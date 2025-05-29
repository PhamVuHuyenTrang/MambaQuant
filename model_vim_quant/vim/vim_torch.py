import math
from dataclasses import dataclass
from typing import Union

from attr import has
import torch
import torch.nn as nn
import torch.nn.functional as F

# from hmquant.qat_torch.qtensor import quantize
from vim.pscan import pscan,PScan
from .normalized_modules import *
from quantize.hadamard_utils import matmul_hadUt_cuda

@dataclass
class MambaConfig:
    d_model: int # D
    n_layers: int
    dt_rank: Union[int, str] = 'auto'
    d_state: int = 16 # N in paper/comments
    expand_factor: int = 2 # E in paper/comments
    d_conv: int = 4

    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init: str = "random" # "random" or "constant"
    dt_scale: float = 1.0
    dt_init_floor = 1e-4

    rms_norm_eps: float = 1e-5

    bias: bool = False
    conv_bias: bool = True
    inner_layernorms: bool = False # apply layernorms to internal activations

    pscan: bool = False # use parallel scan mode or sequential mode when training
    use_cuda: bool = False # use official CUDA implementation when training (not compatible with (b)float16)

    bidirectional: bool = True # use bidirectional MambaBlock

    divide_output: bool = True

    def __post_init__(self):
        self.d_inner = self.expand_factor * self.d_model # E*D = ED in comments

        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)

class Dt_proj_Linear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(Dt_proj_Linear, self).__init__(in_features, out_features, bias)
    
    def forward(self, input):
        # 使用 F.linear 进行矩阵乘法和偏置添加
        output = F.linear(input, self.weight)
        # 调整维度 (batch_size, out_features, seq_len)
        output = output.transpose(1, 2)
        return output

class Mamba(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()

        self.config = config

        self.layers = nn.ModuleList([ResidualBlock(config) for _ in range(config.n_layers)])

    def forward(self, x):
        # x : (B, L, D)

        # y : (B, L, D)

        for layer in self.layers:
            x = layer(x)

        return x
    
    def step(self, x, caches):
        # x : (B, L, D)
        # caches : [cache(layer) for all layers], cache : (h, inputs)

        # y : (B, L, D)
        # caches : [cache(layer) for all layers], cache : (h, inputs)

        for i, layer in enumerate(self.layers):
            x, caches[i] = layer.step(x, caches[i])

        return x, caches

class ResidualBlock(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()

        self.mixer = MambaBlock(config)
        self.norm = RMSNorm(config.d_model, config.rms_norm_eps)

    def forward(self, x):
        # x : (B, L, D)

        # output : (B, L, D)
        return self.mixer(self.norm(x)) + x
    
    def step(self, x, cache):
        # x : (B, D)
        # cache : (h, inputs)
                # h : (B, ED, N)
                # inputs: (B, ED, d_conv-1)

        # output : (B, D)
        # cache : (h, inputs)

        output, cache = self.mixer.step(self.norm(x), cache)
        output = output + x
        return output, cache

class MambaBlock(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()

        self.config = config

        # projects block input from D to 2*ED (two branches)
        self.in_proj = nn.Linear(config.d_model, 2 * config.d_inner, bias=config.bias)

        self.conv1d = nn.Conv1d(in_channels=config.d_inner, out_channels=config.d_inner, 
                              kernel_size=config.d_conv, bias=config.conv_bias, 
                              groups=config.d_inner,
                              padding=config.d_conv - 1)
        
        # projects x to input-dependent delta, B, C
        self.x_proj = nn.Linear(config.d_inner, config.dt_rank + 2 * config.d_state, bias=False)

        # projects delta from dt_rank to d_inner
        self.dt_proj = nn.Linear(config.dt_rank, config.d_inner, bias=True)

        # dt initialization
        # dt weights
        dt_init_std = config.dt_rank**-0.5 * config.dt_scale
        if config.dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif config.dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        
        # delta bias
        dt = torch.exp(
            torch.rand(config.d_inner) * (math.log(config.dt_max) - math.log(config.dt_min)) + math.log(config.dt_min)
        ).clamp(min=config.dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt)) # inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        #self.dt_proj.bias._no_reinit = True # initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # todo : explain why removed

        # S4D real initialization
        A = torch.arange(1, config.d_state + 1, dtype=torch.float32).repeat(config.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A)) # why store A in log ? to keep A < 0 (cf -torch.exp(...)) ? for gradient stability ?
        self.A_log._no_weight_decay = True

        self.D = nn.Parameter(torch.ones(config.d_inner))
        self.D._no_weight_decay = True

        # Backward Parameters
        if config.bidirectional:
            A_b = torch.arange(1, config.d_state + 1, dtype=torch.float32).repeat(config.d_inner, 1)
            self.A_log_b = nn.Parameter(torch.log(A_b))
            self.A_log_b._no_weight_decay = True

            self.conv1d_b = nn.Conv1d(in_channels=config.d_inner, out_channels=config.d_inner,
                                    kernel_size=config.d_conv, bias=config.conv_bias,
                                    groups=config.d_inner,
                                    padding=config.d_conv - 1)
            
            self.x_proj_b = nn.Linear(config.d_inner, config.dt_rank + 2 * config.d_state, bias=False)

            self.dt_proj_b = nn.Linear(config.dt_rank, config.d_inner, bias=True)

            self.D_b = nn.Parameter(torch.ones(config.d_inner))
            self.D_b._no_weight_decay = True

        # projects block output from ED back to D
        self.out_proj = nn.Linear(config.d_inner, config.d_model, bias=config.bias)

        # used in jamba
        if self.config.inner_layernorms:
            self.dt_layernorm = RMSNorm(self.config.dt_rank, config.rms_norm_eps)
            self.B_layernorm = RMSNorm(self.config.d_state, config.rms_norm_eps)
            self.C_layernorm = RMSNorm(self.config.d_state, config.rms_norm_eps)
        else:
            self.dt_layernorm = None
            self.B_layernorm = None
            self.C_layernorm = None

        if self.config.use_cuda:
            try:
                from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
                self.selective_scan_cuda = selective_scan_fn
            except ImportError:
                print("Failed to import mamba_ssm. Falling back to mamba.py.")
                self.config.use_cuda = False

        self.mul_delta_A = Mul()
        self.mul_delta_A_b = Mul()
        self.exp = Exp()
        self.exp_b = Exp()
        self.mul_delta_B = Mul()
        self.mul_delta_B_b = Mul()
        self.mul_deltaB_x = Mul()
        self.mul_deltaB_x_b = Mul()
        self.pscan = pscan
        self.pscan_b = pscan
        self.matmul = MatMul()
        self.matmul_b = MatMul()
        self.mul_D_x = Mul()
        self.mul_D_x_b = Mul()
        self.add_y_Dx = Add()
        self.add_y_Dx_b = Add()

        self.silu_conv1d = nn.SiLU()
        self.silu_conv1d_b = nn.SiLU()
        self.silu_z = nn.SiLU()
        self.silu_z_b = nn.SiLU()
        self.mul_y_z = Mul()
        self.mul_y_z_b = Mul()
        self.softplus = nn.Softplus()
        self.softplus_b = nn.Softplus()

        self.add_y = Add()
        self.mul_y = Mul()

    def _apply_layernorms(self, dt, B, C):
        if self.dt_layernorm is not None:
            dt = self.dt_layernorm(dt)
        if self.B_layernorm is not None:
            B = self.B_layernorm(B)
        if self.C_layernorm is not None:
            C = self.C_layernorm(C)
        return dt, B, C

    def forward(self, x, inference_params=None):
        # x : (B, L, D)  (batch, 197, 192)
        
        # y : (B, L, D)  (batch, 197, 192)
        
        _, L, _ = x.shape
        xz = self.in_proj(x) # (B, L, 2*ED)  (batch, 197, 768)
        chunk_size = xz.shape[-1] // 2
        x, z = xz[..., :chunk_size],xz[..., chunk_size:] # (B, L, ED), (B, L, ED)
        x = x.transpose(1, 2) # (B, ED, L)
        if isinstance(self.conv1d, nn.Conv1d):
            conv1d_out = self.conv1d(x)[:, :, :L]# depthwise convolution over time, with a short filter
        elif isinstance(self.conv1d, nn.Conv2d):
            if self.conv1d.weight.data.shape[2] == 1:
                conv1d_out = self.conv1d(x.unsqueeze(2)).squeeze(2)[:, :, :L]
            else:
                conv1d_out = self.conv1d(x.unsqueeze(3)).squeeze(3)[:, :, :L]
        else:
            raise "Not implemented"
        # conv1d_out = self.conv1d(x)[:, :, :L] 
        conv1d_out = conv1d_out.transpose(1, 2) # (B, L, ED)
        x = self.silu_conv1d(conv1d_out)
        y = self.ssm(x=x, 
                     z=z,
                     b_branch=False)

        if self.config.bidirectional:
            xz_b = xz[:,torch.arange(xz.size(1)-1, -1, -1)] # (B, L, 2*ED)
            chunk_size_b = xz_b.shape[-1] // 2
            x_b, z_b = xz_b[..., :chunk_size_b],xz_b[..., chunk_size_b:] # (B, L, ED), (B, L, ED)
            x_b = x_b.transpose(1, 2) # (B, ED, L)
            if isinstance(self.conv1d_b, nn.Conv1d):
                x_b = self.conv1d_b(x_b)[:, :, :L]# depthwise convolution over time, with a short filter
            elif isinstance(self.conv1d_b, nn.Conv2d):
                if self.conv1d_b.weight.data.shape[2] == 1:
                    x_b = self.conv1d_b(x_b.unsqueeze(2)).squeeze(2)[:, :, :L]
                else:
                    x_b = self.conv1d_b(x_b.unsqueeze(3)).squeeze(3)[:, :, :L]
            else:
                raise "Not implemented"
            # x_b = self.conv1d_b(x_b)[:, :, :L] # depthwise convolution over time, with a short filter
            x_b = x_b.transpose(1, 2) # (B, L, ED)
            x_b = self.silu_conv1d_b(x_b)
            y_b = self.ssm(x=x_b,
                           z=z_b,
                           b_branch=True)

        if self.config.use_cuda:
            if not self.config.bidirectional:
                return self.out_proj(y)
            else:
                if self.config.divide_output:
                    return self.out_proj(self.mul_y(self.add_y(y, y_b[:,torch.arange(y_b.size(1)-1, -1, -1)]), 1/2))
                else:
                    return self.out_proj(y + y_b.flip([1]))
        
        z = self.silu_z(z)
        y = self.mul_y_z(y, z)
        if not self.config.bidirectional:
            return self.out_proj(y)
        else:
            z_b = self.silu_z_b(z_b)
            y_b = self.mul_y_z_b(y_b, z_b)
            if self.config.divide_output:
                y = self.mul_y(self.add_y(y, y_b[:,torch.arange(y_b.size(1)-1, -1, -1)]), 1/2)
                if hasattr(self.config, "use_split"):
                    cls_token = y[:,98:99]
                    y = torch.concat([y[:,:98],y[:,99:]],dim=1)
                    cls_token = self.out_proj(cls_token)
                    y = self.out_proj(y)
                    y = torch.concat([y[:,:98],cls_token,y[:,98:]],dim=1)
                    return y
                else:
                    return self.out_proj(y)
            else:
                return self.out_proj(y + y_b.flip([1]))
    
    def ssm(self, x, z, b_branch=False): 
        # x : (B, L, ED)

        # y : (B, L, ED)
        if not b_branch:
            A_log = self.A_log
            D = self.D
            x_proj = self.x_proj
            dt_proj = self.dt_proj
            softplus = self.softplus
        else:
            A_log = self.A_log_b
            D = self.D_b
            x_proj = self.x_proj_b
            dt_proj = self.dt_proj_b
            softplus = self.softplus_b


        A = -torch.exp(A_log.float()) # (ED, N)

        D = D.float()


        deltaBC = x_proj(x) # (B, L, dt_rank+2*N)
        delta, B, C = torch.split(deltaBC, [self.config.dt_rank, self.config.d_state, self.config.d_state], dim=-1) # (B, L, dt_rank), (B, L, N), (B, L, N)
        delta, B, C = self._apply_layernorms(delta, B, C)
        # delta = dt_proj.weight @ delta.transpose(1, 2) # (ED, dt_rank) @ (B, L, dt_rank) -> (B, ED, L)
        # here we just apply the matrix mul operation of delta = softplus(dt_proj(delta))
        # the rest will be applied later (fused if using cuda)
        
        # choose which selective_scan function to use, according to config
        if self.config.use_cuda:
            # these are unfortunately needed for the selective_scan_cuda function
            x = x.transpose(1, 2)
            B = B.transpose(1, 2)
            C = C.transpose(1, 2)
            z = z.transpose(1, 2)
            delta = self.matmul(delta, dt_proj.weight.transpose(0, 1)).transpose(1,2)

            # "softplus" + "bias" + "y * silu(z)" operations are fused
            y = self.selective_scan_cuda(x, delta, A, B, C, D, z=z, delta_softplus=True, delta_bias=dt_proj.bias.float())
            y = y.transpose(1, 2) # (B, L, ED)
        
        else:
            delta = dt_proj(delta)
            delta = softplus(delta)

            if self.config.pscan:
                y = self.selective_scan(x, delta, A, B, C, D)
            else:
                y = self.selective_scan_seq(x, delta, A, B, C, D, b_branch=b_branch)

        return y
    
    def selective_scan(self, x, delta, A, B, C, D):
        # x : (B, L, ED)
        # Δ : (B, L, ED)
        # A : (ED, N)
        # B : (B, L, N)
        # C : (B, L, N)
        # D : (ED)

        # y : (B, L, ED)

        deltaA = torch.exp(delta.unsqueeze(-1) * A) # (B, L, ED, N)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2) # (B, L, ED, N)

        BX = deltaB * (x.unsqueeze(-1)) # (B, L, ED, N)
        
        hs = pscan(deltaA, BX)

        y = (hs @ C.unsqueeze(-1)).squeeze(3) # (B, L, ED, N) @ (B, L, N, 1) -> (B, L, ED, 1)

        y = y + D * x

        return y
    
    def selective_scan_seq(self, x, delta, A, B, C, D, b_branch=False): # type: ignore
        # x : (B, L, ED)
        # Δ : (B, L, ED)
        # A : (ED, N)
        # B : (B, L, N)
        # C : (B, L, N)
        # D : (ED)

        # y : (B, L, ED)

        if b_branch:
            mul_delta_A = self.mul_delta_A
            exp = self.exp
            mul_delta_B = self.mul_delta_B
            mul_deltaB_x = self.mul_deltaB_x
            matmul = self.matmul
            mul_D_x = self.mul_D_x
            add_y_Dx = self.add_y_Dx
            pscan = self.pscan
        else:
            mul_delta_A = self.mul_delta_A_b
            exp = self.exp_b
            mul_delta_B = self.mul_delta_B_b
            mul_deltaB_x = self.mul_deltaB_x_b
            matmul = self.matmul_b
            mul_D_x = self.mul_D_x_b
            add_y_Dx = self.add_y_Dx_b
            pscan = self.pscan_b

        _, L, _ = x.shape

        deltaA = exp(mul_delta_A(delta.unsqueeze(-1), A)) # (B, L, ED, N)
        deltaB = mul_delta_B(delta.unsqueeze(-1), B.unsqueeze(2)) # (B, L, ED, N)

        BX = mul_deltaB_x(deltaB, (x.unsqueeze(-1))) # (B, L, ED, N)
        hs = pscan(deltaA,BX)

        # h = torch.zeros(x.size(0), self.config.d_inner, self.config.d_state, device=deltaA.device) # (B, ED, N)
        # hs = []
        # for t in range(0, L):
        #     h = deltaA[:, t] * h + BX[:, t]
        #     hs.append(h)
            
        # hs = torch.stack(hs, dim=1) # (B, L, ED, N)
        if hasattr(matmul,"R4") and matmul.R4 is not None:
            hs=hs@self.R4;  C=C@self.R4
        if hasattr(matmul,"R3") and matmul.R3 is not None:
            if  hasattr(matmul,"S3") and matmul.S3 is not None:
                hs = hs/(matmul.S3.unsqueeze(1).to(hs))
                # hs = (hs.permute(0, 1, 3, 2)@matmul.R3).permute(0, 1, 3, 2)
                hs = (hs.permute(0, 1, 3, 2)@matmul.R3).permute(0, 1, 3, 2)
                y = (matmul(hs,C.unsqueeze(-1))).squeeze(3)
                y = y@matmul.R3.T
                y = y*matmul.S3.to(y)           
            else:
                hs = (hs.permute(0, 1, 3, 2)@matmul.R3).permute(0, 1, 3, 2)
                y = (matmul(hs,C.unsqueeze(-1))).squeeze(3)
                y = y@matmul.R3.T
            # hs = (hs.permute(0,2,3,1)@matmul.R3).permute(0,3,1,2)
            # y = (matmul(hs,C.unsqueeze(-1))).squeeze(3)
            # y = ((y.permute(0,2,1))@matmul.R3.T).permute(0,2,1)
        else:
            y = (matmul(hs, C.unsqueeze(-1))).squeeze(3) # (B, L, ED, N) @ (B, L, N, 1) -> (B, L, ED, 1)

        y = add_y_Dx(y, mul_D_x(D, x))

        return y
    
    # -------------------------- inference -------------------------- #
    """
    Concerning auto-regressive inference

    The cool part of using Mamba : inference is constant wrt to sequence length
    We just have to keep in cache, for each layer, two things :
    - the hidden state h (which is (B, ED, N)), as you typically would when doing inference with a RNN
    - the last d_conv-1 inputs of the layer, to be able to compute the 1D conv which is a convolution over the time dimension
      (d_conv is fixed so this doesn't incur a growing cache as we progress on generating the sequence)
      (and d_conv is usually very small, like 4, so we just have to "remember" the last 3 inputs)

    Concretely, these two quantities are put inside a cache tuple, and are named h and inputs respectively.
    h is (B, ED, N), and inputs is (B, ED, d_conv-1)
    The MambaBlock.step() receives this cache, and, along with outputing the output, alos outputs the updated cache for the next call.

    The cache object is initialized as follows : (None, torch.zeros()).
    When h is None, the selective scan function detects it and start with h=0.
    The torch.zeros() isn't a problem (it's same as just feeding the input, because the conv1d is padded)

    As we need one such cache variable per layer, we store a caches object, which is simply a list of cache object. (See mamba_lm.py)
    """
    
    def step(self, x, cache):
        # x : (B, D)
        # cache : (h, inputs)
                # h : (B, ED, N)
                # inputs : (B, ED, d_conv-1)
        
        # y : (B, D)
        # cache : (h, inputs)
        
        h, inputs = cache
        
        xz = self.in_proj(x) # (B, 2*ED)
        x, z = xz.chunk(2, dim=1) # (B, ED), (B, ED)

        # x branch
        x_cache = x.unsqueeze(2)
        x = self.conv1d(torch.cat([inputs, x_cache], dim=2))[:, :, self.config.d_conv-1] # (B, ED)

        x = F.silu(x)
        y, h = self.ssm_step(x, h)

        # z branch
        z = F.silu(z)

        output = y * z
        output = self.out_proj(output) # (B, D)

        # prepare cache for next call
        inputs = torch.cat([inputs[:, :, 1:], x_cache], dim=2) # (B, ED, d_conv-1)
        cache = (h, inputs)
        
        return output, cache

    def ssm_step(self, x, h):
        # x : (B, ED)
        # h : (B, ED, N)

        # y : (B, ED)
        # h : (B, ED, N)

        A = -torch.exp(self.A_log.float()) # (ED, N) # todo : ne pas le faire tout le temps, puisque c'est indépendant de la timestep
        D = self.D.float()

        deltaBC = self.x_proj(x) # (B, dt_rank+2*N)

        delta, B, C = torch.split(deltaBC, [self.config.dt_rank, self.config.d_state, self.config.d_state], dim=-1) # (B, dt_rank), (B, N), (B, N)
        delta, B, C = self._apply_layernorms(delta, B, C)
        delta = F.softplus(self.dt_proj(delta)) # (B, ED)

        deltaA = torch.exp(delta.unsqueeze(-1) * A) # (B, ED, N)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(1) # (B, ED, N)

        BX = deltaB * (x.unsqueeze(-1)) # (B, ED, N)

        if h is None:
            h = torch.zeros(x.size(0), self.config.d_inner, self.config.d_state, device=deltaA.device) # (B, ED, N)

        h = deltaA * h + BX # (B, ED, N)

        y = (h @ C.unsqueeze(-1)).squeeze(2) # (B, ED, N) @ (B, N, 1) -> (B, ED, 1)

        y = y + D * x

        return y, h



from copy import deepcopy
class MambaBlock_optimize(nn.Module):
    def __init__(self, ori_module):
        super().__init__()

        self.config = ori_module.config

        # projects block input from D to 2*ED (two branches)
        # self.in_proj = deepcopy(ori_module.in_proj)
        self.in_proj_states = deepcopy(ori_module.in_proj) 
        self.in_proj_states.weight.data = self.in_proj_states.weight.data[:self.config.d_inner]
        self.in_proj_states.out_features = self.config.d_inner
        self.in_proj_gates = deepcopy(ori_module.in_proj) 
        self.in_proj_gates.weight.data = self.in_proj_gates.weight.data[-self.config.d_inner:]
        self.in_proj_gates.out_features = self.config.d_inner


        self.conv1d = ori_module.conv1d
        
        # projects x to input-dependent delta, B, C
        # self.x_proj = deepcopy(ori_module.x_proj)
        self.x_proj_B = deepcopy(ori_module.x_proj)
        self.x_proj_B.weight.data = ori_module.x_proj.weight.data[-2*self.config.d_state:-self.config.d_state]
        self.x_proj_B.out_features = self.config.d_state
        self.x_proj_C = deepcopy(ori_module.x_proj)
        self.x_proj_C.weight.data = ori_module.x_proj.weight.data[-self.config.d_state:]
        self.x_proj_C.out_features = self.config.d_state
        self.x_proj_dt = deepcopy(ori_module.x_proj)
        self.x_proj_dt.weight.data = ori_module.x_proj.weight.data[:-2*self.config.d_state]
        self.x_proj_dt.out_features = self.config.dt_rank


        # projects delta from dt_rank to d_inner
        self.dt_proj = ori_module.dt_proj
        self.A_log = ori_module.A_log 
        self.D = ori_module.D
        if self.config.bidirectional:
            self.A_log_b = ori_module.A_log_b
            self.conv1d_b = ori_module.conv1d_b

            # self.x_proj_b = deepcopy(ori_module.x_proj_b)
            self.x_proj_B_b = deepcopy(ori_module.x_proj_b)
            self.x_proj_B_b.weight.data = ori_module.x_proj_b.weight.data[-2*self.config.d_state:-self.config.d_state]
            self.x_proj_B_b.out_features = self.config.d_state
            self.x_proj_C_b = deepcopy(ori_module.x_proj_b)
            self.x_proj_C_b.weight.data = ori_module.x_proj_b.weight.data[-self.config.d_state:]
            self.x_proj_C_b.out_features = self.config.d_state
            self.x_proj_dt_b = deepcopy(ori_module.x_proj_b)
            self.x_proj_dt_b.weight.data = ori_module.x_proj_b.weight.data[:-2*self.config.d_state]
            self.x_proj_dt_b.out_features = self.config.dt_rank

            self.dt_proj_b = ori_module.dt_proj_b

            self.D_b = ori_module.D_b


        # projects block output from ED back to D
        self.out_proj = ori_module.out_proj

        # used in jamba
        self.dt_layernorm = None
        self.B_layernorm = None
        self.C_layernorm = None

        if self.config.use_cuda:
            try:
                from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
                self.selective_scan_cuda = selective_scan_fn
            except ImportError:
                print("Failed to import mamba_ssm. Falling back to mamba.py.")
                self.config.use_cuda = False

        self.mul_delta_A = Mul()
        self.mul_delta_A_b = Mul()
        self.exp = Exp()
        self.exp_b = Exp()
        self.mul_delta_B = Mul()
        self.mul_delta_B_b = Mul()
        self.mul_deltaB_x = Mul()
        self.mul_deltaB_x_b = Mul()
        self.pscan = pscan
        self.pscan_b = pscan
        self.matmul = ori_module.matmul
        self.matmul_b = ori_module.matmul_b
        self.mul_D_x = Mul()
        self.mul_D_x_b = Mul()
        self.add_y_Dx = Add()
        self.add_y_Dx_b = Add()

        self.silu_conv1d = nn.SiLU()
        self.silu_conv1d_b = nn.SiLU()
        self.silu_z = nn.SiLU()
        self.silu_z_b = nn.SiLU()
        self.mul_y_z = Mul()
        self.mul_y_z_b = Mul()
        self.softplus = nn.Softplus()
        self.softplus_b = nn.Softplus()

        self.add_y = Add()
        self.mul_y = Mul()

    def _apply_layernorms(self, dt, B, C):
        if self.dt_layernorm is not None:
            dt = self.dt_layernorm(dt)
        if self.B_layernorm is not None:
            B = self.B_layernorm(B)
        if self.C_layernorm is not None:
            C = self.C_layernorm(C)
        return dt, B, C

    def forward(self, x, inference_params=None):
        # x : (B, L, D)  (batch, 197, 192)
        
        # y : (B, L, D)  (batch, 197, 192)
        
        _, L, _ = x.shape
        # xz = self.in_proj(x) # (B, L, 2*ED)  (batch, 197, 768)
        # chunk_size = xz.shape[-1] // 2
        # x, z = xz[..., :chunk_size],xz[..., chunk_size:] # (B, L, ED), (B, L, ED)
        z = self.in_proj_gates(x)
        x = self.in_proj_states(x)
        xz = torch.cat([x, z], dim=-1)
        x = x.transpose(1, 2) # (B, ED, L)
        if isinstance(self.conv1d, nn.Conv1d):
            conv1d_out = self.conv1d(x)[:, :, :L]# depthwise convolution over time, with a short filter
        elif isinstance(self.conv1d, nn.Conv2d):
            if self.conv1d.weight.data.shape[2] == 1:
                conv1d_out = self.conv1d(x.unsqueeze(2)).squeeze(2)[:, :, :L]
            else:
                conv1d_out = self.conv1d(x.unsqueeze(3)).squeeze(3)[:, :, :L]
        else:
            raise "Not implemented"
        # conv1d_out = self.conv1d(x)[:, :, :L] 
        conv1d_out = conv1d_out.transpose(1, 2) # (B, L, ED)
        x = self.silu_conv1d(conv1d_out)
        y = self.ssm(x=x, 
                     z=z,
                     b_branch=False)

        if self.config.bidirectional:
            xz_b = xz[:,torch.arange(xz.size(1)-1, -1, -1)] # (B, L, 2*ED)
            chunk_size_b = xz_b.shape[-1] // 2
            x_b, z_b = xz_b[..., :chunk_size_b],xz_b[..., chunk_size_b:] # (B, L, ED), (B, L, ED)
            x_b = x_b.transpose(1, 2) # (B, ED, L)
            if isinstance(self.conv1d_b, nn.Conv1d):
                x_b = self.conv1d_b(x_b)[:, :, :L]# depthwise convolution over time, with a short filter
            elif isinstance(self.conv1d_b, nn.Conv2d):
                if self.conv1d_b.weight.data.shape[2] == 1:
                    x_b = self.conv1d_b(x_b.unsqueeze(2)).squeeze(2)[:, :, :L]
                else:
                    x_b = self.conv1d_b(x_b.unsqueeze(3)).squeeze(3)[:, :, :L]
            else:
                raise "Not implemented"
            # x_b = self.conv1d_b(x_b)[:, :, :L] # depthwise convolution over time, with a short filter
            x_b = x_b.transpose(1, 2) # (B, L, ED)
            x_b = self.silu_conv1d_b(x_b)
            y_b = self.ssm(x=x_b,
                           z=z_b,
                           b_branch=True)

        if self.config.use_cuda:
            if not self.config.bidirectional:
                return self.out_proj(y)
            else:
                if self.config.divide_output:
                    return self.out_proj(self.mul_y(self.add_y(y, y_b[:,torch.arange(y_b.size(1)-1, -1, -1)]), 1/2))
                else:
                    return self.out_proj(y + y_b.flip([1]))
        
        z = self.silu_z(z)
        y = self.mul_y_z(y, z)

        z_b = self.silu_z_b(z_b)
        y_b = self.mul_y_z_b(y_b, z_b)
        if self.config.divide_output:
            y = self.mul_y(self.add_y(y, y_b[:,torch.arange(y_b.size(1)-1, -1, -1)]), 1/2)
            if hasattr(self,"R2"):
                y = y@self.R2
            if hasattr(self.config, "use_split"):
                cls_token = y[:,98:99]
                y = torch.concat([y[:,:98],y[:,99:]],dim=1)
                cls_token = self.out_proj(cls_token)
                y = self.out_proj(y)
                y = torch.concat([y[:,:98],cls_token,y[:,98:]],dim=1)
                return y
            else:
                return self.out_proj(y)
        else:
            return self.out_proj(y + y_b.flip([1]))
    
    def ssm(self, x, z, b_branch=False): 
        # x : (B, L, ED)

        # y : (B, L, ED)
        if not b_branch:
            A_log = self.A_log
            D = self.D
            # x_proj = self.x_proj
            x_proj_b = self.x_proj_B
            x_proj_c = self.x_proj_C
            x_proj_dt = self.x_proj_dt
            dt_proj = self.dt_proj
            softplus = self.softplus
        else:
            A_log = self.A_log_b
            D = self.D_b
            # x_proj = self.x_proj_b
            x_proj_b = self.x_proj_B_b
            x_proj_c = self.x_proj_C_b
            x_proj_dt = self.x_proj_dt_b
            dt_proj = self.dt_proj_b
            softplus = self.softplus_b


        A = -torch.exp(A_log.float()) # (ED, N)
        D = D.float()
        
        x_rotated = x.clone()
        if hasattr(self,"R6"):
            x_rotated = x_rotated@self.R6
            
        delta = x_proj_dt(x_rotated)
        B = x_proj_b(x_rotated)
        C = x_proj_c(x_rotated)
        # deltaBC = x_proj(x) # (B, L, dt_rank+2*N)
        # delta, B, C = torch.split(deltaBC, [self.config.dt_rank, self.config.d_state, self.config.d_state], dim=-1) # (B, L, dt_rank), (B, L, N), (B, L, N)
        # delta, B, C = self._apply_layernorms(delta, B, C)

        # choose which selective_scan function to use, according to config
        delta = dt_proj(delta)
        delta = softplus(delta)

        if self.config.pscan:
            y = self.selective_scan(x, delta, A, B, C, D)
        else:
            y = self.selective_scan_seq(x, delta, A, B, C, D, b_branch=b_branch)

        return y
    
    def selective_scan(self, x, delta, A, B, C, D):
        # x : (B, L, ED)
        # Δ : (B, L, ED)
        # A : (ED, N)
        # B : (B, L, N)
        # C : (B, L, N)
        # D : (ED)

        # y : (B, L, ED)

        deltaA = torch.exp(delta.unsqueeze(-1) * A) # (B, L, ED, N)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2) # (B, L, ED, N)

        BX = deltaB * (x.unsqueeze(-1)) # (B, L, ED, N)
        
        hs = pscan(deltaA, BX)

        y = (hs @ C.unsqueeze(-1)).squeeze(3) # (B, L, ED, N) @ (B, L, N, 1) -> (B, L, ED, 1)

        y = y + D * x

        return y
    
    def selective_scan_seq(self, x, delta, A, B, C, D, b_branch=False): # type: ignore
        # x : (B, L, ED)
        # Δ : (B, L, ED)
        # A : (ED, N)
        # B : (B, L, N)
        # C : (B, L, N)
        # D : (ED)

        # y : (B, L, ED)

        if b_branch:
            mul_delta_A = self.mul_delta_A
            exp = self.exp
            mul_delta_B = self.mul_delta_B
            mul_deltaB_x = self.mul_deltaB_x
            matmul = self.matmul
            mul_D_x = self.mul_D_x
            add_y_Dx = self.add_y_Dx
            pscan = self.pscan
        else:
            mul_delta_A = self.mul_delta_A_b
            exp = self.exp_b
            mul_delta_B = self.mul_delta_B_b
            mul_deltaB_x = self.mul_deltaB_x_b
            matmul = self.matmul_b
            mul_D_x = self.mul_D_x_b
            add_y_Dx = self.add_y_Dx_b
            pscan = self.pscan_b

        _, L, _ = x.shape

        deltaA = exp(mul_delta_A(delta.unsqueeze(-1), A)) # (B, L, ED, N)
        deltaB = mul_delta_B(delta.unsqueeze(-1), B.unsqueeze(2)) # (B, L, ED, N)

        BX = mul_deltaB_x(deltaB, (x.unsqueeze(-1))) # (B, L, ED, N)
        hs = pscan(deltaA,BX)

        # h = torch.zeros(x.size(0), self.config.d_inner, self.config.d_state, device=deltaA.device) # (B, ED, N)
        # hs = []
        # for t in range(0, L):
        #     h = deltaA[:, t] * h + BX[:, t]
        #     hs.append(h)
            
        # hs = torch.stack(hs, dim=1) # (B, L, ED, N)
        if hasattr(self,"R4") and self.R4 is not None:
            hs=hs@self.R4;  #C=C@self.R4
        if hasattr(matmul,"S4") and matmul.S4 is not None:
            hs = hs/(matmul.S4.to(hs))
            C = C*(matmul.S4.to(hs))
        if hasattr(matmul,"R3") and matmul.R3 is not None:
            if  hasattr(matmul,"S3") and matmul.S3 is not None:
                hs = hs/(matmul.S3.unsqueeze(1).to(hs))
                hs = (hs.permute(0, 1, 3, 2)@matmul.R3).permute(0, 1, 3, 2)
                y = (matmul(hs,C.unsqueeze(-1))).squeeze(3)
                y = y@matmul.R3.T
                y = y*matmul.S3.to(y)           
            else:
                hs = (hs.permute(0, 1, 3, 2)@matmul.R3).permute(0, 1, 3, 2)
                y = (matmul(hs,C.unsqueeze(-1))).squeeze(3)
                y = y@matmul.R3.T
            # hs = (hs.permute(0,2,3,1)@matmul.R3).permute(0,3,1,2)
            # y = (matmul(hs,C.unsqueeze(-1))).squeeze(3)
            # y = ((y.permute(0,2,1))@matmul.R3.T).permute(0,2,1)
        else:
            if  hasattr(matmul,"S3") and matmul.S3 is not None:
                hs = hs/(matmul.S3.unsqueeze(1).to(hs))
                B, L, ED, N = hs.shape
                # matmul.pertoken = True
                y = (matmul(hs, C.reshape(B,L,N,1))) # (B, L, ED, N) @ (B, L, N, 1) -> (B, L, ED, 1)
                y = y.reshape(B,L,ED)
                y = y*matmul.S3.to(y) 
                
            else:
                B, L, ED, N = hs.shape
                matmul.pertoken = True
                y = (matmul(hs, C.reshape(B,L,N,1))) # (B, L, ED, N) @ (B, L, N, 1) -> (B, L, ED, 1)
                y = y.reshape(B,L,ED)

        y = add_y_Dx(y, mul_D_x(D, x))

        return y



class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()

        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = None

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

        return output

