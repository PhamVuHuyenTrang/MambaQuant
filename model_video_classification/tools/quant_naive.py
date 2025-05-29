# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import sys
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
from src.mamba import Mamba3DModel
from mmengine.config import Config, DictAction
from mmengine.runner import Runner
from einops import rearrange
from mmaction.registry import RUNNERS
import torch.nn as nn
import torch
import numpy as np
from torch import Tensor
from typing import Optional
from quantize.utils import set_seed,Logger
from copy import deepcopy
set_seed(10)

# 将 sys.stdout 重定向到 Logger 类实例
sys.stdout = Logger()


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMAction2 test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing evaluation metrics')
    parser.add_argument(
        '--dump',
        type=str,
        help='dump predictions to a pickle file for offline evaluation')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--show-dir',
        help='directory where the visualization images will be saved.')
    parser.add_argument(
        '--show',
        action='store_true',
        help='whether to display the prediction results in a window.')
    parser.add_argument(
        '--interval',
        type=int,
        default=1,
        help='visualize per interval samples.')
    parser.add_argument(
        '--wait-time',
        type=float,
        default=2,
        help='display time of every window. (second)')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    parser.add_argument('--use_smoothquant', default=False, type=bool)
    parser.add_argument("--use_gptq", action="store_true")
    parser.add_argument("--use_hadmard", action="store_true")
    parser.add_argument('--use_hadmard_R3S', action="store_true")
    parser.add_argument('--use_hadmard_R1', action="store_true")
    parser.add_argument('--use_hadmard_R4', action="store_true")
    parser.add_argument('--use_hadmard_R5', action="store_true")
    parser.add_argument("--use_perkernel", action="store_true")
    parser.add_argument('--static_quant', action='store_true')
    parser.add_argument('--quant_weight', action="store_true")
    parser.add_argument('--quant_act', action="store_true")
    parser.add_argument('--w_bit', type=int,default=8)
    parser.add_argument('--a_bit', type=int,default=8)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def merge_args(cfg, args):
    """Merge CLI arguments to config."""
    # -------------------- visualization --------------------
    if args.show or (args.show_dir is not None):
        assert 'visualization' in cfg.default_hooks, \
            'VisualizationHook is not set in the `default_hooks` field of ' \
            'config. Please set `visualization=dict(type="VisualizationHook")`'

        cfg.default_hooks.visualization.enable = True
        cfg.default_hooks.visualization.show = args.show
        cfg.default_hooks.visualization.wait_time = args.wait_time
        cfg.default_hooks.visualization.out_dir = args.show_dir
        cfg.default_hooks.visualization.interval = args.interval

    # -------------------- Dump predictions --------------------
    if args.dump is not None:
        assert args.dump.endswith(('.pkl', '.pickle')), \
            'The dump file must be a pkl file.'
        dump_metric = dict(type='DumpResults', out_file_path=args.dump)
        if isinstance(cfg.test_evaluator, (list, tuple)):
            cfg.test_evaluator = list(cfg.test_evaluator)
            cfg.test_evaluator.append(dump_metric)
        else:
            cfg.test_evaluator = [cfg.test_evaluator, dump_metric]

    return cfg


def main():
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)
    cfg = merge_args(cfg, args)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    cfg.load_from = args.checkpoint

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)
        
    device = runner.model.backbone.pos_embed.device
    # w_cfg = {"dynamic_method":"per_tensor","n_bits":args.w_bit}
    w_cfg = {"dynamic_method":"per_channel","per_channel_axes":[0],"n_bits":args.w_bit}
    if args.use_perkernel:
        conv1d_w_cfg = {"dynamic_method":"per_channel","per_channel_axes":[2],"n_bits":args.w_bit}
    else:
        conv1d_w_cfg = w_cfg
    a_cfg = {"dynamic_method":"per_tensor","n_bits":args.a_bit}
    from utils.utils import convert_vim_2_vim_torch
    from utils.normalized_modules import MatMul
    from quantize import QuantMatMul,QuantLinear,QuantConv1d,QuantConv2d
    from quantize.utils import set_quant_state,set_static_quant,set_observing
    from quantize.smoothquant import mamband_mambablock_smootquant
    def replace_layers(model, target_class, replacement_class):
        for name, child in model.named_children():
            if isinstance(child, target_class):
                # print(name)
                # Replace the layer with the new quantized version
                if target_class == MatMul:
                    setattr(model, name, replacement_class(x1_quant_params=a_cfg,x2_quant_params=a_cfg))
                elif "conv1d" in name:
                    setattr(model, name, replacement_class(child,weight_quant_params=conv1d_w_cfg,act_quant_params=a_cfg))
                else:
                    setattr(model, name, replacement_class(child,weight_quant_params=w_cfg,act_quant_params=a_cfg))
            else:
                # Recursively call this function on the child module
                replace_layers(child, target_class, replacement_class)
    
    # start testing
    def test(self) -> dict:
        """Launch test.

        Returns:
            dict: A dict of metrics on testing set.
        """
        if self._test_loop is None:
            raise RuntimeError(
                '`self._test_loop` should not be None when calling test '
                'method. Please provide `test_dataloader`, `test_cfg` and '
                '`test_evaluator` arguments when initializing runner.')

        self._test_loop = self.build_test_loop(self._test_loop)  # type: ignore

        self.call_hook('before_run')

        # make sure checkpoint-related hooks are triggered after `before_run`
        self.load_or_resume()
        convert_vim_2_vim_torch(self.model.backbone,"cuda")
        if args.use_smoothquant:
            # from quantize.smoothquant_generate_act_scale_shift import mamba3d_video_generate_act_scale_shift
            # mamba3d_video_generate_act_scale_shift(self)#产生初始scale和shift值
            act_scales = torch.load("ckpt/"+cfg.filename.split("/")[-1].split(".")[0]+"_scale.pt")
            mamband_mambablock_smootquant(self.model.backbone.layers,act_scales)
            
            
        if args.use_gptq:
            from quantize.gptq import gptq_fwrd_mamba3d
            args.nsamples=8;args.w_bits=w_cfg['n_bits'];args.w_groupsize=128
            args.percdamp=0.01;args.act_order=False
            # quantizers = gptq_fwrd_mamba3d(self.model, self.test_dataloader, device, args)
            # torch.save(self.model,args.checkpoint[:-4]+"_gptq_weight_"+str(args.w_bits)+"bit.pt")
            # torch.save(quantizers,args.checkpoint[:-4]+"_gptq_scales_"+str(args.w_bits)+"bit.pt")
            self.model = torch.load(args.checkpoint[:-4]+"_gptq_weight_"+str(args.w_bits)+"bit.pt",map_location='cpu')
            quantizers = torch.load(args.checkpoint[:-4]+"_gptq_scales_"+str(args.w_bits)+"bit.pt",map_location='cpu')
            self.model.to(device)
        
        replace_layers(self.model, MatMul, QuantMatMul)
        replace_layers(self.model, nn.Linear, QuantLinear)
        replace_layers(self.model, nn.Conv1d, QuantConv1d)
        replace_layers(self.model, nn.Conv2d, QuantConv2d)
        set_quant_state(self.model,weight_quant=args.quant_weight,act_quant=args.quant_act)
        
        if args.use_hadmard:
            # matmul_scale = torch.load("ckpt/"+cfg.filename.split("/")[-1].split(".")[0]+"_matmul_scale.pt")
            from quantize.hm_model_utils import fuse_layer_norms, RotateModule, RQuantLinear,RMSNorm
            from quantize.hadmard import random_hadamard_matrix
            from mmpretrain.models.utils import resize_pos_embed
            import types
            h1 = random_hadamard_matrix(self.model.backbone.layers[0].mixer.in_proj.in_features,device)
            R1 = RotateModule(h1)
            h3 = random_hadamard_matrix(self.model.backbone.layers[0].mixer.out_proj.in_features,device)
            R3 = RotateModule(h3)
            if args.use_hadmard_R3S:
                def substitute_layers(model):
                    for name,module in model.named_modules():
                        if 'matmul' in name and 'quantizer' not in name:
                            if 'matmul_b' in name: continue
                            # module.register_parameter("matmul_scale",torch.nn.Parameter(matmul_scale[name]))
                            module.register_parameter("R3",R3.weight)
                            # module.register_parameter("R4",R4.weight)
                substitute_layers(self.model)
            if args.use_hadmard_R1:
                def predict(self,
                    inputs: torch.Tensor,
                    data_samples = None,
                    **kwargs):
                    if not hasattr(self,"R1"):
                        self.R1 = R1
                        self.cls_head.fc_cls.weight.data = self.cls_head.fc_cls.weight.data@self.R1.weight.T
                    feats, predict_kwargs = self.extract_feat(inputs, test_mode=True)
                    # feats = tuple([feats[0]@self.R1.weight.T])
                    predictions = self.cls_head.predict(feats, data_samples,
                                                        **predict_kwargs)
                    return predictions
                
                def mamband_forward(self, x):
                    B = x.shape[0]
                    b, _, _, h, w = x.shape
                    #1.  将ln的减均值操作吸收到weight里
                    if not hasattr(self,"obsorted_mean_2_linear"):
                        self.patch_embed.projection.weight.data = self.patch_embed.projection.weight.data - self.patch_embed.projection.weight.data.mean(dim=0,keepdim=True)
                        self.patch_embed.projection.bias.data = self.patch_embed.projection.bias.data - self.patch_embed.projection.bias.data.mean(dim=0,keepdim=True)
                        for i, layer in enumerate(self.layers):
                            layer.mixer.out_proj.weight.data = layer.mixer.out_proj.weight.data - layer.mixer.out_proj.weight.data.mean(dim=0,keepdim=True) 
                        self.obsorted_mean_2_linear = True 
                    
                    if not hasattr(self, 'R1'): #I
                        self.R1=R1
                        self.patch_embed.projection.weight.data = self.R1.weight@self.patch_embed.projection.weight.data  #I: 右乘QT
                        self.patch_embed.projection.bias.data = (self.patch_embed.projection.bias.data.view(1,-1)@self.R1.weight.T).view(-1)
                        
                    x, patch_resolution = self.patch_embed(x)
                    patch_resolution = (patch_resolution[0],patch_resolution[1],patch_resolution[1])
                    pos = resize_pos_embed(
                        self.pos_embed,
                        self.patch_resolution,
                        patch_resolution,
                        mode=self.interpolate_mode,
                        num_extra_tokens=self.num_extra_tokens)[:,self.num_extra_tokens:]
                    pos = pos-pos.mean(dim=-1,keepdim=True)
                    pos = pos@self.R1.weight.T
                    x = x + pos
                    x = self.drop_after_pos(x)

                    outs = []
                    orders = (
                            't l h w',
                            't l w h',
                            'w h t l'
                    )

                    n_dim_pos = [self.n_dim_pos ] * 3

                    if self.factorization is not None:
                        if self.factorization == 'hw_t':
                            n_dim_pos = (2,2,4)
                        elif self.factorization == 'h_w_t':
                            n_dim_pos = (1,1,2)
                    shape = (patch_resolution[0],1,patch_resolution[1],patch_resolution[2])
                    for i,blk in enumerate(self.layers):
                        z = i // 2
                        d = z % len(orders)
                        
                        x = blk(x,order=orders[d],shape=shape,n_dim_pos=n_dim_pos[d])
                            
                        if i in self.out_indices:
                            outs.append(self._format_output(x, patch_resolution))
                    return outs[-1]
                
                def block_forward(
                    self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None,order='t l h w',
                    shape=None,skip=True,n_dim_pos=4
                ):
                    r"""Pass the input through the encoder layer.

                    Args:
                        hidden_states: the sequence to the encoder layer (required).
                        residual: hidden_states = Mixer(LN(residual))
                    """
                    h = w = 0
                    assert shape is not None
                    t,l,h,w = shape
                    if n_dim_pos != 4:
                        order = order.split(' ')
                        assert len(order) == 4
                        trunc_n = 4 - n_dim_pos
                        tgt_order = f"(n {' '.join(order[:trunc_n])}) ({' '.join(order[trunc_n:])}) c"
                    else:
                        tgt_order = f'n ({order}) c'
                    hidden_states =  rearrange(hidden_states,f'n (t l h w ) c -> {tgt_order}',t=t,l=l,h=h,w=w)
                    if self.reverse:
                        hidden_states = hidden_states.flip(1)
                        if residual is not None:
                            residual = residual.flip(1)
                    if not self.fused_add_norm:
                        #2.  需要做一些等效变换,将ln的w和b吸收到mixer中的input_proj和residual中
                        if not hasattr(self,"norm_weight"):
                            self.norm_weight = self.norm.weight.clone()
                            linear_dtype = self.mixer.in_proj.weight.dtype
                            self.norm_weight = deepcopy(self.norm.weight.to(linear_dtype))
                            if hasattr( self.norm, 'bias') and self.norm.bias is not None:
                                self.norm_bias = deepcopy(self.norm.bias.to(linear_dtype) )
                            else:
                                self.norm_bias = None
                            # Calculating new weight and bias
                            W_ = self.mixer.in_proj.weight.data.double()
                            self.mixer.in_proj.weight.data = (W_ * self.norm_weight.double()).to(linear_dtype)
                            if hasattr( self.norm, 'bias') and  self.norm.bias is not None:
                                if self.mixer.in_proj.bias is None:
                                    self.mixer.in_proj.bias = torch.nn.Parameter(torch.zeros(self.mixer.in_proj.out_features, dtype=torch.float64))
                                    self.mixer.in_proj.bias.data = self.mixer.in_proj.bias.data.to(self.norm_weight)
                                self.mixer.in_proj.bias.data = self.mixer.in_proj.bias.data.double() + torch.matmul(W_, self.norm.bias.double())
                                self.mixer.in_proj.bias.data = self.mixer.in_proj.bias.data.to(linear_dtype)
                                self.norm.bias = None
                            with torch.no_grad():
                                self.norm.weight.fill_(1.)
                            if self.norm_bias is not None:
                                if self.mixer.out_proj.bias is not None:
                                    self.mixer.out_proj.bias.data = self.mixer.out_proj.bias.data.to(linear_dtype) + self.norm_bias.data
                                else:
                                     self.mixer.out_proj.bias = self.norm_bias
                        
                        #3.  将ln转成rmsnorm
                        if not hasattr(self,"change_ln_2_rmsnorm"):
                            self.change_ln_2_rmsnorm = True
                            norm_weight = self.norm.weight.clone()
                            self.norm = RMSNorm(self.norm.weight.shape[0], eps=self.norm.eps).to(self.norm.weight.device)
                            self.norm.weight.data = norm_weight.data
                        hidden_states = self.norm(hidden_states)
                        
                        #4.  插入旋转矩阵
                        if not hasattr(self,"R1"):
                            
                            self.R1=R1
                            self.mixer.in_proj.weight.data=self.mixer.in_proj.weight.data@self.R1.weight.T  #II: 左乘Q
                            self.mixer.out_proj.weight.data=self.R1.weight@self.mixer.out_proj.weight.data  #III:右乘QT
                            self.mixer.out_proj.bias.data = (self.mixer.out_proj.bias.data.view(1,-1)@self.R1.weight.T).view(-1)
                        
                        if self.split_head:
                            l = hidden_states.shape[1]
                            h = w = int(np.sqrt(l))
                            hidden_states = SplitHead2D.apply(hidden_states,4,h,w)
                        if skip:
                            x = self.dropout(self.mixer(hidden_states, inference_params=inference_params))
                            if hasattr(self,"R1"): #IV
                                hidden_states = ((hidden_states@self.R1.weight)*self.norm_weight@self.R1.weight.T) + self.drop_path(x)  
                            else:
                                hidden_states = hidden_states*self.norm_weight + self.drop_path(x)
                            
                        else:
                            hidden_states = self.drop_path(self.dropout(self.mixer(hidden_states, inference_params=inference_params)))
                        if self.split_head:
                            hidden_states = SplitHead2D.apply(hidden_states,4,h,w)
                    else:
                        fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
                        hidden_states, residual = fused_add_norm_fn(
                            hidden_states,
                            self.norm.weight,
                            self.norm.bias,
                            residual=residual,
                            prenorm=True,
                            residual_in_fp32=self.residual_in_fp32,
                            eps=self.norm.eps,
                        )
                        hidden_states = self.drop_path(self.mixer(hidden_states, inference_params=inference_params))
                    if self.ffn is not None:
                        hidden_states = self.ffn(self.ln2(hidden_states),identity=hidden_states)
                    if self.reverse:
                        hidden_states = hidden_states.flip(1)
                        if residual is not None:
                            residual = residual.flip(1)
                    hidden_states =  rearrange(hidden_states,f'{tgt_order}->n (t l h w ) c ',t=t,l=l,h=h,w=w)
                    return hidden_states 
                
                self.model.predict = types.MethodType(predict, self.model)
                self.model.backbone.forward = types.MethodType(mamband_forward, self.model.backbone)
                for i, layer in enumerate(self.model.backbone.layers):
                    layer.forward = types.MethodType(block_forward, layer)
            if args.use_hadmard_R4 or args.use_hadmard_R5:
                h4 = random_hadamard_matrix(self.model.backbone.layers[0].mixer.config.d_state,device)
                R4 = RotateModule(h4).weight.to(self.model.backbone.layers[0].mixer.out_proj.weight.data)
                h5 = random_hadamard_matrix(self.model.backbone.layers[0].mixer.config.dt_rank,device)
                R5 = RotateModule(h5).weight.to(self.model.backbone.layers[0].mixer.out_proj.weight.data)
                def forward(self, x, inference_params=None):
                    if self.in_proj is not None:
                        self.in_proj_states = deepcopy(self.in_proj) 
                        self.in_proj_states.weight.data = self.in_proj_states.weight.data[:self.config.d_inner]
                        self.in_proj_states.bias.data = self.in_proj_states.bias.data[:self.config.d_inner]
                        self.in_proj_states.out_features = self.config.d_inner
                        self.in_proj_gates = deepcopy(self.in_proj) 
                        self.in_proj_gates.weight.data = self.in_proj_gates.weight.data[-self.config.d_inner:]
                        self.in_proj_gates.bias.data = self.in_proj_gates.bias.data[-self.config.d_inner:]
                        self.in_proj_gates.out_features = self.config.d_inner
                        self.in_proj = None
                    if self.x_proj is not None:
                        self.x_proj_B = deepcopy(self.x_proj)
                        self.x_proj_B.weight.data = self.x_proj.weight.data[-2*self.config.d_state:-self.config.d_state]
                        self.x_proj_B.out_features = self.config.d_state
                        self.x_proj_C = deepcopy(self.x_proj)
                        self.x_proj_C.weight.data = self.x_proj.weight.data[-self.config.d_state:]
                        self.x_proj_C.out_features = self.config.d_state
                        self.x_proj_dt = deepcopy(self.x_proj)
                        self.x_proj_dt.weight.data = self.x_proj.weight.data[:-2*self.config.d_state]
                        self.x_proj_dt.out_features = self.config.dt_rank
                        self.x_proj = None
                    if not hasattr(self,"rotate_R4") and args.use_hadmard_R4:
                        self.R4 = R4
                        self.x_proj_C.weight.data = R4.T@self.x_proj_C.weight.data
                        self.rotate_R4 = True
                    if not hasattr(self,"rotate_R5") and args.use_hadmard_R5:
                        self.x_proj_dt.weight.data = R5@self.x_proj_dt.weight.data
                        self.dt_proj.weight.data = self.dt_proj.weight.data@R5.T
                        self.rotate_R5 = True


                    _, L, _ = x.shape
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
                    
                    z = self.silu_z(z)
                    y = self.mul_y_z(y, z)
                    return self.out_proj(y)
                
                def ssm(self, x, z, b_branch=False): 
                    # x : (B, L, ED)

                    # y : (B, L, ED)
                    A_log = self.A_log
                    D = self.D
                    x_proj_b = self.x_proj_B
                    x_proj_c = self.x_proj_C
                    x_proj_dt = self.x_proj_dt
                    dt_proj = self.dt_proj
                    softplus = self.softplus


                    A = -torch.exp(A_log.float()) # (ED, N)

                    D = D.float()
                 
                    delta = x_proj_dt(x)
                    B = x_proj_b(x)
                    C = x_proj_c(x)
                    # deltaBC = x_proj(x) # (B, L, dt_rank+2*N)
                    # delta, B, C = torch.split(deltaBC, [self.config.dt_rank, self.config.d_state, self.config.d_state], dim=-1) # (B, L, dt_rank), (B, L, N), (B, L, N)
                    delta, B, C = self._apply_layernorms(delta, B, C)
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

                def selective_scan_seq(self, x, delta, A, B, C, D, b_branch=False): # type: ignore

                    mul_delta_A = self.mul_delta_A
                    exp = self.exp
                    mul_delta_B = self.mul_delta_B
                    mul_deltaB_x = self.mul_deltaB_x
                    matmul = self.matmul
                    mul_D_x = self.mul_D_x
                    add_y_Dx = self.add_y_Dx
                    pscan = self.pscan

                    _, L, _ = x.shape

                    deltaA = exp(mul_delta_A(delta.unsqueeze(-1), A)) # (B, L, ED, N)
                    deltaB = mul_delta_B(delta.unsqueeze(-1), B.unsqueeze(2)) # (B, L, ED, N)

                    BX = mul_deltaB_x(deltaB, (x.unsqueeze(-1))) # (B, L, ED, N)
                    hs = pscan(deltaA,BX)
                    if hasattr(self,"R4") and self.R4 is not None:
                        hs=hs@self.R4;  #C=C@matmul.R4
                    if hasattr(matmul,"R3") and matmul.R3 is not None:
                        if  hasattr(matmul,"matmul_scale") and matmul.matmul_scale is not None:
                            hs = hs/(matmul.matmul_scale.unsqueeze(1).to(hs))
                            hs = (hs.permute(0, 1, 3, 2)@matmul.R3).permute(0, 1, 3, 2)
                            y = (matmul(hs,C.unsqueeze(-1))).squeeze(3)
                            y = y@matmul.R3.T
                            y = y*matmul.matmul_scale.to(y)           
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
                
                
                
                for i, layer in enumerate(self.model.backbone.layers):
                    layer.mixer.forward = types.MethodType(forward, layer.mixer)
                    layer.mixer.ssm = types.MethodType(ssm, layer.mixer)
                    layer.mixer.selective_scan_seq = types.MethodType(selective_scan_seq, layer.mixer)
            
        # if args.use_gptq:
        #     from quantize.gptq import gptq_fwrd_mamba3d
        #     args.nsamples=8;args.w_bits=w_cfg['n_bits'];args.w_groupsize=128
        #     args.percdamp=0.01;args.act_order=False
        #     quantizers = gptq_fwrd_mamba3d(self.model, self.test_dataloader, device, args)
        #     # torch.save(self.model,args.checkpoint[:-4]+"_gptq_weight_"+str(args.w_bits)+"bit.pt")
        #     # torch.save(quantizers,args.checkpoint[:-4]+"_gptq_scales_"+str(args.w_bits)+"bit.pt")
        #     # self.model = torch.load(args.checkpoint[:-4]+"_gptq_weight_"+str(args.w_bits)+"bit.pt",map_location='cpu')
        #     # quantizers = torch.load(args.checkpoint[:-4]+"_gptq_scales_"+str(args.w_bits)+"bit.pt",map_location='cpu')
        #     self.model.to(device)
        

        if args.static_quant:#先较准
            set_static_quant(self.model,True)
            set_observing(self.model,True)
            self.test_loop.runner.call_hook('before_test')
            self.test_loop.runner.call_hook('before_test_epoch')
            self.test_loop.runner.model.eval()
            for idx, data_batch in enumerate(self.test_loop.dataloader):
                self.test_loop.run_iter(idx, data_batch)
                if idx >= 15:
                    break
            set_observing(self.model,False)

        metrics = self.test_loop.run()  # type: ignore
        self.call_hook('after_run')
        return metrics
    with torch.no_grad():
        metrics = test(runner)
        print(metrics)
    
    # runner.test()


if __name__ == '__main__':
    main()
