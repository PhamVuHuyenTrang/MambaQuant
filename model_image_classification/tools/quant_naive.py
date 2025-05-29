# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import sys

from regex import F
from zmq import has
sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.getcwd()))
import os.path as osp
from copy import deepcopy

import mmengine
from mmengine.config import Config, ConfigDict, DictAction
from mmengine.evaluator import DumpResults
from mmengine.registry import RUNNERS
from mmengine.runner import Runner
import torch
import torch.nn as nn
from src.mamba import Mamba2DModel
import numpy as np
from torch import Tensor
from typing import Optional
from einops import rearrange
from quantize.utils import set_seed,Logger
set_seed(10)

# 将 sys.stdout 重定向到 Logger 类实例
sys.stdout = Logger()

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMPreTrain test (and eval) a model')
    parser.add_argument('--config', default="",help='test config file path')
    parser.add_argument('--checkpoint',default="", help='checkpoint file')
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing evaluation metrics')
    parser.add_argument('--out', help='the file to output results.')
    parser.add_argument(
        '--out-item',
        choices=['metrics', 'pred'],
        help='To output whether metrics or predictions. '
        'Defaults to output predictions.')
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
        '--amp',
        action='store_true',
        help='enable automatic-mixed-precision test')
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
        '--no-pin-memory',
        action='store_true',
        help='whether to disable the pin_memory option in dataloaders.')
    parser.add_argument(
        '--tta',
        action='store_true',
        help='Whether to enable the Test-Time-Aug (TTA). If the config file '
        'has `tta_pipeline` and `tta_model` fields, use them to determine the '
        'TTA transforms and how to merge the TTA results. Otherwise, use flip '
        'TTA by averaging classification score.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    parser.add_argument('--use_smoothquant', default=False, type=bool)
    parser.add_argument('--use_gptq', action="store_true")
    parser.add_argument('--use_hadmard', action="store_true")
    parser.add_argument('--use_hadmard_R3', action="store_true")
    parser.add_argument('--use_hadmard_R1', action="store_true")
    parser.add_argument('--use_hadmard_R4', action="store_true")
    parser.add_argument('--use_hadmard_R5', action="store_true")
    parser.add_argument('--use_S2', action="store_true")
    parser.add_argument('--use_S3', action="store_true")
    parser.add_argument('--use_S4', action="store_true")
    parser.add_argument('--use_perkernel', action="store_true")
    parser.add_argument('--use_split', action="store_true")
    parser.add_argument('--static_quant', action='store_true')
    parser.add_argument('--observe', default="minmax")
    parser.add_argument('--quant_weight', action="store_true")
    parser.add_argument('--quant_act', action="store_true")
    parser.add_argument('--w_bit', type=int,default=8)
    parser.add_argument('--a_bit', type=int,default=8)
    parser.add_argument('--fake_online_hadamard', action='store_true')

    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def merge_args(cfg, args):
    """Merge CLI arguments to config."""
    cfg.launcher = args.launcher

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    cfg.load_from = args.checkpoint

    # enable automatic-mixed-precision test
    if args.amp:
        cfg.test_cfg.fp16 = True

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

    # -------------------- TTA related args --------------------
    if args.tta:
        if 'tta_model' not in cfg:
            cfg.tta_model = dict(type='mmpretrain.AverageClsScoreTTA')
        if 'tta_pipeline' not in cfg:
            test_pipeline = cfg.test_dataloader.dataset.pipeline
            cfg.tta_pipeline = deepcopy(test_pipeline)
            flip_tta = dict(
                type='TestTimeAug',
                transforms=[
                    [
                        dict(type='RandomFlip', prob=1.),
                        dict(type='RandomFlip', prob=0.)
                    ],
                    [test_pipeline[-1]],
                ])
            cfg.tta_pipeline[-1] = flip_tta
        cfg.model = ConfigDict(**cfg.tta_model, module=cfg.model)
        cfg.test_dataloader.dataset.pipeline = cfg.tta_pipeline

    # ----------------- Default dataloader args -----------------
    default_dataloader_cfg = ConfigDict(
        pin_memory=True,
        collate_fn=dict(type='default_collate'),
    )

    def set_default_dataloader_cfg(cfg, field):
        if cfg.get(field, None) is None:
            return
        dataloader_cfg = deepcopy(default_dataloader_cfg)
        dataloader_cfg.update(cfg[field])
        cfg[field] = dataloader_cfg
        if args.no_pin_memory:
            cfg[field]['pin_memory'] = False

    set_default_dataloader_cfg(cfg, 'test_dataloader')

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    return cfg


def main():
    args = parse_args()
    print(args)
    if args.out is None and args.out_item is not None:
        raise ValueError('Please use `--out` argument to specify the '
                         'path of the output file before using `--out-item`.')

    # load config
    cfg = Config.fromfile(args.config)

    # merge cli arguments to config
    cfg = merge_args(cfg, args)

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    if args.out and args.out_item in ['pred', None]:
        runner.test_evaluator.metrics.append(
            DumpResults(out_file_path=args.out))
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
                # Replace the layer with the new quantized version
                if target_class == MatMul:
                    setattr(model, name, replacement_class(x1_quant_params=a_cfg,x2_quant_params=a_cfg,observe=args.observe))
                elif "conv1d" in name:
                    setattr(model, name, replacement_class(child,weight_quant_params=conv1d_w_cfg,act_quant_params=a_cfg,observe=args.observe))
                else:
                    setattr(model, name, replacement_class(child,weight_quant_params=w_cfg,act_quant_params=a_cfg,observe=args.observe))
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
        self.hooks[1]._swap_ema_parameters()
        convert_vim_2_vim_torch(self.model.backbone,"cuda")
        # # convert_vim_2_vim_torch(self.hooks[1].ema_model.module.backbone,"cuda")
        
        act_scales = torch.load("ckpt/"+cfg.filename.split("/")[-1].split(".")[0]+"_scale.pt")    
        if args.use_smoothquant:
            mamband_mambablock_smootquant(self.model.backbone.layers,act_scales)
            # # mamband_mambablock_smootquant(self.hooks[1].ema_model.module.backbone.layers,act_scales)
        

        replace_layers(self.model, MatMul, QuantMatMul)
        replace_layers(self.model, nn.Linear, QuantLinear)
        replace_layers(self.model, nn.Conv1d, QuantConv1d)
        replace_layers(self.model, nn.Conv2d, QuantConv2d)
        set_quant_state(self.model,weight_quant=args.quant_weight,act_quant=args.quant_act)

        if args.use_hadmard:
            from quantize.hm_model_utils import fuse_layer_norms, RotateModule, RQuantLinear,RMSNorm
            from quantize.hadmard import random_hadamard_matrix
            from mmpretrain.models.utils import resize_pos_embed
            import types
            from src.mamba import Block
            h1 = random_hadamard_matrix(self.model.backbone.layers[0].mixer.in_proj.in_features,device)
            R1 = RotateModule(h1)
            matmul_scale = torch.load("ckpt/"+cfg.filename.split("/")[-1].split(".")[0]+"_matmul_scale.pt")
            h3 = random_hadamard_matrix(self.model.backbone.layers[0].mixer.out_proj.in_features,device)
            R3 = RotateModule(h3)
            if args.fake_online_hadamard:
                R3.weight.data = R3.weight.data.T
                R1.weight.data = R1.weight.data.T
            
            if args.use_S2:
                class Swiglu(nn.Module):
                    def __init__(self, s):
                        super().__init__()
                        self.s = s
                        self.sigmod = nn.Sigmoid()
                    def forward(self, x):
                        return x*self.sigmod(x*self.s)

                for i,layer in enumerate(self.model.backbone.layers):    
                    act = act_scales[f"backbone.layers.{i}.mixer.out_proj"].to(device=device)
                    weight_scales = layer.mixer.out_proj.weight.abs().max(dim=0, keepdim=True)[0].clamp(min=1e-5)
                    alpha = 0.5
                    scales = ((act.pow(alpha) / weight_scales.pow(1 - alpha)).clamp(min=1e-2).to(device))
                    oc,ic = layer.mixer.in_proj.weight.data.shape
                    layer.mixer.in_proj.weight.data[oc//2:] = \
                        (1/scales.reshape(-1,1))*layer.mixer.in_proj.weight.data[oc//2:] 
                    layer.mixer.out_proj.weight.data = scales*layer.mixer.out_proj.weight.data
                    layer.mixer.silu_z = Swiglu(scales)

            if args.use_hadmard_R3:
                def substitute_layers(model):
                    for name,module in model.named_modules():
                        if 'matmul' in name and 'quantizer' not in name:
                            if 'matmul_b' in name: continue
                            module.register_parameter("matmul_scale",torch.nn.Parameter(matmul_scale[name]))
                            module.register_parameter("R3",R3.weight)
                            # module.register_parameter("R4",R4.weight)
                substitute_layers(self.model)
            
            if args.use_hadmard_R1:
                MambaND = nn.Identity

                def predict(self,
                    inputs: torch.Tensor,
                    data_samples = None,
                    **kwargs):
                    if not hasattr(self,"R1"):
                        self.R1 = R1
                        self.head.fc.weight.data = self.head.fc.weight.data@self.R1.weight.T
                    feats = self.extract_feat(inputs)
                    feats = tuple([feats[0]@self.R1.weight.T])
                    return self.head.predict(feats, data_samples, **kwargs)
                
                def mamba2d_forward(self, x):
                    B = x.shape[0]
                    #1.  将ln的减均值操作吸收到weight里
                    if not hasattr(self,"obsorted_mean_2_linear"):
                        self.patch_embed.projection.weight.data = self.patch_embed.projection.weight.data - self.patch_embed.projection.weight.data.mean(dim=0,keepdim=True)
                        self.patch_embed.projection.bias.data = self.patch_embed.projection.bias.data - self.patch_embed.projection.bias.data.mean(dim=0,keepdim=True)
                        
                        
                        for i, layer in enumerate(self.layers):
                            layer.mixer.out_proj.weight.data = layer.mixer.out_proj.weight.data - layer.mixer.out_proj.weight.data.mean(dim=0,keepdim=True)
                            if i == 9:
                                linear = layer.down_sample_layer.reduction
                                linear.weight.data = linear.weight.data - linear.weight.data.mean(dim=0,keepdim=True)
                                
                        self.obsorted_mean_2_linear = True         
                    
                    if not hasattr(self, 'R1'):
                        self.R1=R1
                        self.patch_embed.projection.weight.data = self.R1.weight@self.patch_embed.projection.weight.data  #I: 右乘QT
                        self.patch_embed.projection.bias.data = (self.patch_embed.projection.bias.data.view(1,-1)@self.R1.weight.T).view(-1)
                    
                    x, patch_resolution = self.patch_embed(x)
                    pos = resize_pos_embed(
                        self.pos_embed,
                        self.patch_resolution,
                        patch_resolution,
                        mode=self.interpolate_mode,
                        num_extra_tokens=self.num_extra_tokens)[:,self.num_extra_tokens:]
                    pos = pos-pos.mean(dim=-1,keepdim=True)
                    if hasattr(self, 'R1'):pos = pos@self.R1.weight.T
                    x = x + pos
                    x = self.drop_after_pos(x)

                    x = self.pre_norm(x)
                    h = patch_resolution[0]
                    w = patch_resolution[1]
                    outs = []
                    residual = None

                    for i, layer in enumerate(self.layers):
                        layer.index = i
                        x,residual = layer(x,residual,h=h,w=w)
                        if layer.downsample:
                            h,w = residual
                            residual = None

                        if i == len(self.layers) - 1:
                            if hasattr(self,"R1"):x=x@self.R1.weight  #V:激活右乘QT
                            x = (x + residual) if residual is not None else x
                            
                                
                        if i == len(self.layers) - 1 and self.final_norm:
                            if isinstance(self.ln1,tuple):
                                x = self.ln1[1].to(x)(x)
                            else:
                                x = self.ln1(x)

                        if i in self.out_indices:
                            outs.append(self._format_output(x, patch_resolution))

                    return tuple(outs)

                def block_forward( self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None,skip=True,**kwargs
                ):
                    r"""Pass the input through the encoder layer.

                    Args:
                        hidden_states: the sequence to the encoder layer (required).
                        residual: hidden_states = Mixer(LN(residual))
                    """
                    h = w = 0
                    if self.transpose:
                        l = hidden_states.shape[1]
                        h = w = int(np.sqrt(l))
                        # assert h * w == l
                        hidden_states = rearrange(hidden_states,'n (h w) c -> n (w h) c',h=h,w=w)
                        if residual is not None:
                            residual = rearrange(residual,'n (h w) c -> n (w h) c',h=h,w=w)
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

                        # hidden_states = hidden_states - hidden_states.mean(dim=-1,keepdim=True)
                        
                        #3.  将ln转成rmsnorm
                        if not hasattr(self,"change_ln_2_rmsnorm"):
                            self.change_ln_2_rmsnorm = True
                            norm_weight = self.norm.weight.clone()
                            self.norm = RMSNorm(self.norm.weight.shape[0], eps=self.norm.eps).to(self.norm.weight.device)
                            self.norm.weight.data = norm_weight.data
                        hidden_states = self.norm(hidden_states)
                        
                        #4.  插入旋转矩阵
                        if not hasattr(self,"R1"):
                            if self.index <= 23:
                                self.R1=R1
                                self.mixer.in_proj.weight.data=self.mixer.in_proj.weight.data@self.R1.weight.T  #II: 左乘Q
                                self.mixer.out_proj.weight.data=self.R1.weight@self.mixer.out_proj.weight.data  #III:右乘QT
                                self.mixer.out_proj.bias.data = (self.mixer.out_proj.bias.data.view(1,-1)@self.R1.weight.T).view(-1)
                                if self.downsample:
                                    self.down_sample_layer.reduction.weight.data = self.R1.weight@self.down_sample_layer.reduction.weight.data  #VI:  卷积层右乘QT
                            
                        if skip:
                            x = self.mixer(hidden_states, inference_params=inference_params,**(kwargs if isinstance(self.mixer,MambaND) else {}))    
                            if hasattr(self,"R1"):
                                hidden_states = ((hidden_states@self.R1.weight)*self.norm_weight@self.R1.weight.T) + self.drop_path(x)  #II: 右乘Q     #III: 右乘QT
                            else:
                                hidden_states = hidden_states*self.norm_weight + self.drop_path(x)
                            # if self.index==23:hidden_states = hidden_states@R1.weight
                        else:
                            hidden_states = self.drop_path(self.dropout(self.mixer(hidden_states, inference_params=inference_params)))
                    if self.reverse:
                        hidden_states = hidden_states.flip(1)
                        if residual is not None:
                            residual = residual.flip(1)
                    if self.transpose:
                        hidden_states = rearrange(hidden_states,'n (w h) c -> n (h w) c',h=h,w=w)
                        if residual is not None:
                            residual = rearrange(residual,'n (w h) c -> n (h w) c',h=h,w=w)
                    if self.downsample:
                        if 'h' in kwargs:
                            h,w = kwargs['h'],kwargs['w']
                        if hasattr(self,'R1'):
                            hidden_states = hidden_states@self.R1.weight  #IV:  激活右乘QT
                        hidden_states,(h,w) = self.down_sample_layer(hidden_states,(h,w))
                       
                        assert residual is None
                        residual = (h,w)
                    return hidden_states, residual
                
                self.model.predict = types.MethodType(predict, self.model)
                self.model.backbone.forward = types.MethodType(mamba2d_forward, self.model.backbone)
                for i, layer in enumerate(self.model.backbone.layers):
                    if isinstance(layer, Block):
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
                    if isinstance(layer, Block):
                        layer.mixer.forward = types.MethodType(forward, layer.mixer)
                        layer.mixer.ssm = types.MethodType(ssm, layer.mixer)
                        layer.mixer.selective_scan_seq = types.MethodType(selective_scan_seq, layer.mixer)
            

        if args.use_gptq:
            from quantize.gptq import gptq_fwrd_mamba2d
            args.nsamples=8;args.w_bits=w_cfg['n_bits'];args.w_groupsize=128
            args.percdamp=0.01;args.act_order=False
            quantizers = gptq_fwrd_mamba2d(self.model, self.test_dataloader, device, args)
            # torch.save(self.model,args.checkpoint[:-4]+"_gptq_weight_"+str(args.w_bits)+"bit_smoothed.pt")
            # torch.save(quantizers,args.checkpoint[:-4]+"_gptq_scales_"+str(args.w_bits)+"bit_smoothed.pt")
            # self.model = torch.load(args.checkpoint[:-4]+"_gptq_weight_"+str(args.w_bits)+"bit.pt",map_location='cpu')
            # quantizers = torch.load(args.checkpoint[:-4]+"_gptq_scales_"+str(args.w_bits)+"bit.pt",map_location='cpu')
            self.model.to(device)

        if args.static_quant:#先较准
            set_static_quant(self.model,True)
            set_observing(self.model,True)
            from model_image_classification.utils.datasets import build_dataset
            from utils.utils import evaluate
            from torch.utils.data import Subset
            from contextlib import suppress
            args.data_set = 'IMNET'
            args.data_path = "/data01/datasets/imagenet"
            dataset_val, _ = build_dataset(is_train=False, args=args)
            subset_indices = list(range(1,50000,200))#校准数据选择
            calibration = Subset(dataset_val, subset_indices)
            calibration = torch.utils.data.DataLoader(
                calibration, 
                batch_size=64,
                num_workers=4,
                pin_memory=True,
                drop_last=False
            )
            test_stats = evaluate(calibration, self.model, device, suppress)
            print(f"Fp Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
            set_observing(self.model,False)


        metrics = self.test_loop.run()  # type: ignore
        self.call_hook('after_run')
        return metrics
    
    
    with torch.no_grad():
        metrics = test(runner)
    # metrica = runner.test()

    if args.out and args.out_item == 'metrics':
        mmengine.dump(metrics, args.out)


if __name__ == '__main__':
    main()
