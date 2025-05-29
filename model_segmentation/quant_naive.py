# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.parallel
import torch.utils.data.distributed
from networks.unetr import UNETR
from optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from trainer import run_training
from utils.data_utils import get_loader
from einops import rearrange
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss, DiceLoss
from monai.metrics import DiceMetric
from monai.transforms import Activations, AsDiscrete, Compose
from monai.utils.enums import MetricReduction
from count_parms import count_parameters
from copy import deepcopy
import wandb

def parse_args():
    parser = argparse.ArgumentParser(description="UNETR segmentation pipeline")
    parser.add_argument("--checkpoint", default=None, help="start training from saved checkpoint")
    parser.add_argument("--logdir", default="test", type=str, help="directory to save the tensorboard logs")
    parser.add_argument(
        "--pretrained_dir", default="./pretrained_models/", type=str, help="pretrained checkpoint directory"
    )
    parser.add_argument("--data_dir", default="/dataset/dataset0/", type=str, help="dataset directory")
    parser.add_argument("--json_list", default="dataset_0.json", type=str, help="dataset json file")
    parser.add_argument(
        "--pretrained_model_name", default="UNETR_model_best_acc.pth", type=str, help="pretrained model name"
    )
    parser.add_argument("--save_checkpoint", action="store_true", help="save checkpoint during training")
    parser.add_argument("--max_epochs", default=5000, type=int, help="max number of training epochs")
    parser.add_argument("--batch_size", default=1, type=int, help="number of batch size")
    parser.add_argument("--sw_batch_size", default=1, type=int, help="number of sliding window batch size")
    parser.add_argument("--optim_lr", default=1e-4, type=float, help="optimization learning rate")
    parser.add_argument("--optim_name", default="adamw", type=str, help="optimization algorithm")
    parser.add_argument("--reg_weight", default=1e-5, type=float, help="regularization weight")
    parser.add_argument("--momentum", default=0.99, type=float, help="momentum")
    parser.add_argument("--noamp", action="store_true", help="do NOT use amp for training")
    parser.add_argument("--val_every", default=100, type=int, help="validation frequency")
    parser.add_argument("--distributed", action="store_true", help="start distributed training")
    parser.add_argument("--world_size", default=1, type=int, help="number of nodes for distributed training")
    parser.add_argument("--rank", default=0, type=int, help="node rank for distributed training")
    parser.add_argument("--dist-url", default="tcp://127.0.0.1:23456", type=str, help="distributed url")
    parser.add_argument("--dist-backend", default="nccl", type=str, help="distributed backend")
    parser.add_argument("--workers", default=8, type=int, help="number of workers")
    parser.add_argument("--model_name", default="unetr", type=str, help="model name")
    parser.add_argument("--pos_embed", default="perceptron", type=str, help="type of position embedding")
    parser.add_argument("--norm_name", default="instance", type=str, help="normalization layer type in decoder")
    parser.add_argument("--num_heads", default=4, type=int, help="number of attention heads in ViT encoder") # not used by mamba, but if not divisible will make constructor unhappy
    parser.add_argument("--mlp_dim", default=3072, type=int, help="mlp dimention in ViT encoder")
    parser.add_argument("--hidden_size", default=768, type=int, help="hidden size dimention in ViT encoder")
    parser.add_argument("--feature_size", default=32, type=int, help="feature size dimention")
    parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
    parser.add_argument("--out_channels", default=14, type=int, help="number of output channels")
    parser.add_argument("--ssm_policy", default='alt', type=str, help="number of output channels")
    parser.add_argument("--res_block", action="store_true", help="use residual blocks")
    parser.add_argument("--conv_block", action="store_true", help="use conv blocks")
    parser.add_argument("--use_mamba", action="store_true", help="use conv blocks")
    parser.add_argument("--use_normal_dataset", action="store_true", help="use monai Dataset class")
    parser.add_argument("--a_min", default=-175.0, type=float, help="a_min in ScaleIntensityRanged")
    parser.add_argument("--a_max", default=250.0, type=float, help="a_max in ScaleIntensityRanged")
    parser.add_argument("--b_min", default=0.0, type=float, help="b_min in ScaleIntensityRanged")
    parser.add_argument("--b_max", default=1.0, type=float, help="b_max in ScaleIntensityRanged")
    parser.add_argument("--space_x", default=1.5, type=float, help="spacing in x direction")
    parser.add_argument("--space_y", default=1.5, type=float, help="spacing in y direction")
    parser.add_argument("--space_z", default=2.0, type=float, help="spacing in z direction")
    parser.add_argument("--roi_x", default=96, type=int, help="roi size in x direction")
    parser.add_argument("--roi_y", default=96, type=int, help="roi size in y direction")
    parser.add_argument("--roi_z", default=96, type=int, help="roi size in z direction")
    parser.add_argument("--dropout_rate", default=0.0, type=float, help="dropout rate")
    parser.add_argument("--RandFlipd_prob", default=0.2, type=float, help="RandFlipd aug probability")
    parser.add_argument("--RandRotate90d_prob", default=0.2, type=float, help="RandRotate90d aug probability")
    parser.add_argument("--RandScaleIntensityd_prob", default=0.1, type=float, help="RandScaleIntensityd aug probability")
    parser.add_argument("--RandShiftIntensityd_prob", default=0.1, type=float, help="RandShiftIntensityd aug probability")
    parser.add_argument("--infer_overlap", default=0.5, type=float, help="sliding window inference overlap")
    parser.add_argument("--lrschedule", default="warmup_cosine", type=str, help="type of learning rate scheduler")
    parser.add_argument("--warmup_epochs", default=50, type=int, help="number of warmup epochs")
    parser.add_argument("--resume_ckpt", action="store_true", help="resume training from pretrained checkpoint")
    parser.add_argument("--resume_jit", action="store_true", help="resume training from pretrained torchscript checkpoint")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--val", action="store_true")
    parser.add_argument("--smooth_dr", default=1e-6, type=float, help="constant added to dice denominator to avoid nan")
    parser.add_argument("--smooth_nr", default=0.0, type=float, help="constant added to dice numerator to avoid zero")
    parser.add_argument("--name", default=None, type=str, help="wandb name")
    parser.add_argument("--load_ckpt", default=None, type=str, help="wandb name")
    parser.add_argument("--use_smoothquant", action="store_true")
    parser.add_argument("--use_gptq", action="store_true")
    parser.add_argument("--use_hadmard", action="store_true")
    parser.add_argument('--use_hadmard_R3S', action="store_true")
    parser.add_argument('--use_hadmard_R1', action="store_true")
    parser.add_argument("--use_perkernel", action="store_true")
    parser.add_argument("--static_quant", action="store_true")
    parser.add_argument('--quant_weight', action="store_true")
    parser.add_argument('--quant_act', action="store_true")
    parser.add_argument('--w_bit', type=int,default=8)
    parser.add_argument('--a_bit', type=int,default=8)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    args.amp = not args.noamp
    args.logdir = "./runs/" + args.logdir
    if args.distributed:
        args.ngpus_per_node = torch.cuda.device_count()
        print("Found total gpus", args.ngpus_per_node)
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args,))
    else:
        main_worker(gpu=0, args=args)


def main_worker(gpu, args):
    if args.distributed:
        torch.multiprocessing.set_start_method("fork", force=True)
    np.set_printoptions(formatter={"float": "{: 0.3f}".format}, suppress=True)
    args.gpu = gpu
    if args.rank == 0:
        wandb.init(sync_tensorboard=True,name=args.name)
    if args.distributed:
        args.rank = args.rank * args.ngpus_per_node + gpu
        dist.init_process_group(
            backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank
        )
    torch.cuda.set_device(args.gpu)
    torch.backends.cudnn.benchmark = True
    args.test_mode = False
    print(args.rank, " gpu", args.gpu)
    if args.rank == 0:
        print("Batch size is:", args.batch_size, "epochs", args.max_epochs)
    inf_size = [args.roi_x, args.roi_y, args.roi_z]
    pretrained_dir = args.pretrained_dir
    if (args.model_name is None) or args.model_name == "unetr":
        model = UNETR(
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            img_size=(args.roi_x, args.roi_y, args.roi_z),
            feature_size=args.feature_size,
            hidden_size=args.hidden_size,
            mlp_dim=args.mlp_dim,
            ssm_policy=args.ssm_policy,
            num_heads=args.num_heads,
            pos_embed=args.pos_embed,
            norm_name=args.norm_name,
            conv_block=True,
            use_mamba=args.use_mamba,
            res_block=True,
            dropout_rate=args.dropout_rate,
        )
        if args.test:
            model.cuda()
            x = torch.rand(1,args.in_channels,args.roi_x, args.roi_y, args.roi_z)
            y = model(x.cuda())
            count_parameters(model.state_dict(),form_dict=True)
            print(x.shape,y.shape)
            return
        if args.resume_ckpt:
            path_load = args.load_ckpt or os.path.join(pretrained_dir, args.pretrained_model_name)
            model_dict = torch.load(path_load)['state_dict']
            r = model.load_state_dict(model_dict,strict=False)
            print(r)
            # breakpoint()
            print("Use pretrained weights")
        loader = get_loader(args)


        if args.resume_jit:
            if not args.noamp:
                print("Training from pre-trained checkpoint does not support AMP\nAMP is disabled.")
                args.amp = args.noamp
            model = torch.jit.load(os.path.join(pretrained_dir, args.pretrained_model_name))
    else:
        raise ValueError("Unsupported model " + str(args.model_name))

    dice_loss = DiceCELoss(
        to_onehot_y=True, softmax=True, squared_pred=True, smooth_nr=args.smooth_nr, smooth_dr=args.smooth_dr
    )
    post_label = AsDiscrete(to_onehot=args.out_channels, n_classes=args.out_channels)
    post_pred = AsDiscrete(argmax=True, to_onehot=args.out_channels, n_classes=args.out_channels)
    dice_acc = DiceMetric(include_background=True, reduction=MetricReduction.MEAN, get_not_nans=True)
    model_inferer = partial(
        sliding_window_inference,
        roi_size=inf_size,
        sw_batch_size=args.sw_batch_size,
        predictor=model,
        overlap=args.infer_overlap,
    )

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total parameters count", pytorch_total_params)

    best_acc = 0
    start_epoch = 0

    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        from collections import OrderedDict

        new_state_dict = OrderedDict()
        for k, v in checkpoint["state_dict"].items():
            new_state_dict[k.replace("backbone.", "")] = v
        model.load_state_dict(new_state_dict, strict=False)
        if "epoch" in checkpoint:
            start_epoch = checkpoint["epoch"]
        if "best_acc" in checkpoint:
            best_acc = checkpoint["best_acc"]
        print("=> loaded checkpoint '{}' (epoch {}) (bestacc {})".format(args.checkpoint, start_epoch, best_acc))

    model.cuda(args.gpu)

    if args.distributed:
        torch.cuda.set_device(args.gpu)
        if args.norm_name == "batch":
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model.cuda(args.gpu)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], output_device=args.gpu, find_unused_parameters=True
        )
    if args.optim_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.optim_lr, weight_decay=args.reg_weight)
    elif args.optim_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.optim_lr, weight_decay=args.reg_weight)
    elif args.optim_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.optim_lr, momentum=args.momentum, nesterov=True, weight_decay=args.reg_weight
        )
    else:
        raise ValueError("Unsupported Optimization Procedure: " + str(args.optim_name))

    if args.lrschedule == "warmup_cosine":
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer, warmup_epochs=args.warmup_epochs, max_epochs=args.max_epochs
        )
    elif args.lrschedule == "cosine_anneal":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs)
        if args.checkpoint is not None:
            scheduler.step(epoch=start_epoch)
    else:
        scheduler = None
       
    
    device = torch.device("cuda")
    from utils.normalized_modules import MatMul
    from quantize import QuantMatMul,QuantLinear,QuantConv1d,QuantConv2d,QuantConv3d
    from quantize.utils import set_quant_state,set_static_quant,set_observing
    from utils.utils import convert_vim_2_vim_torch
    convert_vim_2_vim_torch(model_inferer.keywords['predictor'].vit,"cuda")
    if args.use_smoothquant:
            model_name = args.load_ckpt.split("/")[-1].split(".")[0]
            # from quantize.smoothquant_generate_act_scale_shift import mamband_seg_generate_act_scale_shift
            # mamband_seg_generate_act_scale_shift(model_inferer,loader[1],model_name)#产生初始scale和shift值
            act_scales = torch.load("ckpt/"+model_name+"_scale.pt")
            from quantize.smoothquant import mamband_vit_mambablock_smootquant
            mamband_vit_mambablock_smootquant(model_inferer.keywords['predictor'].vit.layers,act_scales)
    
    
    w_cfg = {"dynamic_method":"per_tensor","n_bits":args.w_bit}
    # w_cfg = {"dynamic_method":"per_channel","per_channel_axes":[0],"n_bits":4}
    if args.use_perkernel:
        conv1d_w_cfg = {"dynamic_method":"per_channel","per_channel_axes":[2],"n_bits":args.w_bit}
    else:
        conv1d_w_cfg = w_cfg
    a_cfg = {"dynamic_method":"per_tensor","n_bits":args.a_bit}
    if args.use_gptq:
        from quantize.gptq import gptq_fwrd_mamba3d_seg
        args.nsamples=6;args.w_bits=w_cfg['n_bits'];args.w_groupsize=128
        args.percdamp=0.01;args.act_order=False
        # quantizers = gptq_fwrd_mamba3d_seg(model_inferer, loader[1], device, args)
        # torch.save(model_inferer.keywords['predictor'],args.load_ckpt[:-3]+"_gptq_weight_"+str(args.w_bits)+"bit.pt")
        # torch.save(quantizers,args.load_ckpt[:-3]+"_gptq_scales_"+str(args.w_bits)+"bit.pt")
        model_inferer.keywords['predictor'] = torch.load(args.load_ckpt[:-3]+"_gptq_weight_"+str(args.w_bits)+"bit.pt",map_location='cpu')
        quantizers = torch.load(args.load_ckpt[:-3]+"_gptq_scales_"+str(args.w_bits)+"bit.pt",map_location='cpu')
        model_inferer.keywords['predictor'].to('cuda')
    
    
    def replace_layers(model, target_class, replacement_class):
        for name, child in model.named_children():
            if isinstance(child, target_class):
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
    replace_layers(model_inferer.keywords['predictor'], MatMul, QuantMatMul)
    replace_layers(model_inferer.keywords['predictor'], nn.Linear, QuantLinear)
    replace_layers(model_inferer.keywords['predictor'], nn.Conv1d, QuantConv1d)
    replace_layers(model_inferer.keywords['predictor'], nn.Conv2d, QuantConv2d)
    replace_layers(model_inferer.keywords['predictor'], nn.Conv3d, QuantConv3d)
    set_quant_state(model_inferer.keywords['predictor'],weight_quant=args.quant_weight,act_quant=args.quant_act)
    
    if args.use_hadmard:
        # matmul_scale = torch.load("ckpt/"+cfg.filename.split("/")[-1].split(".")[0]+"_matmul_scale.pt")
        from quantize.hm_model_utils import fuse_layer_norms, RotateModule, RQuantLinear,RMSNorm
        from quantize.hadmard import random_hadamard_matrix
        from mmpretrain.models.utils import resize_pos_embed
        import types
        h1 = random_hadamard_matrix(model_inferer.keywords['predictor'].vit.layers[0].mixer.in_proj.in_features,device)
        R1 = RotateModule(h1)
        h3 = random_hadamard_matrix(model_inferer.keywords['predictor'].vit.layers[0].mixer.out_proj.in_features,device)
        R3 = RotateModule(h3)
        if args.use_hadmard_R3S:
            def substitute_layers(model):
                for name,module in model.named_modules():
                    if 'matmul' in name and 'quantizer' not in name:
                        if 'matmul_b' in name: continue
                        # module.register_parameter("matmul_scale",torch.nn.Parameter(matmul_scale[name]))
                        module.register_parameter("R3",R3.weight)
                        # module.register_parameter("R4",R4.weight)
            substitute_layers(model_inferer.keywords['predictor'])
        if args.use_hadmard_R1:
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
                    # if i == len(self.layers) - 1:
                    #     if hasattr(self,"R1"):x=x@self.R1.weight
                        
                    if i in self.out_indices:
                        if hasattr(self,"R1"):x=x@self.R1.weight
                        outs.append(self._format_output(x, patch_resolution))
                return outs[-1],outs
            
            def block_forward(
                self, hidden_states, residual = None, inference_params=None,order='t l h w',
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
            model_inferer.keywords['predictor'].vit.forward = types.MethodType(mamband_forward, model_inferer.keywords['predictor'].vit)
            for i, layer in enumerate(model_inferer.keywords['predictor'].vit.layers):
                layer.forward = types.MethodType(block_forward, layer)

    if args.static_quant:
        set_static_quant(model_inferer.keywords['predictor'],True)
        set_observing(model_inferer.keywords['predictor'],True)
        with torch.no_grad():
            for idx, batch_data in enumerate(loader[0]):
                if isinstance(batch_data, list):
                    data, target = batch_data
                else:
                    data, target = batch_data["image"], batch_data["label"]
                data, target = data.cuda(args.rank), target.cuda(args.rank)
                logits = model_inferer(data)
                if idx == 6:break
        set_observing(model_inferer.keywords['predictor'],False)
                

    with torch.cuda.amp.autocast():
        accuracy = run_training(
            model=model,
            train_loader=loader[0],
            val_loader=loader[1],
            optimizer=optimizer,
            loss_func=dice_loss,
            acc_func=dice_acc, # ACC is actually DICE
            args=args,
            model_inferer=model_inferer,
            scheduler=scheduler,
            start_epoch=start_epoch,
            post_label=post_label,
            post_pred=post_pred,
        )
    return accuracy


if __name__ == "__main__":
    main()
