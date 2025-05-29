# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import argparse
import datetime
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import json

from pathlib import Path

from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma

import os
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.getcwd()))

from vim.datasets import build_dataset
from vim.engine import train_one_epoch, evaluate
from vim.losses import DistillationLoss
from vim.samplers import RASampler
from vim.augment import new_data_aug_generator

from contextlib import suppress

import vim.models_mamba

from vim import utils

# log about
import mlflow

from quantize.utils import Logger
folder = 'logs'
sys.stdout = Logger(folder=folder)

def get_args_parser():
    parser = argparse.ArgumentParser('DeiT training and evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--bce-loss', action='store_true')
    parser.add_argument('--unscale-lr', action='store_true')

    # Model parameters
    parser.add_argument('--model', default='deit_base_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input-size', default=224, type=int, help='images input size')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    parser.add_argument('--model-ema', action='store_true')
    parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
    parser.set_defaults(model_ema=True)
    parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.3, metavar='PCT',
                        help='Color jitter factor (default: 0.3)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=True)
    
    parser.add_argument('--train-mode', action='store_true')
    parser.add_argument('--no-train-mode', action='store_false', dest='train_mode')
    parser.set_defaults(train_mode=True)
    
    parser.add_argument('--ThreeAugment', action='store_true') #3augment
    
    parser.add_argument('--src', action='store_true') #simple random crop
    
    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # Distillation parameters
    parser.add_argument('--teacher-model', default='regnety_160', type=str, metavar='MODEL',
                        help='Name of teacher model to train (default: "regnety_160"')
    parser.add_argument('--teacher-path', type=str, default='')
    parser.add_argument('--distillation-type', default='none', choices=['none', 'soft', 'hard'], type=str, help="")
    parser.add_argument('--distillation-alpha', default=0.5, type=float, help="")
    parser.add_argument('--distillation-tau', default=1.0, type=float, help="")
    
    # * Cosub params
    parser.add_argument('--cosub', action='store_true') 
    
    # * Finetuning params
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')
    parser.add_argument('--attn-only', action='store_true') 
    
    # Dataset parameters
    parser.add_argument('--data-path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--data-set', default='IMNET', choices=['CIFAR', 'IMNET', 'INAT', 'INAT19'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--inat-category', default='name',
                        choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--eval-crop-ratio', default=0.875, type=float, help="Crop ratio for evaluation")
    parser.add_argument('--dist-eval', action='store_true', default=False, help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--distributed', action='store_true', default=False, help='Enabling distributed training')
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    
    # amp about
    parser.add_argument('--if_amp', action='store_true')
    parser.add_argument('--no_amp', action='store_false', dest='if_amp')
    parser.set_defaults(if_amp=True)

    # if continue with inf
    parser.add_argument('--if_continue_inf', action='store_true')
    parser.add_argument('--no_continue_inf', action='store_false', dest='if_continue_inf')
    parser.set_defaults(if_continue_inf=False)

    # if use nan to num
    parser.add_argument('--if_nan2num', action='store_true')
    parser.add_argument('--no_nan2num', action='store_false', dest='if_nan2num')
    parser.set_defaults(if_nan2num=False)

    # if use random token position
    parser.add_argument('--if_random_cls_token_position', action='store_true')
    parser.add_argument('--no_random_cls_token_position', action='store_false', dest='if_random_cls_token_position')
    parser.set_defaults(if_random_cls_token_position=False)    

    # if use random token rank
    parser.add_argument('--if_random_token_rank', action='store_true')
    parser.add_argument('--no_random_token_rank', action='store_false', dest='if_random_token_rank')
    parser.set_defaults(if_random_token_rank=False)
    parser.add_argument('--local-rank', default=0, type=int)

    parser.add_argument('--use_vim_torch', default=False, type=bool)
    parser.add_argument('--static_quant', action='store_true')
    parser.add_argument('--observe', default="minmax")
    parser.add_argument('--quant_weight', action='store_true')
    parser.add_argument('--quant_act', action='store_true')
    parser.add_argument('--a_bit', default=8, type=int)
    parser.add_argument('--w_bit', default=8, type=int)
    parser.add_argument('--use_smoothquant', default=False, type=bool)
    parser.add_argument('--use_gptq', action='store_true')
    parser.add_argument('--use_hadmard', action='store_true')
    parser.add_argument('--use_S1', action='store_true')
    parser.add_argument('--use_S2', action='store_true')
    parser.add_argument('--use_S3', action='store_true')
    parser.add_argument('--use_S4', action='store_true')
    parser.add_argument('--use_S5', action='store_true')
    parser.add_argument('--use_S7', action='store_true')
    parser.add_argument('--use_hadmard_R1', action='store_true')
    parser.add_argument('--use_hadmard_R2', action='store_true')
    parser.add_argument('--use_hadmard_R3', action='store_true')
    parser.add_argument('--use_hadmard_R4', action='store_true')
    parser.add_argument('--use_hadmard_R5', action='store_true')
    parser.add_argument('--use_hadmard_R6', action='store_true')
    parser.add_argument('--use_reduce_mean', action='store_true')
    parser.add_argument('--use_pertoken', action='store_true')
    parser.add_argument('--use_split', action='store_true')
    parser.add_argument('--use_klt', action='store_true')
    parser.add_argument('--generate_klt', action='store_true')
    parser.add_argument('--use_perkernel', action='store_true')
    parser.add_argument('--w_perchannel', action='store_true')
    parser.add_argument('--fake_online_hadamard', action='store_true')
    parser.add_argument('--analyse_and_plot', action='store_true')
    parser.add_argument('--use_adaround', action='store_true')
    # parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--adaround-iter', default=200, type=int)
    parser.add_argument('--b_start', default=20, type=int, help='temperature at the beginning of calibration')
    parser.add_argument('--b_end', default=2, type=int, help='temperature at the end of calibration')
    parser.add_argument('--warmup', default=0.2, type=float, help='in the warmup period no regularization is applied')
    
    return parser


def main(args):
    utils.init_distributed_mode(args)
    
    for var_name, var_value in vars(args).items():
        print(f"{var_name}: {var_value}")

    if args.analyse_and_plot:
        args.batch_size = 66

    if "base" in args.model:args.batch_size = args.batch_size // 2
    
    if args.distillation_type != 'none' and args.finetune and not args.eval:
        raise NotImplementedError("Finetuning with distillation not yet supported")

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    cudnn.benchmark = True

    # log about
    run_name = args.output_dir.split("/")[-1]
    if args.local_rank == 0:
        mlflow.start_run(run_name=run_name)
        for key, value in vars(args).items():
            mlflow.log_param(key, value)

    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    dataset_val, _ = build_dataset(is_train=False, args=args)

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        if args.repeated_aug:
            sampler_train = RASampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        else:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    if args.ThreeAugment:
        data_loader_train.dataset.transform = new_data_aug_generator(args)

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=False,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        img_size=args.input_size
    )

                    
    if args.finetune:
        if args.finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.finetune, map_location='cpu')

        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        # only the position tokens are interpolated
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        checkpoint_model['pos_embed'] = new_pos_embed

        model.load_state_dict(checkpoint_model, strict=False)
        
    if args.attn_only:
        for name_p,p in model.named_parameters():
            if '.attn.' in name_p:
                p.requires_grad = True
            else:
                p.requires_grad = False
        try:
            model.head.weight.requires_grad = True
            model.head.bias.requires_grad = True
        except:
            model.fc.weight.requires_grad = True
            model.fc.bias.requires_grad = True
        try:
            model.pos_embed.requires_grad = True
        except:
            print('no position encoding')
        try:
            for p in model.patch_embed.parameters():
                p.requires_grad = False
        except:
            print('no patch embed')
            
    model.to(device)

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    if not args.unscale_lr:
        linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 512.0
        args.lr = linear_scaled_lr
    optimizer = create_optimizer(args, model_without_ddp)
    
    # amp about
    amp_autocast = suppress
    loss_scaler = "none"
    if args.if_amp:
        amp_autocast = torch.cuda.amp.autocast
        loss_scaler = NativeScaler()

    lr_scheduler, _ = create_scheduler(args, optimizer)

    criterion = LabelSmoothingCrossEntropy()

    if mixup_active:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()
        
    if args.bce_loss:
        criterion = torch.nn.BCEWithLogitsLoss()
        
    teacher_model = None
    if args.distillation_type != 'none':
        assert args.teacher_path, 'need to specify teacher-path when using distillation'
        print(f"Creating teacher model: {args.teacher_model}")
        teacher_model = create_model(
            args.teacher_model,
            pretrained=False,
            num_classes=args.nb_classes,
            global_pool='avg',
        )
        if args.teacher_path.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.teacher_path, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.teacher_path, map_location='cpu')
        teacher_model.load_state_dict(checkpoint['model'])
        teacher_model.to(device)
        teacher_model.eval()

    # wrap the criterion in our custom DistillationLoss, which
    # just dispatches to the original criterion if args.distillation_type is 'none'
    criterion = DistillationLoss(
        criterion, teacher_model, args.distillation_type, args.distillation_alpha, args.distillation_tau
    )

    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            if args.model_ema:
                utils._load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])
            if 'scaler' in checkpoint and args.if_amp: # change loss_scaler if not amp
                loss_scaler.load_state_dict(checkpoint['scaler'])
            elif 'scaler' in checkpoint and not args.if_amp:
                loss_scaler = 'none'
        lr_scheduler.step(args.start_epoch)
        
    if args.use_vim_torch:
        from vim.utils import convert_vim_2_vim_torch
        convert_vim_2_vim_torch(model,device)
   
    if args.w_perchannel:
        w_cfg = {"dynamic_method":"per_channel","per_channel_axes":[0],"n_bits":args.w_bit}
    else:
        w_cfg = {"dynamic_method":"per_tensor","n_bits":args.w_bit}
 
    # conv1d_w_cfg = w_cfg
    if args.use_perkernel:
        conv1d_w_cfg = {"dynamic_method":"per_channel","per_channel_axes":[3],"n_bits":args.w_bit}
    else:
        conv1d_w_cfg = w_cfg
    a_cfg = {"dynamic_method":"per_tensor","n_bits":args.a_bit}

    import copy
    if args.use_adaround:
        fp_model = copy.deepcopy(model)

    ##if you want to use smoothquant, you need blow 3 lines
    if args.use_smoothquant:
        from quantize.smoothquant import vim_mambablock_smootquant
        act_scales = torch.load("saved_checkpoint/"+"-".join(args.model.split("_")[:2])+"_scale.pt")
        vim_mambablock_smootquant(model.layers,act_scales)
        
    #ptq
    from vim.normalized_modules import MatMul,MulAdd
    from quantize import QuantMatMul,QuantLinear,QuantConv1d,QuantConv2d
    def replace_layers(model, target_class, replacement_class):
        for name, child in model.named_children():
            if  "patch" in name:
                continue
            if isinstance(child, target_class):
                # Replace the layer with the new quantized version
                if target_class == MatMul:
                    if args.use_pertoken:
                        setattr(model, name, replacement_class(
                            x1_quant_params={"dynamic_method":"per_channel","per_channel_axes":[1],"n_bits":8,"percent":0.999},
                            x2_quant_params={"dynamic_method":"per_channel","per_channel_axes":[1],"n_bits":8,"percent":0.999},
                            observe="minmax"))#args.observe
                    else:
                        setattr(model, name, replacement_class(x1_quant_params=a_cfg,x2_quant_params=a_cfg,observe=args.observe))
                elif "conv1d" in name:
                    setattr(model, name, replacement_class(child,weight_quant_params=conv1d_w_cfg,act_quant_params=a_cfg,observe=args.observe))
                else:
                    setattr(model, name, replacement_class(child,weight_quant_params=w_cfg,act_quant_params=a_cfg,observe=args.observe))
            else:
                # Recursively call this function on the child module
                replace_layers(child, target_class, replacement_class)

    # Usage example:
    # Assuming QuantMatMul, QuantLinear, QuantConv1d, QuantConv2d are defined
    replace_layers(model, MatMul, QuantMatMul)
    replace_layers(model, nn.Linear, QuantLinear)
    replace_layers(model, nn.Conv1d, QuantConv1d)
    replace_layers(model, nn.Conv2d, QuantConv2d)
    from quantize.utils import set_quant_state,set_static_quant,set_observing
    
    set_quant_state(model,weight_quant=args.quant_weight,act_quant=args.quant_act) 
    
    from quantize.hm_model_utils import fuse_layer_norms, fuse_layer_norms_2, RotateModule, RQuantLinear
    from quantize.hadmard import random_hadamard_matrix,random_walsh_matrix
    # from quantize.hadmard import random_walsh_matrix as random_hadamard_matrix

    if args.use_klt:
        if os.path.exists("saved_checkpoint/"+"-".join(args.model.split("_")[:2])+"_act_klt.pt"):
            matmul_klt = torch.load("saved_checkpoint/"+"-".join(args.model.split("_")[:2])+"_act_klt.pt",map_location=device)
        else:
            from quantize.get_klt_matrix import vim_generate_act_klt
            matmul_klt = vim_generate_act_klt(model,data_loader_val,args)
            torch.save(matmul_klt,"saved_checkpoint/"+"-".join(args.model.split("_")[:2])+"_act_klt.pt")

    if args.use_hadmard:

        from vim.vim_torch import MambaBlock_optimize
        for layer in model.layers:
            layer.mixer = MambaBlock_optimize(layer.mixer)

        if not os.path.exists("saved_checkpoint/"+"-".join(args.model.split("_")[:2])+"_scale.pt"):
            from quantize.smoothquant_generate_act_scale_shift import vim_generate_act_scale_shift
            act_scales,act_shifts = vim_generate_act_scale_shift(args)
            torch.save(act_scales,"saved_checkpoint/"+"-".join(args.model.split("_")[:2])+"_scale.pt")
            torch.save(act_shifts,"saved_checkpoint/"+"-".join(args.model.split("_")[:2])+"_shift.pt")
        else:
            act_scales = torch.load("saved_checkpoint/"+"-".join(args.model.split("_")[:2])+"_scale.pt")
            act_shifts = torch.load("saved_checkpoint/"+"-".join(args.model.split("_")[:2])+"_shift.pt")
        S3_scale = torch.load("saved_checkpoint/"+"-".join(args.model.split("_")[:2])+"_matmul_scale_S3.pt",map_location=device)
        S4_scales = torch.load("saved_checkpoint/"+"-".join(args.model.split("_")[:2])+"_matmul_scale_S4.pt")
        device, dtype = model.layers[0].mixer.out_proj.weight.device, model.layers[0].mixer.out_proj.weight.dtype

        if args.use_reduce_mean:
            for i,layer in enumerate(model.layers): 
                shift = act_shifts[f"layers.{i}.mixer.dt_proj"].to(device=device, dtype=dtype)
                layer.mixer.x_proj_dt.bias = nn.Parameter(-shift).to(device).to(dtype)
                layer.mixer.dt_proj.bias.data = layer.mixer.dt_proj.bias.data + \
                    (shift@layer.mixer.dt_proj.weight.data.T).data.reshape(-1)

                shift_b = act_shifts[f"layers.{i}.mixer.dt_proj_b"].to(device=device, dtype=dtype)
                layer.mixer.x_proj_dt_b.bias = nn.Parameter(-shift_b).to(device).to(dtype)
                layer.mixer.dt_proj_b.bias.data = layer.mixer.dt_proj_b.bias.data + \
                    (shift_b@layer.mixer.dt_proj_b.weight.data.T).data.reshape(-1)
                
            #需要重新较准之前生成的dt_proj的R5的scale
            # from quantize.smoothquant_generate_act_scale_shift import get_act_scales
            # sampler_val=torch.utils.data.SequentialSampler(dataset_val)
            # tmp_data_loader_val = torch.utils.data.DataLoader(
            #     dataset_val, sampler=sampler_val,
            #     batch_size=int(128),
            #     num_workers=32,
            #     pin_memory=True,
            #     drop_last=False
            # )
            # set_quant_state(model,weight_quant=False,act_quant=False)
            # tmp_scale = get_act_scales(model,tmp_data_loader_val)
            # set_quant_state(model,weight_quant=args.quant_weight,act_quant=args.quant_act)
            # for key,val in tmp_scale.items():
            #     if "dt_proj" in key:
            #         act_scales[key] = val

        if args.use_S1:
            for i,layer in enumerate(model.layers): 
                act = act_scales[f"layers.{i}.mixer.in_proj"].to(device=device, dtype=dtype)
                weight_states = layer.mixer.in_proj_states.weight.data
                weight_gates  = layer.mixer.in_proj_gates.weight.data
                weight = torch.cat([weight_states,weight_gates],dim=0)
                weight_scales = weight.abs().max(dim=0, keepdim=True)[0].reshape(-1).clamp(min=1e-5)
                alpha = 0.5
                scales = ((act.pow(alpha) / weight_scales.pow(1 - alpha)).clamp(min=1e-2).to(device).to(dtype))
                layer.norm.weight.data = (1/scales.reshape(-1))*layer.norm.weight.data
                
                layer.mixer.in_proj_states.weight.data = scales*layer.mixer.in_proj_states.weight.data
                layer.mixer.in_proj_gates.weight.data = scales*layer.mixer.in_proj_gates.weight.data

        if args.use_S2:
                class Swiglu(nn.Module):
                    def __init__(self, s):
                        super().__init__()
                        self.s = s
                        self.sigmod = nn.Sigmoid()
                    def forward(self, x):
                        return x*self.sigmod(x*self.s)

                for i,layer in enumerate(model.layers):    
                    act = act_scales[f"layers.{i}.mixer.out_proj"].to(device=device, dtype=dtype)
                    weight_scales = layer.mixer.out_proj.weight.abs().max(dim=0, keepdim=True)[0].clamp(min=1e-5)
                    alpha = 0.5
                    scales = ((act.pow(alpha) / weight_scales.pow(1 - alpha)).clamp(min=1e-2).to(device).to(dtype))
                    layer.register_parameter("s1", nn.Parameter(scales))
                    layer.mixer.register_parameter("s1", nn.Parameter(scales))
                
                for layer in model.layers:
                    layer.mixer.in_proj_gates.weight.data = (1/layer.s1.reshape(-1,1))*layer.mixer.in_proj_gates.weight.data
                    layer.mixer.out_proj.weight.data = layer.s1*layer.mixer.out_proj.weight.data
                    layer.mixer.silu_z_b= Swiglu(layer.s1)
                    layer.mixer.silu_z = Swiglu(layer.s1)

        if args.use_S3:
            for name,module in model.named_modules():
                if 'matmul' in name  and 'quantizer' not in name:    
                    module.register_parameter("S3",torch.nn.Parameter(S3_scale[name]))

        if args.use_S4:
            for i,layer in enumerate(model.layers):
                name = f"layers.{i}.mixer.matmul"
                s4 = S4_scales[name].clamp(min=1e-2,max=100).to(device).to(dtype)
                layer.mixer.x_proj_C.weight.data = (s4.reshape(-1,1))*layer.mixer.x_proj_C.weight.data
                layer.mixer.x_proj_B.weight.data = (1/s4.reshape(-1,1))*layer.mixer.x_proj_B.weight.data
                layer.mixer.mul_delta_A = MulAdd(torch.log(1/s4))
                
                layer.mixer.x_proj_C_b.weight.data = (s4.reshape(-1,1))*layer.mixer.x_proj_C_b.weight.data
                layer.mixer.x_proj_B_b.weight.data = (1/s4.reshape(-1,1))*layer.mixer.x_proj_B_b.weight.data
                layer.mixer.mul_delta_A_b = MulAdd(torch.log(1/s4))
                
            
            # for name,module in model.named_modules():
            #     if 'matmul' in name and 'quantizer' not in name:    
            #         module.register_parameter("S4",torch.nn.Parameter(S4_scales[name]))
                    
        if args.use_S5:
            for i,layer in enumerate(model.layers): 
                act = act_scales[f"layers.{i}.mixer.dt_proj"].to(device=device, dtype=dtype)
                weight_scales = layer.mixer.dt_proj.weight.abs().max(dim=0, keepdim=True)[0].reshape(-1).clamp(min=1e-5)
                alpha = 0.5
                scales = ((act.pow(alpha) / weight_scales.pow(1 - alpha)).clamp(min=1e-2).to(device).to(dtype))
                if layer.mixer.x_proj_dt.bias is not None:
                    layer.mixer.x_proj_dt.bias.data = layer.mixer.x_proj_dt.bias.data*(1/scales)
                layer.mixer.x_proj_dt.weight.data = (1/scales.reshape(-1,1))*layer.mixer.x_proj_dt.weight.data
                layer.mixer.dt_proj.weight.data = scales*layer.mixer.dt_proj.weight.data

                act = act_scales[f"layers.{i}.mixer.dt_proj_b"].to(device=device, dtype=dtype)
                weight_scales = layer.mixer.dt_proj_b.weight.abs().max(dim=0, keepdim=True)[0].reshape(-1).clamp(min=1e-5)
                alpha = 0.5
                scales = ((act.pow(alpha) / weight_scales.pow(1 - alpha)).clamp(min=1e-5).to(device).to(dtype))
                if layer.mixer.x_proj_dt_b.bias is not None:
                    layer.mixer.x_proj_dt_b.bias.data = layer.mixer.x_proj_dt_b.bias.data*(1/scales)
                layer.mixer.x_proj_dt_b.weight.data = (1/scales.reshape(-1,1))*layer.mixer.x_proj_dt_b.weight.data
                layer.mixer.dt_proj_b.weight.data = scales*layer.mixer.dt_proj_b.weight.data
                
        if args.use_S7:
            for i,layer in enumerate(model.layers):    
                act = act_scales[f"layers.{i}.mixer.conv1d"].to(device=device, dtype=dtype)
                weight_scales = (layer.mixer.conv1d.weight.abs().max(dim=-1)[0]).reshape(-1).clamp(min=1e-5)
                alpha = 0.5
                scales = ((act.pow(alpha) / weight_scales.pow(1 - alpha)).clamp(min=1e-2).to(device).to(dtype))
                layer.mixer.in_proj_states.weight.data = (1/scales.reshape(-1,1))*layer.mixer.in_proj_states.weight.data
                layer.mixer.conv1d.weight.data = scales.reshape(-1,1,1,1)*layer.mixer.conv1d.weight.data
                layer.mixer.conv1d_b.weight.data = scales.reshape(-1,1,1,1)*layer.mixer.conv1d_b.weight.data
        
        R1 = random_hadamard_matrix(model.layers[0].mixer.in_proj_gates.in_features,device).to(device=device, dtype=dtype)
        R2 = random_hadamard_matrix(model.layers[0].mixer.out_proj.in_features,device).to(device=device, dtype=dtype)
        R3 = random_hadamard_matrix(model.layers[0].mixer.out_proj.in_features,device).to(device=device, dtype=dtype)
        R3_ = random_walsh_matrix(model.layers[0].mixer.out_proj.in_features,device).to(device=device, dtype=dtype)
        R4 = random_hadamard_matrix(model.layers[0].mixer.config.d_state,device).to(device=device, dtype=dtype)
        R5 = random_hadamard_matrix(model.layers[0].mixer.config.dt_rank,device).to(device=device, dtype=dtype)
        R6 = random_hadamard_matrix(model.layers[0].mixer.x_proj_B.in_features,device).to(device=device, dtype=dtype)
        if args.fake_online_hadamard:
            # R1=R1.T
            R2=R2.T
            R3=R3.T
            R4=R4.T
            # R5=R5.T
            R6=R6.T

        if args.use_hadmard_R1:
            if args.use_klt:
                length = len(model.layers)
                name = f"layers.{length-1}.mixer.in_proj_gates"
                if name not in matmul_klt:
                    name = f"layers.{length-1}.mixer.in_proj"
                K = matmul_klt[name].to(device=device, dtype=dtype)
            else:
                K = torch.eye(R1.shape[0]).to(R1)
                
            R1 = K@R1
            if hasattr(model.layers[0].mixer,"in_proj"):
                fuse_layer_norms(model)
            else:
                fuse_layer_norms_2(model)
            for i,layer in enumerate(model.layers):
                if hasattr(layer.mixer,"in_proj"):
                    layer.mixer.in_proj.weight.data = layer.mixer.in_proj.weight.data@R1
                else:
                    layer.mixer.in_proj_states.weight.data = layer.mixer.in_proj_states.weight.data@R1
                    layer.mixer.in_proj_gates.weight.data = layer.mixer.in_proj_gates.weight.data@R1
                layer.mixer.out_proj.weight.data = R1.T@layer.mixer.out_proj.weight.data
            model.R1 =R1 

        if args.use_hadmard_R2:
            for i,layer in enumerate(model.layers):
                # K = torch.eye(model.layers[0].mixer.out_proj.in_features).to(R2) \
                #           if not args.use_klt else matmul_klt[f"layers.{i}.mixer.out_proj"].to(R2)
                # R2 = K@R2
                layer.mixer.R2 = R2.to(layer.mixer.out_proj.weight.data)
                layer.mixer.out_proj.weight.data = layer.mixer.out_proj.weight.data@R2.to(layer.mixer.out_proj.weight.data)
        
        if args.use_hadmard_R3:
            for name,module in model.named_modules():
                if 'matmul' in name and 'quantizer' not in name:    
                    K = torch.eye(model.layers[0].mixer.out_proj.in_features).to(R3) \
                          if not args.use_klt else matmul_klt[name].to(R3)
                    # R3 = K@R3
                    module.R3 =R3.to(model.layers[0].mixer.out_proj.weight)

        if args.use_hadmard_R4 :
            for i,layer in enumerate(model.layers):
                layer.mixer.R4 = R4
                layer.mixer.x_proj_C.weight.data = R4.T@layer.mixer.x_proj_C.weight.data
                layer.mixer.x_proj_C_b.weight.data = R4.T@layer.mixer.x_proj_C_b.weight.data

        if args.use_hadmard_R5:
            R5_b = R5     
            for layer in model.layers:
                K = torch.eye(model.layers[0].mixer.config.dt_rank).to(R5) \
                          if not args.use_klt else matmul_klt[f"layers.{i}.mixer.dt_proj"].to(R5)
                K_b = torch.eye(model.layers[0].mixer.config.dt_rank).to(R5) \
                          if not args.use_klt else matmul_klt[f"layers.{i}.mixer.dt_proj_b"].to(R5)
                R5 = K@R5
                R5_b = K_b@R5_b

                if layer.mixer.x_proj_dt.bias is not None:
                    layer.mixer.x_proj_dt.bias.data = (layer.mixer.x_proj_dt.bias.data.reshape(1,-1)@R5.T).reshape(-1)
                layer.mixer.x_proj_dt.weight.data = R5@layer.mixer.x_proj_dt.weight.data
                layer.mixer.dt_proj.weight.data = layer.mixer.dt_proj.weight.data@R5.T

                if layer.mixer.x_proj_dt_b.bias is not None:
                    layer.mixer.x_proj_dt_b.bias.data = (layer.mixer.x_proj_dt_b.bias.data.reshape(1,-1)@R5_b.T).reshape(-1)
                layer.mixer.x_proj_dt_b.weight.data = R5_b@layer.mixer.x_proj_dt_b.weight.data
                layer.mixer.dt_proj_b.weight.data = layer.mixer.dt_proj_b.weight.data@R5_b.T

        if args.use_hadmard_R6:        
            for layer in model.layers:
                layer.mixer.R6 = R6
                layer.mixer.x_proj_dt.weight.data = layer.mixer.x_proj_dt.weight.data@R6
                layer.mixer.x_proj_C.weight.data = layer.mixer.x_proj_C.weight.data@R6
                layer.mixer.x_proj_B.weight.data = layer.mixer.x_proj_B.weight.data@R6
                
                layer.mixer.x_proj_dt_b.weight.data = layer.mixer.x_proj_dt_b.weight.data@R6
                layer.mixer.x_proj_C_b.weight.data = layer.mixer.x_proj_C_b.weight.data@R6
                layer.mixer.x_proj_B_b.weight.data = layer.mixer.x_proj_B_b.weight.data@R6

        if args.use_S1:
            for i,layer in enumerate(model.layers): 
                # act = act_scales[f"layers.{i}.mixer.in_proj"].to(device=device, dtype=dtype)
                # weight_states = layer.mixer.in_proj_states.weight.data
                # weight_gates  = layer.mixer.in_proj_gates.weight.data
                # weight = torch.cat([weight_states,weight_gates],dim=0)
                # weight_scales = weight.abs().max(dim=0, keepdim=True)[0].reshape(-1).clamp(min=1e-5)
                # alpha = 0.5
                # scales = ((act.pow(alpha) / weight_scales.pow(1 - alpha)).clamp(min=1e-5).to(device).to(dtype))
                
                weight_states = layer.mixer.in_proj_states.weight.data
                weight_gates  = layer.mixer.in_proj_gates.weight.data
                weight = torch.cat([weight_states,weight_gates],dim=0)
                weight_scales = weight.abs().max(dim=0, keepdim=True)[0].reshape(-1).clamp(min=1e-5)
                max_val = torch.max(weight_scales)
                scales = weight_scales.clamp(min=1e-5).to(device).to(dtype)
                
                layer.norm.weight.data = (1/scales.reshape(-1))*layer.norm.weight.data
                
                layer.mixer.in_proj_states.weight.data = scales.reshape(-1)*layer.mixer.in_proj_states.weight.data
                layer.mixer.in_proj_gates.weight.data = scales.reshape(-1)*layer.mixer.in_proj_gates.weight.data


    if args.use_gptq:
        from quantize.gptq import gptq_fwrd_vim
        args.nsamples=128;args.w_bits=w_cfg['n_bits'];args.w_groupsize=128
        args.percdamp=0.01;args.act_order=False;args.perchannel=True
        quantizers = gptq_fwrd_vim(model,data_loader_val,device,args)
        # torch.save(model,"./saved_checkpoint/"+"-".join(args.model.split("_")[:2])+"_gptq_weight_"+str(args.w_bits)+"bit_R1.pt")
        # torch.save(quantizers,"./saved_checkpoint/"+"-".join(args.model.split("_")[:2])+"_gptq_scales_"+str(args.w_bits)+"bit_smoothed.pt")
        # model = torch.load("./saved_checkpoint/"+"-".join(args.model.split("_")[:2])+"_gptq_weight_"+str(args.w_bits)+"bit_R1.pt",map_location='cpu')
        # quantizers = torch.load("./saved_checkpoint/"+"-".join(args.model.split("_")[:2])+"_gptq_scales_"+str(args.w_bits)+"bit.pt",map_location='cpu')
        model.to(device)
    
    if args.use_split:
        for name, module in model.named_modules():
            from vim.vim_torch import MambaBlock,MambaBlock_optimize
            if isinstance(module, (MambaBlock,MambaBlock_optimize)):
                module.config.use_split = args.use_split

    if args.use_adaround:
        from torch.utils.data import Subset
        subset_indices = list(range(1,50000,200))
        calibration = Subset(dataset_val, subset_indices)
        calibration = torch.utils.data.DataLoader(
            calibration, 
            batch_size=16,
        )

        model = fp_model
        device = torch.device("cuda")
        from adaround_quant.utils import inplace_quantize_layers,enable_calibrate,disable_calibrate,calibrate_adaround
        inplace_quantize_layers(model)

        from tqdm import tqdm
        def calibrate():
            model.eval()
            with torch.no_grad(): 
                for i,(img,label) in tqdm(enumerate(calibration),total=len(calibration), desc="calibrating"):  
                    if i==1:  
                        break
                    model(img.to(device))
                   
        model = model.to(device)
        enable_calibrate(model)  
        calibrate()  
        disable_calibrate(model)

        print("==> adaround...")
        calibrate_adaround(model, args.adaround_iter, args.b_start, args.b_end, args.warmup, calibration, device)  
        torch.save(model.state_dict(),"./saved_checkpoint/"+"-".join(args.model.split("_")[:2])+"_adaround.pt")

    if args.static_quant:#先较准
        set_static_quant(model,True)
        set_observing(model,True)
        from torch.utils.data import Subset
        subset_indices = list(range(1,50000,200))
        calibration = Subset(dataset_val, subset_indices)
        calibration = torch.utils.data.DataLoader(
            calibration, 
            batch_size=int(1.5 * args.batch_size),
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )
        test_stats = evaluate(calibration, model, device, amp_autocast)
        print(f"Fp Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        set_observing(model,False)

    if args.analyse_and_plot:
        global input_name
        input_name = "fig_after_r1r2r3r5r6_k1k5_base"
        
        register_hooks(model.layers,name=input_name)
        register_hooks_2(model.head,name=input_name)

        metric_logger = utils.MetricLogger(delimiter="  ")
        header = 'Test:'
        with torch.no_grad():
            for images, target in metric_logger.log_every(data_loader_val, 10, header):
                images = images.to(device, non_blocking=True)
                # images = images.cpu()
                # model.cpu()
                model(images)
                break
    else:   
        test_stats = evaluate(data_loader_val, model, device, amp_autocast)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        return


import functools
from quantize import QuantConv1d,QuantConv2d,QuantLinear,QuantMatMul
from quantize.plot_utils.utils import plot_line_fig,plot_quantile_fig,plot_box_data_perchannel_fig, plot_bar_fig, plot_bar3d_fig,concat_images,find_images

def count_suffixes(directory):
    """
    统计指定文件夹中图片文件名中'mixer.'后面部分的后缀名称及其数量。
    
    参数:
    - directory: 包含图片的文件夹路径。
    
    返回:
    - 一个字典，键为后缀名称，值为该后缀名称出现的次数。
    """
    suffix_counts = {}
    for filename in os.listdir(directory):
        # 分割文件名以找到'mixer.'后面的部分
        if filename.endswith('.jpg'):
            parts = filename.split('mixer.')
            if len(parts) > 1:
                suffix = parts[-1]  # 获取'mixer.'后面的部分
                suffix_counts[suffix] = suffix_counts.get(suffix, 0) + 1
    return suffix_counts

def resize_tensor(tensor, size=(500, 500)):
    """
    将二维 tensor 重新采样到指定大小。

    Args:
        tensor: 输入的二维 tensor。
        size: 输出的大小，默认为 (500, 500)。

    Returns:
        重新采样后的 tensor。
    """
    h, w = tensor.shape
    target_h, target_w = size

    # 如果某个维度小于目标大小，不进行降采样
    new_h = min(h, target_h)
    new_w = min(w, target_w)
    
    # 调整到新尺寸
    resized_tensor = F.interpolate(tensor.unsqueeze(0).unsqueeze(0), size=(new_h, new_w), mode='nearest')
    
    # 将 tensor 的维度恢复
    return resized_tensor.squeeze(0).squeeze(0)

def analyse_hook(module, input, output,name): 
    module_name = name
    os.makedirs(f"data/analyse_fig/{input_name}/{quant_name}/", exist_ok=True)
    save_dir = f"data/analyse_fig/{input_name}/{quant_name}"
    
    # 分析权重
    if isinstance(module, (QuantConv2d,QuantLinear)):
        weight = module.weight.data
        print(module_name,"_weight  shape: ",weight.shape)
        if not os.path.exists(f"{save_dir}/{module_name}_weight_inchannel_quantile.jpg"):
            plot_quantile_fig(weight, f"{save_dir}/{module_name}_weight_inchannel_quantile.jpg", axis=1)
        if not os.path.exists(f"{save_dir}/{module_name}_weight_outchannel_quantile.jpg"):
            plot_quantile_fig(weight, f"{save_dir}/{module_name}_weight_outchannel_quantile.jpg", axis=0)
        # if not os.path.exists(f"{save_dir}/{module_name}_weight_box_data_perchannel.jpg"):
        #     plot_box_data_perchannel_fig(weight, f"{save_dir}/{module_name}_weight_box_data_perchannel.jpg", axis=-1)
        if not os.path.exists(f"{save_dir}/{module_name}_weight_bar_data.jpg"):
            plot_bar_fig(weight, f"{save_dir}/{module_name}_weight_bar_data.jpg")
        # if not os.path.exists(f"{save_dir}/{module_name}_weight_bar3d_data.jpg"):
        #     plot_bar3d_fig(weight, f"{save_dir}/{module_name}_weight_bar3d_data.jpg")
            # plot_bar3d_fig(resize_tensor(weight), f"{save_dir}/{module_name}_weight_bar3d_data.jpg")
    
    # 分析权重
    if isinstance(module, QuantConv1d):
        weight = module.weight.data[:,0]
        print(module_name,"_weight  shape: ",weight.shape)
        if not os.path.exists(f"{save_dir}/{module_name}_weight_inchannel_quantile.jpg"):
            plot_quantile_fig(weight, f"{save_dir}/{module_name}_weight_inchannel_quantile.jpg", axis=1)
        if not os.path.exists(f"{save_dir}/{module_name}_weight_outchannel_quantile.jpg"):
            plot_quantile_fig(weight, f"{save_dir}/{module_name}_weight_outchannel_quantile.jpg", axis=0)
        # if not os.path.exists(f"{save_dir}/{module_name}_weight_box_data_perchannel.jpg"):
        #     plot_box_data_perchannel_fig(weight, f"{save_dir}/{module_name}_weight_box_data_perchannel.jpg", axis=-1)
        if not os.path.exists(f"{save_dir}/{module_name}_weight_bar_data.jpg"):
            plot_bar_fig(weight, f"{save_dir}/{module_name}_weight_bar_data.jpg")
        # if not os.path.exists(f"{save_dir}/{module_name}_weight_bar3d_data.jpg"):
        #     plot_bar3d_fig(weight, f"{save_dir}/{module_name}_weight_bar3d_data.jpg")
            # plot_bar3d_fig(resize_tensor(weight), f"{save_dir}/{module_name}_weight_bar3d_data.jpg")

    for i,temp_input in enumerate(input):
        if isinstance(temp_input, torch.Tensor):  # 确保有输入且为Tensor类型
            print(module_name,f"_input_{i}  shape: ",temp_input.shape)
            if not os.path.exists(f"{save_dir}/{module_name}_input{i}_token_quantile.jpg"):
                plot_quantile_fig(temp_input, f"{save_dir}/{module_name}_input{i}_token_quantile.jpg", axis=1)
            if not os.path.exists(f"{save_dir}/{module_name}_input{i}_channel_quantile.jpg"):
                plot_quantile_fig(temp_input, f"{save_dir}/{module_name}_input{i}_channel_quantile.jpg", axis=2)
            # if not os.path.exists(f"{save_dir}/{module_name}_input{i}_box_data_perchannel.jpg"):
            #     plot_box_data_perchannel_fig(torch.amax(temp_input,dim=0), f"{save_dir}/{module_name}_input{i}_box_data_perchannel.jpg", axis=-1)
            if not os.path.exists(f"{save_dir}/{module_name}_input{i}_bar_data.jpg"):
                plot_bar_fig(temp_input, f"{save_dir}/{module_name}_input{i}_bar_data.jpg")
            # if not os.path.exists(f"{save_dir}/{module_name}_input{i}_bar3d_data.jpg"):
            #     plot_bar3d_fig(torch.amax(temp_input,dim=0), f"{save_dir}/{module_name}_input{i}_bar3d_data.jpg")
                # plot_bar3d_fig(resize_tensor(torch.amax(temp_input,dim=0)), f"{save_dir}/{module_name}_input{i}_bar3d_data.jpg")

def analyse_hook_2(module, input, output,name):  # matmul画图
    module_name = name
    os.makedirs(f"data/analyse_fig/{input_name}/{quant_name}/", exist_ok=True)
    save_dir = f"data/analyse_fig/{input_name}/{quant_name}"
    
    for i,temp_input in enumerate(input):
        if isinstance(temp_input, torch.Tensor):  # 确保有输入且为Tensor类型
            print(module_name,f"_input_{i}  shape: ",temp_input.shape)
            if not os.path.exists(f"{save_dir}/{module_name}_input{i}_token_quantile.jpg"):
                plot_quantile_fig(temp_input, f"{save_dir}/{module_name}_input{i}_token_quantile.jpg", axis=1)
            if not os.path.exists(f"{save_dir}/{module_name}_input{i}_channel_quantile.jpg"):
                plot_quantile_fig(temp_input, f"{save_dir}/{module_name}_input{i}_channel_quantile.jpg", axis=2)
            if not os.path.exists(f"{save_dir}/{module_name}_input{i}_box_data_perchannel.jpg"):
                plot_box_data_perchannel_fig(torch.amax(temp_input,dim=0), f"{save_dir}/{module_name}_input{i}_box_data_perchannel.jpg", axis=-1)
            if not os.path.exists(f"{save_dir}/{module_name}_input{i}_bar_data.jpg"):
                plot_bar_fig(temp_input, f"{save_dir}/{module_name}_input{i}_bar_data.jpg")
            # if not os.path.exists(f"{save_dir}/{module_name}_input{i}_bar3d_data.jpg"):
            #     plot_bar3d_fig(torch.amax(temp_input,dim=0), f"{save_dir}/{module_name}_input{i}_bar3d_data.jpg")
                # plot_bar3d_fig(resize_tensor(torch.amax(temp_input,dim=0)), f"{save_dir}/{module_name}_input{i}_bar3d_data.jpg")

def analyse_hook_3(module, input, output,name): 
    module_name = name
    os.makedirs(f"data/analyse_fig/{input_name}/{quant_name}/", exist_ok=True)
    save_dir = f"data/analyse_fig/{input_name}/{quant_name}"
    
    # 分析权重
    if isinstance(module, (QuantConv2d,QuantLinear,nn.Linear)):
        weight = module.weight.data
        print(module_name,"_weight  shape: ",weight.shape)
        if not os.path.exists(f"{save_dir}/{module_name}_weight_inchannel_quantile.jpg"):
            plot_quantile_fig(weight, f"{save_dir}/{module_name}_weight_inchannel_quantile.jpg", axis=1)
        if not os.path.exists(f"{save_dir}/{module_name}_weight_outchannel_quantile.jpg"):
            plot_quantile_fig(weight, f"{save_dir}/{module_name}_weight_outchannel_quantile.jpg", axis=0)
        # if not os.path.exists(f"{save_dir}/{module_name}_weight_box_data_perchannel.jpg"):
        #     plot_box_data_perchannel_fig(weight, f"{save_dir}/{module_name}_weight_box_data_perchannel.jpg", axis=-1)
        if not os.path.exists(f"{save_dir}/{module_name}_weight_bar_data.jpg"):
            plot_bar_fig(weight, f"{save_dir}/{module_name}_weight_bar_data.jpg")
        # if not os.path.exists(f"{save_dir}/{module_name}_weight_bar3d_data.jpg"):
        #     plot_bar3d_fig(weight, f"{save_dir}/{module_name}_weight_bar3d_data.jpg")
            # plot_bar3d_fig(resize_tensor(weight), f"{save_dir}/{module_name}_weight_bar3d_data.jpg")

    for i,temp_input in enumerate(input):
        if isinstance(temp_input, torch.Tensor):  # 确保有输入且为Tensor类型
            print(module_name,f"_input_{i}  shape: ",temp_input.shape)
            if not os.path.exists(f"{save_dir}/{module_name}_input{i}_token_quantile.jpg"):
                plot_quantile_fig(temp_input.unsqueeze(1), f"{save_dir}/{module_name}_input{i}_token_quantile.jpg", axis=1)
            if not os.path.exists(f"{save_dir}/{module_name}_input{i}_channel_quantile.jpg"):
                plot_quantile_fig(temp_input.unsqueeze(1), f"{save_dir}/{module_name}_input{i}_channel_quantile.jpg", axis=2)
            # if not os.path.exists(f"{save_dir}/{module_name}_input{i}_box_data_perchannel.jpg"):
            #     plot_box_data_perchannel_fig(torch.amax(temp_input,dim=0), f"{save_dir}/{module_name}_input{i}_box_data_perchannel.jpg", axis=-1)
            if not os.path.exists(f"{save_dir}/{module_name}_input{i}_bar_data.jpg"):
                plot_bar_fig(temp_input, f"{save_dir}/{module_name}_input{i}_bar_data.jpg")
            # if not os.path.exists(f"{save_dir}/{module_name}_input{i}_bar3d_data.jpg"):
            #     plot_bar3d_fig(torch.amax(temp_input,dim=0), f"{save_dir}/{module_name}_input{i}_bar3d_data.jpg")
                # plot_bar3d_fig(resize_tensor(torch.amax(temp_input,dim=0)), f"{save_dir}/{module_name}_input{i}_bar3d_data.jpg")

def register_hooks(model,name="fig"):
    global module_to_name 
    global quant_name
    global input_name
    input_name = name
    quant_name = "fp_data"
    handles = []
    for i,layer in enumerate(model):
        # 新增后处理hook
        handles.append(layer.mixer.conv1d.register_forward_hook(functools.partial(analyse_hook, name=f"{i}.layer.mixer.conv1d")))
        # handles.append(layer.mixer.conv1d_b.register_forward_hook(functools.partial(analyse_hook, name=f"{i}.layer.mixer.conv1d_b")))
        
        handles.append(layer.mixer.in_proj_states.register_forward_hook(functools.partial(analyse_hook, name=f"{i}.layer.mixer.in_proj_states")))
        handles.append(layer.mixer.in_proj_gates.register_forward_hook(functools.partial(analyse_hook, name=f"{i}.layer.mixer.in_proj_gates")))
        # # handles.append(layer.mixer.in_proj.register_forward_hook(functools.partial(analyse_hook, name=f"{i}.layer.mixer.conv1d")))
        handles.append(layer.mixer.x_proj_B.register_forward_hook(functools.partial(analyse_hook, name=f"{i}.layer.mixer.x_proj_B")))
        handles.append(layer.mixer.x_proj_C.register_forward_hook(functools.partial(analyse_hook, name=f"{i}.layer.mixer.x_proj_C")))
        handles.append(layer.mixer.x_proj_dt.register_forward_hook(functools.partial(analyse_hook, name=f"{i}.layer.mixer.x_proj_dt")))
        # handles.append(layer.mixer.x_proj_B_b.register_forward_hook(functools.partial(analyse_hook, name=f"{i}.layer.mixer.x_proj_B_b")))
        # handles.append(layer.mixer.x_proj_C_b.register_forward_hook(functools.partial(analyse_hook, name=f"{i}.layer.mixer.x_proj_C_b")))
        # handles.append(layer.mixer.x_proj_dt_b.register_forward_hook(functools.partial(analyse_hook, name=f"{i}.layer.mixer.x_proj_dt_b")))
        # handles.append(layer.mixer.x_proj.register_forward_hook(functools.partial(analyse_hook, name=f"{i}.layer.mixer.conv1d")))
        handles.append(layer.mixer.dt_proj.register_forward_hook(functools.partial(analyse_hook, name=f"{i}.layer.mixer.dt_proj")))
        handles.append(layer.mixer.out_proj.register_forward_hook(functools.partial(analyse_hook, name=f"{i}.layer.mixer.out_proj")))
        # handles.append(layer.mixer.conv_matmul.register_forward_hook(analyse_hook_2))
        # handles.append(layer.mixer.matmul.register_forward_hook(functools.partial(analyse_hook_2, name=f"{i}.layer.mixer.matmul")))
    return handles

def register_hooks_2(model,name="fig"):
    global module_to_name 
    global quant_name
    global input_name
    input_name = name
    quant_name = "fp_data"
    module_to_name = {module: name for name, module in model.named_modules()}
    handles = []
    handles.append(model.register_forward_hook(functools.partial(analyse_hook_3, name=f"head")))
    return handles

if __name__ == '__main__':
    parser = argparse.ArgumentParser('DeiT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
