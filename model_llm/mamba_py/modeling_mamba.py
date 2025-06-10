# coding=utf-8
# Copyright 2024 state-spaces/mamba org and HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""PyTorch MAMBA model."""

import math
from dataclasses import dataclass
from re import T
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import alpha_dropout, nn
from torch.nn import CrossEntropyLoss

from transformers.activations import ACT2FN
from transformers.generation import GenerationMixin
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
)
from transformers.utils.import_utils import is_causal_conv1d_available, is_mamba_ssm_available
from transformers.models.mamba.configuration_mamba import MambaConfig
from zmq import has


logger = logging.get_logger(__name__)

if is_mamba_ssm_available():
    from mamba_ssm.ops.selective_scan_interface import mamba_inner_fn, selective_scan_fn
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
else:
    selective_state_update, selective_scan_fn, mamba_inner_fn = None, None, None

if is_causal_conv1d_available():
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
else:
    causal_conv1d_update, causal_conv1d_fn = None, None

is_fast_path_available = all(
    (selective_state_update, selective_scan_fn, causal_conv1d_fn, causal_conv1d_update, mamba_inner_fn)
)

_CHECKPOINT_FOR_DOC = "state-spaces/mamba-130m-hf"
_CONFIG_FOR_DOC = "MambaConfig"


#from transformers.models.deprecated._archive_maps import MAMBA_PRETRAINED_MODEL_ARCHIVE_LIST  # noqa: F401, E402


class MambaCache:
    """
    Arguments:
        config: MambaConfig
        batch_size: int
        dtype: torch.dtype
        device: torch.device

    Attributes:
        seqlen_offset: int
        dtype: torch.dtype
        conv_states: Dict[int, torch.Tensor] # layer_idx -> [batch_size, intermediate_size, conv_kernel_size]
        ssm_states: Dict[int, torch.Tensor] # layer_idx -> [batch_size, intermediate_size, ssm_state_size]
    """

    def __init__(
        self, config: MambaConfig, batch_size: int, dtype: torch.dtype = torch.float16, device: Optional[str] = None
    ):
        self.seqlen_offset = 0
        self.dtype = dtype
        intermediate_size = config.intermediate_size
        ssm_state_size = config.state_size
        conv_kernel_size = config.conv_kernel

        self.conv_states = {
            i: torch.zeros(batch_size, intermediate_size, conv_kernel_size, device=device, dtype=dtype)
            for i in range(config.num_hidden_layers)
        }
        self.ssm_states = {
            i: torch.zeros(batch_size, intermediate_size, ssm_state_size, device=device, dtype=dtype)
            for i in range(config.num_hidden_layers)
        }

from mamba_py.normalized_modules import MatMul
class MambaMixer(nn.Module):
    def __init__(self, config: MambaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.ssm_state_size = config.state_size
        self.conv_kernel_size = config.conv_kernel
        self.intermediate_size = config.intermediate_size
        self.time_step_rank = int(config.time_step_rank)
        self.layer_idx = layer_idx
        self.use_conv_bias = config.use_conv_bias
        self.conv1d = nn.Conv1d(
            in_channels=self.intermediate_size,
            out_channels=self.intermediate_size,
            bias=config.use_conv_bias,
            kernel_size=config.conv_kernel,
            groups=self.intermediate_size,
            padding=config.conv_kernel - 1,
        )

        self.activation = config.hidden_act
        self.act = ACT2FN[config.hidden_act]

        # projection of the input hidden states
        self.in_proj = nn.Linear(self.hidden_size, self.intermediate_size * 2, bias=config.use_bias)
        # selective projection used to make dt, B and C input dependant
        self.x_proj = nn.Linear(self.intermediate_size, self.time_step_rank + self.ssm_state_size * 2, bias=False)
        # time step projection (discretization)
        self.dt_proj = nn.Linear(self.time_step_rank, self.intermediate_size, bias=True)

        # The core is to load them, compute the discrete states, then write the updated state. Keeps the memory bounded
        A = torch.arange(1, self.ssm_state_size + 1, dtype=torch.float32)[None, :]
        A = A.expand(self.intermediate_size, -1).contiguous()

        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.intermediate_size))
        self.out_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.use_bias)
        self.use_bias = config.use_bias

        if not is_fast_path_available:
            logger.warning_once(
                "The fast path is not available because on of `(selective_state_update, selective_scan_fn, causal_conv1d_fn, causal_conv1d_update, mamba_inner_fn)`"
                " is None. Falling back to the naive implementation. To install follow https://github.com/state-spaces/mamba/#installation and"
                " https://github.com/Dao-AILab/causal-conv1d"
            )

        self.conv_matmul = MatMul()
        self.ssm_matmul = MatMul()
        
    def cuda_kernels_forward(self, hidden_states: torch.Tensor, cache_params: Optional[MambaCache] = None):
        # 1. Gated MLP's linear projection
        projected_states = self.in_proj(hidden_states).transpose(1, 2)

        if self.training and cache_params is None:  # Doesn't support outputting the states -> used for training
            contextualized_states = mamba_inner_fn(
                projected_states,
                self.conv1d.weight,
                self.conv1d.bias if self.use_conv_bias else None,
                self.x_proj.weight,
                self.dt_proj.weight,
                self.out_proj.weight,
                self.out_proj.bias.float() if self.use_bias else None,
                -torch.exp(self.A_log.float()),
                None,  # input-dependent B
                None,  # input-dependent C
                self.D.float(),
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
            )

        else:
            hidden_states, gate = projected_states.chunk(2, dim=1)

            # 2. Convolution sequence transformation
            conv_weights = self.conv1d.weight.view(self.conv1d.weight.size(0), self.conv1d.weight.size(2))
            if cache_params is not None and cache_params.seqlen_offset > 0:
                hidden_states = causal_conv1d_update(
                    hidden_states.squeeze(-1),
                    cache_params.conv_states[self.layer_idx],
                    conv_weights,
                    self.conv1d.bias,
                    self.activation,
                )
                hidden_states = hidden_states.unsqueeze(-1)
            else:
                if cache_params is not None:
                    conv_states = nn.functional.pad(
                        hidden_states, (self.conv_kernel_size - hidden_states.shape[-1], 0)
                    )
                    cache_params.conv_states[self.layer_idx].copy_(conv_states)
                hidden_states = causal_conv1d_fn(
                    hidden_states, conv_weights, self.conv1d.bias, activation=self.activation
                )

            # 3. State Space Model sequence transformation
            # 3.a. input varying initialization of time_step, B and C
            ssm_parameters = self.x_proj(hidden_states.transpose(1, 2))
            time_step, B, C = torch.split(
                ssm_parameters, [self.time_step_rank, self.ssm_state_size, self.ssm_state_size], dim=-1
            )
            discrete_time_step = torch.matmul(self.dt_proj.weight, time_step.transpose(1, 2))

            A = -torch.exp(self.A_log.float())
            # 3.c perform the recurrence y ← SSM(A, B, C)(x)
            time_proj_bias = self.dt_proj.bias.float() if hasattr(self.dt_proj, "bias") else None
            if cache_params is not None and cache_params.seqlen_offset > 0:
                scan_outputs = selective_state_update(
                    cache_params.ssm_states[self.layer_idx],
                    hidden_states[..., 0],
                    discrete_time_step[..., 0],
                    A,
                    B[:, 0],
                    C[:, 0],
                    self.D,
                    gate[..., 0],
                    time_proj_bias,
                    dt_softplus=True,
                ).unsqueeze(-1)
            else:
                scan_outputs, ssm_state = selective_scan_fn(
                    hidden_states,
                    discrete_time_step,
                    A,
                    B.transpose(1, 2),
                    C.transpose(1, 2),
                    self.D.float(),
                    gate,
                    time_proj_bias,
                    delta_softplus=True,
                    return_last_state=True,
                )
                if ssm_state is not None and cache_params is not None:
                    cache_params.ssm_states[self.layer_idx].copy_(ssm_state)

            # 4. Final linear projection
            contextualized_states = self.out_proj(scan_outputs.transpose(1, 2))
        return contextualized_states

    # fmt: off
    def slow_forward(self, input_states, cache_params: Optional[MambaCache]=None):
        batch_size, seq_len, _ = input_states.shape # [batch, seq_len, hidden_size]
        dtype = input_states.dtype
        # 1. Gated MLP's linear projection
        projected_states = self.in_proj(input_states).transpose(1, 2)                   # [batch, 2 * intermediate_size, seq_len]
        hidden_states, gate = projected_states.chunk(2, dim=1)

        # 2. Convolution sequence transformation
        if cache_params is not None:
            ssm_state = cache_params.ssm_states[self.layer_idx].clone()
            if cache_params.seqlen_offset > 0:
                conv_state = cache_params.conv_states[self.layer_idx]                   # [batch, intermediate_size, conv_kernel_size]
                conv_state = torch.roll(conv_state, shifts=-1, dims=-1)
                conv_state[:, :, -1] = hidden_states[:, :, 0]
                cache_params.conv_states[self.layer_idx].copy_(conv_state)
                # hidden_states = torch.sum(conv_state * self.conv1d.weight[:, 0, :], dim=-1)
                weight = self.conv1d.weight[:, 0, :].unsqueeze(1)  # [out_channels, 1, kernel_size]
                hidden_states = self.conv_matmul(conv_state, weight.transpose(1, 2)).squeeze(-1)
                if self.use_conv_bias:
                    hidden_states += self.conv1d.bias
                hidden_states = self.act(hidden_states).to(dtype).unsqueeze(-1)         # [batch, intermediate_size, 1] : decoding
            else:
                conv_state = nn.functional.pad(
                    hidden_states,
                    (self.conv_kernel_size - hidden_states.shape[-1], 0)
                )
                cache_params.conv_states[self.layer_idx].copy_(conv_state)
                hidden_states = self.act(self.conv1d(hidden_states.to(self.conv1d.weight))[..., :seq_len])     # [batch, intermediate_size, seq_len]
        else:
            ssm_state = torch.zeros(
                (batch_size, self.intermediate_size, self.ssm_state_size),
                device=hidden_states.device, dtype=dtype
            )
            hidden_states = self.act(self.conv1d(hidden_states)[..., :seq_len])         # [batch, intermediate_size, seq_len]

        # 3. State Space Model sequence transformation
        # 3.a. Selection:  [batch, seq_len, self.time_step_rank + self.ssm_state_size * 2]
        ssm_parameters = self.x_proj(hidden_states.transpose(1, 2).to(self.x_proj.weight))
        time_step, B, C = torch.split(
            ssm_parameters, [self.time_step_rank, self.ssm_state_size, self.ssm_state_size], dim=-1
        )
        discrete_time_step = self.dt_proj(time_step.to(self.dt_proj.weight))            # [batch, seq_len, intermediate_size]
        discrete_time_step = nn.functional.softplus(discrete_time_step).transpose(1, 2) # [batch, intermediate_size, seq_len]

        # 3.b. Discretization: B and C to [batch, seq_len, intermediate_size, ssm_state_size] (SRAM)
        A = -torch.exp(self.A_log.float())                                              # [intermediate_size, ssm_state_size]
        discrete_A = torch.exp(A[None, :, None, :] * discrete_time_step[:, :, :, None]) # [batch, intermediate_size, seq_len, ssm_state_size]
        discrete_B = discrete_time_step[:, :, :, None] * B[:, None, :, :].float()       # [batch, intermediade_size, seq_len, ssm_state_size]
        deltaB_u = discrete_B * hidden_states[:, :, :, None].float()
        
        # 3.c perform the recurrence y ← SSM(A, B, C)(x)
        scan_outputs = []
        h_list = []
        for i in range(seq_len):
            ssm_state = discrete_A[:, :, i, :] * ssm_state + deltaB_u[:, :, i, :]      # [batch, intermediade_size, ssm_state]
            print("********************")
            print("SSM state shape: ", ssm_state.shape)
            h_list.append(ssm_state)
        h = torch.stack(h_list,dim=1).to(C)
        
        if hasattr(self,"R4") and self.R4 is not None:
            h=h@self.R4;  C=C@self.R4
        
        if hasattr(self,"R3") and self.R3 is not None:
            if  hasattr(self,"matmul_scale") and self.matmul_scale is not None:
                h = h/(self.matmul_scale.unsqueeze(1).to(h))
                h = (h.permute(0, 1, 3, 2)@self.R3).permute(0, 1, 3, 2)
                y = (self.ssm_matmul(h,C.unsqueeze(-1))).squeeze(3)
                y = y@self.R3.T
                y = y*self.matmul_scale.to(y)   
                scan_output = y.permute(0,2,1)       
            else:
                h = (h.permute(0, 1, 3, 2)@self.R3.to(h)).permute(0, 1, 3, 2)
                y = (self.ssm_matmul(h,C.unsqueeze(-1))).squeeze(3)
                y = y@self.R3.T.to(y)
                scan_output = y.permute(0,2,1)
        else:
            scan_output = self.ssm_matmul(h,C.unsqueeze(-1)).squeeze(-1).permute(0,2,1)
        
        #     scan_output = torch.matmul(ssm_state.to(dtype), C[:, i, :].unsqueeze(-1))  # [batch, intermediade_size, 1]
        #     scan_outputs.append(scan_output[:, :, 0])
        # scan_output = torch.stack(scan_outputs, dim=-1)                                # [batch, seq_len, intermediade_size]
        scan_output = scan_output + (hidden_states * self.D[None, :, None])
        scan_output = (scan_output * self.act(gate))

        if cache_params is not None:
            cache_params.ssm_states[self.layer_idx].copy_(ssm_state)


        # 4. Final linear projection
        contextualized_states = self.out_proj(scan_output.transpose(1, 2).to(self.out_proj.weight)) # [batch, seq_len, hidden_size]
        return contextualized_states
    # fmt: on

    def forward(self, hidden_states, cache_params: Optional[MambaCache] = None):
        # if is_fast_path_available and "cuda" in self.x_proj.weight.device.type:
        #     return self.cuda_kernels_forward(hidden_states, cache_params)
        return self.slow_forward(hidden_states, cache_params)

class Swiglu(nn.Module):
    def __init__(self, s):
        super().__init__()
        self.s = s
        self.sigmod = nn.Sigmoid()
    def forward(self, x):
        return x*self.sigmod(x*self.s)

from copy import deepcopy
class MambaMixer_optimize(nn.Module):
    def __init__(self, ori_module):
        super().__init__()
        self.hidden_size = ori_module.hidden_size
        self.ssm_state_size = ori_module.ssm_state_size
        self.conv_kernel_size = ori_module.conv_kernel_size
        self.intermediate_size = ori_module.intermediate_size
        self.time_step_rank = ori_module.time_step_rank
        self.layer_idx = ori_module.layer_idx
        self.use_conv_bias = ori_module.use_conv_bias
        self.conv1d = ori_module.conv1d

        self.activation = ori_module.activation
        self.act = ori_module.act
        self.act_gate = deepcopy(ori_module.act) 

        # self.in_proj = ori_module.in_proj
        self.in_proj_states = deepcopy(ori_module.in_proj) 
        self.in_proj_states.weight.data = self.in_proj_states.weight.data[:self.intermediate_size]
        self.in_proj_states.out_features = self.intermediate_size
        self.in_proj_gates = deepcopy(ori_module.in_proj) 
        self.in_proj_gates.weight.data = self.in_proj_gates.weight.data[-self.intermediate_size:]
        self.in_proj_gates.out_features = self.intermediate_size
        
        # self.x_proj = ori_module.x_proj
        self.x_proj_b = deepcopy(ori_module.x_proj)
        self.x_proj_b.weight.data = self.x_proj_b.weight.data[-2*self.ssm_state_size:-self.ssm_state_size]
        self.x_proj_b.out_features = self.ssm_state_size
        self.x_proj_c = deepcopy(ori_module.x_proj)
        self.x_proj_c.weight.data = self.x_proj_c.weight.data[-self.ssm_state_size:]
        self.x_proj_c.out_features = self.ssm_state_size
        self.x_proj_dt = deepcopy(ori_module.x_proj)
        self.x_proj_dt.weight.data = self.x_proj_dt.weight.data[:-2*self.ssm_state_size]
        self.x_proj_dt.out_features = self.time_step_rank
        

        self.dt_proj = ori_module.dt_proj

        self.A_log = ori_module.A_log
        self.D = ori_module.D
        self.out_proj = ori_module.out_proj
        self.use_bias = ori_module.use_bias

        self.conv_matmul = ori_module.conv_matmul
        self.ssm_matmul = ori_module.ssm_matmul
          
    def forward(self, input_states, cache_params: Optional[MambaCache]=None):

        batch_size, seq_len, _ = input_states.shape # [batch, seq_len, hidden_size]
        dtype = input_states.dtype
        # 1. Gated MLP's linear projection
        # projected_states = self.in_proj(input_states).transpose(1, 2)                   # [batch, 2 * intermediate_size, seq_len]
        # hidden_states, gate = projected_states.chunk(2, dim=1)
        hidden_states = self.in_proj_states(input_states).transpose(1, 2) 
        gate = self.in_proj_gates(input_states).transpose(1, 2)

        # 2. Convolution sequence transformation
        if cache_params is not None:
            ssm_state = cache_params.ssm_states[self.layer_idx].clone()
            if cache_params.seqlen_offset > 0:
                conv_state = cache_params.conv_states[self.layer_idx]                   # [batch, intermediate_size, conv_kernel_size]
                conv_state = torch.roll(conv_state, shifts=-1, dims=-1)
                conv_state[:, :, -1] = hidden_states[:, :, 0]
                cache_params.conv_states[self.layer_idx].copy_(conv_state)
                # hidden_states = torch.sum(conv_state * self.conv1d.weight[:, 0, :], dim=-1)
                weight = self.conv1d.weight[:, 0, :].unsqueeze(1)  # [out_channels, 1, kernel_size]
                hidden_states = self.conv_matmul(conv_state, weight.transpose(1, 2)).squeeze(-1)
                if self.use_conv_bias:
                    hidden_states += self.conv1d.bias
                hidden_states = self.act(hidden_states).to(dtype).unsqueeze(-1)         # [batch, intermediate_size, 1] : decoding
            else:
                conv_state = nn.functional.pad(
                    hidden_states,
                    (self.conv_kernel_size - hidden_states.shape[-1], 0)
                )
                cache_params.conv_states[self.layer_idx].copy_(conv_state)
                hidden_states = self.act(self.conv1d(hidden_states.to(self.conv1d.weight))[..., :seq_len])     # [batch, intermediate_size, seq_len]
        else:
            ssm_state = torch.zeros(
                (batch_size, self.intermediate_size, self.ssm_state_size),
                device=hidden_states.device, dtype=dtype
            )
            hidden_states = self.act(self.conv1d(hidden_states)[..., :seq_len])         # [batch, intermediate_size, seq_len]

        # 3. State Space Model sequence transformation
        # 3.a. Selection:  [batch, seq_len, self.time_step_rank + self.ssm_state_size * 2]
        # ssm_parameters = self.x_proj(hidden_states.transpose(1, 2).to(self.x_proj.weight))
        # time_step, B, C = torch.split(
        #     ssm_parameters, [self.time_step_rank, self.ssm_state_size, self.ssm_state_size], dim=-1
        # )
        if hasattr(self, "R6") and self.R6 is not None:
            tmp_states = hidden_states.transpose(1, 2)
            tmp_states = tmp_states@self.R6
            time_step = self.x_proj_dt(tmp_states.to(self.x_proj_dt.weight))
            B = self.x_proj_b(tmp_states.to(self.x_proj_b.weight))
            C = self.x_proj_c(tmp_states.to(self.x_proj_c.weight))
        else:
            time_step = self.x_proj_dt(hidden_states.transpose(1, 2).to(self.x_proj_dt.weight))
            B = self.x_proj_b(hidden_states.transpose(1, 2).to(self.x_proj_b.weight))
            C = self.x_proj_c(hidden_states.transpose(1, 2).to(self.x_proj_c.weight))
        
        
        discrete_time_step = self.dt_proj(time_step.to(self.dt_proj.weight))            # [batch, seq_len, intermediate_size]
        discrete_time_step = nn.functional.softplus(discrete_time_step).transpose(1, 2) # [batch, intermediate_size, seq_len]

        # 3.b. Discretization: B and C to [batch, seq_len, intermediate_size, ssm_state_size] (SRAM)
        A = -torch.exp(self.A_log.float())                                              # [intermediate_size, ssm_state_size]
        discrete_A = torch.exp(A[None, :, None, :] * discrete_time_step[:, :, :, None]) # [batch, intermediate_size, seq_len, ssm_state_size]
        discrete_B = discrete_time_step[:, :, :, None] * B[:, None, :, :].float()       # [batch, intermediade_size, seq_len, ssm_state_size]
        deltaB_u = discrete_B * hidden_states[:, :, :, None].float()
        
        # 3.c perform the recurrence y ← SSM(A, B, C)(x)
        scan_outputs = []
        h_list = []
        for i in range(seq_len):
            ssm_state = discrete_A[:, :, i, :] * ssm_state + deltaB_u[:, :, i, :]      # [batch, intermediade_size, ssm_state]
            h_list.append(ssm_state)
        h = torch.stack(h_list,dim=1).to(C)
        
        if hasattr(self,"R4") and self.R4 is not None:
            h=h@self.R4; 
        if hasattr(self,"R3") and self.R3 is not None:     
            if  hasattr(self,"S3") and self.S3 is not None:
                h = h/(self.S3.unsqueeze(1).to(h))
                h = (h.permute(0, 1, 3, 2)@self.R3).permute(0, 1, 3, 2)
                y = (self.ssm_matmul(h,C.unsqueeze(-1))).squeeze(3)
                y = y@self.R3.T
                y = y*self.S3.to(y)   
                scan_output = y.permute(0,2,1)       
            else:
                h = (h.permute(0, 1, 3, 2)@self.R3.to(h)).permute(0, 1, 3, 2)
                y = (self.ssm_matmul(h,C.unsqueeze(-1))).squeeze(3)
                y = y@self.R3.T.to(y)
                scan_output = y.permute(0,2,1)
        else:
            scan_output = self.ssm_matmul(h,C.unsqueeze(-1)).squeeze(-1).permute(0,2,1)
        
        #     scan_output = torch.matmul(ssm_state.to(dtype), C[:, i, :].unsqueeze(-1))  # [batch, intermediade_size, 1]
        #     scan_outputs.append(scan_output[:, :, 0])
        # scan_output = torch.stack(scan_outputs, dim=-1)                                # [batch, seq_len, intermediade_size]
        scan_output = scan_output + (hidden_states * self.D[None, :, None])
        scan_output = (scan_output * (self.act_gate(gate.permute(0,2,1))).permute(0,2,1))

        if cache_params is not None:
            cache_params.ssm_states[self.layer_idx].copy_(ssm_state)

        # 4. Final linear projection
        scan_output = scan_output.transpose(1, 2)
        if hasattr(self,'R2') and self.R2 is not None:
            scan_output = scan_output@self.R2.to(scan_output)
        contextualized_states = self.out_proj(scan_output.to(self.out_proj.weight)) # [batch, seq_len, hidden_size]
        return contextualized_states

class MambaBlock_optimize(nn.Module):
    def __init__(self, ori_module):
        super().__init__()
        self.config = ori_module.config
        self.layer_idx = ori_module.layer_idx
        self.residual_in_fp32 = ori_module.residual_in_fp32
        self.norm = ori_module.norm
        self.mixer = MambaMixer_optimize(ori_module.mixer)
        
        self.R2 = None
        self.R4 = None #between x_proj_c and matmul
        self.S_R4 = None
        self.R5 = None #between x_proj_dt and dt_proj
        self.R6 = None #between rmsnorm and in_proj_gate/in_proj_state
        self.s1 = None #between in_proj_gates and out_proj
        self.rotated = False
    
    def rotate(self):
        if self.R6 is not None:
            self.mixer.R6 = self.R6
            self.mixer.x_proj_dt.weight.data = self.mixer.x_proj_dt.weight.data@self.R6
            self.mixer.x_proj_c.weight.data = self.mixer.x_proj_c.weight.data@self.R6
            self.mixer.x_proj_b.weight.data = self.mixer.x_proj_b.weight.data@self.R6
            self.R6 = None
        if self.R5 is not None:
            self.mixer.x_proj_dt.weight.data = self.R5@self.mixer.x_proj_dt.weight.data
            self.mixer.dt_proj.weight.data = self.mixer.dt_proj.weight.data@self.R5.T
            self.R5 = None
        if self.R4 is not None:
            self.mixer.R4 = self.R4
            self.mixer.x_proj_c.weight.data = self.R4.T@self.mixer.x_proj_c.weight.data
            self.R4 = None
        if self.R2 is not None:
            self.mixer.R2 = self.R2
            self.mixer.out_proj.weight.data = self.mixer.out_proj.weight.data@self.R2
            self.R2 = None
        if self.S_R4 is not None:
            self.mixer.S_R4 = self.S_R4

    def forward(self, hidden_states, cache_params: Optional[MambaCache] = None):
        # if not self.rotated:
        #     self.rotate()
        #     self.rotated = True
            
        residual = hidden_states
        hidden_states = self.norm(hidden_states.to(dtype=self.norm.weight.dtype))
        if self.residual_in_fp32:
            residual = residual.to(torch.float32)

        hidden_states = self.mixer(hidden_states, cache_params=cache_params)
        hidden_states = residual + hidden_states
        return hidden_states

class MambaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        MambaRMSNorm is equivalent to T5LayerNorm and LlamaRMSNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class MambaBlock(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.residual_in_fp32 = config.residual_in_fp32
        self.norm = MambaRMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.mixer = MambaMixer(config, layer_idx=layer_idx)

    def forward(self, hidden_states, cache_params: Optional[MambaCache] = None):
        residual = hidden_states
        hidden_states = self.norm(hidden_states.to(dtype=self.norm.weight.dtype))
        if self.residual_in_fp32:
            residual = residual.to(torch.float32)

        hidden_states = self.mixer(hidden_states, cache_params=cache_params)
        hidden_states = residual + hidden_states
        return hidden_states


class MambaPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = MambaConfig
    base_model_prefix = "backbone"
    _no_split_modules = ["MambaBlock"]
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, MambaMixer):
            module.A_log._no_weight_decay = True
            module.D._no_weight_decay = True

            dt_init_std = self.config.time_step_rank**-0.5 * self.config.time_step_scale
            if self.config.time_step_init_scheme == "constant":
                nn.init.constant_(module.dt_proj.weight, dt_init_std)
            elif self.config.time_step_init_scheme == "random":
                nn.init.uniform_(module.dt_proj.weight, -dt_init_std, dt_init_std)

            dt = torch.exp(
                torch.rand(self.config.intermediate_size)
                * (math.log(self.config.time_step_max) - math.log(self.config.time_step_min))
                + math.log(self.config.time_step_min)
            ).clamp(min=self.config.time_step_floor)
            # # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            with torch.no_grad():
                module.dt_proj.bias.copy_(inv_dt)
            module.dt_proj.bias._no_reinit = True

        if isinstance(module, nn.Linear):
            if module.bias is not None:
                if not getattr(module.bias, "_no_reinit", False):
                    nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=self.config.initializer_range)

        if self.config.rescale_prenorm_residual:
            # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
            #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
            #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
            #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
            #
            # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
            for name, p in module.named_parameters():
                if name in ["out_proj.weight"]:
                    # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                    # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                    # We need to reinit p since this code could be called multiple times
                    # Having just p *= scale would repeatedly scale it down
                    nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                    with torch.no_grad():
                        p /= math.sqrt(self.config.num_layers)


@dataclass
class MambaOutput(ModelOutput):
    """
    Class for the MAMBA model outputs.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        cache_params (`MambaCache`):
            The state of the model at the last time step. Can be used in a forward method with the next `input_ids` to
            avoid providing the old `input_ids`.

            Includes both the State space model state matrices after the selective scan, and the Convolutional states
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    """

    last_hidden_state: Optional[torch.FloatTensor] = None
    cache_params: Optional[MambaCache] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class MambaCausalLMOutput(ModelOutput):
    """
    Base class for causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        cache_params (`MambaCache`):
            The state of the model at the last time step. Can be used in a forward method with the next `input_ids` to
            avoid providing the old `input_ids`.

            Includes both the State space model state matrices after the selective scan, and the Convolutional states
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    cache_params: Optional[MambaCache] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None


MAMBA_START_DOCSTRING = r"""

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`MambaConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

MAMBA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, input_ids_length)`):
            Indices of input sequence tokens in the vocabulary.

            If `cache_params.seqlen_offset>0`, only `input_ids` that do not have their past calculated should be passed as
            `input_ids`.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        cache_params (`MambaCache`, *optional*):
            If passed along, the model uses the previous state in all the blocks (which will give the output for the
            `input_ids` provided as if the model add `state_input_ids + input_ids` as context).
        use_cache (`bool`, *optional*):
            If set to `True`, the `cache_params` is returned and can be used to quickly generate the next logits.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare MAMBA Model transformer outputting raw hidden-states without any specific head on top.",
    MAMBA_START_DOCSTRING,
)
class MambaModel(MambaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([MambaBlock(config, layer_idx=idx) for idx in range(config.num_hidden_layers)])

        self.gradient_checkpointing = False
        self.norm_f = MambaRMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        # Initialize weights and apply final processing
        self._register_load_state_dict_pre_hook(self.load_hook)
        self.post_init()

    def load_hook(self, state_dict, prefix, *args):
        for k in state_dict:
            if "embedding." in k:
                state_dict[k.replace("embedding.", "embeddings.")] = state_dict.pop(k)
                break

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, new_embeddings):
        self.embeddings = new_embeddings

    @add_start_docstrings_to_model_forward(MAMBA_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=MambaOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.LongTensor] = None,
        cache_params: Optional[MambaCache] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,  # `attention_mask` is passed by the tokenizer and we don't want it
    ) -> Union[Tuple, MambaOutput]:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else (self.config.use_cache if not self.training else False)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):  # ^ is python for xor
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)

        if hasattr(self,"R1"):
           inputs_embeds = (inputs_embeds.to(torch.float64) @ self.R1.to(torch.float64)).to(inputs_embeds.dtype)

        if self.gradient_checkpointing and self.training and use_cache:
            use_cache = False

        if cache_params is None and use_cache:
            cache_params = MambaCache(
                self.config, inputs_embeds.size(0), device=inputs_embeds.device, dtype=inputs_embeds.dtype
            )

        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        for i, mixer_block in enumerate(self.layers):
            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(mixer_block.__call__, hidden_states, cache_params)
            else:
                hidden_states = mixer_block(hidden_states, cache_params=cache_params)

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        if hasattr(self,"R1"):
           hidden_states = (hidden_states.to(torch.float64) @ self.R1.T.to(torch.float64)).to(hidden_states.dtype)

        if use_cache:
            cache_params.seqlen_offset += inputs_embeds.shape[1]

        hidden_states = self.norm_f(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, cache_params, all_hidden_states] if v is not None)

        return MambaOutput(
            last_hidden_state=hidden_states,
            cache_params=cache_params if use_cache else None,
            hidden_states=all_hidden_states,
        )


@add_start_docstrings(
    """
    The MAMBA Model transformer with a language modeling head on top (linear layer with weights tied to the input
    embeddings).
    """,
    MAMBA_START_DOCSTRING,
)
class MambaForCausalLM(MambaPreTrainedModel,GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.backbone = MambaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_input_embeddings(self):
        return self.backbone.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        return self.backbone.set_input_embeddings(new_embeddings)

    def _update_model_kwargs_for_generation(
        self, outputs: ModelOutput, model_kwargs: Dict[str, Any], **kwargs
    ) -> Dict[str, Any]:
        model_kwargs["cache_params"] = outputs.get("cache_params", None)
        return model_kwargs

    def prepare_inputs_for_generation(
        self, input_ids, cache_params: Optional[MambaCache] = None, inputs_embeds=None, attention_mask=None, **kwargs
    ):
        # only last token for inputs_ids if the state is passed along.
        if cache_params is not None:
            input_ids = input_ids[:, -1].unsqueeze(-1)

        if inputs_embeds is not None and cache_params is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs["cache_params"] = cache_params
        return model_inputs

    @add_start_docstrings_to_model_forward(MAMBA_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=MambaCausalLMOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_params: Optional[MambaCache] = None,
        labels: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        use_cache: Optional[bool] = None,
        **kwargs,  # for now we need this for generation
    ) -> Union[Tuple, MambaCausalLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        mamba_outputs = self.backbone(
            input_ids,
            cache_params=cache_params,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            use_cache=use_cache,
        )
        hidden_states = mamba_outputs[0]

        type_ = self.lm_head.weight.dtype
        if hasattr(self.backbone, 'R1'):
            hidden_states = (hidden_states.double()@self.backbone.R1.double()).to(type_)
        else:
            hidden_states = hidden_states
            
        if hasattr(self.backbone, 'R1') and not hasattr(self, 'lm_head_rotated'):
            self.lm_head_rotated = deepcopy(self.lm_head)
            self.lm_head_rotated.weight.data = (self.lm_head_rotated.weight.data.double()@self.backbone.R1.double()).to(type_)
        
        if hasattr(self, 'lm_head_rotated'):
            logits = self.lm_head_rotated(hidden_states.to(self.lm_head_rotated.weight.dtype)).float()
        else:
            logits = self.lm_head(hidden_states.to(self.lm_head.weight.dtype)).float()
        

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (logits,) + mamba_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return MambaCausalLMOutput(
            loss=loss,
            logits=logits,
            cache_params=mamba_outputs.cache_params,
            hidden_states=mamba_outputs.hidden_states,
        )
