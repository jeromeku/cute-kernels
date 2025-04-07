from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..torch_implementation import Experts_Torch, MoE_Torch
from .ops import bincount, scattered_experts


class Experts_Triton(Experts_Torch):
    def forward(
        self,
        inputs,
        k,
        sorted_expert_idxs,
        sorted_scattered_idxs,
        expert_offsets,
        gates=None,
        grouped_in=False,
        grouped_out=False,
    ):
        return scattered_experts(
            inputs,
            self.weight.permute(0, 2, 1),
            k,
            sorted_expert_idxs,
            sorted_scattered_idxs,
            expert_offsets,
            gates,
            grouped_in,
            grouped_out,
        )


class MoE_Triton(MoE_Torch):
    def __init__(
        self,
        num_experts: int,
        num_experts_per_tok: int,
        hidden_size: int,
        intermediate_size: int,
        add_bias: bool,
        std: float,
        activation_function: Callable = F.silu,
    ) -> None:
        nn.Module.__init__(self)

        self.num_experts = num_experts
        self.top_k = num_experts_per_tok

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        self.gate = nn.Linear(
            in_features=self.hidden_size, out_features=num_experts, bias=False
        )

        self.c_fc = Experts_Triton(
            num_experts=num_experts,
            in_features=self.hidden_size,
            out_features=2 * self.intermediate_size,
            add_bias=add_bias,
            std=std,
        )

        self.act = activation_function

        self.c_proj = Experts_Triton(
            num_experts=num_experts,
            in_features=self.intermediate_size,
            out_features=self.hidden_size,
            add_bias=add_bias,
            std=std,
        )

    def _compute_experts(
        self,
        hidden_states: torch.Tensor,
        router_weights: torch.Tensor,
        selected_experts: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            sorted_expert_idx, sorted_scattered_idx = (
                selected_experts.flatten().sort()
            )
            expert_offsets = bincount(
                sorted_expert_idx, self.num_experts
            ).cumsum(-1)
        breakpoint()
        hidden_states = self.c_fc(
            hidden_states,
            self.top_k,
            sorted_expert_idx,
            sorted_scattered_idx,
            expert_offsets,
            grouped_out=True,
        )
        gate, up = hidden_states.chunk(2, dim=-1)
        hidden_states = self.act(gate) * up
        hidden_states = self.c_proj(
            hidden_states,
            1, # hardcoded in original code
            sorted_expert_idx,
            sorted_scattered_idx,
            expert_offsets,
            grouped_in=True,
            gates=router_weights,
        )
        return hidden_states
