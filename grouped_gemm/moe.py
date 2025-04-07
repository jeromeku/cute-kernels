import random
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Experts_Torch(nn.Module):
    def __init__(
        self,
        num_experts: int,
        in_features: int,
        out_features: int,
        add_bias: bool = True,
        std: float | None = None,
    ) -> None:
        super().__init__()

        self.weight = nn.Parameter(torch.empty(num_experts, out_features, in_features))

        self.bias = None
        if add_bias:
            self.bias = nn.Parameter(torch.empty(num_experts, out_features))

        self.std = std

        self.num_experts = num_experts
        self.in_features = in_features
        self.out_features = out_features

        self.reset_parameters()

    def forward(
        self,
        input: torch.Tensor | tuple[torch.Tensor],
        expert_frequency: torch.Tensor,
        return_list: bool,
    ) -> torch.Tensor | list[torch.Tensor]:
        if isinstance(input, torch.Tensor):
            input = input.split(expert_frequency.tolist(), dim=0)

        input = [
            F.linear(
                input[i],
                self.weight[i],
                None if self.bias is None else self.bias[i],
            )
            for i in range(self.num_experts)
        ]

        if not return_list:
            input = torch.cat(input)

        return input

    def extra_repr(self):
        return "num_experts={}, in_features={}, out_features={}".format(
            self.num_experts, self.in_features, self.out_features
        )

    @torch.no_grad()
    def reset_parameters(self) -> None:
        nn.init.normal_(self.weight, mean=0, std=self.std)
        if hasattr(self, "bias") and self.bias is not None:
            self.bias.zero_()


class MoE_Torch(nn.Module):
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
        super().__init__()

        self.num_experts = num_experts
        self.top_k = num_experts_per_tok

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        self.gate = nn.Linear(
            in_features=self.hidden_size, out_features=num_experts, bias=False
        )

        self.c_fc = Experts_Torch(
            num_experts=num_experts,
            in_features=self.hidden_size,
            out_features=2 * self.intermediate_size,
            add_bias=add_bias,
            std=std,
        )

        self.act = activation_function

        self.c_proj = Experts_Torch(
            num_experts=num_experts,
            in_features=self.intermediate_size,
            out_features=self.hidden_size,
            add_bias=add_bias,
            std=std,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        original_shape = hidden_states.shape
        # hidden_states -> (batch_size, query_length, hidden_size)
        hidden_states = hidden_states.view(-1, self.hidden_size)
        # hidden_states -> (total_q, hidden_size)
        router_logits, router_weights, selected_experts = self._compute_routing_weights(
            hidden_states
        )

        # router_logits -> (total_q, num_experts)
        # router_weights -> (total_q, top_k)
        # selected_experts -> (total_q, top_k)

        hidden_states = self._compute_experts(
            hidden_states, router_weights, selected_experts
        )
        hidden_states = hidden_states.view(original_shape)

        # hidden_states -> (batch_size, query_length, hidden_size)

        return hidden_states, router_logits


def compute_routing_weights(
    router_logits: torch.Tensor,
    top_k: int = 1,
) -> tuple[torch.Tensor]:
    router_logits = router_logits.view(-1, router_logits.size(-1))
    # router_logits [bs * seq_len, num_experts]

    router_weights, selected_experts = get_topk(router_logits, top_k)

    # router_weights -> (total_q, top_k)
    # selected_experts -> (total_q, top_k)

    router_weights = F.sigmoid(router_weights.float())
    router_weights = router_weights.to(router_logits.dtype)

    return router_weights, selected_experts


def get_topk(x: torch.Tensor, top_k: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    if top_k == 1:
        x, indices = x.max(dim=-1, keepdim=True)
    else:
        x, indices = x.topk(top_k, dim=-1)

    return x, indices


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    NUM_EXPERTS = 8
    TOPK = 1
    HIDDEN_SIZE = 5120
    INTERMEDIATE_SIZE = 8192
    BATCH_SIZE = 1
    SEQLEN = 16
    DTYPE = torch.float32
    DEVICE = "cuda"
    SEED = 42

    set_seed(SEED)

    x = torch.randn(BATCH_SIZE, SEQLEN, HIDDEN_SIZE, dtype=DTYPE, device=DEVICE)
    gate = nn.Linear(HIDDEN_SIZE, NUM_EXPERTS, bias=False).to(DEVICE)
    router_logits = gate(x)
    router_weights, selected_experts = compute_routing_weights(router_logits, TOPK)
    print(f"Router_logits: {router_logits.shape}")
    print(f"Router_weights: {router_weights}")
    print(f"Selected experts:\n{selected_experts}")

    with torch.no_grad():
        sorted_expert_idx, sorted_scattered_idx = selected_experts.flatten().sort()
        expert_offsets = torch.bincount(sorted_expert_idx, minlength=NUM_EXPERTS)
        tok_assignment_check = selected_experts.flatten().argsort()
        print(f"Tok assignment check: {tok_assignment_check}")
        assert (tok_assignment_check == sorted_scattered_idx).all(), f"Tok assignment check != sorted scattered idx: {tok_assignment_check} != {sorted_scattered_idx}"
        
    print(f"Sorted expert idxs: {sorted_expert_idx}")
    print(f"Sorted scattered idxs: {sorted_scattered_idx}")
    print(f"Expert offsets: {expert_offsets}")
    start_idx = 0
    for e in range(NUM_EXPERTS):
        num_assigned_tokens = expert_offsets[e]
        expert_assignment = sorted_expert_idx[start_idx:start_idx+num_assigned_tokens]
        assert (expert_assignment == e).all(), f"Expert {e} assigned {num_assigned_tokens}, sorted expert idx != {e}: {expert_assignment}"
        token_assignment = sorted_scattered_idx[start_idx:start_idx+num_assigned_tokens]
        print(f"Expert {e} assigned {num_assigned_tokens} tokens: {token_assignment}")
        
        start_idx += num_assigned_tokens

