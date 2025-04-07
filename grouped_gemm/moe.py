import random
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


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

def test_grouped_gemm_bf16(
    G: int,
    M: int,
    N: int,
    K: int,
) -> None:
    device = torch.device("cuda")
    a = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    b = torch.randn(N * G, K, dtype=torch.bfloat16, device=device)
    m_ends, _ = torch.sort(
        torch.randint(
            low=0, high=M, size=[G - 1], device=device, dtype=torch.int32
        )
        if M > 0
        else torch.zeros([G - 1], device=device, dtype=torch.int32)
    )
    m_ends = m_ends.tolist()
    m_starts = [0] + m_ends
    m_ends = m_ends + [M]
    m_sizes = torch.tensor(
        [m_ends[i] - m_starts[i] for i in range(G)], device=device
    ).to(torch.int32)
    print(f"M sizes: {m_sizes} {m_sizes.sum().item()}")

def test_router(
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    num_experts: int,
    topk: int,
    verbose: bool = False,
) -> None:
    device = torch.device("cuda")
    x = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float32, device=device)
    gate = nn.Linear(hidden_size, num_experts, bias=False).to(device)
    router_logits = gate(x)
    router_weights, selected_experts = compute_routing_weights(router_logits, topk)
    if verbose:
        print(f"Router_logits: {router_logits.shape}")
        print(f"Router_weights: {router_weights}")
        print(f"Selected experts:\n{selected_experts}")

    with torch.no_grad():
        sorted_expert_idx, sorted_token_idx = selected_experts.flatten().sort()
        expert_offsets = torch.bincount(sorted_expert_idx, minlength=NUM_EXPERTS)
        tok_assignment_check = selected_experts.flatten().argsort()
        assert (tok_assignment_check == sorted_token_idx).all(), f"Tok assignment check != sorted scattered idx: {tok_assignment_check} != {sorted_token_idx}"

    if verbose:
        print(f"Sorted expert idxs: {sorted_expert_idx}")
        print(f"Sorted scattered idxs: {sorted_token_idx}")
        print(f"Expert offsets: {expert_offsets}")
    start_idx = 0
    for e in range(NUM_EXPERTS):
        num_assigned_tokens = expert_offsets[e]
        expert_assignment = sorted_expert_idx[start_idx:start_idx+num_assigned_tokens]
        assert (expert_assignment == e).all(), f"Expert {e} assigned {num_assigned_tokens}, sorted expert idx != {e}: {expert_assignment}"
        token_assignment = sorted_token_idx[start_idx:start_idx+num_assigned_tokens]

        if verbose:
            print(f"Expert {e} assigned {num_assigned_tokens} tokens: {token_assignment}")
        
        start_idx += num_assigned_tokens
    return sorted_expert_idx, sorted_token_idx

def test_gather(A, sorted_token_idx, sorted_expert_idx, num_experts) -> None:
    # Group sizes correspond to M in grouped gemm for each expert
    group_sizes = torch.bincount(sorted_expert_idx, minlength=num_experts)

    assert A.ndim == 2, f"A must be 2D, got {A.ndim}"
    assert sorted_expert_idx.shape[0] == A.shape[0], f"sorted_expert_idx.shape[0] must match A.shape[0], got {sorted_expert_idx.shape[0]} != {A.shape[0]}"
    gather_idx = sorted_token_idx.flatten().unsqueeze(-1).expand_as(A)

    print(f"A: \n{A}")
    group_start = 0
    for e in range(num_experts):
        group_size = group_sizes[e]
        if group_size == 0:
            continue
        group_end = group_start + group_size
        group_row_idx = gather_idx[group_start:group_end]
        assigned_rows = sorted_token_idx[group_start:group_end]
        A_group = A.gather(0, group_row_idx)
        print(f"Expert {e} group size: {group_size}:\nAssigned rows: {assigned_rows}, A_group: {A_group.shape}\n{A_group}")
        group_start = group_end


@triton.jit
def histogram_kernel(x_ptr, z_ptr, M: tl.constexpr, N: tl.constexpr):
    x_idx = tl.arange(0, M)
    z_idx = tl.arange(0, N)
    x = tl.load(x_ptr + x_idx)
    tl.device_print("x", x)
    z = tl.histogram(x.to(tl.int32), num_bins=N)
    tl.device_print("z", z)
    tl.store(z_ptr + z_idx, z)

if __name__ == "__main__":
    NUM_EXPERTS = 8
    TOPK = 1
    HIDDEN_SIZE = 4
    INTERMEDIATE_SIZE = 8192
    BATCH_SIZE = 1
    SEQLEN = 16
    DTYPE = torch.float32
    DEVICE = "cuda"
    SEED = 42

    set_seed(SEED)

    # test_router(BATCH_SIZE, SEQLEN, HIDDEN_SIZE, NUM_EXPERTS, TOPK)
    # test_grouped_gemm_bf16(G=NUM_EXPERTS, M=BATCH_SIZE * SEQLEN, N=INTERMEDIATE_SIZE, K=HIDDEN_SIZE)
    A = torch.arange(BATCH_SIZE * SEQLEN * HIDDEN_SIZE, dtype=DTYPE, device=DEVICE).view(BATCH_SIZE * SEQLEN, HIDDEN_SIZE)
    sorted_expert_idx, sorted_token_idx = test_router(BATCH_SIZE, SEQLEN, HIDDEN_SIZE, NUM_EXPERTS, TOPK)
    print(f"Sorted expert idx: {sorted_expert_idx}")
    print(f"Sorted token idx: {sorted_token_idx}")
    # test_gather(A, sorted_token_idx, sorted_expert_idx, NUM_EXPERTS)
    M = sorted_token_idx.shape[0]
    N = NUM_EXPERTS
    z = torch.empty(N, dtype=torch.int32, device=DEVICE)
    histogram_kernel[(1,)](sorted_expert_idx, z, M, N)

    ref_hist = sorted_expert_idx.bincount(minlength=N)
    print(f"Z: {z}")
    print(f"Ref hist: {ref_hist}")
    assert (z == ref_hist).all(), f"Z != ref_hist: {z} != {ref_hist}"