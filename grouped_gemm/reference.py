# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.functional as F


def silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    return F.silu(x[..., :d]) * x[..., d:]

def torch_moe(a, w1, w2, score, topk, renormalize = False, score_func = F.sigmoid, return_topk_weights = False, return_selected_experts = False):
    B, D = a.shape
    a = a.view(B, -1, D).repeat(1, topk, 1).reshape(-1, D)
    out = torch.zeros(B * topk, w2.shape[1], dtype=a.dtype, device=a.device)

    if score_func == F.sigmoid:
        topk_weight = F.sigmoid(score)
    elif score_func == F.softmax:
        topk_weight = score.softmax(dim=-1, dtype=torch.float32)
    else:
        raise ValueError(f"Unsupported score function: {score_func}")
    topk_weight, topk_ids = torch.topk(topk_weight, topk)
    if renormalize:
        topk_weight = topk_weight / topk_weight.sum(dim=-1, keepdim=True)
    topk_weight = topk_weight.view(-1)
    topk_ids = topk_ids.view(-1)

    for i in range(w1.shape[0]):
        mask = topk_ids == i
        if mask.sum():
            out[mask] = silu_and_mul(
                a[mask] @ w1[i].transpose(0, 1)) @ w2[i].transpose(0, 1)
    out = (out.view(B, -1, w2.shape[1]) *
            topk_weight.view(B, -1, 1).to(out.dtype)).sum(dim=1)
    
    outputs = [out]
    if return_topk_weights:
        outputs.append(topk_weight)
    if return_selected_experts:
        outputs.append(topk_ids)
    return tuple(outputs) if len(outputs) > 1 else outputs[0]

def iterative_moe(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    global_num_experts: int,
    renormalize: bool = False,
    score_func = F.sigmoid,
    return_topk_weights: bool = False,
    return_selected_experts: bool = False,
) -> torch.Tensor:
    """
    Args:
        hidden_states: [*, hidden_size]
        w1: [num_experts, intermediate_size * 2, hidden_size]
        w2: [num_experts, hidden_size, intermediate_size]
        gating_output: [*, num_experts]
        expert_map: [num_experts]
    """
    orig_shape = hidden_states.shape
    hidden_size = hidden_states.shape[-1]
    num_tokens = hidden_states.shape[:-1].numel()
    num_experts = w1.shape[0]
    intermediate_size = w2.shape[-1]
    dtype = hidden_states.dtype

    hidden_states = hidden_states.view(num_tokens, hidden_size)
    gating_output = gating_output.view(num_tokens, global_num_experts)
    if score_func == F.sigmoid:
        topk_weights = F.sigmoid(gating_output)
    elif score_func == F.softmax:
        topk_weights = gating_output.softmax(dim=-1, dtype=torch.float)
    else:
        raise ValueError(f"Unsupported score function: {score_func}")
    topk_weights, selected_experts = topk_weights.topk(topk, dim=-1)
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    topk_weights = topk_weights.to(dtype)

    final_hidden_states = None
    
    for expert_idx in range(num_experts):
        expert_w1 = w1[expert_idx]
        expert_w2 = w2[expert_idx]
        expert_mask = (selected_experts == expert_idx)
        expert_weights = (topk_weights * expert_mask).sum(dim=-1, keepdim=True)
        x = F.linear(hidden_states, expert_w1)
        gate = F.silu(x[:, :intermediate_size])
        x = x[:, intermediate_size:] * gate
        x = F.linear(x, expert_w2)
        current_hidden_states = x * expert_weights
        if final_hidden_states is None:
            final_hidden_states = current_hidden_states
        else:
            final_hidden_states = final_hidden_states + current_hidden_states

    outputs = [final_hidden_states.view(orig_shape)]
    if return_topk_weights:
        outputs.append(topk_weights)
    if return_selected_experts:
        outputs.append(selected_experts)
    return tuple(outputs) if len(outputs) > 1 else outputs[0]

def test_fused_moe(M, N, K, E, topk, dtype):
    a = torch.randn((M, K), device="cuda", dtype=dtype) / 10
    w1 = torch.randn((E, 2 * N, K), device="cuda", dtype=dtype) / 10
    w2 = torch.randn((E, K, N), device="cuda", dtype=dtype) / 10

    score = torch.randn((M, E), device="cuda", dtype=dtype)


    torch_out, torch_weights, torch_selected_experts = torch_moe(a, w1, w2, score, topk, return_topk_weights=True, return_selected_experts=True)
    iterative_out, iterative_weights, iterative_selected_experts = iterative_moe(a,
                                     w1,
                                     w2,
                                     score,
                                     topk,
                                     global_num_experts=E,
                                     renormalize=False,
                                     return_topk_weights=True,
                                     return_selected_experts=True)
    print(f"torch_weights: {torch_weights}")
    print(f"iterative_weights: {iterative_weights}")
    print(f"torch_selected_experts: {torch_selected_experts}")
    print(f"iterative_selected_experts: {iterative_selected_experts}")
    diff = (torch_out - iterative_out).abs().max()

    print(f"Diff: {diff}")
    assert torch_selected_experts.equal(iterative_selected_experts.view_as(torch_selected_experts))
    assert torch_weights.equal(iterative_weights.view_as(torch_weights))
    assert diff < 1e-5

if __name__ == "__main__":
    BS = 1
    SEQLEN = 16
    M = BS * SEQLEN # num tokens
    K = 128 # hidden_size
    N = 256 # intermediate_size
    E = 4 # num_experts
    TOPK = 1
    DTYPE = torch.float32
    test_fused_moe(M, N, K, E, TOPK, DTYPE)
