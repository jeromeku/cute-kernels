# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.functional as F


def silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    return F.silu(x[..., :d]) * x[..., d:]


def make_inputs(M, N, K, E, topk, dtype):
    a = torch.randn((M, K), device="cuda", dtype=dtype) / 10
    w1 = torch.randn((E, 2 * N, K), device="cuda", dtype=dtype) / 10
    w2 = torch.randn((E, K, N), device="cuda", dtype=dtype) / 10
    score = torch.randn((M, E), device="cuda", dtype=dtype)
    return a, w1, w2, score


def calculate_topk(gating_output, topk, score_func=F.sigmoid, renormalize=False):
    if score_func == F.sigmoid:
        kwargs = {}
    elif score_func == F.softmax:
        kwargs = {"dim": -1, "dtype": torch.float32}
    else:
        raise ValueError(f"Unsupported score function: {score_func}")
    topk_weights = score_func(gating_output, **kwargs)
    topk_weights, topk_ids = torch.topk(topk_weights, topk)
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    topk_weights = topk_weights.to(gating_output.dtype)

    return topk_weights, topk_ids


def torch_moe(
    *,
    a,
    w1,
    w2,
    gating_output,
    topk,
    renormalize=False,
    score_func=F.sigmoid,
    return_topk_weights=False,
    return_selected_experts=False,
):
    B, D = a.shape
    a = a.view(B, -1, D).repeat(1, topk, 1).reshape(-1, D)
    out = torch.zeros(B * topk, w2.shape[1], dtype=a.dtype, device=a.device)

    topk_weight, topk_ids = calculate_topk(
        gating_output, topk, score_func=score_func, renormalize=renormalize
    )
    topk_weight = topk_weight.view(-1)
    topk_ids = topk_ids.view(-1)

    for i in range(w1.shape[0]):
        mask = topk_ids == i
        if mask.sum():
            out[mask] = silu_and_mul(a[mask] @ w1[i].transpose(0, 1)) @ w2[i].transpose(
                0, 1
            )
    out = (out.view(B, -1, w2.shape[1]) * topk_weight.view(B, -1, 1).to(out.dtype)).sum(
        dim=1
    )

    outputs = [out]
    if return_topk_weights:
        outputs.append(topk_weight)
    if return_selected_experts:
        outputs.append(topk_ids)
    return tuple(outputs) if len(outputs) > 1 else outputs[0]


def iterative_moe(
    *,
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    global_num_experts: int,
    renormalize: bool = False,
    score_func=F.sigmoid,
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
    assert hidden_states.dtype == gating_output.dtype

    hidden_states = hidden_states.view(num_tokens, hidden_size)
    gating_output = gating_output.view(num_tokens, global_num_experts)

    topk_weights, topk_ids = calculate_topk(
        gating_output, topk, score_func=score_func, renormalize=renormalize
    )
    final_hidden_states = None

    for expert_idx in range(num_experts):
        expert_w1 = w1[expert_idx]
        expert_w2 = w2[expert_idx]
        expert_mask = topk_ids == expert_idx
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
        outputs.append(topk_ids)
    return tuple(outputs) if len(outputs) > 1 else outputs[0]


def get_sorted_tokens_by_expert(selected_experts, num_experts):
    """
    sorted_expert_idx: [num_tokens] needed to calculate tokens per expert -- see torchtitan for alternative impl that does not require bincount
    sorted_token_idx: [num_tokens]
    token_counts_by_expert: [num_experts]
    """
    with torch.no_grad():
        sorted_expert_idx, sorted_token_idx = selected_experts.flatten().sort()
        token_counts_by_expert = torch.bincount(
            sorted_expert_idx, minlength=num_experts
        )
        return sorted_expert_idx, sorted_token_idx, token_counts_by_expert


def gather_moe(
    *,
    a,
    w1,
    w2,
    gating_output,
    topk,
    renormalize=False,
    score_func=F.sigmoid,
    debug=False,
    verbose=False,
):
    a = a.view(-1, a.shape[-1])
    B, K = a.shape  # B = num_tokens, K = hidden_size
    E, K, N = w2.shape  # E = num_experts, K = hidden_size, N = intermediate_size
    assert w1.shape == (E, 2 * N, K)
    topk_weights, topk_ids = calculate_topk(
        gating_output, topk, score_func=score_func, renormalize=renormalize
    )
    topk_weights = topk_weights.view(-1)  # num_tokens * topk
    topk_ids = topk_ids.view(-1)  # num_tokens * topk

    # 1. Sort token assignments by expert
    sorted_expert_idx, sorted_token_idx, token_counts_by_expert = (
        get_sorted_tokens_by_expert(topk_ids, num_experts=E)
    )

    # Simulate grouped gemm by running num_expert gemms
    # The size of each gemm is M_size x K x N where M_size is the number of tokens assigned to expert e
    acc = torch.zeros((B, K), device=a.device, dtype=a.dtype)

    token_start = 0
    for e in range(E):
        M_size = token_counts_by_expert[e]
        if M_size == 0:
            continue
        token_end = token_start + M_size
        assert (sorted_expert_idx == e).sum() == M_size, (
            f"Expert {e} has {token_counts_by_expert[e]} tokens, but {sorted_expert_idx.shape[0]} tokens are assigned to expert {e}"
        )

        token_idx = sorted_token_idx[token_start:token_end]
        A_ = a[token_idx]  # [M_size, K] -> select tokens assigned to expert e
        W1_ = w1[e]  # [2 * N, K] -> select expert e
        W2_ = w2[e]  # [K, N] -> select expert e
        # Accumulate does not apply for topk == 1
        intermediate_out = silu_and_mul(A_ @ W1_.transpose(0, 1))  # [M_size, N]
        assert intermediate_out.shape == (M_size, N)
        out = intermediate_out @ W2_.transpose(0, 1)  # [M_size, K]
        assert out.shape == (M_size, K)
        # Apply topk weights, make sure to broadcast to the correct shape
        out = out * topk_weights[token_idx, None]
        assert out.shape == (M_size, K)
        # Accumulate
        acc[token_idx] += out
        token_start = token_end

    return (
        (
            acc,
            topk_weights,
            topk_ids,
            sorted_expert_idx,
            sorted_token_idx,
            token_counts_by_expert,
        )
        if debug
        else acc
    )


def test_fused_moe(
    M, N, K, E, topk, dtype, verbose=False, debug=False, test_iterative=False
):
    a, w1, w2, gating_output = make_inputs(M, N, K, E, topk, dtype)

    torch_out = torch_moe(
        a=a,
        w1=w1,
        w2=w2,
        gating_output=gating_output,
        topk=topk,
        return_topk_weights=debug,
        return_selected_experts=debug,
    )
    if debug:
        torch_out, torch_topk_weights, torch_selected_experts = torch_out

    gather_out = gather_moe(
        a=a,
        w1=w1,
        w2=w2,
        gating_output=gating_output,
        topk=topk,
        score_func=score_func,
        debug=debug,
        renormalize=renormalize,
    )
    if debug:
        (
            gather_out,
            gather_topk_weights,
            gather_selected_experts,
            sorted_expert_idx,
            sorted_token_idx,
            token_counts_by_expert,
        ) = gather_out
        assert torch_topk_weights.equal(
            gather_topk_weights.view_as(torch_topk_weights)
        ), (
            f"torch_topk_weights: {torch_topk_weights}\ngather_topk_weights: {gather_topk_weights}"
        )
        assert torch_selected_experts.equal(
            gather_selected_experts.view_as(torch_selected_experts)
        ), (
            f"torch_selected_experts: {torch_selected_experts}\ngather_selected_experts: {gather_selected_experts}"
        )

    diff = (torch_out - gather_out).abs().max()
    print(f"torch vs gather: {diff}")
    assert diff < 1e-5

    if test_iterative:
        iterative_out = iterative_moe(
            a=a,
            w1=w1,
            w2=w2,
            gating_output=gating_output,
            topk=topk,
            global_num_experts=E,
            renormalize=False,
            return_topk_weights=debug,
            return_selected_experts=debug,
        )
        if debug:
            iterative_out, iterative_topk_weights, iterative_selected_experts = (
                iterative_out
            )
        if verbose:
            print(f"torch_weights: {torch_topk_weights}")
            print(f"iterative_weights: {iterative_topk_weights}")
            print(f"torch_selected_experts: {torch_selected_experts}")
            print(f"iterative_selected_experts: {iterative_selected_experts}")
        if debug:
            assert torch_selected_experts.equal(
                iterative_selected_experts.view_as(torch_selected_experts)
            )
            assert torch_topk_weights.equal(
                iterative_topk_weights.view_as(torch_topk_weights)
            )

        diff = (torch_out - iterative_out).abs().max()

        print(f"torch vs iterative: {diff}")
        assert diff < 1e-5


def test_bincompile(num_tokens, num_experts):
    selected_experts = torch.randint(0, num_experts, (num_tokens,), device="cuda")
    sorted_expert_idx, sorted_token_idx = selected_experts.sort()
    token_counts_by_expert = torch.bincount(sorted_expert_idx, minlength=num_experts)
    token_counts_compiled = torch.compile(torch.bincount)(
        sorted_expert_idx, minlength=num_experts
    )
    assert token_counts_compiled.equal(token_counts_by_expert)


def get_grouped_gemm_inputs(gating_output, topk, num_experts, use_bincount=True):
    topk_weights, topk_ids = calculate_topk(
        gating_output, topk, score_func=score_func, renormalize=renormalize
    )
    topk_weights = topk_weights.view(-1)  # num_tokens * topk
    topk_ids = topk_ids.view(-1)  # num_tokens * topk
    
    num_tokens = topk_ids.shape[0]
    # Alternative to bincount
    
    if use_bincount:
        # 1. Sort token assignments by expert
        sorted_expert_idx, sorted_token_idx, token_counts_by_expert = (
            get_sorted_tokens_by_expert(topk_ids, num_experts=E)
        )
        return topk_weights, topk_ids, sorted_expert_idx, sorted_token_idx, token_counts_by_expert

    else:
        counts = topk_ids.new_zeros((num_tokens, num_experts))
        counts.scatter_(1, topk_ids.unsqueeze(-1), 1)
        token_counts_by_expert_no_bincount = counts.sum(dim=0)
        return topk_weights, topk_ids, token_counts_by_expert_no_bincount

if __name__ == "__main__":
    BS = 1
    SEQLEN = 16
    M = BS * SEQLEN  # num tokens
    K = 128  # hidden_size
    N = 256  # intermediate_size
    E = 4  # num_experts
    TOPK = 1
    DTYPE = torch.float32
    renormalize = False
    score_func = F.sigmoid
    debug = False
    test_iterative = False
    #    test_fused_moe(M, N, K, E, TOPK, DTYPE, debug=debug, test_iterative=test_iterative)
    a, w1, w2, gating_output = make_inputs(M, N, K, E, TOPK, DTYPE)
    _, _, token_counts_by_expert_no_bincount = get_grouped_gemm_inputs(gating_output, topk=TOPK, num_experts=E, use_bincount=False)
    #torch._dynamo.config.capture_dynamic_output_shape_ops = True
    *_, token_counts_by_expert_compiled = torch.compile(get_grouped_gemm_inputs, fullgraph=True)(gating_output, TOPK, E, use_bincount=False)
    assert token_counts_by_expert_no_bincount.equal(token_counts_by_expert_compiled)
    