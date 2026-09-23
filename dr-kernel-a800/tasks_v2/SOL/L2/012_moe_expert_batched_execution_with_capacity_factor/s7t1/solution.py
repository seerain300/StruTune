import torch
import torch.nn.functional as F
import math
import triton
import triton.language as tl


@triton.jit
def bmm_gate_kernel(
    A_ptr,              # [1, K] view pointer (hidden_inputs[e, n, :]), bfloat16
    B_ptr,              # [num_experts, K, J] pointer (expert_gate_weights), bfloat16
    C_ptr,              # [1, J] pointer (output gate_out[e, n, :]), bfloat16
    K, J,               # ints: K=hidden_size, J=moe_intermediate_size
    stride_B_e, stride_B_k, stride_B_j,
    BLOCK_K: tl.constexpr, BLOCK_J: tl.constexpr,
):
    # Single program computes one output row (e, n) into C[0, :]
    # Load A as 1xK vector
    offs_k = tl.arange(0, BLOCK_K)
    k_idx = offs_k
    mask_k = k_idx < K

    a_ptrs = A_ptr + k_idx * 1  # A is 1D contiguous, stride over k
    a = tl.load(a_ptrs, mask=mask_k, other=0.0)

    offs_j = tl.arange(0, BLOCK_J)
    acc = tl.zeros((BLOCK_J,), dtype=tl.bfloat16)

    # Reduce over K in tiles
    for k0 in range(0, K, BLOCK_K):
        k_idx = k0 + offs_k
        mask_k = k_idx < K
        a = tl.load(A_ptr + k_idx * 1, mask=mask_k, other=0.0)  # [BLOCK_K]
        b_ptrs = B_ptr + 0 * stride_B_e + k_idx[:, None] * stride_B_k + offs_j[None, :] * stride_B_j  # e fixed at 0 for this program
        b = tl.load(b_ptrs, mask=mask_k[:, None], other=0.0)    # [BLOCK_K, BLOCK_J]
        acc += tl.sum(b * a[:, None], axis=0)

    # Store to C[0, :]
    c_ptrs = C_ptr + offs_j * 1
    tl.store(c_ptrs, acc, mask=offs_j < J)


@triton.jit
def bmm_up_kernel(
    A_ptr,              # [1, K] view pointer (hidden_inputs[e, n, :]), bfloat16
    B_ptr,              # [num_experts, K, J] pointer (expert_up_weights), bfloat16
    C_ptr,              # [1, J] pointer (output up_out[e, n, :]), bfloat16
    K, J,               # ints: K=hidden_size, J=moe_intermediate_size
    stride_B_e, stride_B_k, stride_B_j,
    BLOCK_K: tl.constexpr, BLOCK_J: tl.constexpr,
):
    offs_k = tl.arange(0, BLOCK_K)
    k_idx = offs_k
    mask_k = k_idx < K
    a_ptrs = A_ptr + k_idx * 1
    a = tl.load(a_ptrs, mask=mask_k, other=0.0)

    offs_j = tl.arange(0, BLOCK_J)
    acc = tl.zeros((BLOCK_J,), dtype=tl.bfloat16)

    for k0 in range(0, K, BLOCK_K):
        k_idx = k0 + offs_k
        mask_k = k_idx < K
        a = tl.load(A_ptr + k_idx * 1, mask=mask_k, other=0.0)
        b_ptrs = B_ptr + 0 * stride_B_e + k_idx[:, None] * stride_B_k + offs_j[None, :] * stride_B_j
        b = tl.load(b_ptrs, mask=mask_k[:, None], other=0.0)
        acc += tl.sum(b * a[:, None], axis=0)

    c_ptrs = C_ptr + offs_j * 1
    tl.store(c_ptrs, acc, mask=offs_j < J)


@triton.jit
def bmm_down_all_kernel(
    A_ptr,              # [1, K] pointer (activated[e, n, :]), bfloat16, K=J
    B_ptr,              # [num_experts, K, H] pointer (expert_down_weights), bfloat16
    C_ptr,              # [1, H] pointer (output expert_outputs_all[e, n, :]), bfloat16
    K, H,               # ints: K=J=intermediate_size, H=hidden_size
    stride_B_e, stride_B_k, stride_B_j,
    BLOCK_K: tl.constexpr, BLOCK_H: tl.constexpr,
):
    offs_k = tl.arange(0, BLOCK_K)
    k_idx = offs_k
    mask_k = k_idx < K
    a_ptrs = A_ptr + k_idx * 1
    a = tl.load(a_ptrs, mask=mask_k, other=0.0)

    offs_h = tl.arange(0, BLOCK_H)
    acc = tl.zeros((BLOCK_H,), dtype=tl.bfloat16)

    for k0 in range(0, K, BLOCK_K):
        k_idx = k0 + offs_k
        mask_k = k_idx < K
        a = tl.load(A_ptr + k_idx * 1, mask=mask_k, other=0.0)
        b_ptrs = B_ptr + 0 * stride_B_e + k_idx[:, None] * stride_B_k + offs_h[None, :] * stride_B_j
        b = tl.load(b_ptrs, mask=mask_k[:, None], other=0.0)  # [BLOCK_K, BLOCK_H]
        acc += tl.sum(b * a[:, None], axis=0)

    c_ptrs = C_ptr + offs_h * 1
    tl.store(c_ptrs, acc, mask=offs_h < H)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                selected_experts: torch.Tensor,
                routing_weights: torch.Tensor,
                expert_gate_weights: torch.Tensor,
                expert_up_weights: torch.Tensor,
                expert_down_weights: torch.Tensor):
        # Inputs:
        # hidden_states: [num_tokens, hidden_size], bfloat16
        # selected_experts: [num_tokens, num_experts_per_tok], int64
        # routing_weights: [num_tokens, num_experts_per_tok], bfloat16
        # expert_gate_weights: [num_experts, hidden_size, intermediate_size], bfloat16
        # expert_up_weights: [num_experts, hidden_size, intermediate_size], bfloat16
        # expert_down_weights: [num_experts, intermediate_size, hidden_size], bfloat16

        num_tokens, hidden_size = hidden_states.shape
        num_experts = expert_gate_weights.shape[0]
        K = hidden_size
        J = expert_gate_weights.shape[2]  # intermediate size
        H = expert_down_weights.shape[2]  # final hidden size

        # Capacity
        num_experts_per_tok = selected_experts.shape[1]
        total_selected = num_tokens * num_experts_per_tok
        capacity = max(int((total_selected * 1.25) // num_experts), 1)

        # Flatten and sort by selected expert ID
        flat_experts = selected_experts.reshape(-1)                 # [num_tokens * num_experts_per_tok]
        flat_weights = routing_weights.reshape(-1)                 # [num_tokens * num_experts_per_tok]
        flat_token_ids = torch.arange(num_tokens, device=hidden_states.device).repeat_interleave(num_experts_per_tok)

        sorted_experts, sorted_indices = torch.sort(flat_experts, stable=True)
        sorted_weights = flat_weights[sorted_indices]
        sorted_token_ids = flat_token_ids[sorted_indices]

        counts = torch.bincount(sorted_experts, minlength=num_experts)
        starts = torch.zeros(num_experts, dtype=torch.long, device=hidden_states.device)
        starts[1:] = counts[:-1].cumsum(0)
        within_pos = torch.arange(len(sorted_experts), device=hidden_states.device) - starts[sorted_experts]
        valid = within_pos < capacity

        v_exp = sorted_experts[valid]
        v_pos = within_pos[valid]
        v_tok = sorted_token_ids[valid]
        v_wt = sorted_weights[valid]

        num_valid = v_exp.numel()

        # Build expert_inputs: [num_experts, capacity, hidden_size]
        expert_inputs = torch.zeros((num_experts, capacity, K), dtype=hidden_states.dtype, device=hidden_states.device)
        if num_valid > 0:
            expert_inputs[v_exp, v_pos] = hidden_states[v_tok]

        # Allocate outputs for gate_out, up_out, expert_outputs_all
        gate_out = torch.empty((num_experts, capacity, J), dtype=hidden_states.dtype, device=hidden_states.device)
        up_out = torch.empty((num_experts, capacity, J), dtype=hidden_states.dtype, device=hidden_states.device)
        expert_outputs_all = torch.empty((num_experts, capacity, H), dtype=hidden_states.dtype, device=hidden_states.device)

        # Launch Triton kernels: one program per (e, n)
        # BLOCK sizes chosen conservatively; adjust as needed.
        BLOCK_K = 64
        BLOCK_J = 64
        BLOCK_H = 64

        # Gate BMM
        for e in range(num_experts):
            for n in range(capacity):
                grid = (1,)  # single program per (e, n)
                bmm_gate_kernel[grid](
                    expert_inputs[e, n],      # 1D contiguous vector of length K
                    expert_gate_weights,      # [num_experts, K, J]
                    gate_out[e, n],           # 1D output vector of length J
                    K, J,
                    expert_gate_weights.stride(0), expert_gate_weights.stride(1), expert_gate_weights.stride(2),
                    BLOCK_K=BLOCK_K, BLOCK_J=BLOCK_J,
                    num_warps=4
                )

        # Up BMM
        for e in range(num_experts):
            for n in range(capacity):
                grid = (1,)
                bmm_up_kernel[grid](
                    expert_inputs[e, n],
                    expert_up_weights,
                    up_out[e, n],
                    K, J,
                    expert_up_weights.stride(0), expert_up_weights.stride(1), expert_up_weights.stride(2),
                    BLOCK_K=BLOCK_K, BLOCK_J=BLOCK_J,
                    num_warps=4
                )

        # Down BMM: compute activated per (e,n), then use down kernel (we need activated first). To avoid an extra Python loop, we compute activated via PyTorch for simplicity, since this workload isn't the bottleneck. This keeps the Triton work strictly on BMMs, as required.
        # activated = SiLU(gate_out) * up_out
        for e in range(num_experts):
            for n in range(capacity):
                g = gate_out[e, n]   # [J], bfloat16
                u = up_out[e, n]     # [J], bfloat16
                activated = F.silu(g.float()).to(g.dtype) * u  # PyTorch elementwise
                # Now do down bmm
                grid = (1,)
                bmm_down_all_kernel[grid](
                    activated,          # 1D vector of length J
                    expert_down_weights,
                    expert_outputs_all[e, n],
                    J, H,
                    expert_down_weights.stride(0), expert_down_weights.stride(1), expert_down_weights.stride(2),
                    BLOCK_K=BLOCK_J, BLOCK_H=BLOCK_H,
                    num_warps=4
                )

        # Final aggregation: scatter-add weighted outputs into result [num_tokens, H]
        result = torch.zeros((num_tokens, H), dtype=hidden_states.dtype, device=hidden_states.device)
        for i in range(num_valid):
            e = int(v_exp[i].item())
            pos = int(v_pos[i].item())
            tok = int(v_tok[i].item())
            wt = v_wt[i]  # bfloat16 tensor
            out_vec = expert_outputs_all[e, pos]  # [H]
            result[tok] += out_vec * wt

        return result


def run(*args):
    return ModelNew()(*args)
