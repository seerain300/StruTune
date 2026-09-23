import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _bmm_gate_kernel(
    A_ptr,         # *bf16, (B, H) padded inputs per expert
    W_ptr,         # *bf16, (N, C) gate weights per expert (N=H, C=I)
    Out_ptr,       # *bf16, (B, C) output (B=capacity, C=I)
    B, H, C,       # sizes: B=batch (capacity), H=hidden_size, C=intermediate_size
    BLOCK_N: tl.constexpr,  # tile over C
    BLOCK_K: tl.constexpr,  # tile over H
):
    pid_b = tl.program_id(0)  # batch row index
    pid_n = tl.program_id(1)  # output feature index tile

    if pid_b >= B:
        return

    n_start = pid_n * BLOCK_N
    n_idx = n_start + tl.arange(0, BLOCK_N)
    n_mask = n_idx < C

    # Accumulator for this (pid_b, n tile)
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # Loop over K dimension (hidden_size) in tiles
    for k_start in range(0, H, BLOCK_K):
        k_idx = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_idx < H

        # Load A[pid_b, k] -> (BLOCK_K,)
        a_off = pid_b * H + k_idx
        a = tl.load(A_ptr + a_off, mask=k_mask, other=0.0).to(tl.float32)  # (BLOCK_K,)

        # Load W[k, n] -> (BLOCK_K, BLOCK_N)
        w_off = k_idx[:, None] * C + n_idx[None, :]  # k*intermediate + n
        w = tl.load(W_ptr + w_off, mask=k_mask[:, None] & n_mask[None, :], other=0.0).to(tl.float32)

        # Accumulate dot: sum over k dimension
        acc += tl.sum(w * a[:, None], axis=0)

    # Store result (bf16)
    out_off = pid_b * C + n_idx
    tl.store(Out_ptr + out_off, acc, mask=n_mask)


@triton.jit
def _bmm_up_kernel(
    A_ptr,         # *bf16, (B, H) padded inputs per expert
    W_ptr,         # *bf16, (N, C) up weights per expert (N=H, C=I)
    Out_ptr,       # *bf16, (B, C) output (B=capacity, C=I)
    B, H, C,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)

    if pid_b >= B:
        return

    n_start = pid_n * BLOCK_N
    n_idx = n_start + tl.arange(0, BLOCK_N)
    n_mask = n_idx < C

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for k_start in range(0, H, BLOCK_K):
        k_idx = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_idx < H

        a_off = pid_b * H + k_idx
        a = tl.load(A_ptr + a_off, mask=k_mask, other=0.0).to(tl.float32)

        w_off = k_idx[:, None] * C + n_idx[None, :]
        w = tl.load(W_ptr + w_off, mask=k_mask[:, None] & n_mask[None, :], other=0.0).to(tl.float32)

        acc += tl.sum(w * a[:, None], axis=0)

    out_off = pid_b * C + n_idx
    tl.store(Out_ptr + out_off, acc, mask=n_mask)


@triton.jit
def _bmm_down_kernel(
    A_ptr,         # *bf16, (B, C) activated values (B=capacity, C=I)
    W_ptr,         # *bf16, (C, N) down weights per expert (C=I, N=H)
    Out_ptr,       # *bf16, (B, N) output (B=capacity, N=H)
    B, C, N,       # sizes: B=batch, C=intermediate_size, N=hidden_size
    BLOCK_N: tl.constexpr,  # tile over N
    BLOCK_K: tl.constexpr,  # tile over C
):
    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)

    if pid_b >= B:
        return

    n_start = pid_n * BLOCK_N
    n_idx = n_start + tl.arange(0, BLOCK_N)
    n_mask = n_idx < N

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for k_start in range(0, C, BLOCK_K):
        k_idx = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_idx < C

        a_off = pid_b * C + k_idx
        a = tl.load(A_ptr + a_off, mask=k_mask, other=0.0).to(tl.float32)  # (BLOCK_K,)

        # W has shape (C, N): loading W[k, n] -> (BLOCK_K, BLOCK_N)
        w_off = k_idx[:, None] * N + n_idx[None, :]  # k*hidden + n
        w = tl.load(W_ptr + w_off, mask=k_mask[:, None] & n_mask[None, :], other=0.0).to(tl.float32)

        acc += tl.sum(w * a[:, None], axis=0)

    out_off = pid_b * N + n_idx
    tl.store(Out_ptr + out_off, acc, mask=n_mask)


def _launch_bmm(kernel, A, W, Out, B, H, C, block_n=128, block_k=64, num_warps=4):
    grid = (B, triton.cdiv(C, block_n))
    kernel[grid](A, W, Out, B, H, C, BLOCK_N=block_n, BLOCK_K=block_k, num_warps=num_warps, num_stages=2)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                selected_experts: torch.Tensor,
                routing_weights: torch.Tensor,
                expert_gate_weights: torch.Tensor,
                expert_up_weights: torch.Tensor,
                expert_down_weights: torch.Tensor) -> torch.Tensor:
        # Ensure CUDA tensors
        assert hidden_states.is_cuda and selected_experts.is_cuda and routing_weights.is_cuda \
               and expert_gate_weights.is_cuda and expert_up_weights.is_cuda and expert_down_weights.is_cuda, \
            "All inputs must be CUDA tensors for Triton execution."
        assert hidden_states.dtype == torch.bfloat16, "hidden_states must be bfloat16"
        assert routing_weights.dtype == torch.bfloat16, "routing_weights must be bfloat16"

        num_tokens, hidden_size = hidden_states.shape
        num_experts, _, intermediate_size = expert_gate_weights.shape
        K = selected_experts.shape[1]
        device = hidden_states.device

        # Original preprocessing: flatten, stable sort, bincount, starts, within_pos, capacity
        flat_experts = selected_experts.reshape(-1).to(torch.long)  # (num_tokens*K,)
        flat_weights = routing_weights.reshape(-1).to(torch.bfloat16)
        flat_token_ids = torch.arange(num_tokens, device=device).repeat_interleave(K)

        sorted_experts, sorted_indices = torch.sort(flat_experts, stable=True)
        sorted_weights = flat_weights[sorted_indices]
        sorted_token_ids = flat_token_ids[sorted_indices]

        counts = torch.bincount(sorted_experts, minlength=num_experts)
        starts = torch.cumsum(counts, dim=0)  # starts[j] = sum_{i<j} counts[i]
        starts = starts - counts  # move starts to prefix of each expert
        within_pos = torch.arange(len(sorted_experts), device=device) - starts[sorted_experts]

        capacity = max(int((num_tokens * K) * 1.25 / num_experts), 1)
        valid = within_pos < capacity
        v_exp = sorted_experts[valid]
        v_pos = within_pos[valid]
        v_tok = sorted_token_ids[valid]
        v_wt = sorted_weights[valid]

        # Build padded expert_inputs: (num_experts, capacity, hidden_size)
        expert_inputs = torch.zeros((num_experts, capacity, hidden_size),
                                    dtype=torch.bfloat16, device=device)
        expert_inputs[v_exp, v_pos] = hidden_states[v_tok]

        # Allocate outputs (float32 for accumulation)
        gate_out = torch.empty((num_experts, capacity, intermediate_size), dtype=torch.float32, device=device)
        up_out   = torch.empty((num_experts, capacity, intermediate_size), dtype=torch.float32, device=device)
        expert_outputs = torch.empty((num_experts, capacity, hidden_size), dtype=torch.float32, device=device)

        # Triton BMMs per expert
        for e in range(num_experts):
            A = expert_inputs[e]                  # (B, H), B=capacity, H=hidden_size
            W_gate = expert_gate_weights[e]       # (H, C)
            W_up = expert_up_weights[e]           # (H, C)

            # Gate: (B, C)
            _launch_bmm(_bmm_gate_kernel, A, W_gate, gate_out[e], A.shape[0], A.shape[1], W_gate.shape[1],
                        block_n=128, block_k=64, num_warps=4)
            # Up: (B, C)
            _launch_bmm(_bmm_up_kernel, A, W_up, up_out[e], A.shape[0], A.shape[1], W_up.shape[1],
                        block_n=128, block_k=64, num_warps=4)

            # SiLU and gated-up in PyTorch (pointwise ops)
            silu_gate = F.silu(gate_out[e])      # (B, C)
            activated = silu_gate * up_out[e]    # (B, C)

            # Down: (B, N)
            W_down = expert_down_weights[e]      # (C, N)
            _launch_bmm(_bmm_down_kernel, activated, W_down, expert_outputs[e], A.shape[0], W_down.shape[0], W_down.shape[1],
                        block_n=128, block_k=64, num_warps=4)

        # Gather valid outputs and weighted contributions
        valid_out = expert_outputs[v_exp, v_pos]  # (num_valid, hidden_size) float32
        weighted_out = (v_wt.unsqueeze(1) * valid_out).to(torch.bfloat16)  # cast to bf16 for final aggregation

        # Scatter-add into result
        result = torch.zeros((num_tokens, hidden_size), dtype=torch.bfloat16, device=device)
        result.index_add_(0, v_tok, weighted_out)

        return result


def run(*args):
    return ModelNew()(*args)
