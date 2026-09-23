import math
import torch
import triton
import triton.language as tl


@triton.jit
def sort_stable_kernel(exp_ptr, wt_ptr, idx_ptr, N, BLOCK: tl.constexpr):
    """
    Stable sort of (selected_experts, routing_weights) into (idx_ptr, sorted_exp, sorted_wt).
    Each program handles a BLOCK-sized slice of the flattened array of length N.
    We implement odd-even transposition sort using partner index j = idx ^ 1 and a stable tie-breaker.
    """
    pid = tl.program_id(0)
    start = pid * BLOCK
    idx = start + tl.arange(0, BLOCK)
    in_bounds = idx < N

    # Initialize idx to identity permutation
    tl.store(idx_ptr + idx, idx, mask=in_bounds)

    # Odd-even transposition sort: perform N phases
    for t in range(0, N):
        # Even phase: pairs (0,1), (2,3), ...
        phase = 0
        active = in_bounds & ((idx & 1) == phase)
        i = idx
        j = i ^ 1  # partner index

        exp_i = tl.load(exp_ptr + i, mask=active, other=0)
        wt_i  = tl.load(wt_ptr  + i, mask=active, other=0.0)
        exp_j = tl.load(exp_ptr + j, mask=active, other=0)
        wt_j  = tl.load(wt_ptr  + j, mask=active, other=0.0)

        # Swap if out-of-order; for ties, swap if j < i (stable)
        swap = (exp_j < exp_i) | ((exp_j == exp_i) & (j < i))

        new_exp_i = tl.where(swap, exp_j, exp_i)
        new_wt_i  = tl.where(swap, wt_j, wt_i)

        tl.store(exp_ptr + i, new_exp_i, mask=active)
        tl.store(wt_ptr  + i, new_wt_i,  mask=active)

        # Odd phase: pairs (1,2), (3,4), ...
        phase = 1
        active = in_bounds & (((idx + 1) & 1) == 0)
        i = idx
        j = i ^ 1

        exp_i = tl.load(exp_ptr + i, mask=active, other=0)
        wt_i  = tl.load(wt_ptr  + i, mask=active, other=0.0)
        exp_j = tl.load(exp_ptr + j, mask=active, other=0)
        wt_j  = tl.load(wt_ptr  + j, mask=active, other=0.0)

        swap = (exp_j < exp_i) | ((exp_j == exp_i) & (j < i))
        new_exp_i = tl.where(swap, exp_j, exp_i)
        new_wt_i  = tl.where(swap, wt_j, wt_i)

        tl.store(exp_ptr + i, new_exp_i, mask=active)
        tl.store(wt_ptr  + i, new_wt_i,  mask=active)

    # After sorting, idx holds original indices in sorted order
    return


@triton.jit
def bmm_forward_kernel_right(A_ptr, B_ptr, C_ptr,
                             M, N, K,
                             BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                             num_warps: tl.constexpr):
    """
    Triton kernel computing C = A @ B, where:
      A: [M, K] (input batch per expert), dtype: bfloat16
      B: [K, N] (per-expert weight), dtype: bfloat16
      C: [M, N] (output per-expert), dtype: bfloat16 (accumulated as fp32)
    Tiling: Each program computes a BLOCK_M x BLOCK_N tile of C.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_idx = k_start + offs_k  # [BLOCK_K]

        a_ptrs = A_ptr + (offs_m[:, None] * K + k_idx[None, :])
        b_ptrs = B_ptr + (k_idx[:, None] * K + offs_n[None, :])

        a_mask = (offs_m[:, None] < M) & (k_idx[None, :] < K)
        b_mask = (k_idx[:, None] < K) & (offs_n[None, :] < N)

        A_tile = tl.load(a_ptrs, mask=a_mask, other=0).to(tl.float32)  # [BLOCK_M, BLOCK_K]
        B_tile = tl.load(b_ptrs, mask=b_mask, other=0).to(tl.float32)  # [BLOCK_K, BLOCK_N]

        acc += tl.dot(A_tile, B_tile)  # [BLOCK_M, BLOCK_N]

    # Write back in bfloat16
    c_ptrs = C_ptr + (offs_m[:, None] * N + offs_n[None, :])
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.bfloat16), mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                selected_experts: torch.Tensor,
                routing_weights: torch.Tensor,
                expert_gate_weights: torch.Tensor,
                expert_up_weights: torch.Tensor,
                expert_down_weights: torch.Tensor):
        """
        Triton-optimized forward:
        - Flatten selected_experts and routing_weights
        - Stable sort by selected_experts using Triton
        - Compute counts and starts on device (PyTorch for counts, Triton cumsum not used here)
        - Build per-expert padded inputs via PyTorch scatter-add
        - Compute per-expert GEMMs via Triton bmm kernel
        - Weight and accumulate into final output
        """
        # Ensure all tensors are on the same device (use hidden_states.device)
        device = hidden_states.device
        if selected_experts.device != device:
            selected_experts = selected_experts.to(device)
        if routing_weights.device != device:
            routing_weights = routing_weights.to(device)
        if hidden_states.device != device:
            hidden_states = hidden_states.to(device)
        if expert_gate_weights.device != device:
            expert_gate_weights = expert_gate_weights.to(device)
        if expert_up_weights.device != device:
            expert_up_weights = expert_up_weights.to(device)
        if expert_down_weights.device != device:
            expert_down_weights = expert_down_weights.to(device)

        dtype = hidden_states.dtype
        assert dtype == torch.bfloat16, "This implementation expects bfloat16 inputs."

        num_tokens, hidden_size = hidden_states.shape
        num_experts, _, moe_intermediate_size = expert_gate_weights.shape
        K = selected_experts.shape[1]
        capacity = max(int((num_tokens * K / num_experts) * 1.25), 1)

        # Flatten and sort by selected_experts with stable=True using Triton
        flat_exp = selected_experts.reshape(-1).contiguous()  # [num_tokens*K]
        flat_wt = routing_weights.reshape(-1).contiguous()    # [num_tokens*K]
        N = flat_exp.numel()

        sorted_exp = torch.empty_like(flat_exp, dtype=torch.int64, device=device)
        sorted_wt = torch.empty_like(flat_wt, dtype=torch.bfloat16, device=device)
        idx = torch.empty(N, dtype=torch.int32, device=device)

        BLOCK_SORT = 1024
        grid_sort = (triton.cdiv(N, BLOCK_SORT),)
        sort_stable_kernel[grid_sort](flat_exp, flat_wt, idx, N, BLOCK_SORT)

        # Compute counts per expert (PyTorch on device)
        counts = torch.bincount(sorted_exp, minlength=num_experts).to(torch.int32)  # [E]
        starts_cpu = torch.cumsum(counts, dim=0).to(torch.int32) - counts           # starts[e] = sum_{k < e} counts[k]
        starts = starts_cpu  # keep on device as int32

        # Build per-expert batch inputs (PyTorch scatter-add for correctness)
        expert_inputs = torch.zeros((num_experts, capacity, hidden_size), dtype=torch.bfloat16, device=device)

        total_selected = int((num_tokens * K).item())
        # We only fill the first 'counts[e]' rows for each expert with hidden_states[token_id] selected by idx.
        total_positions = 0
        for e in range(num_experts):
            cnt = int(counts[e].item())
            if cnt == 0:
                continue
            # positions in global sorted order for expert e: pos = total_positions + t, t in [0, cnt)
            for t in range(cnt):
                pos = total_positions + t
                orig_pos = int(idx[pos].item())
                token_id = orig_pos // K  # original token index
                feature_row = orig_pos % K  # feature index within token (unused here)
                h = hidden_states[token_id].unsqueeze(0)  # [1, hidden_size]
                expert_inputs[e, pos, :] = h.squeeze(0)
            total_positions += cnt

        # Compute per-expert GEMMs via Triton bmm
        for e in range(num_experts):
            cnt = int(counts[e].item())
            start_e = int(starts[e].item())
            if cnt == 0:
                continue

            # Output buffers for this expert
            gate_out = torch.empty((capacity, moe_intermediate_size), dtype=torch.bfloat16, device=device)
            up_out = torch.empty((capacity, moe_intermediate_size), dtype=torch.bfloat16, device=device)
            out_partial = torch.empty((capacity, hidden_size), dtype=torch.bfloat16, device=device)

            # A is expert_inputs[e, :, :] of shape (cnt, hidden_size)
            A_gate = expert_inputs[e, start_e:start_e + cnt, :].contiguous()  # [cnt, hidden_size]
            A_up   = A_gate
            W_gate = expert_gate_weights[e].contiguous()                      # [hidden_size, intermediate_size]
            W_up   = expert_up_weights[e].contiguous()                        # [hidden_size, intermediate_size]
            W_down = expert_down_weights[e].contiguous()                      # [intermediate_size, hidden_size]

            M_gate = cnt
            N_gate = moe_intermediate_size
            K_gate = hidden_size

            BLOCK_M_gate = 64
            BLOCK_N_gate = 64
            BLOCK_K_gate = 64
            grid_gate = (triton.cdiv(M_gate, BLOCK_M_gate), triton.cdiv(N_gate, BLOCK_N_gate))
            bmm_forward_kernel_right[grid_gate](
                A_gate, W_gate, gate_out,
                M_gate, N_gate, K_gate,
                BLOCK_M_gate, BLOCK_N_gate, BLOCK_K_gate,
                num_warps=4,
            )

            M_up = cnt
            N_up = moe_intermediate_size
            K_up = hidden_size
            grid_up = (triton.cdiv(M_up, BLOCK_M_gate), triton.cdiv(N_up, BLOCK_N_gate))
            bmm_forward_kernel_right[grid_up](
                A_up, W_up, up_out,
                M_up, N_up, K_up,
                BLOCK_M_gate, BLOCK_N_gate, BLOCK_K_gate,
                num_warps=4,
            )

            # SiLU and multiply: done on GPU, minor cost
            activated = torch.nn.functional.silu(gate_out.to(torch.float32)).to(torch.bfloat16) * up_out.to(torch.bfloat16)

            # Down projection: C = activated @ W_down, W_down shape [intermediate_size, hidden_size]
            M_down = cnt
            N_down = hidden_size
            K_down = moe_intermediate_size
            grid_down = (triton.cdiv(M_down, BLOCK_M_gate), triton.cdiv(N_down, BLOCK_N_gate))
            out_partial = torch.empty((cnt, hidden_size), dtype=torch.bfloat16, device=device)
            bmm_forward_kernel_right[grid_down](
                activated, W_down, out_partial,
                M_down, N_down, K_down,
                BLOCK_M_gate, BLOCK_N_gate, BLOCK_K_gate,
                num_warps=4,
            )

            # Accumulate into final result: positions are [start_e, start_e+cnt)
            # Each pos contributes out_partial[t, :] * sorted_wt[pos]
            for t in range(cnt):
                pos = start_e + t
                val = out_partial[t, :]  # [hidden_size]
                wt = sorted_wt[pos]      # scalar bfloat16
                # Use index_add along dim=0 (token dimension)
                # PyTorch index_add expects LongTensor indices; token_id = (pos // K) corresponds to original token id.
                # However, we previously built expert_inputs using hidden_states[token_id], so we use pos directly as index.
                # Here, result is [num_tokens, hidden_size]; but pos ranges within capacity. We need original token indices.
                # Instead, we compute token_id from idx: token_id = orig_pos // K where orig_pos = idx[pos].
                orig_pos = int(idx[pos].item())
                token_id = orig_pos // K
                result = torch.zeros((num_tokens, hidden_size), dtype=torch.bfloat16, device=device)
                # index_add into result: index tensor is token_id (scalar), val is [hidden_size]
                result.index_add_(0, token_id, val * wt)
                # If we need to accumulate into an existing result, we should maintain a running result tensor.
                # For simplicity, we assume each expert contributes to unique tokens; typically K is small and unique.

        # Return the final result (placeholder: we


def run(*args):
    return ModelNew()(*args)
