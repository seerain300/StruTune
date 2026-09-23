import math
import torch
import triton
import triton.language as tl


@triton.jit
def sort_stable_kernel(exp_ptr, wt_ptr, idx_ptr, N, BLOCK: tl.constexpr):
    """
    Stable sort of (selected_experts, routing_weights) into (idx_ptr, sorted_exp, sorted_wt).
    Each program handles a BLOCK-sized slice of the flattened array of length N.
    We implement odd-even transposition sort using partner index j = idx ^ 1.
    """
    pid = tl.program_id(0)
    start = pid * BLOCK
    idx = start + tl.arange(0, BLOCK)
    in_bounds = idx < N

    # Track permutations via idx (original positions). Initialize to identity.
    init_mask = in_bounds & (idx_ptr + idx == 0)
    tl.store(idx_ptr + idx, idx, mask=init_mask)

    # Number of phases: N passes are sufficient to sort.
    for t in range(0, N):
        # Even phase: compare (0,1), (2,3), ...
        phase = 0
        sel = in_bounds & ((idx & 1) == phase)
        j = idx ^ 1  # partner index
        # Load current and partner values and permutations
        exp_i = tl.load(exp_ptr + idx, mask=sel, other=0)
        wt_i = tl.load(wt_ptr + idx, mask=sel, other=0.0)
        idx_i = tl.load(idx_ptr + idx, mask=sel, other=0)

        exp_j = tl.load(exp_ptr + j, mask=sel, other=0)
        wt_j = tl.load(wt_ptr + j, mask=sel, other=0.0)
        idx_j = tl.load(idx_ptr + j, mask=sel, other=0)

        # If out-of-order, swap both values and permutations
        cond_swap = exp_i > exp_j  # ascending order by expert
        new_exp_i = tl.where(cond_swap, exp_j, exp_i)
        new_exp_j = tl.where(cond_swap, exp_i, exp_j)
        new_wt_i = tl.where(cond_swap, wt_j, wt_i)
        new_wt_j = tl.where(cond_swap, wt_i, wt_j)

        new_idx_i = tl.where(cond_swap, idx_j, idx_i)
        new_idx_j = tl.where(cond_swap, idx_i, idx_j)

        # Write back
        tl.store(exp_ptr + idx, new_exp_i, mask=sel)
        tl.store(wt_ptr + idx, new_wt_i, mask=sel)
        tl.store(idx_ptr + idx, new_idx_i, mask=sel)

        tl.store(exp_ptr + j, new_exp_j, mask=sel)
        tl.store(wt_ptr + j, new_wt_j, mask=sel)
        tl.store(idx_ptr + j, new_idx_j, mask=sel)

        # Odd phase: compare (1,2), (3,4), ...
        phase = 1
        sel = in_bounds & ((idx & 1) == phase)
        j = idx ^ 1
        exp_i = tl.load(exp_ptr + idx, mask=sel, other=0)
        wt_i = tl.load(wt_ptr + idx, mask=sel, other=0.0)
        idx_i = tl.load(idx_ptr + idx, mask=sel, other=0)

        exp_j = tl.load(exp_ptr + j, mask=sel, other=0)
        wt_j = tl.load(wt_ptr + j, mask=sel, other=0.0)
        idx_j = tl.load(idx_ptr + j, mask=sel, other=0)

        cond_swap = exp_i > exp_j
        new_exp_i = tl.where(cond_swap, exp_j, exp_i)
        new_exp_j = tl.where(cond_swap, exp_i, exp_j)
        new_wt_i = tl.where(cond_swap, wt_j, wt_i)
        new_wt_j = tl.where(cond_swap, wt_j, wt_i)

        new_idx_i = tl.where(cond_swap, idx_j, idx_i)
        new_idx_j = tl.where(cond_swap, idx_i, idx_j)

        tl.store(exp_ptr + idx, new_exp_i, mask=sel)
        tl.store(wt_ptr + idx, new_wt_i, mask=sel)
        tl.store(idx_ptr + idx, new_idx_i, mask=sel)

        tl.store(exp_ptr + j, new_exp_j, mask=sel)
        tl.store(wt_ptr + j, new_wt_j, mask=sel)
        tl.store(idx_ptr + j, new_idx_j, mask=sel)


@triton.jit
def bincount_kernel(exp_ptr, counts_ptr, N, E, BLOCK: tl.constexpr):
    """
    counts[e] = number of occurrences of expert id e in exp_ptr[0:N], int32.
    """
    pid = tl.program_id(0)
    start = pid * BLOCK
    idx = start + tl.arange(0, BLOCK)
    in_bounds = idx < E

    # Accumulate local counts per expert id
    local_counts = tl.zeros((BLOCK,), dtype=tl.int32)
    for k in range(0, N):
        val = tl.load(exp_ptr + k)
        # val may be int64; cast to int32 for counting
        local_counts += in_bounds & ((val >= 0) & (val < E)) & (val == idx)

    # Reduce local_counts into global counts using atomic add
    for kk in range(0, BLOCK):
        v = local_counts[kk]
        # Only add if v > 0 and in_bounds[kk]
        tl.atomic_add(counts_ptr + idx[kk], v, mask=in_bounds[kk])


@triton.jit
def cumsum_kernel(counts_ptr, starts_ptr, E, BLOCK: tl.constexpr):
    """
    starts[e] = sum of counts[0:e] (exclusive), int64.
    """
    # Single pass: compute prefix sums and store starts.
    acc = tl.zeros((), dtype=tl.int64)
    for e in range(0, E):
        c = tl.load(counts_ptr + e)
        acc += c
        tl.store(starts_ptr + e, acc)


@triton.jit
def bmm_forward_kernel_right(A_ptr, B_ptr, C_ptr,
                             M, N, K,
                             BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                             num_warps: tl.constexpr):
    """
    Compute C = A @ B, where:
      A: [M, K], B: [K, N], C: [M, N]
    Use tiles with fp32 accumulation. Assumes bfloat16 input, output.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        a_ptrs = A_ptr + (offs_m[:, None] * K + offs_k[None, :])
        b_ptrs = B_ptr + (offs_k[:, None] * N + offs_n[None, :])

        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0).to(tl.float32)  # [BLOCK_M, BLOCK_K]
        b = tl.load(b_ptrs, mask=b_mask, other=0.0).to(tl.float32)  # [BLOCK_K, BLOCK_N]

        acc += tl.dot(a, b)

    # Store result (C is bfloat16)
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
        device = hidden_states.device
        dtype = hidden_states.dtype

        # Shapes
        num_tokens, hidden_size = hidden_states.shape
        num_experts, gate_in, moe_intermediate_size = expert_gate_weights.shape
        K = selected_experts.shape[1]
        capacity = max(int((num_tokens * K / num_experts) * 1.25), 1)

        # Flatten selected_experts and routing_weights
        exp_flat = selected_experts.reshape(-1)  # [N]
        wt_flat = routing_weights.reshape(-1)    # [N]
        N = exp_flat.shape[0]

        # 1) Stable sort by selected_experts; compute sorted indices
        sorted_exp = torch.empty(N, dtype=torch.int64, device=device)
        sorted_wt = torch.empty(N, dtype=torch.bfloat16, device=device)
        idx = torch.empty(N, dtype=torch.int64, device=device)

        # Launch stable sort kernel
        BLOCK = 1024
        grid = (triton.cdiv(N, BLOCK),)
        sort_stable_kernel[grid](exp_flat, wt_flat, idx, N, BLOCK=BLOCK, num_warps=4)

        # 2) Bincount counts per expert
        counts = torch.zeros(num_experts, dtype=torch.int32, device=device)
        bincount_kernel[grid](sorted_exp, counts, N, num_experts, BLOCK=BLOCK, num_warps=1)

        # 3) Cumsum to get starts (exclusive prefix sum)
        starts = torch.empty(num_experts, dtype=torch.int64, device=device)
        cumsum_kernel[(num_experts,)](counts, starts, num_experts, BLOCK=1, num_warps=1)

        # 4) Build padded per-expert inputs via PyTorch scatter-add (correctness)
        # We need to map original token positions to global flat positions.
        # After sorting, flat positions are just [0, 1, 2, ...]. We use idx to retrieve original token id for each sorted position.
        # However, we only need to scatter hidden_states[v_tok] into expert_inputs[v_exp, v_pos].
        # valid positions: within_pos < capacity
        # Compute v_exp, v_pos, v_tok, v_wt
        # within_pos = global_sorted_index - starts[sorted_exp]
        # invalid positions are not written; we'll mask when reading later.
        # Note: idx is original token id for each sorted position. We can use it to read hidden_states at original positions for scatter.
        # But scatter-add requires dynamic index, implemented via PyTorch for simplicity and safety.

        # First compute global sorted index (position) per entry using idx
        global_sorted_index = idx  # [N]
        within_pos = global_sorted_index - starts[sorted_exp].to(torch.int64)  # [N], int64
        valid = within_pos < capacity  # [N], bool

        # Filter valid entries
        exp_v = sorted_exp[valid]       # [M'] int64
        pos_v = within_pos[valid]       # [M'] int64
        tok_v = idx[valid]              # [M'] int64, original token id
        wt_v = sorted_wt[valid].to(torch.float32)  # [M'] float32

        # Build per-expert inputs using PyTorch scatter-add
        # Initialize expert_inputs
        expert_inputs = torch.empty((num_experts, capacity, hidden_size), dtype=torch.bfloat16, device=device)
        # Scatter hidden_states[tok_v] into expert_inputs[exp_v, pos_v]
        # We need to ensure capacity fits total selected tokens. If not, we cap to capacity.
        M_prime = exp_v.shape[0]
        if M_prime > 0:
            # Convert to 2D indices
            # Note: M' may be huge in some workloads. Scatter-add is acceptable here.
            exp_2d = exp_v.view(M_prime, 1).expand(M_prime, capacity).contiguous()
            pos_2d = pos_v.view(M_prime, 1).expand(M_prime, capacity).contiguous()
            # Prepare value matrix: each row has hidden_states[tok_v[i]] replicated at pos_v[i]
            # We cannot directly fill, so we do scatter_add row by row
            for i in range(M_prime):
                row_vals = hidden_states[tok_v[i]].unsqueeze(1).expand(1, capacity).clone()
                # Only positions matching pos_v[i] should receive values; but scatter-add will distribute by index.
                # However, pos_v may repeat across different experts. To avoid overwrite, ensure M' <= capacity per expert.
                # Given capacity is defined as 1.25 * average tokens per expert, M' should be <= capacity.
                expert_inputs[exp_v[i], pos_v[i]] += row_vals.squeeze(0)

        # 5) Run Triton batched matmuls per expert (bmm_forward_kernel_right) on expert_inputs
        # expert_gate_weights: [E, H, I], expert_up_weights: [E, H, I], expert_down_weights: [E, I, H]
        # We need to call the kernel for each expert. Launch a grid over (num_experts, 1).
        # For each expert e, compute:
        #   A: expert_inputs[e, :capacity, :]   -> (M_e, H)
        #   gate_out: A @ expert_gate_weights[e] -> (M_e, I)
        #   up_out:   A @ expert_up_weights[e]   -> (M_e, I)
        #   activated: SiLU(gate_out) * up_out  -> (M_e, I)
        #   expert_outputs: activated @ expert_down_weights[e] -> (M_e, H)
        # We don't have 'M_e' exactly; but we can compute per-expert outputs by masking valid positions.

        # Precompute grid size for bmm
        # However, since we created expert_inputs with arbitrary capacity across experts based on global sorting, we need to compute M_e per expert.
        # To avoid complexity, we compute outputs only for those valid positions (M') and accumulate at the end.
        # Create output buffers for each expert: but Triton can't index by variable M per expert easily; instead, we'll compute in chunks.

        # Instead of per-expert launch, we aggregate across all experts and valid positions into a single tensor:
        # Initialize result
        result = torch.zeros((num_tokens, hidden_size), dtype=dtype, device=device)

        # Since we cannot separate per-expert here, we aggregate using valid positions. But we need to weight by routing_weights.
        # We already have exp_v, pos_v, tok_v, wt_v. We need expert_outputs for each valid pair. But building expert_outputs per valid pair is costly and requires per-expert weights.
        # To simplify and ensure correctness, we will compute per-expert outputs by enumerating all possible token-expert pairs, but that is O(N*K) which is fine for the benchmarks.

        # 6) Compute per-expert outputs by enumerating all pairs (i, e) with i in valid set and selected_exp[i] == e, and capacity mask
        # This avoids complexity in Triton grid and preserves correctness.
        # For each expert e, find all i where exp_v[i] == e and pos_v[i] < capacity; then compute and scatter-add into result[tok_v[i]].

        # Prepare outputs per valid pair (gate_out, up_out, activated, final output). We can compute small batches.

        # We'll compute for each expert e in range(num_experts):
        # Find indices in exp_v where expert == e
        for e in range(num_experts):
            mask_exp = (exp_v == e)
            if mask_exp.any():
                local_valid = mask_exp & (pos_v < capacity)
                if local_valid.any():
                    i = torch.nonzero(local_valid, as_tuple=False).flatten()  # indices in exp_v array
                    # Retrieve original token id and weight
                    tok_local = tok_v[i]  # [L]
                    wt_local = wt_v[i]    # [L], float32
                    # Select hidden state rows
                    hs_local = hidden_states[tok_local]  # [L, H]
                    # Build A for this expert (only rows corresponding to valid positions in exp_v with expert==e)
                    # expert_inputs[e, pos_v[i]] already contains values; but we need A from hidden_states.
                    # We need to find which rows of hs_local match tok_v[i] relative to original hidden_states order.
                    # However, tok_v[i] refers to original token id; we can retrieve rows directly.
                    # For each i, A_row = hidden_states[tok_v[i]], B = corresponding weight matrices.

                    # We don't have A directly from hs_local because we filled expert_inputs via scatter; A must be constructed from hidden_states rows.
                    # To avoid complexity, we approximate by using hs_local as A. For exactness, we will compute A from original hidden_states using tok_v.
                    # But tok_v maps to original token id in [0..num_tokens-1]; we can gather those rows.

                    # Build A: shape (L, H)
                    # For each i, A[i] = hidden_states[tok_local[i]]
                    # However, gathering per row is expensive; instead, we will compute by recomputing with original indices.
                    # Since we already have hs_local, we can use it as A for the Triton kernel by materializing into a contiguous tensor and launching bmm_forward_kernel_right for (A, weight_e) -> output (L, I).

                    # We need per-expert weights: gather slices
                    W_gate_e = expert_gate_weights[e]  # [H, I]
                    W_up_e = expert_up_weights[e]      # [H, I]
                    W_down_e = expert_down_weights[e]  # [I, H]

                    # For simplicity, materialize A as bfloat16, then run bmm on it
                    A = hs_local.to(torch.bfloat16)  # [L, H]

                    # We need M, N, K for kernel: M=L, N=I, K=H
                    L = A.shape[0]
                    I = W_gate_e.shape[1]
                    H = hidden_size

                    # Allocate outputs
                    gate_out = torch.empty((L, I), dtype=torch.bfloat16, device=device)
                    up_out = torch.empty((L, I), dtype=torch.bfloat16, device=device)
                    activated = torch.empty((L, I), dtype=torch.bfloat16, device=device)
                    expert_outputs = torch.empty((L, H), dtype=torch.bfloat16, device=device)

                    # Launch Triton bmm for gate_out
                    grid_bmm = (triton.cdiv(L, 64), triton.cdiv(I, 64))
                    bmm_forward_kernel_right[grid_bmm](A, W_gate_e.to(torch.bfloat16), gate_out,
                                                       L, I, H, BLOCK_M=64, BLOCK_N=64, BLOCK_K=64, num_warps=4)

                    # Launch Triton bmm for up_out
                    bmm_forward_kernel_right[grid_bmm](A, W_up_e.to(torch.bfloat16), up_out,
                                                       L, I, H, BLOCK_M=64, BLOCK_N=64, BLOCK_K=64, num_warps=4)

                    # SiLU and multiply
                    # Implement SiLU in Triton is not straightforward in this snippet; use torch for activation
                    gate_silu = torch.nn.functional.silu(gate_out.to(torch.float32)).to(torch.bfloat16)
                    activated = gate_silu * up_out

                    # Third bmm: activated @ W_down_e -> (L, H)
                    B_down = W_down_e.to(torch.bfloat16)  # [I, H]
                    grid_bmm3 = (triton.cdiv(L, 64), triton.cdiv(H, 64))
                    bmm_forward_kernel_right[grid_bmm3](activated, B_down, expert_outputs,
                                                        L, H, I, BLOCK_M=64, BLOCK_N=64, BLOCK_K=64, num_warps=4)

                    # Accumulate into result: for each i, result[tok_local[i]] += wt_local[i] * expert_outputs[i]
                    # result uses original token ids, gathered via tok_local.
                    # Weights are float32; expert_outputs is bfloat16; convert to float32 for accumulation.
                    contrib = expert_outputs.to(torch.float32) * wt_local.view(-1, 1)  # [L, H]
                    result.index_add_(0, tok_local, contrib)

        return result


def run(*args):
    return ModelNew()(*args)
