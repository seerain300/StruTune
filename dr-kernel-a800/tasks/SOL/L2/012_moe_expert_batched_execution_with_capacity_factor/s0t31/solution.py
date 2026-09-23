import math
import torch
import triton
import triton.language as tl


@triton.jit
def sort_stable_kernel(exp_ptr, wt_ptr,
                        N,
                        BLOCK: tl.constexpr):
    """
    Stable sort by selected_experts using odd-even transposition sort per element.
    exp_ptr: int64 array [N], selected_experts flattened
    wt_ptr:  bfloat16/float32 array [N], routing_weights flattened
    """
    pid = tl.program_id(0)
    start = pid * BLOCK
    idx = start + tl.arange(0, BLOCK)
    in_bounds = idx < N

    # Precompute partner indices j = idx + 1
    j = idx + 1
    j_in_bounds = j < N

    # Even phase: pairs (0,1), (2,3), ...
    # Odd phase:  pairs (1,2), (3,4), ...
    for t in range(0, N):
        if (t % 2 == 0):
            is_even_pair = ((idx % 2) == 0) & in_bounds
            # partner is j
            exp_i = tl.load(exp_ptr + idx, mask=in_bounds, other=0)
            wt_i  = tl.load(wt_ptr  + idx, mask=in_bounds, other=0.0)
            exp_j = tl.load(exp_ptr + j,   mask=j_in_bounds, other=0)
            wt_j  = tl.load(wt_ptr  + j,   mask=j_in_bounds, other=0.0)

            # Swap when out-of-order and within-pair bounds
            swap = (exp_i > exp_j) & is_even_pair
            # Swap both arrays
            new_exp_i = tl.where(swap, exp_j, exp_i)
            new_exp_j = tl.where(swap, exp_i, exp_j)
            new_wt_i  = tl.where(swap, wt_j,  wt_i)
            new_wt_j  = tl.where(swap, wt_i,  wt_j)

            # Store back
            tl.store(exp_ptr + idx, new_exp_i, mask=in_bounds)
            tl.store(wt_ptr  + idx, new_wt_i,  mask=in_bounds)
            tl.store(exp_ptr + j,  new_exp_j, mask=j_in_bounds)
            tl.store(wt_ptr  + j,  new_wt_j,  mask=j_in_bounds)

        else:
            is_odd_pair  = (((idx + 1) % 2) == 0) & in_bounds
            # partner is j-1 (since odd pairs are (1,2), (3,4), ...)
            j_m1 = j - 1
            j_m1_in_bounds = (j > 0) & (j > 1) & in_bounds  # redundant, but keep for clarity

            exp_i = tl.load(exp_ptr + idx, mask=in_bounds, other=0)
            wt_i  = tl.load(wt_ptr  + idx, mask=in_bounds, other=0.0)
            exp_jm1 = tl.load(exp_ptr + j_m1, mask=j_m1_in_bounds, other=0)
            wt_jm1  = tl.load(wt_ptr  + j_m1, mask=j_m1_in_bounds, other=0.0)

            # Swap when out-of-order and within-pair bounds
            swap = (exp_i > exp_jm1) & is_odd_pair
            new_exp_i = tl.where(swap, exp_jm1, exp_i)
            new_exp_jm1 = tl.where(swap, exp_i, exp_jm1)
            new_wt_i  = tl.where(swap, wt_jm1, wt_i)
            new_wt_jm1  = tl.where(swap, wt_i, wt_jm1)

            # Store back
            tl.store(exp_ptr + idx, new_exp_i, mask=in_bounds)
            tl.store(wt_ptr  + idx, new_wt_i,  mask=in_bounds)
            tl.store(exp_ptr + j_m1, new_exp_jm1, mask=j_m1_in_bounds)
            tl.store(wt_ptr  + j_m1, new_wt_jm1, mask=j_m1_in_bounds)


@triton.jit
def bincount_kernel(exp_ptr, counts_ptr,
                    N, E,
                    BLOCK: tl.constexpr):
    """
    Triton bincount: counts per expert id in exp_ptr [N], int64 values.
    Write counts into counts_ptr [E], int64.
    """
    pid = tl.program_id(0)
    start = pid * BLOCK
    idx = start + tl.arange(0, BLOCK)
    in_bounds = idx < N

    # Load values
    vals = tl.load(exp_ptr + idx, mask=in_bounds, other=0)
    # Build per-value counts: one pass over idx
    # We'll count occurrences of each value using atomic adds.
    # For each idx, if in_bounds, atomic add 1 to counts[vals[idx]]
    # Note: vals may be out of bounds for idx >= N; mask ensures no contribution.
    for i in range(0, BLOCK):
        v = vals[i]
        if in_bounds[i]:
            # Atomic add 1 to counts[v]
            tl.atomic_add(counts_ptr + v, 1)


@triton.jit
def cumsum_kernel(counts_ptr, starts_ptr,
                  E,
                  BLOCK: tl.constexpr):
    """
    Triton cumsum of counts_ptr [E] -> starts_ptr [E].
    starts[i] = sum_{k < i} counts[k], int64.
    """
    # Single program computes prefix sums sequentially for simplicity.
    # This is acceptable for moderate E.
    pid = tl.program_id(0)
    if pid != 0:
        return
    running = tl.zeros((), dtype=tl.int64)
    for i in range(0, E):
        c = tl.load(counts_ptr + i)
        running += c
        tl.store(starts_ptr + i, running)


@triton.jit
def bmm_forward_kernel_right(A_ptr, B_ptr, C_ptr,
                             M, N, K,
                             BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    """
    Triton kernel computing C = A @ B, where:
      A: [M, K], B: [K, N], C: [M, N]
    Performs batched matmul for M rows, N columns, looping over K in tiles.
    Accumulates in fp32 and stores as bfloat16.
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

        A_tile = tl.load(a_ptrs, mask=a_mask, other=0.0)  # [BLOCK_M, BLOCK_K]
        B_tile = tl.load(b_ptrs, mask=b_mask, other=0.0)  # [BLOCK_K, BLOCK_N]

        # Cast to fp32 for accumulation
        A_tile = A_tile.to(tl.float32)
        B_tile = B_tile.to(tl.float32)

        # acc += A_tile @ B_tile
        acc += tl.dot(A_tile, B_tile)

    # Store results as bfloat16
    c_ptrs = C_ptr + (offs_m[:, None] * N + offs_n[None, :])
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.bfloat16), mask=c_mask)


def _stable_sort_triton(exp_ptr, wt_ptr, N, device, BLOCK=1024):
    # Launch stable sort kernel
    grid = (triton.cdiv(N, BLOCK),)
    sort_stable_kernel[grid](exp_ptr, wt_ptr, N, BLOCK=BLOCK)
    # Return sorted arrays
    # exp_ptr and wt_ptr are modified in-place by the kernel
    sorted_exp = exp_ptr
    sorted_wt = wt_ptr
    return sorted_exp, sorted_wt


def _bincount_triton(exp_ptr, N, E, device):
    counts = torch.zeros(E, dtype=torch.int64, device=device)
    grid = (triton.cdiv(N, 1024),)
    bincount_kernel[grid](exp_ptr, counts, N, E, BLOCK=1024)
    return counts


def _cumsum_triton(counts_ptr, E, device):
    starts = torch.empty(E, dtype=torch.int64, device=device)
    grid = (1,)
    cumsum_kernel[grid](counts_ptr, starts, E, BLOCK=1024)
    return starts


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states, selected_experts, routing_weights,
                expert_gate_weights, expert_up_weights, expert_down_weights):
        """
        hidden_states: [num_tokens, hidden_size], bfloat16
        selected_experts: [num_tokens, K], int64
        routing_weights: [num_tokens, K], bfloat16
        expert_*_weights: [num_experts, hidden_size, intermediate_size] and [num_experts, intermediate_size, hidden_size], bfloat16
        """
        num_tokens = hidden_states.shape[0]
        hidden_size = hidden_states.shape[1]
        num_experts, _, moe_intermediate_size = expert_gate_weights.shape
        K = selected_experts.shape[1]

        device = hidden_states.device
        dtype = hidden_states.dtype

        # Flatten and sort by selected_experts (stable=True) using Triton
        selected_exp = selected_experts.reshape(-1).to(torch.int64).contiguous()     # [N]
        wt = routing_weights.reshape(-1).to(torch.bfloat16).contiguous()             # [N]
        N = selected_exp.numel()

        grid_sort = (triton.cdiv(N, 1024),)
        _ = _stable_sort_triton(selected_exp, wt, N, device, BLOCK=1024)             # modify in-place

        # Triton bincount per expert
        per_exp_counts = _bincount_triton(selected_exp, N, num_experts, device)      # [E], int64
        starts = _cumsum_triton(per_exp_counts, num_experts, device)                 # [E], int64

        # capacity heuristic
        total_selected = int(per_exp_counts.sum().item())
        capacity = max(int((num_tokens * K / num_experts) * 1.25), 1)
        # Ensure capacity covers total_selected (it should, per problem setup)
        if capacity < total_selected:
            capacity = total_selected

        # Build per-expert batch inputs using PyTorch scatter-add: expert_inputs [E, capacity, hidden_size]
        # For each expert e, fill the first per_exp_counts[e] entries by copying hidden_states rows
        # corresponding to tokens that selected expert e, in position within the sorted group.
        expert_inputs = torch.zeros((num_experts, capacity, hidden_size), dtype=torch.bfloat16, device=device)

        # Compute token_id per flattened index pos: token_id = pos // K
        pos = torch.arange(N, device=device)
        token_id = (pos // K).to(torch.int64)  # [N]

        # For each expert e, fill positions [starts[e] : starts[e] + per_exp_counts[e]]
        for e in range(num_experts):
            cnt = int(per_exp_counts[e].item())
            start = int(starts[e].item())
            if cnt > 0:
                # Build mapping of token_id to hidden_states row in PyTorch
                # We'll scatter-add rows into expert_inputs[e, 0:cnt, :]
                # For each token t, its K selections are at positions where token_id == t and selected_exp == e.
                # Since we already have sorted_exp and sorted_wt, we can reconstruct token_id per index.
                # However, reconstructing token_id from sorted arrays isn't necessary for scatter; we can directly use
                # the positions as token_id.
                # We need to map each position p to a hidden_states row. Using token_id = floor(p / K) is the correct mapping.
                # So: for p in [start, start+cnt), copy hidden_states[token_id[p]] into expert_inputs[e, p - start, :]
                # Implement via PyTorch scatter-add:
                p = torch.arange(0, cnt, device=device)
                indices = p + start  # flattened positions
                token_ids = token_id[indices]  # [cnt]
                # Validate uniqueness (not guaranteed, but for our setup it is by construction).
                # Copy rows
                rows = hidden_states[token_ids]  # [cnt, hidden_size]
                expert_inputs[e, p, :] = rows

        # Now compute per-expert gate_out, up_out, activated, expert_outputs using Triton bmm
        # gate_out = expert_inputs @ expert_gate_weights -> [capacity, intermediate_size]
        # up_out   = expert_inputs @ expert_up_weights   -> [capacity, intermediate_size]
        # activated = SiLU(gate_out) * up_out
        # expert_outputs = activated @ expert_down_weights -> [capacity, hidden_size]

        # Prepare B matrices contiguous
        B_gate = expert_gate_weights.contiguous()      # [E, H, I]
        B_up   = expert_up_weights.contiguous()        # [E, H, I]
        B_down = expert_down_weights.contiguous()      # [E, I, H]

        # We need to compute these for each expert e. We will launch a grid over (M=capacity, N=hidden_size) tiles.
        # But Triton kernel expects fixed M,N,K. We can compute per-expert outputs in separate launches.
        outputs_all = torch.empty((num_experts, capacity, hidden_size), dtype=torch.bfloat16, device=device)

        # Define grid for bmm
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 64
        grid_bmm = lambda meta: (triton.cdiv(capacity, meta['BLOCK_M']), triton.cdiv(hidden_size, meta['BLOCK_N']))

        # Launch per-expert bmm
        for e in range(num_experts):
            A = expert_inputs[e]                        # [capacity, hidden_size]
            # Compute gate_out
            C_gate = torch.empty((capacity, moe_intermediate_size), dtype=torch.bfloat16, device=device)
            bmm_forward_kernel_right[grid_bmm](A, B_gate[e], C_gate, capacity, moe_intermediate_size, hidden_size,
                                               BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                                               num_warps=4)
            # Compute up_out
            C_up = torch.empty((capacity, moe_intermediate_size), dtype=torch.bfloat16, device=device)
            bmm_forward_kernel_right[grid_bmm](A, B_up[e], C_up, capacity, moe_intermediate_size, hidden_size,
                                               BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                                               num_warps=4)
            # SiLU: x * sigmoid(x)
            # Triton kernel for SiLU: elementwise on C_gate
            gate_out = C_gate
            up_out = C_up
            # activated = SiLU(gate_out) * up_out
            # Implement SiLU via Triton: y = gate_out * sigmoid(gate_out)
            # We can do this elementwise on gate_out:
            # Note: Triton doesn't have sigmoid; implement as 1 / (1 + exp(-x))
            # Convert to fp32 for stability
            gate_out_fp32 = gate_out.to(torch.float32)
            silu = gate_out_fp32 * (1.0 / (1.0 + torch.exp(-gate_out_fp32)))
            activated = (silu * up_out.to(torch.float32)).to(torch.bfloat16)  # [capacity, I]

            # expert_outputs = activated @ B_down[e] -> [capacity, hidden_size]
            C_expert = torch.empty((capacity, hidden_size), dtype=torch.bfloat16, device=device)
            grid_down = lambda meta: (triton.cdiv(capacity, meta['BLOCK_M']), triton.cdiv(hidden_size, meta['BLOCK_N']))
            bmm_forward_kernel_right[grid_down](activated, B_down[e], C_expert, capacity, hidden_size, moe_intermediate_size,
                                                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                                                num_warps=4)

            # Store into outputs_all
            outputs_all[e] = C_expert

        # Gather valid outputs and accumulate into final result
        # We need to map positions back to token indices. After sorting and capacity masking, the valid
        # positions are within_pos < capacity for each expert. We have:
        # sorted_exp[i] = selected_exp_id, sorted_wt[i] = routing_weight
        # For each expert e, positions [starts[e] : starts[e] + per_exp_counts[e]) correspond to valid tokens.
        # But since we pre-filled expert_inputs with these positions, we can directly use outputs_all
        # and multiply by routing_weights at those positions.

        # Build final output: result[num_tokens, hidden_size]
        result = torch.zeros((num_tokens, hidden_size), dtype=torch.bfloat16, device=device)

        # We need to scatter-add outputs to result using token_id mapping. For each position i after sorting:
        # if within_pos < capacity, then add outputs_all[e, within_pos] * sorted_wt[i] to result[token_id[i], :]
        # Implement in PyTorch:
        for e in range(num_experts):
            cnt = int(per_exp_counts[e].item())
            start = int(starts[e].item())
            if cnt > 0:
                p = torch.arange(0, cnt, device=device)
                indices = p + start  # flattened positions
                token_ids = token_id[indices]  # [cnt]
                selected_ids = selected_exp[indices]  # [cnt], should equal e (true by construction of positions)
                # Compute within_pos
                within_pos = p  # since we filled first cnt positions
                # Valid mask
                valid = within_pos < capacity
                # Gathers outputs_all[e, within_pos] and weights
                # Convert indices to int32 for scatter_add
                token_ids_i32 = token_ids.to(torch.int32)
                outputs_e = outputs_all[e]  # [capacity, hidden_size]
                # Slice valid rows
                outputs_valid = outputs_e[valid]  # [cnt, hidden_size]
                weights = wt[indices]  # [cnt], bfloat16
                # Scale by weights
                scaled = outputs_valid * weights[:, None]  # [cnt, hidden_size]
                # Accumulate into result
                result.index_add_(0, token_ids_i32, scaled.to(torch.bfloat16))

        return result


def run(*args):
    return ModelNew()(*args)
