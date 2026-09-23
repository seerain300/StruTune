import math
import torch
import triton
import triton.language as tl


@triton.jit
def sort_stable_kernel(exp_ptr, wt_ptr, N, BLOCK: tl.constexpr):
    """
    Stable sort of arrays exp_ptr (int64) and wt_ptr (same length) of length N.
    Uses odd-even transposition sort: swap adjacent pairs per phase.
    Grid: 1D with size = ceil_div(N, BLOCK).
    """
    pid = tl.program_id(0)
    start = pid * BLOCK
    idx = start + tl.arange(0, BLOCK)
    in_bounds = idx < N

    # Loop for N phases (sufficient for stable sort)
    for t in range(0, N):
        # Even phase: pairs (0,1), (2,3), ...
        is_even_phase = (t % 2 == 0)
        i = idx
        j = i + 1
        j_in_bounds = j < N

        even_mask = in_bounds & j_in_bounds & (i % 2 == 0)

        exp_i = tl.load(exp_ptr + i, mask=in_bounds, other=0)
        exp_j = tl.load(exp_ptr + j, mask=j_in_bounds, other=0)
        swap_mask = (exp_i > exp_j) & even_mask

        new_i = tl.where(swap_mask, exp_j, exp_i)
        new_j = tl.where(swap_mask, exp_i, exp_j)

        tl.store(exp_ptr + i, new_i, mask=in_bounds)
        tl.store(exp_ptr + j, new_j, mask=j_in_bounds)

        # Swap corresponding weights under the same mask
        wt_i = tl.load(wt_ptr + i, mask=in_bounds, other=0)
        wt_j = tl.load(wt_ptr + j, mask=j_in_bounds, other=0)
        swap_mask_wt = swap_mask

        new_wt_i = tl.where(swap_mask_wt, wt_j, wt_i)
        new_wt_j = tl.where(swap_mask_wt, wt_i, wt_j)

        tl.store(wt_ptr + i, new_wt_i, mask=in_bounds)
        tl.store(wt_ptr + j, new_wt_j, mask=j_in_bounds)

        # Odd phase: pairs (1,2), (3,4), ...
        is_odd_phase = (t % 2 == 1)
        if is_odd_phase:
            i = idx
            j = i + 1
            j_in_bounds = j < N
            odd_mask = in_bounds & j_in_bounds & (((i + 1) % 2) == 0)

            exp_i = tl.load(exp_ptr + i, mask=in_bounds, other=0)
            exp_j = tl.load(exp_ptr + j, mask=j_in_bounds, other=0)
            swap_mask = (exp_i > exp_j) & odd_mask

            new_i = tl.where(swap_mask, exp_j, exp_i)
            new_j = tl.where(swap_mask, exp_i, exp_j)

            tl.store(exp_ptr + i, new_i, mask=in_bounds)
            tl.store(exp_ptr + j, new_j, mask=j_in_bounds)

            wt_i = tl.load(wt_ptr + i, mask=in_bounds, other=0)
            wt_j = tl.load(wt_ptr + j, mask=j_in_bounds, other=0)
            swap_mask_wt = swap_mask

            new_wt_i = tl.where(swap_mask_wt, wt_j, wt_i)
            new_wt_j = tl.where(swap_mask_wt, wt_i, wt_j)

            tl.store(wt_ptr + i, new_wt_i, mask=in_bounds)
            tl.store(wt_ptr + j, new_wt_j, mask=j_in_bounds)


@triton.jit
def bincount_kernel(vals_ptr, out_ptr, N, num_buckets: tl.constexpr):
    """
    Triton kernel for bincount: counts occurrences of each integer in [0, num_buckets)
    in vals_ptr (int64), writes counts to out_ptr (int32).
    Each program handles a block of indices and atomically adds to out.
    """
    pid = tl.program_id(0)
    start = pid * 1024
    idx = start + tl.arange(0, 1024)
    in_bounds = idx < N

    counts = tl.zeros((1024,), dtype=tl.int32)
    # Load values and count
    vals = tl.load(vals_ptr + idx, mask=in_bounds, other=0)  # int64
    # Compute indices into counts
    # We assume vals are within [0, num_buckets). If not, mask invalid with in_bounds.
    idxs = vals  # type inference: int64
    # Atomic add per index
    for k in range(0, 1024):
        if in_bounds[k]:
            # Atomic add to out_ptr[idxs[k]]
            # Triton supports atomic_add on int32. Cast idxs[k] to int32 for pointer offset.
            # Note: idxs[k] may be int64; Triton pointer arithmetic expects int32 offsets for indexing.
            # We assume num_buckets <= 2^31-1; cast is safe for typical sizes.
            cnt = tl.atomic_add(out_ptr + idxs[k].to(tl.int32), 1)
            # We only need to add once; cnt is returned value but ignored.


@triton.jit
def cumsum_kernel(in_ptr, out_ptr, L: tl.constexpr):
    """
    Triton kernel to compute prefix sum (cumsum) of int64 vector in_ptr of length L.
    out_ptr receives int64 cumsum results.
    Uses sequential scan within a single program for simplicity (L is num_experts, small).
    """
    # Single program computes cumsum
    running = 0
    for i in range(0, L):
        val = tl.load(in_ptr + i)  # int64
        running += val
        tl.store(out_ptr + i, running)


@triton.jit
def bmm_forward_kernel_right(A_ptr, B_ptr, C_ptr,
                             M, N, K,
                             BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    """
    Compute C = A @ B, where:
      A: [M, K], B: [K, N], C: [M, N]
    Tiling with fp32 accumulation. Each program computes a BLOCK_M x BLOCK_N tile of C.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K in tiles
    for k_start in range(0, K, BLOCK_K):
        k_idx = k_start + offs_k  # [BLOCK_K]

        # Pointers for current tiles
        a_ptrs = A_ptr + (offs_m[:, None] * K + k_idx[None, :])  # (BLOCK_M, BLOCK_K)
        b_ptrs = B_ptr + (k_idx[:, None] * N + offs_n[None, :])  # (BLOCK_K, BLOCK_N)

        # Bounds masks
        a_mask = (offs_m[:, None] < M) & (k_idx[None, :] < K)
        b_mask = (k_idx[:, None] < K) & (offs_n[None, :] < N)

        # Load tiles (promote to fp32 for accumulation)
        A_tile = tl.load(a_ptrs, mask=a_mask, other=0.0)
        B_tile = tl.load(b_ptrs, mask=b_mask, other=0.0)

        A_tile = A_tile.to(tl.float32)
        B_tile = B_tile.to(tl.float32)

        # Accumulate
        acc += tl.dot(A_tile, B_tile)  # (BLOCK_M, BLOCK_N)

    # Store results to C (C is fp32)
    c_ptrs = C_ptr + (offs_m[:, None] * N + offs_n[None, :])
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                selected_experts: torch.Tensor,
                routing_weights: torch.Tensor,
                expert_gate_weights: torch.Tensor,
                expert_up_weights: torch.Tensor,
                expert_down_weights: torch.Tensor):
        """
        Triton-optimized version of the original run function.
        - Sorts selected_experts and routing_weights in stable=True fashion using Triton.
        - Computes per-expert counts and cumsum starts using Triton kernels.
        - Assembles per-expert inputs via PyTorch scatter-add (Triton scatter not used here).
        - Launches Triton matmul kernel to compute gate_out, up_out, and expert_outputs for each selected token within capacity.
        - Accumulates final result using weights.
        """
        device = hidden_states.device

        # Shapes
        num_tokens, hidden_size = hidden_states.shape
        num_experts, _, moe_intermediate_size = expert_gate_weights.shape
        K = selected_experts.shape[1]

        # Flatten selected_experts and routing_weights
        selected_exp = selected_experts.reshape(-1).contiguous()     # [num_tokens*K]
        routing_flat = routing_weights.reshape(-1).contiguous()      # [num_tokens*K]
        N = len(selected_exp)

        # 1) Stable sort with Triton
        BLOCK_SORT = 1024
        grid_sort = (triton.cdiv(N, BLOCK_SORT),)
        sorted_exp = torch.empty(N, dtype=torch.int64, device=device)
        sorted_wt = torch.empty(N, dtype=torch.bfloat16, device=device)
        sort_stable_kernel[grid_sort](selected_exp, routing_flat, N, BLOCK=BLOCK_SORT)

        # 2) Triton bincount to compute per-expert counts
        per_exp_counts = torch.empty(num_experts, dtype=torch.int32, device=device)
        grid_bin = (triton.cdiv(num_experts, 1),)  # single program; fine
        bincount_kernel[grid_bin](sorted_exp, per_exp_counts, N, num_buckets=num_experts)

        # 3) Triton cumsum to get starts (prefix sum of counts)
        starts = torch.empty(num_experts, dtype=torch.int64, device=device)
        grid_cum = (triton.cdiv(num_experts, 1),)
        cumsum_kernel[grid_cum](per_exp_counts, starts, L=num_experts)

        # 4) Compute capacity
        total_selected = int((num_tokens * K))
        avg_tokens_per_expert = total_selected // num_experts
        capacity = max(int(avg_tokens_per_expert * 1.25), 1)
        per_exp_counts_cpu = per_exp_counts.tolist()

        # 5) Assemble per-expert inputs via PyTorch scatter-add
        expert_inputs = torch.zeros((num_experts, capacity, hidden_size), dtype=torch.bfloat16, device=device)

        # For each position p in the sorted list, map to original token and expert, and scatter hidden_states
        for p in range(N):
            original_token = p // K
            expert_id = int(sorted_exp[p].item())
            pos = int(p - int(starts[expert_id].item()))  # position within expert after sorting
            if pos < int(per_exp_counts_cpu[expert_id]) and pos < capacity:
                expert_inputs[expert_id, pos] = hidden_states[original_token]

        # 6) Compute gate_out, up_out, expert_outputs via Triton bmm for each selected token within capacity
        result = torch.zeros((num_tokens, hidden_size), dtype=torch.bfloat16, device=device)

        for p in range(N):
            original_token = p // K
            expert_id = int(sorted_exp[p].item())
            pos = int


def run(*args):
    return ModelNew()(*args)
