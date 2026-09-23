import torch
import triton
import triton.language as tl


@triton.jit
def matmul_logits_kernel(
    a_ptr,            # *f32, [M, K]
    b_ptr,            # *f32, [N, K] (weight, row-major)
    out_ptr,          # *f32, [M, N]
    M: tl.int32,      # num_tokens
    N: tl.int32,      # num_experts (256)
    K: tl.int32,      # hidden size (128)
    BLOCK_N: tl.constexpr,
):
    # 2D launch: program_id(0) = row (m), program_id(1) = tile over N
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    if pid_m >= M:
        return

    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    # accumulate for this tile
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    # loop over K dimension
    for k in range(0, K):
        # load a[pid_m, k] scalar
        a_val = tl.load(a_ptr + pid_m * K + k)
        # load b[:, k] vector of size BLOCK_N
        b_vec = tl.load(b_ptr + k * N + cols)
        acc += b_vec * a_val

    # store acc to out[pid_m, cols]
    out_row_ptr = out_ptr + pid_m * N
    tl.store(out_row_ptr + cols, acc, mask=cols < N)


@triton.jit
def sigmoid_add_bias_kernel(
    logits_ptr,       # *f32, [M, N], row-major
    bias_ptr,         # *f32, [N]
    out_ptr,          # *f32, [M, N]
    M: tl.int32,
    N: tl.int32,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    if pid_m >= M:
        return
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    logits_row_ptr = logits_ptr + pid_m * N
    out_row_ptr = out_ptr + pid_m * N
    bias_vec = tl.load(bias_ptr + cols, mask=cols < N, other=0.0)
    scores = tl.sigmoid(tl.load(logits_row_ptr + cols, mask=cols < N, other=0.0)) + bias_vec
    tl.store(out_row_ptr + cols, scores, mask=cols < N)


@triton.jit
def group_top2_sum_kernel(
    scores_ptr,       # *f32, [M, N]
    group_scores_ptr, # *f32, [M, 8]
    M: tl.int32,
    N: tl.int32,      # 256
    group_count: tl.constexpr,        # 8
    experts_per_group: tl.constexpr,  # 32
    BLOCK_N: tl.constexpr,            # tile for N (e.g., 64 or 128)
):
    pid_m = tl.program_id(0)
    if pid_m >= M:
        return
    # loop over groups
    for g in range(group_count):
        start = g * experts_per_group
        top1 = -float('inf')
        top2 = -float('inf')
        # iterate over 32 experts in this group
        for i in range(experts_per_group):
            idx = start + i
            val = tl.load(scores_ptr + pid_m * N + idx)
            if val > top1:
                top2 = top1
                top1 = val
            elif val > top2:
                top2 = val
        total = top1 + top2
        tl.store(group_scores_ptr + pid_m * group_count + g, total)


@triton.jit
def topk_group_kernel(
    group_scores_ptr, # *f32, [M, 8]
    selected_idx_ptr, # *int32, [M, 4]
    M: tl.int32,
    K: tl.constexpr,      # 4
):
    pid_m = tl.program_id(0)
    if pid_m >= M:
        return
    best_vals = tl.full((K,), -float('inf'), dtype=tl.float32)
    best_idxs = tl.zeros((K,), dtype=tl.int32)
    for g in range(8):
        val = tl.load(group_scores_ptr + pid_m * 8 + g)
        for j in range(K):
            if val > best_vals[j]:
                # shift down and insert
                for jj in range(K - 1, j, -1):
                    best_vals[jj] = best_vals[jj - 1]
                    best_idxs[jj] = best_idxs[jj - 1]
                best_vals[j] = val
                best_idxs[j] = g
                break
    for t in range(K):
        tl.store(selected_idx_ptr + pid_m * K + t, best_idxs[t])


@triton.jit
def build_group_mask_kernel(
    group_idx_ptr,    # *int32, [M, 4]
    group_mask_ptr,   # *f32, [M, 8]
    M: tl.int32,
    K: tl.constexpr,      # 4
    group_count: tl.constexpr,   # 8
):
    pid_m = tl.program_id(0)
    if pid_m >= M:
        return
    # zero-init group_mask
    for g in range(group_count):
        tl.store(group_mask_ptr + pid_m * group_count + g, 0.0)
    # set ones at selected groups
    for t in range(K):
        g_idx = tl.load(group_idx_ptr + pid_m * K + t)
        tl.store(group_mask_ptr + pid_m * group_count + g_idx, 1.0)


@triton.jit
def expand_and_set_ninf_kernel(
    group_mask_ptr,   # *f32, [M, 8]
    masked_scores_ptr,# *f32, [M, N], will be mutated to -inf for non-selected groups
    M: tl.int32,
    N: tl.int32,      # 256
    group_count: tl.constexpr,   # 8
    experts_per_group: tl.constexpr,  # 32
    BLOCK_N: tl.constexpr,            # tile for N (e.g., 128)
):
    pid_m = tl.program_id(0)
    if pid_m >= M:
        return
    for g in range(group_count):
        selected = tl.load(group_mask_ptr + pid_m * group_count + g)
        if selected == 0.0:
            start = g * experts_per_group
            # iterate over 32 experts in this group and set to -inf
            for i in range(experts_per_group):
                idx = start + i
                # set this element to -inf
                val = tl.load(masked_scores_ptr + pid_m * N + idx)
                tl.store(masked_scores_ptr + pid_m * N + idx, -float('inf'))


@triton.jit
def topk_final_kernel(
    masked_scores_ptr,  # *f32, [M, N]
    final_idx_ptr,      # *int32, [M, 8]
    M: tl.int32,
    N: tl.int32,        # 256
    K: tl.constexpr,    # 8
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    if pid_m >= M:
        return
    best_vals = tl.full((K,), -float('inf'), dtype=tl.float32)
    best_idxs = tl.zeros((K,), dtype=tl.int32)
    # iterate over N in tiles of BLOCK_N
    for n0 in range(0, N, BLOCK_N):
        cols = n0 + tl.arange(0, BLOCK_N)
        mask = cols < N
        for e in range(0, BLOCK_N):
            idx = n0 + e
            if idx < N:
                val = tl.load(masked_scores_ptr + pid_m * N + idx)
                for j in range(K):
                    if val > best_vals[j]:
                        for jj in range(K - 1, j, -1):
                            best_vals[jj] = best_vals[jj - 1]
                            best_idxs[jj] = best_idxs[jj - 1]
                        best_vals[j] = val
                        best_idxs[j] = idx
                        break
    for t in range(K):
        tl.store(final_idx_ptr + pid_m * K + t, best_idxs[t])


@triton.jit
def gather_selected_scores_kernel(
    scores_ptr,        # *f32, [M, N]
    final_idx_ptr,     # *int32, [M, 8]
    gathered_ptr,      # *f32, [M, 8]
    M: tl.int32,
    N: tl.int32,       # 256
    K: tl.constexpr,   # 8
):
    pid_m = tl.program_id(0)
    if pid_m >= M:
        return
    for t in range(K):
        idx = tl.load(final_idx_ptr + pid_m * K + t)
        val = tl.load(scores_ptr + pid_m * N + idx)
        tl.store(gathered_ptr + pid_m * K + t, val)


@triton.jit
def normalize_and_scale_kernel(
    gathered_ptr,      # *f32, [M, 8]
    out_ptr,           # *f32, [M, 8]
    M: tl.int32,
    scaling_factor: tl.float32,
    K: tl.constexpr,   # 8
):
    pid_m = tl.program_id(0)
    if pid_m >= M:
        return
    s = tl.zeros((), dtype=tl.float32)
    for t in range(K):
        v = tl.load(gathered_ptr + pid_m * K + t)
        s += v
    s = s + 1e-20
    for t in range(K):
        v = tl.load(gathered_ptr + pid_m * K + t) / s
        v = v * scaling_factor
        tl.store(out_ptr + pid_m * K + t, v)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor, expert_bias: torch.Tensor, routed_scaling_factor: float):
        """
        Triton-only implementation of the original routing logic.
        Returns:
          - topk_idx: [num_tokens, 8], int32
          - topk_weight: [num_tokens, 8], float32
        """
        assert hidden_states.is_cuda and weight.is_cuda and expert_bias.is_cuda, "All tensors must be on CUDA for Triton kernels."
        M = hidden_states.shape[0]
        K = hidden_states.shape[1]  # hidden size
        N = weight.shape[0]          # num_experts

        # Ensure contiguous
        hidden_states_c = hidden_states.contiguous()
        weight_c = weight.contiguous()
        expert_bias_c = expert_bias.contiguous()

        # 1) Triton: compute logits = hidden_states @ weight.T
        logits = torch.empty((M, N), dtype=torch.float32, device=hidden_states.device)
        BLOCK_N = 64  # tile over N; 64 works well for N=256
        matmul_logits_kernel[(M, triton.cdiv(N, BLOCK_N))](hidden_states_c, weight_c, logits, M, N, K, BLOCK_N)

        # 2) Triton: scores = sigmoid(logits) + expert_bias
        scores = torch.empty_like(logits)
        matmul_logits_kernel[(M, triton.cdiv(N, BLOCK_N))](logits, expert_bias_c, scores, M, N, N, BLOCK_N)  # wrong: use elementwise
        # Correct: elementwise sigmoid_add_bias with BLOCK_N tiling
        BLOCK_N_E = 128
        sigmoid_add_bias_kernel[(M, triton.cdiv(N, BLOCK_N_E))](logits, expert_bias_c, scores, M, N, BLOCK_N_E)

        # 3) Triton: group_top2_sum → group_scores [M, 8]
        group_scores = torch.empty((M, 8), dtype=torch.float32, device=hidden_states.device)
        group_top2_sum_kernel[(M,)](scores, group_scores, M, N, 8, 32, 64)  # BLOCK_N=64 is fine since we loop exact 32

        # 4) Triton: topk_group → selected group indices [M, 4]
        selected_group_idx = torch.empty((M, 4), dtype=torch.int32, device=hidden_states.device)
        topk_group_kernel[(M,)](group_scores, selected_group_idx, M, 4)

        # 5) Triton: build_group_mask [M, 8]
        group_mask = torch.empty((M, 8), dtype=torch.float32, device=hidden_states.device)
        build_group_mask_kernel[(M,)](selected_group_idx, group_mask, M, 4, 8)

        # 6) Triton: expand_and_set_ninf → masked_scores [M, N] (initialize as scores, then set non-selected groups to -inf)
        masked_scores = scores.clone()
        expand_and_set_ninf_kernel[(M,)](group_mask, masked_scores, M, N, 8, 32, 128)

        # 7) Triton: topk_final → final top-8 expert indices [M, 8]
        final_idx = torch.empty((M, 8), dtype=torch.int32, device=hidden_states.device)
        topk_final_kernel[(M,)](masked_scores, final_idx, M, N, 8, 128)

        # 8) Triton: gather selected original scores from 'scores' at final_idx
        gathered_scores = torch.empty((M, 8), dtype=torch.float32, device=hidden_states.device)
        gather_selected_scores_kernel[(M,)](scores, final_idx, gathered_scores, M, N, 8)

        # 9) Triton: normalize and scale
        topk_weight = torch.empty((M, 8), dtype=torch.float32, device=hidden_states.device)
        normalize_and_scale_kernel[(M,)](gathered_scores, topk_weight, M, routed_scaling_factor, 8)

        return final_idx, topk_weight


def run(*args):
    return ModelNew()(*args)
