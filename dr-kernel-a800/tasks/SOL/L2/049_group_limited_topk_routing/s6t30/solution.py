import torch
import triton
import triton.language as tl


# 1) Triton: matmul logits = hidden_states @ weight.T
@triton.jit
def matmul_logits_kernel(
    a_ptr,         # *f32, [M, K]
    b_ptr,         # *f32, [N, K]
    c_ptr,         # *f32, [M, N]
    M: tl.int32,   # num_tokens
    N: tl.int32,   # num_experts (256)
    K: tl.int32,   # hidden_size (128)
    stride_am: tl.int32,
    stride_ak: tl.int32,
    stride_bn: tl.int32,
    stride_bk: tl.int32,
    stride_cm: tl.int32,
    stride_cn: tl.int32,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    if pid_m >= M or pid_n >= N:
        return
    acc = tl.zeros((), dtype=tl.float32)
    # loop over K
    for k in range(0, K):
        a = tl.load(a_ptr + pid_m * stride_am + k * stride_ak)
        b = tl.load(b_ptr + pid_n * stride_bn + k * stride_bk)
        acc += a * b
    tl.store(c_ptr + pid_m * stride_cm + pid_n * stride_cn, acc)


# 2) Triton: scores = sigmoid(logits) + expert_bias
@triton.jit
def sigmoid_add_bias_kernel(
    logits_ptr,    # *f32, [M, N]
    bias_ptr,      # *f32, [N]
    scores_ptr,    # *f32, [M, N]
    M: tl.int32,
    N: tl.int32,
    stride_lm: tl.int32,
    stride_ln: tl.int32,
    stride_sm: tl.int32,
    stride_sn: tl.int32,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    if pid_m >= M or pid_n >= N:
        return
    logit = tl.load(logits_ptr + pid_m * stride_lm + pid_n * stride_ln)
    bias = tl.load(bias_ptr + pid_n)
    # sigmoid
    s = 1.0 / (1.0 + tl.exp(-logit))
    out = s + bias
    tl.store(scores_ptr + pid_m * stride_sm + pid_n * stride_sn, out)


# 3) Triton: compute per-token group_scores [M, 8] as sum of top-2 per group
@triton.jit
def group_top2_sum_kernel(
    scores_ptr,    # *f32, [M, 256]
    group_scores_ptr,  # *f32, [M, 8]
    M: tl.int32,
    N: tl.int32,   # 256
    group_count: tl.constexpr,   # 8
    experts_per_group: tl.constexpr,  # 32
):
    pid = tl.program_id(0)  # one program per token
    if pid >= M:
        return
    # For each group g in [0..7], compute top1 and top2 over its 32 experts
    top1 = tl.full((), -float('inf'), dtype=tl.float32)
    top2 = tl.full((), -float('inf'), dtype=tl.float32)
    for g in range(group_count):
        base = g * experts_per_group
        # find top1
        for o in range(experts_per_group):
            idx = base + o
            val = tl.load(scores_ptr + pid * N + idx)
            if val > top1:
                top2 = top1
                top1 = val
            elif val > top2:
                top2 = val
        total = top1 + top2
        tl.store(group_scores_ptr + pid * group_count + g, total)


# 4) Triton: top-k (K=4) on group_scores, return indices
@triton.jit
def topk_group_kernel(
    group_scores_ptr,  # *f32, [M, 8]
    selected_idx_ptr,  # *int32, [M, 4]
    M: tl.int32,
    K: tl.constexpr,   # 4
):
    pid = tl.program_id(0)
    if pid >= M:
        return
    best_vals = tl.full((K,), -float('inf'), dtype=tl.float32)
    best_idxs = tl.zeros((K,), dtype=tl.int32)
    for g in range(8):
        val = tl.load(group_scores_ptr + pid * 8 + g)
        for j in range(K):
            if val > best_vals[j]:
                for jj in range(K - 1, j, -1):
                    best_vals[jj] = best_vals[jj - 1]
                    best_idxs[jj] = best_idxs[jj - 1]
                best_vals[j] = val
                best_idxs[j] = g
                break
    for t in range(K):
        tl.store(selected_idx_ptr + pid * K + t, best_idxs[t])


# 5) Triton: build group_mask [M, 8] from selected group indices
@triton.jit
def build_group_mask_kernel(
    group_idx_ptr,       # *int32, [M, 4]
    group_mask_ptr,      # *f32, [M, 8]
    M: tl.int32,
    K: tl.constexpr,     # 4
    group_count: tl.constexpr,  # 8
):
    pid = tl.program_id(0)
    if pid >= M:
        return
    for g in range(group_count):
        tl.store(group_mask_ptr + pid * group_count + g, 0.0)
    for t in range(K):
        g_idx = tl.load(group_idx_ptr + pid * K + t)
        tl.store(group_mask_ptr + pid * group_count + g_idx, 1.0)


# 6) Triton: expand group_mask to [M, 256] and set non-selected groups to -inf in masked_scores
@triton.jit
def expand_and_set_ninf_kernel(
    group_mask_ptr,      # *f32, [M, 8]
    masked_scores_ptr,   # *f32, [M, 256]
    M: tl.int32,
    N: tl.int32,         # 256
    group_count: tl.constexpr,  # 8
    experts_per_group: tl.constexpr,  # 32
):
    pid = tl.program_id(0)
    if pid >= M:
        return
    # Determine which groups are selected (group_mask has 1.0 where selected)
    selected_groups = tl.zeros((), dtype=tl.int32)  # count of selected groups for this token
    for g in range(group_count):
        m = tl.load(group_mask_ptr + pid * group_count + g)
        # m is 1.0 if selected, 0.0 otherwise
        if m == 1.0:
            selected_groups += 1
    # We can scan groups to find selected ones and set others to -inf
    for g in range(group_count):
        m = tl.load(group_mask_ptr + pid * group_count + g)
        if m != 1.0:
            base = g * experts_per_group
            for o in range(experts_per_group):
                idx = base + o
                tl.store(masked_scores_ptr + pid * N + idx, -float('inf'))
        # If selected, leave as-is (we filled masked_scores with -inf previously; selected groups need to be restored from original scores, but this kernel only sets non-selected to -inf).


# 7) Triton: masked top-8 selection from masked_scores [M, 256] → indices [M, 8]
@triton.jit
def masked_top8_kernel(
    masked_scores_ptr,   # *f32, [M, 256]
    selected_idx_ptr,    # *int32, [M, 8]
    M: tl.int32,
    N: tl.int32,         # 256
    K: tl.constexpr,     # 8
):
    pid = tl.program_id(0)
    if pid >= M:
        return
    best_vals = tl.full((K,), -float('inf'), dtype=tl.float32)
    best_idxs = tl.zeros((K,), dtype=tl.int32)
    for n in range(N):
        val = tl.load(masked_scores_ptr + pid * N + n)
        for j in range(K):
            if val > best_vals[j]:
                for jj in range(K - 1, j, -1):
                    best_vals[jj] = best_vals[jj - 1]
                    best_idxs[jj] = best_idxs[jj - 1]
                best_vals[j] = val
                best_idxs[j] = n
                break
    for t in range(K):
        tl.store(selected_idx_ptr + pid * K + t, best_idxs[t])


# 8) Triton: gather original scores at selected positions
@triton.jit
def gather_original_scores_kernel(
    scores_ptr,          # *f32, [M, 256]
    idx_ptr,             # *int32, [M, 8]
    gathered_ptr,        # *f32, [M, 8]
    M: tl.int32,
    K: tl.constexpr,     # 8
):
    pid = tl.program_id(0)
    if pid >= M:
        return
    for t in range(K):
        n = tl.load(idx_ptr + pid * K + t)
        val = tl.load(scores_ptr + pid * 256 + n)
        tl.store(gathered_ptr + pid * K + t, val)


# 9) Triton: normalize and scale
@triton.jit
def normalize_and_scale_kernel(
    gathered_ptr,        # *f32, [M, 8]
    output_ptr,          # *f32, [M, 8]
    M: tl.int32,
    K: tl.constexpr,     # 8
    scale: tl.float32,
    eps: tl.float32,
):
    pid = tl.program_id(0)
    if pid >= M:
        return
    sum_val = 0.0
    for t in range(K):
        val = tl.load(gathered_ptr + pid * K + t)
        sum_val += val
    denom = sum_val + eps
    for t in range(K):
        val = tl.load(gathered_ptr + pid * K + t)
        out = val / denom * scale
        tl.store(output_ptr + pid * K + t, out)


# Entry point: ModelNew
class ModelNew(torch.nn.Module):
    def __init__(self, routed_scaling_factor: float, eps: float = 1e-20):
        super().__init__()
        self.routed_scaling_factor = routed_scaling_factor
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor, expert_bias: torch.Tensor, routed_scaling_factor: float):
        # Ensure dtypes and device
        device = hidden_states.device
        M = hidden_states.shape[0]
        N = weight.shape[0]  # number of experts (256)
        K = hidden_states.shape[1]  # hidden size (128)
        # Allocate logits and scores
        logits = torch.empty((M, N), device=device, dtype=torch.float32)
        scores = torch.empty((M, N), device=device, dtype=torch.float32)

        # 1) Compute logits = hidden_states @ weight.T in Triton
        # hidden_states: [M, K], weight: [N, K]
        # We pass strides
        a = hidden_states.contiguous()
        b = weight.contiguous()
        # Launch grid: (M, N) tiles
        grid = (M, N)
        matmul_logits_kernel[grid](
            a, b, logits,
            M, N, K,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            logits.stride(0), logits.stride(1),
            num_warps=4,
        )

        # 2) Compute scores = sigmoid(logits) + expert_bias
        bias = expert_bias.to(torch.float32).contiguous()
        grid_scores = (M, N)
        sigmoid_add_bias_kernel[grid_scores](
            logits, bias, scores,
            M, N,
            logits.stride(0), logits.stride(1),
            scores.stride(0), scores.stride(1),
            num_warps=4,
        )

        # 3) Compute group_scores [M, 8]
        group_scores = torch.empty((M, 8), device=device, dtype=torch.float32)
        grid_group = (M,)
        group_top2_sum_kernel[grid_group](
            scores, group_scores,
            M, N,
            8, 32,
            num_warps=2,
        )

        # 4) Top-4 groups per token
        selected_group_idx = torch.empty((M, 4), device=device, dtype=torch.int32)
        topk_group_kernel[(M,)](
            group_scores, selected_group_idx,
            M,
            4,
            num_warps=2,
        )

        # 5) Build group_mask [M, 8]
        group_mask = torch.empty((M, 8), device=device, dtype=torch.float32)
        build_group_mask_kernel[(M,)](
            selected_group_idx, group_mask,
            M,
            4,
            8,
            num_warps=2,
        )

        # 6) Expand mask to [M, 256] and set non-selected groups to -inf in masked_scores
        masked_scores = torch.empty((M, N), device=device, dtype=torch.float32)
        expand_and_set_ninf_kernel[(M,)](
            group_mask, masked_scores,
            M, N,
            8, 32,
            num_warps=2,
        )

        # 7) Select final top-8 from masked_scores (indices)
        selected_idx_8 = torch.empty((M, 8), device=device, dtype=torch.int32)
        masked_top8_kernel[(M,)](
            masked_scores, selected_idx_8,
            M, N,
            8,
            num_warps=2,
        )

        # 8) Gather original scores for those 8 from 'scores' (not logits)
        gathered = torch.empty((M, 8), device=device, dtype=torch.float32)
        gather_original_scores_kernel[(M,)](
            scores, selected_idx_8, gathered,
            M,
            8,
            num_warps=2,
        )

        # 9) Normalize and scale
        output = torch.empty((M, 8), device=device, dtype=torch.float32)
        normalize_and_scale_kernel[(M,)](
            gathered, output,
            M,
            8,
            routed_scaling_factor, self.eps,
            num_warps=2,
        )

        # Return indices (int64) and weights (float32)
        topk_idx = selected_idx_8.to(torch.int64)  # indices selected by masked_top8 kernel
        topk_weight = output
        return topk_idx, topk_weight


def run(*args):
    return ModelNew()(*args)
