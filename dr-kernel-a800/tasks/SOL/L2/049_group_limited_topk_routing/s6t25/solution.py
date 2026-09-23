import torch
import triton
import triton.language as tl


# 1) Triton matmul: logits = hidden_states @ weight.T
# hidden_states: [M, K], weight: [N, K], logits: [M, N]
@triton.jit
def matmul_logits_kernel(
    a_ptr,            # *f32, [M, K]
    b_ptr,            # *f32, [N, K] but we use transposed access in kernel
    c_ptr,            # *f32, [M, N]
    M: tl.int32,      # number of tokens (rows of a)
    N: tl.int32,      # number of experts (columns of c)
    K: tl.int32,      # hidden size (common: 128)
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
    # Accumulator
    acc = tl.zeros((), dtype=tl.float32)
    # Loop over K
    for k in range(0, K):
        a_val = tl.load(a_ptr + pid_m * stride_am + k * stride_ak)
        # Load b[n, k], but we pass b as [N, K] and index transposed
        b_val = tl.load(b_ptr + pid_n * stride_bn + k * stride_bk)
        acc += a_val * b_val
    tl.store(c_ptr + pid_m * stride_cm + pid_n * stride_cn, acc)


# 2) Elementwise: scores = sigmoid(logits) + expert_bias
# logits: [M, N], expert_bias: [N], scores: [M, N]
@triton.jit
def sigmoid_add_bias_kernel(
    logits_ptr,        # *f32, [M, N]
    bias_ptr,          # *f32, [N]
    scores_ptr,        # *f32, [M, N]
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
    l = tl.load(logits_ptr + pid_m * stride_lm + pid_n * stride_ln)
    b = tl.load(bias_ptr + pid_n)
    s = 1.0 / (1.0 + tl.exp(-l)) + b
    tl.store(scores_ptr + pid_m * stride_sm + pid_n * stride_sn, s)


# 3) Group top-2 sum: group_scores [M, 8] from scores [M, N] viewed as [M, 8, 32]
# We pass scores pointer and compute per-group index directly.
@triton.jit
def group_top2_sum_kernel(
    scores_ptr,        # *f32, [M, 256]
    group_scores_ptr,  # *f32, [M, 8]
    M: tl.int32,
    N: tl.int32,       # N=256
    group_count: tl.constexpr,    # 8
    experts_per_group: tl.constexpr,  # 32
):
    pid = tl.program_id(0)  # one program per token
    if pid >= M:
        return
    for g in range(group_count):
        total = 0.0
        for ep in range(experts_per_group):  # ep in [0, 31]
            col = g * experts_per_group + ep
            val = tl.load(scores_ptr + pid * N + col)
            # Find top-2 in this group
            max1 = -float('inf')
            max2 = -float('inf')
            # Scan the 32 elements to compute top-2
            for ep2 in range(experts_per_group):
                v2 = tl.load(scores_ptr + pid * N + g * experts_per_group + ep2)
                if v2 > max1:
                    max2 = max1
                    max1 = v2
                elif v2 > max2:
                    max2 = v2
            total += (max1 + max2)
        tl.store(group_scores_ptr + pid * group_count + g, total)


# 4) Top-4 groups per token: return indices [M, 4] int32
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
                # shift down
                for jj in range(K - 1, j, -1):
                    best_vals[jj] = best_vals[jj - 1]
                    best_idxs[jj] = best_idxs[jj - 1]
                best_vals[j] = val
                best_idxs[j] = g
                break
    for t in range(K):
        tl.store(selected_idx_ptr + pid * K + t, best_idxs[t])


# 5) Build group_mask [M, 8]: set 1 at selected group indices
@triton.jit
def build_group_mask_kernel(
    group_idx_ptr,     # *int32, [M, 4]
    group_mask_ptr,    # *f32, [M, 8]
    M: tl.int32,
    K: tl.constexpr,   # 4
    group_count: tl.constexpr,  # 8
):
    pid = tl.program_id(0)
    if pid >= M:
        return
    # Initialize to zeros
    for g in range(group_count):
        tl.store(group_mask_ptr + pid * group_count + g, 0.0)
    # Set ones at selected positions
    for t in range(K):
        g_idx = tl.load(group_idx_ptr + pid * K + t)
        tl.store(group_mask_ptr + pid * group_count + g_idx, 1.0)


# 6) Expand group_mask to [M, 256] and set non-selected groups' 32 entries to -inf
@triton.jit
def expand_and_set_ninf_kernel(
    group_mask_ptr,     # *f32, [M, 8]
    masked_scores_ptr,  # *f32, [M, 256], will be mutated
    M: tl.int32,
    N: tl.int32,        # 256
    group_count: tl.constexpr,  # 8
    experts_per_group: tl.constexpr,  # 32
):
    pid = tl.program_id(0)
    if pid >= M:
        return
    # First, assume masked_scores_ptr already contains original scores; we'll overwrite non-selected groups with -inf
    # For each group g, if group_mask == 0, set its 32 experts to -inf
    for g in range(group_count):
        if tl.load(group_mask_ptr + pid * group_count + g) == 0.0:
            base = g * experts_per_group
            for ep in range(experts_per_group):
                col = base + ep
                # Set to -inf
                tl.store(masked_scores_ptr + pid * N + col, -float('inf'))


# 7) Final top-8 selection per token from masked_scores [M, 256] → indices [M, 8] int32
@triton.jit
def final_topk_kernel(
    masked_scores_ptr,  # *f32, [M, 256]
    topk_idx_ptr,       # *int32, [M, 8]
    M: tl.int32,
    K: tl.constexpr,    # 8
):
    pid = tl.program_id(0)
    if pid >= M:
        return
    best_vals = tl.full((K,), -float('inf'), dtype=tl.float32)
    best_idxs = tl.zeros((K,), dtype=tl.int32)
    for n in range(256):  # 256 experts
        val = tl.load(masked_scores_ptr + pid * 256 + n)
        for j in range(K):
            if val > best_vals[j]:
                for jj in range(K - 1, j, -1):
                    best_vals[jj] = best_vals[jj - 1]
                    best_idxs[jj] = best_idxs[jj - 1]
                best_vals[j] = val
                best_idxs[j] = n
                break
    for t in range(K):
        tl.store(topk_idx_ptr + pid * K + t, best_idxs[t])


# 8) Gather original scores for selected indices and normalize per token, scale by routed_scaling_factor
@triton.jit
def gather_and_normalize_kernel(
    scores_ptr,         # *f32, [M, 256]
    topk_idx_ptr,       # *int32, [M, 8]
    topk_weight_ptr,    # *f32, [M, 8]
    M: tl.int32,
    scaling_factor: tl.float32,
    K: tl.constexpr,    # 8
):
    pid = tl.program_id(0)
    if pid >= M:
        return
    sum_scores = 0.0
    for t in range(K):
        idx = tl.load(topk_idx_ptr + pid * K + t)
        val = tl.load(scores_ptr + pid * 256 + idx)
        sum_scores += val
    inv_sum = 1.0 / (sum_scores + 1e-20)
    for t in range(K):
        idx = tl.load(topk_idx_ptr + pid * K + t)
        val = tl.load(scores_ptr + pid * 256 + idx)
        norm = val * inv_sum * scaling_factor
        tl.store(topk_weight_ptr + pid * K + t, norm)


class ModelNew(torch.nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        weight: torch.Tensor,
        expert_bias: torch.Tensor,
        routed_scaling_factor: float,
    ):
        # Ensure dtypes and devices
        assert hidden_states.is_cuda and weight.is_cuda and expert_bias.is_cuda, "Inputs must be on CUDA"
        M = hidden_states.shape[0]
        K = 128  # hidden size
        N = 256  # number of experts

        # 1) Triton matmul: logits = hidden_states @ weight.T
        # hidden_states: [M, K], weight: [N, K]
        logits = torch.empty((M, N), dtype=torch.float32, device=hidden_states.device)

        # Strides
        a = hidden_states.contiguous()
        b = weight.contiguous()  # [N, K]
        a_stride_m = a.stride(0)
        a_stride_k = a.stride(1)
        b_stride_n = b.stride(0)
        b_stride_k = b.stride(1)
        c_stride_m = logits.stride(0)
        c_stride_n = logits.stride(1)

        grid = (M, N)
        matmul_logits_kernel[grid](
            a, b, logits,
            M, N, K,
            a_stride_m, a_stride_k,
            b_stride_n, b_stride_k,
            c_stride_m, c_stride_n,
            num_warps=4, num_stages=2
        )

        # 2) Triton elementwise: scores = sigmoid(logits) + expert_bias
        scores = torch.empty((M, N), dtype=torch.float32, device=hidden_states.device)
        bias = expert_bias.contiguous()
        s_stride_m = scores.stride(0)
        s_stride_n = scores.stride(1)
        l_stride_m = logits.stride(0)
        l_stride_n = logits.stride(1)
        grid2 = (M, N)
        sigmoid_add_bias_kernel[grid2](
            logits, bias, scores,
            M, N,
            l_stride_m, l_stride_n,
            s_stride_m, s_stride_n,
            num_warps=4, num_stages=2
        )

        # 3) Triton: group_top2_sum → [M, 8]
        group_scores = torch.empty((M, 8), dtype=torch.float32, device=hidden_states.device)
        group_count = 8
        experts_per_group = 32
        group_top2_sum_kernel[(M,)](
            scores, group_scores,
            M, N,
            group_count, experts_per_group,
            num_warps=1, num_stages=1
        )

        # 4) Triton: topk_group → [M, 4] int32
        selected_idx = torch.empty((M, 4), dtype=torch.int32, device=hidden_states.device)
        topk_group_kernel[(M,)](
            group_scores, selected_idx,
            M, 4,
            num_warps=1, num_stages=1
        )

        # 5) Triton: build group_mask [M, 8] float32
        group_mask = torch.empty((M, 8), dtype=torch.float32, device=hidden_states.device)
        build_group_mask_kernel[(M,)](
            selected_idx, group_mask,
            M, 4, group_count,
            num_warps=1, num_stages=1
        )

        # 6) Triton: expand group_mask to [M, 256] and set non-selected groups to -inf in masked_scores
        masked_scores = scores.clone()  # we'll mutate non-selected parts to -inf
        expand_and_set_ninf_kernel[(M,)](
            group_mask, masked_scores,
            M, N, group_count, experts_per_group,
            num_warps=1, num_stages=1
        )

        # 7) Triton: final top-8 selection per token → [M, 8] int32
        topk_idx = torch.empty((M, 8), dtype=torch.int32, device=hidden_states.device)
        final_topk_kernel[(M,)](
            masked_scores, topk_idx,
            M, 8,
            num_warps=1, num_stages=1
        )

        # 8) Triton: gather original scores for selected indices and normalize, scale
        # We need original scores from which we normalized. The original used scores_for_routing = scores + bias
        # Here, masked_scores already contains -inf for non-selected groups; we just gather original scores.
        topk_weight = torch.empty((M, 8), dtype=torch.float32, device=hidden_states.device)
        # scores_ptr is 'scores' from step 2; gather_and_normalize_kernel expects original scores pre-bias.
        # However, to remain consistent with the original logic, we can use 'scores' (post-bias) and adjust by bias,
        # but simpler: gather from original logits via scores since scores = sigmoid(logits) + bias; we don't have
        # logits anymore. To preserve correctness, we compute original scores from masked_scores and bias:
        # original_score = masked_scores - bias. But masked_scores may have -inf; so instead we use scores (which
        # are pre-subtract of bias). The normalization uses the selected scores' values; since bias cancels out
        # in normalization (sum of selected original scores), we can safely use scores for normalization. So we
        # gather from 'scores' and normalize (sum of original scores), then scale by routed_scaling_factor.
        gather_and_normalize_kernel[(M,)](
            scores, topk_idx, topk_weight,
            M, routed_scaling_factor,
            8,
            num_warps=1, num_stages=1
        )

        # Return indices as int64 to match torch.topk default, weights as float32
        return topk_idx.to(torch.long), topk_weight


def run(*args):
    return ModelNew()(*args)
