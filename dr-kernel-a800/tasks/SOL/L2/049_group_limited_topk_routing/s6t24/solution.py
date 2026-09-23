import torch
import triton
import triton.language as tl


@triton.jit
def matmul_logits_kernel(
    a_ptr,      # *f32, [M, K]
    b_ptr,      # *f32, [N, K] (note: weight is [N, K], we use b[j, k] = weight[k, j] for matmul)
    c_ptr,      # *f32, [M, N]
    M: tl.int32,
    N: tl.int32,
    K: tl.int32,
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
        a_val = tl.load(a_ptr + pid_m * stride_am + k * stride_ak)
        b_val = tl.load(b_ptr + pid_n * stride_bn + k * stride_bk)
        acc += a_val * b_val

    tl.store(c_ptr + pid_m * stride_cm + pid_n * stride_cn, acc)


@triton.jit
def sigmoid_add_bias_kernel(
    logits_ptr,    # *f32, [M, N]
    bias_ptr,      # *f32, [N]
    scores_ptr,    # *f32, [M, N]
    M: tl.int32,
    N: tl.int32,
    stride_lm: tl.int32,
    stride_ln: tl.int32,
    stride_sn: tl.int32,
):
    pid = tl.program_id(0)
    if pid >= M * N:
        return
    m = pid // N
    n = pid % N
    if m >= M or n >= N:
        return
    val = tl.load(logits_ptr + m * stride_lm + n * stride_ln)
    b = tl.load(bias_ptr + n)
    val = 1.0 / (1.0 + tl.exp(-val))  # sigmoid
    val = val + b
    tl.store(scores_ptr + m * stride_sn + n * stride_sn, val)


@triton.jit
def group_top2_sum_kernel(
    scores_ptr,    # *f32, [M, N]
    group_scores_ptr,  # *f32, [M, 8]
    M: tl.int32,
    N: tl.int32,
    group_count: tl.constexpr,   # 8
    per_group: tl.constexpr,     # 32
    stride_sm: tl.int32,
    stride_sn: tl.int32,
):
    pid = tl.program_id(0)
    if pid >= M:
        return
    # Each program handles one token
    total = tl.zeros((), dtype=tl.float32)
    for g in range(group_count):
        base = g * per_group
        top1 = tl.full((), -float('inf'), dtype=tl.float32)
        top2 = tl.full((), -float('inf'), dtype=tl.float32)
        for i in range(per_group):
            idx = base + i
            val = tl.load(scores_ptr + pid * stride_sm + idx * stride_sn)
            if val > top1:
                top2 = top1
                top1 = val
            elif val > top2:
                top2 = val
        total = total + top1 + top2
    tl.store(group_scores_ptr + pid * group_count + g, total)  # store per group total, but overwrite correct index
    # Note: we need per-group totals per group, so we'll recompute using pid*group_count + g indices
    # Better: compute per-group totals into group_scores_ptr[pid, g] via direct store:
    # Here we compute total for group g and store at [pid, g]
    tl.store(group_scores_ptr + pid * group_count + g, total)


# We'll fix the above issue by computing totals into a 2D tensor directly in host and pass it to Triton as 1D [M,8].
# For now, we'll implement a correct version below.

@triton.jit
def group_top2_sum_kernel_v2(
    scores_ptr,    # *f32, [M, N]
    group_scores_ptr,  # *f32, [M, 8]
    M: tl.int32,
    N: tl.int32,
    group_count: tl.constexpr,   # 8
    per_group: tl.constexpr,     # 32
    stride_sm: tl.int32,
    stride_sn: tl.int32,
):
    pid = tl.program_id(0)
    if pid >= M:
        return
    # Reshape [M, N] into [M, 8, 32] by decoding group and offset
    for g in range(group_count):
        base = g * per_group
        top1 = tl.full((), -float('inf'), dtype=tl.float32)
        top2 = tl.full((), -float('inf'), dtype=tl.float32)
        for i in range(per_group):
            idx = base + i
            val = tl.load(scores_ptr + pid * stride_sm + idx * stride_sn)
            if val > top1:
                top2 = top1
                top1 = val
            elif val > top2:
                top2 = val
        total = top1 + top2
        tl.store(group_scores_ptr + pid * group_count + g, total)


@triton.jit
def topk_group_kernel(
    group_scores_ptr,   # *f32, [M, 8]
    selected_idx_ptr,   # *int32, [M, 4]
    M: tl.int32,
    K: tl.constexpr,    # 4
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


@triton.jit
def build_group_mask_kernel(
    group_idx_ptr,      # *int32, [M, 4]
    group_mask_ptr,     # *f32, [M, 8]
    M: tl.int32,
    K: tl.constexpr,    # 4
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


@triton.jit
def expand_and_set_ninf_kernel(
    group_mask_ptr,     # *f32, [M, 8]
    masked_scores_ptr,  # *f32, [M, N]
    M: tl.int32,
    N: tl.int32,
    group_count: tl.constexpr,  # 8
    per_group: tl.constexpr,    # 32
    stride_sm: tl.int32,
    stride_sn: tl.int32,
):
    pid = tl.program_id(0)
    if pid >= M:
        return
    for g in range(group_count):
        if tl.load(group_mask_ptr + pid * group_count + g) == 1.0:
            continue
        base = g * per_group
        for i in range(per_group):
            idx = base + i
            tl.store(masked_scores_ptr + pid * stride_sm + idx * stride_sn, -float('inf'))


@triton.jit
def masked_top8_experts_kernel(
    masked_scores_ptr,  # *f32, [M, N]
    selected_idx_ptr,   # *int32, [M, 8]
    M: tl.int32,
    K: tl.constexpr,    # 8
    N: tl.int32,
    stride_sm: tl.int32,
    stride_sn: tl.int32,
):
    pid = tl.program_id(0)
    if pid >= M:
        return
    best_vals = tl.full((K,), -float('inf'), dtype=tl.float32)
    best_idxs = tl.zeros((K,), dtype=tl.int32)
    for e in range(N):
        val = tl.load(masked_scores_ptr + pid * stride_sm + e * stride_sn)
        for j in range(K):
            if val > best_vals[j]:
                for jj in range(K - 1, j, -1):
                    best_vals[jj] = best_vals[jj - 1]
                    best_idxs[jj] = best_idxs[jj - 1]
                best_vals[j] = val
                best_idxs[j] = e
                break
    for t in range(K):
        tl.store(selected_idx_ptr + pid * K + t, best_idxs[t])


@triton.jit
def gather_selected_scores_kernel(
    scores_ptr,         # *f32, [M, N]
    selected_idx_ptr,   # *int32, [M, 8]
    gathered_ptr,       # *f32, [M, 8]
    M: tl.int32,
    K: tl.constexpr,    # 8
    N: tl.int32,
    stride_sm: tl.int32,
    stride_sn: tl.int32,
):
    pid = tl.program_id(0)
    if pid >= M:
        return
    for t in range(K):
        idx = tl.load(selected_idx_ptr + pid * K + t)
        val = tl.load(scores_ptr + pid * stride_sm + idx * stride_sn)
        tl.store(gathered_ptr + pid * K + t, val)


@triton.jit
def normalize_and_scale_kernel(
    gathered_ptr,       # *f32, [M, 8]
    out_ptr,            # *f32, [M, 8]
    scale: tl.float32,
    M: tl.int32,
    K: tl.constexpr,    # 8
):
    pid = tl.program_id(0)
    if pid >= M:
        return
    row_sum = tl.zeros((), dtype=tl.float32)
    for t in range(K):
        row_sum = row_sum + tl.load(gathered_ptr + pid * K + t)
    for t in range(K):
        val = tl.load(gathered_ptr + pid * K + t)
        norm = val / (row_sum + 1e-20)
        tl.store(out_ptr + pid * K + t, norm * scale)


class ModelNew(torch.nn.Module):
    def __init__(self, num_experts=256, num_hidden=128, routed_scaling_factor=1.0):
        super().__init__()
        self.num_experts = num_experts
        self.num_hidden = num_hidden
        self.routed_scaling_factor = routed_scaling_factor
        self.group_count = 8
        self.per_group = num_experts // self.group_count  # 32
        self.top_k = 8
        self.k_group = 4

    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor, expert_bias: torch.Tensor, routed_scaling_factor: float):
        # Ensure device/dtype
        device = hidden_states.device
        M, K = hidden_states.shape
        N, Kw = weight.shape
        assert Kw == self.num_hidden, f"weight second dim must be {self.num_hidden}, got {Kw}"
        assert N == self.num_experts, f"num_experts must be {self.num_experts}, got {N}"
        assert expert_bias.shape[0] == self.num_experts, "expert_bias shape mismatch"
        # 1) Triton: compute logits = hidden_states @ weight.T
        logits = torch.empty((M, N), dtype=torch.float32, device=device)
        # Note: pass weight as [N, K], Triton will access b[j, k] which equals weight[k, j] for matmul
        matmul_logits_kernel[(M, N)](
            hidden_states, weight, logits,
            M, N, K,
            hidden_states.stride(0), hidden_states.stride(1),
            weight.stride(0), weight.stride(1),
            logits.stride(0), logits.stride(1),
            num_warps=4
        )

        # 2) Triton: scores = sigmoid(logits) + expert_bias
        scores = torch.empty((M, N), dtype=torch.float32, device=device)
        sigmoid_add_bias_kernel[(M * N,)](
            logits, expert_bias, scores,
            M, N,
            logits.stride(0), logits.stride(1),
            scores.stride(0),
            num_warps=1
        )

        # 3) Triton: group_top2_sum → group_scores [M, 8]
        group_scores = torch.empty((M, self.group_count), dtype=torch.float32, device=device)
        group_top2_sum_kernel_v2[(M,)](
            scores, group_scores,
            M, N,
            self.group_count, self.per_group,
            scores.stride(0), scores.stride(1),
            num_warps=4
        )

        # 4) Triton: topk_group_kernel (K=4) → selected group indices [M, 4]
        selected_group_idx = torch.empty((M, self.k_group), dtype=torch.int32, device=device)
        topk_group_kernel[(M,)](
            group_scores, selected_group_idx,
            M, self.k_group,
            num_warps=1
        )

        # 5) Triton: build_group_mask [M, 8]
        group_mask = torch.empty((M, self.group_count), dtype=torch.float32, device=device)
        build_group_mask_kernel[(M,)](
            selected_group_idx, group_mask,
            M, self.k_group, self.group_count,
            num_warps=1
        )

        # 6) Triton: expand_and_set_ninf — produce masked_scores [M, N]
        masked_scores = torch.empty((M, N), dtype=torch.float32, device=device)
        masked_scores.copy_(scores)  # initialize with scores
        expand_and_set_ninf_kernel[(M,)](
            group_mask, masked_scores,
            M, N, self.group_count, self.per_group,
            masked_scores.stride(0), masked_scores.stride(1),
            num_warps=4
        )

        # 7) Triton: final top-8 expert indices from masked_scores → [M, 8]
        final_selected_idx = torch.empty((M, self.top_k), dtype=torch.int32, device=device)
        masked_top8_experts_kernel[(M,)](
            masked_scores, final_selected_idx,
            M, self.top_k, N,
            masked_scores.stride(0), masked_scores.stride(1),
            num_warps=4
        )

        # 8) Triton: gather original scores for those 8
        gathered_selected_scores = torch.empty((M, self.top_k), dtype=torch.float32, device=device)
        gather_selected_scores_kernel[(M,)](
            scores, final_selected_idx, gathered_selected_scores,
            M, self.top_k, N,
            scores.stride(0), scores.stride(1),
            num_warps=1
        )

        # 9) Triton: normalize and scale
        topk_weight = torch.empty((M, self.top_k), dtype=torch.float32, device=device)
        normalize_and_scale_kernel[(M,)](
            gathered_selected_scores, topk_weight,
            self.routed_scaling_factor,
            M, self.top_k,
            num_warps=1
        )

        # Return indices and weights, with indices int64 as in original
        topk_idx = final_selected_idx.to(torch.int64)
        topk_weight = topk_weight  # already float32, scaled and normalized
        return topk_idx, topk_weight


def run(*args):
    return ModelNew()(*args)
