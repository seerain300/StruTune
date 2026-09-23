import torch
import triton
import triton.language as tl

# 1) Triton matmul: logits = hidden_states @ weight.T
# hidden_states: [M, K], weight: [N, K], logits: [M, N]
@triton.jit
def logits_kernel(
    hidden_ptr,   # *f32, [M, K]
    weight_ptr,   # *f32, [N, K]
    logits_ptr,   # *f32, [M, N]
    M: tl.int32,
    N: tl.int32,
    K: tl.int32,
    stride_hs_m: tl.int32, stride_hs_k: tl.int32,
    stride_w_n: tl.int32, stride_w_k: tl.int32,
    stride_l_m: tl.int32, stride_l_n: tl.int32,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    if pid_m >= M or pid_n >= N:
        return
    acc = tl.zeros((), dtype=tl.float32)
    for k in range(0, K):
        hs = tl.load(hidden_ptr + pid_m * stride_hs_m + k * stride_hs_k)
        w = tl.load(weight_ptr + pid_n * stride_w_n + k * stride_w_k)
        acc += hs * w
    tl.store(logits_ptr + pid_m * stride_l_m + pid_n * stride_l_n, acc)

# 2) Triton elementwise: scores = sigmoid(logits) + expert_bias
# logits: [M, N], expert_bias: [N], scores: [M, N]
@triton.jit
def sigmoid_add_bias_kernel(
    logits_ptr,    # *f32, [M, N]
    bias_ptr,      # *f32, [N]
    scores_ptr,    # *f32, [M, N]
    M: tl.int32,
    N: tl.int32,
    stride_l_m: tl.int32, stride_l_n: tl.int32,
    stride_s_m: tl.int32, stride_s_n: tl.int32,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    if pid_m >= M or pid_n >= N:
        return
    val = tl.load(logits_ptr + pid_m * stride_l_m + pid_n * stride_l_n)
    bias = tl.load(bias_ptr + pid_n)
    out = 1.0 / (1.0 + tl.exp(-val)) + bias
    tl.store(scores_ptr + pid_m * stride_s_m + pid_n * stride_s_n, out)

# 3) Triton: compute per-group top-2 sums → group_scores [M, 8]
# We logically reshape scores as [M, 8, 32]. For each token g in [0..7], loop 32 experts.
@triton.jit
def group_top2_sum_kernel(
    scores_ptr,    # *f32, [M, N]
    group_scores_ptr,  # *f32, [M, 8]
    M: tl.int32,
    group_count: tl.constexpr,  # 8
    per_group: tl.constexpr,    # 32
):
    pid = tl.program_id(0)
    if pid >= M:
        return
    base = pid * (group_count * per_group)
    top1 = tl.full((), -float('inf'), dtype=tl.float32)
    top2 = tl.full((), -float('inf'), dtype=tl.float32)
    for g in range(group_count):
        start = base + g * per_group
        for j in range(per_group):
            val = tl.load(scores_ptr + start + j)
            if val > top1:
                top2 = top1
                top1 = val
            elif val > top2:
                top2 = val
    total = top1 + top2
    tl.store(group_scores_ptr + pid * group_count + g, total)

# 4) Triton: topk_group_kernel (K=4) — returns selected group indices [M, 4] (int32)
@triton.jit
def topk_group_kernel(
    group_scores_ptr,    # *f32, [M, 8]
    selected_idx_ptr,    # *int32, [M, 4]
    M: tl.int32,
    K: tl.constexpr,     # 4
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

# 5) Triton: build_group_mask_kernel — given selected group_idx [M, 4], set group_mask [M, 8] to 1 at selected positions
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
    # initialize zeros
    for g in range(group_count):
        tl.store(group_mask_ptr + pid * group_count + g, 0.0)
    # set ones at selected groups
    for t in range(K):
        g_idx = tl.load(group_idx_ptr + pid * K + t)
        tl.store(group_mask_ptr + pid * group_count + g_idx, 1.0)

# 6) Triton: expand_and_set_ninf_kernel — expand group_mask [M, 8] to [M, 256], set non-selected groups' 32 entries to -inf
@triton.jit
def expand_and_set_ninf_kernel(
    group_mask_ptr,      # *f32, [M, 8]
    masked_scores_ptr,   # *f32, [M, 256], will be mutated
    M: tl.int32,
    N: tl.int32,         # 256
    group_count: tl.constexpr,  # 8
    per_group: tl.constexpr,    # 32
):
    pid = tl.program_id(0)
    if pid >= M:
        return
    base = pid * N
    for g in range(group_count):
        flag = tl.load(group_mask_ptr + pid * group_count + g)
        if flag == 0.0:
            start = g * per_group
            for j in range(per_group):
                idx = base + start + j
                tl.store(masked_scores_ptr + idx, -float('inf'))

# 7) Triton: masked_top8_experts_kernel — perform top-k=8 selection on masked_scores [M, 256] → indices [M, 8]
@triton.jit
def masked_top8_experts_kernel(
    masked_scores_ptr,   # *f32, [M, 256]
    selected_idx_ptr,    # *int32, [M, 8]
    M: tl.int32,
    K: tl.constexpr,     # 8
    N: tl.int32,         # 256
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

# 8) Triton: gather_selected_scores — given scores [M, N] and selected_idx [M, 8], write selected scores [M, 8]
@triton.jit
def gather_selected_scores_kernel(
    scores_ptr,          # *f32, [M, N]
    selected_idx_ptr,    # *int32, [M, 8]
    selected_scores_ptr, # *f32, [M, 8]
    M: tl.int32,
    K: tl.constexpr,     # 8
    N: tl.int32,
):
    pid = tl.program_id(0)
    if pid >= M:
        return
    for t in range(K):
        idx = tl.load(selected_idx_ptr + pid * K + t)
        val = tl.load(scores_ptr + pid * N + idx)
        tl.store(selected_scores_ptr + pid * K + t, val)

# 9) Triton: normalize_and_scale_kernel — selected_scores [M, 8], scale routed_scaling_factor
@triton.jit
def normalize_and_scale_kernel(
    selected_scores_ptr, # *f32, [M, 8]
    scaled_ptr,          # *f32, [M, 8]
    M: tl.int32,
    K: tl.constexpr,     # 8
    scale: tl.float32,
):
    pid = tl.program_id(0)
    if pid >= M:
        return
    total = tl.zeros((), dtype=tl.float32)
    for t in range(K):
        val = tl.load(selected_scores_ptr + pid * K + t)
        total += val
    for t in range(K):
        val = tl.load(selected_scores_ptr + pid * K + t)
        out = val / total * scale
        tl.store(scaled_ptr + pid * K + t, out)

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Constants from the original setup
        self.group_count = 8
        self.per_group = 32
        self.top_k = 8
        self.k_group = 4
        self.num_experts = self.group_count * self.per_group  # 256

    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor, expert_bias: torch.Tensor, routed_scaling_factor: float):
        # Ensure device and dtype
        device = hidden_states.device
        M = hidden_states.shape[0]
        N = self.num_experts  # 256

        # 1) Triton matmul logits: [M, N]
        logits = torch.empty((M, N), dtype=torch.float32, device=device)
        stride_hs_m, stride_hs_k = hidden_states.stride(0), hidden_states.stride(1)
        stride_w_n, stride_w_k = weight.stride(0), weight.stride(1)
        stride_l_m, stride_l_n = logits.stride(0), logits.stride(1)
        logits_kernel[(M, N)](
            hidden_states, weight, logits,
            M, N, 128,
            stride_hs_m, stride_hs_k,
            stride_w_n, stride_w_k,
            stride_l_m, stride_l_n,
        )

        # 2) Triton: scores = sigmoid(logits) + expert_bias
        scores = torch.empty((M, N), dtype=torch.float32, device=device)
        stride_l_m, stride_l_n = logits.stride(0), logits.stride(1)
        stride_s_m, stride_s_n = scores.stride(0), scores.stride(1)
        expert_bias = expert_bias.to(torch.float32).to(device)
        sigmoid_add_bias_kernel[(M, N)](
            logits, expert_bias, scores,
            M, N,
            stride_l_m, stride_l_n,
            stride_s_m, stride_s_n,
        )

        # 3) Triton: group_top2_sum → group_scores [M, 8]
        group_scores = torch.empty((M, self.group_count), dtype=torch.float32, device=device)
        group_top2_sum_kernel[(M,)](
            scores, group_scores,
            M, self.group_count, self.per_group
        )

        # 4) Triton: topk_group_kernel (K=4) → selected group indices [M, 4]
        selected_group_idx = torch.empty((M, self.k_group), dtype=torch.int32, device=device)
        topk_group_kernel[(M,)](
            group_scores, selected_group_idx,
            M, self.k_group
        )

        # 5) Triton: build_group_mask [M, 8]
        group_mask = torch.empty((M, self.group_count), dtype=torch.float32, device=device)
        build_group_mask_kernel[(M,)](
            selected_group_idx, group_mask,
            M, self.k_group, self.group_count
        )

        # 6) Triton: expand_and_set_ninf_kernel — produce masked_scores [M, 256]
        masked_scores = torch.empty((M, N), dtype=torch.float32, device=device)
        masked_scores.copy_(scores)  # initialize with scores
        expand_and_set_ninf_kernel[(M,)](
            group_mask, masked_scores,
            M, N, self.group_count, self.per_group
        )

        # 7) Triton: masked_top8_experts_kernel → selected expert indices [M, 8]
        final_selected_idx = torch.empty((M, self.top_k), dtype=torch.int32, device=device)
        masked_top8_experts_kernel[(M,)](
            masked_scores, final_selected_idx,
            M, self.top_k, N
        )

        # 8) Triton: gather_selected_scores from original scores (pre-mask)
        gathered_selected_scores = torch.empty((M, self.top_k), dtype=torch.float32, device=device)
        gather_selected_scores_kernel[(M,)](
            scores, final_selected_idx, gathered_selected_scores,
            M, self.top_k, N
        )

        #


def run(*args):
    return ModelNew()(*args)
