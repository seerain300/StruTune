import torch
import triton
import triton.language as tl

# Constants used in the original logic
GROUP_COUNT = 8
EXP_PER_GROUP = 32
TOP_K = 8
K_GROUP = 4

@triton.jit
def matmul_logits_kernel(
    hidden_ptr,      # *f32, [M, K]
    weight_ptr,      # *f32, [N, K]
    logits_ptr,      # *f32, [M, N]
    M: tl.int32, N: tl.int32, K: tl.int32,
    stride_h_m: tl.int32, stride_h_k: tl.int32,
    stride_w_n: tl.int32, stride_w_k: tl.int32,
    stride_l_m: tl.int32, stride_l_n: tl.int32,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    if pid_m >= M or pid_n >= N:
        return
    acc = tl.zeros((), dtype=tl.float32)
    for k in range(0, K):
        h = tl.load(hidden_ptr + pid_m * stride_h_m + k * stride_h_k)
        w = tl.load(weight_ptr + pid_n * stride_w_n + k * stride_w_k)
        acc += h * w
    tl.store(logits_ptr + pid_m * stride_l_m + pid_n * stride_l_n, acc)


@triton.jit
def sigmoid_add_bias_kernel(
    logits_ptr,      # *f32, [M, N]
    bias_ptr,        # *f32, [N]
    scores_ptr,      # *f32, [M, N]
    M: tl.int32, N: tl.int32,
    stride_l_m: tl.int32, stride_l_n: tl.int32,
    stride_s_m: tl.int32, stride_s_n: tl.int32,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    if pid_m >= M or pid_n >= N:
        return
    logit = tl.load(logits_ptr + pid_m * stride_l_m + pid_n * stride_l_n)
    b = tl.load(bias_ptr + pid_n)
    score = 1.0 / (1.0 + tl.exp(-logit)) + b
    tl.store(scores_ptr + pid_m * stride_s_m + pid_n * stride_s_n, score)


@triton.jit
def group_top2_sum_kernel(
    scores_ptr,      # *f32, [M, N], N = group_count * per_group
    group_scores_ptr,  # *f32, [M, group_count]
    M: tl.int32, group_count: tl.int32, per_group: tl.int32,
):
    pid = tl.program_id(0)
    if pid >= M:
        return
    for g in range(group_count):
        total = tl.zeros((), dtype=tl.float32)
        base = g * per_group
        # find top-2 in this group slice
        max1 = tl.full((), -float('inf'), dtype=tl.float32)
        max2 = tl.full((), -float('inf'), dtype=tl.float32)
        for j in range(per_group):
            idx = base + j
            val = tl.load(scores_ptr + pid * N + idx)
            if val > max1:
                max2 = max1
                max1 = val
            elif val > max2:
                max2 = val
        total = max1 + max2
        tl.store(group_scores_ptr + pid * group_count + g, total)


@triton.jit
def topk_group_kernel(
    group_scores_ptr,     # *f32, [M, group_count]
    selected_idx_ptr,     # *int32, [M, k_group]
    M: tl.int32, k_group: tl.constexpr, group_count: tl.int32,
):
    pid = tl.program_id(0)
    if pid >= M:
        return
    best_vals = tl.full((k_group,), -float('inf'), dtype=tl.float32)
    best_idxs = tl.zeros((k_group,), dtype=tl.int32)
    for g in range(group_count):
        val = tl.load(group_scores_ptr + pid * group_count + g)
        for j in range(k_group):
            if val > best_vals[j]:
                for jj in range(k_group - 1, j, -1):
                    best_vals[jj] = best_vals[jj - 1]
                    best_idxs[jj] = best_idxs[jj - 1]
                best_vals[j] = val
                best_idxs[j] = g
                break
    for t in range(k_group):
        tl.store(selected_idx_ptr + pid * k_group + t, best_idxs[t])


@triton.jit
def build_group_mask_kernel(
    selected_idx_ptr,     # *int32, [M, k_group]
    group_mask_ptr,       # *f32, [M, group_count]
    M: tl.int32, k_group: tl.constexpr, group_count: tl.int32,
):
    pid = tl.program_id(0)
    if pid >= M:
        return
    for g in range(group_count):
        tl.store(group_mask_ptr + pid * group_count + g, 0.0)
    for t in range(k_group):
        g_idx = tl.load(selected_idx_ptr + pid * k_group + t)
        tl.store(group_mask_ptr + pid * group_count + g_idx, 1.0)


@triton.jit
def expand_and_set_ninf_kernel(
    group_mask_ptr,       # *f32, [M, group_count]
    masked_scores_ptr,    # *f32, [M, N]
    M: tl.int32, N: tl.int32, group_count: tl.int32, per_group: tl.int32,
):
    pid = tl.program_id(0)
    if pid >= M:
        return
    for g in range(group_count):
        if tl.load(group_mask_ptr + pid * group_count + g) == 0.0:
            base = g * per_group
            for j in range(per_group):
                idx = base + j
                tl.store(masked_scores_ptr + pid * N + idx, -float('inf'))
        else:
            # keep as-is (already equal to scores)
            pass


@triton.jit
def masked_top8_experts_kernel(
    masked_scores_ptr,    # *f32, [M, N]
    selected_exp_ptr,     # *int32, [M, top_k]
    M: tl.int32, top_k: tl.constexpr, N: tl.int32,
):
    pid = tl.program_id(0)
    if pid >= M:
        return
    best_vals = tl.full((top_k,), -float('inf'), dtype=tl.float32)
    best_idxs = tl.zeros((top_k,), dtype=tl.int32)
    for e in range(N):
        val = tl.load(masked_scores_ptr + pid * N + e)
        for j in range(top_k):
            if val > best_vals[j]:
                for jj in range(top_k - 1, j, -1):
                    best_vals[jj] = best_vals[jj - 1]
                    best_idxs[jj] = best_idxs[jj - 1]
                best_vals[j] = val
                best_idxs[j] = e
                break
    for t in range(top_k):
        tl.store(selected_exp_ptr + pid * top_k + t, best_idxs[t])


@triton.jit
def gather_selected_scores_kernel(
    scores_ptr,           # *f32, [M, N]
    selected_exp_ptr,     # *int32, [M, top_k]
    gathered_ptr,         # *f32, [M, top_k]
    M: tl.int32, top_k: tl.constexpr, N: tl.int32,
):
    pid = tl.program_id(0)
    if pid >= M:
        return
    for t in range(top_k):
        e = tl.load(selected_exp_ptr + pid * top_k + t)
        val = tl.load(scores_ptr + pid * N + e)
        tl.store(gathered_ptr + pid * top_k + t, val)


@triton.jit
def normalize_and_scale_kernel(
    gathered_ptr,         # *f32, [M, top_k]
    output_ptr,           # *f32, [M, top_k]
    scale: tl.float32,
    M: tl.int32, top_k: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= M:
        return
    total = tl.zeros((), dtype=tl.float32)
    for t in range(top_k):
        total += tl.load(gathered_ptr + pid * top_k + t)
    for t in range(top_k):
        val = tl.load(gathered_ptr + pid * top_k + t)
        out = val / (total + 1e-20) * scale
        tl.store(output_ptr + pid * top_k + t, out)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # No parameters; everything is computed in Triton

    def forward(
        self,
        hidden_states: torch.Tensor,
        weight: torch.Tensor,
        expert_bias: torch.Tensor,
        routed_scaling_factor: float,
    ):
        # Ensure device and dtype
        device = hidden_states.device
        M = hidden_states.shape[0]
        K = hidden_states.shape[1]
        N = weight.shape[0]
        assert weight.shape[1] == K, "weight second dim must equal hidden_size"
        assert expert_bias.shape[0] == N, "expert_bias must have size equal to num_experts"
        assert hidden_states.dtype == torch.float32 and weight.dtype == torch.float32 and expert_bias.dtype == torch.float32, "use float32 tensors"

        # 1) Triton matmul: logits = hidden_states @ weight.T
        logits = torch.empty((M, N), dtype=torch.float32, device=device)
        matmul_logits_kernel[(M, N)](
            hidden_states, weight, logits,
            M, N, K,
            hidden_states.stride(0), hidden_states.stride(1),
            weight.stride(0), weight.stride(1),
            logits.stride(0), logits.stride(1),
        )

        # 2) Triton: scores = sigmoid(logits) + expert_bias
        scores = torch.empty((M, N), dtype=torch.float32, device=device)
        sigmoid_add_bias_kernel[(M, N)](
            logits, expert_bias, scores,
            M, N,
            logits.stride(0), logits.stride(1),
            scores.stride(0), scores.stride(1),
        )

        # 3) Triton: group_top2_sum → group_scores [M, 8]
        group_scores = torch.empty((M, GROUP_COUNT), dtype=torch.float32, device=device)
        group_top2_sum_kernel[(M,)](
            scores, group_scores,
            M, GROUP_COUNT, EXP_PER_GROUP,
        )

        # 4) Triton: topk_group_kernel (K=4) → selected group indices [M, 4]
        selected_group_idx = torch.empty((M, K_GROUP), dtype=torch.int32, device=device)
        topk_group_kernel[(M,)](
            group_scores, selected_group_idx,
            M, K_GROUP, GROUP_COUNT,
        )

        # 5) Triton: build_group_mask [M, 8]
        group_mask = torch.empty((M, GROUP_COUNT), dtype=torch.float32, device=device)
        build_group_mask_kernel[(M,)](
            selected_group_idx, group_mask,
            M, K_GROUP, GROUP_COUNT,
        )

        # 6) Triton: expand_and_set_ninf → masked_scores [M, N]
        masked_scores = scores.clone()  # initialize with scores
        expand_and_set_ninf_kernel[(M,)](
            group_mask, masked_scores,
            M, N, GROUP_COUNT, EXP_PER_GROUP,
        )

        # 7) Triton: masked_top8_experts_kernel → final selected expert indices [M, 8]
        final_selected_idx = torch.empty((M, TOP_K), dtype=torch.int32, device=device)
        masked_top8_experts_kernel[(M,)](
            masked_scores, final_selected_idx,
            M, TOP_K, N,
        )

        # 8) Triton: gather_selected_scores from original scores (post-mask)
        gathered_selected_scores = torch.empty((M, TOP_K), dtype=torch.float32, device=device)
        gather_selected_scores_kernel[(M,)](
            scores, final_selected_idx, gathered_selected_scores,
            M, TOP_K, N,
        )

        # 9) Triton: normalize_and_scale_kernel → topk_weight
        topk_weight = torch.empty((M, TOP_K), dtype=torch.float32, device=device)
        normalize_and_scale_kernel[(M,)](
            gathered_selected_scores, topk_weight,
            routed_scaling_factor,
            M, TOP_K,
        )

        # Return indices and weights; convert indices to int64 to match PyTorch behavior
        # The original PyTorch returns (topk_idx, topk_weight) where topk_idx are expert indices [M, 8]
        # We have final_selected_idx [M, 8], and topk_weight [M, 8]
        topk_idx = final_selected_idx.to(torch.int64)
        topk_weight = topk_weight  # already normalized and scaled

        return topk_idx, topk_weight


def run(*args):
    return ModelNew()(*args)
