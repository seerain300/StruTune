import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Constants used by the model
NUM_EXPERTS = 256
N_GROUP = 8
EXP_PER_GROUP = NUM_EXPERTS // N_GROUP  # 32
TOP_K = 8
TOPK_GROUP = 4


# Triton kernel 1: elementwise sigmoid on input tensor X (M, N)
@triton.jit
def _sigmoid_kernel(X_ptr, Y_ptr, M, N):
    pid = tl.program_id(0)  # one program per row
    row = pid
    base = row * N
    cols = tl.arange(0, N)
    mask = cols < N
    x = tl.load(X_ptr + base + cols, mask=mask, other=0.0)
    y = 1.0 / (1.0 + tl.exp(-x))
    tl.store(Y_ptr + base + cols, y, mask=mask)


# Triton kernel 2: add bias vector b of size N to each row of X (M, N) -> Y (M, N)
@triton.jit
def _add_bias_kernel(X_ptr, B_ptr, Y_ptr, M, N):
    pid = tl.program_id(0)  # one program per row
    row = pid
    base_x = row * N
    base_y = row * N
    cols = tl.arange(0, N)
    mask = cols < N
    x = tl.load(X_ptr + base_x + cols, mask=mask, other=0.0)
    b = tl.load(B_ptr + cols, mask=mask, other=0.0)  # bias vector [N]
    y = x + b
    tl.store(Y_ptr + base_y + cols, y, mask=mask)


# Triton kernel 3: compute per-group top-2 sum from scores_for_routing reshaped as [M, N_GROUP, EXP_PER_GROUP]
# Input: S (M, N), Output: group_scores (M, N_GROUP)
@triton.jit
def _group_top2_sum_kernel(S_ptr, G_ptr, M, N, N_GROUP, EXP_PER_GROUP):
    pid = tl.program_id(0)  # one program per token (row)
    row = pid
    base = row * N
    # Iterate groups
    for g in range(N_GROUP):
        group_start = g * EXP_PER_GROUP
        best1 = tl.full((), -1.0e30, tl.float32)
        best2 = tl.full((), -1.0e30, tl.float32)
        # Loop over 32 experts in group
        for i in range(EXP_PER_GROUP):
            idx = group_start + i
            val = tl.load(S_ptr + base + idx)
            if val > best1:
                best2 = best1
                best1 = val
            elif val > best2:
                best2 = val
        tl.store(G_ptr + row * N_GROUP + g, best1 + best2)


# Triton kernel 4: select top-4 groups per token from group_scores (M, N_GROUP)
# Output: group_idx (M, TOPK_GROUP), int32
@triton.jit
def _select_top4_groups_kernel(GS_ptr, IDX_ptr, M, N_GROUP, TOPK_GROUP):
    pid = tl.program_id(0)  # one program per token (row)
    row = pid
    gs_row_ptr = GS_ptr + row * N_GROUP
    idx_row_ptr = IDX_ptr + row * TOPK_GROUP
    # Iteratively find top-4 indices without torch.topk
    top_vals = tl.full((TOPK_GROUP,), -1.0e30, tl.float32)
    top_idx = tl.zeros((TOPK_GROUP,), dtype=tl.int32)
    for g in range(N_GROUP):
        val = tl.load(gs_row_ptr + g)
        # Linear scan insertion; TOPK_GROUP=4 is small
        for i in range(TOPK_GROUP):
            if val > top_vals[i]:
                for j in range(TOPK_GROUP - 1, i, -1):
                    top_vals[j] = top_vals[j - 1]
                    top_idx[j] = top_idx[j - 1]
                top_vals[i] = val
                top_idx[i] = g
                break
    for i in range(TOPK_GROUP):
        tl.store(idx_row_ptr + i, top_idx[i])


# Triton kernel 5: build expert-level mask from group_idx (M, TOPK_GROUP)
# Output: score_mask (M, N), int32 mask 1 for selected groups' 32 experts, 0 otherwise
@triton.jit
def _build_group_mask_from_idx_kernel(
    group_idx_ptr, score_mask_ptr, M, N, N_GROUP, EXP_PER_GROUP, TOPK_GROUP
):
    pid = tl.program_id(0)  # one program per token
    row = pid
    gip_row = group_idx_ptr + row * TOPK_GROUP
    sm_row_base = score_mask_ptr + row * N
    # Iterate over selected groups
    for i in range(TOPK_GROUP):
        g = tl.load(gip_row + i)  # group index in 0..7
        group_start = g * EXP_PER_GROUP
        ones = tl.full((EXP_PER_GROUP,), 1, tl.int32)
        zeros = tl.zeros((EXP_PER_GROUP,), dtype=tl.int32)
        # Write 1s into score_mask[row, group_start:group_start+32], 0 elsewhere
        tl.store(sm_row_base + group_start + tl.arange(0, EXP_PER_GROUP), ones)


# Triton kernel 6: masked fill: if score_mask[row, e] == 0 then set S[row, e] = -inf, else keep
@triton.jit
def _masked_fill_kernel(S_ptr, MASK_ptr, OUT_ptr, M, N, NEG_INF):
    pid = tl.program_id(0)  # one program per row
    row = pid
    s_base = S_ptr + row * N
    out_base = OUT_ptr + row * N
    mask_base = MASK_ptr + row * N
    cols = tl.arange(0, N)
    mask = cols < N
    s = tl.load(s_base + cols, mask=mask, other=0.0)
    m = tl.load(mask_base + cols, mask=mask, other=0)  # int32
    out = tl.where(m != 0, s, NEG_INF)
    tl.store(out_base + cols, out, mask=mask)


# Triton kernel 7: top-8 selection across N experts per token from masked_scores (M, N)
# Output: top8_idx (M, TOP_K) int32, top8_vals (M, TOP_K) float32
@triton.jit
def _top8_indices_kernel(S_ptr, IDX_ptr, VAL_ptr, M, N, TOP_K):
    pid = tl.program_id(0)  # one program per row
    row = pid
    s_base = S_ptr + row * N
    idx_base = IDX_ptr + row * TOP_K
    val_base = VAL_ptr + row * TOP_K
    cols = tl.arange(0, N)
    mask = cols < N
    s = tl.load(s_base + cols, mask=mask, other=-1.0e30)  # load, -inf for oob
    top_vals = tl.full((TOP_K,), -1.0e30, tl.float32)
    top_idx = tl.zeros((TOP_K,), dtype=tl.int32)
    for i in range(N):
        val = s[i]
        for j in range(TOP_K):
            if val > top_vals[j]:
                for k in range(TOP_K - 1, j, -1):
                    top_vals[k] = top_vals[k - 1]
                    top_idx[k] = top_idx[k - 1]
                top_vals[j] = val
                top_idx[j] = i
                break
    for j in range(TOP_K):
        tl.store(idx_base + j, top_idx[j])
        tl.store(val_base + j, top_vals[j])


# Triton kernel 8: normalize selected values and apply scaling factor
# Inputs: top8_vals (M, TOP_K), top8_idx (M, TOP_K), scaling_factor (float32)
# Output: normalized weight (M, TOP_K) scaled by factor
@triton.jit
def _normalize_scale_kernel(VAL_ptr, IDX_ptr, OUT_ptr, M, TOP_K, SCALING):
    pid = tl.program_id(0)  # one program per row
    row = pid
    val_base = VAL_ptr + row * TOP_K
    idx_base = IDX_ptr + row * TOP_K
    out_base = OUT_ptr + row * TOP_K
    total = 0.0
    for i in range(TOP_K):
        v = tl.load(val_base + i)
        total += v
    if total <= 0:
        total = 1.0
    norm_factor = 1.0 / total
    for i in range(TOP_K):
        v = tl.load(val_base + i)
        scaled = v * norm_factor * SCALING
        tl.store(out_base + i, scaled)


class ModelNew(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = NUM_EXPERTS
        self.n_group = N_GROUP
        self.experts_per_group = EXP_PER_GROUP
        self.top_k = TOP_K
        self.topk_group = TOPK_GROUP

    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor, expert_bias: torch.Tensor):
        # Triton-only forward: use PyTorch for initial linear for robustness, Triton for rest.
        if not hidden_states.is_cuda or not weight.is_cuda or not expert_bias.is_cuda:
            raise RuntimeError("Inputs must be CUDA tensors for Triton execution.")
        hidden = hidden_states.contiguous().to(torch.float32)      # [M, K]
        weight = weight.contiguous().to(torch.float32)             # [N, K]
        bias = expert_bias.contiguous().to(torch.float32)          # [N]

        M = hidden.shape[0]
        K = hidden.shape[1]
        N = weight.shape[0]
        assert N == self.num_experts and K == self.hidden_dim, "Shape mismatch: weight must be [256, hidden_dim], hidden [M, hidden_dim]."

        # 1) Compute logits via PyTorch F.linear (FP32) for robustness
        logits = F.linear(hidden, weight)  # [M, N], FP32

        # 2) Sigmoid in Triton
        scores = torch.empty((M, N), dtype=torch.float32, device=hidden.device)
        _sigmoid_kernel[(M,)](logits, scores, M, N, num_warps=4)

        # 3) Add expert bias in Triton
        scores_for_routing = torch.empty((M, N), dtype=torch.float32, device=hidden.device)
        _add_bias_kernel[(M,)](scores, bias, scores_for_routing, M, N, num_warps=4)

        # 4) Group top-2 sum
        group_scores = torch.empty((M, self.n_group), dtype=torch.float32, device=hidden.device)
        _group_top2_sum_kernel[(M,)](scores_for_routing, group_scores, M, N, self.n_group, self.experts_per_group, num_warps=2)

        # 5) Select top-4 groups per token
        group_idx = torch.empty((M, self.topk_group), dtype=torch.int32, device=hidden.device)
        _select_top4_groups_kernel[(M,)](group_scores, group_idx, M, self.n_group, self.topk_group, num_warps=2)

        # 6) Build expert-level mask from group_idx
        score_mask = torch.empty((M, N), dtype=torch.int32, device=hidden.device)
        _build_group_mask_from_idx_kernel[(M,)](group_idx, score_mask, M, N, self.n_group, self.experts_per_group, self.topk_group, num_warps=2)

        # 7) Masked fill: non-selected -> -inf
        masked_scores = torch.empty((M, N), dtype=torch.float32, device=hidden.device)
        _masked_fill_kernel[(M,)](scores_for_routing, score_mask, masked_scores, M, N, NEG_INF=-1.0e20, num_warps=4)

        # 8) Top-8 selection across masked_scores
        top8_idx = torch.empty((M, self.top_k), dtype=torch.int32, device=hidden.device)
        top8_vals = torch.empty((M, self.top_k), dtype=torch.float32, device=hidden.device)
        _top8_indices_kernel[(M,)](masked_scores, top8_idx, top8_vals, M, N, self.top_k, num_warps=4)

        # 9) Normalize and scale
        routed_scaling_factor = 1.0  # default as in original; can be adjusted
        topk_weight = torch.empty((M, self.top_k), dtype=torch.float32, device=hidden.device)
        _normalize_scale_kernel[(M,)](top8_vals, top8_idx, topk_weight, M, self.top_k, routed_scaling_factor, num_warps=2)

        return top8_idx, topk_weight


# Example usage:
# model = ModelNew(hidden_dim=768).cuda()
# hidden_states = torch.randn(2048, 768, device="cuda", dtype=torch.float32)
# weight = torch.randn(256, 768, device="cuda", dtype=torch.float32)
# expert_bias = torch.randn(256, device="cuda", dtype=torch.float32)
# top8_idx, topk_weight = model(hidden_states, weight, expert_bias)


def run(*args):
    return ModelNew()(*args)
