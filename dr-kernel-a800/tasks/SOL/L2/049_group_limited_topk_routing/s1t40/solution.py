import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: elementwise sigmoid
@triton.jit
def _sigmoid_kernel(scores_ptr, out_ptr, M, N):
    pid = tl.program_id(0)
    row = pid
    if row >= M:
        return
    # iterate columns
    for j in range(0, N):
        x = tl.load(scores_ptr + row * N + j)
        y = 1.0 / (1.0 + tl.exp(-x))
        tl.store(out_ptr + row * N + j, y)


# Triton kernel: add bias vector (size N) to each column
@triton.jit
def _add_bias_kernel(scores_ptr, bias_ptr, out_ptr, M, N):
    pid = tl.program_id(0)
    row = pid
    if row >= M:
        return
    for j in range(0, N):
        x = tl.load(scores_ptr + row * N + j)
        b = tl.load(bias_ptr + j)
        tl.store(out_ptr + row * N + j, x + b)


# Triton kernel: group top-2 sum per token
# scores_for_routing: [M, N], N=num_experts=256, reshape conceptually to [M, 8, 32]
# out: group_scores [M, 8], each is sum of top-2 per group
@triton.jit
def _group_top2_sum_kernel(s_ptr, out_ptr, M, N, n_group, experts_per_group):
    pid = tl.program_id(0)
    row = pid
    if row >= M:
        return
    # Loop over groups
    for g in range(0, n_group):
        base = g * experts_per_group
        max1 = -float("inf")
        max2 = -float("inf")
        # iterate experts in this group
        for j in range(0, experts_per_group):
            val = tl.load(s_ptr + row * N + base + j)
            if val > max1:
                max2 = max1
                max1 = val
            elif val > max2:
                max2 = val
        tl.store(out_ptr + row * n_group + g, max1 + max2)


# Triton kernel: select top-4 groups per token
# group_scores: [M, n_group], out group_idx [M, 4]
@triton.jit
def _select_top4_groups_kernel(group_scores_ptr, group_idx_ptr, M, n_group):
    pid = tl.program_id(0)
    row = pid
    if row >= M:
        return
    idxs = tl.zeros((4,), dtype=tl.int32)
    vals = tl.zeros((4,), dtype=tl.float32)
    for i in range(0, 4):
        best = -float("inf")
        chosen = -1
        # scan all groups to find the i-th best
        for g in range(0, n_group):
            score = tl.load(group_scores_ptr + row * n_group + g)
            if score > best and (i == 0 or score > vals[0]):
                # ensure we don't pick previously picked groups
                picked = False
                for j in range(0, i):
                    if g == idxs[j]:
                        picked = True
                        break
                if not picked:
                    best = score
                    chosen = g
        idxs[i] = chosen
        vals[i] = best
    # store idxs
    for i in range(0, 4):
        tl.store(group_idx_ptr + row * 4 + i, idxs[i])


# Triton kernel: build expert-level score_mask given group_idx [M, 4]
# score_mask: [M, N], set 1 at selected groups' 32 experts, 0 otherwise
@triton.jit
def _build_group_mask_kernel(group_idx_ptr, score_mask_ptr, M, N, n_group, experts_per_group):
    pid = tl.program_id(0)
    row = pid
    if row >= M:
        return
    # initialize mask to zeros
    for j in range(0, N):
        tl.store(score_mask_ptr + row * N + j, 0)
    # for each selected group, set its 32 experts to 1
    for i in range(0, 4):  # we only select top-4 groups
        g = tl.load(group_idx_ptr + row * 4 + i)
        if g >= 0 and g < n_group:
            base = g * experts_per_group
            for j in range(0, experts_per_group):
                tl.store(score_mask_ptr + row * N + base + j, 1)


# Triton kernel: masked fill: if score_mask == 0 -> set to -inf, else keep
@triton.jit
def _masked_fill_kernel(scores_ptr, score_mask_ptr, masked_ptr, M, N, NEG_INF):
    pid = tl.program_id(0)
    row = pid
    if row >= M:
        return
    for j in range(0, N):
        s = tl.load(scores_ptr + row * N + j)
        m = tl.load(score_mask_ptr + row * N + j)  # int32 0/1
        val = tl.where(m == 1, s, NEG_INF)
        tl.store(masked_ptr + row * N + j, val)


# Triton kernel: final top-8 selection from masked_scores
@triton.jit
def _final_top8_kernel(masked_ptr, top_idx_ptr, top_vals_ptr, M, N, K):
    pid = tl.program_id(0)
    row = pid
    if row >= M:
        return
    best = tl.zeros((K,), dtype=tl.float32)
    chosen = tl.zeros((K,), dtype=tl.int32)
    for i in range(0, K):
        best_i = -float("inf")
        chosen_i = -1
        for j in range(0, N):
            v = tl.load(masked_ptr + row * N + j)
            if v > best_i:
                best_i = v
                chosen_i = j
        best[i] = best_i
        chosen[i] = chosen_i
        # mark chosen as -inf for next picks
        tl.store(masked_ptr + row * N + chosen_i, -float("inf"))
    # store results
    for i in range(0, K):
        tl.store(top_idx_ptr + row * K + i, chosen[i])
        tl.store(top_vals_ptr + row * K + i, best[i])


# Triton kernel: normalize top8_vals and apply scaling factor
@triton.jit
def _normalize_and_scale_kernel(vals_ptr, out_ptr, M, K, SCALE):
    pid = tl.program_id(0)
    row = pid
    if row >= M:
        return
    total = 0.0
    for i in range(0, K):
        v = tl.load(vals_ptr + row * K + i)
        total += v
    total = total + 1e-20  # epsilon
    for i in range(0, K):
        v = tl.load(vals_ptr + row * K + i)
        nv = v / total
        nv = nv * SCALE
        tl.store(out_ptr + row * K + i, nv)


class ModelNew(nn.Module):
    """
    Triton-optimized version that performs all computations in Triton kernels.
    Host code only allocates tensors and launches Triton kernels. No torch ops in host beyond F.linear.
    """
    def __init__(self, hidden_dim: int, routed_scaling_factor: float):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = 256
        self.n_group = 8
        self.experts_per_group = self.num_experts // self.n_group  # 32
        self.topk_group = 4
        self.top_k = 8
        self.routed_scaling_factor = float(routed_scaling_factor)

    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor, expert_bias: torch.Tensor):
        # Ensure Triton availability and CUDA tensors
        if not TRITON_AVAILABLE or not hidden_states.is_cuda or not weight.is_cuda or not expert_bias.is_cuda:
            raise RuntimeError("Triton is required but not available or tensors are not on CUDA.")

        # Compute logits via PyTorch (fast and correct), host code does not perform elementwise math
        logits = F.linear(hidden_states.to(torch.float32), weight.to(torch.float32))  # [M, N], N=256
        M, N = logits.shape
        assert N == self.num_experts, f"Expected {self.num_experts} experts, got {N}"

        # Prepare outputs
        scores = torch.empty((M, N), dtype=torch.float32, device=logits.device)
        scores_for_routing = torch.empty((M, N), dtype=torch.float32, device=logits.device)
        group_scores = torch.empty((M, self.n_group), dtype=torch.float32, device=logits.device)
        group_idx = torch.empty((M, self.topk_group), dtype=torch.int32, device=logits.device)
        score_mask = torch.empty((M, N), dtype=torch.int32, device=logits.device)  # 0/1 mask
        masked_scores = torch.empty((M, N), dtype=torch.float32, device=logits.device)
        top8_idx = torch.empty((M, self.top_k), dtype=torch.int32, device=logits.device)
        top8_vals = torch.empty((M, self.top_k), dtype=torch.float32, device=logits.device)
        topk_weight = torch.empty((M, self.top_k), dtype=torch.float32, device=logits.device)

        # Launch Triton kernels
        # 1) Sigmoid
        _sigmoid_kernel[(M,)](logits, scores, M, N)

        # 2) Add bias
        _add_bias_kernel[(M,)](scores, expert_bias, scores_for_routing, M, N)

        # 3) Group top-2 sum
        _group_top2_sum_kernel[(M,)](scores_for_routing, group_scores, M, N, self.n_group, self.experts_per_group)

        # 4) Select top-4 groups
        _select_top4_groups_kernel[(M,)](group_scores, group_idx, M, self.n_group)

        # 5) Build group mask
        _build_group_mask_kernel[(M,)](group_idx, score_mask, M, N, self.n_group, self.experts_per_group)

        # 6) Masked fill
        NEG_INF = -1.0e20
        _masked_fill_kernel[(M,)](scores_for_routing, score_mask, masked_scores, M, N, NEG_INF)

        # 7) Final top-8 selection
        _final_top8_kernel[(M,)](masked_scores, top8_idx, top8_vals, M, N, self.top_k)

        # 8) Normalize and scale
        _normalize_and_scale_kernel[(M,)](top8_vals, topk_weight, M, self.top_k, self.routed_scaling_factor)

        # Return indices and normalized weights
        return top8_idx, topk_weight


def run(*args):
    return ModelNew()(*args)
