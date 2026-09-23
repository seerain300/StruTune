import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernels
@triton.jit
def _sigmoid_flat_kernel(inp_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(inp_ptr + offs, mask=mask, other=0.0)
    y = 1.0 / (1.0 + tl.exp(-x))
    tl.store(out_ptr + offs, y, mask=mask)


@triton.jit
def _add_bias_kernel(scores_ptr, bias_ptr, out_ptr,
                     M, N,
                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    # Each program handles one row (token)
    row_id = tl.program_id(0)
    if row_id >= M:
        return
    acc = tl.zeros((N,), dtype=tl.float32)
    # iterate over columns in tiles
    for n_start in range(0, N, BLOCK_N):
        col = n_start + tl.arange(0, BLOCK_N)
        # load row tile from scores and bias
        s = tl.load(scores_ptr + row_id * N + col, mask=col < N, other=0.0)
        b = tl.load(bias_ptr + col, mask=col < N, other=0.0)
        acc += s + b
    # store back
    tl.store(out_ptr + row_id * N + tl.arange(0, N), acc, mask=True)


@triton.jit
def _group_top2_sum_kernel(scores_ptr, group_scores_ptr,
                           M, N, EXP_PER_GROUP: tl.constexpr, NUM_GROUPS: tl.constexpr):
    # one program per token
    row_id = tl.program_id(0)
    if row_id >= M:
        return
    # base pointers for this token
    base = scores_ptr + row_id * N
    # compute per-group top2 sum
    for g in range(NUM_GROUPS):
        group_base = base + g * EXP_PER_GROUP
        # compute top-2 within this group of 32
        top1 = -1.0e20
        top2 = -1.0e20
        for e in range(EXP_PER_GROUP):
            val = tl.load(group_base + e)
            if val > top1:
                top2 = top1
                top1 = val
            elif val > top2:
                top2 = val
        group_scores_ptr[row_id * NUM_GROUPS + g] = top1 + top2


@triton.jit
def _select_top4_groups_kernel(group_scores_ptr, group_idx_ptr,
                                M, NUM_GROUPS: tl.constexpr, TOPK_GROUP: tl.constexpr):
    # one program per token
    row_id = tl.program_id(0)
    if row_id >= M:
        return
    gs = tl.load(group_scores_ptr + row_id * NUM_GROUPS + tl.arange(0, NUM_GROUPS))
    # iterative argmax for top4
    # Python loop is allowed here as NUM_GROUPS is constexpr
    for t in range(TOPK_GROUP):
        # find max and its index
        idx = 0
        maxv = -1.0e20
        for i in range(NUM_GROUPS):
            v = gs[i]
            if v > maxv:
                maxv = v
                idx = i
        # mark selected
        tl.store(group_idx_ptr + row_id * TOPK_GROUP + t, idx)
        # zero it for remaining selections
        gs[idx] = -1.0e20


@triton.jit
def _build_group_mask_kernel(group_idx_ptr, score_mask_ptr,
                             M, N, EXP_PER_GROUP: tl.constexpr, NUM_GROUPS: tl.constexpr):
    # one program per token
    row_id = tl.program_id(0)
    if row_id >= M:
        return
    g_idx = tl.load(group_idx_ptr + row_id * NUM_GROUPS + tl.arange(0, NUM_GROUPS))  # [NUM_GROUPS]
    # write mask across N experts
    for e in range(N):
        g = e // EXP_PER_GROUP  # group index for expert e
        # score_mask[row_id, e] = 1 if any of the 4 selected groups == g
        # initialize to 0
        tl.store(score_mask_ptr + row_id * N + e, 0)
        # check each of the 4 selected groups
        for t in range(NUM_GROUPS):
            if g_idx[t] == g:
                tl.store(score_mask_ptr + row_id * N + e, 1)


@triton.jit
def _masked_fill_kernel(scores_ptr, score_mask_ptr, masked_ptr,
                        M, N, NEG_INF: tl.constexpr):
    row_id = tl.program_id(0)
    if row_id >= M:
        return
    for e in range(N):
        mask_val = tl.load(score_mask_ptr + row_id * N + e)
        val = tl.load(scores_ptr + row_id * N + e)
        if mask_val == 0:
            val = NEG_INF
        tl.store(masked_ptr + row_id * N + e, val)


@triton.jit
def _top8_select_kernel(masked_ptr, top_idx_ptr, top_vals_ptr,
                        M, N, TOPK: tl.constexpr):
    row_id = tl.program_id(0)
    if row_id >= M:
        return
    for t in range(TOPK):
        maxv = -1.0e20
        arg = 0
        for e in range(N):
            v = tl.load(masked_ptr + row_id * N + e)
            if v > maxv:
                maxv = v
                arg = e
        # write arg and value
        tl.store(top_idx_ptr + row_id * TOPK + t, arg)
        tl.store(top_vals_ptr + row_id * TOPK + t, maxv)
        # mask it out for next selections
        # (we can write -inf to masked_ptr; here we just avoid considering it next time)
        # Note: Triton loop doesn't mutate the pointer; we re-load masked_ptr next iteration.
        # So no explicit masking needed.


@triton.jit
def _normalize_scale_kernel(top_vals_ptr, out_ptr,
                            M, TOPK: tl.constexpr, eps: tl.constexpr, scale: tl.constexpr):
    row_id = tl.program_id(0)
    if row_id >= M:
        return
    s = 0.0
    for t in range(TOPK):
        v = tl.load(top_vals_ptr + row_id * TOPK + t)
        s += v
    for t in range(TOPK):
        v = tl.load(top_vals_ptr + row_id * TOPK + t)
        v = v / (s + eps) * scale
        tl.store(out_ptr + row_id * TOPK + t, v)


class ModelNew(nn.Module):
    def __init__(self, num_experts: int = 256, hidden_dim: int = None, routed_scaling_factor: float = 1.0):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.n_group = 8
        self.experts_per_group = num_experts // self.n_group  # 32
        self.topk_group = 4
        self.top_k = 8
        self.routed_scaling_factor = float(routed_scaling_factor)

    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor, expert_bias: torch.Tensor):
        # Ensure Triton/CUDA and FP32
        if not TRITON_AVAILABLE or not hidden_states.is_cuda or not weight.is_cuda or not expert_bias.is_cuda:
            raise RuntimeError("Triton is required but not available or tensors are not on CUDA.")
        # Make inputs contiguous and FP32
        hidden = hidden_states.contiguous().to(torch.float32)        # [M, hidden_dim]
        weight = weight.contiguous().to(torch.float32)               # [num_experts, hidden_dim]
        bias = expert_bias.contiguous().to(torch.float32)            # [num_experts]

        M = hidden.shape[0]
        K = hidden.shape[1]
        N = weight.shape[0]
        assert N == self.num_experts and K == self.hidden_dim, "Shape mismatch: weight must be [256, hidden_dim] and hidden [M, hidden_dim]."

        # 1) Linear in torch (robust)
        logits = torch.nn.functional.linear(hidden, weight)  # [M, N], FP32

        # 2) Sigmoid in Triton
        logits_flat = logits.view(-1)                           # [M*N]
        sig_out_flat = torch.empty_like(logits_flat, dtype=torch.float32, device=logits.device)
        BLOCK_SIZE = 1024
        grid_sig = (triton.cdiv(logits_flat.numel(), BLOCK_SIZE),)
        _sigmoid_flat_kernel[grid_sig](logits_flat, sig_out_flat, logits_flat.numel(), BLOCK_SIZE)

        scores = sig_out_flat.view(M, N)

        # 3) Add bias in Triton
        scores_for_routing = torch.empty((M, N), dtype=torch.float32, device=scores.device)
        # Launch add_bias_kernel with reasonable tiles
        BLOCK_M, BLOCK_N = 64, 64
        grid_add = (M,)
        _add_bias_kernel[grid_add](scores, bias, scores_for_routing, M, N, BLOCK_M, BLOCK_N)

        # 4) Group top-2 sum
        group_scores = torch.empty((M, self.n_group), dtype=torch.float32, device=scores.device)
        _group_top2_sum_kernel[(M,)](scores_for_routing, group_scores, M, N, self.experts_per_group, self.n_group)

        # 5) Select top-4 groups
        group_idx = torch.empty((M, self.topk_group), dtype=torch.int32, device=scores.device)
        _select_top4_groups_kernel[(M,)](group_scores, group_idx, M, self.n_group, self.topk_group)

        # 6) Build group mask [M, N]
        score_mask = torch.empty((M, N), dtype=torch.int32, device=scores.device)
        _build_group_mask_kernel[(M,)](group_idx, score_mask, M, N, self.experts_per_group, self.n_group)

        # 7) Masked fill: set non-selected to -inf
        masked_scores = torch.empty((M, N), dtype=torch.float32, device=scores.device)
        NEG_INF = -1.0e20
        _masked_fill_kernel[(M,)](scores_for_routing, score_mask, masked_scores, M, N, NEG_INF)

        # 8) Final top-8 selection
        top8_idx = torch.empty((M, self.top_k), dtype=torch.int32, device=scores.device)
        top8_vals = torch.empty((M, self.top_k), dtype=torch.float32, device=scores.device)
        _top8_select_kernel[(M,)](masked_scores, top8_idx, top8_vals, M, N, self.top_k)

        # 9) Normalize and scale
        topk_weight = torch.empty((M, self.top_k), dtype=torch.float32, device=scores.device)
        eps = 1e-20
        _normalize_scale_kernel[(M,)](top8_vals, topk_weight, M, self.top_k, eps, self.routed_scaling_factor)

        # Return indices and normalized scaled weights
        return top8_idx, topk_weight


def run(*args):
    return ModelNew()(*args)
