import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Kernel 1: Triton matmul to compute logits = hidden @ weight^T
# hidden: [M, K], weight: [N, K], out: [M, N]
@triton.jit
def _matmul_kernel(
    hidden_ptr,        # *f32, [M, K]
    weight_ptr,        # *f32, [N, K]
    out_ptr,           # *f32, [M, N]
    M, K, N,           # int32 sizes
    stride_hm, stride_hk,   # strides for hidden
    stride_wk, stride_wn,   # strides for weight
    stride_om, stride_on,   # strides for out
    BLOCK_M: tl.constexpr,  # tile sizes
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        # hidden block: [BLOCK_M, BLOCK_K]
        h_ptrs = hidden_ptr + (offs_m[:, None] * stride_hm + offs_k[None, :] * stride_hk)
        h_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        h = tl.load(h_ptrs, mask=h_mask, other=0.0)

        # weight block: [BLOCK_K, BLOCK_N]
        w_ptrs = weight_ptr + (offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn)
        w_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)

        acc += tl.dot(h, w)

    out_ptrs = out_ptr + (offs_m[:, None] * stride_om + offs_n[None, :] * stride_on)
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, acc, mask=out_mask)


# Kernel 2: elementwise sigmoid on out_ptr, write to dst_ptr
@triton.jit
def _sigmoid_kernel(out_ptr, dst_ptr, M, N,
                    stride_out, stride_dst):
    pid = tl.program_id(0)
    total = M * N
    offsets = pid * 64 + tl.arange(0, 64)
    mask = offsets < total
    row = offsets // N
    col = offsets % N
    ptrs = out_ptr + row * stride_out + col
    x = tl.load(ptrs, mask=mask, other=0.0)
    s = 1.0 / (1.0 + tl.exp(-x))
    dst_ptrs = dst_ptr + row * stride_dst + col
    tl.store(dst_ptrs, s, mask=mask)


# Kernel 3: add bias (length N) to each column of scores, write to scores_for_routing
@triton.jit
def _add_bias_kernel(scores_ptr, bias_ptr, dst_ptr,
                     M, N,
                     stride_s, stride_d, stride_b):
    pid = tl.program_id(0)
    total = M * N
    offsets = pid * 64 + tl.arange(0, 64)
    mask = offsets < total
    row = offsets // N
    col = offsets % N
    s_ptrs = scores_ptr + row * stride_s + col
    s = tl.load(s_ptrs, mask=mask, other=0.0)
    b_ptrs = bias_ptr + col * stride_b
    b = tl.load(b_ptrs, mask=mask, other=0.0)
    tl.store(dst_ptr + row * stride_d + col, s + b, mask=mask)


# Kernel 4: group-top-2 sum per group
# scores_for_routing: [M, N], group_scores: [M, n_group]
# Assumes N = n_group * experts_per_group (here 256 = 8 * 32)
@triton.jit
def _group_top2_sum_kernel(scores_ptr, group_scores_ptr,
                           M, N, n_group,
                           stride_s0, stride_s1,
                           stride_gs0, stride_gs1,
                           experts_per_group: tl.constexpr):
    pid = tl.program_id(0)
    # Each program handles one token
    total = M  # grid is (M,)
    # Loop over groups
    for g in range(n_group):
        start_expert = g * experts_per_group
        # Initialize top2 with the first two values
        top1 = -1.0e20
        top1_idx = 0
        top2 = -1.0e20
        top2_idx = 0
        for i in range(experts_per_group):
            e = start_expert + i
            ptr = scores_ptr + pid * stride_s0 + e * stride_s1
            v = tl.load(ptr)
            if v > top1:
                top2 = top1
                top2_idx = top1_idx
                top1 = v
                top1_idx = i
            elif v > top2:
                top2 = v
                top2_idx = i
        group_scores_ptr[pid * stride_gs0 + g * stride_gs1] = top1 + top2


# Kernel 5: select top-4 groups per token, write indices [M, 4]
@triton.jit
def _select_top4_groups_kernel(group_scores_ptr, group_idx_ptr,
                                M, n_group,
                                stride_gs0, stride_gs1,
                                stride_gi0, stride_gi1):
    pid = tl.program_id(0)
    total = M  # grid is (M,)
    # Find top4 by repeated argmax
    for r in range(4):
        maxv = -1.0e20
        pos = -1
        for g in range(n_group):
            v = group_scores_ptr[pid * stride_gs0 + g * stride_gs1]
            if v > maxv:
                maxv = v
                pos = g
        # store pos
        tl.store(group_idx_ptr + pid * stride_gi0 + r * stride_gi1, pos)
        # mark as used (set to -inf)
        group_scores_ptr[pid * stride_gs0 + pos * stride_gs1] = -1.0e20


# Kernel 6: build expert-level mask from selected group indices
# score_mask: [M, N], int32 0/1
@triton.jit
def _build_group_mask_kernel(group_idx_ptr, score_mask_ptr,
                             M, N,
                             stride_gi0, stride_gi1,
                             stride_sm0, stride_sm1,
                             n_group: tl.constexpr,
                             experts_per_group: tl.constexpr):
    pid = tl.program_id(0)
    # Each program handles one token
    total = M  # grid is (M,)
    for g in range(n_group):
        idx = tl.load(group_idx_ptr + pid * stride_gi0 + g * stride_gi1)  # int32
        start_expert = g * experts_per_group
        for i in range(experts_per_group):
            e = start_expert + i
            ptr = score_mask_ptr + pid * stride_sm0 + e * stride_sm1
            tl.store(ptr, 1)


# Kernel 7: masked fill: set non-selected to -inf
@triton.jit
def _masked_fill_kernel(scores_ptr, score_mask_ptr, masked_ptr,
                        M, N,
                        stride_sp, stride_msp, stride_mp):
    pid = tl.program_id(0)
    total = M * N
    offsets = pid * 64 + tl.arange(0, 64)
    mask = offsets < total
    row = offsets // N
    col = offsets % N
    s = tl.load(scores_ptr + row * stride_sp + col * stride_sp, mask=mask, other=0.0)  # NOTE: using sp twice, intended to be scores_ptr
    m = tl.load(score_mask_ptr + row * stride_msp + col * stride_msp, mask=mask, other=0.0)  # int32
    neg_inf = -1.0e20
    out = tl.where(m != 0, s, neg_inf)
    tl.store(masked_ptr + row * stride_mp + col * stride_mp, out, mask=mask)


# Kernel 8: final top-8 selection from masked_scores
# Returns indices and values in top8_idx [M, 8], top8_vals [M, 8]
@triton.jit
def _final_top8_kernel(masked_ptr, top8_idx_ptr, top8_vals_ptr,
                       M, N,
                       stride_ms, stride_tmi, stride_tmv):
    pid = tl.program_id(0)
    total = M  # grid is (M,)
    for r in range(8):
        maxv = -1.0e20
        pos = -1
        for e in range(N):
            ptr = masked_ptr + pid * stride_ms + e * stride_ms
            v = tl.load(ptr)
            if v > maxv:
                maxv = v
                pos = e
        tl.store(top8_idx_ptr + pid * stride_tmi + r * stride_tmi, pos)
        tl.store(top8_vals_ptr + pid * stride_tmv + r * stride_tmv, maxv)


# Kernel 9: normalize selected vals and apply scaling
@triton.jit
def _normalize_scale_kernel(vals_ptr, topk_weight_ptr,
                            M, K,  # K is top_k
                            stride_vp, stride_tw):
    pid = tl.program_id(0)
    total = M  # grid is (M,)
    # sum of first K values
    s = 0.0
    for r in range(K):
        v = tl.load(vals_ptr + pid * stride_vp + r * stride_vp)
        s += v
    eps = 1e-20
    denom = s + eps
    scale = 1.0
    # Apply scaling factor if provided (here we assume routed_scaling_factor is passed via vals_ptr's dtype or constexpr),
    # but since it's a scalar, we keep it as a constexpr parameter. We'll pass it at launch.
    # For simplicity, treat routed_scaling_factor as a scalar from host.
    # We'll implement as: normalize then scale by scale_val (float32 scalar)
    # However, since Triton kernel parameters don't allow per-token scale, we'll pass it as a pointer to scalar
    # Here we assume host passes routed_scaling_factor as a scalar argument; Triton allows scalar kernel args.
    # We'll define scale_val as a scalar argument next.
    scale_val = 1.0  # placeholder; host will set this via kernel call
    # Now write normalized and scaled
    for r in range(K):
        v = tl.load(vals_ptr + pid * stride_vp + r * stride_vp)
        w = (v / denom) * scale_val
        tl.store(topk_weight_ptr + pid * stride_tw + r * stride_tw, w)


class ModelNew(nn.Module):
    def __init__(self, hidden_dim: int = 768, routed_scaling_factor: float = 1.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = 256
        self.n_group = 8
        self.experts_per_group = self.num_experts // self.n_group  # 32
        self.top_k = 8
        self.topk_group = 4
        self.routed_scaling_factor = routed_scaling_factor

    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor, expert_bias: torch.Tensor):
        # Triton-only forward
        if not TRITON_AVAILABLE or not hidden_states.is_cuda or not weight.is_cuda or not expert_bias.is_cuda:
            raise RuntimeError("Triton is required but not available or tensors are not on CUDA.")

        hidden = hidden_states.contiguous().to(torch.float32)       # [M, K]
        weight = weight.contiguous().to(torch.float32)              # [N, K]
        bias = expert_bias.contiguous().to(torch.float32)           # [N]

        M = hidden.shape[0]
        K = hidden.shape[1]
        N = weight.shape[0]
        assert N == self.num_experts and K == self.hidden_dim, "Shape mismatch: weight must be [256, hidden_dim] and hidden [M, hidden_dim]."

        # Outputs and intermediates
        logits = torch.empty((M, N), dtype=torch.float32, device=hidden.device)
        scores = torch.empty((M, N), dtype=torch.float32, device=hidden.device)
        scores_for_routing = torch.empty((M, N), dtype=torch.float32, device=hidden.device)
        group_scores = torch.empty((M, self.n_group), dtype=torch.float32, device=hidden.device)
        group_idx = torch.empty((M, self.topk_group), dtype=torch.int32, device=hidden.device)
        score_mask = torch.empty((M, N), dtype=torch.int32, device=hidden.device)  # expert-level mask
        masked_scores = torch.empty((M, N), dtype=torch.float32, device=hidden.device)
        top8_idx = torch.empty((M, self.top_k), dtype=torch.int32, device=hidden.device)
        top8_vals = torch.empty((M, self.top_k), dtype=torch.float32, device=hidden.device)
        topk_weight = torch.empty((M, self.top_k), dtype=torch.float32, device=hidden.device)

        # 1) Triton matmul for logits
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 64
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        _matmul_kernel[grid](
            hidden, weight, logits,
            M, K, N,
            hidden.stride(0), hidden.stride(1),
            weight.stride(1), weight.stride(0),  # weight stride: (K, N) => (stride_wn, stride_wk) = weight.stride(1), weight.stride(0)
            logits.stride(0), logits.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4,
        )

        # 2) Sigmoid activation (Triton)
        _sigmoid_kernel[(M,)](
            logits, scores,
            M, N,
            logits.stride(0), scores.stride(0),
            num_warps=4,
        )

        # 3) Add expert bias (Triton)
        _add_bias_kernel[(M,)](
            scores, bias, scores_for_routing,
            M, N,
            scores.stride(0), scores_for_routing.stride(0), bias.stride(0),
            num_warps=4,
        )

        # 4) Group top-2 sum per group (Triton)
        _group_top2_sum_kernel[(M,)](
            scores_for_routing, group_scores,
            M, N, self.n_group,
            scores_for_routing.stride(0), scores_for_routing.stride(1),
            group_scores.stride(0), group_scores.stride(1),
            self.experts_per_group,
            num_warps=1,
        )

        # 5) Select top-4 groups (Triton)
        _select_top4_groups_kernel[(M,)](
            group_scores, group_idx,
            M, self.n_group,
            group_scores.stride(0), group_scores.stride(1),
            group_idx.stride(0), group_idx.stride(1),
            num_warps=1,
        )

        # 6) Build expert-level mask (Triton)
        _build_group_mask_kernel[(M,)](
            group_idx, score_mask,
            M, N,
            group_idx.stride(0), group_idx.stride(1),
            score_mask.stride(0), score_mask.stride(1),
            self.n_group,
            self.experts_per_group,
            num_warps=1,
        )

        # 7) Masked fill (Triton)
        _masked_fill_kernel[(M,)](
            scores_for_routing, score_mask, masked_scores,
            M, N,
            scores_for_routing.stride(0), score_mask.stride(0), masked_scores.stride(0),
            num_warps=4,
        )

        # 8) Final top-8 selection (Triton)
        _final_top8_kernel[(M,)](
            masked_scores, top8_idx, top8_vals,
            M, N,
            masked_scores.stride(0), top8_idx.stride(0), top8_vals.stride(0),
            num_warps=1,
        )

        # 9) Normalize and scale (Triton). Note: routed_scaling_factor is a scalar; pass via host.
        _normalize_scale_kernel[(M,)](
            top8_vals, topk_weight,
            M, self.top_k,
            top8_vals.stride(0), topk_weight.stride(0),
            scale_val=self.routed_scaling_factor,
            num_warps=1,
        )

        # Return indices and weights
        return top8_idx, topk_weight


def run(*args):
    return ModelNew()(*args)
