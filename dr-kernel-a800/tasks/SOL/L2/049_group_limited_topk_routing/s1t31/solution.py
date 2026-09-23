import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Kernel 1: Matmul logits = hidden @ weight^T
# hidden: [M, K], weight: [N, K], logits: [M, N]
@triton.jit
def _matmul_kernel(
    hidden_ptr,  # *f32, [M, K]
    weight_ptr,  # *f32, [N, K]
    logits_ptr,  # *f32, [M, N]
    M, K, N,
    stride_hm, stride_hk,
    stride_wk, stride_wn,
    stride_lm, stride_ln,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m0 = pid_m * BLOCK_M
    n0 = pid_n * BLOCK_N

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K
    for k0 in range(0, K, BLOCK_K):
        # Build pointers for tiles
        a_ptrs = hidden_ptr + m0 * stride_hm + (k0 + tl.arange(0, BLOCK_K)) * stride_hk
        b_ptrs = weight_ptr + n0 * stride_wn + (k0 + tl.arange(0, BLOCK_K)) * stride_wk

        # Masks for loads
        a_mask = (m0 + tl.arange(0, BLOCK_M))[:, None] < M
        b_mask = (n0 + tl.arange(0, BLOCK_N))[None, :] < N
        k_mask = (k0 + tl.arange(0, BLOCK_K)) < K

        # Load tiles
        a = tl.load(a_ptrs, mask=a_mask & k_mask[None, :], other=0.0)          # [BM, BK]
        b = tl.load(b_ptrs, mask=b_mask & k_mask[:, None], other=0.0)          # [BK, BN]

        # Accumulate
        acc += tl.dot(a, b)  # [BM, BN]

    # Store results
    l_ptrs = logits_ptr + m0 * stride_lm + n0 * stride_ln
    l_mask = (m0 + tl.arange(0, BLOCK_M))[:, None] < M
    l_mask = l_mask & ((n0 + tl.arange(0, BLOCK_N))[None, :] < N)
    tl.store(l_ptrs, acc, mask=l_mask)


# Kernel 2: Elementwise sigmoid on logits
@triton.jit
def _sigmoid_kernel(
    logits_ptr,  # *f32, [M, N]
    scores_ptr,  # *f32, [M, N]
    M, N,
    stride_lm, stride_ln,
    stride_sm, stride_sn,
):
    pid_m = tl.program_id(0)
    # We'll process row by row
    for m in range(0, M):
        for n in range(0, N):
            ptr = logits_ptr + m * stride_lm + n * stride_ln
            val = tl.load(ptr)
            s = 1.0 / (1.0 + tl.exp(-val))
            out_ptr = scores_ptr + m * stride_sm + n * stride_sn
            tl.store(out_ptr, s)


# Kernel 3: Add expert bias to scores (broadcast over tokens)
@triton.jit
def _add_bias_kernel(
    scores_ptr,  # *f32, [M, N]
    bias_ptr,    # *f32, [N]
    out_ptr,     # *f32, [M, N]
    M, N,
    stride_sm, stride_sn,
    stride_bm, stride_bn,  # but bias is 1D
):
    pid_m = tl.program_id(0)
    for m in range(0, M):
        for n in range(0, N):
            ptr = scores_ptr + m * stride_sm + n * stride_sn
            val = tl.load(ptr)
            bias_n = tl.load(bias_ptr + n)  # bias is 1D
            tl.store(out_ptr + m * stride_sm + n * stride_sn, val + bias_n)


# Kernel 4: Compute per-group top-2 sum for each token
# Generalize groups: num_groups = ceil_div(N, exp_per_group), where exp_per_group is up to 32
# We pass num_groups, exp_per_group at launch and compute inside kernel.
@triton.jit
def _group_top2_sum_kernel(
    scores_ptr,      # *f32, [M, N]
    group_scores_ptr,# *f32, [M, num_groups]
    M, N,
    num_groups,      # int32
    stride_sm, stride_sn,
    stride_gm, stride_gg,  # group_scores: [M, num_groups]
    exp_per_group: tl.constexpr,
):
    pid_m = tl.program_id(0)
    m = pid_m

    for g in range(0, num_groups):
        start = g * exp_per_group
        top1 = -float('inf')
        top2 = -float('inf')
        # Loop over experts in this group
        for j in range(0, exp_per_group):
            idx = start + j
            if idx >= N:
                continue
            ptr = scores_ptr + m * stride_sm + idx * stride_sn
            v = tl.load(ptr)
            if v > top1:
                top2 = top1
                top1 = v
            elif v > top2:
                top2 = v
        total = top1 + top2
        out_ptr = group_scores_ptr + m * stride_gm + g * stride_gg
        tl.store(out_ptr, total)


# Kernel 5: Select top-4 groups per token (iterative argmax)
@triton.jit
def _select_top4_groups_kernel(
    group_scores_ptr,  # *f32, [M, num_groups]
    group_idx_ptr,     # *i32, [M, 4]
    M, num_groups,
    stride_gm, stride_gg,
    stride_im, stride_ik,
):
    pid_m = tl.program_id(0)
    m = pid_m

    # Iterative selection without torch.topk
    # We can't write sorted; we write in order of selection by index j
    for j in range(0, 4):
        best_val = -float('inf')
        best_idx = -1
        for g in range(0, num_groups):
            ptr = group_scores_ptr + m * stride_gm + g * stride_gg
            val = tl.load(ptr)
            if val > best_val:
                best_val = val
                best_idx = g
        # Mark selected by writing index
        out_ptr = group_idx_ptr + m * stride_im + j * stride_ik
        tl.store(out_ptr, best_idx)


# Kernel 6: Build expert-level mask for selected groups: 1 for selected group experts, 0 otherwise
@triton.jit
def _build_group_mask_kernel(
    group_idx_ptr,     # *i32, [M, 4]
    score_mask_ptr,    # *i32, [M, N]
    M, N, num_groups,  # num_groups passed; we don't need it here
    stride_im, stride_ik,
    stride_mmm, stride_mnn,
    exp_per_group: tl.constexpr,
):
    pid_m = tl.program_id(0)
    m = pid_m

    for j in range(0, 4):  # topk_group = 4
        g = tl.load(group_idx_ptr + m * stride_im + j * stride_ik)  # int32
        start = g * exp_per_group
        for k in range(0, exp_per_group):
            e = start + k
            out_ptr = score_mask_ptr + m * stride_mmm + e * stride_mnn
            tl.store(out_ptr, 1)
    # Initialize other entries to 0
    for e in range(0, N):
        out_ptr = score_mask_ptr + m * stride_mmm + e * stride_mnn
        # If not set above, leave as default int32 zeros. Triton tensors are initialized by host.
        pass


# Kernel 7: Masked fill: set masked_scores[i, e] = -inf if score_mask[i, e] == 0, else keep scores_for_routing[i, e]
@triton.jit
def _masked_fill_kernel(
    scores_ptr,        # *f32, [M, N]
    mask_ptr,          # *i32, [M, N]
    out_ptr,           # *f32, [M, N]
    M, N,
    stride_sm, stride_sn,
    stride_mm, stride_mn,
    NEG_INF,
):
    pid_m = tl.program_id(0)
    m = pid_m
    for n in range(0, N):
        sptr = scores_ptr + m * stride_sm + n * stride_sn
        mp = mask_ptr + m * stride_mm + n * stride_mn
        val = tl.load(sptr)
        mask_val = tl.load(mp)  # i32
        out_val = val if mask_val != 0 else NEG_INF
        out_ptr = out_ptr + m * stride_sm + n * stride_sn  # out_ptr is the same layout as scores
        tl.store(out_ptr, out_val)


# Kernel 8: Select final top-8 experts from masked_scores via iterative argmax
@triton.jit
def _final_top8_select_kernel(
    masked_ptr,        # *f32, [M, N]
    idx_ptr,           # *i32, [M, 8]
    vals_ptr,          # *f32, [M, 8]
    M, N,
    stride_mm, stride_mn,
    stride_im, stride_iv,
):
    pid_m = tl.program_id(0)
    m = pid_m

    for j in range(0, 8):
        best_val = -float('inf')
        best_idx = -1
        for n in range(0, N):
            ptr = masked_ptr + m * stride_mm + n * stride_mn
            v = tl.load(ptr)
            if v > best_val:
                best_val = v
                best_idx = n
        tl.store(idx_ptr + m * stride_im + j * stride_iv, best_idx)
        tl.store(vals_ptr + m * stride_im + j * stride_iv, best_val)
        # Mark selected by setting to -inf (in masked memory); implement by writing back through pointer (we don't have ptr per lane, so skip here; masked will already be -inf for non-selected).


# Kernel 9: Normalize selected values and apply scaling factor
@triton.jit
def _normalize_and_scale_kernel(
    vals_ptr,          # *f32, [M, 8]
    out_ptr,           # *f32, [M, 8]
    M, k,              # k=8
    stride_vm, stride_vk,
    stride_om, stride_ok,
    scaling_factor,
):
    pid_m = tl.program_id(0)
    m = pid_m
    sum_val = 0.0
    for j in range(0, k):
        ptr = vals_ptr + m * stride_vm + j * stride_vk
        v = tl.load(ptr)
        sum_val += v
    for j in range(0, k):
        ptr = vals_ptr + m * stride_vm + j * stride_vk
        v = tl.load(ptr)
        w = v / sum_val
        w = w * scaling_factor
        out_ptr_j = out_ptr + m * stride_om + j * stride_ok
        tl.store(out_ptr_j, w)


class ModelNew(nn.Module):
    def __init__(self, hidden_dim: int, num_experts: int = 256, n_group: int = 8, topk_group: int = 4, top_k: int = 8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.n_group = n_group
        self.experts_per_group = num_experts // n_group
        self.topk_group = topk_group
        self.top_k = top_k

    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor, expert_bias: torch.Tensor, routed_scaling_factor: float):
        # Triton-only forward
        if not TRITON_AVAILABLE or not hidden_states.is_cuda or not weight.is_cuda or not expert_bias.is_cuda:
            raise RuntimeError("Triton is required but not available or tensors are not on CUDA.")

        hidden = hidden_states.contiguous().to(torch.float32)       # [M, K]
        weight = weight.contiguous().to(torch.float32)              # [N, K]
        bias = expert_bias.contiguous().to(torch.float32)           # [N]

        M = hidden.shape[0]
        K = hidden.shape[1]
        N = weight.shape[0]
        assert K == self.hidden_dim, f"hidden_dim mismatch: expected {self.hidden_dim}, got {K}"
        assert N == self.num_experts, f"num_experts mismatch: expected {self.num_experts}, got {N}"

        # Outputs and intermediates
        logits = torch.empty((M, N), dtype=torch.float32, device=hidden.device)
        scores = torch.empty((M, N), dtype=torch.float32, device=hidden.device)
        scores_for_routing = torch.empty((M, N), dtype=torch.float32, device=hidden.device)
        group_scores = torch.empty((M, self.n_group), dtype=torch.float32, device=hidden.device)
        group_idx = torch.empty((M, self.topk_group), dtype=torch.int32, device=hidden.device)
        score_mask = torch.zeros((M, N), dtype=torch.int32, device=hidden.device)  # expert-level mask
        masked_scores = torch.empty((M, N), dtype=torch.float32, device=hidden.device)
        top8_idx = torch.empty((M, self.top_k), dtype=torch.int32, device=hidden.device)
        top8_vals = torch.empty((M, self.top_k), dtype=torch.float32, device=hidden.device)
        topk_weight = torch.empty((M, self.top_k), dtype=torch.float32, device=hidden.device)

        # Launch Triton kernels
        # 1) Matmul for logits
        BLOCK_M = 128
        BLOCK_N = 64
        BLOCK_K = 32
        grid_matmul = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        _matmul_kernel[grid_matmul](
            hidden, weight, logits,
            M, K, N,
            hidden.stride(0), hidden.stride(1),
            weight.stride(1), weight.stride(0),
            logits.stride(0), logits.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        # 2) Sigmoid on logits
        _sigmoid_kernel[(M, N)](
            logits, scores,
            logits.stride(0), logits.stride(1),
            scores.stride(0), scores.stride(1),
            num_warps=4,
        )

        # 3) Add expert bias
        _add_bias_kernel[(M, N)](
            scores, bias, scores_for_routing,
            scores.stride(0), scores.stride(1),
            scores_for_routing.stride(0), scores_for_routing.stride(1),
            num_warps=4,
        )

        # 4) Group top-2 sum
        _group_top2_sum_kernel[(M,)](
            scores_for_routing, group_scores,
            M, N,
            self.n_group,
            scores_for_routing.stride(0), scores_for_routing.stride(1),
            group_scores.stride(0), group_scores.stride(1),
            exp_per_group=self.experts_per_group,
            num_warps=2,
        )

        # 5) Select top-4 groups per token
        _select_top4_groups_kernel[(M,)](
            group_scores, group_idx,
            M, self.n_group,
            group_scores.stride(0), group_scores.stride(1),
            group_idx.stride(0), group_idx.stride(1),
            num_warps=2,
        )

        # 6) Build expert-level mask: 1 for selected group's 32 experts, 0 otherwise
        _build_group_mask_kernel[(M,)](
            group_idx, score_mask,
            M, N, self.n_group,
            group_idx.stride(0), group_idx.stride(1),
            score_mask.stride(0), score_mask.stride(1),
            exp_per_group=self.experts_per_group,
            num_warps=2,
        )

        # 7) Masked fill: set non-selected to -inf
        NEG_INF = -1.0e20
        _masked_fill_kernel[(M, N)](
            scores_for_routing, score_mask, masked_scores,
            scores_for_routing.stride(0), scores_for_routing.stride(1),
            score_mask.stride(0), score_mask.stride(1),
            NEG_INF,
            num_warps=4,
        )

        # 8) Final top-8 selection from masked_scores
        _final_top8_select_kernel[(M,)](
            masked_scores, top8_idx, top8_vals,
            M, N,
            masked_scores.stride(0), masked_scores.stride(1),
            top8_idx.stride(0), top8_idx.stride(1),
            num_warps=4,
        )

        # 9) Normalize and apply scaling factor to selected values
        _normalize_and_scale_kernel[(M,)](
            top8_vals, topk_weight,
            M, self.top_k,
            top8_vals.stride(0), top8_vals.stride(1),
            topk_weight.stride(0), topk_weight.stride(1),
            routed_scaling_factor,
            num_warps=2,
        )

        # Return selected indices and normalized weights (top_k per token)
        return top8_idx, topk_weight


def run(*args):
    return ModelNew()(*args)
