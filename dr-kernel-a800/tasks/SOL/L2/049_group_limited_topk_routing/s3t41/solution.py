import torch
import triton
import triton.language as tl


@triton.jit
def _matmul_kernel(
    A_ptr,  # [M, K] = hidden, float32, contiguous
    B_ptr,  # [K, N] = weight.T, float32, contiguous
    C_ptr,  # [M, N] = logits, float32, contiguous
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # 2D launch: one program per tile
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    k = 0
    while k < K:
        a_ptrs = A_ptr + (offs_m[:, None] * stride_am + (k + offs_k[None, :]) * stride_ak)
        b_ptrs = B_ptr + ((k + offs_k[:, None]) * stride_bk + offs_n[None, :] * stride_bn)
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (k + offs_k[None, :] < K),
                    other=0.0)
        b = tl.load(b_ptrs, mask=(k + offs_k[:, None] < K) & (offs_n[None, :] < N),
                    other=0.0)
        acc += tl.dot(a, b)
        k += BLOCK_K

    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


@triton.jit
def _sigmoid_bias_kernel(
    X_ptr,   # [M, N] logits
    Bias_ptr, # [N] float32
    Y_ptr,   # [M, N] sigmoid + bias
    M, N,
    stride_xm, stride_xn,
    stride_ym, stride_yn,
    stride_b,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * 64 + tl.arange(0, 64)
    offs_n = pid_n * 64 + tl.arange(0, 64)
    x = tl.load(
        X_ptr + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
        other=0.0,
    )
    b = tl.load(Bias_ptr + offs_n * stride_b, mask=offs_n < N, other=0.0)  # [N]
    y = 1.0 / (1.0 + tl.exp(-x)) + b  # broadcast b over rows
    tl.store(
        Y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn,
        y,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


@triton.jit
def _group_top2_and_select_kernel(
    SCORES_ptr,          # [M, 256] float32
    GROUPSCORE_ptr,      # [M, 8] float32
    GROUP_IDX_ptr,       # [M, 4] int32
    M, NUM_EXPERTS,
    stride_sm, stride_sn,
    stride_gsm, stride_gsn,
    stride_gipm, stride_gin,
):
    pid_m = tl.program_id(0)

    # For each group g=0..7: compute top-2 within [g*32 .. (g+1)*32] and accumulate
    # Also track top-4 groups based on accumulated score
    for g in range(0, 8):
        start = g * 32
        # Initialize top-2 for this group
        t1 = tl.full((), -1e20, tl.float32)
        t2 = tl.full((), -1e20, tl.float32)
        # Compute top-2 among 32 experts
        for e in range(0, 32):
            e_abs = start + e
            score = tl.load(SCORES_ptr + pid_m * stride_sm + e_abs * stride_sn)
            if score > t1:
                t2 = t1
                t1 = score
            elif score > t2:
                t2 = score
        group_score = t1 + t2
        tl.store(GROUPSCORE_ptr + pid_m * stride_gsm + g * stride_gsn, group_score)

        # After computing group_score for all 8 groups, select top-4
        # Maintain a small top-4 buffer
        top_vals = [tl.full((), -1e20, tl.float32) for _ in range(4)]
        top_idx = [-1 for _ in range(4)]
        for gg in range(0, 8):
            val = tl.load(GROUPSCORE_ptr + pid_m * stride_gsm + gg * stride_gsn)
            if val > top_vals[0]:
                top_vals[3] = top_vals[2]
                top_idx[3] = top_idx[2]
                top_vals[2] = top_vals[1]
                top_idx[2] = top_idx[1]
                top_vals[1] = top_vals[0]
                top_idx[1] = top_idx[0]
                top_vals[0] = val
                top_idx[0] = gg
            elif val > top_vals[1]:
                top_vals[3] = top_vals[2]
                top_idx[3] = top_idx[2]
                top_vals[2] = top_vals[1]
                top_idx[2] = top_idx[1]
                top_vals[1] = val
                top_idx[1] = gg
            elif val > top_vals[2]:
                top_vals[3] = top_vals[2]
                top_idx[3] = top_idx[2]
                top_vals[2] = val
                top_idx[2] = gg
            elif val > top_vals[3]:
                top_vals[3] = val
                top_idx[3] = gg
        # Store the 4 selected group indices for this token
        for j in range(0, 4):
            tl.store(GROUP_IDX_ptr + pid_m * stride_gipm + j * stride_gin, top_idx[j])


@triton.jit
def _build_group_mask_kernel(
    GROUP_IDX_ptr,      # [M, 4] int32
    GROUPMASK_ptr,      # [M, 8] float32
    M,
    stride_gip_m, stride_gin,
    stride_gmm_m, stride_gmn,
):
    pid_m = tl.program_id(0)
    for j in range(0, 4):
        idx = tl.load(GROUP_IDX_ptr + pid_m * stride_gip_m + j * stride_gin)
        # set one-hot for selected group, others zero
        for g in range(0, 8):
            one = 1.0 if g == idx else 0.0
            tl.store(GROUPMASK_ptr + pid_m * stride_gmm_m + g * stride_gmn, one)


@triton.jit
def _mask_non_selected_kernel(
    SCORES_ptr,         # [M, 256] float32
    GROUPMASK_ptr,      # [M, 8] float32
    MASKED_ptr,         # [M, 256] float32
    M, NUM_EXPERTS,
    stride_s_m, stride_s_n,
    stride_gm_m, stride_gm_n,
    stride_ms_m, stride_ms_n,
):
    pid_m = tl.program_id(0)
    # For each expert e, if its group g is masked (groupmask[g] == 0), set score to -1e20
    for e in range(0, NUM_EXPERTS):
        g = e // 32  # 256 / 8 = 32 per group
        score = tl.load(SCORES_ptr + pid_m * stride_s_m + e * stride_s_n)
        mask_val = tl.load(GROUPMASK_ptr + pid_m * stride_gm_m + g * stride_gm_n)
        masked = tl.where(mask_val > 0.0, score, -1e20)
        tl.store(MASKED_ptr + pid_m * stride_ms_m + e * stride_ms_n, masked)


@triton.jit
def _select_top8_final_kernel(
    MASKED_ptr,        # [M, 256] float32
    TOPK_idx_ptr,      # [M, 8] int32
    M, NUM_EXPERTS,
    stride_msk_m, stride_msk_n,
    stride_topk_m, stride_topk_n,
):
    pid_m = tl.program_id(0)
    # Maintain top-8 maxima and indices
    top_vals = [tl.full((), -1e20, tl.float32) for _ in range(8)]
    top_idxs = [tl.full((), -1, tl.int32) for _ in range(8)]
    for e in range(0, NUM_EXPERTS):
        score = tl.load(MASKED_ptr + pid_m * stride_msk_m + e * stride_msk_n)
        # Compare with current top-8; shift down if better
        for k in range(0, 8):
            if score > top_vals[k]:
                # shift lower positions down
                for kk in range(7, k, -1):
                    top_vals[kk] = top_vals[kk - 1]
                    top_idxs[kk] = top_idxs[kk - 1]
                top_vals[k] = score
                top_idxs[k] = e
                break
    for k in range(0, 8):
        tl.store(TOPK_idx_ptr + pid_m * stride_topk_m + k * stride_topk_n, top_idxs[k])


@triton.jit
def _normalize_and_scale_kernel(
    SELECTED_ptr,      # [M, 8] float32, selected expert scores from masked
    TOPK_idx_ptr,      # [M, 8] int32, indices of selected experts
    OUTPUT_ptr,        # [M, 8] float32, normalized and scaled
    M,
    stride_sel_m, stride_sel_n,
    stride_idx_m, stride_idx_n,
    stride_out_m, stride_out_n,
    scale,
):
    pid_m = tl.program_id(0)
    for k in range(0, 8):
        idx = tl.load(TOPK_idx_ptr + pid_m * stride_idx_m + k * stride_idx_n)
        val = tl.load(SELECTED_ptr + pid_m * stride_sel_m + idx * stride_sel_n)
        denom = 0.0
        for kk in range(0, 8):
            id2 = tl.load(TOPK_idx_ptr + pid_m * stride_idx_m + kk * stride_idx_n)
            dval = tl.load(SELECTED_ptr + pid_m * stride_sel_m + id2 * stride_sel_n)
            denom += dval
        norm = val / (denom + 1e-20) * scale
        tl.store(OUTPUT_ptr + pid_m * stride_out_m + k * stride_out_n, norm)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor, expert_bias: torch.Tensor, routed_scaling_factor: float):
        # Ensure dtype and contiguity
        M = hidden_states.shape[0]
        K = weight.shape[0]
        N = weight.shape[1]
        assert N == 256, "num_experts must be 256"
        # 1) Compute logits = hidden @ weight.T
        hidden = hidden_states.to(torch.float32).contiguous()  # [M, K]
        weight_T = weight.to(torch.float32).transpose(0, 1).contiguous()  # [K, N]
        logits = torch.empty((M, N), dtype=torch.float32, device=hidden.device)
        grid_mm = (triton.cdiv(M, 64), triton.cdiv(N, 64))
        _matmul_kernel[grid_mm](
            hidden, weight_T, logits,
            M, N, K,
            hidden.stride(0), hidden.stride(1),
            weight_T.stride(0), weight_T.stride(1),
            logits.stride(0), logits.stride(1),
            BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
        )

        # 2) Sigmoid + expert bias
        bias = expert_bias.to(torch.float32).contiguous()  # [N]
        scores = torch.empty_like(logits)
        grid_sigmoid = (triton.cdiv(M, 64), triton.cdiv(N, 64))
        _sigmoid_bias_kernel[grid_sigmoid](
            logits, bias, scores,
            M, N,
            logits.stride(0), logits.stride(1),
            scores.stride(0), scores.stride(1),
            bias.stride(0),
        )

        # 3) Reshape and compute group top-2, group scores, select top-4 groups
        scores_contig = scores.contiguous()
        group_scores = torch.empty((M, 8), dtype=torch.float32, device=scores.device)  # [M, 8]
        group_idx = torch.empty((M, 4), dtype=torch.int32, device=scores.device)       # [M, 4]
        grid_g = (M,)
        _group_top2_and_select_kernel[grid_g](
            scores_contig, group_scores, group_idx,
            M, N,
            scores_contig.stride(0), scores_contig.stride(1),
            group_scores.stride(0), group_scores.stride(1),
            group_idx.stride(0), group_idx.stride(1),
        )

        # 4) Build group mask [M, 8]
        group_mask = torch.empty((M, 8), dtype=torch.float32, device=scores.device)  # one-hot
        _build_group_mask_kernel[(M,)](
            group_idx, group_mask,
            M,
            group_idx.stride(0), group_idx.stride(1),
            group_mask.stride(0), group_mask.stride(1),
        )

        # 5) Mask out non-selected groups: set their scores to -1e20
        masked_scores = torch.empty((M, N), dtype=torch.float32, device=scores.device)
        _mask_non_selected_kernel[(M,)](
            scores_contig, group_mask, masked_scores,
            M, N,
            scores_contig.stride(0), scores_contig.stride(1),
            group_mask.stride(0), group_mask.stride(1),
            masked_scores.stride(0), masked_scores.stride(1),
        )

        # 6) Select final top-8 experts from masked scores
        top8_idx = torch.empty((M, 8), dtype=torch.int32, device=scores.device)  # [M, 8]
        _select_top8_final_kernel[(M,)](
            masked_scores, top8_idx,
            M, N,
            masked_scores.stride(0), masked_scores.stride(1),
            top8_idx.stride(0), top8_idx.stride(1),
        )

        # 7) Gather selected expert scores and normalize, then apply scaling
        selected_scores = torch.empty((M, 8), dtype=torch.float32, device=scores.device)
        # We need to gather from masked_scores using top8_idx
        # Manually gather inside kernel: we don't have indices pointing into masked_scores,
        # but we have indices into original N=256. So we can directly load masked_scores[idx].
        # Create dummy gather by reading masked_scores using top8_idx.
        # We'll do this via another kernel: load each idx and store selected_score.
        # However, Triton kernel can compute this: for each row, load 8 selected entries.
        # But Triton function only reads pointers, so we'll implement a kernel that writes.
        for k in range(8):
            idx = top8_idx[:, k]  # int32 vector
            # Load selected score
            # We need to broadcast idx across the row and load. Triton supports vectorized indexing.
            # We'll construct pointers: for each row, load masked_scores[row, idx[row]].
            # Triton loop over rows handled by grid=(M,).
            # Here we'll assume grid=(1, M) launch: but Triton expects scalar pid_m. So we use grid=(M,).
            # For kernel, we need a 1D launch and compute row from pid. We’ll relaunch a small kernel.
            pass  # Placeholder: we will implement via gather kernel below

        # Implement a gather kernel to collect selected scores based on top8_idx
        # We need to gather values from masked_scores at positions given by top8_idx for each row.
        # Triton does not support dynamic row-wise gather easily, so we do it in Python by launching per row.
        # However, Triton expects a single kernel launch. We will instead compute selected_scores in Python.
        # To keep everything Triton-only, we can implement a kernel that reads masked_scores at positions
        # given by top8_idx for each row. Triton supports scalar loads in a kernel; we can use a 2D grid:
        # grid=(M, 8), and for each (m, k), load masked_scores[m, top8_idx[m, k]].
        # Note: Triton JIT requires compile-time constants for loops. We can use a separate kernel:
        # We'll write a kernel that takes per-row pointers and idx vector, but Triton does not support
        # dynamic indexing into a vector of pointers. So we will implement selected_scores in host code
        # by computing via torch.gather on CPU, but that would break Triton-only rule. Therefore, we
        # implement a Triton kernel that writes selected_scores via row-wise gather.

        # Since we must stay within Triton, we will compute selected_scores using torch.gather on the
        # masked_scores (but that's not Triton). Given strict requirement, we will instead compute selected_scores
        # by loading directly from masked_scores using top8_idx in Python. To comply, we will instead
        # implement a Triton kernel that writes selected_scores by reading masked_scores at positions
        # given by top8_idx for each row using a small inner loop. Triton supports loops; we can pass
        # top8_idx as an argument and load per row. For simplicity, we will write the selected_scores
        # using a small Python wrapper around Triton: launch a kernel per row. But Triton requires a
        # single kernel launch; so we will implement a single kernel that handles all rows via a loop.

        # Implement a Triton kernel that writes selected_scores per row: grid=(1,), and inside the kernel,
        # we iterate over M rows and use top8_idx row pointer. Triton supports scalar loads from tensors.

        # Define kernel that writes selected_scores
        @triton.jit
        def _write_selected_scores_kernel(
            MASKED_ptr,        # [M, 256] float32
            TOPK_idx_ptr,      # [M, 8] int32
            SELECTED_ptr,      # [M, 8] float32
            M,
            stride_msk_m, stride_msk_n,
            stride_idx_m, stride_idx_n,
            stride_sel_m, stride_sel_n,
        ):
            # Single program writes all rows. Loop over rows.
            for m in range(0, M):
                for k in range(0, 8):
                    idx = tl.load(TOPK_idx_ptr + m * stride_idx_m + k * stride_idx_n)  # scalar int32
                    val = tl.load(MASKED_ptr + m * stride_msk_m + idx * stride_msk_n)  # scalar float32
                    tl.store(SELECTED_ptr + m * stride_sel_m + k * stride_sel_n, val)

        selected_scores = torch.empty((M, 8), dtype=torch.float32, device=scores.device)
        _write_selected_scores_kernel[(1,)](
            masked_scores, top8_idx, selected_scores,
            M,
            masked_scores.stride(0), masked_scores.stride(1),
            top8_idx.stride(0), top8_idx.stride(1),
            selected_scores.stride(0), selected_scores.stride(1),
        )

        # 8) Normalize and apply scaling
        output = torch.empty((M, 8), dtype=torch.float32, device=scores.device)
        _normalize_and_scale_kernel[(M,)](
            selected_scores, top8_idx, output,
            M,
            selected_scores.stride(0), selected_scores.stride(1),
            top8_idx.stride(0), top8_idx.stride(1),
            output.stride(0), output.stride(1),
            routed_scaling_factor,
        )

        # Return indices and normalized weights (top8_idx and output correspond to normalized and scaled weights)
        # Note: Original code returns (topk_idx, topk_weight). Our output is (top8_idx, output).
        # To match original, we return top8_idx as indices and output as normalized/scaled weights.
        # If you need to return topk_idx, it's top8_idx. If you need to return topk_weight, it's 'output'.
        # We'll return both to be safe: top_idx indices used for routing, and normalized+scaled weights.
        # However, original returns two tensors; to avoid mismatch, we return top_idx and output.

        # The original returns (topk_idx, topk_weight). We'll return top8_idx and output.
        # top_idx = top8_idx (indices of selected experts)
        # topk_weight = output (normalized and scaled)
        return top8_idx, output


def run(*args):
    return ModelNew()(*args)
