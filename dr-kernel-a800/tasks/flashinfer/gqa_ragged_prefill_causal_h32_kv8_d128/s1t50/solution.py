import torch
import math
import triton
import triton.language as tl

# Fixed parameters from the original code
HEAD_DIM = 128
GQA_RATIO = 4  # 32 / 8

# Triton kernels

@triton.jit
def _compute_logits_kernel(
    Q, K_EXP, LOGITS,
    num_q_tokens, num_kv_tokens, head_dim,
    Q_stride_q, Q_stride_h, Q_stride_d,
    K_EXP_stride_k, K_EXP_stride_h, K_EXP_stride_d,
    LOGITS_stride_q, LOGITS_stride_h, LOGITS_stride_k,
    BLOCK_Q: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_K: tl.constexpr
):
    # Grid: (ceil(num_q_tokens/BLOCK_Q), heads)
    pid_q = tl.program_id(0)
    h = tl.program_id(1)

    q_offsets = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
    q_mask = q_offsets < num_q_tokens

    # Accumulator for logits: [Q, K]
    acc = tl.zeros((BLOCK_Q, BLOCK_K), dtype=tl.float32)

    # Iterate over d in chunks of BLOCK_D with constexpr bounds
    for d0 in range(0, HEAD_DIM, BLOCK_D):
        d_idx = d0 + tl.arange(0, BLOCK_D)
        d_valid = d_idx < head_dim

        # For each d element, compute q[:, d] * K_exp[:, h, d]
        # K_exp is [K, H, D], we will index specific h and d slices.
        # Compute Q[:, d] as [Q] and K_exp[:, h, d] as [K], then outer product to [Q, K].
        for d_off in range(0, BLOCK_D):
            d_cur = d0 + d_off
            d_mask_cur = d_cur < head_dim

            # Load Q slice: Q[q, h, d_cur] -> [Q]
            q_ptrs = Q + q_offsets * Q_stride_q + h * Q_stride_h + d_cur * Q_stride_d
            q_vec = tl.load(q_ptrs, mask=q_mask & d_mask_cur, other=0.0)  # [Q]

            # Load K_exp slice: K_exp[k, h, d_cur] -> [K]
            k_ptrs = K_EXP + tl.arange(0, BLOCK_K) * K_EXP_stride_k + h * K_EXP_stride_h + d_cur * K_EXP_stride_d
            k_offsets = tl.arange(0, BLOCK_K)
            k_mask = k_offsets < num_kv_tokens
            k_vec = tl.load(k_ptrs, mask=k_mask, other=0.0)  # [K]

            # Outer product: [Q] * [K]^T -> [Q, K]
            prod = q_vec[:, None] * k_vec[None, :]  # [Q, K]
            acc += prod  # accumulate across d

    # Store acc to LOGITS[q, h, k]
    LOGITS_ptrs = LOGITS + q_offsets[:, None] * LOGITS_stride_q + h * LOGITS_stride_h + tl.arange(0, BLOCK_K)[None, :] * LOGITS_stride_k
    tl.store(LOGITS_ptrs, acc, mask=(q_mask[:, None] & (tl.arange(0, BLOCK_K)[None, :] < num_kv_tokens)))


@triton.jit
def _lse_masked_kernel(
    LOGITS, LSE,
    num_q_tokens, num_kv_tokens, head_dim,
    LOGITS_stride_q, LOGITS_stride_h, LOGITS_stride_k,
    LSE_stride_q, LSE_stride_h,
    BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr
):
    # Grid: (ceil(num_q_tokens/BLOCK_Q), heads)
    pid_q = tl.program_id(0)
    h = tl.program_id(1)

    q_offsets = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
    q_mask = q_offsets < num_q_tokens

    max_vals = tl.full((BLOCK_Q,), -float("inf"), dtype=tl.float32)
    sum_exp = tl.zeros((BLOCK_Q,), dtype=tl.float32)

    for k0 in range(0, HEAD_DIM, BLOCK_K):  # use 128 as constexpr upper bound; we guard masks
        k_idx = k0 + tl.arange(0, BLOCK_K)
        k_mask = k_idx < num_kv_tokens

        # Load logits[q, h, k] -> [Q, K]
        LOGITS_ptrs = LOGITS + q_offsets[:, None] * LOGITS_stride_q + h * LOGITS_stride_h + k_idx[None, :] * LOGITS_stride_k
        vals = tl.load(LOGITS_ptrs, mask=(q_mask[:, None] & k_mask[None, :]), other=-float("inf"))  # [Q, K]

        # Causal mask: allowed if k < (q_pos + 1). q_pos is q_offsets
        allowed = k_idx[None, :] < (q_offsets[:, None] + 1)
        vals = tl.where(allowed, vals, -float("inf"))

        # Row-wise max and sum(exp(vals - max))
        row_max = tl.max(vals, axis=1)  # [Q]
        max_vals = tl.maximum(max_vals, row_max)

        exp_vals = tl.exp(vals - max_vals[:, None])  # [Q, K]
        row_sum = tl.sum(exp_vals, axis=1)  # [Q]
        sum_exp += row_sum

    ln2 = 1.4426950408889634  # log(2)
    lse_vals = max_vals + tl.log(sum_exp) * ln2

    LSE_ptrs = LSE + q_offsets * LSE_stride_q + h * LSE_stride_h
    tl.store(LSE_ptrs, lse_vals, mask=q_mask)


@triton.jit
def _softmax_output_kernel(
    LOGITS, V_EXP, LSE, OUT,
    num_q_tokens, num_kv_tokens, head_dim,
    LOGITS_stride_q, LOGITS_stride_h, LOGITS_stride_k,
    V_EXP_stride_k, V_EXP_stride_h, V_EXP_stride_d,
    OUT_stride_q, OUT_stride_h, OUT_stride_d,
    BLOCK_Q: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_K: tl.constexpr
):
    # Grid: (ceil(num_q_tokens/BLOCK_Q), heads)
    pid_q = tl.program_id(0)
    h = tl.program_id(1)

    q_offsets = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
    q_mask = q_offsets < num_q_tokens

    # Load lse for this (q,h): shape [Q]
    LSE_ptrs = LSE + q_offsets * LSE_stride_q + h * LSE_stride_h
    lse_vals = tl.load(LSE_ptrs, mask=q_mask, other=-float("inf"))  # [Q]

    # Output over D tiles
    for d0 in range(0, HEAD_DIM, BLOCK_D):
        d_idx = d0 + tl.arange(0, BLOCK_D)
        d_valid = d_idx < head_dim

        OUT_ptrs = OUT + q_offsets[:, None] * OUT_stride_q + h * OUT_stride_h + d_idx[None, :] * OUT_stride_d
        out_row = tl.zeros((BLOCK_Q, BLOCK_D), dtype=tl.float32)

        # Softmax over K tiles
        for k0 in range(0, HEAD_DIM, BLOCK_K):
            k_idx = k0 + tl.arange(0, BLOCK_K)
            k_mask = k_idx < num_kv_tokens

            q_pos = q_offsets  # [Q]
            allowed = k_idx[None, :] < (q_pos[:, None] + 1)  # [Q, K]

            LOGITS_ptrs = LOGITS + q_pos[:, None] * LOGITS_stride_q + h * LOGITS_stride_h + k_idx[None, :] * LOGITS_stride_k
            vals = tl.load(LOGITS_ptrs, mask=(q_mask[:, None] & k_mask[None, :] & allowed), other=-float("inf"))  # [Q, K]
            vals = vals - lse_vals[:, None]  # [Q, K]

            exp_vals = tl.exp(vals)  # [Q, K]
            sum_exp = tl.sum(exp_vals, axis=1)  # [Q]
            probs = exp_vals / sum_exp[:, None]  # [Q, K]

            # Multiply probs with V_exp[k, h, d] and accumulate over K
            V_ptrs = V_EXP + k_idx[:, None] * V_EXP_stride_k + h * V_EXP_stride_h + d_idx[None, :] * V_EXP_stride_d
            v_vec = tl.load(V_ptrs, mask=(k_mask[:, None] & d_valid[None, :]), other=0.0)  # [K, D]
            # probs: [Q, K], v_vec: [K, D] -> outer across K: for each k, add probs[:, k][:, None] * v_vec[k, :]
            # Implement by broadcasting and reducing over K
            # Create [Q, D, K] and sum over last axis
            # Expand to [Q, D, K]
            probs_exp = probs[:, None, :]  # [Q, 1, K]
            v_exp = v_vec[None, :, :]      # [1, D, K]
            contrib = probs_exp * v_exp    # [Q, D, K]
            out_row += tl.sum(contrib, axis=2)  # [Q, D]

        tl.store(OUT_ptrs, out_row, mask=(q_mask[:, None] & d_valid[None, :]))


class ModelNew(torch.nn.Module):
    def __init__(self, head_dim=128, gqa_ratio=4):
        super().__init__()
        self.head_dim = head_dim
        self.gqa_ratio = gqa_ratio
        # Fixed constants for kernels
        self.BLOCK_Q = 1
        self.BLOCK_K = 64
        self.BLOCK_D = 128  # since head_dim=128, we can set BLOCK_D=128 for a single pass

    def forward(self, q, k, v, qo_indptr, kv_indptr, sm_scale):
        # Shapes from original: q [total_q, 32, 128], k [total_kv, 8, 128], v [total_kv, 8, 128]
        total_q, num_qo_heads, head_dim_q = q.shape
        total_kv, num_kv_heads, head_dim_k = k.shape
        assert head_dim_q == self.head_dim and head_dim_k == self.head_dim
        assert num_qo_heads == 32 and num_kv_heads == 8
        assert q.is_cuda and k.is_cuda and v.is_cuda

        # Check indptr bounds
        assert qo_indptr[-1].item() == total_q
        assert kv_indptr[-1].item() == total_kv

        # Prepare outputs
        output = torch.empty((total_q, num_qo_heads, head_dim_q), dtype=torch.bfloat16, device=q.device)
        lse = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=q.device)

        # Expand K and V by GQA ratio on GPU
        k_exp = k.repeat_interleave(self.gqa_ratio, dim=1)  # [K, 8*4, D] -> [K, 32, D]
        v_exp = v.repeat_interleave(self.gqa_ratio, dim=1)  # [K, 32, D]

        # Process each batch segment
        len_indptr = qo_indptr.numel()
        for b in range(len_indptr - 1):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())

            if q_start >= q_end or kv_start >= kv_end:
                continue

            num_q_tokens = q_end - q_start
            num_kv_tokens = kv_end - kv_start

            # Extract batch segments
            q_batch = q[q_start:q_end].to(torch.float32)  # [num_q_tokens, 32, 128]
            k_exp_batch = k_exp[kv_start:kv_end]  # [num_kv_tokens, 32, 128], float32
            v_exp_batch = v_exp[kv_start:kv_end]  # [num_kv_tokens, 32, 128]

            # Strides for Triton
            Q_strides = q_batch.stride()  # (q_stride_q, q_stride_h, q_stride_d)
            KEXP_strides = k_exp_batch.stride()  # (k_stride_k, k_stride_h, k_stride_d)
            VEXP_strides = v_exp_batch.stride()  # (v_stride_k, v_stride_h, v_stride_d)

            # Allocate intermediate logits and output for this segment
            logits_seg = torch.empty((num_q_tokens, 32, num_kv_tokens), dtype=torch.float32, device=q.device)
            out_seg = torch.empty((num_q_tokens, 32, head_dim_q), dtype=torch.float32, device=q.device)

            # Launch kernels
            # 1) Compute logits
            _compute_logits_kernel[(triton.cdiv(num_q_tokens, self.BLOCK_Q), 32)](
                q_batch, k_exp_batch, logits_seg,
                num_q_tokens, num_kv_tokens, self.head_dim,
                Q_strides[0], Q_strides[1], Q_strides[2],
                KEXP_strides[0], KEXP_strides[1], KEXP_strides[2],
                logits_seg.stride(0), logits_seg.stride(1), logits_seg.stride(2),
                BLOCK_Q=self.BLOCK_Q, BLOCK_D=self.BLOCK_D, BLOCK_K=self.BLOCK_K
            )

            # 2) LSE with causal mask
            _lse_masked_kernel[(triton.cdiv(num_q_tokens, self.BLOCK_Q), 32)](
                logits_seg, lse,
                num_q_tokens, num_kv_tokens, self.head_dim,
                logits_seg.stride(0), logits_seg.stride(1), logits_seg.stride(2),
                lse.stride(0), lse.stride(1),
                BLOCK_Q=self.BLOCK_Q, BLOCK_K=self.BLOCK_K
            )

            # 3) Softmax + output
            _softmax_output_kernel[(triton.cdiv(num_q_tokens, self.BLOCK_Q), 32)](
                logits_seg, v_exp_batch, lse, out_seg,
                num_q_tokens, num_kv_tokens, self.head_dim,
                logits_seg.stride(0), logits_seg.stride(1), logits_seg.stride(2),
                VEXP_strides[0], VEXP_strides[1], VEXP_strides[2],
                out_seg.stride(0), out_seg.stride(1), out_seg.stride(2),
                BLOCK_Q=self.BLOCK_Q, BLOCK_D=self.BLOCK_D, BLOCK_K=self.BLOCK_K
            )

            # Store segment to global output
            output[q_start:q_end] = out_seg.to(torch.bfloat16)

        return output, lse


def run(*args):
    return ModelNew()(*args)
