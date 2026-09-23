import torch
import math
import triton
import triton.language as tl


@triton.jit
def _compute_logits_kernel(
    Q, K_EXP, LOGITS,
    num_q_tokens, num_kv_tokens, head_dim,
    Q_stride_q, Q_stride_h, Q_stride_d,
    K_EXP_stride_k, K_EXP_stride_h, K_EXP_stride_d,
    LOGITS_stride_q, LOGITS_stride_h, LOGITS_stride_k,
    BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_D: tl.constexpr
):
    # Grid: (ceil(num_q_tokens/BLOCK_Q), heads)
    pid_q = tl.program_id(0)
    h = tl.program_id(1)

    q_offsets = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)  # [BLOCK_Q]
    k_offsets = tl.arange(0, BLOCK_K)                    # [BLOCK_K]

    # Accumulator for logits [q_tile, k_tile]
    acc = tl.zeros((BLOCK_Q, BLOCK_K), dtype=tl.float32)

    # Reduce over d in fixed tiles
    for d0 in range(0, 128, BLOCK_D):
        d_idx = d0 + tl.arange(0, BLOCK_D)               # [BLOCK_D]
        d_valid = d_idx < 128

        # Load Q[q,h,d] -> [Q, D]
        Q_ptrs = Q + q_offsets[:, None] * Q_stride_q + h * Q_stride_h + d_idx[None, :] * Q_stride_d
        q_mat = tl.load(
            Q_ptrs,
            mask=(q_offsets[:, None] < num_q_tokens) & (d_valid[None, :]),
            other=0.0
        )  # [Q, D]

        # Load K[k,h,d] -> [K, D]
        K_ptrs = K_EXP + k_offsets[:, None] * K_EXP_stride_k + h * K_EXP_stride_h + d_idx[None, :] * K_EXP_stride_d
        k_mat = tl.load(
            K_ptrs,
            mask=(k_offsets[:, None] < num_kv_tokens) & (d_valid[None, :]),
            other=0.0
        )  # [K, D]

        # Outer-product contribution: q_mat[:, None, :] * k_mat[None, :, :] over D -> [Q, K]
        prod = q_mat[:, None, :] * k_mat[None, :, :]  # [Q, D, K]
        acc += tl.sum(prod, axis=1)  # [Q, K]

    # Store logits[q_tile, h, k_tile]
    LOGITS_ptrs = LOGITS + q_offsets[:, None] * LOGITS_stride_q + h * LOGITS_stride_h + k_offsets[None, :] * LOGITS_stride_k
    store_mask = (q_offsets[:, None] < num_q_tokens) & (k_offsets[None, :] < num_kv_tokens)
    tl.store(LOGITS_ptrs, acc, mask=store_mask)


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

    q_offsets = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)  # [Q]
    max_vals = tl.full((BLOCK_Q,), -float("inf"), dtype=tl.float32)

    for k0 in range(0, 128, BLOCK_K):
        k_idx = k0 + tl.arange(0, BLOCK_K)  # [K]
        k_valid = k_idx < 128
        k_mask = k_idx < num_kv_tokens

        LOGITS_ptrs = LOGITS + q_offsets[:, None] * LOGITS_stride_q + h * LOGITS_stride_h + k_idx[None, :] * LOGITS_stride_k
        mask_qk = (q_offsets[:, None] < num_q_tokens) & k_mask[None, :]
        vals = tl.load(LOGITS_ptrs, mask=mask_qk, other=-float("inf"))  # [Q, K]

        # Max across K for each q
        max_vals = tl.maximum(max_vals, tl.max(vals, axis=1))
        # Sum of exp(vals - max_vals) across K for each q
        sum_exp = tl.sum(tl.exp(vals - max_vals[:, None]), axis=1)

    lse_vals = max_vals + tl.log(sum_exp)  # logsumexp
    LSE_ptrs = LSE + q_offsets * LSE_stride_q + h * LSE_stride_h
    store_mask = q_offsets < num_q_tokens
    tl.store(LSE_ptrs, lse_vals, mask=store_mask)


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

    q_offsets = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)  # [1 when BLOCK_Q=1]
    q_mask = q_offsets < num_q_tokens  # for BLOCK_Q=1 always true

    # Load lse for this (q,h)
    LSE_ptrs = LSE + q_offsets * LSE_stride_q + h * LSE_stride_h
    lse_vals = tl.load(LSE_ptrs, mask=q_mask, other=-float("inf"))  # [1]

    # Compute output[q,h,d] across d tiles
    for d0 in range(0, 128, BLOCK_D):
        d_idx = d0 + tl.arange(0, BLOCK_D)  # [16]
        d_valid = d_idx < head_dim

        OUT_ptrs = OUT + q_offsets[:, None] * OUT_stride_q + h * OUT_stride_h + d_idx[None, :] * OUT_stride_d
        out_row = tl.zeros((BLOCK_Q, BLOCK_D), dtype=tl.float32)

        # Softmax over K tiles with causal mask
        for k0 in range(0, 128, BLOCK_K):
            k_idx = k0 + tl.arange(0, BLOCK_K)  # [64]
            k_mask = k_idx < num_kv_tokens

            q_pos = q_offsets  # [1]
            allowed = k_idx[None, :] < (q_pos[:, None] + 1)  # [1, K]

            LOGITS_ptrs = LOGITS + q_pos * LOGITS_stride_q + h * LOGITS_stride_h + k_idx[None, :] * LOGITS_stride_k
            vals = tl.load(LOGITS_ptrs, mask=(q_mask[:, None] & k_mask[None, :] & allowed), other=-float("inf"))  # [1, K]
            vals = vals - lse_vals[:, None]  # [1, K]

            exp_vals = tl.exp(vals)
            sum_exp = tl.sum(exp_vals, axis=1)  # [1]
            probs = exp_vals / sum_exp[:, None]  # [1, K]

            V_ptrs = V_EXP + k_idx[:, None] * V_EXP_stride_k + h * V_EXP_stride_h + d_idx[None, :] * V_EXP_stride_d
            v_vec = tl.load(V_ptrs, mask=(k_mask[:, None] & d_valid[None, :]), other=0.0)  # [K, D]
            out_row += tl.sum(probs[:, :, None] * v_vec[None, :, :], axis=1)  # [Q, D]

        store_mask = (q_offsets[:, None] < num_q_tokens) & (d_valid[None, :])
        tl.store(OUT_ptrs, out_row, mask=store_mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Fixed constants
        self.num_qo_heads = 32
        self.num_kv_heads = 8
        self.head_dim = 128
        self.gqa_ratio = self.num_qo_heads // self.num_kv_heads  # 4
        # Triton tiling parameters (compile-time constants)
        self.BLOCK_Q = 1
        self.BLOCK_K = 64
        self.BLOCK_D = 16

    def forward(self, q, k, v, qo_indptr, kv_indptr, sm_scale):
        # All tensors must be CUDA
        assert q.is_cuda and k.is_cuda and v.is_cuda, "Tensors must be on CUDA for Triton kernels."
        device = q.device

        total_q, num_qo_heads, head_dim = q.shape
        total_kv, num_kv_heads, _ = k.shape
        assert num_qo_heads == self.num_qo_heads and head_dim == self.head_dim and num_kv_heads == self.num_kv_heads, \
            f"Expected num_qo_heads=32, head_dim=128, num_kv_heads=8, got ({num_qo_heads}, {head_dim}, {num_kv_heads})"

        len_indptr = qo_indptr.shape[0]
        assert qo_indptr.shape[0] == len_indptr and kv_indptr.shape[0] == len_indptr, "Indptr shapes must match len_indptr."
        assert qo_indptr.dtype in (torch.int32, torch.int64) and kv_indptr.dtype in (torch.int32, torch.int64)
        assert total_q == int(qo_indptr[-1].item()), "qo_indptr[-1] must equal total_q."
        assert total_kv == int(kv_indptr[-1].item()), "kv_indptr[-1] must equal total_kv."

        # Prepare outputs
        output = torch.empty((total_q, self.num_qo_heads, self.head_dim), dtype=torch.bfloat16, device=device)
        lse = torch.empty((total_q, self.num_qo_heads), dtype=torch.float32, device=device)

        for b in range(len_indptr - 1):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())

            if q_start >= q_end or kv_start >= kv_end:
                continue

            num_q_tokens = q_end - q_start
            num_kv_tokens = kv_end - kv_start

            # Slice and make contiguous in float32 for compute
            q_batch = q[q_start:q_end].contiguous().to(torch.float32)  # [Q, H, D]
            k_batch = k[kv_start:kv_end].contiguous().to(torch.float32)  # [KV, H, D]
            v_batch = v[kv_start:kv_end].contiguous().to(torch.float32)  # [KV, H, D]

            # Expand K/V by GQA ratio
            k_expanded = k_batch.repeat_interleave(self.gqa_ratio, dim=1)  # [Q, H, D] note: repeat_interleave is a data layout change
            v_expanded = v_batch.repeat_interleave(self.gqa_ratio, dim=1)  # [KV, H, D]

            # Allocate intermediates
            logits = torch.empty((num_q_tokens, self.num_qo_heads, num_kv_tokens), dtype=torch.float32, device=device)
            output_seg = torch.empty((num_q_tokens, self.num_qo_heads, self.head_dim), dtype=torch.float32, device=device)
            lse_seg = torch.empty((num_q_tokens, self.num_qo_heads), dtype=torch.float32, device=device)

            # 1) Compute logits
            grid_q = (triton.cdiv(num_q_tokens, self.BLOCK_Q), self.num_qo_heads)
            _compute_logits_kernel[grid_q](
                q_batch, k_expanded, logits,
                num_q_tokens, num_kv_tokens, self.head_dim,
                q_batch.stride(0), q_batch.stride(1), q_batch.stride(2),
                k_expanded.stride(0), k_expanded.stride(1), k_expanded.stride(2),
                logits.stride(0), logits.stride(1), logits.stride(2),
                BLOCK_Q=self.BLOCK_Q, BLOCK_K=self.BLOCK_K, BLOCK_D=self.BLOCK_D
            )

            # 2) Compute lse per (q, head) with causal mask
            grid_lse = (triton.cdiv(num_q_tokens, self.BLOCK_Q), self.num_qo_heads)
            _lse_masked_kernel[grid_lse](
                logits, lse_seg,
                num_q_tokens, num_kv_tokens, self.head_dim,
                logits.stride(0), logits.stride(1), logits.stride(2),
                lse_seg.stride(0), lse_seg.stride(1),
                BLOCK_Q=self.BLOCK_Q, BLOCK_K=self.BLOCK_K
            )

            # 3) Compute output = softmax(logits) @ V_expanded
            grid_out = (triton.cdiv(num_q_tokens, self.BLOCK_Q), self.num_qo_heads)
            _softmax_output_kernel[grid_out](
                logits, v_expanded, lse_seg, output_seg,
                num_q_tokens, num_kv_tokens, self.head_dim,
                logits.stride(0), logits.stride(1), logits.stride(2),
                v_expanded.stride(0), v_expanded.stride(1), v_expanded.stride(2),
                output_seg.stride(0), output_seg.stride(1), output_seg.stride(2),
                BLOCK_Q=self.BLOCK_Q, BLOCK_D=self.BLOCK_D, BLOCK_K=self.BLOCK_K
            )

            # Store segment to global output/lse
            output[q_start:q_end] = output_seg.to(torch.bfloat16)
            lse[q_start:q_end] = lse_seg

        return output, lse


def run(*args):
    return ModelNew()(*args)
