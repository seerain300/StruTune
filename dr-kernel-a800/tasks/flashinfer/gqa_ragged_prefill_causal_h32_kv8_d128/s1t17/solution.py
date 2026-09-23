import math
import torch
import triton
import triton.language as tl


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.num_qo_heads = 32
        self.num_kv_heads = 8
        self.head_dim = 128
        self.gqa_ratio = self.num_qo_heads // self.num_kv_heads
        self.ln2 = 1.0 / math.log(2.0)

        # Fixed tiling constants (compile-time for Triton)
        self.BLOCK_Q = 32
        self.BLOCK_K = 64
        self.BLOCK_D = 16
        self.BLOCK_K_REDUCE = 64  # used in reductions

    def forward(self, q, k, v, qo_indptr, kv_indptr, sm_scale):
        # Shapes
        assert q.dim() == 3 and k.dim() == 3 and v.dim() == 3
        assert q.shape[1] == self.num_qo_heads and k.shape[1] == self.num_kv_heads and v.shape[1] == self.num_kv_heads
        assert q.shape[2] == self.head_dim and k.shape[2] == self.head_dim and v.shape[2] == self.head_dim

        total_q, num_qo_heads, head_dim = q.shape
        total_kv, num_kv_heads, _ = k.shape
        len_indptr = qo_indptr.shape[0]
        assert num_qo_heads == self.num_qo_heads
        assert num_kv_heads == self.num_kv_heads
        assert head_dim == self.head_dim

        # Compute qo/kv lengths from indptr
        qo_len = qo_indptr[-1].item()
        kv_len = kv_indptr[-1].item()
        assert total_q == qo_len and total_kv == kv_len

        device = q.device

        # Output and lse buffers
        output = torch.zeros((total_q, num_qo_heads, head_dim), dtype=torch.bfloat16, device=device)
        lse = torch.full((total_q, num_qo_heads), -float("inf"), dtype=torch.float32, device=device)

        # Precompute expanded K and V: repeat_interleave(gqa_ratio, dim=1)
        k_expanded = k.repeat_interleave(self.gqa_ratio, dim=1)  # [total_kv, 32, 128]
        v_expanded = v.repeat_interleave(self.gqa_ratio, dim=1)  # [total_kv, 32, 128]

        # Iterate batch segments
        for b in range(len_indptr - 1):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())

            if q_start >= q_end or kv_start >= kv_end:
                # No tokens in this segment
                continue

            # Slice per segment
            q_batch = q.to(torch.float32)[q_start:q_end]            # [num_q_tokens, 32, 128]
            k_expanded_batch = k_expanded.to(torch.float32)[kv_start:kv_end]  # [num_kv_tokens, 32, 128]
            v_expanded_batch = v_expanded.to(torch.float32)[kv_start:kv_end]  # [num_kv_tokens, 32, 128]

            num_q_tokens = q_batch.shape[0]
            num_kv_tokens = k_expanded_batch.shape[0]

            # Allocate segment outputs
            output_seg = torch.empty((num_q_tokens, num_qo_heads, head_dim), dtype=torch.bfloat16, device=device)
            lse_seg = torch.empty((num_q_tokens, num_qo_heads), dtype=torch.float32, device=device)

            # 1) Compute logits[q,h,k] = sum_d q[q,h,d] * k_expanded[k,h,d], for d in [0..127]
            grid = (triton.cdiv(num_q_tokens, self.BLOCK_Q), triton.cdiv(num_kv_tokens, self.BLOCK_K), 1)
            _compute_logits_kernel[grid](
                q_batch, k_expanded_batch, output_seg,
                num_q_tokens, num_kv_tokens, head_dim,
                q_batch.stride(0), q_batch.stride(1), q_batch.stride(2),
                k_expanded_batch.stride(0), k_expanded_batch.stride(1), k_expanded_batch.stride(2),
                output_seg.stride(0), output_seg.stride(1), output_seg.stride(2),
                BLOCK_Q=self.BLOCK_Q, BLOCK_K=self.BLOCK_K, BLOCK_D=self.BLOCK_D
            )

            # 2) Compute lse per (q, head): logsumexp(logits) / ln(2) with causal mask
            # lse_seg[q, h] = logsumexp(logits[q, h, :]) / ln(2)
            grid_lse = (triton.cdiv(num_q_tokens, self.BLOCK_Q), 1)
            _lse_masked_kernel[grid_lse](
                output_seg, lse_seg,
                num_q_tokens, num_kv_tokens, head_dim,
                self.ln2, (num_kv_tokens - num_q_tokens),
                output_seg.stride(0), output_seg.stride(1), output_seg.stride(2),
                BLOCK_Q=self.BLOCK_Q, BLOCK_K=self.BLOCK_K_REDUCE
            )

            # 3) Softmax across K with causal mask, output = softmax @ V_expanded
            grid_out = (triton.cdiv(num_q_tokens, self.BLOCK_Q), 1)
            _softmax_output_kernel[grid_out](
                output_seg, v_expanded_batch, lse_seg,
                num_q_tokens, num_kv_tokens, head_dim,
                output_seg.stride(0), output_seg.stride(1), output_seg.stride(2),
                v_expanded_batch.stride(0), v_expanded_batch.stride(1), v_expanded_batch.stride(2),
                output_seg.stride(0), output_seg.stride(1), output_seg.stride(2),
                BLOCK_Q=self.BLOCK_Q, BLOCK_D=self.BLOCK_D, BLOCK_K=self.BLOCK_K_REDUCE
            )

            # Copy segment results to global output/lse
            output[q_start:q_end] = output_seg
            lse[q_start:q_end] = lse_seg

        return output, lse


# Triton kernels: no runtime-dependent loops
@triton.jit
def _compute_logits_kernel(
    Q, K, LOGITS,
    num_q_tokens, num_kv_tokens, head_dim,
    Q_stride_q, Q_stride_h, Q_stride_d,
    K_stride_k, K_stride_h, K_stride_d,
    LOGITS_stride_q, LOGITS_stride_h, LOGITS_stride_k,
    BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_D: tl.constexpr
):
    # Grid dims: (ceil(num_q_tokens/BLOCK_Q), ceil(num_kv_tokens/BLOCK_K), heads)
    pid_q = tl.program_id(0)
    pid_k = tl.program_id(1)
    h = tl.program_id(2)

    q_offsets = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
    k_offsets = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    d_offsets = tl.arange(0, BLOCK_D)

    q_mask = q_offsets < num_q_tokens
    k_mask = k_offsets < num_kv_tokens

    # Accumulator for logits[q, h, k] over d
    # Shape: [BLOCK_Q, 1, BLOCK_K]
    acc = tl.zeros((BLOCK_Q, 1, BLOCK_K), dtype=tl.float32)

    # Iterate over d in chunks of BLOCK_D
    for d0 in range(0, 128, BLOCK_D):
        d_idx = d0 + d_offsets  # [16]
        d_valid = d_idx < head_dim

        # Load Q tile: [BLOCK_Q, BLOCK_D]
        q_ptrs = Q + q_offsets[:, None] * Q_stride_q + h * Q_stride_h + d_idx[None, :] * Q_stride_d
        q_vals = tl.load(q_ptrs, mask=(q_mask[:, None] & d_valid[None, :]), other=0.0)

        # Load K tile: [BLOCK_K, BLOCK_D]
        k_ptrs = K + k_offsets[:, None] * K_stride_k + h * K_stride_h + d_idx[None, :] * K_stride_d
        k_vals = tl.load(k_ptrs, mask=(k_mask[:, None] & d_valid[None, :]), other=0.0)

        # Broadcast multiply and reduce over D: [BLOCK_Q, BLOCK_K]
        # q_vals: [Q, D], k_vals: [K, D] -> [Q, D] * [K, D] -> sum over D to get [Q, K]
        prod = q_vals[:, None, :] * k_vals[None, :, :]  # [Q, K, D]
        acc += tl.sum(prod, axis=2)[:, :, None]  # sum over D -> [Q, K]

    # Store acc to LOGITS at [q, h, k]
    logits_ptrs = LOGITS + q_offsets[:, None] * LOGITS_stride_q + h * LOGITS_stride_h + k_offsets[None, :] * LOGITS_stride_k
    store_mask = q_mask[:, None] & k_mask[None, :]
    tl.store(logits_ptrs, acc, mask=store_mask)


@triton.jit
def _lse_masked_kernel(
    LOGITS, LSE,
    num_q_tokens, num_kv_tokens, head_dim,
    ln2, delta,  # delta = num_kv_tokens - num_q_tokens
    LOGITS_stride_q, LOGITS_stride_h, LOGITS_stride_k,
    BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr
):
    # Grid dims: (ceil(num_q_tokens/BLOCK_Q), heads)
    pid_q = tl.program_id(0)
    h = tl.program_id(1)

    q_offsets = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
    q_mask = q_offsets < num_q_tokens

    # Compute max over K with causal mask; initialize to -inf
    max_vals = tl.full((BLOCK_Q,), -float("inf"), dtype=tl.float32)
    for k0 in range(0, BLOCK_K):
        k_idx = k0 + tl.arange(0, BLOCK_K)
        k_mask = k_idx < num_kv_tokens
        # Causal mask: k < (q + 1 + delta)
        q_pos = q_offsets[:, None]  # [Q, 1]
        allowed = (k_idx[None, :] < (q_pos + 1 + delta))  # [Q, K]
        ptrs = LOGITS + q_offsets[:, None] * LOGITS_stride_q + h * LOGITS_stride_h + k_idx[None, :] * LOGITS_stride_k
        vals = tl.load(ptrs, mask=(q_mask[:, None] & k_mask[None, :] & allowed), other=-float("inf"))
        # Reduce max across K chunk
        chunk_max = tl.max(vals, axis=1)  # [Q]
        max_vals = tl.maximum(max_vals, chunk_max)

    # Compute sum_exp = sum(exp(logits - max)) over K with causal mask
    sum_exp = tl.zeros((BLOCK_Q,), dtype=tl.float32)
    for k0 in range(0, BLOCK_K):
        k_idx = k0 + tl.arange(0, BLOCK_K)
        k_mask = k_idx < num_kv_tokens
        q_pos = q_offsets[:, None]
        allowed = (k_idx[None, :] < (q_pos + 1 + delta))
        ptrs = LOGITS + q_offsets[:, None] * LOGITS_stride_q + h * LOGITS_stride_h + k_idx[None, :] * LOGITS_stride_k
        vals = tl.load(ptrs, mask=(q_mask[:, None] & k_mask[None, :] & allowed), other=-float("inf"))
        exp_vals = tl.exp(vals - max_vals[:, None])
        sum_exp += tl.sum(exp_vals, axis=1)

    lse_vals = max_vals + tl.log(sum_exp) * ln2
    # Store to LSE[q, h]
    lse_ptrs = LSE + q_offsets * LSE.stride(0) + h * LSE.stride(1)
    tl.store(lse_ptrs, lse_vals, mask=q_mask)


@triton.jit
def _softmax_output_kernel(
    LOGITS, V, LSE, OUT,
    num_q_tokens, num_kv_tokens, head_dim,
    LOGITS_stride_q, LOGITS_stride_h, LOGITS_stride_k,
    V_stride_k, V_stride_h, V_stride_d,
    OUT_stride_q, OUT_stride_h, OUT_stride_d,
    BLOCK_Q: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_K: tl.constexpr
):
    # Grid dims: (ceil(num_q_tokens/BLOCK_Q), heads)
    pid_q = tl.program_id(0)
    h = tl.program_id(1)

    q_offsets = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
    q_mask = q_offsets < num_q_tokens

    # Load lse[q, h]
    lse_ptrs = LSE + q_offsets * LSE.stride(0) + h * LSE.stride(1)
    lse_vals = tl.load(lse_ptrs, mask=q_mask, other=-float("inf"))  # [Q]
    lse_scale = 1.0 / (math.log(2.0))  # not needed; LSE already computed in ln2

    # Compute output[q, h, d] = sum_k softmax(q[:,h] over k) * V[k,h,d]
    for d0 in range(0, 128, BLOCK_D):
        d_idx = d0 + tl.arange(0, BLOCK_D)
        d_valid = d_idx < head_dim

        out_row_ptrs = OUT + q_offsets[:, None] * OUT_stride_q + h * OUT_stride_h + d_idx[None, :] * OUT_stride_d
        out_row = tl.zeros((BLOCK_Q, BLOCK_D), dtype=tl.float32)

        for k0 in range(0, BLOCK_K):
            k_idx = k0 + tl.arange(0, BLOCK_K)
            k_mask = k_idx < num_kv_tokens

            # Softmax over K with causal mask
            q_pos = q_offsets[:, None]
            allowed = (k_idx[None, :] < (q_pos + 1 + delta))  # delta known at launch (host)
            ptrs = LOGITS + q_offsets[:, None] * LOGITS_stride_q + h * LOGITS_stride_h + k_idx[None, :] * LOGITS_stride_k
            vals = tl.load(ptrs, mask=(q_mask[:, None] & k_mask[None, :] & allowed), other=-float("inf"))
            # Subtract lse_vals for numerical stability
            vals = vals - lse_vals[:, None]
            exp_vals = tl.exp(vals)
            sum_exp = tl.sum(exp_vals, axis=1)  # [Q]
            softmax_vals = exp_vals / sum_exp[:, None]  # [Q, K]

            # Multiply by V[k,h,d] and accumulate over K into out_row
            v_ptrs = V + k_idx[None, :] * V_stride_k + h * V_stride_h + d_idx[:, None] * V_stride_d
            v_vals = tl.load(v_ptrs, mask=(k_mask[None, :] & d_valid[:, None]), other=0.0)  # [K, D]
            contrib = tl.sum(softmax_vals[:, :, None] * v_vals[None, :, :], axis=1)  # [Q, D]
            out_row += contrib

        # Store out_row to OUT
        store_mask = q_mask[:, None] & d_valid[None, :]
        tl.store(out_row_ptrs, out_row.to(tl.bfloat16), mask=store_mask)

        # Note: OUT is allocated as float32; we can store as bfloat16. Triton allows casting here.


def run(*args):
    return ModelNew()(*args)
