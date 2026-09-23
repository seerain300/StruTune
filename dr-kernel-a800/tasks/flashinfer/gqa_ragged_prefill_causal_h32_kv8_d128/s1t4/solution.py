import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False

# Single Triton kernel: per-segment attention (compute logits, masked lse, softmax, and output)
@triton.jit
def _segment_attention_kernel(
    q_ptr, k_ptr, v_ptr, output_ptr, lse_ptr,
    num_q_tokens, num_kv_tokens, num_qo_heads, head_dim,
    q_stride_q, q_stride_h, q_stride_d,
    k_stride_k, k_stride_h, k_stride_d,
    v_stride_k, v_stride_h, v_stride_d,
    output_stride_q, output_stride_h, output_stride_d,
    lse_stride_q, lse_stride_h,
    ln2, delta,  # delta = num_kv_tokens - num_q_tokens
    BLOCK_Q: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid_q = tl.program_id(0)  # tiles along Q
    pid_h = tl.program_id(1)  # tiles along H

    q_start = pid_q * BLOCK_Q
    h_start = pid_h * BLOCK_H

    q_offsets = q_start + tl.arange(0, BLOCK_Q)[:, None, None]   # (Q, 1, 1)
    h_offsets = h_start + tl.arange(0, BLOCK_H)[None, :, None]   # (1, H, 1)
    k_offsets = tl.arange(0, BLOCK_K)[None, None, :]             # (1, 1, K)

    q_mask = q_offsets < num_q_tokens
    h_mask = h_offsets < num_qo_heads
    k_mask = k_offsets < num_kv_tokens

    # Accumulator for logits: [Q, H, K]
    acc = tl.zeros((BLOCK_Q, BLOCK_H, BLOCK_K), dtype=tl.float32)

    # Compute logits = Q @ K^T across head_dim in chunks of 128
    for d in range(0, head_dim, 128):
        # Load Q block: [Q, H, 128]
        q_block = tl.load(
            q_ptr + q_offsets * q_stride_q + h_offsets * q_stride_h + (d + tl.arange(0, 128)) * q_stride_d,
            mask=q_mask[:, None] & h_mask[None, :] & ((d + tl.arange(0, 128)) < head_dim),
            other=0.0
        )
        # Load K block: [H, 128]
        k_block = tl.load(
            k_ptr + h_offsets * k_stride_k + (d + tl.arange(0, 128)) * k_stride_d,  # k does not depend on q here
            mask=h_mask[None, :, None] & ((d + tl.arange(0, 128)) < head_dim) & k_mask[None, None, :],
            other=0.0
        )
        # Accumulate acc += sum over H of q_block[:, h, :] * k_block[h, :]
        for h_off in range(BLOCK_H):
            q_h = q_block[:, h_off, :]           # (Q, 128)
            k_h = k_block[h_off, :]              # (128,)
            acc[:, h_off, :] += tl.sum(q_h * k_h[None, :], axis=1)

    # Now apply causal mask: for each q, allowed keys k < min(num_kv_tokens, q + 1 + delta)
    # We need to compute allowed_max for each q in this tile
    allowed_max = q_start + 1 + delta  # scalar per tile
    allowed_max = tl.full((1,), allowed_max, dtype=tl.int32)  # Triton supports scalar broadcasting; keep consistent

    # Build causal mask across (Q, H, K)
    # Note: Triton supports elementwise comparison with scalar. We create k_vec and compare.
    for q_off in range(0, BLOCK_Q):
        q_idx = q_start + q_off
        q_valid = q_idx < num_q_tokens
        for h_off in range(0, BLOCK_H):
            h_idx = h_start + h_off
            h_valid = h_idx < num_qo_heads
            # k_vec: (K,)
            k_vec = k_offsets[0, 0, :]  # vector (BLOCK_K,)
            causal_mask = k_vec < allowed_max  # elementwise
            # Apply mask: set acc to -inf where not causal
            acc[q_off, h_off, :] = tl.where(causal_mask, acc[q_off, h_off, :], -float('inf'))

    # Compute per-(q,h) lse = logsumexp(acc) / ln(2)
    max_val = tl.full((BLOCK_Q, BLOCK_H), -float('inf'), dtype=tl.float32)
    sum_exp = tl.zeros((BLOCK_Q, BLOCK_H), dtype=tl.float32)

    # Loop over K chunks to compute max and sum_exp
    for k0 in range(0, num_kv_tokens, BLOCK_K):
        k_vec = k0 + tl.arange(0, BLOCK_K)
        k_mask_chunk = k_vec < num_kv_tokens
        for q_off in range(0, BLOCK_Q):
            q_idx = q_start + q_off
            q_valid = q_idx < num_q_tokens
            for h_off in range(0, BLOCK_H):
                h_idx = h_start + h_off
                h_valid = h_idx < num_qo_heads
                # Gather acc[q_off, h_off, k] for this chunk
                vals = acc[q_off, h_off, k0:k0+BLOCK_K]
                # Apply causal mask (k_vec < allowed_max); allowed_max is scalar per tile
                mask_causal = k_vec < allowed_max
                vals = tl.where(mask_causal, vals, -float('inf'))
                # Compute max and sum_exp
                max_val[q_off, h_off] = tl.maximum(max_val[q_off, h_off], tl.max(vals, axis=0))
                sum_exp[q_off, h_off] += tl.sum(tl.exp(vals - max_val[q_off, h_off]), axis=0)

    # Final lse: max + log(sum_exp) - ln2
    lse_seg = max_val + tl.log(sum_exp) - ln2

    # Compute softmax along K and output: output[q,h,:] = sum_k softmax[q,h,k] * v[k,h,:]
    for d in range(0, head_dim, 128):
        # Initialize output segment for this d chunk: [Q, H, 128]
        out_d = tl.zeros((BLOCK_Q, BLOCK_H, 128), dtype=tl.float32)
        for q_off in range(0, BLOCK_Q):
            q_idx = q_start + q_off
            q_valid = q_idx < num_q_tokens
            for h_off in range(0, BLOCK_H):
                h_idx = h_start + h_off
                h_valid = h_idx < num_qo_heads
                # Compute softmax for this (q,h) across K
                sum_exp_qh = 0.0
                for k0 in range(0, num_kv_tokens, BLOCK_K):
                    k_vec = k0 + tl.arange(0, BLOCK_K)
                    k_mask_chunk = k_vec < num_kv_tokens
                    vals = acc[q_off, h_off, k0:k0+BLOCK_K]
                    mask_causal = k_vec < allowed_max
                    vals = tl.where(mask_causal, vals, -float('inf'))
                    m = tl.max(vals, axis=0)
                    s = tl.sum(tl.exp(vals - m), axis=0)
                    sum_exp_qh = s  # scalar for this (q,h)
                # Now compute softmax and accumulate output
                for k0 in range(0, num_kv_tokens, BLOCK_K):
                    k_vec = k0 + tl.arange(0, BLOCK_K)
                    k_mask_chunk = k_vec < num_kv_tokens
                    vals = acc[q_off, h_off, k0:k0+BLOCK_K]
                    mask_causal = k_vec < allowed_max
                    vals = tl.where(mask_causal, vals, -float('inf'))
                    m = tl.max(vals, axis=0)
                    soft = tl.exp(vals - m) / tl.maximum(sum_exp_qh, 1e-20)
                    # Load v chunk: V[k,h,d] for d this chunk
                    v_chunk = tl.load(
                        v_ptr + k_vec * v_stride_k + h_idx * v_stride_h + (d + tl.arange(0, 128)) * v_stride_d,
                        mask=k_mask_chunk[None, :] & h_valid & ((d + tl.arange(0, 128)) < head_dim),
                        other=0.0
                    )  # (K, 128)
                    # Since we have soft of shape (K,), and v_chunk (K,128), we need to broadcast soft over 128-dim.
                    # out_d[q_off, h_off, d:d+128] += sum_k soft[k] * v_chunk[k, :]
                    # Implement per kk:
                    for kk in range(0, BLOCK_K):
                        k_idx = k0 + kk
                        k_valid = k_idx < num_kv_tokens
                        s_val = soft[kk]  # scalar
                        v_vec = v_chunk[kk, :]  # (128,)
                        out_d[q_off, h_off, :] += s_val * v_vec

        # Store output for this d chunk into output buffer at q_idx, h_idx
        # output_ptr points to a [num_q_tokens, num_qo_heads, head_dim] buffer; we write out_d
        # We need to compute pointers: output_ptr + q_idx*output_stride_q + h_idx*output_stride_h + d*output_stride_d
        for q_off in range(0, BLOCK_Q):
            q_idx = q_start + q_off
            q_valid = q_idx < num_q_tokens
            for h_off in range(0, BLOCK_H):
                h_idx = h_start + h_off
                h_valid = h_idx < num_qo_heads
                tl.store(
                    output_ptr + q_idx * output_stride_q + h_idx * output_stride_h + (d + tl.arange(0, 128)) * output_stride_d,
                    out_d[q_off, h_off, :].to(tl.bfloat16),
                    mask=q_valid & h_valid & ((d + tl.arange(0, 128)) < head_dim)
                )

        # Store lse_seg for this tile
        for q_off in range(0, BLOCK_Q):
            q_idx = q_start + q_off
            q_valid = q_idx < num_q_tokens
            for h_off in range(0, BLOCK_H):
                h_idx = h_start + h_off
                h_valid = h_idx < num_qo_heads
                tl.store(
                    lse_ptr + q_idx * lse_stride_q + h_idx * lse_stride_h,
                    lse_seg[q_off, h_off].to(tl.float32),
                    mask=q_valid & h_valid
                )

# ModelNew forward (Triton-only)
class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Constants for this task
        self.num_qo_heads = 32
        self.num_kv_heads = 8
        self.head_dim = 128
        self.gqa_ratio = self.num_qo_heads // self.num_kv_heads  # 4
        self.sm_scale = 1.0 / math.sqrt(self.head_dim)
        self.ln2 = math.log(2.0)

    def forward(self, q, k, v, qo_indptr, kv_indptr, sm_scale):
        # Triton availability and device
        assert TRITON_AVAILABLE, "Triton is not available"
        assert q.is_cuda and k.is_cuda and v.is_cuda, "Inputs must be CUDA tensors"

        # Cast to float32 for compute
        q_f32 = q.to(torch.float32).contiguous()
        k_f32 = k.to(torch.float32).contiguous()
        v_f32 = v.to(torch.float32).contiguous()

        # Expand K/V for GQA (repeat heads by ratio)
        k_expanded = k_f32.repeat_interleave(self.gqa_ratio, dim=1).contiguous()
        v_expanded = v_f32.repeat_interleave(self.gqa_ratio, dim=1).contiguous()

        # Output and lse buffers
        total_q, num_qo_heads, head_dim = q_f32.shape
        total_kv, num_kv_heads, _ = k_f32.shape
        assert num_qo_heads == self.num_qo_heads
        assert num_kv_heads == self.num_kv_heads
        assert head_dim == self.head_dim

        output = torch.empty((total_q, num_qo_heads, head_dim), dtype=torch.bfloat16, device=q_f32.device)
        lse = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=q_f32.device)

        # Process segments
        n_batches = qo_indptr.shape[0] - 1
        for b in range(n_batches):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())

            if q_start >= q_end or kv_start >= kv_end:
                continue

            num_q_tokens = q_end - q_start
            num_kv_tokens = kv_end - kv_start

            # Allocate per-segment output and lse
            output_seg = torch.empty((num_q_tokens, num_qo_heads, head_dim), dtype=torch.bfloat16, device=q_f32.device)
            lse_seg = torch.empty((num_q_tokens, num_qo_heads), dtype=torch.float32, device=q_f32.device)

            # Prepare q_batch, k_expanded_batch, v_expanded_batch (views or copies)
            q_batch = q_f32[q_start:q_end]
            k_expanded_batch = k_expanded[kv_start:kv_end]
            v_expanded_batch = v_expanded[kv_start:kv_end]

            # Strides
            q_stride_q = q_batch.stride(0)
            q_stride_h = q_batch.stride(1)
            q_stride_d = q_batch.stride(2)

            k_stride_k = k_expanded_batch.stride(0)
            k_stride_h = k_expanded_batch.stride(1)
            k_stride_d = k_expanded_batch.stride(2)

            v_stride_k = v_expanded_batch.stride(0)
            v_stride_h = v_expanded_batch.stride(1)
            v_stride_d = v_expanded_batch.stride(2)

            output_stride_q = output_seg.stride(0)
            output_stride_h = output_seg.stride(1)
            output_stride_d = output_seg.stride(2)

            lse_stride_q = lse_seg.stride(0)
            lse_stride_h = lse_seg.stride(1)

            # Launch Triton kernel for this segment
            grid = (triton.cdiv(num_q_tokens, 32), triton.cdiv(num_qo_heads, 8))
            _segment_attention_kernel[grid](
                q_batch, k_expanded_batch, v_expanded_batch, output_seg, lse_seg,
                num_q_tokens, num_kv_tokens, num_qo_heads, head_dim,
                q_stride_q, q_stride_h, q_stride_d,
                k_stride_k, k_stride_h, k_stride_d,
                v_stride_k, v_stride_h, v_stride_d,
                output_stride_q, output_stride_h, output_stride_d,
                lse_stride_q, lse_stride_h,
                self.ln2, num_kv_tokens - num_q_tokens,  # delta
                BLOCK_Q=32, BLOCK_H=8, BLOCK_K=64
            )

            # Copy segment results to global output/lse
            output[q_start:q_end] = output_seg
            lse[q_start:q_end] = lse_seg

        return output, lse


def run(*args):
    return ModelNew()(*args)
