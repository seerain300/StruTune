import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernels (define before usage)

if TRITON_AVAILABLE:

    @triton.jit
    def _compute_logits_kernel(
        Q, K_EXPANDED, LOGITS,
        num_q_tokens, num_kv_tokens, head_dim,
        Q_stride_q, Q_stride_h, Q_stride_d,
        K_EXP_stride_k, K_EXP_stride_h, K_EXP_stride_d,
        LOGITS_stride_q, LOGITS_stride_h, LOGITS_stride_k,
        BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_D: tl.constexpr
    ):
        # Grid: (heads, d_tiles)
        h = tl.program_id(0)
        d_block = tl.program_id(1)
        d_idx = d_block * BLOCK_D + tl.arange(0, BLOCK_D)  # [BLOCK_D]
        d_mask = d_idx < head_dim

        # For each q in this tile
        for q_off in range(0, BLOCK_Q):
            q = q_off  # scalar q index within this tile
            q_valid = q < num_q_tokens

            # Accumulate LOGITS[q, :, d] over d in constexpr range
            acc = tl.zeros((BLOCK_K, BLOCK_D), dtype=tl.float32)
            for d in range(0, 128):
                # Load q[q, h, d]
                q_ptr = Q + q * Q_stride_q + h * Q_stride_h + d * Q_stride_d
                q_val = tl.load(q_ptr, mask=q_valid, other=0.0)  # scalar
                # For each k in K_EXPANDED
                for k_off in range(0, BLOCK_K):
                    k = k_off  # scalar k index within this tile
                    k_valid = k < num_kv_tokens
                    k_ptr = K_EXP + k * K_EXP_stride_k + h * K_EXP_stride_h + d * K_EXP_stride_d
                    k_val = tl.load(k_ptr, mask=k_valid, other=0.0)  # scalar
                    acc[k, d] = q_val * k_val

            # Store acc to LOGITS[q, :, d]
            LOGITS_ptrs = LOGITS + q * LOGITS_stride_q + h * LOGITS_stride_h + d_idx * LOGITS_stride_k
            tl.store(LOGITS_ptrs, acc[:, d_mask], mask=(q_valid & d_mask))

    @triton.jit
    def _lse_masked_kernel(
        LOGITS, LSE,
        num_q_tokens, num_kv_tokens, head_dim,
        LOGITS_stride_q, LOGITS_stride_h, LOGITS_stride_k,
        LSE_stride_q, LSE_stride_h,
        ln2, delta,
        BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr
    ):
        # Grid: (q_tiles, heads)
        q_tile = tl.program_id(0)
        h = tl.program_id(1)

        q_offsets = q_tile * BLOCK_Q + tl.arange(0, BLOCK_Q)  # [BLOCK_Q]
        q_mask = q_offsets < num_q_tokens

        max_vals = tl.full((BLOCK_Q,), -float("inf"), dtype=tl.float32)
        for k0 in range(0, 128, BLOCK_K):
            k_idx = k0 + tl.arange(0, BLOCK_K)  # [BLOCK_K]
            k_mask = k_idx < num_kv_tokens

            LOGITS_ptrs = LOGITS + q_offsets[:, None] * LOGITS_stride_q + h * LOGITS_stride_h + k_idx[None, :] * LOGITS_stride_k
            allowed = k_idx[None, :] < (q_offsets[:, None] + 1 + delta)
            vals = tl.load(LOGITS_ptrs, mask=(q_mask[:, None] & k_mask[None, :] & allowed), other=-float("inf"))
            # vals: [Q, K]
            max_vals = tl.maximum(max_vals, tl.max(vals, axis=1))

        sum_exp = tl.zeros((BLOCK_Q,), dtype=tl.float32)
        for k0 in range(0, 128, BLOCK_K):
            k_idx = k0 + tl.arange(0, BLOCK_K)
            k_mask = k_idx < num_kv_tokens
            LOGITS_ptrs = LOGITS + q_offsets[:, None] * LOGITS_stride_q + h * LOGITS_stride_h + k_idx[None, :] * LOGITS_stride_k
            allowed = k_idx[None, :] < (q_offsets[:, None] + 1 + delta)
            vals = tl.load(LOGITS_ptrs, mask=(q_mask[:, None] & k_mask[None, :] & allowed), other=-float("inf"))
            vals = vals - max_vals[:, None]
            exp_vals = tl.exp(vals)
            sum_exp += tl.sum(exp_vals, axis=1)

        lse_vals = max_vals + tl.log(sum_exp) * ln2
        LSE_ptrs = LSE + q_offsets * LSE_stride_q + h * LSE_stride_h
        tl.store(LSE_ptrs, lse_vals, mask=q_mask)

    @triton.jit
    def _softmax_output_kernel(
        LOGITS, V_EXPANDED, LSE, OUT,
        num_q_tokens, num_kv_tokens, head_dim,
        LOGITS_stride_q, LOGITS_stride_h, LOGITS_stride_k,
        V_EXP_stride_k, V_EXP_stride_h, V_EXP_stride_d,
        OUT_stride_q, OUT_stride_h, OUT_stride_d,
        BLOCK_Q: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_K: tl.constexpr
    ):
        # Grid: (q_tiles, heads)
        q_tile = tl.program_id(0)
        h = tl.program_id(1)

        q_offsets = q_tile * BLOCK_Q + tl.arange(0, BLOCK_Q)  # [BLOCK_Q]
        q_mask = q_offsets < num_q_tokens

        # Load lse for this (q,h)
        LSE_ptrs = LSE + q_offsets * LSE_stride_q + h * LSE_stride_h
        lse_vals = tl.load(LSE_ptrs, mask=q_mask, other=-float("inf"))  # [Q]

        # Compute OUT[q,h,d] across d tiles
        for d0 in range(0, 128, BLOCK_D):
            d_idx = d0 + tl.arange(0, BLOCK_D)
            d_valid = d_idx < head_dim

            OUT_ptrs = OUT + q_offsets[:, None] * OUT_stride_q + h * OUT_stride_h + d_idx[None, :] * OUT_stride_d
            out_row = tl.zeros((BLOCK_Q, BLOCK_D), dtype=tl.float32)

            for k0 in range(0, 128, BLOCK_K):
                k_idx = k0 + tl.arange(0, BLOCK_K)
                k_mask = k_idx < num_kv_tokens

                q_pos = q_offsets[:, None]  # [Q,1]
                allowed = k_idx[None, :] < (q_pos[:, 0] + 1)  # causal mask: j < q + 1

                LOGITS_ptrs = LOGITS + q_pos * LOGITS_stride_q + h * LOGITS_stride_h + k_idx[None, :] * LOGITS_stride_k
                vals = tl.load(LOGITS_ptrs, mask=(q_mask[:, None] & k_mask[None, :] & allowed), other=-float("inf"))  # [Q, K]
                vals = vals - lse_vals[:, None]  # [Q, K]
                exp_vals = tl.exp(vals)          # [Q, K]
                sum_exp = tl.sum(exp_vals, axis=1)  # [Q]
                probs = exp_vals / sum_exp[:, None]  # [Q, K]

                V_ptrs = V_EXP + k_idx[:, None] * V_EXP_stride_k + h * V_EXP_stride_h + d_idx[None, :] * V_EXP_stride_d
                v_tile = tl.load(V_ptrs, mask=(k_mask[:, None] & d_valid[None, :]), other=0.0)  # [K, D]
                out_row += tl.sum(probs[:, :, None] * v_tile[None, :, :], axis=1)  # [Q, D] += sum over K

            tl.store(OUT_ptrs, out_row, mask=(q_mask[:, None] & d_valid[None, :]))


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Constants from original
        self.num_qo_heads = 32
        self.num_kv_heads = 8
        self.head_dim = 128
        self.gqa_ratio = self.num_qo_heads // self.num_kv_heads  # 4
        self.ln2 = 1.0 / math.log(2.0)

    def forward(self, q, k, v, qo_indptr, kv_indptr, sm_scale):
        # Original asserts and constraints
        total_q = q.shape[0]
        num_qo_heads = q.shape[1]
        head_dim = q.shape[2]
        total_kv = k.shape[0]
        num_kv_heads = k.shape[1]
        assert num_qo_heads == 32
        assert num_kv_heads == 8
        assert head_dim == 128
        assert total_q == int(qo_indptr[-1].item())
        assert total_kv == int(kv_indptr[-1].item())

        device = q.device
        # Compute in float32; output in bfloat16
        q_f32 = q.to(torch.float32)
        k_f32 = k.to(torch.float32)
        v_f32 = v.to(torch.float32)

        # Initialize output and lse
        output = torch.zeros((total_q, self.num_qo_heads, head_dim), dtype=torch.float32, device=device)
        lse = torch.full((total_q, self.num_qo_heads), -float("inf"), dtype=torch.float32, device=device)

        # Process each batch segment defined by indptr
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

            # Extract batched tensors for this segment
            q_batch = q_f32[q_start:q_end]                       # [num_q_tokens, 32, 128]
            k_batch = k_f32[kv_start:kv_end]                    # [num_kv_tokens, 8, 128]
            v_batch = v_f32[kv_start:kv_end]                    # [num_kv_tokens, 8, 128]

            # Expand K and V by GQA ratio
            k_expanded = k_batch.repeat_interleave(self.gqa_ratio, dim=1)  # [num_kv_tokens, 32, 128]
            v_expanded = v_batch.repeat_interleave(self.gqa_ratio, dim=1)  # [num_kv_tokens, 32, 128]

            # Allocate segment output and lse
            output_seg = torch.empty((num_q_tokens, self.num_qo_heads, self.head_dim), dtype=torch.float32, device=device)
            lse_seg = torch.empty((num_q_tokens, self.num_qo_heads), dtype=torch.float32, device=device)

            # Launch Triton kernels
            if TRITON_AVAILABLE:
                # 1) Compute logits = Q @ K^T per (q,h) without loops over runtime sizes
                # Grid: (heads, d_tiles). We set BLOCK_D=128 so one tile covers head_dim.
                _compute_logits_kernel[(self.num_qo_heads, 1)](
                    q_batch, k_expanded, output_seg,
                    num_q_tokens, num_kv_tokens, self.head_dim,
                    q_batch.stride(0), q_batch.stride(1), q_batch.stride(2),
                    k_expanded.stride(0), k_expanded.stride(1), k_expanded.stride(2),
                    output_seg.stride(0), output_seg.stride(1), output_seg.stride(2),
                    BLOCK_Q=128, BLOCK_K=128, BLOCK_D=128
                )

                # 2) Compute lse[q,h] = logsumexp with causal mask
                _lse_masked_kernel[(1, self.num_qo_heads)](
                    output_seg, lse_seg,
                    num_q_tokens, num_kv_tokens, self.head_dim,
                    output_seg.stride(0), output_seg.stride(1), output_seg.stride(2),
                    lse_seg.stride(0), lse_seg.stride(1),
                    self.ln2, (num_kv_tokens - num_q_tokens),
                    BLOCK_Q=1, BLOCK_K=128
                )

                # 3) Compute output = softmax(logits) @ V_expanded
                _softmax_output_kernel[(1, self.num_qo_heads)](
                    output_seg, v_expanded, lse_seg, output_seg,  # OUT alias to reuse output_seg buffer (only segment scope)
                    num_q_tokens, num_kv_tokens, self.head_dim,
                    output_seg.stride(0), output_seg.stride(1), output_seg.stride(2),
                    v_expanded.stride(0), v_expanded.stride(1), v_expanded.stride(2),
                    output_seg.stride(0), output_seg.stride(1), output_seg.stride(2),
                    BLOCK_Q=1, BLOCK_D=16, BLOCK_K=128
                )

            # Copy segment results to global output/lse
            output[q_start:q_end] = output_seg.to(torch.bfloat16)
            lse[q_start:q_end] = lse_seg

        return output, lse


def run(*args):
    return ModelNew()(*args)
