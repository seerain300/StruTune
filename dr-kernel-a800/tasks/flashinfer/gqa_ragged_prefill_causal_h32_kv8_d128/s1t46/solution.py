import torch
import math
import triton
import triton.language as tl


# Triton kernels: all attention math performed inside Triton.

@triton.jit
def _compute_logits_single_qh_kernel(
    Q, KEXP, LOGITS,
    num_q_tokens, num_kv_tokens, head_dim,
    Q_stride_q, Q_stride_h, Q_stride_d,
    KEXP_stride_k, KEXP_stride_h, KEXP_stride_d,
    LOGITS_stride_q, LOGITS_stride_h, LOGITS_stride_k,
    BLOCK_D: tl.constexpr, BLOCK_K: tl.constexpr, SM_SCALE: tl.constexpr
):
    # Grid: (num_q_tokens, num_qo_heads)
    q = tl.program_id(0)
    h = tl.program_id(1)

    # Initialize accumulator for logits[q, h, k]
    acc = tl.zeros((num_kv_tokens,), dtype=tl.float32)

    # Loop over d dimension in constexpr chunks
    for d0 in range(0, head_dim, BLOCK_D):
        d_idx = d0 + tl.arange(0, BLOCK_D)
        d_valid = d_idx < head_dim  # mask not needed because head_dim=128 but used for safety

        # Load q[q, h, d_idx]
        q_ptrs = Q + q * Q_stride_q + h * Q_stride_h + d_idx * Q_stride_d
        q_vals = tl.load(q_ptrs)  # [BLOCK_D]

        # Load k_expanded[:, h, d_idx]
        k_ptrs = KEXP + tl.arange(0, num_kv_tokens)[:, None] * KEXP_stride_k + h * KEXP_stride_h + d_idx[None, :] * KEXP_stride_d
        k_mat = tl.load(k_ptrs)  # [num_kv_tokens, BLOCK_D]

        # Accumulate dot for each k in the tile: acc += sum_d q_vals[d] * k_mat[k, d]
        # We need a per-k reduction over BLOCK_D. Triton supports reductions across axis.
        # Use nested loop to avoid dynamic reductions:
        for kd in range(0, BLOCK_D):
            # valid_d = d0 + kd < head_dim
            valid_d = (d0 + kd) < head_dim
            # Select q_val if valid else 0
            q_val = q_vals[kd] if valid_d else 0.0
            # Multiply and sum over k
            k_vec = k_mat[:, kd]  # [num_kv_tokens]
            acc += q_val * k_vec

    # Apply scaling and store acc to LOGITS[q, h, :]
    LOGITS_ptrs = LOGITS + q * LOGITS_stride_q + h * LOGITS_stride_h + tl.arange(0, num_kv_tokens) * LOGITS_stride_k
    tl.store(LOGITS_ptrs, acc * SM_SCALE)


@triton.jit
def _lse_single_qh_kernel(
    LOGITS, LSE,
    num_q_tokens, num_kv_tokens, head_dim,
    LOGITS_stride_q, LOGITS_stride_h, LOGITS_stride_k,
    LSE_stride_q, LSE_stride_h,
    ln2: tl.constexpr, delta: tl.constexpr,  # delta = num_kv_tokens - num_q_tokens
    BLOCK_K: tl.constexpr
):
    # Grid: (num_q_tokens, num_qo_heads)
    q = tl.program_id(0)
    h = tl.program_id(1)

    max_val = tl.full((), -float("inf"), dtype=tl.float32)
    sum_val = tl.full((), 0.0, dtype=tl.float32)

    # Reduce over K tiles
    for k0 in range(0, head_dim, BLOCK_K):
        k_idx = k0 + tl.arange(0, BLOCK_K)
        k_mask = k_idx < num_kv_tokens

        LOGITS_ptrs = LOGITS + q * LOGITS_stride_q + h * LOGITS_stride_h + k_idx * LOGITS_stride_k
        vals = tl.load(LOGITS_ptrs, mask=k_mask, other=-float("inf"))  # [BLOCK_K]

        # Causal mask: allowed keys j < (q + 1 + delta)
        allowed = k_idx < (q + 1 + delta)
        vals = tl.where(allowed, vals, -float("inf"))

        m = tl.max(vals, axis=0)
        s = tl.sum(tl.exp(vals - m), axis=0)
        max_val = tl.maximum(max_val, m)
        sum_val += s

    lse_val = max_val + tl.log(sum_val) * ln2
    LSE_ptr = LSE + q * LSE_stride_q + h * LSE_stride_h
    tl.store(LSE_ptr, lse_val)


@triton.jit
def _output_single_qh_kernel(
    LOGITS, VEXP, LSE, OUT,
    num_q_tokens, num_kv_tokens, head_dim,
    LOGITS_stride_q, LOGITS_stride_h, LOGITS_stride_k,
    VEXP_stride_k, VEXP_stride_h, VEXP_stride_d,
    OUT_stride_q, OUT_stride_h, OUT_stride_d,
    BLOCK_K: tl.constexpr
):
    # Grid: (num_q_tokens, num_qo_heads)
    q = tl.program_id(0)
    h = tl.program_id(1)

    lse_ptr = LSE + q * LSE.stride(0) + h * LSE.stride(1)
    lse_val = tl.load(lse_ptr)

    # Output across d in fixed chunks
    for d0 in range(0, head_dim, head_dim):  # single iteration, but keeps kernel constexpr-friendly
        # We can process d = 0..127 in a loop; Triton supports constexpr loops
        for d in range(0, head_dim):
            out_val = tl.full((), 0.0, dtype=tl.float32)
            # Reduce over K tiles to compute softmax and accumulate with VEXP
            for k0 in range(0, head_dim, BLOCK_K):
                k_idx = k0 + tl.arange(0, BLOCK_K)
                k_mask = k_idx < num_kv_tokens

                LOGITS_ptrs = LOGITS + q * LOGITS_stride_q + h * LOGITS_stride_h + k_idx * LOGITS_stride_k
                vals = tl.load(LOGITS_ptrs, mask=k_mask, other=-float("inf"))  # [BLOCK_K]

                # Causal mask: allowed keys j < min(num_kv_tokens, q + 1 + (num_kv_tokens - num_q_tokens))
                allowed = k_idx < (q + 1 + (num_kv_tokens - num_q_tokens))
                vals = tl.where(allowed, vals, -float("inf"))

                m = tl.max(vals, axis=0)
                exp_vals = tl.exp(vals - m)
                s = tl.sum(exp_vals, axis=0)
                probs = exp_vals / s  # [BLOCK_K]

                VEXP_ptrs = VEXP + k_idx * VEXP_stride_k + h * VEXP_stride_h + d * VEXP_stride_d
                v_vals = tl.load(VEXP_ptrs, mask=k_mask, other=0.0)  # [BLOCK_K]
                out_val += tl.sum(probs * v_vals, axis=0)

            OUT_ptr = OUT + q * OUT_stride_q + h * OUT_stride_h + d * OUT_stride_d
            tl.store(OUT_ptr, out_val)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Precompute constants
        self.head_dim = 128
        self.gqa_ratio = 32 // 8  # 4

    def forward(self, q, k, v, qo_indptr, kv_indptr, sm_scale):
        # Shapes
        total_q, num_qo_heads, head_dim = q.shape
        total_kv, num_kv_heads, _ = k.shape
        assert num_qo_heads == 32
        assert num_kv_heads == 8
        assert head_dim == 128
        assert total_q == int(qo_indptr[-1].item())
        assert total_kv == int(kv_indptr[-1].item())

        device = q.device

        # Output buffers
        output = torch.empty((total_q, num_qo_heads, head_dim), dtype=torch.bfloat16, device=device)
        lse = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=device)

        # Expand k and v by GQA ratio (heads)
        k_expanded = k.repeat_interleave(self.gqa_ratio, dim=1).to(torch.float32)
        v_expanded = v.repeat_interleave(self.gqa_ratio, dim=1).to(torch.float32)

        # Process each batch segment
        for b in range(qo_indptr.numel() - 1):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())

            num_q_tokens = q_end - q_start
            num_kv_tokens = kv_end - kv_start
            if num_q_tokens <= 0 or num_kv_tokens <= 0:
                continue

            # Slice segment tensors
            q_batch = q[q_start:q_end]  # [num_q_tokens, 32, 128]
            k_expanded_batch = k_expanded[kv_start:kv_end]  # [num_kv_tokens, 32, 128]
            v_expanded_batch = v_expanded[kv_start:kv_end]  # [num_kv_tokens, 32, 128]

            # Per-segment outputs
            out_seg = torch.empty((num_q_tokens, num_qo_heads, head_dim), dtype=torch.bfloat16, device=device)
            lse_seg = torch.empty((num_q_tokens, num_qo_heads), dtype=torch.float32, device=device)

            # Strides (element strides, not bytes)
            q_strides = q_batch.stride()
            kexp_strides = k_expanded_batch.stride()
            vexp_strides = v_expanded_batch.stride()
            out_seg_strides = out_seg.stride()
            lse_seg_strides = lse_seg.stride()

            # Allocate LOGITS buffer for this segment: [num_q_tokens, 32, num_kv_tokens]
            logits = torch.empty((num_q_tokens, num_qo_heads, num_kv_tokens), dtype=torch.float32, device=device)
            logits_strides = logits.stride()

            # 1) Compute LOGITS[q, h, k] = sum_d q[q,h,d] * k_expanded[k,h,d]
            grid = (num_q_tokens, num_qo_heads)
            _compute_logits_single_qh_kernel[grid](
                q_batch, k_expanded_batch, logits,
                num_q_tokens, num_kv_tokens, self.head_dim,
                q_strides[0], q_strides[1], q_strides[2],
                kexp_strides[0], kexp_strides[1], kexp_strides[2],
                logits_strides[0], logits_strides[1], logits_strides[2],
                BLOCK_D=self.head_dim, BLOCK_K=64, SM_SCALE=sm_scale
            )

            # 2) Compute lse per (q,h) with causal mask, divide by ln(2)
            _lse_single_qh_kernel[grid](
                logits, lse_seg,
                num_q_tokens, num_kv_tokens, self.head_dim,
                logits_strides[0], logits_strides[1], logits_strides[2],
                lse_seg_strides[0], lse_seg_strides[1],
                1.4426950408889634,  # ln(2)
                num_kv_tokens - num_q_tokens,  # delta
                BLOCK_K=64
            )

            # 3) Compute output per (q,h): softmax(LOGITS) @ V_expanded
            _output_single_qh_kernel[grid](
                logits, v_expanded_batch, lse_seg, out_seg,
                num_q_tokens, num_kv_tokens, self.head_dim,
                logits_strides[0], logits_strides[1], logits_strides[2],
                vexp_strides[0], vexp_strides[1], vexp_strides[2],
                out_seg_strides[0], out_seg_strides[1], out_seg_strides[2],
                BLOCK_K=64
            )

            # Copy back to global output/lse
            output[q_start:q_end] = out_seg
            lse[q_start:q_end] = lse_seg

        return output, lse


def run(*args):
    return ModelNew()(*args)
