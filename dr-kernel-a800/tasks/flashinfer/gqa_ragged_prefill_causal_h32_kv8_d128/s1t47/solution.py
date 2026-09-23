import torch
import math

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernels: operate on a single (q, h) pair per program. All loops are constexpr over head_dim=128.

if TRITON_AVAILABLE:
    @triton.jit
    def _compute_logits_single_qh(
        Q, KEXP, LOGITS,
        num_q_tokens, num_kv_tokens, head_dim,
        Q_stride_q, Q_stride_h, Q_stride_d,
        KEXP_stride_k, KEXP_stride_h, KEXP_stride_d,
        LOGITS_stride_q, LOGITS_stride_h, LOGITS_stride_k,
        SM_SCALE: tl.constexpr  # scale applied to logits
    ):
        # Grid: (num_q_tokens, num_qo_heads)
        q = tl.program_id(0)
        h = tl.program_id(1)

        # We compute LOGITS[q, h, k] for all k in 0..127
        for k in range(0, 128):
            # Pointers for Q[q, h, :] and KEXP[:, h, k]
            q_ptrs = Q + q * Q_stride_q + h * Q_stride_h + tl.arange(0, head_dim) * Q_stride_d
            k_ptrs = KEXP + k * KEXP_stride_k + h * KEXP_stride_h + tl.arange(0, head_dim) * KEXP_stride_d
            q_vec = tl.load(q_ptrs)
            k_vec = tl.load(k_ptrs)
            # Dot product over d=0..127
            dot = tl.sum(q_vec * k_vec, axis=0)  # scalar
            # Store with scaling
            LOGITS_ptr = LOGITS + q * LOGITS_stride_q + h * LOGITS_stride_h + k * LOGITS_stride_k
            tl.store(LOGITS_ptr, dot * SM_SCALE)


    @triton.jit
    def _lse_single_qh(
        LOGITS, LSE,
        num_q_tokens, num_kv_tokens, head_dim,
        LOGITS_stride_q, LOGITS_stride_h, LOGITS_stride_k,
        LSE_stride_q, LSE_stride_h,
        ln2: tl.constexpr,
        delta: tl.constexpr  # num_kv_tokens - num_q_tokens
    ):
        # Grid: (num_q_tokens, num_qo_heads)
        q = tl.program_id(0)
        h = tl.program_id(1)

        # Compute max over K for numerical stability, ignoring causal mask for max (since mask gives -inf, max stays unaffected)
        max_vals = tl.full((), -float("inf"), dtype=tl.float32)

        for k in range(0, 128):
            LOGITS_ptr = LOGITS + q * LOGITS_stride_q + h * LOGITS_stride_h + k * LOGITS_stride_k
            val = tl.load(LOGITS_ptr)  # scalar
            max_vals = tl.maximum(max_vals, val)

        # Compute sum of exp(LOGITS - max_vals) over K, but apply causal mask so invalid positions contribute 0
        sum_exp = tl.full((), 0.0, dtype=tl.float32)
        for k in range(0, 128):
            LOGITS_ptr = LOGITS + q * LOGITS_stride_q + h * LOGITS_stride_h + k * LOGITS_stride_k
            val = tl.load(LOGITS_ptr)  # scalar
            allowed = k < (q + 1 + delta)  # scalar boolean
            contrib = tl.exp(val - max_vals) * allowed
            sum_exp += contrib

        lse_val = max_vals + tl.log(sum_exp) * ln2
        LSE_ptr = LSE + q * LSE_stride_q + h * LSE_stride_h
        tl.store(LSE_ptr, lse_val, mask=(q < num_q_tokens))


    @triton.jit
    def _output_single_qh(
        LOGITS, VEXP, LSE, OUT,
        num_q_tokens, num_kv_tokens, head_dim,
        LOGITS_stride_q, LOGITS_stride_h, LOGITS_stride_k,
        VEXP_stride_k, VEXP_stride_h, VEXP_stride_d,
        OUT_stride_q, OUT_stride_h, OUT_stride_d,
        ln2: tl.constexpr,  # not directly used here
        delta: tl.constexpr  # num_kv_tokens - num_q_tokens
    ):
        # Grid: (num_q_tokens, num_qo_heads)
        q = tl.program_id(0)
        h = tl.program_id(1)

        LSE_ptr = LSE + q * LSE.stride(0) + h * LSE.stride(1)
        lse_val = tl.load(LSE_ptr, mask=(q < num_q_tokens), other=-float("inf"))  # scalar

        # For each output d, compute sum over K of softmax(LOGITS[q, h, :]) * VEXP[:, h, d]
        for d in range(0, 128):
            out_elem = tl.full((), 0.0, dtype=tl.float32)
            # Compute sum_exp for normalization
            sum_exp = tl.full((), 0.0, dtype=tl.float32)
            for k in range(0, 128):
                LOGITS_ptr = LOGITS + q * LOGITS_stride_q + h * LOGITS_stride_h + k * LOGITS_stride_k
                val = tl.load(LOGITS_ptr)  # scalar
                allowed = k < (q + 1 + delta)  # scalar boolean
                sum_exp += tl.exp(val - lse_val) * allowed

            inv_sum = 1.0 / sum_exp

            # Accumulate out_elem
            for k in range(0, 128):
                LOGITS_ptr = LOGITS + q * LOGITS_stride_q + h * LOGITS_stride_h + k * LOGITS_stride_k
                val = tl.load(LOGITS_ptr)  # scalar
                allowed = k < (q + 1 + delta)  # scalar boolean
                prob = tl.exp(val - lse_val) * allowed * inv_sum

                VEXP_ptr = VEXP + k * VEXP_stride_k + h * VEXP_stride_h + d * VEXP_stride_d
                v_elem = tl.load(VEXP_ptr)  # scalar
                out_elem += prob * v_elem

            # Store output[q, h, d]
            OUT_ptr = OUT + q * OUT_stride_q + h * OUT_stride_h + d * OUT_stride_d
            tl.store(OUT_ptr, out_elem, mask=(q < num_q_tokens))


class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, qo_indptr, kv_indptr, sm_scale):
        """
        Triton-optimized attention that mirrors the original logic:
        - GQA: num_qo_heads=32, num_kv_heads=8 -> repeat k/v along head dim by ratio 4.
        - For each segment defined by qo_indptr, kv_indptr, compute attention over q/k/v.
        - Returns (output [total_q, 32, 128], lse [total_q, 32]).
        """
        device = q.device
        total_q = qo_indptr[-1].item()
        total_kv = kv_indptr[-1].item()

        assert q.shape[1] == 32 and k.shape[1] == 8 and v.shape[1] == 8 and q.shape[2] == 128 and k.shape[2] == 128 and v.shape[2] == 128
        assert total_q == qo_indptr[-1].item()
        assert total_kv == kv_indptr[-1].item()

        output = torch.empty((total_q, 32, 128), dtype=torch.bfloat16, device=device)
        lse = torch.empty((total_q, 32), dtype=torch.float32, device=device)

        # Prepare expanded K and V for GQA
        k_expanded = k.repeat_interleave(4, dim=1)  # [*, 32, 128]
        v_expanded = v.repeat_interleave(4, dim=1)  # [*, 32, 128]

        # Compute per-segment
        for b in range(len(qo_indptr) - 1):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())

            if q_start >= q_end or kv_start >= kv_end:
                # No tokens in this segment
                continue

            num_q_tokens = q_end - q_start
            num_kv_tokens = kv_end - kv_start

            # Prepare per-segment output buffers (flattened for simplicity; but we can just compute into output directly)
            # We'll compute into a temporary LOGITS, then lse, then output. For simplicity and correctness, reuse output tensor per segment and lse per segment by slicing.
            # However, Triton kernels expect pointers; we'll allocate per-segment tensors, but since we need to keep full output, we'll compute directly into output[q_start:q_end, :, :], which is not supported in Triton. Instead, we compute into LOGITS, then lse, then output, writing at specific indices.

            # Allocate per-segment LOGITS, LSE, VEXP slices
            LOGITS = torch.empty((num_q_tokens, 32, num_kv_tokens), dtype=torch.float32, device=device)
            LSE_seg = torch.empty((num_q_tokens, 32), dtype=torch.float32, device=device)

            # Copy relevant q, k_expanded, v_expanded into per-segment tensors via slicing
            # Note: Triton kernels receive pointers; slicing here is for clarity but not used directly in Triton math.
            # We will launch kernels with full tensors and use masks; q, k_expanded, v_expanded are already defined on device.

            # 1) Compute logits
            grid = (num_q_tokens, 32)
            _compute_logits_single_qh[grid](
                q, k_expanded, LOGITS,
                num_q_tokens, num_kv_tokens, 128,
                q.stride(0), q.stride(1), q.stride(2),
                k_expanded.stride(0), k_expanded.stride(1), k_expanded.stride(2),
                LOGITS.stride(0), LOGITS.stride(1), LOGITS.stride(2),
                SM_SCALE=sm_scale
            )

            # 2) Compute LSE per (q, h)
            delta = num_kv_tokens - num_q_tokens
            ln2 = math.log(2.0)
            _lse_single_qh[grid](
                LOGITS, LSE_seg,
                num_q_tokens, num_kv_tokens, 128,
                LOGITS.stride(0), LOGITS.stride(1), LOGITS.stride(2),
                LSE_seg.stride(0), LSE_seg.stride(1),
                ln2=ln2,
                delta=delta
            )

            # 3) Compute output
            OUT_seg = torch.empty((num_q_tokens, 32, 128), dtype=torch.float32, device=device)
            _output_single_qh[grid](
                LOGITS, v_expanded, LSE_seg, OUT_seg,
                num_q_tokens, num_kv_tokens, 128,
                LOGITS.stride(0), LOGITS.stride(1), LOGITS.stride(2),
                v_expanded.stride(0), v_expanded.stride(1), v_expanded.stride(2),
                OUT_seg.stride(0), OUT_seg.stride(1), OUT_seg.stride(2),
                ln2=ln2, delta=delta
            )

            # Store into global output
            # output[q_start:q_end, :, :] = OUT_seg.to(torch.bfloat16)
            # However, Triton kernel wrote float32; we need to move OUT_seg to output with slicing
            # Note: Direct assignment here is PyTorch; but since we cannot index Triton output in host, we'll recompute per segment directly into output by launching kernels that write at offsets. To avoid that, we instead compute per segment into a temporary and copy.

            # Copy segment output into final output tensor
            output[q_start:q_end] = OUT_seg.to(torch.bfloat16)
            lse[q_start:q_end] = LSE_seg

        return output, lse


def run(*args):
    return ModelNew()(*args)
