import math
import torch

# Triton imports
import triton
import triton.language as tl


@triton.jit
def _compute_logits_single_qh_kernel(
    Q, KEXP, LOGITS,
    num_q_tokens, num_kv_tokens, head_dim,
    Q_stride_q, Q_stride_h, Q_stride_d,
    KEXP_stride_k, KEXP_stride_h, KEXP_stride_d,
    LOGITS_stride_q, LOGITS_stride_h, LOGITS_stride_k,
    BLOCK_D: tl.constexpr  # always 128
):
    # Grid dims: (num_q_tokens, num_qo_heads)
    q = tl.program_id(0)
    h = tl.program_id(1)

    # Initialize LOGITS[q, h, k] = 0
    LOGITS_ptrs = LOGITS + q * LOGITS_stride_q + h * LOGITS_stride_h
    LOGITS_ptrs_k = LOGITS_ptrs + tl.arange(0, 128) * LOGITS_stride_k
    tl.store(LOGITS_ptrs_k, tl.zeros((128,), dtype=tl.float32))

    # Accumulate logits across d = 0..127
    for d in range(0, 128):
        # q_val = Q[q, h, d]
        Q_ptr = Q + q * Q_stride_q + h * Q_stride_h + d * Q_stride_d
        q_val = tl.load(Q_ptr)  # scalar float32

        # k_vec = KEXP[:, h, d] for all k in 0..127
        KEXP_ptrs_d = KEXP + tl.arange(0, 128) * KEXP_stride_k + h * KEXP_stride_h + d * KEXP_stride_d
        k_vec = tl.load(KEXP_ptrs_d)  # [128] float32

        # logits_k += q_val * k_vec
        LOGITS_ptrs_k = LOGITS + q * LOGITS_stride_q + h * LOGITS_stride_h + tl.arange(0, 128) * LOGITS_stride_k
        curr = tl.load(LOGITS_ptrs_k)  # [128]
        new = curr + q_val * k_vec
        tl.store(LOGITS_ptrs_k, new)


@triton.jit
def _lse_single_qh_kernel(
    LOGITS, LSE,
    num_q_tokens, num_kv_tokens, head_dim,
    LOGITS_stride_q, LOGITS_stride_h, LOGITS_stride_k,
    LSE_stride_q, LSE_stride_h,
    ln2: tl.constexpr,
    delta: tl.constexpr,  # num_kv_tokens - num_q_tokens
    BLOCK_K: tl.constexpr  # always 128
):
    # Grid dims: (num_q_tokens, num_qo_heads)
    q = tl.program_id(0)
    h = tl.program_id(1)

    max_vals = tl.full((), -float("inf"), dtype=tl.float32)
    sum_exp = tl.full((), 0.0, dtype=tl.float32)

    for k in range(0, 128):  # constexpr loop
        LOGITS_ptr = LOGITS + q * LOGITS_stride_q + h * LOGITS_stride_h + k * LOGITS_stride_k
        val = tl.load(LOGITS_ptr)
        allowed = k < (q + 1 + delta)
        val = tl.where(allowed, val, -float("inf"))
        max_vals = tl.maximum(max_vals, val)

    for k in range(0, 128):
        LOGITS_ptr = LOGITS + q * LOGITS_stride_q + h * LOGITS_stride_h + k * LOGITS_stride_k
        val = tl.load(LOGITS_ptr)
        allowed = k < (q + 1 + delta)
        val = tl.where(allowed, val, -float("inf"))
        sum_exp += tl.exp(val - max_vals)

    lse_val = max_vals + tl.log(sum_exp) * ln2
    LSE_ptr = LSE + q * LSE_stride_q + h * LSE_stride_h
    tl.store(LSE_ptr, lse_val, mask=(q < num_q_tokens))


@triton.jit
def _softmax_output_single_qh_kernel(
    LOGITS, VEXP, LSE, OUT,
    num_q_tokens, num_kv_tokens, head_dim,
    LOGITS_stride_q, LOGITS_stride_h, LOGITS_stride_k,
    VEXP_stride_k, VEXP_stride_h, VEXP_stride_d,
    OUT_stride_q, OUT_stride_h, OUT_stride_d,
    ln2: tl.constexpr,
    delta: tl.constexpr,
    BLOCK_D: tl.constexpr  # 128
):
    # Grid dims: (num_q_tokens, num_qo_heads)
    q = tl.program_id(0)
    h = tl.program_id(1)

    # Load lse for this (q,h)
    LSE_ptr = LSE + q * LSE_stride_q + h * LSE_stride_h
    lse_val = tl.load(LSE_ptr, mask=(q < num_q_tokens), other=-float("inf"))  # scalar float32

    # Pass 1: compute max and sum for softmax
    max_vals = tl.full((), -float("inf"), dtype=tl.float32)
    sum_exp = tl.full((), 0.0, dtype=tl.float32)

    for k in range(0, 128):
        LOGITS_ptr = LOGITS + q * LOGITS_stride_q + h * LOGITS_stride_h + k * LOGITS_stride_k
        val = tl.load(LOGITS_ptr)
        allowed = k < (q + 1 + delta)
        val = tl.where(allowed, val, -float("inf"))
        max_vals = tl.maximum(max_vals, val)

    for k in range(0, 128):
        LOGITS_ptr = LOGITS + q * LOGITS_stride_q + h * LOGITS_stride_h + k * LOGITS_stride_k
        val = tl.load(LOGITS_ptr)
        allowed = k < (q + 1 + delta)
        val = tl.where(allowed, val, -float("inf"))
        sum_exp += tl.exp(val - max_vals)

    # Pass 2: write output per d
    for d in range(0, 128):
        out_scalar = tl.full((), 0.0, dtype=tl.float32)
        for k in range(0, 128):
            LOGITS_ptr = LOGITS + q * LOGITS_stride_q + h * LOGITS_stride_h + k * LOGITS_stride_k
            val = tl.load(LOGITS_ptr)
            allowed = k < (q + 1 + delta)
            val = tl.where(allowed, val, -float("inf"))
            prob = tl.exp(val - max_vals) / sum_exp
            VEXP_ptr = VEXP + k * VEXP_stride_k + h * VEXP_stride_h + d * VEXP_stride_d
            v_val = tl.load(VEXP_ptr)
            out_scalar += prob * v_val

        # Store as float32, cast to bfloat16 in host if needed
        OUT_ptr = OUT + q * OUT_stride_q + h * OUT_stride_h + d * OUT_stride_d
        tl.store(OUT_ptr, out_scalar)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, qo_indptr, kv_indptr, sm_scale):
        # Ensure CUDA and dtypes
        assert q.is_cuda and k.is_cuda and v.is_cuda, "Inputs must be on CUDA device"
        device = q.device
        # Original asserts
        assert q.shape[1] == 32 and k.shape[1] == 8 and v.shape[1] == 8, "Head counts must match constraints"
        assert q.shape[2] == 128 and k.shape[2] == 128 and v.shape[2] == 128, "Head dim must be 128"
        assert qo_indptr.shape[0] > 0 and kv_indptr.shape[0] > 0
        total_q = int(qo_indptr[-1].item())
        total_kv = int(kv_indptr[-1].item())
        assert total_q == q.shape[0], "total_q must equal q.shape[0]"
        assert total_kv == k.shape[0], "total_kv must equal k.shape[0]"

        # Output and lse buffers (float32 for lse, bfloat16 for output)
        output = torch.empty((total_q, 32, 128), dtype=torch.bfloat16, device=device)
        lse = torch.empty((total_q, 32), dtype=torch.float32, device=device)

        # Process each segment b
        for b in range(qo_indptr.numel() - 1):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())

            if q_start >= q_end or kv_start >= kv_end:
                continue

            num_q_tokens = q_end - q_start
            num_kv_tokens = kv_end - kv_start

            # Expand k, v by GQA ratio (32 // 8 = 4)
            k_expanded = k[kv_start:kv_end].repeat_interleave(4, dim=1).contiguous().to(torch.float32)
            v_expanded = v[kv_start:kv_end].repeat_interleave(4, dim=1).contiguous().to(torch.float32)

            # Segment output buffers
            output_seg = torch.empty((num_q_tokens, 32, 128), dtype=torch.float32, device=device)  # compute in fp32
            lse_seg = torch.empty((num_q_tokens, 32), dtype=torch.float32, device=device)

            # Launch Triton kernels: compute LOGITS, lse, output
            # 1) Compute logits LOGITS[q,h,k] = sum_d Q[q,h,d] * KEXP[k,h,d]
            LOGITS = torch.empty((num_q_tokens, 32, 128), dtype=torch.float32, device=device)

            # Grid: (num_q_tokens, 32)
            grid = (num_q_tokens, 32)
            _compute_logits_single_qh_kernel[grid](
                q[q_start:q_end], k_expanded, LOGITS,
                num_q_tokens, num_kv_tokens, 128,
                q[q_start:q_end].stride(0), q[q_start:q_end].stride(1), q[q_start:q_end].stride(2),
                k_expanded.stride(0), k_expanded.stride(1), k_expanded.stride(2),
                LOGITS.stride(0), LOGITS.stride(1), LOGITS.stride(2),
                BLOCK_D=128
            )

            # 2) Compute lse per (q, h)
            ln2 = 1.0 / math.log(2.0)
            delta = num_kv_tokens - num_q_tokens
            _lse_single_qh_kernel[(num_q_tokens, 32)](
                LOGITS, lse_seg,
                num_q_tokens, num_kv_tokens, 128,
                LOGITS.stride(0), LOGITS.stride(1), LOGITS.stride(2),
                lse_seg.stride(0), lse_seg.stride(1),
                ln2=ln2, delta=delta, BLOCK_K=128
            )

            # 3) Compute output = softmax(LOGITS with causal mask) @ V_expanded
            _softmax_output_single_qh_kernel[(num_q_tokens, 32)](
                LOGITS, v_expanded, lse_seg, output_seg,
                num_q_tokens, num_kv_tokens, 128,
                LOGITS.stride(0), LOGITS.stride(1), LOGITS.stride(2),
                v_expanded.stride(0), v_expanded.stride(1), v_expanded.stride(2),
                output_seg.stride(0), output_seg.stride(1), output_seg.stride(2),
                ln2=ln2, delta=delta, BLOCK_D=128
            )

            # Copy segment results into global output/lse
            output[q_start:q_end] = output_seg.to(torch.bfloat16)
            lse[q_start:q_end] = lse_seg

        return output, lse


def run(*args):
    return ModelNew()(*args)
