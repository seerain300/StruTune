import torch
import math

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: compute logits[H, L] per head h
@triton.jit
def compute_logits_kernel(
    qn_ptr, Kc_ptr, qp_ptr, Kp_ptr,
    logits_ptr,
    H, L, K, Kp,
    scale,
    stride_qn_h, stride_qn_k,
    stride_Kc_l, stride_Kc_k,
    stride_qp_h, stride_qp_kp,
    stride_Kp_l, stride_Kp_kp,
    stride_log_h, stride_log_l,
    head: tl.constexpr,
):
    # One program per head, writes logits[head, :] of length L
    l = 0
    acc = tl.zeros((), dtype=tl.float32)
    while l < L:
        acc = 0.0
        k = 0
        while k < K:
            q_val = tl.load(qn_ptr + head * stride_qn_h + k * stride_qn_k)
            kc_row_ptr = Kc_ptr + l * stride_Kc_l + k * stride_Kc_k
            kc_val = tl.load(kc_row_ptr)
            acc += q_val * kc_val
            k += 1
        kprime = 0
        while kprime < Kp:
            q_val = tl.load(qp_ptr + head * stride_qp_h + kprime * stride_qp_kp)
            kp_row_ptr = Kp_ptr + l * stride_Kp_l + kprime * stride_Kp_kp
            kp_val = tl.load(kp_row_ptr)
            acc += q_val * kp_val
            kprime += 1
        acc = acc * scale
        tl.store(logits_ptr + head * stride_log_h + l * stride_log_l, acc)
        l += 1


# Triton kernel: compute logsumexp (base 2) for a row with causal masking
@triton.jit
def compute_lse_kernel(
    logits_ptr, lse_ptr,
    H, L,
    scale,  # logits are already scaled by sm_scale
    prefix_len,  # kv_len - q_len for this batch
    i,  # current query index within batch
    base2_scale,  # 1 / ln(2)
    stride_log_h, stride_log_l,
    stride_lse_h,
):
    h = 0
    while h < H:
        row_ptr = logits_ptr + h * stride_log_h
        row_max = -float("inf")
        l = 0
        while l < L:
            val = tl.load(row_ptr + l * stride_log_l)
            # causal: if l > (prefix_len + i), set to -inf
            if (l > (prefix_len + i)):
                val = -float("inf")
            if val > row_max:
                row_max = val
            l += 1
        sum_exp = 0.0
        l = 0
        while l < L:
            val = tl.load(row_ptr + l * stride_log_l)
            if (l > (prefix_len + i)):
                val = -float("inf")
            sum_exp += tl.exp(val - row_max)
            l += 1
        lse_val = tl.log(sum_exp) * base2_scale
        tl.store(lse_ptr + h * stride_lse_h, lse_val)
        h += 1


# Triton kernel: softmax along L per head, with causal masking (invalid positions contribute 0)
@triton.jit
def softmax_kernel(
    logits_ptr, attn_ptr,
    H, L,
    prefix_len,
    i,
    stride_log_h, stride_log_l,
    stride_attn_h, stride_attn_l,
):
    h = 0
    while h < H:
        row_ptr = logits_ptr + h * stride_log_h
        row_max = -float("inf")
        l = 0
        while l < L:
            val = tl.load(row_ptr + l * stride_log_l)
            if (l > (prefix_len + i)):
                val = -float("inf")
            if val > row_max:
                row_max = val
            l += 1
        sum_exp = 0.0
        l = 0
        while l < L:
            val = tl.load(row_ptr + l * stride_log_l)
            if (l > (prefix_len + i)):
                val = -float("inf")
            sum_exp += tl.exp(val - row_max)
            l += 1
        inv_sum = 1.0 / sum_exp
        l = 0
        row_attn_ptr = attn_ptr + h * stride_attn_h
        while l < L:
            val = tl.load(row_ptr + l * stride_log_l)
            if (l > (prefix_len + i)):
                soft = 0.0
            else:
                soft = tl.exp(val - row_max) * inv_sum
            tl.store(row_attn_ptr + l * stride_attn_l, soft)
            l += 1
        h += 1


# Triton kernel: GEMV out[h, K] = attn[h, :] @ Kc[L, K] per head
@triton.jit
def gemv_out_kernel(
    attn_ptr, Kc_ptr, out_ptr,
    H, L, K,
    stride_attn_h, stride_attn_l,
    stride_Kc_l, stride_Kc_k,
    stride_out_h, stride_out_k,
    head: tl.constexpr,
):
    k = 0
    acc = tl.zeros((), dtype=tl.float32)
    while k < K:
        # sum over l of attn[head, l] * Kc[l, k]
        l = 0
        while l < L:
            attn_val = tl.load(attn_ptr + head * stride_attn_h + l * stride_attn_l)
            kc_val = tl.load(Kc_ptr + l * stride_Kc_l + k * stride_Kc_k)
            acc += attn_val * kc_val
            l += 1
        tl.store(out_ptr + head * stride_out_h + k * stride_out_k, acc)
        k += 1


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Triton-only version requires CUDA and Triton
        if not TRITON_AVAILABLE or not torch.cuda.is_available():
            raise RuntimeError("Triton or CUDA not available for Triton version.")

        # Move tensors to CUDA and ensure contiguity
        q_nope = q_nope.to('cuda').contiguous()
        q_pe = q_pe.to('cuda').contiguous()
        ckv_cache = ckv_cache.to('cuda').contiguous()
        kpe_cache = kpe_cache.to('cuda').contiguous()
        qo_indptr = qo_indptr.to('cuda').contiguous()
        kv


def run(*args):
    return ModelNew()(*args)
