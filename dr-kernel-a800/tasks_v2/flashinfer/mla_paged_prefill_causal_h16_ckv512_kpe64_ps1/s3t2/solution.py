import torch
import math

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Kernel: compute logits[H, L] per head h, vectorized over L
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
    # One program computes the logits vector for a single head h
    l = 0
    acc = tl.zeros((), dtype=tl.float32)
    while l < L:
        # Accumulate qn @ Kc.T for this l
        k = 0
        while k < K:
            q_val = tl.load(qn_ptr + head * stride_qn_h + k * stride_qn_k)
            kc_val = tl.load(Kc_ptr + l * stride_Kc_l + k * stride_Kc_k)
            acc += q_val * kc_val
            k += 1
        # Accumulate qp @ Kp.T for this l
        kprime = 0
        while kprime < Kp:
            q_val = tl.load(qp_ptr + head * stride_qp_h + kprime * stride_qp_kp)
            kp_val = tl.load(Kp_ptr + l * stride_Kp_l + kprime * stride_Kp_kp)
            acc += q_val * kp_val
            kprime += 1
        acc = acc * scale
        tl.store(logits_ptr + head * stride_log_h + l * stride_log_l, acc)
        l += 1


# Kernel: compute logsumexp (base 2) for a row with causal masking, per head
@triton.jit
def compute_lse_kernel(
    logits_ptr, lse_ptr,
    H, L,
    scale,  # logits are already scaled by sm_scale
    prefix_len,  # kv_len - q_len for this batch
    i,  # current query index within batch element
    base2_scale,  # 1 / ln(2)
    stride_log_h, stride_log_l,
    stride_lse_h,
):
    h = 0
    while h < H:
        row_ptr = logits_ptr + h * stride_log_h
        # Compute max with causal mask: invalid positions set to -inf
        row_max = -float("inf")
        l = 0
        while l < L:
            val = tl.load(row_ptr + l * stride_log_l)
            if (l > (prefix_len + i)):
                val = -float("inf")
            if val > row_max:
                row_max = val
            l += 1
        # Compute sum of exp(logits - max) with mask
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


# Kernel: softmax along L per head, with causal masking (invalid positions contribute 0)
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
        # Compute row_max with mask
        row_max = -float("inf")
        l = 0
        while l < L:
            val = tl.load(row_ptr + l * stride_log_l)
            if (l > (prefix_len + i)):
                val = -float("inf")
            if val > row_max:
                row_max = val
            l += 1
        # Compute sum of exp with mask
        sum_exp = 0.0
        l = 0
        while l < L:
            val = tl.load(row_ptr + l * stride_log_l)
            if (l > (prefix_len + i)):
                val = -float("inf")
            sum_exp += tl.exp(val - row_max)
            l += 1
        inv_sum = 1.0 / sum_exp
        # Write softmax to attn buffer
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


# Kernel: GEMV out[h, K] = attn[h, :] @ Kc[h, :]
@triton.jit
def gemv_out_kernel(
    attn_ptr, Kc_ptr, out_ptr,
    H, L, K,
    stride_attn_h, stride_attn_l,
    stride_Kc_l, stride_Kc_k,
    stride_out_h, stride_out_k,
    head: tl.constexpr,
):
    # One program per head; compute out[h, :] of length K
    k = 0
    acc = tl.zeros((), dtype=tl.float32)
    while k < K:
        # Sum over l of attn[head, l] * Kc[l, k]
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
        kv_indptr = kv_indptr.to('cuda').contiguous()
        kv_indices = kv_indices.to('cuda').contiguous()

        # Dimensions
        total_q, num_qo_heads, head_dim_ckv = q_nope.shape
        _, _, head_dim_kpe = q_pe.shape
        assert num_qo_heads == 16, "num_qo_heads must be 16"
        assert head_dim_ckv == 512, "head_dim_ckv must be 512"
        assert head_dim_kpe == 64, "head_dim_kpe must be 64"

        # Squeeze cache singleton dim
        Kc_all = ckv_cache.squeeze(1).contiguous()  # [num_pages, 512]
        Kp_all = kpe_cache.squeeze(1).contiguous()  # [num_pages, 64]

        device = q_nope.device
        H = num_qo_heads
        K = head_dim_ckv
        Kp = head_dim_kpe

        # Output buffers
        output = torch.empty((total_q, H, K), dtype=torch.bfloat16, device=device)
        lse = torch.full((total_q, H), -float("inf"), dtype=torch.float32, device=device)

        # Process batches using PyTorch loops (no torch compute in kernels)
        b = 0
        while b < (len(qo_indptr) - 1):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            q_len = q_end - q_start

            if q_len == 0:
                b += 1
                continue

            # Compute token indices and gather Kc, Kp for this batch
            page_beg = int(kv_indptr[b].item())
            page_end = int(kv_indptr[b + 1].item())
            L = page_end - page_beg
            if L == 0:
                b += 1
                continue

            tok_idx = kv_indices[page_beg:page_end].to(torch.int32).to(device)
            Kc = Kc_all[tok_idx].to(torch.float32).contiguous()  # [L, K]
            Kp = Kp_all[tok_idx].to(torch.float32).contiguous()  # [L, Kp]

            # Initialize per-query outputs
            for i in range(q_len):
                q_start_i = q_start + i

                # Prepare qn and qp for this query (H, K) and (H, Kp)
                qn = q_nope[q_start_i].to(torch.float32).contiguous()  # [H, K]
                qp = q_pe[q_start_i].to(torch.float32).contiguous()    # [H, Kp]

                # Allocate intermediate buffers (per head, vector of length L)
                logits = torch.empty((H, L), dtype=torch.float32, device=device)
                attn = torch.empty((H, L), dtype=torch.float32, device=device)

                # Launch Triton kernels
                # 1) compute logits
                # Strides for qn: (H, K) contiguous => stride_h=K, stride_k=1
                stride_qn_h, stride_qn_k = K, 1
                # Strides for Kc: (L, K) => stride_l=K, stride_k=1
                stride_Kc_l, stride_Kc_k = K, 1
                # Strides for qp: (H, Kp) => stride_h=Kp, stride_kp=1
                stride_qp_h, stride_qp_kp = Kp, 1
                # Strides for Kp: (L, Kp) => stride_l=Kp, stride_kp=1
                stride_Kp_l, stride_Kp_kp = Kp, 1
                # Strides for logits: (H, L) => stride_h=L, stride_l=1
                stride_log_h, stride_log_l = L, 1

                # Launch compute_logits_kernel: one program per head
                grid = (H,)
                compute_logits_kernel[grid](
                    qn, Kc, qp, Kp,
                    logits,
                    H, L, K, Kp,
                    sm_scale,
                    stride_qn_h, stride_qn_k,
                    stride_Kc_l, stride_Kc_k,
                    stride_qp_h, stride_qp_kp,
                    stride_Kp_l, stride_Kp_kp,
                    stride_log_h, stride_log_l,
                )

                # 2) compute lse
                # Strides for lse: (H,) => stride_h=1
                stride_lse_h = 1
                grid_lse = (H,)
                lse[q_start_i] = torch.zeros(H, dtype=torch.float32, device=device)  # placeholder, will be overwritten
                compute_lse_kernel[grid_lse](
                    logits, lse[q_start_i],
                    H, L,
                    sm_scale,
                    (L - q_len),  # prefix_len = kv_len - q_len
                    i,
                    1.0 / math.log(2.0),  # base2_scale
                    stride_log_h, stride_log_l,
                    stride_lse_h,
                )

                # 3) softmax
                stride_attn_h, stride_attn_l = L, 1
                grid_softmax = (H,)
                softmax_kernel[grid_softmax](
                    logits, attn,
                    H, L,
                    (L - q_len),
                    i,
                    stride_log_h, stride_log_l,
                    stride_attn_h, stride_attn_l,
                )

                # 4) GEMV to produce out[h, K]
                # Strides for out: (H, K) => stride_h=K, stride_k=1
                stride_out_h, stride_out_k = K, 1
                grid_gemv = (H,)
                out_vec = torch.empty((H, K), dtype=torch.float32, device=device)
                gemv_out_kernel[grid_gemv](
                    attn, Kc, out_vec,
                    H, L, K,
                    stride_attn_h, stride_attn_l,
                    stride_Kc_l, stride_Kc_k,
                    stride_out_h, stride_out_k,
                    head=0,  # we launch one program per head; H=16, so we loop externally
                )

                # Store output as bfloat16
                output[q_start_i] = out_vec.to(torch.bfloat16)

            b += 1

        return output, lse


def run(*args):
    return ModelNew()(*args)
