import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


@triton.jit
def matmul_add_row_kernel(
    qn_ptr, qp_ptr, Kc_ptr, Kp_ptr, logits_ptr, lse_ptr,
    Hc: tl.constexpr, Hp: tl.constexpr, L: tl.constexpr,
    qn_stride, qp_stride,
    Kc_stride0, Kc_stride1,
    Kp_stride0, Kp_stride1,
    out_stride,
    sm_scale: tl.float32,
    BLOCK_K: tl.constexpr
):
    # One program per (b, h). We don't take h as program_id since we loop h in host.
    # Loop over K dimension in chunks and accumulate.
    acc = tl.zeros([Hc], dtype=tl.float32)
    for k in range(0, L, BLOCK_K):
        offs = k + tl.arange(0, BLOCK_K)
        mask = offs < L
        # Load qn_chunk and qp_chunk
        qn_chunk = tl.load(qn_ptr + offs * qn_stride, mask=mask, other=0.0)
        qp_chunk = tl.load(qp_ptr + offs * qp_stride, mask=mask, other=0.0)
        # Compute contributions: qn_chunk @ Kc[offs, :] and qp_chunk @ Kp[offs, :]
        # Kc and Kp are (L, dim) with strides (Kc_stride0, Kc_stride1)
        # We want Kc[offs, :] which is along dim axis; load vectors of length Hc for each offs
        kc_vec = tl.zeros([Hc], dtype=tl.float32)
        kp_vec = tl.zeros([Hp], dtype=tl.float32)
        # We load each vector element by scalar loop over dim index j
        # Note: Triton doesn't support arbitrary 2D indexing in vectorized form; emulate with loop.
        for j in range(Hc):
            kc_vec[j] = tl.load(Kc_ptr + offs * Kc_stride0 + j * Kc_stride1, mask=mask, other=0.0)
        for j in range(Hp):
            kp_vec[j] = tl.load(Kp_ptr + offs * Kp_stride0 + j * Kp_stride1, mask=mask, other=0.0)
        # Accumulate dot products
        acc += tl.sum(qn_chunk * kc_vec, axis=0) + tl.sum(qp_chunk * kp_vec, axis=0)
    # Scale by sm_scale
    acc *= sm_scale
    # Store logits and update max for lse
    # Write acc to logits
    for i in range(Hc):
        tl.store(logits_ptr + i * out_stride, acc[i])
    # Compute lse = log(sum(exp(x - max))) / log(2)
    # We need max of acc; use tl.max over vector
    row_max = tl.max(acc, axis=0)
    sum_exp = 0.0
    for i in range(Hc):
        sum_exp += tl.exp(acc[i] - row_max)
    lse_val = tl.log(sum_exp) / 1.4426950408889634  # 1 / ln(2)
    tl.store(lse_ptr, lse_val)


@triton.jit
def softmax_row_kernel(
    logits_ptr, attn_ptr, lse_ptr,
    Hc: tl.constexpr, L: tl.constexpr,
    in_stride, out_stride,
    BLOCK: tl.constexpr
):
    # One program per row
    # Pass 1: compute max
    row_max = -float('inf')
    for i in range(0, Hc):
        x = tl.load(logits_ptr + i * in_stride)
        if x > row_max:
            row_max = x
    # Pass 2: compute sum and write attn; also compute lse
    sum_exp = 0.0
    for i in range(0, Hc):
        x = tl.load(logits_ptr + i * in_stride)
        e = tl.exp(x - row_max)
        tl.store(attn_ptr + i * out_stride, e)
        sum_exp += e
    lse_val = tl.log(sum_exp) / 1.4426950408889634  # log(2)
    tl.store(lse_ptr, lse_val)


@triton.jit
def matvec_row_kernel(
    attn_ptr, K_ptr, out_ptr,
    Hc: tl.constexpr, L: tl.constexpr,
    K_stride0, K_stride1,
    out_stride,
    BLOCK_N: tl.constexpr, BLOCK_M: tl.constexpr
):
    # One program per output column chunk
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    for m in range(0, L, BLOCK_M):
        offs_m = m + tl.arange(0, BLOCK_M)
        mask_m = offs_m < L
        # Load attn slice: attn[offs_m]
        attn_vec = tl.load(attn_ptr + offs_m * 1, mask=mask_m, other=0.0)  # stride 1 assumed for attn
        # Load Kc slice: K[offs_m, offs_n] where K is (L, Hc)
        K_sub = tl.load(K_ptr + offs_m[:, None] * K_stride0 + offs_n[None, :] * K_stride1,
                        mask=mask_m[:, None], other=0.0)
        # Multiply and reduce over m
        acc += tl.sum(K_sub * attn_vec[:, None], axis=0)
    # Store result
    tl.store(out_ptr + offs_n * out_stride, acc, mask=offs_n < Hc)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure CUDA and contiguity
        device = q_nope.device
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda
        # Squeeze cache to (num_pages, dim)
        Kc_all = ckv_cache.squeeze(1).contiguous().to(torch.float32)
        Kp_all = kpe_cache.squeeze(1).contiguous().to(torch.float32)

        batch_size, num_qo_heads, head_dim_ckv = q_nope.shape
        _, _, head_dim_kpe = q_pe.shape

        # Prepare output and lse
        output = torch.empty((batch_size, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=device)
        lse = torch.empty((batch_size, num_qo_heads), dtype=torch.float32, device=device)

        # Loop over batch elements
        for b in range(batch_size):
            # Compute tok_idx for this batch
            page_beg = int(kv_indptr[b].item())
            page_end = int(kv_indptr[b + 1].item())
            if page_beg >= page_end:
                # No KV cache for this batch element
                output[b].zero_()
                lse[b] = torch.tensor(-float('inf'), dtype=torch.float32, device=device)
                continue

            tok_idx = kv_indices[page_beg:page_end].to(torch.int32).to(device)  # indices are int32
            L_tokens = tok_idx.numel()

            # Gather Kc and Kp for these tokens
            Kc = Kc_all[tok_idx].contiguous()  # (L_tokens, head_dim_ckv)
            Kp = Kp_all[tok_idx].contiguous()  # (L_tokens, head_dim_kpe)

            # Per-head computations
            for h in range(num_qo_heads):
                # Prepare qn and qp for this head
                qn = q_nope[b, h, :].contiguous().to(torch.float32)  # (head_dim_ckv,)
                qp = q_pe[b, h, :].contiguous().to(torch.float32)   # (head_dim_kpe,)

                # Logits buffer and lse buffer
                logits = torch.empty(head_dim_ckv, dtype=torch.float32, device=device)
                lse_buf = torch.empty((), dtype=torch.float32, device=device)  # scalar

                # Kernel 1: matmul_add_row_kernel computes logits and lse for this (b,h)
                # We pass Hc=512, Hp=64, L=L_tokens as constexprs
                # BLOCK_K controls reduction chunk size; choose 128 for typical dims
                matmul_add_row_kernel[(1,)](
                    qn, qp, Kc, Kp, logits, lse_buf,
                    Hc=head_dim_ckv, Hp=head_dim_kpe, L=L_tokens,
                    qn_stride=1, qp_stride=1,
                    Kc_stride0=Kc.stride(0), Kc_stride1=Kc.stride(1),
                    Kp_stride0=Kp.stride(0), Kp_stride1=Kp.stride(1),
                    out_stride=1,
                    sm_scale=sm_scale,
                    BLOCK_K=128,
                    num_warps=2, num_stages=2
                )

                # Kernel 2: softmax_row_kernel computes attn and lse from logits
                attn = torch.empty(head_dim_ckv, dtype=torch.float32, device=device)
                lse_scalar = torch.empty((), dtype=torch.float32, device=device)
                softmax_row_kernel[(1,)](
                    logits, attn, lse_scalar,
                    Hc=head_dim_ckv, L=L_tokens,
                    in_stride=1, out_stride=1,
                    BLOCK=128,
                    num_warps=2, num_stages=2
                )
                lse[b, h] = lse_scalar

                # Kernel 3: matvec_row_kernel computes output row = attn @ Kc
                out_row = torch.empty(head_dim_ckv, dtype=torch.float32, device=device)
                BLOCK_N = 128
                matvec_row_kernel[(triton.cdiv(head_dim_ckv, BLOCK_N),)](
                    attn, Kc, out_row,
                    Hc=head_dim_ckv, L=L_tokens,
                    K_stride0=Kc.stride(0), K_stride1=Kc.stride(1),
                    out_stride=1,
                    BLOCK_N=BLOCK_N, BLOCK_M=64,
                    num_warps=2, num_stages=2
                )
                output[b, h, :] = out_row.to(torch.bfloat16)

        return output, lse


def run(*args):
    return ModelNew()(*args)
