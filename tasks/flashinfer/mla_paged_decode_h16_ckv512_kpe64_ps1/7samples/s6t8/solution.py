import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Kernel 1: Compute per-head logits vector for all tokens: logits = qn @ Kc.T + qp @ Kp.T
# Input:
#   qn_ptr: [Hc] float32 (single head's q_nope)
#   qp_ptr: [Hp] float32 (single head's q_pe)
#   Kc_ptr: [L_tokens, Hc] float32
#   Kp_ptr: [L_tokens, Hp] float32
#   logits_ptr: [L_tokens] float32
#   sm_scale: float32
# Launch: grid = (1,)
@triton.jit
def matmul_add_row_kernel(
    qn_ptr, qp_ptr, Kc_ptr, Kp_ptr, logits_ptr,
    L_tokens: tl.constexpr, Hc: tl.constexpr, Hp: tl.constexpr, sm_scale: tl.constexpr,
    Kc_stride0, Kc_stride1, Kp_stride0, Kp_stride1,
    BLOCK_K: tl.constexpr
):
    acc = tl.zeros((L_tokens,), dtype=tl.float32)
    # Loop over tokens in chunks of BLOCK_K
    for k_start in range(0, L_tokens, BLOCK_K):
        offs = k_start + tl.arange(0, BLOCK_K)
        mask = offs < L_tokens
        # Load Kc and Kp slices as [BLOCK_K]
        Kc_row = tl.load(Kc_ptr + offs * Kc_stride0, mask=mask, other=0.0)  # shape [BLOCK_K]
        Kp_row = tl.load(Kp_ptr + offs * Kp_stride0, mask=mask, other=0.0)  # shape [BLOCK_K]
        # qn and qp are scalars for this head
        qn_val = tl.load(qn_ptr + 0)  # single scalar
        qp_val = tl.load(qp_ptr + 0)  # single scalar
        # Accumulate contributions: qn * sum(Kc_row) + qp * sum(Kp_row)
        Kc_sum = tl.sum(Kc_row, axis=0)
        Kp_sum = tl.sum(Kp_row, axis=0)
        acc += qn_val * Kc_sum + qp_val * Kp_sum
    # Scale logits
    acc = acc * sm_scale
    # Store logits
    # We write acc into logits_ptr[0:L_tokens]
    for i in range(0, L_tokens):
        tl.store(logits_ptr + i, acc[i])


# Kernel 2: Row-wise softmax with logsumexp in Triton (two-pass). Input: logits_ptr [L_tokens], output: lse_ptr scalar and attn_ptr [L_tokens]
@triton.jit
def softmax_logsumexp_row_kernel(
    logits_ptr, attn_ptr, lse_ptr,
    L_tokens: tl.constexpr,
    BLOCK: tl.constexpr  # chunk size for reduction passes
):
    # Pass 1: compute max
    max_val = -float('inf')
    for i in range(0, L_tokens, BLOCK):
        offs = i + tl.arange(0, BLOCK)
        mask = offs < L_tokens
        x = tl.load(logits_ptr + offs, mask=mask, other=-float('inf'))
        # compute max of this chunk
        chunk_max = tl.max(x, axis=0)
        max_val = tl.maximum(max_val, chunk_max)
    # Pass 2: compute sum(exp(x - max))
    sum_exp = 0.0
    for i in range(0, L_tokens, BLOCK):
        offs = i + tl.arange(0, BLOCK)
        mask = offs < L_tokens
        x = tl.load(logits_ptr + offs, mask=mask, other=-float('inf'))
        expx = tl.exp(x - max_val)
        sum_exp += tl.sum(expx, axis=0)
    # lse = log(sum_exp) / log(2)
    lse = tl.log(sum_exp) / 1.4426950408889634  # 1 / ln(2)
    tl.store(lse_ptr, lse)
    # Write attn = exp(logits - max) / sum_exp (not used on host)
    for i in range(0, L_tokens, BLOCK):
        offs = i + tl.arange(0, BLOCK)
        mask = offs < L_tokens
        x = tl.load(logits_ptr + offs, mask=mask, other=-float('inf'))
        attn_chunk = tl.exp(x - max_val) / sum_exp
        tl.store(attn_ptr + offs, attn_chunk, mask=mask)


# Kernel 3: GEMV for output: out_row = attn_row @ Kc (per head). Input: attn_ptr [L_tokens], Kc_ptr [L_tokens, Hc], out_ptr [Hc]
@triton.jit
def matvec_row_kernel(
    attn_ptr, Kc_ptr, out_ptr,
    L_tokens: tl.constexpr, Hc: tl.constexpr,
    Kc_stride0, Kc_stride1,
    out_stride,
    BLOCK_N: tl.constexpr, BLOCK_M: tl.constexpr
):
    # Grid over output columns in chunks of BLOCK_N
    # We pass out_stride=1; out_ptr is [Hc] contiguous
    for n_start in range(0, Hc, BLOCK_N):
        out_offs = n_start + tl.arange(0, BLOCK_N)
        out_mask = out_offs < Hc
        acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
        # Loop over tokens in chunks
        for m_start in range(0, L_tokens, BLOCK_M):
            offs_m = m_start + tl.arange(0, BLOCK_M)
            m_mask = offs_m < L_tokens
            attn_chunk = tl.load(attn_ptr + offs_m, mask=m_mask, other=0.0)  # [BLOCK_M]
            Kc_chunk = tl.load(Kc_ptr + offs_m[:, None] * Kc_stride0 + 0 * Kc_stride1, mask=m_mask[:, None], other=0.0)  # [BLOCK_M, 1] but stride1 is ignored since other dim is 0
            # Multiply and reduce over M
            acc += tl.sum(attn_chunk[:, None] * Kc_chunk, axis=0)  # [BLOCK_N]
        # Store
        tl.store(out_ptr + out_offs, acc, mask=out_mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure inputs are CUDA tensors and contiguous
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda, "All inputs must be CUDA tensors"
        q_nope = q_nope.contiguous()
        q_pe = q_pe.contiguous()
        # Squeeze caches: original caches have shape (num_pages, 1, D). We want (num_pages, D)
        Kc_all = ckv_cache.squeeze(1).contiguous().to(torch.float32)
        Kp_all = kpe_cache.squeeze(1).contiguous().to(torch.float32)

        batch_size = q_nope.shape[0]
        num_qo_heads = q_nope.shape[1]
        head_dim_ckv = q_nope.shape[2]
        head_dim_kpe = q_pe.shape[2]

        # Gather token indices per batch element
        # kv_indptr: [len_indptr], len_indptr == batch_size + 1
        len_indptr = kv_indptr.shape[0]
        assert len_indptr == batch_size + 1, "kv_indptr length must be batch_size + 1"
        output = torch.empty((batch_size, num_qo_heads, head_dim_ckv), dtype=torch.float32, device=q_nope.device)  # will store float32, cast to bfloat16 at end
        lse = torch.empty((batch_size, num_qo_heads), dtype=torch.float32, device=q_nope.device)

        for b in range(batch_size):
            # Compute tok_idx for this batch
            page_beg = int(kv_indptr[b].item())
            page_end = int(kv_indptr[b + 1].item())
            L_tokens = page_end - page_beg
            if L_tokens <= 0:
                # No valid tokens for this batch element
                output[b] = torch.zeros((num_qo_heads, head_dim_ckv), dtype=torch.float32, device=q_nope.device)
                lse[b] = -float('inf')
                continue

            tok_idx = kv_indices[page_beg:page_end].to(torch.int32)
            # Gather Kc and Kp for these tokens
            Kc = Kc_all[tok_idx]  # [L_tokens, head_dim_ckv]
            Kp = Kp_all[tok_idx]  # [L_tokens, head_dim_kpe]

            # Per-head vectors
            for h in range(num_qo_heads):
                # Prepare pointers for this head
                qn_ptr = q_nope[b, h].contiguous().to(torch.float32)  # [Hc]
                qp_ptr = q_pe[b, h].contiguous().to(torch.float32)   # [Hp]
                Kc_ptr = Kc.contiguous()  # [L_tokens, Hc]
                Kp_ptr = Kp.contiguous()  # [L_tokens, Hp]
                logits = torch.empty((L_tokens,), dtype=torch.float32, device=q_nope.device)

                # Launch matmul_add_row_kernel to compute logits_scaled
                matmul_add_row_kernel[(1,)](
                    qn_ptr, qp_ptr, Kc_ptr, Kp_ptr, logits,
                    L_tokens=L_tokens, Hc=head_dim_ckv, Hp=head_dim_kpe, sm_scale=float(sm_scale),
                    Kc_stride0=Kc_ptr.stride(0), Kc_stride1=Kc_ptr.stride(1),
                    Kp_stride0=Kp_ptr.stride(0), Kp_stride1=Kp_ptr.stride(1),
                    BLOCK_K=128
                )

                # Launch softmax_logsumexp_row_kernel to compute lse
                attn = torch.empty((L_tokens,), dtype=torch.float32, device=q_nope.device)
                lse_vec = torch.empty((1,), dtype=torch.float32, device=q_nope.device)
                softmax_logsumexp_row_kernel[(1,)](
                    logits, attn, lse_vec,
                    L_tokens=L_tokens,
                    BLOCK=128
                )
                lse[b, h] = lse_vec[0]

                # Launch matvec_row_kernel to compute output[b, h, :]
                out_row = torch.empty((head_dim_ckv,), dtype=torch.float32, device=q_nope.device)
                matvec_row_kernel[(triton.cdiv(head_dim_ckv, 128),)](
                    attn, Kc_ptr, out_row,
                    L_tokens=L_tokens, Hc=head_dim_ckv,
                    Kc_stride0=Kc_ptr.stride(0), Kc_stride1=Kc_ptr.stride(1),
                    out_stride=1,
                    BLOCK_N=128, BLOCK_M=64
                )
                output[b, h, :] = out_row

        # Return output in bfloat16 to match original dtype, lse as float32
        output_bf16 = output.to(torch.bfloat16)
        return output_bf16, lse


def run(*args):
    return ModelNew()(*args)
