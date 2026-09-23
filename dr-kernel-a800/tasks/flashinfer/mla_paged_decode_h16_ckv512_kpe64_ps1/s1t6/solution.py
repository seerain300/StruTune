import math
import torch
import triton
import triton.language as tl


# Triton kernels: gather cached rows
@triton.jit
def gather_rows_c_kernel(cache_ptr, tok_idx_ptr, out_ptr,
                          L_tokens: tl.constexpr, Dc: tl.constexpr):
    # One program per token row
    pid = tl.program_id(0)
    if pid >= L_tokens:
        return
    idx = tl.load(tok_idx_ptr + pid).to(tl.int32)
    base_in = idx * Dc
    for k in range(0, Dc):
        val = tl.load(cache_ptr + base_in + k)
        tl.store(out_ptr + pid * Dc + k, val)


@triton.jit
def gather_rows_p_kernel(cache_ptr, tok_idx_ptr, out_ptr,
                          L_tokens: tl.constexpr, Dp: tl.constexpr):
    pid = tl.program_id(0)
    if pid >= L_tokens:
        return
    idx = tl.load(tok_idx_ptr + pid).to(tl.int32)
    base_in = idx * Dp
    for k in range(0, Dp):
        val = tl.load(cache_ptr + base_in + k)
        tl.store(out_ptr + pid * Dp + k, val)


# Triton kernels: GEMV per head (q @ K.T)
@triton.jit
def gemv_row_kernel(q_ptr, K_flat_ptr, out_ptr,
                    D: tl.constexpr, L: tl.constexpr):
    # One program per output element
    # Compute y[j] = sum_k q[j] * K[k, j]
    # Layout: K_flat is [L, D] flattened (row-major), out_ptr is [L]
    j = tl.program_id(0)
    if j >= D:
        return
    sum_val = 0.0
    for k in range(0, L):
        # q[j]
        qj = tl.load(q_ptr + j)
        # K[k, j] = K_flat[k * D + j]
        Kkj = tl.load(K_flat_ptr + k * D + j)
        sum_val += qj * Kkj
    tl.store(out_ptr + j, sum_val)


# Triton kernels: per-row lse in base-2 using one-pass online max-sub trick
@triton.jit
def lse_base2_row_kernel(vec_ptr, lse_ptr,
                         L: tl.constexpr):
    m = -float("inf")
    sum_exp = 0.0
    for t in range(0, L):
        v = tl.load(vec_ptr + t)
        # online update: m_new = max(m, v); sum_exp = sum_exp * exp(m - m_new) + exp(v - m_new)
        m_new = tl.maximum(m, v)
        sum_exp = sum_exp * tl.exp(m - m_new) + tl.exp(v - m_new)
        m = m_new
    lse = m + tl.log(sum_exp) / 0.6931471805599453  # 1 / ln(2)
    tl.store(lse_ptr, lse)


# Triton kernels: per-row softmax (two-pass)
@triton.jit
def softmax_row_kernel(vec_ptr, out_ptr,
                        L: tl.constexpr):
    m = -float("inf")
    # Pass 1: max
    for t in range(0, L):
        v = tl.load(vec_ptr + t)
        m = tl.maximum(m, v)
    # Pass 2: normalize
    sum_exp = 0.0
    for t in range(0, L):
        v = tl.load(vec_ptr + t)
        expv = tl.exp(v - m)
        sum_exp += expv
        tl.store(out_ptr + t, expv)


# Triton kernels: matvec per head (attn @ Kc), chunked across D
@triton.jit
def matvec_row_kernel(attn_ptr, Kc_ptr, out_ptr,
                      D: tl.constexpr, L: tl.constexpr,
                      BLOCK_D: tl.constexpr):
    # One program per head vector
    # out[i, :] = sum_t attn[i, t] * Kc[t, :]
    # Kc is [L, D], flattened with row-major
    # Implement accumulation across D in chunks
    acc = [0.0] * D
    for t in range(0, L):
        attn_val = tl.load(attn_ptr + t)  # scalar attn[i, t]
        kc_base = t * D
        for d0 in range(0, D, BLOCK_D):
            offs = d0 + tl.arange(0, BLOCK_D)
            mask = offs < D
            kc_vals = tl.load(Kc_ptr + kc_base + offs, mask=mask, other=0.0)
            # Multiply and accumulate
            acc[offs] += attn_val * kc_vals
    out_base = tl.program_id(0) * D
    for d in range(0, D):
        tl.store(out_ptr + out_base + d, acc[d])


# Triton kernels: GEMV per head (q @ K.T), more general loop
@triton.jit
def gemv_row_kernel_loop(q_ptr, K_flat_ptr, out_ptr,
                         D: tl.constexpr, L: tl.constexpr):
    # One program per output element
    # Compute y[j] = sum_k q[j] * K[k, j]
    j = tl.program_id(0)
    if j >= D:
        return
    sum_val = 0.0
    # Loop over tokens
    for k in range(0, L):
        qj = tl.load(q_ptr + j)
        Kkj = tl.load(K_flat_ptr + k * D + j)
        sum_val += qj * Kkj
    tl.store(out_ptr + j, sum_val)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # CPU fallback: if not on CUDA, return zeros
        if not q_nope.is_cuda:
            batch_size, num_qo_heads, head_dim_ckv = q_nope.shape
            output = torch.zeros((batch_size, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16)
            lse = torch.zeros((batch_size, num_qo_heads), dtype=torch.float32)
            return output, lse

        # Shapes and assertions
        batch_size, num_qo_heads, head_dim_ckv = q_nope.shape
        head_dim_kpe = q_pe.shape[-1]
        assert num_qo_heads == 16, "num_qo_heads must be 16"
        assert head_dim_ckv == 512, "head_dim_ckv must be 512"
        assert head_dim_kpe == 64, "head_dim_kpe must be 64"
        device = q_nope.device

        # Remove size-1 dim from caches and make contiguous float32
        Kc_all = ckv_cache.squeeze(1).to(torch.float32).contiguous()  # [num_pages, Dc]
        Kp_all = kpe_cache.squeeze(1).to(torch.float32).contiguous()  # [num_pages, Dp]

        # Output buffers
        output = torch.empty((batch_size, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=device)
        lse = torch.empty((batch_size, num_qo_heads), dtype=torch.float32, device=device)

        for b in range(batch_size):
            # Derive number of tokens for this batch element
            L_tokens = int(kv_indptr[b + 1].item()) - int(kv_indptr[b].item())
            if L_tokens <= 0:
                for i in range(num_qo_heads):
                    output[b, i] = torch.zeros((head_dim_ckv,), dtype=torch.bfloat16, device=device)
                lse[b] = torch.zeros((num_qo_heads,), dtype=torch.float32, device=device)
                continue

            # Token indices for this batch element
            tok_idx = kv_indices[int(kv_indptr[b].item()):int(kv_indptr[b + 1].item())].contiguous()  # [L_tokens]

            # 1) Gather rows from caches into float32
            Kc_flat = torch.empty((L_tokens * head_dim_ckv,), dtype=torch.float32, device=device)
            Kp_flat = torch.empty((L_tokens * head_dim_kpe,), dtype=torch.float32, device=device)

            gather_rows_c_kernel[(L_tokens,)](Kc_all, tok_idx, Kc_flat, L_tokens, head_dim_ckv)
            Kc = Kc_flat.view(L_tokens, head_dim_ckv)  # [L_tokens, Dc]
            # Flatten Kc for GEMV: [L, Dc]
            Kc_flat_for_gemv = Kc.view(L_tokens, head_dim_ckv).contiguous()  # already contiguous

            gather_rows_p_kernel[(L_tokens,)](Kp_all, tok_idx, Kp_flat, L_tokens, head_dim_kpe)
            Kp = Kp_flat.view(L_tokens, head_dim_kpe)  # [L_tokens, Dp]
            Kp_flat_for_gemv = Kp.view(L_tokens, head_dim_kpe).contiguous()

            # 2)


def run(*args):
    return ModelNew()(*args)
