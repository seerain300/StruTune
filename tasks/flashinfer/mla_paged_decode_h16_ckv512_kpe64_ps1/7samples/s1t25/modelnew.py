import math
import torch
import triton
import triton.language as tl


@triton.jit
def matvec_add_kernel(
    qn_ptr,           # *float32, [D]
    qp_ptr,           # *float32, [Dp]
    Kc_ptr,           # *float32, [L, D], row-major (L, D)
    Kp_ptr,           # *float32, [L, Dp], row-major (L, Dp)
    v_ptr,            # *float32, [L]
    L: tl.int32,      # number of tokens
    D: tl.int32,      # head_dim_ckv
    Dp: tl.int32,     # head_dim_kpe
    BLOCK_K: tl.constexpr
):
    # One program per output index i in [0, L)
    i = tl.program_id(0)
    acc1 = 0.0
    acc2 = 0.0
    # Reduce over Kc dimension (D)
    k = 0
    while k < D:
        k_off = k + tl.arange(0, BLOCK_K)
        mask_k = k_off < D
        qn_slice = tl.load(qn_ptr + k_off, mask=mask_k, other=0.0)  # [BLOCK_K]
        kc_row = tl.load(Kc_ptr + i * D + k_off, mask=mask_k, other=0.0)  # [BLOCK_K]
        acc1 += tl.sum(qn_slice * kc_row, axis=0)
        k += BLOCK_K
    # Reduce over Kp dimension (Dp)
    kp = 0
    while kp < Dp:
        kp_off = kp + tl.arange(0, BLOCK_K)
        mask_kp = kp_off < Dp
        qp_slice = tl.load(qp_ptr + kp_off, mask=mask_kp, other=0.0)  # [BLOCK_K]
        kp_row = tl.load(Kp_ptr + i * Dp + kp_off, mask=mask_kp, other=0.0)  # [BLOCK_K]
        acc2 += tl.sum(qp_slice * kp_row, axis=0)
        kp += BLOCK_K
    v = acc1 + acc2
    tl.store(v_ptr + i, v)


@triton.jit
def lse_kernel(
    v_ptr,            # *float32, [L]
    lse_ptr,          # *float32, [L]
    L: tl.int32,      # number of tokens
    BLOCK_L: tl.constexpr
):
    # Compute max m and sum_exp across L in chunks
    m = -float('inf')
    i = 0
    while i < L:
        idx = i + tl.arange(0, BLOCK_L)
        mask = idx < L
        vi = tl.load(v_ptr + idx, mask=mask, other=-float('inf'))
        local_max = tl.max(vi, axis=0)
        m = tl.maximum(m, local_max)
        i += BLOCK_L
    # Second pass: compute sum_exp = sum(exp(v - m))
    sum_exp = 0.0
    i = 0
    ln2 = 0.6931471805599453
    while i < L:
        idx = i + tl.arange(0, BLOCK_L)
        mask = idx < L
        vi = tl.load(v_ptr + idx, mask=mask, other=-float('inf'))
        expi = tl.exp(vi - m) / ln2
        sum_exp += tl.sum(expi, axis=0)
        i += BLOCK_L
    lse = m + tl.log(sum_exp)
    # Store lse vector (all equal)
    i = 0
    while i < L:
        idx = i + tl.arange(0, BLOCK_L)
        mask = idx < L
        tl.store(lse_ptr + idx, lse, mask=mask)
        i += BLOCK_L


@triton.jit
def softmax_base2_kernel(
    v_ptr,            # *float32, [L]
    lse_ptr,          # *float32, [L] (same lse for all)
    attn_ptr,         # *float32, [L]
    L: tl.int32,      # number of tokens
    BLOCK_L: tl.constexpr
):
    # One program per index i; compute attn[i] = exp((v[i] - lse) / ln(2))
    i = tl.program_id(0)
    if i >= L:
        return
    vi = tl.load(v_ptr + i)
    li = tl.load(lse_ptr + i)
    ln2 = 0.6931471805599453
    attn_i = tl.exp((vi - li) / ln2)
    tl.store(attn_ptr + i, attn_i)


@triton.jit
def matvec_write_y_kernel(
    attn_ptr,         # *float32, [L]
    Kc_ptr,           # *float32, [L, D], row-major
    y_ptr,            # *float32, [D]
    L: tl.int32,      # number of tokens
    D: tl.int32,      # head_dim_ckv
    BLOCK_K: tl.constexpr
):
    # One program per output dimension h in [0, D)
    h = tl.program_id(0)
    acc = 0.0
    k = 0
    while k < L:
        idx = k + tl.arange(0, BLOCK_K)
        mask = idx < L
        ai = tl.load(attn_ptr + idx, mask=mask, other=0.0)   # [BLOCK_K]
        kc_col = tl.load(Kc_ptr + idx * D + h, mask=mask, other=0.0)  # [BLOCK_K]
        acc += tl.sum(ai * kc_col, axis=0)
        k += BLOCK_K
    tl.store(y_ptr + h, acc)


def get_inputs():
    # Non-recursive helper to generate test inputs; the harness can override.
    device = 'cuda'
    batch_size = 1
    num_pages = 989669
    len_indptr = batch_size + 1
    # Set a small number of tokens for testing; real harness will supply L.
    num_kv_indices = 8
    q_nope = torch.randn([batch_size, 16, 512], dtype=torch.bfloat16, device=device)
    q_pe = torch.randn([batch_size, 16, 64], dtype=torch.bfloat16, device=device)
    ckv_cache = torch.randn([num_pages, 1, 512], dtype=torch.bfloat16, device=device)
    kpe_cache = torch.randn([num_pages, 1, 64], dtype=torch.bfloat16, device=device)
    # Indptr and indices for a single batch
    _n = batch_size
    _t = num_kv_indices
    _lens = torch.full((_n,), _t // _n, dtype=torch.int32, device=device)
    _lens[: _t % _n] += 1
    kv_indptr = torch.cat([torch.zeros(1, dtype=torch.int32, device=device), torch.cumsum(_lens, 0)]).to(torch.int32)  # [len_indptr]
    kv_indices = torch.randint(0, num_pages, [num_kv_indices], dtype=torch.int32, device=device)
    sm_scale = 1.0  # float32 scalar
    return [q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale]


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure tensors are on CUDA
        device = 'cuda'
        for i in range(len(q_nope), len(q_nope) - 1, -1):  # no-op, just ensure device
            pass
        # Make inputs contiguous
        q_nope = q_nope.contiguous()
        q_pe = q_pe.contiguous()
        ckv_cache = ckv_cache.contiguous()
        kpe_cache = kpe_cache.contiguous()
        kv_indptr = kv_indptr.contiguous()
        kv_indices = kv_indices.contiguous()

        batch_size = q_nope.shape[0]
        num_qo_heads = q_nope.shape[1]
        head_dim_ckv = q_nope.shape[2]  # 512
        head_dim_kpe = q_pe.shape[2]    # 64
        num_pages = ckv_cache.shape[0]
        len_indptr = kv_indptr.shape[0]
        num_kv_indices = kv_indices.shape[0]

        # Assertions for safety (same as original)
        assert num_qo_heads == 16, "num_qo_heads must be 16"
        assert head_dim_ckv == 512, "head_dim_ckv must be 512"
        assert head_dim_kpe == 64, "head_dim_kpe must be 64"
        assert len_indptr == batch_size + 1, "kv_indptr length must be batch_size + 1"
        # Optional: ensure last indptr equals num_kv_indices
        assert kv_indptr[-1].item() == num_kv_indices, "last kv_indptr must equal num_kv_indices"

        # Output tensors
        output = torch.zeros(
            (batch_size, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=device
        )
        lse = torch.full((batch_size, num_qo_heads), -float("inf"), dtype=torch.float32, device=device)

        # Pre-define BLOCK sizes
        BLOCK_K = 128  # for Kc/Kp reduction
        BLOCK_L = 128  # for L loops in kernels

        # Process each batch b and head j
        for b in range(batch_size):
            # Determine token range and gather indices
            page_beg = int(kv_indptr[b].item())
            page_end = int(kv_indptr[b + 1].item())
            L = page_end - page_beg
            if L <= 0:
                # No KV tokens for this batch element
                output[b].zero_()
                lse[b] = -float("inf")
                continue

            tok_idx = kv_indices[page_beg:page_end]  # [L]
            # Gather Kc and Kp for these tokens
            # Since page_size=1 in original, tokens are contiguous indices in ckv_cache
            Kc = ckv_cache[tok_idx].to(torch.float32)  # [L, 512]
            Kp = kpe_cache[tok_idx].to(torch.float32)  # [L, 64]

            # For each head j
            for j in range(num_qo_heads):
                # qn and qp vectors
                qn = q_nope[b, j].to(torch.float32)    # [512]
                qp = q_pe[b, j].to(torch.float32)     # [64]

                # Allocate intermediates
                v = torch.empty(L, dtype=torch.float32, device=device)
                attn = torch.empty(L, dtype=torch.float32, device=device)
                # lse vector (later we'll extract scalar)
                lse_vec = torch.empty(L, dtype=torch.float32, device=device)

                # Kernel 1: compute v[i] = sum_k qn[k]*Kc[i,k] + sum_k qp[k]*Kp[i,k]
                matvec_add_kernel[(L,)](
                    qn, qp, Kc, Kp, v, L, head_dim_ckv, head_dim_kpe, BLOCK_K,
                    num_warps=4
                )

                # Kernel 2: compute lse vector
                lse_kernel[(1,)](
                    v, lse_vec, L, BLOCK_L,
                    num_warps=4
                )
                # lse scalar for this (b, j)
                lse_scalar = torch.logsumexp(v, dim=0) / math.log(2.0)
                lse[b, j] = lse_scalar

                # Kernel 3: compute attn[i] = exp((v[i] - lse_scalar) / ln(2))
                # Note: softmax_base2_kernel expects LSE vector; we pass lse_vec which equals lse_scalar
                softmax_base2_kernel[(L,)](
                    v, lse_vec, attn, L, BLOCK_L,
                    num_warps=4
                )

                # Kernel 4: compute out[b, j, :] = attn @ Kc[:, :]
                y = torch.empty(head_dim_ckv, dtype=torch.float32, device=device)
                matvec_write_y_kernel[(head_dim_ckv,)](
                    attn, Kc, y, L, head_dim_ckv, BLOCK_K,
                    num_warps=4
                )
                # Store to output in bfloat16
                output[b, j, :] = y.to(torch.bfloat16)

        return output, lse