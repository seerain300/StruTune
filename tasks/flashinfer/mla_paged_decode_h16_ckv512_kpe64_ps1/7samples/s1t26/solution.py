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
    sum1 = 0.0
    sum2 = 0.0
    # Reduce over Kc dimension (D)
    for k in range(0, D, BLOCK_K):
        k_off = k + tl.arange(0, BLOCK_K)
        mask_k = k_off < D
        qn_slice = tl.load(qn_ptr + k_off, mask=mask_k, other=0.0)  # [BLOCK_K]
        kc_ptr_i = Kc_ptr + i * D + k_off
        kc_vals = tl.load(kc_ptr_i, mask=mask_k, other=0.0)  # [BLOCK_K]
        sum1 += tl.sum(qn_slice * kc_vals, axis=0)
    # Reduce over Kp dimension (Dp)
    for p in range(0, Dp, BLOCK_K):
        p_off = p + tl.arange(0, BLOCK_K)
        mask_p = p_off < Dp
        qp_slice = tl.load(qp_ptr + p_off, mask=mask_p, other=0.0)  # [BLOCK_K]
        kp_ptr_i = Kp_ptr + i * Dp + p_off
        kp_vals = tl.load(kp_ptr_i, mask=mask_p, other=0.0)  # [BLOCK_K]
        sum2 += tl.sum(qp_slice * kp_vals, axis=0)
    v = sum1 + sum2
    tl.store(v_ptr + i, v)


@triton.jit
def lse_kernel(
    v_ptr,            # *float32, [L]
    lse_ptr,          # *float32, [1]
    L: tl.int32,
    inv_ln2: tl.float32  # 1 / ln(2)
):
    # Pass 1: compute max m
    m = -float('inf')
    for i in range(0, L):
        vi = tl.load(v_ptr + i)
        m = tl.maximum(m, vi)
    # Pass 2: compute sum_exp
    sum_exp = 0.0
    for i in range(0, L):
        vi = tl.load(v_ptr + i)
        sum_exp += tl.exp(vi - m)
    # lse = m + log(sum_exp) / ln(2)
    lse = m + tl.log(sum_exp) * inv_ln2
    tl.store(lse_ptr, lse)


@triton.jit
def softmax_base2_kernel(
    v_ptr,            # *float32, [L]
    lse_ptr,          # *float32, [1]
    attn_ptr,         # *float32, [L]
    L: tl.int32,
    inv_ln2: tl.float32
):
    # Load lse scalar
    lse = tl.load(lse_ptr)
    # Compute attn[i] = exp(v[i] / ln(2) - lse)
    for i in range(0, L):
        vi = tl.load(v_ptr + i)
        attn_i = tl.exp(vi * inv_ln2 - lse)
        tl.store(attn_ptr + i, attn_i)


@triton.jit
def matvec_write_y_kernel(
    attn_ptr,         # *float32, [L]
    Kc_ptr,           # *float32, [L, D], row-major (L, D)
    y_ptr,            # *float32, [D]
    L: tl.int32,
    D: tl.int32,
    BLOCK_K: tl.constexpr
):
    # One program per output dimension h in [0, D)
    h = tl.program_id(0)
    acc = 0.0
    for i in range(0, L, BLOCK_K):
        idx = i + tl.arange(0, BLOCK_K)
        mask_i = idx < L
        attn_i = tl.load(attn_ptr + idx, mask=mask_i, other=0.0)  # [BLOCK_K]
        kc_ptr_h = Kc_ptr + idx * D + h
        kc_vals = tl.load(kc_ptr_h, mask=mask_i, other=0.0)      # [BLOCK_K]
        # acc += sum(attn_i * kc_vals)
        acc += tl.sum(attn_i * kc_vals, axis=0)
    tl.store(y_ptr + h, acc)


def get_inputs():
    # Helper for local testing; evaluator will provide its own inputs.
    # Avoid recursion; return fixed sizes compatible with the original function.
    device = 'cuda'
    batch_size = 1
    num_qo_heads = 16
    head_dim_ckv = 512
    head_dim_kpe = 64
    num_pages = 989669
    L = 8  # typical tokens per batch
    # Create dummy tensors
    q_nope = torch.randn([batch_size, num_qo_heads, head_dim_ckv], dtype=torch.bfloat16, device=device)
    q_pe = torch.randn([batch_size, num_qo_heads, head_dim_kpe], dtype=torch.bfloat16, device=device)
    ckv_cache = torch.randn([num_pages, 1, head_dim_ckv], dtype=torch.bfloat16, device=device)
    kpe_cache = torch.randn([num_pages, 1, head_dim_kpe], dtype=torch.bfloat16, device=device)
    # Indptr and indices for a single batch
    _n = 1
    _t = L
    _lens = torch.full((_n,), _t // _n, dtype=torch.int32, device=device)
    _lens[: _t % _n] += 1
    kv_indptr = torch.cat([torch.zeros(1, dtype=torch.int32, device=device), torch.cumsum(_lens, 0)]).to(torch.int32)  # [2]
    kv_indices = torch.randint(0, num_pages, [L], dtype=torch.int32, device=device)
    sm_scale = 1.0  # float32 scalar
    return [q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale]


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure tensors are on CUDA
        for i in range(len(q_nope), len(q_nope) - 1, -1):  # non-recursive sanity check; do nothing
            pass
        # Cast to float32 for computation
        device = q_nope.device
        batch_size = q_nope.shape[0]
        num_qo_heads = q_nope.shape[1]
        head_dim_ckv = q_nope.shape[2]
        head_dim_kpe = q_pe.shape[2]
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda, "Inputs must be CUDA tensors"

        # Prepare output
        output = torch.empty((batch_size, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=device)
        lse = torch.full((batch_size, num_qo_heads), -float('inf'), dtype=torch.float32, device=device)

        # Constants
        inv_ln2 = 1.0 / math.log(2.0)
        BLOCK_K = 128
        BLOCK_L = 128

        for b in range(batch_size):
            # Determine token range
            L_tokens = int(kv_indptr[b + 1].item()) - int(kv_indptr[b].item())
            if L_tokens <= 0:
                # No KV tokens for this batch, output zeros and lse = -inf
                output[b].zero_()
                lse[b] = -float('inf')
                continue

            # Gather tok_idx and form Kc, Kp (float32)
            tok_idx = kv_indices[kv_indptr[b]: kv_indptr[b + 1]]
            # Squeeze the (1,) dim to [L_tokens, D] and [L_tokens, Dp]
            Kc_all = ckv_cache.squeeze(1).to(torch.float32)
            Kp_all = kpe_cache.squeeze(1).to(torch.float32)
            Kc = Kc_all[tok_idx]  # [L_tokens, head_dim_ckv]
            Kp = Kp_all[tok_idx]  # [L_tokens, head_dim_kpe]
            L = L_tokens

            # Compute qn and qp for each head j
            for j in range(num_qo_heads):
                qn = q_nope[b, j, :].to(torch.float32)  # [D]
                qp = q_pe[b, j, :].to(torch.float32)   # [Dp]

                # Kernel 1: v = qn · Kc + qp · Kp
                v = torch.empty(L, dtype=torch.float32, device=device)
                matvec_add_kernel[(L,)](
                    qn, qp, Kc, Kp, v,
                    L, head_dim_ckv, head_dim_kpe,
                    BLOCK_K, num_warps=4
                )

                # Kernel 2: lse = logsumexp_base2(v)
                lse_b_j = torch.empty(1, dtype=torch.float32, device=device)
                lse_kernel[(1,)](
                    v, lse_b_j, L, inv_ln2, num_warps=1
                )
                lse_val = lse_b_j[0]  # scalar
                lse[b, j] = lse_val

                # Kernel 3: attn[i] = exp(v[i] / ln(2) - lse)
                attn = torch.empty(L, dtype=torch.float32, device=device)
                softmax_base2_kernel[(L,)](
                    v, lse_b_j, attn, L, inv_ln2, num_warps=4
                )

                # Kernel 4: out[b, j, :] = attn @ Kc
                y = torch.empty(head_dim_ckv, dtype=torch.float32, device=device)
                matvec_write_y_kernel[(head_dim_ckv,)](
                    attn, Kc, y, L, head_dim_ckv, BLOCK_K, num_warps=4
                )
                output[b, j, :] = y.to(torch.bfloat16)

        return output, lse


def run(*args):
    return ModelNew()(*args)
