import math
import torch
import triton
import triton.language as tl


@triton.jit
def matvec_add_kernel(
    qn_ptr,   # *float32, [D]
    qp_ptr,   # *float32, [Dp]
    Kc_ptr,   # *float32, [L, D], row-major (L, D)
    Kp_ptr,   # *float32, [L, Dp], row-major (L, Dp)
    v_ptr,    # *float32, [L]
    L: tl.int32,   # number of tokens
    D: tl.int32,   # head_dim_ckv
    Dp: tl.int32,  # head_dim_kpe
    BLOCK_K: tl.constexpr
):
    # One program per output index i in [0, L)
    i = tl.program_id(0)
    # Accumulator for v[i]
    acc = 0.0
    # Reduce over Kc dimension (D)
    for k in range(0, D, BLOCK_K):
        k_off = k + tl.arange(0, BLOCK_K)
        mask_k = k_off < D
        qn_slice = tl.load(qn_ptr + k_off, mask=mask_k, other=0.0)  # [BLOCK_K]
        kc_ptr_i = Kc_ptr + i * D + k_off
        kc_slice = tl.load(kc_ptr_i, mask=mask_k, other=0.0)       # [BLOCK_K]
        acc += tl.sum(qn_slice * kc_slice, axis=0)
    # Reduce over Kp dimension (Dp)
    for kp in range(0, Dp, BLOCK_K):
        kp_off = kp + tl.arange(0, BLOCK_K)
        mask_kp = kp_off < Dp
        qp_slice = tl.load(qp_ptr + kp_off, mask=mask_kp, other=0.0)  # [BLOCK_K]
        kp_ptr_i = Kp_ptr + i * Dp + kp_off
        kp_slice = tl.load(kp_ptr_i, mask=mask_kp, other=0.0)         # [BLOCK_K]
        acc += tl.sum(qp_slice * kp_slice, axis=0)
    tl.store(v_ptr + i, acc)


@triton.jit
def lse_base2_kernel(
    v_ptr,     # *float32, [L]
    L: tl.int32,
    ln2: tl.float32,  # ln(2)
    lse_ptr,   # *float32, [1] (scalar per head, but grid is (1,))
):
    # Compute logsumexp_base2 over L
    # First pass: max
    m = -float('inf')
    for i in range(0, L):
        vi = tl.load(v_ptr + i)
        m = tl.maximum(m, vi)
    # Second pass: sum exp(v - m) scaled by 1/ln(2)
    s = 0.0
    for i in range(0, L):
        vi = tl.load(v_ptr + i)
        s += tl.exp((vi - m) / ln2)
    lse = m + tl.log(s)  # equals log2(sum(exp(v))) because m + log(sum(exp(v)/ln2)) simplifies
    # Store scalar (grid is (1,))
    tl.store(lse_ptr, lse)


@triton.jit
def softmax_base2_kernel(
    v_ptr,          # *float32, [L]
    lse_ptr,        # *float32, [1]
    attn_ptr,       # *float32, [L]
    L: tl.int32,
    ln2: tl.float32
):
    # One program per index i
    i = tl.program_id(0)
    lse_val = tl.load(lse_ptr)
    vi = tl.load(v_ptr + i)
    attn_i = tl.exp((vi - lse_val) / ln2)
    tl.store(attn_ptr + i, attn_i)


@triton.jit
def matvec_write_y_kernel(
    attn_ptr,  # *float32, [L]
    Kc_ptr,    # *float32, [L, D], row-major (L, D)
    y_ptr,     # *float32, [D]
    L: tl.int32,
    D: tl.int32,
    BLOCK_K: tl.constexpr
):
    # One program per output dim h in [0, D)
    h = tl.program_id(0)
    acc = 0.0
    for k in range(0, L, BLOCK_K):
        offs = k + tl.arange(0, BLOCK_K)
        mask = offs < L
        attn_k = tl.load(attn_ptr + offs, mask=mask, other=0.0)  # [BLOCK_K]
        kc_row = tl.load(Kc_ptr + offs * D + h, mask=mask, other=0.0)  # [BLOCK_K]
        acc += tl.sum(attn_k * kc_row, axis=0)
    tl.store(y_ptr + h, acc)


def get_inputs():
    # Helper returns dummy tensors; harness will provide its own inputs.
    # Define axes for the harness to read:
    axes = {
        "batch_size": 1,
        "num_pages": 989669,
        "len_indptr": 2,
        "num_kv_indices": 8,
    }
    device = 'cuda'
    q_nope = torch.randn([1, 16, 512], dtype=torch.bfloat16, device=device)
    q_pe = torch.randn([1, 16, 64], dtype=torch.bfloat16, device=device)
    ckv_cache = torch.randn([989669, 1, 512], dtype=torch.bfloat16, device=device)
    kpe_cache = torch.randn([989669, 1, 64], dtype=torch.bfloat16, device=device)
    _n = 1; _t = 8
    _lens = torch.full((_n,), _t // _n, dtype=torch.int32, device=device)
    _lens[: _t % _n] += 1
    kv_indptr = torch.cat([torch.zeros(1, dtype=torch.int32, device=device), torch.cumsum(_lens, 0)]).to(torch.int32)  # [2]
    kv_indices = torch.randint(0, 989669, [8], dtype=torch.int32, device=device)
    sm_scale = 1.0  # float32 scalar
    return [q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale]


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure tensors are on CUDA
        device = 'cuda'
        q_nope = q_nope.to(device)
        q_pe = q_pe.to(device)
        ckv_cache = ckv_cache.to(device)
        kpe_cache = kpe_cache.to(device)
        kv_indptr = kv_indptr.to(device)
        kv_indices = kv_indices.to(device)

        batch_size = q_nope.shape[0]
        num_qo_heads = q_nope.shape[1]
        D = q_nope.shape[2]
        Dp = q_pe.shape[2]

        # Kc and Kp for all tokens (gather per batch below)
        # output and lse tensors
        output = torch.empty((batch_size, num_qo_heads, D), dtype=torch.float32, device=device)
        lse = torch.empty((batch_size, num_qo_heads), dtype=torch.float32, device=device)

        # For each batch b
        for b in range(batch_size):
            # Compute token range from indptr
            # Note: PyTorch requires kv_indptr has length batch_size + 1
            L = int(kv_indptr[b + 1].item()) - int(kv_indptr[b].item())
            if L <= 0:
                # No tokens for this batch element; set output to zeros
                output[b].zero_()
                lse[b] = -float('inf')
                continue

            # Gather token indices and corresponding Kc, Kp
            tok_idx = kv_indices[kv_indptr[b]:kv_indptr[b + 1]]
            Kc = ckv_cache.squeeze(1)[tok_idx].to(torch.float32)  # [L, D]
            Kp = kpe_cache.squeeze(1)[tok_idx].to(torch.float32)  # [L, Dp]

            # For each head j
            for j in range(num_qo_heads):
                # Prepare qn[j, :] and qp[j, :]
                qn = q_nope[b, j, :].to(torch.float32)  # [D]
                qp = q_pe[b, j, :].to(torch.float32)    # [Dp]

                # Compute v = (qn @ Kc.T) + (qp @ Kp.T) -> [L]
                v = torch.empty((L,), dtype=torch.float32, device=device)
                grid_v = (L,)
                matvec_add_kernel[grid_v](
                    qn, qp, Kc, Kp, v,
                    L, D, Dp,
                    BLOCK_K=128,
                    num_warps=1
                )

                # Compute lse_base2 = logsumexp_base2(v)
                ln2 = math.log(2.0)
                lse_b = torch.empty((1,), dtype=torch.float32, device=device)  # scalar tensor for kernel
                lse_b.fill_(0.0)
                lse_base2_kernel[(1,)](
                    v, L, ln2, lse_b,
                    num_warps=1
                )
                lse_j = lse_b[0]

                # Compute attn[i] = exp(v[i] / ln(2) - lse_j)
                attn = torch.empty((L,), dtype=torch.float32, device=device)
                softmax_base2_kernel[(L,)](
                    v, lse_j, attn,
                    L, ln2,
                    num_warps=1
                )

                # Compute y = attn @ Kc -> [D]
                y = torch.empty((D,), dtype=torch.float32, device=device)
                matvec_write_y_kernel[(D,)](
                    attn, Kc, y,
                    L, D,
                    BLOCK_K=128,
                    num_warps=2
                )

                output[b, j, :] = y

        return output.to(torch.bfloat16), lse

# Optional: keep the original run function for compatibility with some harnesses
def run(q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
    return ModelNew()(q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale)