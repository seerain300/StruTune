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
        kc_ptr = Kc_ptr + i * D + k_off
        kc_slice = tl.load(kc_ptr, mask=mask_k, other=0.0)  # [BLOCK_K]
        sum1 += tl.sum(qn_slice * kc_slice, axis=0)
    # Reduce over Kp dimension (Dp)
    for k in range(0, Dp, BLOCK_K):
        k_off = k + tl.arange(0, BLOCK_K)
        mask_k = k_off < Dp
        qp_slice = tl.load(qp_ptr + k_off, mask=mask_k, other=0.0)  # [BLOCK_K]
        kp_ptr = Kp_ptr + i * Dp + k_off
        kp_slice = tl.load(kp_ptr, mask=mask_k, other=0.0)  # [BLOCK_K]
        sum2 += tl.sum(qp_slice * kp_slice, axis=0)
    v = sum1 + sum2
    tl.store(v_ptr + i, v)


@triton.jit
def lse_kernel(
    v_ptr,            # *float32, [L]
    out_ptr,          # *float32, [1] to hold scalar lse
    L: tl.int32,
    inv_ln2: tl.float32
):
    # First pass: compute max
    m = -float('inf')
    for i in range(0, L):
        vi = tl.load(v_ptr + i)
        m = tl.maximum(m, vi)
    # Second pass: sum exp(v - m)
    sum_exp = 0.0
    for i in range(0, L):
        vi = tl.load(v_ptr + i)
        sum_exp += tl.exp((vi - m) * inv_ln2)  # multiply by inv_ln2 to convert to natural log base
    lse = m + tl.log(sum_exp) / inv_ln2  # logsumexp_base2 = m + log(sum_exp) / ln(2)
    tl.store(out_ptr, lse)


@triton.jit
def softmax_base2_kernel(
    v_ptr,            # *float32, [L]
    lse_ptr,          # *float32, [1] scalar lse
    attn_ptr,         # *float32, [L]
    L: tl.int32,
    inv_ln2: tl.float32
):
    # One program per i
    i = tl.program_id(0)
    vi = tl.load(v_ptr + i)
    lse_val = tl.load(lse_ptr)
    attn = tl.exp((vi - lse_val) * inv_ln2)
    tl.store(attn_ptr + i, attn)


@triton.jit
def matvec_write_y_kernel(
    attn_ptr,         # *float32, [L]
    K_ptr,            # *float32, [L, D]
    y_ptr,            # *float32, [D]
    L: tl.int32,
    D: tl.int32,
    BLOCK_K: tl.constexpr
):
    # One program per output dimension h in [0, D)
    h = tl.program_id(0)
    acc = 0.0
    # Reduce over L
    for i in range(0, L, BLOCK_K):
        idx = i + tl.arange(0, BLOCK_K)
        mask_i = idx < L
        attn_slice = tl.load(attn_ptr + idx, mask=mask_i, other=0.0)  # [BLOCK_K]
        k_ptr = K_ptr + idx * D + h
        k_slice = tl.load(k_ptr, mask=mask_i, other=0.0)  # [BLOCK_K]
        acc += tl.sum(attn_slice * k_slice, axis=0)
    tl.store(y_ptr + h, acc)


def get_inputs():
    # Non-recursive helper for local testing. The evaluator will supply its own inputs.
    batch_size = 1
    num_qo_heads = 16
    head_dim_ckv = 512
    head_dim_kpe = 64
    num_pages = 989669  # irrelevant for this setup; indptr length is batch_size + 1
    # Construct minimal tensors; forward will move to CUDA.
    device = 'cuda'
    q_nope = torch.randn([batch_size, num_qo_heads, head_dim_ckv], dtype=torch.bfloat16, device=device)
    q_pe = torch.randn([batch_size, num_qo_heads, head_dim_kpe], dtype=torch.bfloat16, device=device)
    ckv_cache = torch.randn([num_pages, 1, head_dim_ckv], dtype=torch.bfloat16, device=device)
    kpe_cache = torch.randn([num_pages, 1, head_dim_kpe], dtype=torch.bfloat16, device=device)
    # Simple indptr and indices for a single batch
    kv_indptr = torch.tensor([0, 8], dtype=torch.int32, device=device)
    kv_indices = torch.randint(0, num_pages, [8], dtype=torch.int32, device=device)
    sm_scale = 1.0
    return [q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale]


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Ensure inputs are on CUDA; the evaluator may provide pre-CUDA tensors.
        for i in range(len(args)):
            if isinstance(args[i], torch.Tensor) and args[i].device.type != 'cuda':
                args[i] = args[i].to('cuda')

        q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale = args
        batch_size = q_nope.shape[0]
        num_qo_heads = q_nope.shape[1]
        head_dim_ckv = q_nope.shape[2]
        head_dim_kpe = q_pe.shape[2]

        # Ensure dtype is float32 for computation
        device = q_nope.device
        q_nope_f32 = q_nope.float()
        q_pe_f32 = q_pe.float()
        # Squeeze the 1-size dim
        Kc_all = ckv_cache.squeeze(1).float()
        Kp_all = kpe_cache.squeeze(1).float()

        # Output tensors
        output = torch.empty(
            (batch_size, num_qo_heads, head_dim_ckv),
            dtype=torch.bfloat16,
            device=device
        )
        lse = torch.full((batch_size, num_qo_heads), -float("inf"), dtype=torch.float32, device=device)

        inv_ln2 = 1.0 / math.log(2.0)

        # For each batch element
        for b in range(batch_size):
            # Determine token range
            L = int(kv_indptr[b + 1].item()) - int(kv_indptr[b].item())
            if L <= 0:
                # No tokens for this batch; output zeros and lse = -inf
                output[b].zero_()
                lse[b] = -float("inf")
                continue

            # Gather indices
            tok_idx = kv_indices[int(kv_indptr[b].item()):int(kv_indptr[b + 1].item())].to(torch.int32)

            # Gather Kc and Kp slices
            Kc = Kc_all[tok_idx]  # [L, D]
            Kp = Kp_all[tok_idx]  # [L, Dp]

            # For each head j
            for j in range(num_qo_heads):
                qn = q_nope_f32[b, j, :]             # [D]
                qp = q_pe_f32[b, j, :]               # [Dp]

                # Kernel 1: compute v[j, :] = sum_i (qn · Kc[i, :]) + sum_i (qp · Kp[i, :])
                v = torch.empty(L, dtype=torch.float32, device=device)
                matvec_add_kernel[(L,)](
                    qn, qp, Kc, Kp, v,
                    L, head_dim_ckv, head_dim_kpe,
                    BLOCK_K=64, num_warps=4
                )

                # Kernel 2: compute lse = logsumexp_base2(v)
                lse_b_j = torch.empty(1, dtype=torch.float32, device=device)
                lse_kernel[(1,)](
                    v, lse_b_j, L, inv_ln2,
                    num_warps=1
                )
                lse_val = lse_b_j[0]  # scalar

                # Kernel 3: compute attn[i] = exp(v[i] / ln(2) - lse)
                attn = torch.empty(L, dtype=torch.float32, device=device)
                softmax_base2_kernel[(L,)](
                    v, lse_b_j, attn, L, inv_ln2,
                    num_warps=4
                )

                # Kernel 4: compute out[b, j, :] = attn @ Kc
                y = torch.empty(head_dim_ckv, dtype=torch.float32, device=device)
                matvec_write_y_kernel[(head_dim_ckv,)](
                    attn, Kc, y, L, head_dim_ckv, BLOCK_K=64, num_warps=4
                )
                output[b, j, :] = y.to(torch.bfloat16)

                # Update lse
                lse[b, j] = lse_val

        return output, lse


def run(*args):
    return ModelNew()(*args)
