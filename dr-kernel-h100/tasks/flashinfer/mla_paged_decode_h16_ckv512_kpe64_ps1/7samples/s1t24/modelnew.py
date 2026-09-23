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
    D: tl.int32,      # head_dim_ckv (512)
    Dp: tl.int32,     # head_dim_kpe (64)
    ln2: tl.float32,  # natural log of 2
    BLOCK_K: tl.constexpr,  # reduction block size over K
    BLOCK_KP: tl.constexpr, # reduction block size over Dp
):
    # One program per output index i
    i = tl.program_id(0)
    acc = 0.0
    # Reduce over Kc dimension (D)
    for k in range(0, D, BLOCK_K):
        k_off = k + tl.arange(0, BLOCK_K)
        mask_k = k_off < D
        qn_slice = tl.load(qn_ptr + k_off, mask=mask_k, other=0.0)  # [BLOCK_K]
        kc_ptr_row = Kc_ptr + i * D + k_off
        kc_slice = tl.load(kc_ptr_row, mask=mask_k, other=0.0)     # [BLOCK_K]
        acc += tl.sum(qn_slice * kc_slice, axis=0)

    # Reduce over Kp dimension (Dp)
    for kp in range(0, Dp, BLOCK_KP):
        kp_off = kp + tl.arange(0, BLOCK_KP)
        mask_kp = kp_off < Dp
        qp_slice = tl.load(qp_ptr + kp_off, mask=mask_kp, other=0.0)  # [BLOCK_KP]
        kp_ptr_row = Kp_ptr + i * Dp + kp_off
        kp_slice = tl.load(kp_ptr_row, mask=mask_kp, other=0.0)       # [BLOCK_KP]
        acc += tl.sum(qp_slice * kp_slice, axis=0)

    tl.store(v_ptr + i, acc)


@triton.jit
def lse_base2_kernel(
    v_ptr,            # *float32, [L]
    lse_ptr,          # *float32, [1]
    L: tl.int32,      # number of tokens
    ln2: tl.float32,  # natural log of 2
    BLOCK_L: tl.constexpr,
):
    # One program that loops over L to compute stable lse
    max_v = -float('inf')
    sum_exp = 0.0
    for l in range(0, L, BLOCK_L):
        l_off = l + tl.arange(0, BLOCK_L)
        mask_l = l_off < L
        v_chunk = tl.load(v_ptr + l_off, mask=mask_l, other=-float('inf'))
        max_v = tl.maximum(max_v, tl.max(v_chunk, axis=0))
        # sum of exp(v - max_v) / ln(2)
        sum_exp += tl.sum(tl.exp(v_chunk - max_v) / ln2, axis=0)
    lse = max_v + tl.log(sum_exp)
    tl.store(lse_ptr, lse)


@triton.jit
def softmax_base2_kernel(
    v_ptr,            # *float32, [L]
    lse_scalar,       # scalar float32
    attn_ptr,         # *float32, [L]
    L: tl.int32,      # number of tokens
    ln2: tl.float32,  # natural log of 2
    BLOCK_L: tl.constexpr,
):
    # One program that loops over L to compute attn = exp(v/ln2 - lse_scalar)
    for l in range(0, L, BLOCK_L):
        l_off = l + tl.arange(0, BLOCK_L)
        mask_l = l_off < L
        v_chunk = tl.load(v_ptr + l_off, mask=mask_l, other=0.0)
        attn_chunk = tl.exp(v_chunk / ln2 - lse_scalar)
        tl.store(attn_ptr + l_off, attn_chunk, mask=mask_l)


@triton.jit
def matvec_write_y_kernel(
    attn_ptr,         # *float32, [L]
    Kc_ptr,           # *float32, [L, D], row-major (L, D)
    y_ptr,            # *float32, [D]
    L: tl.int32,      # number of tokens
    D: tl.int32,      # head_dim_ckv
    BLOCK_K: tl.constexpr,
):
    # One program per output dimension h
    h = tl.program_id(0)
    acc = 0.0
    for l in range(0, L, BLOCK_K):
        l_off = l + tl.arange(0, BLOCK_K)
        mask_l = l_off < L
        attn_chunk = tl.load(attn_ptr + l_off, mask=mask_l, other=0.0)  # [BLOCK_K]
        Kc_row = Kc_ptr + l_off * D + h  # since shape (L, D), row-major: row stride = D
        Kc_chunk = tl.load(Kc_row, mask=mask_l, other=0.0)             # [BLOCK_K]
        acc += tl.sum(attn_chunk * Kc_chunk, axis=0)
    tl.store(y_ptr + h, acc)


def run(q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
    # Shapes/assertions
    batch_size, num_qo_heads, head_dim_ckv = q_nope.shape
    head_dim_kpe = q_pe.shape[-1]
    num_pages = ckv_cache.shape[0] - 1  # squeezed removes the first dim; original has [num_pages, 1, D]
    device = q_nope.device
    # We don't assert constants here; we handle general L in kernels.
    output = torch.empty(
        (batch_size, num_qo_heads, head_dim_ckv), dtype=torch.float32, device=device
    )
    lse = torch.empty((batch_size, num_qo_heads), dtype=torch.float32, device=device)

    ln2 = math.log(2.0)

    for b in range(batch_size):
        # Determine token range
        L = int(kv_indptr[b + 1].item()) - int(kv_indptr[b].item())
        if L <= 0:
            # No tokens for this batch element; output zeros and lse -inf
            output[b].zero_()
            lse[b] = -float('inf')
            continue

        # Gather Kc and Kp for these tokens
        tok_idx = kv_indices[int(kv_indptr[b].item()):int(kv_indptr[b + 1].item())].to(device).to(torch.int32)
        Kc = ckv_cache[tok_idx, 0]  # squeeze dim 1 -> [L, D]
        Kp = kpe_cache[tok_idx, 0]  # [L, Dp]
        # Ensure contiguous
        Kc = Kc.contiguous().to(torch.float32)
        Kp = Kp.contiguous().to(torch.float32)

        # qn and qp for this batch, head
        qn = q_nope[b, :, :].contiguous().to(torch.float32)  # [num_qo_heads, D]
        qp = q_pe[b, :].contiguous().to(torch.float32)       # [num_qo_heads, Dp]
        D = head_dim_ckv
        Dp = head_dim_kpe

        # Allocate v for each head
        for j in range(16):
            # Compute v[j, :] using Triton
            v = torch.empty(L, dtype=torch.float32, device=device)
            grid_v = (L,)
            matvec_add_kernel[grid_v](
                qn[j], qp[j], Kc, Kp, v,
                L=L, D=D, Dp=Dp, ln2=ln2,
                BLOCK_K=128, BLOCK_KP=64,
            )

            # Compute lse (base-2) for this head
            lse_scalar = torch.empty(1, dtype=torch.float32, device=device)
            lse_base2_kernel[(1,)](
                v, lse_scalar,
                L=L, ln2=ln2, BLOCK_L=1024,
            )
            lse[b, j] = lse_scalar[0]

            # Compute attn (softmax with base-2 normalization)
            attn = torch.empty(L, dtype=torch.float32, device=device)
            softmax_base2_kernel[(1,)](
                v, lse[b, j], attn,
                L=L, ln2=ln2, BLOCK_L=1024,
            )

            # Compute y[j, :] = attn @ Kc
            y = torch.empty(D, dtype=torch.float32, device=device)
            grid_y = (D,)
            matvec_write_y_kernel[grid_y](
                attn, Kc, y,
                L=L, D=D, BLOCK_K=128,
            )

            # Store y into output
            output[b, j, :] = y

    # Cast output to bfloat16 to match original signature
    output_bf16 = output.to(torch.bfloat16)
    return output_bf16, lse


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Ensure inputs are on CUDA
        for i in range(len(args)):
            if isinstance(args[i], torch.Tensor) and args[i].device.type != 'cuda':
                args[i] = args[i].to('cuda')
        return run(*args)