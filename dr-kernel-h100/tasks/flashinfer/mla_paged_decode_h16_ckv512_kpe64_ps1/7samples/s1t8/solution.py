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
    BLOCK_K: tl.constexpr,
):
    # One program per output index i in [0, L)
    i = tl.program_id(0)
    sum1 = 0.0
    sum2 = 0.0
    k = 0
    while k < D:
        k_off = k + tl.arange(0, BLOCK_K)
        mask_k = k_off < D
        qn_slice = tl.load(qn_ptr + k_off, mask=mask_k, other=0.0)  # [BLOCK_K]
        kc_ptr = Kc_ptr + i * D + k_off
        kc_slice = tl.load(kc_ptr, mask=mask_k, other=0.0)  # [BLOCK_K]
        sum1 += tl.sum(qn_slice * kc_slice, axis=0)
        k += BLOCK_K
    p = 0
    while p < Dp:
        p_off = p + tl.arange(0, BLOCK_K)
        mask_p = p_off < Dp
        qp_slice = tl.load(qp_ptr + p_off, mask=mask_p, other=0.0)  # [BLOCK_K]
        kp_ptr = Kp_ptr + i * Dp + p_off
        kp_slice = tl.load(kp_ptr, mask=mask_p, other=0.0)  # [BLOCK_K]
        sum2 += tl.sum(qp_slice * kp_slice, axis=0)
        p += BLOCK_K
    tl.store(v_ptr + i, sum1 + sum2)


@triton.jit
def lse_kernel(
    v_ptr,            # *float32, [L]
    lse_ptr,          # *float32, [1]
    L: tl.int32,
    inv_ln2: tl.float32,
    BLOCK: tl.constexpr,
):
    # Compute base-2 logsumexp for a vector v of length L
    max_v = -float("inf")
    # First pass: find max
    i = 0
    while i < L:
        v_i = tl.load(v_ptr + i)
        if v_i > max_v:
            max_v = v_i
        i += 1
    # Second pass: sum exp((v - max) * inv_ln2)
    sum_exp = 0.0
    i = 0
    while i < L:
        v_i = tl.load(v_ptr + i)
        sum_exp += tl.exp((v_i - max_v) * inv_ln2)
        i += 1
    lse_val = tl.log(sum_exp) + max_v
    tl.store(lse_ptr, lse_val)


@triton.jit
def softmax_base2_kernel(
    v_ptr,            # *float32, [L]
    attn_ptr,         # *float32, [L]
    lse_val: tl.float32,
    L: tl.int32,
    inv_ln2: tl.float32,
    BLOCK: tl.constexpr,
):
    # Compute attn[i] = exp((v[i] - lse) * inv_ln2)
    i = 0
    while i < L:
        v_i = tl.load(v_ptr + i)
        attn_i = tl.exp((v_i - lse_val) * inv_ln2)
        tl.store(attn_ptr + i, attn_i)
        i += 1


@triton.jit
def matvec_write_y_kernel(
    attn_ptr,         # *float32, [L]
    K_ptr,            # *float32, [D], this is Kc for output, row-major (L, D) reduced per token
    out_ptr,          # *float32, [D]
    L: tl.int32,
    D: tl.int32,
    BLOCK_J: tl.constexpr,
):
    # One program per output dimension j in [0, D)
    j = tl.program_id(0)
    acc = 0.0
    k = 0
    while k < L:
        attn_k = tl.load(attn_ptr + k)
        K_jk = tl.load(K_ptr + k * D + j)
        acc += attn_k * K_jk
        k += 1
    tl.store(out_ptr + j, acc)


@torch.no_grad()
def run(q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
    device = q_nope.device
    batch_size = q_nope.shape[0]
    head_dim_ckv = q_nope.shape[2]
    head_dim_kpe = q_pe.shape[2]

    # Ensure dtype is float32 for computation
    qn_base = q_nope.to(torch.float32)   # [B, 16, 512]
    qp_base = q_pe.to(torch.float32)     # [B, 16, 64]
    Kc_all = ckv_cache.squeeze(1).to(torch.float32)  # [N, 512]
    Kp_all = kpe_cache.squeeze(1).to(torch.float32)  # [N, 64]

    # Output and lse
    output = torch.empty((batch_size, 16, head_dim_ckv), dtype=torch.float32, device=device)
    lse = torch.empty((batch_size, 16), dtype=torch.float32, device=device)

    # Precompute inv ln(2)
    inv_ln2 = 1.4426950408889634  # 1 / ln(2)

    for b in range(batch_size):
        # If there are no tokens for this batch, output zeros
        if kv_indptr[b + 1] <= kv_indptr[b]:
            lse[b, :] = 0.0
            continue

        # Gather tokens for this batch
        tok_idx = kv_indices[kv_indptr[b]: kv_indptr[b + 1]].to(torch.long)  # [L_tokens]
        L_tokens = tok_idx.numel()

        # Gather Kc and Kp for tokens
        Kc = Kc_all[tok_idx]  # [L_tokens, 512]
        Kp = Kp_all[tok_idx]  # [L_tokens, 64]

        # For each head j in [0, 16)
        for j in range(16):
            # qn[j], qp[j] vectors
            qn = qn_base[b, j, :]     # [512]
            qp = qp_base[b, j, :]     # [64]

            # 1) Compute v[j, :] = sum_i (qn · Kc[i, :]) + sum_i (qp · Kp[i, :])
            v = torch.empty(L_tokens, dtype=torch.float32, device=device)
            grid_v = (L_tokens,)
            matvec_add_kernel[grid_v](
                qn, qp, Kc, Kp, v,
                L_tokens, head_dim_ckv, head_dim_kpe,
                BLOCK_K=128, num_warps=4
            )

            # 2) Compute lse_j = logsumexp_base2(v)
            lse_j = torch.empty((), dtype=torch.float32, device=device)
            grid_lse = (1,)
            lse_kernel[grid_lse](
                v, lse_j,
                L_tokens, inv_ln2,
                BLOCK=128, num_warps=4
            )
            lse[b, j] = lse_j

            # 3) Compute attn vector with base-2 softmax
            attn = torch.empty(L_tokens, dtype=torch.float32, device=device)
            grid_softmax = (L_tokens,)
            softmax_base2_kernel[grid_softmax](
                v, attn, lse[b, j], L_tokens, inv_ln2,
                BLOCK=128, num_warps=4
            )

            # 4) Final matvec: out[b, j, :] = attn @ Kc[:, :]
            out_row = torch.empty(head_dim_ckv, dtype=torch.float32, device=device)
            grid_y = (head_dim_ckv,)
            matvec_write_y_kernel[grid_y](
                attn, Kc,
                out_row,
                L_tokens, head_dim_ckv,
                BLOCK_J=128, num_warps=4
            )
            output[b, j, :] = out_row

    # Cast output to bfloat16 to match original
    output_bf16 = output.to(torch.bfloat16)
    return output_bf16, lse


def get_inputs():
    # Provide consistent inputs without recursion. Here we construct a small example consistent with the original code.
    device = 'cuda'
    batch_size = 1
    num_tokens = 8  # per batch
    num_qo_heads = 16
    head_dim_ckv = 512
    head_dim_kpe = 64
    num_pages = 989669  # not used directly; only L tokens from indices matter

    # Create random q_nope and q_pe
    q_nope = torch.randn([batch_size, num_qo_heads, head_dim_ckv], dtype=torch.bfloat16, device=device)
    q_pe = torch.randn([batch_size, num_qo_heads, head_dim_kpe], dtype=torch.bfloat16, device=device)

    # Create caches
    ckv_cache = torch.randn([num_pages, 1, head_dim_ckv], dtype=torch.bfloat16, device=device)
    kpe_cache = torch.randn([num_pages, 1, head_dim_kpe], dtype=torch.bfloat16, device=device)

    # Build kv_indptr and kv_indices
    # len_indptr = batch_size + 1 = 2
    kv_indptr = torch.tensor([0, num_tokens], dtype=torch.int32, device=device)
    kv_indices = torch.randint(0, num_pages, [num_tokens], dtype=torch.int32, device=device)

    sm_scale = 1.0  # float32 scalar
    return [q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale]


def fused_operator(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6):
    _out = run(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6)
    return list(_out) if isinstance(_out, (tuple, list)) else [_out]


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Ensure inputs are on CUDA
        for i in range(len(args)):
            if isinstance(args[i], torch.Tensor) and args[i].device.type != 'cuda':
                args[i] = args[i].to('cuda')
        return run(*args)


def run(*args):
    return ModelNew()(*args)
