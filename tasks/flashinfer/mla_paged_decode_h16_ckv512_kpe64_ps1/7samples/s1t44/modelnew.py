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
    BLOCK_K: tl.constexpr,
):
    # One program per output index i in [0, L)
    i = tl.program_id(0)
    sum1 = 0.0
    sum2 = 0.0
    inv_ln2 = 1.4426950408889634  # 1 / ln(2)

    # Reduce over Kc dimension (D)
    for k in range(0, D, BLOCK_K):
        k_off = k + tl.arange(0, BLOCK_K)
        mask_k = k_off < D
        qn_slice = tl.load(qn_ptr + k_off, mask=mask_k, other=0.0)  # [BLOCK_K]
        kc_ptr_i = Kc_ptr + i * D + k_off
        kc_i = tl.load(kc_ptr_i, mask=mask_k, other=0.0)           # [BLOCK_K]
        sum1 += tl.sum(qn_slice * kc_i, axis=0)

    # Reduce over Kp dimension (Dp)
    for p in range(0, Dp, BLOCK_K):
        p_off = p + tl.arange(0, BLOCK_K)
        mask_p = p_off < Dp
        qp_slice = tl.load(qp_ptr + p_off, mask=mask_p, other=0.0)  # [BLOCK_K]
        kp_ptr_i = Kp_ptr + i * Dp + p_off
        kp_i = tl.load(kp_ptr_i, mask=mask_p, other=0.0)            # [BLOCK_K]
        sum2 += tl.sum(qp_slice * kp_i, axis=0)

    v = sum1 + sum2
    tl.store(v_ptr + i, v)


@triton.jit
def lse_kernel(
    v_ptr,            # *float32, [L]
    lse_ptr,          # *float32, [1] (scalar per head)
    L: tl.int32,
    inv_ln2: tl.float32,
):
    # Numerically stable logsumexp in base 2 for v (size L)
    m = -float("inf")
    # Find max
    for i in range(0, L):
        vi = tl.load(v_ptr + i)
        m = tl.maximum(m, vi)

    sum_exp = 0.0
    for i in range(0, L):
        vi = tl.load(v_ptr + i)
        sum_exp += tl.exp((vi - m) * inv_ln2)
    lse = tl.log(sum_exp) * inv_ln2  # logsumexp(v/ln2)/ln(2)
    tl.store(lse_ptr, lse)


@triton.jit
def softmax_base2_kernel(
    v_ptr,            # *float32, [L]
    lse_ptr,          # *float32, [1]
    attn_ptr,         # *float32, [L]
    L: tl.int32,
    inv_ln2: tl.float32,
):
    lse = tl.load(lse_ptr)
    for i in range(0, L):
        vi = tl.load(v_ptr + i)
        attn = tl.exp((vi - lse) * inv_ln2)
        tl.store(attn_ptr + i, attn)


@triton.jit
def matvec_write_y_kernel(
    attn_ptr,         # *float32, [L]
    Kc_ptr,           # *float32, [L, D]
    y_ptr,            # *float32, [D]
    L: tl.int32,
    D: tl.int32,
    BLOCK_K: tl.constexpr,
):
    # One program per output dimension h in [0, D)
    h = tl.program_id(0)
    sum_y = 0.0
    for k in range(0, D, BLOCK_K):
        k_off = k + tl.arange(0, BLOCK_K)
        mask_k = k_off < D
        kc_ptr_h = Kc_ptr + h * L + k_off  # across rows i for fixed h
        # Build vector attn_i for this h by loading Kc rows and using precomputed attn
        # However, attn is per i; here we load attn[i] for i in this block:
        # Since we only need sum over i of attn[i] * Kc[i, h], we loop i.
        for i in range(0, L):
            attn_i = tl.load(attn_ptr + i)
            kc_i_h = tl.load(Kc_ptr + i * D + h)  # single scalar load
            sum_y += attn_i * kc_i_h
    tl.store(y_ptr + h, sum_y)


def run(q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
    # Assume inputs are already on CUDA and device is q_nope.device
    device = q_nope.device

    # Shape assertions from original
    batch_size, num_qo_heads, head_dim_ckv = q_nope.shape
    head_dim_kpe = q_pe.shape[-1]
    num_pages = ckv_cache.shape[0]
    assert num_qo_heads == 16
    assert head_dim_ckv == 512
    assert head_dim_kpe == 64
    assert ckv_cache.shape[1] == 1 and kpe_cache.shape[1] == 1

    # Prepare pointers and shapes
    D = head_dim_ckv
    Dp = head_dim_kpe

    # Output and lse tensors
    output = torch.zeros((batch_size, num_qo_heads, D), dtype=torch.bfloat16, device=device)
    lse = torch.full((batch_size, num_qo_heads), -float("inf"), dtype=torch.float32, device=device)

    # Precompute constants
    inv_ln2 = 1.4426950408889634  # 1 / ln(2)

    # Process each batch b
    for b in range(batch_size):
        # Determine token range
        if kv_indptr.numel() <= 1:
            # Empty indptr, no tokens
            output[b].zero_()
            continue

        if b + 1 >= kv_indptr.numel():
            # Out-of-range, guard
            output[b].zero_()
            continue

        L = int(kv_indptr[b + 1].item() - kv_indptr[b].item())
        if L <= 0:
            output[b].zero_()
            continue

        # Gather tokens indices for this batch
        tok_idx = kv_indices[kv_indptr[b]:kv_indptr[b + 1]]  # [L], int32
        # Kc_all and Kp_all are [num_pages, head_dim], we slice by tok_idx
        Kc_all = ckv_cache.to(torch.float32).squeeze(1)  # [num_pages, D]
        Kp_all = kpe_cache.to(torch.float32).squeeze(1)  # [num_pages, Dp]
        Kc = Kc_all[tok_idx.to(torch.long)]             # [L, D]
        Kp = Kp_all[tok_idx.to(torch.long)]             # [L, Dp]

        # q_nope and q_pe are [1, 16, D] and [1, 16, Dp] for generality; we take b-th batch slice
        # But inputs given are [B, 16, D]; handle general B:
        # We assume q_nope has shape [B, 16, D]; q_pe [B, 16, Dp]
        # Use b-th slice
        qn = q_nope[b].to(torch.float32)  # [16, D]
        qp = q_pe[b].to(torch.float32)    # [16, Dp]

        # Prepare outputs per head j
        for j in range(num_qo_heads):
            # Compute v[j, :] using Triton kernel
            v = torch.empty(L, dtype=torch.float32, device=device)
            # Launch matvec_add_kernel: grid over i in [0, L)
            grid_v = (L,)
            matvec_add_kernel[grid_v](
                qn[j], qp[j], Kc, Kp, v,
                L, D, Dp, BLOCK_K=128,
                num_warps=4
            )

            # Compute lse_j via Triton
            lse_j = torch.empty(1, dtype=torch.float32, device=device)
            lse_kernel[(1,)](v, lse_j, L, inv_ln2)
            lse[b, j] = lse_j[0]

            # Compute attention attn[j, :] via Triton
            attn = torch.empty(L, dtype=torch.float32, device=device)
            softmax_base2_kernel[(L,)](v, lse[b, j], attn, L, inv_ln2)

            # Compute final output vector out[b, j, :] via Triton matvec
            # Output is bfloat16; compute as float32 then cast
            y = torch.empty(D, dtype=torch.float32, device=device)
            # One program per output dimension h
            grid_y = (D,)
            matvec_write_y_kernel[grid_y](
                attn, Kc, y, L, D, BLOCK_K=128,
                num_warps=4
            )

            # Store to output
            output[b, j] = y.to(torch.bfloat16)

    return output, lse


# Optional local helper (not used by harness); avoids recursion in environment.
def get_inputs():
    device = 'cuda'  # or torch.device('cuda')
    # Dummy tensors; evaluation harness provides real inputs
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


# Entry point: ModelNew
class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure tensors are on CUDA (if needed)
        for i in range(len(q_nope)):
            if isinstance(q_nope[i], torch.Tensor) and q_nope[i].device.type != 'cuda':
                q_nope[i] = q_nope[i].to('cuda')
        # Call run to orchestrate Triton kernels
        return run(q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale)