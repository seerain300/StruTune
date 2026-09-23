import math
import torch
import triton
import triton.language as tl


@triton.jit
def matvec_add_kernel(
    qn_ptr,           # *float32, [D]
    qp_ptr,           # *float32, [Dp]
    Kc_ptr,           # *float32, [L, D], row-major
    Kp_ptr,           # *float32, [L, Dp], row-major
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

    # Reduce over Kc dimension (D)
    for k in range(0, D, BLOCK_K):
        k_off = k + tl.arange(0, BLOCK_K)
        mask_k = k_off < D
        qn_slice = tl.load(qn_ptr + k_off, mask=mask_k, other=0.0)  # [BLOCK_K]
        kc_ptr = Kc_ptr + i * D + k_off
        kc_slice = tl.load(kc_ptr, mask=mask_k, other=0.0)         # [BLOCK_K]
        sum1 += tl.sum(qn_slice * kc_slice, axis=0)

    # Reduce over Kp dimension (Dp)
    for p in range(0, Dp, BLOCK_K):
        p_off = p + tl.arange(0, BLOCK_K)
        mask_p = p_off < Dp
        qp_slice = tl.load(qp_ptr + p_off, mask=mask_p, other=0.0) # [BLOCK_K]
        kp_ptr = Kp_ptr + i * Dp + p_off
        kp_slice = tl.load(kp_ptr, mask=mask_p, other=0.0)         # [BLOCK_K]
        sum2 += tl.sum(qp_slice * kp_slice, axis=0)

    # Write v[i] = sum1 + sum2
    tl.store(v_ptr + i, sum1 + sum2)


@triton.jit
def lse_kernel(
    v_ptr,            # *float32, [L]
    lse_ptr,          # *float32, scalar for this row
    L: tl.int32,
    inv_ln2: tl.float32,
    BLOCK: tl.constexpr,
):
    # Compute lse = log(sum(exp(v * inv_ln2))) / ln(2) via stable max-reduce
    max_v = -float('inf')
    # Pass 1: find max
    for offs in range(0, L, BLOCK):
        idx = offs + tl.arange(0, BLOCK)
        mask = idx < L
        vi = tl.load(v_ptr + idx, mask=mask, other=-float('inf'))
        local_max = tl.max(vi, axis=0)
        max_v = tl.maximum(max_v, local_max)

    # Pass 2: sum exp(v - max_v) scaled by inv_ln2
    sum_exp = 0.0
    for offs in range(0, L, BLOCK):
        idx = offs + tl.arange(0, BLOCK)
        mask = idx < L
        vi = tl.load(v_ptr + idx, mask=mask, other=0.0)
        expv = tl.exp((vi - max_v) * inv_ln2)  # exp((v - max) * 1/ln(2))
        sum_exp += tl.sum(expv, axis=0)

    lse_val = tl.log(sum_exp) + max_v
    tl.store(lse_ptr, lse_val)


@triton.jit
def softmax_base2_kernel(
    v_ptr,            # *float32, [L]
    attn_ptr,         # *float32, [L]
    lse_ptr,          # *float32, scalar lse for this head
    L: tl.int32,
    inv_ln2: tl.float32,
    BLOCK: tl.constexpr,
):
    # One program per output index i in [0, L)
    i = tl.program_id(0)
    # Load lse
    lse = tl.load(lse_ptr)
    # Compute normalized attn[i] = exp((v[i] - lse) * inv_ln2)
    vi = tl.load(v_ptr + i)
    attn = tl.exp((vi - lse) * inv_ln2)
    tl.store(attn_ptr + i, attn)


@triton.jit
def matvec_write_y_kernel(
    attn_ptr,         # *float32, [L]
    Kc_ptr,           # *float32, [L, D], row-major (we want Kc[:, D])
    out_ptr,          # *float32, [D]
    L: tl.int32,      # number of tokens
    D: tl.int32,      # head_dim_ckv
    BLOCK: tl.constexpr,
):
    # One program per output index j in [0, D)
    j = tl.program_id(0)
    acc = 0.0
    # Reduce over tokens: out[j] = sum_i attn[i] * Kc[i, j]
    for i in range(0, L, BLOCK):
        ii = i + tl.arange(0, BLOCK)
        mask_i = ii < L
        attn_i = tl.load(attn_ptr + ii, mask=mask_i, other=0.0)     # [BLOCK]
        kc_ptr = Kc_ptr + ii * D + j
        kc_i = tl.load(kc_ptr, mask=mask_i, other=0.0)              # [BLOCK]
        acc += tl.sum(attn_i * kc_i, axis=0)
    tl.store(out_ptr + j, acc)


# The original get_inputs helper (kept for compatibility with the evaluation harness).
def get_inputs():
    q_nope = torch.randn([1, 16, 512], dtype=torch.bfloat16)
    q_pe = torch.randn([1, 16, 64], dtype=torch.bfloat16)
    ckv_cache = torch.randn([989669, 1, 512], dtype=torch.bfloat16)
    kpe_cache = torch.randn([989669, 1, 64], dtype=torch.bfloat16)
    _n = 1; _t = 8
    _lens = torch.full((_n,), _t // _n, dtype=torch.int32)
    _lens[: _t % _n] += 1
    kv_indptr = torch.cat([torch.zeros(1, dtype=torch.int32), torch.cumsum(_lens, 0)]).to(torch.int32)
    kv_indices = torch.randint(0, 989669, [8], dtype=torch.int32)
    sm_scale = 1.0  # float32 scalar
    return [q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale]


# Fused operator calling the Triton kernels. The evaluation harness will invoke this.
def fused_operator(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6):
    return run(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6)


# Original logic with Triton kernels integrated and invoked from ModelNew.forward.
@torch.no_grad()
def run(q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
    # Ensure CUDA tensors
    device = 'cuda'
    q_nope = q_nope.to(device)
    q_pe = q_pe.to(device)
    ckv_cache = ckv_cache.to(device)
    kpe_cache = kpe_cache.to(device)
    kv_indptr = kv_indptr.to(device)
    kv_indices = kv_indices.to(device)

    batch_size = q_nope.shape[0]
    num_qo_heads = q_nope.shape[1]
    head_dim_ckv = q_nope.shape[2]  # 512
    head_dim_kpe = q_pe.shape[2]    # 64
    num_pages = ckv_cache.shape[0]
    # Checks
    assert num_qo_heads == 16, "num_qo_heads must be 16"
    assert head_dim_ckv == 512, "head_dim_ckv must be 512"
    assert head_dim_kpe == 64, "head_dim_kpe must be 64"
    assert ckv_cache.shape[1] == 1 and kpe_cache.shape[1] == 1, "cache second dim must be 1"
    assert kv_indptr.shape[0] == batch_size + 1, "kv_indptr length must be batch_size + 1"

    Kc_all = ckv_cache.squeeze(1).contiguous().to(torch.float32)  # [num_pages, 512]
    Kp_all = kpe_cache.squeeze(1).contiguous().to(torch.float32)  # [num_pages, 64]

    # Output buffers
    output = torch.zeros((batch_size, num_qo_heads, head_dim_ckv), dtype=torch.float32, device=device)
    lse = torch.full((batch_size, num_qo_heads), -float("inf"), dtype=torch.float32, device=device)

    inv_ln2 = 1.0 / math.log(2.0)

    for b in range(batch_size):
        # Determine token range [page_beg, page_end)
        page_beg = int(kv_indptr[b].item())
        page_end = int(kv_indptr[b + 1].item())

        if page_beg >= page_end:
            # No KV cache for this batch element
            output[b].zero_()
            continue

        L_tokens = page_end - page_beg
        tok_idx = kv_indices[page_beg:page_end].to(torch.int32)  # indices into Kc_all/Kp_all

        # Gather Kc and Kp for this batch
        Kc = Kc_all[tok_idx].contiguous()  # [L_tokens, 512], float32
        Kp = Kp_all[tok_idx].contiguous()  # [L_tokens, 64], float32

        # Prepare qn and qp as float32
        qn = q_nope[b].to(torch.float32).contiguous()  # [16, 512]
        qp = q_pe[b].to(torch.float32).contiguous()    # [16, 64]

        # Compute v[j, :] per head j using Triton
        for j in range(num_qo_heads):
            # v for this head
            v = torch.empty(L_tokens, dtype=torch.float32, device=device)
            # Launch matvec_add_kernel for head j
            grid_v = (L_tokens,)
            matvec_add_kernel[grid_v](
                qn[j], qp[j], Kc, Kp, v, L_tokens, head_dim_ckv, head_dim_kpe,
                BLOCK_K=128, num_warps=4
            )

            # Compute lse for this head using Triton
            lse_j = torch.empty((), dtype=torch.float32, device=device)
            grid_lse = (1,)
            lse_kernel[grid_lse](
                v, lse_j, L_tokens, inv_ln2,
                BLOCK=128, num_warps=4
            )
            lse[b, j] = lse_j

            # Compute attn vector (softmax base-2) for this head using Triton
            attn = torch.empty(L_tokens, dtype=torch.float32, device=device)
            grid_softmax = (L_tokens,)
            softmax_base2_kernel[grid_softmax](
                v, attn, lse[b, j], L_tokens, inv_ln2,
                BLOCK=128, num_warps=4
            )

            # Final matvec: out[b, j, :] = attn @ Kc[:, 512]
            out_row = torch.empty(head_dim_ckv, dtype=torch.float32, device=device)
            grid_y = (head_dim_ckv,)
            matvec_write_y_kernel[grid_y](
                attn, Kc, out_row, L_tokens, head_dim_ckv,
                BLOCK=128, num_warps=4
            )
            output[b, j, :] = out_row

    # Cast output to bfloat16 to match original, keep lse as float32
    output_bf16 = output.to(torch.bfloat16)
    return output_bf16, lse


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Run the Triton-orchestrated computation
        return run(*args)


def run(*args):
    return ModelNew()(*args)
