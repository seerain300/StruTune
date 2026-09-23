import math
import torch
import triton
import triton.language as tl


@triton.jit
def matvec_add_kernel(
    qn_ptr,         # *float32, [D], for a single head j
    qp_ptr,         # *float32, [Dp], for a single head j
    Kc_ptr,         # *float32, [L, D], row-major (tokens, D)
    Kp_ptr,         # *float32, [L, Dp], row-major (tokens, Dp)
    v_ptr,          # *float32, [L]
    L: tl.int32,    # number of tokens
    D: tl.int32,    # head_dim_ckv
    Dp: tl.int32,   # head_dim_kpe
    inv_ln2: tl.float32,
    BLOCK_K: tl.constexpr,
):
    # One program computes one output index i
    i = tl.program_id(0)
    if i >= L:
        return
    sum1 = 0.0
    sum2 = 0.0
    # Reduce over D in blocks
    for k in range(0, D, BLOCK_K):
        k_off = k + tl.arange(0, BLOCK_K)
        mask_k = k_off < D
        qn_k = tl.load(qn_ptr + k_off, mask=mask_k, other=0.0)  # [BLOCK_K]
        kc_ptr_i = Kc_ptr + i * D + k_off
        kc_vec = tl.load(kc_ptr_i, mask=mask_k, other=0.0)      # [BLOCK_K]
        sum1 += tl.sum(qn_k * kc_vec, axis=0)

    # Reduce over Dp in blocks
    for p in range(0, Dp, BLOCK_K):
        p_off = p + tl.arange(0, BLOCK_K)
        mask_p = p_off < Dp
        qp_p = tl.load(qp_ptr + p_off, mask=mask_p, other=0.0)  # [BLOCK_K]
        kp_ptr_i = Kp_ptr + i * Dp + p_off
        kp_vec = tl.load(kp_ptr_i, mask=mask_p, other=0.0)      # [BLOCK_K]
        sum2 += tl.sum(qp_p * kp_vec, axis=0)

    v = sum1 + sum2
    tl.store(v_ptr + i, v)


@triton.jit
def lse_base2_kernel(
    v_ptr,          # *float32, [L]
    lse_ptr,        # *float32, scalar
    L: tl.int32,
    inv_ln2: tl.float32,
):
    # Single program performs stable reduction over L
    max_v = -float("inf")
    sum_exp = 0.0
    for i in range(0, L):
        vi = tl.load(v_ptr + i)
        si = vi * inv_ln2  # scaled by 1/ln(2) for base-2
        # stable: subtract max_v
        e = tl.exp(si - max_v)
        sum_exp += e
        # update max_v if needed
        max_v = tl.maximum(max_v, vi)

    lse = max_v + math.log(2.0) * tl.log(sum_exp)
    tl.store(lse_ptr, lse)


@triton.jit
def softmax_base2_kernel(
    v_ptr,          # *float32, [L]
    lse_ptr,        # *float32, scalar
    attn_ptr,       # *float32, [L]
    L: tl.int32,
    inv_ln2: tl.float32,
):
    lse = tl.load(lse_ptr)
    for i in range(0, L):
        vi = tl.load(v_ptr + i)
        si = vi * inv_ln2
        attn = tl.exp(si - lse)
        tl.store(attn_ptr + i, attn)


@triton.jit
def matvec_write_y_kernel(
    attn_ptr,       # *float32, [L]
    Kc_ptr,         # *float32, [L, D], row-major
    y_ptr,          # *float32, [D]
    L: tl.int32,
    D: tl.int32,
    BLOCK_K: tl.constexpr,
):
    # One program computes one output dimension h
    h = tl.program_id(0)
    if h >= D:
        return
    acc = 0.0
    for k in range(0, L, BLOCK_K):
        i_off = k + tl.arange(0, BLOCK_K)
        mask_i = i_off < L
        attn_vec = tl.load(attn_ptr + i_off, mask=mask_i, other=0.0)  # [BLOCK_K]
        kc_ptr_h = Kc_ptr + i_off * D + h
        kc_vec = tl.load(kc_ptr_h, mask=mask_i, other=0.0)            # [BLOCK_K]
        acc += tl.sum(attn_vec * kc_vec, axis=0)
    tl.store(y_ptr + h, acc)


def run(q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
    # Ensure device is CUDA
    if not q_nope.is_cuda:
        q_nope = q_nope.to('cuda')
    if not q_pe.is_cuda:
        q_pe = q_pe.to('cuda')
    if not ckv_cache.is_cuda:
        ckv_cache = ckv_cache.to('cuda')
    if not kpe_cache.is_cuda:
        kpe_cache = kpe_cache.to('cuda')
    if not kv_indptr.is_cuda:
        kv_indptr = kv_indptr.to('cuda')
    if not kv_indices.is_cuda:
        kv_indices = kv_indices.to('cuda')

    batch_size, num_qo_heads, head_dim_ckv = q_nope.shape
    head_dim_kpe = q_pe.shape[-1]
    assert num_qo_heads == 16
    assert head_dim_ckv == 512
    assert head_dim_kpe == 64
    num_pages = ckv_cache.shape[0]
    assert kpe_cache.shape[0] == num_pages
    len_indptr = kv_indptr.shape[0]
    assert len_indptr == batch_size + 1

    device = q_nope.device
    inv_ln2 = 1.0 / math.log(2.0)

    # Prepare output
    output = torch.zeros((batch_size, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=device)
    lse = torch.full((batch_size, num_qo_heads), -float("inf"), dtype=torch.float32, device=device)

    for b in range(batch_size):
        # Determine token range for this batch
        page_beg = int(kv_indptr[b].item())
        page_end = int(kv_indptr[b + 1].item())
        L = page_end - page_beg
        if L <= 0:
            lse[b, :] = -float("inf")
            continue

        tok_idx = kv_indices[page_beg:page_end].to(torch.long)

        # Gather Kc, Kp for this batch
        Kc = ckv_cache[tok_idx].to(torch.float32)  # [L, D]
        Kp = kpe_cache[tok_idx].to(torch.float32)  # [L, Dp]

        # v buffer for this batch
        v = torch.empty((L,), dtype=torch.float32, device=device)

        # Launch matvec_add_kernel: compute v[j, :] for each head j
        for j in range(num_qo_heads):
            qn = q_nope[b, j].to(torch.float32)  # [D]
            qp = q_pe[b, j].to(torch.float32)   # [Dp]
            # grid = (L,)
            matvec_add_kernel[(L,)](
                qn, qp, Kc, Kp, v,
                L, head_dim_ckv, head_dim_kpe,
                inv_ln2,
                BLOCK_K=64, num_warps=4
            )

            # scaled logits: v * sm_scale
            scaled = v * sm_scale

            # Compute lse for this head (scalar)
            lse_j = torch.empty((), dtype=torch.float32, device=device)
            lse_base2_kernel[(1,)](
                scaled, lse_j,
                L, inv_ln2
            )
            lse[b, j] = lse_j

            # Compute attention vector attn[j, :] = exp((scaled - lse_j) / ln(2))
            attn = torch.empty((L,), dtype=torch.float32, device=device)
            softmax_base2_kernel[(L,)](
                scaled, lse_j, attn,
                L, inv_ln2
            )

            # Write y[b, j, :] = attn @ Kc[:, :]
            y = torch.empty((head_dim_ckv,), dtype=torch.float32, device=device)
            matvec_write_y_kernel[(head_dim_ckv,)](
                attn, Kc, y,
                L, head_dim_ckv,
                BLOCK_K=64, num_warps=4
            )
            output[b, j, :] = y.to(torch.bfloat16)

    return output, lse


def get_inputs():
    # Helper for local testing; harness will provide its own inputs.
    # Construct tensors on CUDA for compatibility.
    batch_size = 1
    num_qo_heads = 16
    head_dim_ckv = 512
    head_dim_kpe = 64
    num_pages = 989669

    q_nope = torch.randn([batch_size, num_qo_heads, head_dim_ckv], dtype=torch.bfloat16, device='cuda')
    q_pe = torch.randn([batch_size, num_qo_heads, head_dim_kpe], dtype=torch.bfloat16, device='cuda')
    ckv_cache = torch.randn([num_pages, 1, head_dim_ckv], dtype=torch.bfloat16, device='cuda')
    kpe_cache = torch.randn([num_pages, 1, head_dim_kpe], dtype=torch.bfloat16, device='cuda')
    # Simple indptr for a single batch
    kv_indptr = torch.tensor([0, 10], dtype=torch.int32, device='cuda')  # length = 2
    # Token indices
    kv_indices = torch.randint(0, num_pages, [10], dtype=torch.int32, device='cuda')
    sm_scale = 1.0  # float32 scalar
    return [q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale]


def fused_operator(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6):
    _out = run(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6)
    return list(_out) if isinstance(_out, (tuple, list)) else [_out]


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Ensure inputs are on CUDA and run Triton kernels
        for i in range(len(args)):
            if isinstance(args[i], torch.Tensor) and args[i].device.type != 'cuda':
                args[i] = args[i].to('cuda')
        return run(*args)