import math
import torch
import triton
import triton.language as tl


# Kernel 1: Compute logits vector v[j, :] = sum_i (qn[j, :] · Kc[i, :]) + sum_i (qp[j, :] · Kp[i, :])
@triton.jit
def matvec_add_kernel(
    qn_ptr,          # *float32, [D]
    qp_ptr,          # *float32, [Dp]
    Kc_ptr,          # *float32, [L, D], row-major
    Kp_ptr,          # *float32, [L, Dp], row-major
    v_ptr,           # *float32, [L]
    L: tl.int32,     # number of tokens
    D: tl.int32,     # head_dim_ckv
    Dp: tl.int32,    # head_dim_kpe
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
        kc_ptr_row = Kc_ptr + i * D + k_off
        kc_row = tl.load(kc_ptr_row, mask=mask_k, other=0.0)        # [BLOCK_K]
        sum1 += tl.sum(qn_slice * kc_row, axis=0)
    # Reduce over Kp dimension (Dp)
    for p in range(0, Dp, BLOCK_K):
        p_off = p + tl.arange(0, BLOCK_K)
        mask_p = p_off < Dp
        qp_slice = tl.load(qp_ptr + p_off, mask=mask_p, other=0.0)  # [BLOCK_K]
        kp_ptr_row = Kp_ptr + i * Dp + p_off
        kp_row = tl.load(kp_ptr_row, mask=mask_p, other=0.0)        # [BLOCK_K]
        sum2 += tl.sum(qp_slice * kp_row, axis=0)
    v[i] = sum1 + sum2


# Kernel 2: Compute lse_j = logsumexp_base2(v[j, :]) = max(v) + log(sum(exp(v - max)) / log(2))
@triton.jit
def lse_base2_kernel(
    v_ptr,           # *float32, [L]
    L: tl.int32,
    lse_ptr,         # *float32, [1] (scalar)
    ln2_inv: tl.float32,   # 1 / log(2)
):
    # One program to reduce over L
    m = -float("inf")
    # Pass 1: find max
    for i in range(0, L):
        m = tl.maximum(m, tl.load(v_ptr + i))
    # Pass 2: sum exp normalized by ln(2)
    sumexp = 0.0
    for i in range(0, L):
        vi = tl.load(v_ptr + i)
        sumexp += tl.exp(vi - m) * ln2_inv  # * (1 / ln(2)) == * log2(e)
    lse_val = m + tl.log(sumexp)
    tl.store(lse_ptr, lse_val)


# Kernel 3: Compute attn[i] = exp(v[j, i] / ln(2) - lse_j) for all i
@triton.jit
def softmax_base2_kernel(
    v_ptr,           # *float32, [L]
    lse_ptr,         # *float32, [1] (scalar lse_j)
    attn_ptr,        # *float32, [L]
    L: tl.int32,
    ln2_inv: tl.float32,   # 1 / log(2)
):
    lse_val = tl.load(lse_ptr)
    for i in range(0, L):
        vi = tl.load(v_ptr + i)
        attn_i = tl.exp(vi * ln2_inv - lse_val)
        tl.store(attn_ptr + i, attn_i)


# Kernel 4: Compute out[j, :] = attn @ Kc (i.e., for each output dim h: sum_i attn[i] * Kc[i, h])
@triton.jit
def matvec_write_y_kernel(
    attn_ptr,        # *float32, [L]
    Kc_ptr,          # *float32, [L, D], row-major
    y_ptr,           # *float32, [D]
    L: tl.int32,
    D: tl.int32,
    BLOCK_K: tl.constexpr,
):
    h = tl.program_id(0)
    # Reduce over L
    sum_h = 0.0
    for i in range(0, L, BLOCK_K):
        idx = i + tl.arange(0, BLOCK_K)
        mask = idx < L
        attn_i = tl.load(attn_ptr + idx, mask=mask, other=0.0)      # [BLOCK_K]
        kc_ptr_row = Kc_ptr + idx * D + h
        kc_row = tl.load(kc_ptr_row, mask=mask, other=0.0)          # [BLOCK_K]
        sum_h += tl.sum(attn_i * kc_row, axis=0)
    tl.store(y_ptr + h, sum_h)


@torch.no_grad()
def run(q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
    # Ensure CUDA tensors
    device = q_nope.device
    batch_size = q_nope.shape[0]
    num_qo_heads = q_nope.shape[1]
    head_dim_ckv = q_nope.shape[2]
    head_dim_kpe = q_pe.shape[2]
    num_kv_indices = kv_indices.shape[0]

    # Constants and assertions from the original
    assert num_qo_heads == 16
    assert head_dim_ckv == 512
    assert head_dim_kpe == 64
    # For this workload, num_pages and len_indptr are dynamic; we respect them.

    # Prepare Kc_all and Kp_all from cache
    Kc_all = ckv_cache.squeeze(1).contiguous().to(torch.float32)  # [num_pages, 512]
    Kp_all = kpe_cache.squeeze(1).contiguous().to(torch.float32)  # [num_pages, 64]

    output = torch.empty(
        (batch_size, num_qo_heads, head_dim_ckv), dtype=torch.float32, device=device
    )
    lse = torch.empty((batch_size, num_qo_heads), dtype=torch.float32, device=device)

    ln2_inv = 1.0 / math.log(2.0)

    for b in range(batch_size):
        # Determine token range for this batch element
        if kv_indptr.numel() < 2 or b + 1 >= kv_indptr.numel():
            # Fallback safe case: no valid indptr, produce zeros
            output[b] = 0.0
            lse[b] = -float("inf")
            continue

        # Gather tokens for this batch
        start = int(kv_indptr[b].item())
        end = int(kv_indptr[b + 1].item())
        L = end - start
        if L <= 0 or start >= end:
            output[b].zero_()
            lse[b] = -float("inf")
            continue

        tok_idx = kv_indices[start:end].to(torch.int64)  # indices into cache

        # Slice Kc and Kp for this batch
        Kc_batch = Kc_all[tok_idx]  # [L, 512], float32
        Kp_batch = Kp_all[tok_idx]  # [L, 64], float32

        # Prepare qn and qp for head j
        for j in range(num_qo_heads):
            qn = q_nope[b, j, :].to(torch.float32).contiguous()  # [512]
            qp = q_pe[b, j, :].to(torch.float32).contiguous()    # [64]

            # 1) Compute logits v[j, :] = qn · Kc^T + qp · Kp^T
            v = torch.empty(L, dtype=torch.float32, device=device)
            # Launch Triton kernel: one program per i
            grid_v = (L,)
            matvec_add_kernel[grid_v](
                qn, qp, Kc_batch, Kp_batch, v, L, 512, 64, BLOCK_K=128, num_warps=4
            )

            # 2) Compute lse_j = logsumexp_base2(v)
            lse_bj = torch.empty(1, dtype=torch.float32, device=device)
            lse_base2_kernel[(1,)](v, L, lse_bj, ln2_inv)
            lse_val = lse_bj[0]
            lse[b, j] = lse_val

            # 3) Compute attention attn[i] = exp(v[i] / ln(2) - lse_val)
            attn = torch.empty(L, dtype=torch.float32, device=device)
            softmax_base2_kernel[(1,)](v, lse_bj, attn, L, ln2_inv)

            # 4) Compute out[b, j, :] = attn @ Kc[:, :]
            y = torch.empty(512, dtype=torch.float32, device=device)
            matvec_write_y_kernel[(512,)](attn, Kc_batch, y, L, 512, BLOCK_K=128, num_warps=4)

            # Store into output
            output[b, j, :] = y

    # Cast output back to bfloat16 to match original signature
    output = output.to(torch.bfloat16)
    return output, lse


def get_inputs():
    # Helper: generate consistent inputs for local testing.
    # The evaluation harness will supply its own inputs; this function is not recursive.
    batch_size = 1
    num_tokens = 8  # The harness axes define actual sizes; this is a placeholder for local testing
    device = 'cuda'
    q_nope = torch.randn([batch_size, 16, 512], dtype=torch.bfloat16, device=device)
    q_pe = torch.randn([batch_size, 16, 64], dtype=torch.bfloat16, device=device)
    num_pages = 989669
    ckv_cache = torch.randn([num_pages, 1, 512], dtype=torch.bfloat16, device=device)
    kpe_cache = torch.randn([num_pages, 1, 64], dtype=torch.bfloat16, device=device)
    kv_indptr = torch.tensor([0, num_tokens], dtype=torch.int32, device=device)
    kv_indices = torch.randint(0, num_pages, [num_tokens], dtype=torch.int32, device=device)
    sm_scale = 1.0
    return [q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale]


def fused_operator(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6):
    _out = run(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6)
    return list(_out) if isinstance(_out, (tuple, list)) else [_out]


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Ensure CUDA tensors and call the Triton-orchestrated run
        for i in range(len(args)):
            if isinstance(args[i], torch.Tensor) and args[i].device.type != 'cuda':
                args[i] = args[i].to('cuda')
        return run(*args)


def run(*args):
    return ModelNew()(*args)
