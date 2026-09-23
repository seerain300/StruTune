import math
import torch
import triton
import triton.language as tl


# Triton kernels: softmax_row_causal (row-wise softmax with causal mask) and
# lse_row_causal (row-wise logsumexp, base-2, with causal mask)
@triton.jit
def softmax_row_causal(
    X_ptr, Y_ptr,
    M: tl.int32, N: tl.int32,
    absolute_pos: tl.int32,
    BLOCK: tl.constexpr,
):
    # One program per row
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    mask = offsets < N

    # Load row x
    x = tl.load(X_ptr + row * N + offsets, mask=mask, other=-float("inf"))
    # Causal mask: positions j > absolute_pos -> -inf
    j = offsets
    causal = j > absolute_pos
    x = tl.where(causal, -float("inf"), x)

    # Stable softmax
    m = tl.max(x, axis=0)
    x = x - m
    e = tl.exp(x)
    # zero-out invalid positions
    e = tl.where(causal, 0.0, e)
    s = tl.sum(e, axis=0)
    y = e / s

    tl.store(Y_ptr + row * N + offsets, y, mask=mask)


@triton.jit
def lse_row_causal(
    X_ptr, Y_ptr,
    M: tl.int32, N: tl.int32,
    absolute_pos: tl.int32,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    mask = offsets < N
    x = tl.load(X_ptr + row * N + offsets, mask=mask, other=-float("inf"))
    j = offsets
    causal = j > absolute_pos
    x = tl.where(causal, -float("inf"), x)

    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(causal, 0.0, e)
    s = tl.sum(e, axis=0)
    lse = tl.log(s) / math.log(2.0)  # base-2 logsumexp
    tl.store(Y_ptr + row, lse)


@torch.no_grad()
def run(q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
    # Ensure CUDA device
    assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda, "Inputs must be CUDA tensors"

    total_q, num_qo_heads, head_dim_ckv = q_nope.shape
    head_dim_kpe = q_pe.shape[-1]
    assert num_qo_heads == 16, "num_qo_heads must be 16"
    assert head_dim_ckv == 512, "head_dim_ckv must be 512"
    assert head_dim_kpe == 64, "head_dim_kpe must be 64"

    # Flatten q_nope and q_pe if needed
    # Kc_all and Kp_all: [num_pages, head_dim], already
    Kc_all = ckv_cache.squeeze(1).to(torch.float32)  # [num_pages, 512]
    Kp_all = kpe_cache.squeeze(1).to(torch.float32)  # [num_pages, 64]

    device = q_nope.device
    output = torch.empty((total_q, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=device)
    lse = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=device)

    batch_size = qo_indptr.shape[0] - 1
    for b in range(batch_size):
        q_start = int(qo_indptr[b].item())
        q_end = int(qo_indptr[b + 1].item())
        if q_start >= q_end:
            continue

        q_len = q_end - q_start

        page_beg = int(kv_indptr[b].item())
        page_end = int(kv_indptr[b + 1].item())
        if page_beg >= page_end:
            continue
        kv_len = page_end - page_beg
        tok_idx = kv_indices[page_beg:page_end].to(torch.int32)
        Kc = Kc_all[tok_idx].contiguous()  # [kv_len, 512]
        Kp = Kp_all[tok_idx].contiguous()  # [kv_len, 64]

        for i in range(q_len):
            abs_q = q_start + i
            qn = q_nope[abs_q]  # [16, 512], bf16 -> cast to fp32
            qp = q_pe[abs_q]   # [16, 64], bf16 -> cast to fp32
            qn = qn.to(torch.float32).contiguous()
            qp = qp.to(torch.float32).contiguous()

            # Compute scores_n = qn @ Kc.T -> [16, kv_len]
            # qn: [M=16, Kq=512], Kc.T: [KcT=512, N=kv_len]
            scores_n = qn @ Kc.transpose(0, 1)  # [16, kv_len]
            # scores_p = qp @ Kp.T -> [16, kv_len]
            scores_p = qp @ Kp.transpose(0, 1)  # [16, kv_len]
            scores = scores_n + scores_p  # [16, kv_len]

            # Causal mask: positions j > (prefix_len + i) => -inf
            prefix_len = kv_len - q_len
            absolute_pos = prefix_len + i

            # Softmax with causal mask (row-wise), Triton kernel
            attn = torch.empty((scores.shape[0], scores.shape[1]), dtype=torch.float32, device=device)
            M = scores.shape[0]
            N = scores.shape[1]
            grid_s = (M,)
            # BLOCK should be a power of two; choose 256
            softmax_row_causal[grid_s](
                scores, attn,
                M, N,
                absolute_pos,
                BLOCK=256,
                num_warps=4,
            )

            # LSE per head (base-2), Triton kernel
            lse_row = torch.empty((M,), dtype=torch.float32, device=device)
            grid_l = (M,)
            lse_row_causal[grid_l](
                scores, lse_row,
                M, N,
                absolute_pos,
                BLOCK=256,
                num_warps=4,
            )
            lse[abs_q] = lse_row  # [16]

            # Output: attn @ Kc -> [16, 512]
            out_row = attn @ Kc  # [16, 512]
            output[abs_q] = out_row.to(torch.bfloat16)

    return output, lse


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        return run(q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale)


def run(*args):
    return ModelNew()(*args)
