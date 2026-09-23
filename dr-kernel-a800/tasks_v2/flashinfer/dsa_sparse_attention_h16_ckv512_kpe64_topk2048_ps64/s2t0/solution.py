import math
import torch
import triton
import triton.language as tl


@triton.jit
def compute_logits_kernel(
    q_nope_ptr, q_pe_ptr, Kc_all_ptr, Kp_all_ptr, sparse_idx_ptr,
    logits_ptr,
    N, H, Dk, Dp, topk,
    BLOCK_K: tl.constexpr  # typically 1024 or 2048 to cover topk candidates
):
    # Each program handles one token t and one head h
    t = tl.program_id(0)
    h = tl.program_id(1)

    # Base offsets
    qn_base = t * H * Dk + h * Dk
    qp_base = t * H * Dp + h * Dp

    # Loop over candidates in chunks of BLOCK_K
    for chunk in range(0, topk, BLOCK_K):
        offs = chunk + tl.arange(0, BLOCK_K)
        mask = offs < topk

        # Load indices for this chunk
        idx_vals = tl.load(sparse_idx_ptr + t * topk + offs, mask=mask, other=-1)

        # Compute Kc/Kp rows for each candidate
        for i in range(0, BLOCK_K):
            if mask[i]:
                tok_idx = idx_vals[i].to(tl.int32)
                Kc_row_ptr = Kc_all_ptr + tok_idx * Dk
                Kp_row_ptr = Kp_all_ptr + tok_idx * Dp

                # Load query vectors qn[h] and qp[h]
                qn_vec = tl.load(q_nope_ptr + qn_base + tl.arange(0, Dk), mask=tl.arange(0, Dk) < Dk, other=0.0)
                qp_vec = tl.load(q_pe_ptr + qp_base + tl.arange(0, Dp), mask=tl.arange(0, Dp) < Dp, other=0.0)

                # Load K rows
                Kc_row = tl.load(Kc_row_ptr + tl.arange(0, Dk), mask=tl.arange(0, Dk) < Dk, other=0.0)
                Kp_row = tl.load(Kp_row_ptr + tl.arange(0, Dp), mask=tl.arange(0, Dp) < Dp, other=0.0)

                # Compute dot products
                dot_qn = tl.sum(qn_vec * Kc_row, axis=0)
                dot_qp = tl.sum(qp_vec * Kp_row, axis=0)
                logit = dot_qn + dot_qp

                # Store logits into logits[t, h, i]
                logits_row_base = t * H * topk + h * topk + i
                tl.store(logits_ptr + logits_row_base, logit)


@triton.jit
def softmax_kernel(
    logits_ptr, attn_ptr,
    N, H, topk,
    sm_scale
):
    # Each program handles one token t and one head h
    t = tl.program_id(0)
    h = tl.program_id(1)

    # Compute logsumexp across topk
    lse_acc = tl.full((), -1e30, dtype=tl.float32)
    sum_exp = tl.full((), 0.0, dtype=tl.float32)

    for i in range(0, topk):
        logit = tl.load(logits_ptr + t * H * topk + h * topk + i)
        logit_scaled = logit * sm_scale
        m = tl.maximum(lse_acc, logit_scaled)
        sum_exp = sum_exp * tl.exp(lse_acc - m) + tl.exp(logit_scaled - m)
        lse_acc = m

    # Write LSE for this head (not strictly needed here, but kept for consistency if used)
    # Compute and store attention weights
    for i in range(0, topk):
        logit = tl.load(logits_ptr + t * H * topk + h * topk + i)
        logit_scaled = logit * sm_scale
        exp_term = tl.exp(logit_scaled - lse_acc)
        attn_val = exp_term / sum_exp
        tl.store(attn_ptr + t * H * topk + h * topk + i, attn_val)


@triton.jit
def output_kernel(
    attn_ptr, Kc_all_ptr, output_ptr,
    N, H, Dk, topk,
    BLOCK_K: tl.constexpr
):
    # Each program handles one token t and one head h
    t = tl.program_id(0)
    h = tl.program_id(1)

    # Accumulator for output vector
    out_vec = tl.zeros((Dk,), dtype=tl.float32)

    # Reduction over candidates: out_vec = sum_j attn[t, h, j] * Kc_row[j]
    for chunk in range(0, topk, BLOCK_K):
        offs = chunk + tl.arange(0, BLOCK_K)
        mask = offs < topk

        for i in range(0, BLOCK_K):
            if mask[i]:
                attn_val = tl.load(attn_ptr + t * H * topk + h * topk + offs[i])
                tok_idx = offs[i]
                Kc_row_ptr = Kc_all_ptr + tok_idx * Dk
                Kc_row = tl.load(Kc_row_ptr + tl.arange(0, Dk), mask=tl.arange(0, Dk) < Dk, other=0.0)
                out_vec = out_vec + attn_val * Kc_row

    # Store output[t, h, :]
    out_row_base = t * H * Dk + h * Dk
    tl.store(output_ptr + out_row_base + tl.arange(0, Dk), out_vec, mask=tl.arange(0, Dk) < Dk)


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
        # Ensure CUDA device
        device = q_nope.device
        assert device.type == "cuda", "Triton kernels require CUDA tensors."

        # Cast inputs to float32 for compute
        q_nope = q_nope.to(torch.float32).contiguous()  # [N, H, Dk]
        q_pe = q_pe.to(torch.float32).contiguous()     # [N, H, Dp]
        N, H, Dk = q_nope.shape
        Dp = q_pe.shape[2]

        # Flatten paged KV cache
        num_pages, _, _ = ckv_cache.shape
        Kc_all = ckv_cache.reshape(-1, Dk).to(torch.float32).contiguous()  # [num_pages*64, Dk]
        Kp_all = kpe_cache.reshape(-1, Dp).to(torch.float32).contiguous()  # [num_pages*64, Dp]

        # Flatten sparse_indices to [N*topk]
        topk = sparse_indices.shape[1]


def run(*args):
    return ModelNew()(*args)
