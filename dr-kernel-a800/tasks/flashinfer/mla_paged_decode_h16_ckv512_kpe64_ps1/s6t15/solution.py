import math
import torch
import triton
import triton.language as tl


@triton.jit
def compute_logits_per_batch_kernel(
    qn_ptr,         # *fp32, shape [H, D], contiguous (D=512, compile-time const via tl.constexpr)
    qp_ptr,         # *fp32, shape [H, Dp], contiguous (Dp=64)
    Kc_ptr,         # *fp32, shape [L_tokens, D], contiguous
    Kp_ptr,         # *fp32, shape [L_tokens, Dp], contiguous
    logits_ptr,     # *fp32, flattened buffer [B*H*L_tokens]
    H,              # int32 (runtime)
    D: tl.constexpr,          # compile-time constant: 512
    Dp: tl.constexpr,         # compile-time constant: 64
    L_tokens,       # int32 (runtime number of valid tokens for this batch element)
    b,              # int32 batch id (runtime scalar)
):
    # 2D grid: (h in 0..H-1, t in 0..L_tokens-1)
    h = tl.program_id(0)
    t = tl.program_id(1)

    # Compute flat index for logits[b, h, t]
    idx = (b * H + h) * L_tokens + t

    # Load qn row for head h: [D]
    acc1 = 0.0
    for kk in range(0, D):
        val = tl.load(qn_ptr + h * D + kk)  # qn_ptr is contiguous across heads: row offset is h * D
        kv = tl.load(Kc_ptr + t * D + kk)   # Kc_ptr contiguous: row offset is t * D, column kk
        acc1 += val * kv

    # Load qp row for head h: [Dp]
    acc2 = 0.0
    for kk in range(0, Dp):
        val = tl.load(qp_ptr + h * Dp + kk)
        kv = tl.load(Kp_ptr + t * Dp + kk)
        acc2 += val * kv

    logit = acc1 + acc2
    tl.store(logits_ptr + idx, logit)


@triton.jit
def compute_lse_per_batch_kernel(
    logits_ptr,     # *fp32, flattened buffer [B*H*L_tokens]
    lse_ptr,        # *fp32, buffer [B*H]
    H,              # int32
    L_tokens,       # int32
    b,              # int32
):
    # Grid: (h,)
    h = tl.program_id(0)
    base = b * H + h
    row_start = base * L_tokens

    # Compute max for numerical stability
    m = -float("inf")
    for t in range(0, L_tokens):
        logit = tl.load(logits_ptr + row_start + t)
        m = tl.maximum(m, logit)

    sum_exp = 0.0
    for t in range(0, L_tokens):
        logit = tl.load(logits_ptr + row_start + t)
        sum_exp += tl.exp(logit - m)

    ln2 = 1.4426950408889634  # 1 / log(2)
    lse = tl.log(sum_exp) / ln2
    tl.store(lse_ptr + (b * H + h), lse)


@triton.jit
def compute_output_per_batch_kernel(
    logits_ptr,     # *fp32, [B*H*L_tokens]
    Kc_ptr,         # *fp32, [L_tokens, D], contiguous
    output_ptr,     # *fp32, [B*H*D], contiguous
    H,              # int32
    D: tl.constexpr,          # 512
    L_tokens,       # int32
    b,              # int32
):
    # Grid: (h in 0..H-1, d in 0..D-1)
    h = tl.program_id(0)
    d = tl.program_id(1)

    base = b * H + h
    row_start = base * L_tokens
    out_row = output_ptr + (base * D + d)

    # Initialize output (store 0); we'll sum contributions below
    tl.store(out_row, 0.0)

    m = -float("inf")
    for t in range(0, L_tokens):
        logit = tl.load(logits_ptr + row_start + t)
        m = tl.maximum(m, logit)

    # Accumulate output[h, d] = sum_t softmax_t * Kc[t, d]
    for t in range(0, L_tokens):
        logit = tl.load(logits_ptr + row_start + t)
        e = tl.exp(logit - m)
        sum_exp = 0.0
        for tt in range(0, L_tokens):
            logit2 = tl.load(logits_ptr + row_start + tt)
            sum_exp += tl.exp(logit2 - m)
        attn = e / sum_exp
        kv = tl.load(Kc_ptr + t * D + d)
        acc = tl.load(out_row) + (attn * kv)
        tl.store(out_row, acc)


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure CUDA and contiguous
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda, "Tensors must be on CUDA device"
        device = q_nope.device

        # Extract shapes
        B, H, D = q_nope.shape  # B=batch_size, H=num_qo_heads, D=head_dim_ckv (512)
        Dp = q_pe.shape[-1]     # 64
        # Sanity checks consistent with original code
        assert H == 16
        assert D == 512
        assert Dp == 64
        assert ckv_cache.shape[1] == 1 and kpe_cache.shape[1] == 1, "Cache second dim must be 1"
        # Squeeze batch dim since cache has [num_pages, 1, D]
        Kc_all = ckv_cache.squeeze(1).contiguous().to(torch.float32)  # [num_pages, D]
        Kp_all = kpe_cache.squeeze(1).contiguous().to(torch.float32)  # [num_pages, Dp]

        # Compute per-batch token counts from kv_indptr
        # kv_indptr shape: [B+1], int32
        L_tokens_list = (kv_indptr[1:] - kv_indptr[:B]).tolist()  # Python list of ints
        assert len(L_tokens_list) == B, "L_tokens_list length must match batch size"

        # Allocate buffers
        logits = torch.empty(B * H * max(L_tokens_list), dtype=torch.float32, device=device)
        lse = torch.empty((B, H), dtype=torch.float32, device=device)
        output = torch.empty((B, H, D), dtype=torch.float32, device=device)  # compute in fp32, cast later

        # Launch kernels: compute logits per batch b using Triton
        for b in range(B):
            L_tokens = int(L_tokens_list[b])
            # Slice kv_indices for this batch element
            tok_idx = kv_indices[kv_indptr[b]: kv_indptr[b + 1]].contiguous().to(torch.int32)

            # Select corresponding rows in Kc_all and Kp_all
            Kc_batch = Kc_all[tok_idx]  # [L_tokens, D]
            Kp_batch = Kp_all[tok_idx]  # [L_tokens, Dp]

            # Cast q_nope and q_pe to float32 for computation
            qn = q_nope[b].contiguous().to(torch.float32)  # [H, D]
            qp = q_pe[b].contiguous().to(torch.float32)   # [H, Dp]

            # Launch Triton kernel to compute logits[b, :, :]
            grid_logits = (H, L_tokens)
            compute_logits_per_batch_kernel[grid_logits](
                qn, qp, Kc_batch, Kp_batch, logits, H, D, Dp, L_tokens, b,
                num_warps=4, num_stages=2
            )

        # Second kernel: compute lse per (b, h) by re-launching per b
        for b in range(B):
            L_tokens = int(L_tokens_list[b])
            lse_b = torch.empty((H,), dtype=torch.float32, device=device)
            compute_lse_per_batch_kernel[(H,)](
                logits, lse_b, H, L_tokens, b,
                num_warps=4, num_stages=2
            )
            lse[b] = lse_b  # lse has shape [B, H]

        # Third kernel: compute output per (b, h) by re-launching per b
        for b in range(B):
            L_tokens = int(L_tokens_list[b])
            grid_output = (H, D)
            compute_output_per_batch_kernel[grid_output](
                logits, Kc_all, output, H, D, L_tokens, b,
                num_warps=4, num_stages=2
            )

        # Cast output to bfloat16 as in original
        output_bf16 = output.to(torch.bfloat16)

        return output_bf16, lse


def run(*args):
    return ModelNew()(*args)
