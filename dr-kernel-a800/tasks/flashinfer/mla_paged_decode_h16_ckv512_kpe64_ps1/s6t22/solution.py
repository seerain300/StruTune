import math
import torch
import triton
import triton.language as tl


@triton.jit
def compute_logits_scaled_per_batch_kernel(
    qn_ptr,         # *fp32, shape [H, D], contiguous
    qp_ptr,         # *fp32, shape [H, Dp], contiguous
    Kc_ptr,         # *fp32, shape [L_tokens, D], contiguous
    Kp_ptr,         # *fp32, shape [L_tokens, Dp], contiguous
    logits_ptr,     # *fp32, flattened buffer [H*L_tokens], row-major: base = h * L_tokens
    H,              # int32
    D: tl.constexpr,          # compile-time constant: 512
    Dp: tl.constexpr,         # compile-time constant: 64
    L_tokens,       # int32
    sm_scale,       # float32
):
    # 2D grid: (h, t)
    h = tl.program_id(0)
    t = tl.program_id(1)

    base = h * L_tokens
    acc1 = 0.0
    # Compute dot(qn[h, :], Kc[t, :])
    for kk in range(0, D):
        val = tl.load(qn_ptr + h * D + kk)
        kv = tl.load(Kc_ptr + t * D + kk)
        acc1 += val * kv

    acc2 = 0.0
    # Compute dot(qp[h, :], Kp[t, :])
    for kk in range(0, Dp):
        val = tl.load(qp_ptr + h * Dp + kk)
        kv = tl.load(Kp_ptr + t * Dp + kk)
        acc2 += val * kv

    logit = (acc1 + acc2) * sm_scale
    tl.store(logits_ptr + base + t, logit)


@triton.jit
def compute_lse_per_row_kernel(
    logits_ptr,     # *fp32, [H*L_tokens], contiguous row-major
    lse_ptr,        # *fp32, [H]
    H,              # int32
    L_tokens,       # int32
):
    # 1D grid: (h,)
    h = tl.program_id(0)
    base = h * L_tokens
    row = tl.load(logits_ptr + base + tl.arange(0, L_tokens))
    m = tl.max(row)
    row = row - m
    exp_row = tl.exp(row)
    sumexp = tl.sum(exp_row)
    lse = tl.log(sumexp) / 0.6931471805599453  # log(2)
    tl.store(lse_ptr + h, lse)


@triton.jit
def compute_output_matvec_kernel(
    Kc_ptr,         # *fp32, [L_tokens, D], contiguous
    logits_ptr,     # *fp32, [H*L_tokens], contiguous
    output_ptr,     # *fp32, [H*D], contiguous row-major
    H,              # int32
    D: tl.constexpr,          # compile-time constant: 512
    L_tokens,       # int32
    sm_scale,       # float32 (kept for signature; not used)
):
    # 1D grid: (h,)
    h = tl.program_id(0)
    base_log = h * L_tokens
    out_vec = tl.zeros((D,), dtype=tl.float32)

    # Compute softmax over logits[h, :] stably: subtract max
    row = tl.load(logits_ptr + base_log + tl.arange(0, L_tokens))
    m = tl.max(row)
    row = row - m
    exp_row = tl.exp(row)
    sumexp = tl.sum(exp_row)
    softmax_row = exp_row / sumexp  # softmax probabilities

    # Accumulate out[h, j] = sum_t softmax(logits[h, t]) * Kc[t, j]
    for t in range(0, L_tokens):
        p = softmax_row[t]
        kc = tl.load(Kc_ptr + t * D + tl.arange(0, D))
        out_vec += p * kc

    tl.store(output_ptr + h * D + tl.arange(0, D), out_vec)


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        """
        q_nope: [B, H, D], bfloat16
        q_pe: [B, H, Dp], bfloat16
        ckv_cache: [num_pages, 1, D], bfloat16
        kpe_cache: [num_pages, 1, Dp], bfloat16
        kv_indptr: [B+1], int32
        kv_indices: [N], int32 (only first kv_indptr[b+1] - kv_indptr[b] indices used per b)
        sm_scale: float32 scalar
        Returns:
        output: [B, H, D] (bfloat16)
        lse: [B, H] (float32)
        """
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda, "All tensors must be on CUDA"
        B, H, D = q_nope.shape
        Bp, Hp, Dp = q_pe.shape
        assert B == Bp and H == Hp, "Batch and head dims must match for q_nope and q_pe"
        # Constants as per original assertions
        assert D == 512 and Dp == 64, "Expected head dims: D=512, Dp=64"
        num_pages = ckv_cache.shape[0]
        assert ckv_cache.shape[1] == 1 and kpe_cache.shape[1] == 1, "Expected single cache dim"
        assert kv_indptr.shape[0] == B + 1, "kv_indptr length must be B+1"

        # Cast queries to fp32 for computation
        q_nope_f = q_nope.to(torch.float32).contiguous()  # [B, H, D]
        q_pe_f = q_pe.to(torch.float32).contiguous()      # [B, H, Dp]
        Kc_all = ckv_cache.squeeze(1).to(torch.float32).contiguous()  # [num_pages, D]
        Kp_all = kpe_cache.squeeze(1).to(torch.float32).contiguous()  # [num_pages, Dp]

        output = torch.empty((B, H, D), dtype=torch.float32, device=q_nope.device)
        lse = torch.empty((B, H), dtype=torch.float32, device=q_nope.device)

        for b in range(B):
            start = int(kv_indptr[b].item())
            end = int(kv_indptr[b + 1].item())
            L_tokens = end - start
            if L_tokens <= 0:
                # No tokens for this batch element: output zeros, lse -inf
                output[b].zero_()
                lse[b].fill_(-float("inf"))
                continue

            # Prepare per-batch queries and cache slices
            qn_b = q_nope_f[b].contiguous()                # [H, D]
            qp_b = q_pe_f[b].contiguous()                 # [H, Dp]
            Kc_batch = Kc_all[start:end].contiguous()     # [L_tokens, D]
            Kp_batch = Kp_all[start:end].contiguous()     # [L_tokens, Dp]

            # 1) Compute logits_scaled buffer [H, L_tokens]
            logits_buf = torch.empty((H, L_tokens), dtype=torch.float32, device=q_nope.device)
            grid_logits = (H, L_tokens)
            compute_logits_scaled_per_batch_kernel[grid_logits](
                qn_b, qp_b, Kc_batch, Kp_batch, logits_buf.view(-1), H, D, Dp, L_tokens, sm_scale
            )

            # 2) Compute lse per head
            grid_lse = (H,)
            compute_lse_per_row_kernel[grid_lse](logits_buf.view(-1), lse[b], H, L_tokens)

            # 3) Compute output[h, :] = softmax(logits_scaled[h, :]) @ Kc[:, :]
            grid_out = (H,)
            compute_output_matvec_kernel[grid_out](
                Kc_batch, logits_buf.view(-1), output[b].view(-1), H, D, L_tokens, sm_scale
            )

        # Return output in bfloat16, lse in float32
        return output.to(torch.bfloat16), lse


def run(*args):
    return ModelNew()(*args)
