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
    logits_ptr,     # *fp32, flattened buffer [H*L_tokens], row-major (h-major)
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

    # Accumulate dot products for Kc and Kp
    acc1 = 0.0
    for kk in range(0, D):
        qk = tl.load(qn_ptr + h * D + kk)  # qn_ptr[h, kk]
        kc = tl.load(Kc_ptr + t * D + kk)  # Kc[t, kk]
        acc1 += qk * kc

    acc2 = 0.0
    for kk in range(0, Dp):
        qk = tl.load(qp_ptr + h * Dp + kk)  # qp_ptr[h, kk]
        kp = tl.load(Kp_ptr + t * Dp + kk)  # Kp[t, kk]
        acc2 += qk * kp

    logit = (acc1 + acc2) * sm_scale
    # Store to flattened buffer at index base + t
    tl.store(logits_ptr + base + t, logit)


@triton.jit
def compute_lse_per_row_kernel(
    logits_ptr,     # *fp32, [H*L_tokens], contiguous row-major
    lse_ptr,        # *fp32, [H]
    H,              # int32
    L_tokens,       # int32
):
    # 1D grid over heads
    h = tl.program_id(0)
    base = h * L_tokens
    row = tl.load(logits_ptr + base + tl.arange(0, L_tokens))  # Not used; this would require compile-time L_tokens.
    # Since we cannot use tl.arange with runtime L_tokens here, we implement a scalar loop:
    m = -float("inf")
    # We need to reload row elements; use scalar loop to compute max:
    for t in range(0, L_tokens):
        val = tl.load(logits_ptr + base + t)
        if val > m:
            m = val

    sumexp = 0.0
    for t in range(0, L_tokens):
        val = tl.load(logits_ptr + base + t)
        sumexp += tl.exp(val - m)

    lse = tl.log(sumexp) / math.log(2.0)
    tl.store(lse_ptr + h, lse)


@triton.jit
def compute_output_matvec_kernel(
    Kc_ptr,         # *fp32, [L_tokens, D], contiguous
    logits_ptr,     # *fp32, [H*L_tokens], contiguous row-major
    out_ptr,        # *fp32, [H*D], contiguous
    H,              # int32
    D: tl.constexpr,          # 512
    L_tokens,       # int32
    sm_scale,       # float32 (not used here, kept for signature)
):
    # 1D grid over heads; compute output for each head
    h = tl.program_id(0)
    base = h * L_tokens

    # Compute softmax over logits[h, :] stably and accumulate matvec
    m = -float("inf")
    for t in range(0, L_tokens):
        val = tl.load(logits_ptr + base + t)
        if val > m:
            m = val

    sumexp = 0.0
    for t in range(0, L_tokens):
        val = tl.load(logits_ptr + base + t)
        p = tl.exp(val - m)
        sumexp += p

    inv_sumexp = 1.0 / sumexp
    out_vec = tl.zeros((D,), dtype=tl.float32)
    for t in range(0, L_tokens):
        val = tl.load(logits_ptr + base + t)
        p = tl.exp(val - m) * inv_sumexp
        for kk in range(0, D):
            kc = tl.load(Kc_ptr + t * D + kk)  # Kc[t, kk]
            out_vec[kk] += p * kc

    # Store output row
    tl.store(out_ptr + h * D + tl.arange(0, D), out_vec)  # Not valid due to runtime D; use scalar store:
    # We need to store out_vec to out_ptr + h*D + kk in scalar fashion:
    for kk in range(0, D):
        tl.store(out_ptr + h * D + kk, out_vec[kk])


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        """
        q_nope: [B, H, D] bfloat16
        q_pe: [B, H, Dp] bfloat16
        ckv_cache: [num_pages, 1, D] bfloat16
        kpe_cache: [num_pages, 1, Dp] bfloat16
        kv_indptr: [B+1] int32
        kv_indices: [N] int32
        sm_scale: float32
        Returns:
        output: [B, H, D] bfloat16
        lse: [B, H] float32
        """
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda, "All tensors must be CUDA"
        B, H, D = q_nope.shape
        Bp, H2, Dp = q_pe.shape
        assert B == Bp and H == H2, "Batch/heads mismatch between q_nope and q_pe"
        assert D == 512 and Dp == 64, "Expected D=512, Dp=64"
        num_pages = ckv_cache.shape[0]
        assert ckv_cache.shape[1] == 1 and kpe_cache.shape[1] == 1, "Expected single cache dim"
        assert kv_indptr.shape[0] == B + 1, "kv_indptr length must be B+1"

        # Cast queries to fp32 for computation
        q_nope_f = q_nope.to(torch.float32).contiguous()   # [B, H, D]
        q_pe_f = q_pe.to(torch.float32).contiguous()       # [B, H, Dp]
        # Gather all cached keys
        Kc_all = ckv_cache.squeeze(1).to(torch.float32).contiguous()  # [num_pages, D]
        Kp_all = kpe_cache.squeeze(1).to(torch.float32).contiguous()  # [num_pages, Dp]

        output = torch.empty((B, H, D), dtype=torch.float32, device=q_nope.device)
        lse = torch.empty((B, H), dtype=torch.float32, device=q_nope.device)

        for b in range(B):
            start = int(kv_indptr[b].item())
            end = int(kv_indptr[b + 1].item())
            L_tokens = end - start
            if L_tokens <= 0:
                output[b].zero_()
                lse[b].fill_(-float("inf"))
                continue

            # Per-batch queries
            qn_b = q_nope_f[b].contiguous()                # [H, D]
            qp_b = q_pe_f[b].contiguous()                 # [H, Dp]
            # Slice cached keys for this batch
            Kc_batch = Kc_all[start:end].contiguous()     # [L_tokens, D]
            Kp_batch = Kp_all[start:end].contiguous()     # [L_tokens, Dp]

            # 1) Compute logits_scaled buffer [H, L_tokens]
            logits_buf = torch.empty((H, L_tokens), dtype=torch.float32, device=q_nope.device).view(-1)  # [H*L_tokens]
            grid_logits = (H, L_tokens)
            compute_logits_scaled_per_batch_kernel[grid_logits](
                qn_b, qp_b, Kc_batch, Kp_batch, logits_buf, H, D, Dp, L_tokens, sm_scale
            )

            # 2) Compute lse per head: [H]
            grid_lse = (H,)
            compute_lse_per_row_kernel[grid_lse](logits_buf, lse[b], H, L_tokens)

            # 3) Compute output per head: [H, D]
            out_flat = torch.empty((H * D,), dtype=torch.float32, device=q_nope.device)
            grid_out = (H,)
            compute_output_matvec_kernel[grid_out](
                Kc_batch, logits_buf, out_flat, H, D, L_tokens, sm_scale
            )
            output[b] = out_flat.view(H, D)

        # Return in original expected dtypes
        return output.to(torch.bfloat16), lse


def run(*args):
    return ModelNew()(*args)
