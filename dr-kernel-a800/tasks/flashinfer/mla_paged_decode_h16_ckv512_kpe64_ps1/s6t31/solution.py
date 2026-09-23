import math
import torch
import triton
import triton.language as tl


@triton.jit
def compute_per_batch_logits_lse_out_kernel(
    qn_ptr,         # *fp32, shape [H, D], contiguous
    qp_ptr,         # *fp32, shape [H, Dp], contiguous
    Kc_ptr,         # *fp32, shape [L_tokens, D], contiguous
    Kp_ptr,         # *fp32, shape [L_tokens, Dp], contiguous
    out_ptr,        # *fp32, shape [H, D], contiguous (final output in fp32)
    lse_ptr,        # *fp32, shape [H], contiguous (lse per head)
    H: tl.constexpr,           # number of heads (e.g., 16)
    D: tl.constexpr,           # head_dim_ckv (e.g., 512)
    Dp: tl.constexpr,          # head_dim_kpe (e.g., 64)
    L_tokens: tl.constexpr,    # number of tokens for this batch element
    b,                      # batch id (runtime scalar, used for output indexing)
    sm_scale,               # scaling factor (float32)
):
    # For each head h in [0, H), compute:
    # - logits_scaled[h, t] for t in [0, L_tokens)
    # - lse[h] = logsumexp(logits_scaled[h, :]) / ln(2)
    # - out[h, :] = sum_t exp(logits_scaled[h, t]) * Kc[t, :]
    h = 0
    while h < H:
        # First pass: compute max for numerical stability
        m = -float("inf")
        t = 0
        while t < L_tokens:
            acc = 0.0
            kk = 0
            while kk < D:
                val_q = tl.load(qn_ptr + h * D + kk)
                val_k = tl.load(Kc_ptr + t * D + kk)
                acc += val_q * val_k
                kk += 1
            kk = 0
            while kk < Dp:
                val_q = tl.load(qp_ptr + h * Dp + kk)
                val_k = tl.load(Kp_ptr + t * Dp + kk)
                acc += val_q * val_k
                kk += 1
            logit = acc * sm_scale
            m = tl.maximum(m, logit)
            t += 1

        # Second pass: compute sum_exp
        sum_exp = 0.0
        t = 0
        while t < L_tokens:
            acc = 0.0
            kk = 0
            while kk < D:
                val_q = tl.load(qn_ptr + h * D + kk)
                val_k = tl.load(Kc_ptr + t * D + kk)
                acc += val_q * val_k
                kk += 1
            kk = 0
            while kk < Dp:
                val_q = tl.load(qp_ptr + h * Dp + kk)
                val_k = tl.load(Kp_ptr + t * Dp + kk)
                acc += val_q * val_k
                kk += 1
            logit = acc * sm_scale
            sum_exp += tl.exp(logit - m)
            t += 1

        lse_val = m + tl.log(sum_exp) / 0.6931471805599453  # 1 / ln(2)
        tl.store(lse_ptr + h, lse_val)  # store per head lse

        # Compute output[h, :] = sum_t exp(logits_scaled[h, t] - m) * Kc[t, :]
        kk = 0
        while kk < D:
            out_row = 0.0
            t = 0
            while t < L_tokens:
                acc = 0.0
                kk_q = 0
                while kk_q < D:
                    val_q = tl.load(qn_ptr + h * D + kk_q)
                    val_k = tl.load(Kc_ptr + t * D + kk_q)
                    acc += val_q * val_k
                    kk_q += 1
                kk_q = 0
                while kk_q < Dp:
                    val_q = tl.load(qp_ptr + h * Dp + kk_q)
                    val_k = tl.load(Kp_ptr + t * Dp + kk_q)
                    acc += val_q * val_k
                    kk_q += 1
                logit = acc * sm_scale
                out_row += tl.exp(logit - m) * tl.load(Kc_ptr + t * D + kk)
                t += 1
            tl.store(out_ptr + h * D + kk, out_row)
            kk += 1

        h += 1


# Host-side forward: per-batch Triton launch to ensure exact semantics
class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure device and dtypes
        device = q_nope.device
        B = q_nope.shape[0]
        H = q_nope.shape[1]
        D = q_nope.shape[2]  # 512
        Dp = q_pe.shape[2]   # 64

        # Extract cache and convert to fp32 contiguous
        Kc_all = ckv_cache.squeeze(1).to(torch.float32).contiguous()  # [num_pages, D]
        Kp_all = kpe_cache.squeeze(1).to(torch.float32).contiguous()  # [num_pages, Dp]

        # Output and lse buffers
        output = torch.empty((B, H, D), dtype=torch.bfloat16, device=device)
        lse = torch.empty((B, H), dtype=torch.float32, device=device)

        # Ensure kv_indptr, kv_indices are on device and int32
        kv_indptr = kv_indptr.to(device=device, dtype=torch.int32)
        kv_indices = kv_indices.to(device=device, dtype=torch.int32)

        for b in range(B):
            # Determine token range for this batch element
            L_tokens = int(kv_indptr[b + 1].item()) - int(kv_indptr[b].item())
            if L_tokens <= 0:
                # No KV entries for this batch element; output zeros, lse = -inf
                output[b].zero_()
                lse[b].fill_(-float("inf"))
                continue

            tok_idx = kv_indices[int(kv_indptr[b].item()): int(kv_indptr[b + 1].item())].to(torch.int32)

            # Slice Kc and Kp for these tokens
            Kc = Kc_all[tok_idx]        # [L_tokens, D], fp32
            Kp = Kp_all[tok_idx]        # [L_tokens, Dp], fp32

            # Ensure q_nope and q_pe are fp32 on device, per-batch
            qn = q_nope[b].to(torch.float32).contiguous()  # [H, D]
            qp = q_pe[b].to(torch.float32).contiguous()    # [H, Dp]

            # Allocate per-head output and lse
            out_buf = torch.empty((H * D), dtype=torch.float32, device=device)  # flattened [H*D]
            lse_buf = torch.empty((H), dtype=torch.float32, device=device)

            # Launch Triton kernel for this batch
            grid = (1,)  # process all heads and tokens inside the kernel
            compute_per_batch_logits_lse_out_kernel[grid](
                qn, qp, Kc, Kp,
                out_buf, lse_buf,
                H=H, D=D, Dp=Dp, L_tokens=L_tokens, b=b, sm_scale=float(sm_scale),
                num_warps=1, num_stages=1
            )

            # Store outputs
            output[b] = out_buf.view(H, D).to(torch.bfloat16)
            lse[b] = lse_buf

        return output, lse


def run(*args):
    return ModelNew()(*args)
