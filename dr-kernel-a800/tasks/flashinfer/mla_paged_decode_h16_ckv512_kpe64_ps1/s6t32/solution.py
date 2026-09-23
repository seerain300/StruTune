import math
import torch
import triton
import triton.language as tl


@triton.jit
def per_batch_logits_lse_out_kernel(
    qn_ptr,         # *fp32, shape [H, D], contiguous
    qp_ptr,         # *fp32, shape [H, Dp], contiguous
    Kc_ptr,         # *fp32, shape [L_tokens, D], contiguous
    Kp_ptr,         # *fp32, shape [L_tokens, Dp], contiguous
    out_ptr,        # *fp32, shape [H, D], flattened [H*D]
    lse_ptr,        # *fp32, shape [H]
    H: tl.constexpr,        # compile-time constant: number of heads
    D: tl.constexpr,        # compile-time constant: 512
    Dp: tl.constexpr,       # compile-time constant: 64
    L_tokens,       # runtime int: number of tokens for this batch element
    sm_scale,       # runtime float32
):
    # We process all H and L_tokens inside this kernel; grid=(1,)
    # 1) Compute logits_scaled[h, t] for all h in 0..H-1 and t in 0..L_tokens-1
    for h in range(0, H):
        # Create a vector out[h, :] to store output and per-head lse accumulator
        # Initialize output row and max for lse
        row_base = h * D
        max_val = -float("inf")
        logits_row = torch.empty((L_tokens,), dtype=torch.float32, device=qn_ptr.device)
        for t in range(0, L_tokens):
            acc1 = 0.0
            # dot(qn[h, :], Kc[t, :])
            for kk in range(0, D):
                val = tl.load(qn_ptr + row_base + kk)  # qn[h, kk]
                kv = tl.load(Kc_ptr + t * D + kk)     # Kc[t, kk]
                acc1 += val * kv
            acc2 = 0.0
            # dot(qp[h, :], Kp[t, :])
            for kk in range(0, Dp):
                val = tl.load(qp_ptr + h * Dp + kk)   # qp[h, kk]
                kv = tl.load(Kp_ptr + t * Dp + kk)    # Kp[t, kk]
                acc2 += val * kv
            logit_scaled = (acc1 + acc2) * sm_scale
            logits_row[t] = logit_scaled
            # Update max for lse
            if logit_scaled > max_val:
                max_val = logit_scaled

        # 2) Compute lse[h] = logsumexp(logits_row) / ln(2)
        sum_exp = 0.0
        for t in range(0, L_tokens):
            sum_exp += tl.exp(logits_row[t] - max_val)
        lse_val = tl.log(sum_exp) + max_val
        lse_val = lse_val / math.log(2.0)  # divide by ln(2)
        tl.store(lse_ptr + h, lse_val)

        # 3) Compute output[h, :] = softmax(logits_row) @ Kc[:, :]
        out_row = torch.empty((D,), dtype=torch.float32, device=qn_ptr.device)
        denom = 0.0
        for t in range(0, L_tokens):
            expv = tl.exp(logits_row[t] - max_val)
            denom += expv
        for kk in range(0, D):
            acc = 0.0
            for t in range(0, L_tokens):
                expv = tl.exp(logits_row[t] - max_val)
                kv = tl.load(Kc_ptr + t * D + kk)  # Kc[t, kk]
                acc += expv * kv
            out_row[kk] = acc / denom
        # Store out[h, :]
        for kk in range(0, D):
            tl.store(out_ptr + h * D + kk, out_row[kk])


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # No parameters; everything is device tensors in forward

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Constants from the original implementation
        D = 512
        Dp = 64

        device = q_nope.device

        # Cast inputs to fp32 for Triton math
        q_nope_fp32 = q_nope.to(torch.float32).contiguous()
        q_pe_fp32 = q_pe.to(torch.float32).contiguous()

        # Flatten K caches: [num_pages, 1, D] -> [num_tokens, D]
        # Note: The original code uses ckv_cache.squeeze(1). We will do the same.
        # Ensure the tensors are contiguous and on device
        Kc_all = ckv_cache.squeeze(1).to(torch.float32).contiguous()  # [num_tokens, D]
        Kp_all = kpe_cache.squeeze(1).to(torch.float32).contiguous()  # [num_tokens, Dp]

        batch_size = q_nope_fp32.shape[0]
        num_qo_heads = q_nope_fp32.shape[1]

        # Output and lse buffers
        output = torch.empty((batch_size, num_qo_heads, D), dtype=torch.float32, device=device)
        lse = torch.empty((batch_size, num_qo_heads), dtype=torch.float32, device=device)

        # Process per batch to avoid Triton grid confusion
        for b in range(batch_size):
            # Compute token range for this batch element
            start = int(kv_indptr[b].item())
            end = int(kv_indptr[b + 1].item())
            L_tokens = end - start
            if L_tokens <= 0:
                # No KV tokens for this batch element; output zeros, lse = -inf
                output[b].zero_()
                lse[b].fill_(-float("inf"))
                continue

            # Slice indices for this batch
            tok_idx = kv_indices[start:end].to(torch.long)  # [L_tokens]
            # Slice caches: only the selected tokens are used
            Kc = Kc_all[tok_idx]  # [L_tokens, D]
            Kp = Kp_all[tok_idx]  # [L_tokens, Dp]

            # Allocate outputs per head
            out_buf = torch.empty((num_qo_heads * D), dtype=torch.float32, device=device)
            lse_buf = torch.empty((num_qo_heads), dtype=torch.float32, device=device)

            # Launch Triton kernel for this batch
            grid = (1,)
            per_batch_logits_lse_out_kernel[grid](
                q_nope_fp32[b], q_pe_fp32[b], Kc, Kp,
                out_buf, lse_buf,
                H=num_qo_heads, D=D, Dp=Dp, L_tokens=L_tokens, sm_scale=float(sm_scale),
                num_warps=1, num_stages=1
            )

            # Reshape and cast to final dtypes
            output[b] = out_buf.view(num_qo_heads, D)
            lse[b] = lse_buf

        # Return output in bfloat16 (matching original), lse in float32
        return output.to(torch.bfloat16), lse


def run(*args):
    return ModelNew()(*args)
