import math
import torch
import triton
import triton.language as tl


@triton.jit
def per_batch_kernel(
    qn_ptr,         # *fp32, shape [H, D], contiguous
    qp_ptr,         # *fp32, shape [H, Dp], contiguous
    Kc_ptr,         # *fp32, shape [L_tokens, D], contiguous (L_tokens is runtime, but we loop over it)
    Kp_ptr,         # *fp32, shape [L_tokens, Dp], contiguous
    out_ptr,        # *fp32, flattened buffer [H*D], will hold logits_scaled per head in slots [h*D + t]
    lse_ptr,        # *fp32, buffer [H]
    H: tl.constexpr,        # number of heads (compile-time constant for loop)
    D: tl.constexpr,        # head dim for Kc (e.g., 512)
    Dp: tl.constexpr,       # head dim for Kp (e.g., 64)
    L_tokens,       # runtime int: number of tokens for this batch element
    b,              # runtime int: batch index (used for pointer base in case qn/qp are larger, but here qn/qp are [H,D])
    sm_scale,       # runtime float32
):
    # Initialize lse for this batch element to -inf
    for h in range(0, H):
        tl.store(lse_ptr + h, -float("inf"))

    # 1) Compute logits_scaled[h, t] for all h in 0..H-1 and t in 0..L_tokens-1
    for h in range(0, H):
        # Row base pointers for qn and qp
        qn_row_ptr = qn_ptr + h * D
        qp_row_ptr = qp_ptr + h * Dp
        # Loop over tokens
        for t in range(0, L_tokens):
            # Compute dot(qn[h, :], Kc[t, :]) and dot(qp[h, :], Kp[t, :])
            acc1 = 0.0
            acc2 = 0.0
            for kk in range(0, D):
                val_qn = tl.load(qn_row_ptr + kk)
                val_Kc = tl.load(Kc_ptr + t * D + kk)
                acc1 += val_qn * val_Kc
            for kp in range(0, Dp):
                val_qp = tl.load(qp_row_ptr + kp)
                val_Kp = tl.load(Kp_ptr + t * Dp + kp)
                acc2 += val_qp * val_Kp
            logit = (acc1 + acc2) * sm_scale
            tl.store(out_ptr + h * D + t, logit)

    # 2) Compute lse[h] = logsumexp(logits_scaled[h, :]) / ln(2)
    ln2 = 1.4426950408889634  # log(2)
    for h in range(0, H):
        row_start = h * D
        max_val = -float("inf")
        # first pass to find max
        for t in range(0, L_tokens):
            val = tl.load(out_ptr + row_start + t)
            if val > max_val:
                max_val = val
        # second pass to compute sum(exp(val - max))
        sum_exp = 0.0
        for t in range(0, L_tokens):
            val = tl.load(out_ptr + row_start + t)
            sum_exp += tl.exp(val - max_val)
        lse = tl.log(sum_exp) + max_val
        lse = lse / ln2
        tl.store(lse_ptr + h, lse)

    # 3) Compute output[h, :] = softmax(logits_scaled[h, :]) @ Kc[:, :]
    # We accumulate output in float32
    for h in range(0, H):
        row_start = h * D
        # Compute output[h, kk] for all kk in 0..D-1
        for kk in range(0, D):
            acc = 0.0
            for t in range(0, L_tokens):
                val = tl.load(out_ptr + row_start + t)  # logits_scaled[h, t]
                expv = tl.exp(val)  # softmax numerator
                kc_val = tl.load(Kc_ptr + t * D + kk)   # Kc[t, kk]
                acc += expv * kc_val
            tl.store(out_ptr + h * D + D + kk, acc)  # place output in a separate buffer starting at D beyond out_ptr


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Assertions to match original constraints
        assert q_nope.shape[1] == 16, "num_qo_heads must be 16"
        assert q_nope.shape[2] == 512, "head_dim_ckv must be 512"
        assert q_pe.shape[2] == 64, "head_dim_kpe must be 64"
        assert ckv_cache.shape[1] == 1 and kpe_cache.shape[1] == 1, "page_size must be 1"
        assert kv_indptr.shape[0] == q_nope.shape[0] + 1, "len_indptr must be batch_size + 1"
        device = q_nope.device

        # Ensure inputs are on the same device and contiguous
        q_nope = q_nope.to(device).to(torch.float32).contiguous()
        q_pe = q_pe.to(device).to(torch.float32).contiguous()
        Kc_all = ckv_cache.squeeze(1).to(device).to(torch.float32).contiguous()  # [num_pages, 512]
        Kp_all = kpe_cache.squeeze(1).to(device).to(torch.float32).contiguous()  # [num_pages, 64]
        batch_size = q_nope.shape[0]
        H = q_nope.shape[1]
        D = q_nope.shape[2]
        Dp = q_pe.shape[2]

        # Output tensors
        output = torch.empty((batch_size, H, D), dtype=torch.float32, device=device)
        lse = torch.empty((batch_size, H), dtype=torch.float32, device=device)

        # Process each batch element individually to ensure correct per-batch kv_indices slice
        for b in range(batch_size):
            # Compute number of tokens for this batch element
            # L_tokens = kv_indptr[b+1] - kv_indptr[b]
            L_tokens = int(kv_indptr[b + 1].item() - kv_indptr[b].item())
            # If no tokens, skip
            if L_tokens <= 0:
                output[b].zero_()
                lse[b].fill_(-float("inf"))
                continue

            # Slice token indices for this batch element
            tok_idx = kv_indices[kv_indptr[b]:kv_indptr[b + 1]].to(torch.int32)

            # Prepare Kc and Kp for these tokens
            Kc = Kc_all[tok_idx]  # [L_tokens, D]
            Kp = Kp_all[tok_idx]  # [L_tokens, Dp]

            # Allocate flattened buffer for logits_scaled and lse
            out_buf = torch.empty((H * D + D), dtype=torch.float32, device=device)  # first H*D for logits_scaled, next D for output
            lse_buf = torch.empty((H,), dtype=torch.float32, device=device)

            # Launch Triton kernel for this batch
            per_batch_kernel[(1,)](
                qn_ptr=q_nope[b],       # [H, D]
                qp_ptr=q_pe[b],         # [H, Dp]
                Kc_ptr=Kc,              # [L_tokens, D]
                Kp_ptr=Kp,              # [L_tokens, Dp]
                out_ptr=out_buf,        # buffer to hold logits_scaled and output
                lse_ptr=lse_buf,
                H=H, D=D, Dp=Dp, L_tokens=L_tokens, b=b, sm_scale=float(sm_scale),
                num_warps=1, num_stages=1
            )

            # Extract output[h, :] from out_buf (last D elements are output for each head)
            # Here, output is stored after H*D logits: out_buf[H*D + kk] corresponds to head h and feature kk
            # But our kernel stores output[h, kk] at out_buf[h*D + D + kk].
            # To match expected output shape (H, D), we construct output[b, h, kk] as out_buf[h*D + D + kk].
            # Convert to bfloat16 for final output
            for h in range(H):
                start = h * D + D
                output[b, h, :] = out_buf[start : start + D]

            # lse[b, h] is stored in lse_buf[h]
            lse[b] = lse_buf

        # Return output in bfloat16 and lse in float32
        return output.to(torch.bfloat16), lse


def run(*args):
    return ModelNew()(*args)
