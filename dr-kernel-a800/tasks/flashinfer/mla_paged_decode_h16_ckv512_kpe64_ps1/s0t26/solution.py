import torch
import triton
import triton.language as tl


@triton.jit
def attention_kernel(
    q_nope_ptr,           # *bf16 [B, H, Dc]
    q_pe_ptr,             # *bf16 [B, H, Dp]
    ckv_cache_ptr,        # *bf16 [N, 1, Dc]
    kpe_cache_ptr,        # *bf16 [N, 1, Dp]
    kv_indptr_ptr,        # *int32 [B+1]
    kv_indices_ptr,       # *int32 [L]
    output_ptr,           # *bf16 [B, H, Dc]
    lse_ptr,              # *float32 [B*H]
    B: tl.constexpr,      # batch_size
    H: tl.constexpr,      # num_qo_heads
    Dc: tl.constexpr,     # head_dim_ckv
    Dp: tl.constexpr,     # head_dim_kpe
    SM_SCALE: tl.constexpr,  # scaling factor for logits
):
    # One program per batch element
    b = tl.program_id(0)

    # Load token range for this batch element
    base = tl.load(kv_indptr_ptr + b)          # int32 start index in kv_indices
    end = tl.load(kv_indptr_ptr + b + 1)       # int32 end index in kv_indices
    L_tokens = end - base                       # number of tokens in this segment

    # Process each head h
    for h in range(H):
        # Load q vectors for head h and convert to float32
        qn_vec = tl.load(q_nope_ptr + b * H * Dc + h * Dc + tl.arange(0, Dc), mask=tl.arange(0, Dc) < Dc, other=0.0).to(tl.float32)
        qp_vec = tl.load(q_pe_ptr + b * H * Dp + h * Dp + tl.arange(0, Dp), mask=tl.arange(0, Dp) < Dp, other=0.0).to(tl.float32)

        # Streaming logsumexp over tokens to compute lse (base-2): lse = log(sum_exp) + max_val, then divide by ln(2)
        max_val = tl.full((), -float("inf"), tl.float32)
        sum_exp = tl.full((), 0.0, tl.float32)

        # Loop over tokens i
        # Note: Triton allows Python range with a scalar; this pattern is supported.
        for i in range(L_tokens):
            idx = tl.load(kv_indices_ptr + base + i)  # int32
            # Load Kc_row [Dc] and Kp_row [Dp] for this token
            Kc_row = tl.load(ckv_cache_ptr + idx * Dc + tl.arange(0, Dc), mask=tl.arange(0, Dc) < Dc, other=0.0).to(tl.float32)
            Kp_row = tl.load(kpe_cache_ptr + idx * Dp + tl.arange(0, Dp), mask=tl.arange(0, Dp) < Dp, other=0.0).to(tl.float32)

            # Compute dot products for this head
            dot1 = tl.sum(qn_vec * Kc_row, axis=0)
            dot2 = tl.sum(qp_vec * Kp_row, axis=0)
            val = (dot1 + dot2) * SM_SCALE  # logits_scaled[h, i] * SM_SCALE (SM_SCALE acts as inverse ln(2))

            # Streaming logsumexp update in natural log:
            # If val > max_val: sum_exp = sum_exp * exp(max_val - val) + 1; max_val = val
            # Else: sum_exp += exp(val - max_val)
            # We will compute lse as log(sum_exp) + max_val and then divide by ln(2) for base-2.
            # Note: If L_tokens == 0, this loop won't run, and we'll skip softmax by default below.
            if val > max_val:
                sum_exp = sum_exp * tl.exp(max_val - val) + 1.0
                max_val = val
            else:
                sum_exp += tl.exp(val - max_val)

        # Compute lse in natural log space, then convert to base-2 by multiplying SM_SCALE (since SM_SCALE = 1/ln(2))
        # But we must divide by ln(2); SM_SCALE is actually the scaling factor applied to logits before logsumexp in the PyTorch code,
        # and the PyTorch code divides the logsumexp result by ln(2). Here, to match, we compute:
        # lse = log(sum_exp) + max_val  (natural log), then divide by ln(2).
        # However, PyTorch uses logits_scaled = logits / ln(2), so logsumexp(logits_scaled) = log(sum(exp((logits/ln(2))))) = logsumexp(logits) / ln(2).
        # In our kernel, we need to emulate: logits_scaled = val * SM_SCALE (because PyTorch divides logits by ln(2)).
        # The correct lse computation is: lse_val = log(sum_exp) + max_val; then since we used natural log, we must divide by ln(2).
        # To simplify, since SM_SCALE is the scale factor before logsumexp, we can compute lse as log(sum_exp) + max_val, then divide by ln(2).
        # The final attention uses (val - lse) * ln(2) for exponentiation.
        # So we will compute lse_val = log(sum_exp) + max_val and store it; attention uses (val - lse_val) * ln(2) for softmax.
        # Here ln(2) is tl.log(2.0).
        lse_val = tl.log(sum_exp) + max_val  # natural log space
        lse_val = lse_val / tl.log(2.0)      # convert to base-2 logsumexp

        # Compute final output vector out[b, h, :] = sum_i attn_i * Kc_row[i, :]
        out_vec = tl.zeros((Dc,), dtype=tl.float32)
        for i in range(L_tokens):
            idx = tl.load(kv_indices_ptr + base + i)  # int32
            Kc_row = tl.load(ckv_cache_ptr + idx * Dc + tl.arange(0, Dc), mask=tl.arange(0, Dc) < Dc, other=0.0).to(tl.float32)
            Kp_row = tl.load(kpe_cache_ptr + idx * Dp + tl.arange(0, Dp), mask=tl.arange(0, Dp) < Dp, other=0.0).to(tl.float32)

            dot1 = tl.sum(qn_vec * Kc_row, axis=0)
            dot2 = tl.sum(qp_vec * Kp_row, axis=0)
            val = (dot1 + dot2) * SM_SCALE

            # attn_i = exp((val - lse) * ln(2))
            attn_i = tl.exp((val - lse_val) * tl.log(2.0))
            out_vec += attn_i * Kc_row

        # Store output for head h
        out_offset = b * H * Dc + h * Dc
        tl.store(output_ptr + out_offset + tl.arange(0, Dc), out_vec)

        # Store lse per (b, h)
        lse_offset = b * H + h
        tl.store(lse_ptr + lse_offset, lse_val)


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        """
        Triton-only forward:
        - Inputs:
          * q_nope: [B, H, Dc], dtype bfloat16 (B=1, H=16, Dc=512)
          * q_pe: [B, H, Dp], dtype bfloat16 (Dp=64)
          * ckv_cache: [N, 1, Dc], dtype bfloat16
          * kpe_cache: [N, 1, Dp], dtype bfloat16
          * kv_indptr: [B+1], int32
          * kv_indices: [L], int32
          * sm_scale: float32 scalar
        - Returns:
          * output: [B, H, Dc], dtype bfloat16
          * lse: [B, H], dtype float32
        """
        # Ensure tensors are contiguous and on the same device
        device = q_nope.device
        q_nope = q_nope.contiguous()
        q_pe = q_pe.contiguous()
        ckv_cache = ckv_cache.contiguous()
        kpe_cache = kpe_cache.contiguous()
        kv_indptr = kv_indptr.contiguous()
        kv_indices = kv_indices.contiguous()

        B = q_nope.shape[0]
        H = q_nope.shape[1]
        Dc = q_nope.shape[2]
        Dp = q_pe.shape[2]

        # Allocate outputs
        output = torch.empty((B, H, Dc), dtype=torch.bfloat16, device=device)
        lse = torch.empty((B, H), dtype=torch.float32, device=device)

        # Launch Triton kernel: one program per batch element
        attention_kernel[(B,)](
            q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices,
            output, lse,
            B=B, H=H, Dc=Dc, Dp=Dp, SM_SCALE=float(sm_scale),
            num_warps=4, num_stages=2
        )

        return output, lse


def run(*args):
    return ModelNew()(*args)
