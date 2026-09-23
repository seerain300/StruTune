import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: for a single (b, h), compute output[b, h, :] and lse[b, h]
@triton.jit
def _batch_elem_kernel(
    q_nope_ptr, q_pe_ptr,
    Kc_all_ptr, Kp_all_ptr,
    out_ptr, lse_ptr,
    B: tl.constexpr, H: tl.constexpr, Dc: tl.constexpr, Dp: tl.constexpr,
    L_tokens: tl.constexpr,
    sm_scale: tl.float32
):
    b = tl.program_id(0)  # batch index
    h = tl.program_id(1)  # head index

    # Load qn and qp for this head, as fp32
    # q_nope has shape [B, H, Dc]; q_ptr for this (b,h) is q_nope_ptr + b*H*stride_b + h*stride_h
    qn = tl.load(
        q_nope_ptr + b * H * Dc + h * Dc,
        mask=tl.arange(0, Dc) < Dc,
        other=0.0
    )  # [Dc], fp32
    qp = tl.load(
        q_pe_ptr + b * H * Dp + h * Dp,
        mask=tl.arange(0, Dp) < Dp,
        other=0.0
    )  # [Dp], fp32

    # Prepare arrays to store logits and final output
    # Since we need row-major access to Kc[t, :] and Kp[t, :], we will index K pointers with offsets.
    # We don't need to store logits, but we need to compute lse and attn. We'll compute directly.

    # First pass: compute lse for this head (stable)
    max_val = -1.0e30
    for t in range(L_tokens):
        sum_qn = 0.0
        sum_qp = 0.0
        # qn @ Kc[t, :]
        for i in range(Dc):
            sum_qn += qn[i] * tl.load(Kc_all_ptr + t * Dc + i)
        # qp @ Kp[t, :]
        for j in range(Dp):
            sum_qp += qp[j] * tl.load(Kp_all_ptr + t * Dp + j)
        logits_t = sum_qn + sum_qp
        max_val = tl.maximum(max_val, logits_t * sm_scale)

    sum_exp = 0.0
    for t in range(L_tokens):
        sum_qn = 0.0
        sum_qp = 0.0
        for i in range(Dc):
            sum_qn += qn[i] * tl.load(Kc_all_ptr + t * Dc + i)
        for j in range(Dp):
            sum_qp += qp[j] * tl.load(Kp_all_ptr + t * Dp + j)
        logits_t = sum_qn + sum_qp
        sum_exp += tl.exp((logits_t * sm_scale) - max_val)

    lse_val = max_val + tl.log(sum_exp)  # base-e logsumexp; we will scale later
    ln2 = 0.6931471805599453  # math.log(2.0)
    lse_val = lse_val / ln2  # base-2 logsumexp

    # Second pass: compute attn and accumulate output[h, :]
    out_vec = tl.zeros([Dc], dtype=tl.float32)
    for t in range(L_tokens):
        sum_qn = 0.0
        sum_qp = 0.0
        for i in range(Dc):
            sum_qn += qn[i] * tl.load(Kc_all_ptr + t * Dc + i)
        for j in range(Dp):
            sum_qp += qp[j] * tl.load(Kp_all_ptr + t * Dp + j)
        logits_t = sum_qn + sum_qp
        attn_t = tl.exp((logits_t * sm_scale) - lse_val) / ln2  # base-2 softmax scaling
        # out[h, :] += attn_t * Kc[t, :]
        for i in range(Dc):
            out_vec[i] += attn_t * tl.load(Kc_all_ptr + t * Dc + i)

    # Store output (bfloat16)
    out_row_ptr = out_ptr + b * H * Dc + h * Dc
    for i in range(Dc):
        tl.store(out_row_ptr + i, out_vec[i].to(tl.bfloat16))

    # Store lse (float32)
    tl.store(lse_ptr + b * H + h, lse_val)


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        B = q_nope.shape[0]
        H = q_nope.shape[1]
        Dc = q_nope.shape[2]
        Dp = q_pe.shape[2]

        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda, "All inputs must be CUDA tensors for Triton"
        # Prepare output and lse tensors
        output = torch.empty((B, H, Dc), dtype=torch.bfloat16, device=q_nope.device)
        lse = torch.empty((B, H), dtype=torch.float32, device=q_nope.device)

        # Ensure inputs are contiguous
        q_nope_c = q_nope.contiguous()
        q_pe_c = q_pe.contiguous()
        Kc_all = ckv_cache.squeeze(1).contiguous()  # [num_pages, Dc]
        Kp_all = kpe_cache.squeeze(1).contiguous()  # [num_pages, Dp]

        for b in range(B):
            # Compute number of tokens for this batch element
            L_tokens = int(kv_indptr[b + 1].item()) - int(kv_indptr[b].item())
            if L_tokens <= 0:
                # No KV for this batch element: out zeros, lse -inf
                output[b].zero_()
                lse[b] = float("-inf")
                continue

            # Gather token indices for this batch
            tok_idx = kv_indices[int(kv_indptr[b].item()): int(kv_indptr[b + 1].item())].to(torch.long)

            # Gather Kc and Kp rows for these tokens (already contiguous)
            Kc = Kc_all[tok_idx]  # [L_tokens, Dc]
            Kp = Kp_all[tok_idx]  # [L_tokens, Dp]

            # Launch Triton kernel: one program per (b, h)
            grid = (B, H)
            _batch_elem_kernel[grid](
                q_nope_c, q_pe_c,
                Kc, Kp,
                output, lse,
                B=B, H=H, Dc=Dc, Dp=Dp,
                L_tokens=L_tokens,
                sm_scale=float(sm_scale),
                num_warps=1, num_stages=1
            )

        return output, lse

# Helper to invoke Triton-only forward
def fused_operator(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6):
    return ModelNew().forward(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6)

# Original Model for reference
class Model(torch.nn.Module):
    def forward(self, *args):
        return run(*args)