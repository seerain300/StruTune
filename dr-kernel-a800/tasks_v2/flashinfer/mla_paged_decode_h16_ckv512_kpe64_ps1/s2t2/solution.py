import math
import torch

import triton
import triton.language as tl


@triton.jit
def _compute_lse_per_head_kernel(
    q_nope_rows_ptr,       # *f32, shape [H, D1], contiguous
    q_pe_rows_ptr,         # *f32, shape [H, D2], contiguous
    Kc_sub_ptr,            # *f32, shape [L_tokens, D1], contiguous
    Kp_sub_ptr,            # *f32, shape [L_tokens, D2], contiguous
    lse_out_ptr,           # *f32, shape [B, H], contiguous
    H: tl.int32,           # number of heads (runtime)
    D1: tl.constexpr,      # head_dim_ckv (512), constexpr
    D2: tl.constexpr,      # head_dim_kpe (64), constexpr
    L_tokens: tl.int32,    # number of tokens in this batch element (runtime)
    sm_scale: tl.float32,  # scaling factor
):
    # One Triton program per batch element b. We loop over heads h to compute lse for each head.
    b = tl.program_id(axis=0)

    for h in range(0, H):
        # Load q vectors for this head as 1D vectors using constexpr ranges
        qn = tl.load(q_nope_rows_ptr + h * D1 + tl.arange(0, D1)).to(tl.float32)  # [D1]
        qp = tl.load(q_pe_rows_ptr + h * D2 + tl.arange(0, D2)).to(tl.float32)   # [D2]

        # Initialize per-column logits buffer
        logits_buf = tl.zeros((D1,), dtype=tl.float32)

        # Loop over tokens t; tl.arange uses constexpr bounds inside the loop for K rows
        for t in range(0, L_tokens):
            Kc_row = tl.load(Kc_sub_ptr + t * D1 + tl.arange(0, D1)).to(tl.float32)  # [D1]
            Kp_row = tl.load(Kp_sub_ptr + t * D2 + tl.arange(0, D2)).to(tl.float32)  # [D2]

            # Compute dot products: sum_i qn[i] * Kc_row[i], sum_j qp[j] * Kp_row[j]
            dot1 = tl.zeros((), dtype=tl.float32)
            dot2 = tl.zeros((), dtype=tl.float32)
            for i in range(0, D1):
                dot1 += qn[i] * Kc_row[i]
            for j in range(0, D2):
                dot2 += qp[j] * Kp_row[j]

            logits_buf += (dot1 + dot2) * sm_scale

        # Compute lse for this head: max + log(sum(exp)) and divide by ln(2)
        max_val = tl.max(logits_buf)
        sum_exp = tl.zeros((), dtype=tl.float32)
        for i in range(0, D1):
            sum_exp += tl.exp(logits_buf[i] - max_val)
        lse_h = max_val + tl.log(sum_exp)
        lse_h = lse_h / math.log(2.0)
        tl.store(lse_out_ptr + b * H + h, lse_h)


@triton.jit
def _compute_out_kernel(
    q_nope_rows_ptr,       # *f32, shape [H, D1], contiguous
    q_pe_rows_ptr,         # *f32, shape [H, D2], contiguous
    Kc_sub_ptr,            # *f32, shape [L_tokens, D1], contiguous
    Kp_sub_ptr,            # *f32, shape [L_tokens, D2], contiguous
    output_ptr,            # *bf16, shape [B, H, D1], contiguous
    H: tl.int32,
    D1: tl.constexpr,
    D2: tl.constexpr,
    L_tokens: tl.int32,
    sm_scale: tl.float32,
):
    # One Triton program per batch element b. Loop over heads h and compute output.
    b = tl.program_id(axis=0)

    for h in range(0, H):
        qn = tl.load(q_nope_rows_ptr + h * D1 + tl.arange(0, D1)).to(tl.float32)  # [D1]
        qp = tl.load(q_pe_rows_ptr + h * D2 + tl.arange(0, D2)).to(tl.float32)   # [D2]

        out_vec = tl.zeros((D1,), dtype=tl.float32)

        # First pass: compute per-token max for softmax stability
        token_max = tl.full((), -float("inf"), dtype=tl.float32)
        for t in range(0, L_tokens):
            Kc_row = tl.load(Kc_sub_ptr + t * D1 + tl.arange(0, D1)).to(tl.float32)  # [D1]
            Kp_row = tl.load(Kp_sub_ptr + t * D2 + tl.arange(0, D2)).to(tl.float32)  # [D2]

            dot1 = tl.zeros((), dtype=tl.float32)
            dot2 = tl.zeros((), dtype=tl.float32)
            for i in range(0, D1):
                dot1 += qn[i] * Kc_row[i]
            for j in range(0, D2):
                dot2 += qp[j] * Kp_row[j]

            logits_t = (dot1 + dot2) * sm_scale
            token_max = tl.maximum(token_max, logits_t)

        # Second pass: compute sum of exp and accumulate output
        token_sum = tl.zeros((), dtype=tl.float32)
        for t in range(0, L_tokens):
            Kc_row = tl.load(Kc_sub_ptr + t * D1 + tl.arange(0, D1)).to(tl.float32)  # [D1]
            Kp_row = tl.load(Kp_sub_ptr + t * D2 + tl.arange(0, D2)).to(tl.float32)  # [D2]

            dot1 = tl.zeros((), dtype=tl.float32)
            dot2 = tl.zeros((), dtype=tl.float32)
            for i in range(0, D1):
                dot1 += qn[i] * Kc_row[i]
            for j in range(0, D2):
                dot2 += qp[j] * Kp_row[j]

            logits_t = (dot1 + dot2) * sm_scale
            attn_t = tl.exp(logits_t - token_max)  # softmax contribution for this token
            out_vec += attn_t * Kc_row

        # Store output for this head
        tl.store(output_ptr + b * (H * D1) + h * D1 + tl.arange(0, D1),
                 out_vec.to(tl.bfloat16))


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure CUDA tensors
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda, \
            "All inputs must be on CUDA device for Triton."

        B, H, D1 = q_nope.shape
        _, _, D2 = q_pe.shape
        assert H == 16, "num_qo_heads must be 16."
        assert D1 == 512, "head_dim_ckv must be 512."
        assert D2 == 64, "head_dim_kpe must be 64."

        device = q_nope.device
        len_indptr = kv_indptr.shape[0]
        assert len_indptr == B + 1, "kv_indptr length must be batch_size + 1."

        # Prepare output and lse
        output = torch.empty((B, H, D1), dtype=torch.bfloat16, device=device)
        lse = torch.empty((B, H), dtype=torch.float32, device=device)

        # Grid: one program per batch element
        grid = (B,)

        for b in range(B):
            start = int(kv_indptr[b].item())
            end = int(kv_indptr[b + 1].item())
            L_tokens = max(end - start, 0)

            # Slice q_nope and q_pe rows for this batch element -> [H, D1] and [H, D2]
            qnb = q_nope[b].to(torch.float32).contiguous()  # [H, D1]
            qpb = q_pe[b].to(torch.float32).contiguous()    # [H, D2]

            # Slice Kc and Kp based on kv_indices for this batch element -> [L_tokens, D1] and [L_tokens, D2]
            if L_tokens > 0:
                idxs = kv_indices[start:start + L_tokens].to(torch.int32).contiguous()  # [L_tokens]
                Kc_sub = ckv_cache[idxs].squeeze(1).contiguous().to(torch.float32)     # [L_tokens, D1]
                Kp_sub = kpe_cache[idxs].squeeze(1).contiguous().to(torch.float32)     # [L_tokens, D2]
            else:
                Kc_sub = torch.empty((0, D1), dtype=torch.float32, device=device)
                Kp_sub = torch.empty((0, D2), dtype=torch.float32, device=device)

            # Kernel 1: compute lse per head
            _compute_lse_per_head_kernel[grid](
                qnb, qpb, Kc_sub, Kp_sub, lse,
                H=H,
                D1=D1,
                D2=D2,
                L_tokens=L_tokens,
                sm_scale=sm_scale,
            )

            # Kernel 2: compute output using softmax and Kc rows
            _compute_out_kernel[grid](
                qnb, qpb, Kc_sub, Kp_sub, output,
                H=H,
                D1=D1,
                D2=D2,
                L_tokens=L_tokens,
                sm_scale=sm_scale,
            )

        return output, lse


def run(*args):
    return ModelNew()(*args)
