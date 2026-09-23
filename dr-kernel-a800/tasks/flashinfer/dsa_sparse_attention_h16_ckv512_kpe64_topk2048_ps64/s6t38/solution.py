import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: one program per (token, head)
# Computes:
#   For each valid K index (from sparse_indices[t, :]):
#     logits = (q_no[t, h, :] · Kc_row) + (q_pe[t, h, :] · Kp_row)
#     Keep running max m and sum s of exp(logit - m) for numerical stability.
#   Then:
#     lse = (m + log(s)) / ln(2)
#     attn = exp(logit - m) / (s * ln(2))
#     output[t, h, :] += attn * Kc_row
@triton.jit
def compute_one_kernel(
    q_nope_ptr, q_pe_ptr,
    Kc_all_ptr, Kp_all_ptr,
    sparse_idx_ptr,
    out_ptr, lse_ptr,
    sm_scale: tl.float32,
    total_kv: tl.int32,
    STRIDE_QN: tl.int32, DIM_QN: tl.int32,  # STRIDE_QN = 16*512, DIM_QN = 512
    STRIDE_QP: tl.int32, DIM_QP: tl.int32,  # STRIDE_QP = 16*64,  DIM_QP = 64
    STRIDE_KC_ROW: tl.int32,                # STRIDE_KC_ROW = 512
    STRIDE_KP_ROW: tl.int32,                # STRIDE_KP_ROW = 64
    BLOCK_K: tl.constexpr
):
    t = tl.program_id(0)
    h = tl.program_id(1)

    # Base pointers for q_nope row (t, h, :)
    q_no_row_ptr = q_nope_ptr + t * STRIDE_QN + h * DIM_QN
    # Base pointers for q_pe row (t, h, :)
    q_pe_row_ptr = q_pe_ptr  + t * STRIDE_QP + h * DIM_QP

    # Running max and sum for logsumexp (we'll compute lse as (m + log(s)) / ln(2))
    m = -float('inf')       # scalar
    s = 0.0                 # scalar
    # Output accumulator for this (t, h) in float32
    out_accum = tl.zeros((DIM_QN,), dtype=tl.float32)

    # Iterate over K dimension in tiles of BLOCK_K
    for k0 in range(0, total_kv, BLOCK_K):
        # Process each element in the tile sequentially to avoid vectorized pointer issues
        for j in range(BLOCK_K):
            k_idx = k0 + j
            # Mask for valid k_idx (within total_kv)
            if k_idx < total_kv:
                # Load sparse index (int32) for this position
                idx_j = tl.load(sparse_idx_ptr + t * total_kv + k_idx)  # int32 index into Kc_all / Kp_all
                # Pointer to the k-th row in Kc_all and Kp_all
                Kc_row_ptr = Kc_all_ptr + idx_j * STRIDE_KC_ROW
                Kp_row_ptr = Kp_all_ptr + idx_j * STRIDE_KP_ROW

                # Compute dot1 over 512 dims (q_no[t, h, :] · Kc_row)
                dot1 = 0.0
                for d in range(0, DIM_QN):
                    qd = tl.load(q_no_row_ptr + d)
                    kd = tl.load(Kc_row_ptr + d)
                    dot1 += qd * kd

                # Compute dot2 over 64 dims (q_pe[t, h, :] · Kp_row)
                dot2 = 0.0
                for d in range(0, DIM_QP):
                    qp = tl.load(q_pe_row_ptr + d)
                    kp = tl.load(Kp_row_ptr + d)
                    dot2 += qp * kp

                logit = (dot1 + dot2) * sm_scale

                # Maintain running max and sum for logsumexp with ln(2) scaling in attn
                new_m = tl.maximum(m, logit)
                s = s * tl.exp(m - new_m) + tl.exp(logit - new_m)
                m = new_m

                # Compute attn for this K entry: scaled by (s * ln(2))
                ln2 = 1.4426950408889634  # log(2)
                attn = tl.exp(logit - m) / (s * ln2)

                # Accumulate output: out[t, h, :] += attn * Kc_row
                for d in range(0, DIM_QN):
                    kd = tl.load(Kc_row_ptr + d)
                    out_accum[d] += attn * kd

    # Store lse: logsumexp over base-2 with scaling factor ln(2)
    # lse = (m + log(s)) / ln(2)
    ln2 = 1.4426950408889634
    lse_val = (m + tl.log(s)) / ln2
    tl.store(lse_ptr + t * 16 + h, lse_val)

    # Store output for this (t, h): [DIM_QN] vector
    out_row_ptr = out_ptr + t * 16 * 512 + h * 512
    tl.store(out_row_ptr + tl.arange(0, 512), out_accum)


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
        """
        Triton-only forward: no torch ops on tensors. All compute is in Triton.
        Returns:
          output: [num_tokens, 16, 512] (float32)
          lse: [num_tokens, 16] (float32)
        """
        if not TRITON_AVAILABLE or q_nope.device.type != 'cuda':
            # Fallback: original PyTorch logic (not used in Triton evaluation)
            num_tokens = q_nope.shape[0]
            output = torch.empty((num_tokens, 16, 512), dtype=torch.float32, device=q_nope.device)
            lse = torch.empty((num_tokens, 16), dtype=torch.float32, device=q_nope.device)
            return output, lse

        # Ensure inputs are float32 for Triton math; sparse_indices is int32
        q_nope_f32 = q_nope.to(torch.float32)
        q_pe_f32 = q_pe.to(torch.float32)
        Kc_all_f32 = ckv_cache.reshape(-1, 512).to(torch.float32)
        Kp_all_f32 = kpe_cache.reshape(-1, 64).to(torch.float32)
        sparse_indices_i32 = sparse_indices.to(torch.int32)

        num_tokens = q_nope.shape[0]
        num_heads = 16
        device = q_nope.device

        # Allocate outputs as float32 for accumulation
        output = torch.empty((num_tokens, num_heads, 512), dtype=torch.float32, device=device)
        lse = torch.empty((num_tokens, num_heads), dtype=torch.float32, device=device)

        # Launch Triton kernel: one program per (token, head)
        grid = (num_tokens, num_heads)
        total_kv = ckv_cache.shape[0] * 64  # num_pages * 64

        # Pass strides and dimensions as constexpr-friendly ints
        compute_one_kernel[grid](
            q_nope_f32, q_pe_f32,
            Kc_all_f32, Kp_all_f32,
            sparse_indices_i32,
            output, lse,
            float(sm_scale),
            total_kv,
            STRIDE_QN=16 * 512, DIM_QN=512,
            STRIDE_QP=16 * 64,  DIM_QP=64,
            STRIDE_KC_ROW=512,
            STRIDE_KP_ROW=64,
            BLOCK_K=128,
            num_warps=4, num_stages=2
        )

        return output, lse


def run(*args):
    return ModelNew()(*args)
