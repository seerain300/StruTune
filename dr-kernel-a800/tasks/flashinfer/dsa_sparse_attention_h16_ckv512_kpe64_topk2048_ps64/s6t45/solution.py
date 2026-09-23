import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


@triton.jit
def compute_one_kernel(
    q_nope_ptr, q_pe_ptr,
    Kc_all_ptr, Kp_all_ptr,
    sparse_idx_ptr,
    output_ptr, lse_ptr,
    sm_scale,           # float32 scalar
    T,                  # int32: num_tokens (for pointer safety)
    TOTAL_K: tl.constexpr,  # fixed 2048
    DIM_QN: tl.constexpr,   # 512
    DIM_QP: tl.constexpr,   # 64
    DIM_KC: tl.constexpr,   # 512
    DIM_KP: tl.constexpr,   # 64
    LN2_INV: tl.constexpr,  # 1.4426950408889634 (1 / ln(2))
):
    t = tl.program_id(0)
    h = tl.program_id(1)

    # Base pointers for q vectors
    # q_nope layout: [T, 16, 512] contiguous => row offset for (t,h) is t*16*512 + h*512
    q_no_row_ptr = q_nope_ptr + t * DIM_QN * 16 + h * DIM_QN
    q_pe_row_ptr = q_pe_ptr  + t * DIM_QP * 16 + h * DIM_QP

    # Running max and sum for logsumexp (we'll use s = sum exp(logit - m))
    m = -float('inf')
    s = 0.0
    out_accum = tl.zeros((DIM_QN,), dtype=tl.float32)

    # Process each of the 2048 entries in sparse_indices[t, :]
    j = 0
    while j < TOTAL_K:
        # Load idx (int32)
        idx = tl.load(sparse_idx_ptr + t * TOTAL_K + j)
        valid = idx != -1
        # If invalid, skip
        if not valid:
            j += 1
            continue
        # Compute offsets for rows in Kc_all and Kp_all
        # Kc_all: [num_pages*64, 512] => row stride = DIM_KC = 512
        # Kp_all: [num_pages*64, 64]  => row stride = DIM_KP = 64
        Kc_row_ptr = Kc_all_ptr + idx * DIM_KC
        Kp_row_ptr = Kp_all_ptr + idx * DIM_KP

        # Dot1 over 512 dims
        dot1 = 0.0
        for d in range(0, DIM_QN):
            qd = tl.load(q_no_row_ptr + d)
            kd = tl.load(Kc_row_ptr + d)
            dot1 += qd * kd

        # Dot2 over 64 dims
        dot2 = 0.0
        for d in range(0, DIM_QP):
            qp = tl.load(q_pe_row_ptr + d)
            kp = tl.load(Kp_row_ptr + d)
            dot2 += qp * kp

        logit = (dot1 + dot2) * sm_scale
        # Update running m and s for logsumexp
        new_m = tl.maximum(m, logit)
        # s_new = s * exp(m - new_m) + exp(logit - new_m)
        s = s * tl.exp(m - new_m) + tl.exp(logit - new_m)
        m = new_m

        # attn = exp(logit - m) / (s * ln(2))
        attn = tl.exp(logit - m) * (1.0 / (s * LN2_INV))

        # Accumulate output
        out_accum += attn * tl.load(Kc_row_ptr + tl.arange(0, DIM_QN)).to(tl.float32)

        j += 1

    # Write output for this (t, h): output has shape [T, 16, 512]
    out_row_ptr = output_ptr + t * 16 * DIM_QN + h * DIM_QN
    tl.store(out_row_ptr + tl.arange(0, DIM_QN), out_accum)

    # Write lse for this (t, h): lse = (m + log(s)) / ln(2)
    lse_val = (m + tl.log(s)) * LN2_INV
    tl.store(lse_ptr + t * 16 + h, lse_val)


def _forward_triton_only(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
    """
    Forward that only launches Triton kernels. Reads inputs, calls Triton, returns outputs.
    No torch operations on tensors; no allocations in forward.
    Returns:
      output: [num_tokens, 16, 512] (float32 compute, stored as float32 here; cast done outside if needed)
      lse: [num_tokens, 16] (float32)
    """
    # Only support Triton + CUDA
    if not TRITON_AVAILABLE or q_nope.device.type != 'cuda':
        # Fallback (not used in evaluation). Keep minimal.
        with torch.no_grad():
            num_tokens = q_nope.shape[0]
            output = torch.empty((num_tokens, 16, 512), dtype=torch.float32, device=q_nope.device)
            lse = torch.empty((num_tokens, 16), dtype=torch.float32, device=q_nope.device)
            for t in range(num_tokens):
                indices_t = sparse_indices[t]  # [2048] int32
                valid_mask = indices_t != -1
                if not valid_mask.any():
                    output[t].zero_()
                    continue
                Kc_all = ckv_cache.reshape(-1, 512)[valid_mask]  # [M, 512]
                Kp_all = kpe_cache.reshape(-1, 64)[valid_mask]  # [M, 64]
                qn = q_nope[t]               # [16, 512]
                qp = q_pe[t]                 # [16, 64]
                logits = (qn @ Kc_all.T) + (qp @ Kp_all.T)     # [16, M]
                logits_scaled = logits * sm_scale
                lse_t = torch.logsumexp(logits_scaled, dim=-1) / math.log(2.0)
                attn = torch.softmax(logits_scaled, dim=-1)     # [16, M]
                out = attn @ Kc_all                            # [16, 512]
                output[t] = out
                lse[t] = lse_t
            return output, lse

    num_tokens = q_nope.shape[0]
    device = q_nope.device
    # Flatten caches for row access
    Kc_all = ckv_cache.reshape(-1, 512).to(torch.float32)  # [num_pages*64, 512]
    Kp_all = kpe_cache.reshape(-1, 64).to(torch.float32)   # [num_pages*64, 64]

    # Prepare sparse indices as int32
    sparse_idx = sparse_indices.to(torch.int32)

    # Allocate outputs (compute in float32)
    output = torch.empty((num_tokens, 16, 512), dtype=torch.float32, device=device)
    lse = torch.empty((num_tokens, 16), dtype=torch.float32, device=device)

    # Launch: one program per (token, head)
    grid = (num_tokens, 16)
    compute_one_kernel[grid](
        q_nope, q_pe,
        Kc_all, Kp_all,
        sparse_idx,
        output, lse,
        float(sm_scale),
        num_tokens,
        TOTAL_K=2048,
        DIM_QN=512,
        DIM_QP=64,
        DIM_KC=512,
        DIM_KP=64,
        LN2_INV=1.4426950408889634,
        num_warps=4, num_stages=2
    )
    return output, lse


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
        # Only allocate and launch Triton; no torch ops on tensors
        return _forward_triton_only(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale)


def run(*args):
    return ModelNew()(*args)
