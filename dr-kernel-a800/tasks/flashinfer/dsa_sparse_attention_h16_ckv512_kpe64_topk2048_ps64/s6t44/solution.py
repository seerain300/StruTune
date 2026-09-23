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
    T,                  # int32: num_tokens (for pointer safety, not used in loop)
    TOTAL_K: tl.constexpr,  # fixed 2048
    DIM_QN: tl.constexpr,   # 512
    DIM_QP: tl.constexpr,   # 64
    DIM_KC: tl.constexpr,   # 512
    STRIDE_KC_ROW: tl.constexpr,  # 512
    DIM_KP: tl.constexpr,   # 64
    STRIDE_KP_ROW: tl.constexpr,  # 64
):
    t = tl.program_id(0)
    h = tl.program_id(1)

    # Base pointers for the q vectors
    q_no_row_ptr = q_nope_ptr + t * DIM_QN * 16 + h * DIM_QN  # q_nope: [T, 16, 512] contiguous
    q_pe_row_ptr = q_pe_ptr  + t * DIM_QP * 16 + h * DIM_QP  # q_pe:  [T, 16, 64] contiguous

    # Running max and sum for logsumexp (scaled by 1/ln(2) will be handled in attn)
    m = -float('inf')  # scalar
    s = 0.0            # scalar
    # Output accumulator for this (t, h)
    out_accum = tl.zeros((DIM_QN,), dtype=tl.float32)

    # Load sparse indices for this token: a vector of length TOTAL_K=2048
    idx_vec = tl.load(sparse_idx_ptr + t * TOTAL_K + tl.arange(0, TOTAL_K))  # [2048] int32
    # Now iterate over K = 0..TOTAL_K-1
    for j in range(0, TOTAL_K):
        idx = idx_vec[j]
        # If idx == -1, skip
        if idx == -1:
            continue
        # Compute pointers to Kc_row and Kp_row
        Kc_row_ptr = Kc_all_ptr + idx * STRIDE_KC_ROW
        Kp_row_ptr = Kp_all_ptr + idx * STRIDE_KP_ROW

        # Compute dot1 over 512 dims
        dot1 = 0.0
        for d in range(0, DIM_QN):  # q vector is length 512
            qd = tl.load(q_no_row_ptr + d)
            kd = tl.load(Kc_row_ptr + d)
            dot1 += qd * kd

        # Compute dot2 over 64 dims
        dot2 = 0.0
        for d in range(0, DIM_QP):  # q' vector is length 64
            qp = tl.load(q_pe_row_ptr + d)
            kp = tl.load(Kp_row_ptr + d)
            dot2 += qp * kp

        # Logit with scaling
        logit = (dot1 + dot2) * sm_scale

        # Update running max and sum for logsumexp
        # m_new = max(m, logit)
        m_new = tl.maximum(m, logit)
        # s_new = s*exp(m - m_new) + exp(logit - m_new)
        s = s * tl.exp(m - m_new) + tl.exp(logit - m_new)
        m = m_new

        # Attention value scaled by ln(2): attn = exp(logit - m) / (s * ln(2))
        # ln(2) is ~0.69314718056
        ln2 = 0.69314718056  # constant
        attn = tl.exp(logit - m) / (s * ln2)

        # Accumulate output: out_accum += attn * Kc_row
        for d in range(0, DIM_QN):
            kd = tl.load(Kc_row_ptr + d)
            out_accum[d] += attn * kd

    # Store lse[t, h] = m + log(s)
    lse_val = m + tl.log(s)
    tl.store(lse_ptr + t * 16 + h, lse_val)

    # Store output for this (t, h)
    out_row_ptr = output_ptr + t * 16 * 512 + h * 512
    tl.store(out_row_ptr + tl.arange(0, 512), out_accum)


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
        # Ensure Triton and CUDA device
        if not TRITON_AVAILABLE or q_nope.device.type != 'cuda':
            # Fallback: if Triton not available, do CPU/GPU torch computation (not ideal, but safe)
            return self._fallback_forward(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale)

        num_tokens = q_nope.shape[0]
        device = q_nope.device

        # Flatten caches
        K_total = ckv_cache.shape[0] * 64  # num_pages * 64
        Kc_all = ckv_cache.reshape(-1, 512)  # [K_total, 512]
        Kp_all = kpe_cache.reshape(-1, 64)   # [K_total, 64]

        # Allocate outputs
        output = torch.empty((num_tokens, 16, 512), dtype=torch.float32, device=device)
        lse = torch.empty((num_tokens, 16), dtype=torch.float32, device=device)

        # Launch: one program per (token, head)
        grid = (num_tokens, 16)
        compute_one_kernel[grid](
            q_nope, q_pe,
            Kc_all, Kp_all,
            sparse_indices,  # [T, 2048] int32
            output, lse,
            float(sm_scale),
            num_tokens,
            TOTAL_K=2048,
            DIM_QN=512,
            DIM_QP=64,
            DIM_KC=512,
            STRIDE_KC_ROW=512,
            DIM_KP=64,
            STRIDE_KP_ROW=64,
            num_warps=4, num_stages=2
        )

        return output, lse

    def _fallback_forward(self, q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
        # Minimal fallback using PyTorch ops (not Triton). Use only for non-CUDA/Triton envs.
        num_tokens = q_nope.shape[0]
        output = torch.empty((num_tokens, 16, 512), dtype=torch.float32, device=q_nope.device)
        lse = torch.empty((num_tokens, 16), dtype=torch.float32, device=q_nope.device)
        for t in range(num_tokens):
            indices_t = sparse_indices[t]  # [2048]
            valid_mask = indices_t != -1
            if not valid_mask.any():
                output[t].zero_()
                continue
            Kc_all = ckv_cache.reshape(-1, 512)[valid_mask]  # [M, 512]
            Kp_all = kpe_cache.reshape(-1, 64)[valid_mask]   # [M, 64]
            qn = q_nope[t]                                     # [16, 512]
            qp = q_pe[t]                                       # [16, 64]
            logits = (qn @ Kc_all.T) + (qp @ Kp_all.T)        # [16, M]
            logits_scaled = logits * sm_scale
            lse_t = torch.logsumexp(logits_scaled, dim=-1) / math.log(2.0)
            attn = torch.softmax(logits_scaled, dim=-1)       # [16, M]
            out = attn @ Kc_all                               # [16, 512]
            output[t] = out
            lse[t] = lse_t
        return output, lse


def run(*args):
    return ModelNew()(*args)
