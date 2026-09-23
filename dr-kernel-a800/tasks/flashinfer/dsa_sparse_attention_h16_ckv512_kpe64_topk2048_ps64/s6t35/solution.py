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
    sm_scale: tl.constexpr,
    DIM_QN: tl.constexpr,       # 512
    DIM_QP: tl.constexpr,       # 64
    DIM_KC: tl.constexpr,       # 512
    DIM_KP: tl.constexpr,       # 64
    total_kv: tl.constexpr,     # num_pages * 64
    STRIDE_QN: tl.constexpr,    # elements per row in q_nope = DIM_QN
    STRIDE_QP: tl.constexpr,    # elements per row in q_pe   = DIM_QP
    STRIDE_KC_ROW: tl.constexpr,# elements per row in Kc_all = DIM_KC
    STRIDE_KP_ROW: tl.constexpr,# elements per row in Kp_all = DIM_KP
    BLOCK_K: tl.constexpr       # tile size for K, e.g., 128
):
    t = tl.program_id(0)
    h = tl.program_id(1)

    q_no_row_ptr = q_nope_ptr + t * STRIDE_QN + h * DIM_QN
    q_pe_row_ptr = q_pe_ptr  + t * STRIDE_QP + h * DIM_QP

    m = -float('inf')  # running max for logsumexp
    s = 0.0            # running sum of exp shifted by m
    out_accum = tl.zeros((DIM_QN,), dtype=tl.float32)

    inv_ln2 = 1.4426950408889634  # 1 / ln(2)

    for k0 in range(0, total_kv, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        valid = offs_k < total_kv
        idx_vec = tl.load(sparse_idx_ptr + t * total_kv + offs_k, mask=valid, other=-1)  # int32
        active = valid & (idx_vec != -1)

        for j in range(BLOCK_K):
            if not active[j]:
                continue
            k_idx = idx_vec[j]
            Kc_row_ptr = Kc_all_ptr + k_idx * STRIDE_KC_ROW
            Kp_row_ptr = Kp_all_ptr + k_idx * STRIDE_KP_ROW

            # dot1: q_no[t,h,:] · Kc_row over 512
            dot1 = 0.0
            for d in range(0, DIM_QN):
                qd = tl.load(q_no_row_ptr + d)
                kd = tl.load(Kc_row_ptr + d)
                dot1 += qd * kd

            # dot2: q_pe[t,h,:] · Kp_row over 64
            dot2 = 0.0
            for d in range(0, DIM_QP):
                qp = tl.load(q_pe_row_ptr + d)
                kp = tl.load(Kp_row_ptr + d)
                dot2 += qp * kp

            logit = (dot1 + dot2) * sm_scale

            # update logsumexp
            new_m = tl.maximum(m, logit)
            s = s * tl.exp(m - new_m) + tl.exp(logit - new_m)
            m = new_m

            # attention value for this row
            attn = tl.exp(logit - m) / (s * inv_ln2)

            # accumulate output: out += attn * Kc_row
            for d in range(0, DIM_QN):
                kd = tl.load(Kc_row_ptr + d)
                out_accum[d] += attn * kd

    # lse = (m + log(s)) / ln(2)
    lse_val = (m + tl.log(s)) * inv_ln2
    tl.store(lse_ptr + t * 16 + h, lse_val)

    # store output row for this (t, h)
    out_row_ptr = output_ptr + t * 16 * DIM_QN + h * DIM_QN
    for d in range(0, DIM_QN):
        tl.store(out_row_ptr + d, out_accum[d])


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
        # Triton-only execution: no torch ops for compute
        if not TRITON_AVAILABLE or q_nope.device.type != "cuda":
            return None, None

        # Shapes and assertions
        num_tokens, num_qo_heads, head_dim_ckv = q_nope.shape
        head_dim_kpe = q_pe.shape[-1]
        num_pages, page_size, _ = ckv_cache.shape
        total_kv = num_pages * page_size
        assert num_qo_heads == 16
        assert head_dim_ckv == 512
        assert head_dim_kpe == 64
        assert page_size == 64
        assert sparse_indices.shape[0] == num_tokens
        assert sparse_indices.shape[1] == 2048

        # Flatten caches
        Kc_all = ckv_cache.reshape(-1, 512)  # [total_kv, 512]
        Kp_all = kpe_cache.reshape(-1, 64)   # [total_kv, 64]
        sparse_indices_i32 = sparse_indices.to(torch.int32)

        # Allocate outputs (float32 during compute)
        output = torch.empty((num_tokens, 16, 512), dtype=torch.float32, device=q_nope.device)
        lse = torch.empty((num_tokens, 16), dtype=torch.float32, device=q_nope.device)

        # Launch Triton kernel: one program per (token, head)
        grid = (num_tokens, 16)
        compute_one_kernel[grid](
            q_nope, q_pe,
            Kc_all, Kp_all,
            sparse_indices_i32,
            output, lse,
            float(sm_scale),
            512, 64, 512, 64, total_kv,
            q_nope.stride(0), q_nope.stride(1),
            q_pe.stride(0), q_pe.stride(1),
            Kc_all.stride(0), Kp_all.stride(0),
            BLOCK_K=128,
            num_warps=4, num_stages=2
        )

        # Return output in bfloat16 to match original code's dtype
        return output.to(torch.bfloat16), lse


def run(*args):
    return ModelNew()(*args)
