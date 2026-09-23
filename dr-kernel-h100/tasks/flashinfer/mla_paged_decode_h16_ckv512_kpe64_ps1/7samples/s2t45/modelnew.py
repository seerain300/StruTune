import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: compute logits (per token) and lse for a single (b, h).
# Inputs: q_nope[b, h, :], Kc[t, :], Kp[t, :], outputs: logits, lse
@triton.jit
def _compute_logits_and_lse(
    qn_ptr,    # [Dc] contiguous
    qp_ptr,    # [Dp] contiguous
    Kc_ptr,    # [L_tokens, Dc] contiguous
    Kp_ptr,    # [L_tokens, Dp] contiguous
    logits_ptr,    # [L_tokens] contiguous
    lse_ptr,        # scalar output for lse[b, h] (float32)
    B: tl.constexpr, H: tl.constexpr,
    Dc: tl.constexpr, Dp: tl.constexpr,
    L_tokens: tl.constexpr,
    sm_scale: tl.float32,
):
    # We only have one program per (b,h) launched from ModelNew, so no pid needed here.
    # Compute logits vector: qn @ Kc.T + qp @ Kp.T
    # Initialize logits vector
    logits = tl.zeros([L_tokens], dtype=tl.float32)

    # Accumulate two parts
    # sum_qn_Kc[t] = sum_i qn[i] * Kc[t, i]
    for t in range(0, L_tokens):
        sum_qn = 0.0
        for i in range(0, Dc):
            qval = tl.load(qn_ptr + i)
            kval = tl.load(Kc_ptr + t * Dc + i)
            sum_qn += qval * kval
        sum_qp = 0.0
        for j in range(0, Dp):
            qval = tl.load(qp_ptr + j)
            kval = tl.load(Kp_ptr + t * Dp + j)
            sum_qp += qval * kval
        logits[t] = sum_qn + sum_qp

    # Scale
    logits_scaled = logits * sm_scale

    # Compute lse = logsumexp(logits_scaled) / ln(2)
    m = tl.max(logits_scaled)
    sum_exp = 0.0
    for t in range(0, L_tokens):
        sum_exp += tl.exp(logits_scaled[t] - m)
    lse_val = m + tl.log(sum_exp)  # natural log
    lse_val = lse_val / 0.6931471805599453  # 1 / ln(2)

    # Store lse
    tl.store(lse_ptr, lse_val)

    # Store logits
    for t in range(0, L_tokens):
        tl.store(logits_ptr + t, logits_scaled[t])


# Triton kernel: compute attention vector from logits_scaled and lse
@triton.jit
def _compute_attention(
    logits_scaled_ptr,  # [L_tokens]
    lse_ptr,            # scalar lse[b, h] (float32)
    attn_ptr,           # [L_tokens] output attention
    L_tokens: tl.constexpr,
):
    inv_ln2 = 0.6931471805599453
    lse_val = tl.load(lse_ptr)
    for t in range(0, L_tokens):
        attn_val = tl.exp(logits_scaled_ptr[t] - lse_val) * inv_ln2
        tl.store(attn_ptr + t, attn_val)


# Triton kernel: accumulate output vector out[h, :] = sum_t attn[t] * Kc[t, :]
@triton.jit
def _accumulate_output(
    attn_ptr,         # [L_tokens]
    Kc_ptr,           # [L_tokens, Dc]
    out_vec_ptr,      # [Dc] output vector
    Dc: tl.constexpr,
    L_tokens: tl.constexpr,
):
    for i in range(0, Dc):
        sum_attn = 0.0
        for t in range(0, L_tokens):
            attn_val = tl.load(attn_ptr + t)
            kval = tl.load(Kc_ptr + t * Dc + i)
            sum_attn += attn_val * kval
        tl.store(out_vec_ptr + i, sum_attn)


def _run_triton_only(q_nope, q_pe, Kc_all, Kp_all, kv_indptr, kv_indices, sm_scale):
    """
    Triton-only forward: no torch operations for core math.
    Returns (output [B, H, Dc] bfloat16, lse [B, H] float32).
    """
    assert TRITON_AVAILABLE, "Triton not available"
    assert q_nope.is_cuda and q_pe.is_cuda and Kc_all.is_cuda and Kp_all.is_cuda, "Tensors must be CUDA for Triton"

    B = q_nope.shape[0]
    H = q_nope.shape[1]
    Dc = q_nope.shape[2]
    Dp = q_pe.shape[2]
    # We assert the fixed dims per original code
    assert H == 16
    assert Dc == 512
    assert Dp == 64

    # Prepare output and lse
    output = torch.empty((B, H, Dc), dtype=torch.bfloat16, device=q_nope.device)
    lse = torch.empty((B, H), dtype=torch.float32, device=q_nope.device)

    # Process each batch element
    for b in range(B):
        # Compute token range
        page_beg = int(kv_indptr[b].item())
        page_end = int(kv_indptr[b + 1].item())
        L_tokens = page_end - page_beg
        if L_tokens <= 0:
            output[b].zero_()
            lse[b].fill_(-float("inf"))
            continue

        # Gather indices
        tok_idx = kv_indices[page_beg:page_end].to(torch.long)
        # Gather Kc and Kp
        Kc = Kc_all[tok_idx]  # [L_tokens, Dc]
        Kp = Kp_all[tok_idx]  # [L_tokens, Dp]

        # Prepare qn and qp: shape [Dc] and [Dp] contiguous
        qn = q_nope[b].to(torch.float32).contiguous()  # [H, Dc], we take h-th row later
        # Note: Triton kernels expect 1D vectors; we'll create per-head views inside launch by slicing.
        # To avoid host-side loops over H, we compute per head by calling helper functions and loops.
        # But Triton cannot index Python b; we'll compute per head in a loop below in Python side,
        # launching kernels per head.

        # We'll loop over heads and launch kernels per (b, h). For this, we can use Python loops
        # and pass q_nope[b,h] by slicing.
        for h in range(H):
            # Slice qn and qp for this head
            qn_vec = q_nope[b, h].to(torch.float32).contiguous()  # [Dc]
            qp_vec = q_pe[b, h].to(torch.float32).contiguous()   # [Dp]

            # Allocate logits, attention, and out vector
            logits = torch.empty((L_tokens,), dtype=torch.float32, device=q_nope.device)
            # lse scalar for this (b, h)
            lse_bh = torch.empty((), dtype=torch.float32, device=q_nope.device)

            # Kernel 1: compute logits and lse
            _compute_logits_and_lse[(1,)](
                qn_vec, qp_vec,
                Kc, Kp,
                logits, lse_bh,
                B, H,
                Dc, Dp,
                L_tokens,
                sm_scale,
                num_warps=4, num_stages=2,
            )

            # Kernel 2: compute attention from logits and lse
            attn = torch.empty((L_tokens,), dtype=torch.float32, device=q_nope.device)
            _compute_attention[(1,)](
                logits, lse_bh,
                attn,
                L_tokens,
                num_warps=4, num_stages=2,
            )

            # Kernel 3: accumulate output vector for this head
            out_vec = torch.empty((Dc,), dtype=torch.float32, device=q_nope.device)
            _accumulate_output[(1,)](
                attn, Kc,
                out_vec,
                Dc, L_tokens,
                num_warps=4, num_stages=2,
            )

            # Store result
            output[b, h, :] = out_vec.to(torch.bfloat16)
            lse[b, h] = lse_bh[0]

    return output, lse


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        """
        Triton-only forward: computes output and lse using Triton kernels.
        """
        # Ensure CUDA for Triton
        if not q_nope.is_cuda or not q_pe.is_cuda:
            raise RuntimeError("ModelNew requires CUDA tensors for Triton kernels")
        # Convert caches to fp32 contiguous
        Kc_all = ckv_cache.to(torch.float32).contiguous()
        Kp_all = kpe_cache.to(torch.float32).contiguous()
        # Make q_nope and q_pe contiguous and fp32 for kernel loads
        q_nope = q_nope.to(torch.float32).contiguous()
        q_pe = q_pe.to(torch.float32).contiguous()
        kv_indptr = kv_indptr.to(torch.int32)
        kv_indices = kv_indices.to(torch.int32)
        sm_scale = float(sm_scale)  # Triton scalar

        out, lse = _run_triton_only(q_nope, q_pe, Kc_all, Kp_all, kv_indptr, kv_indices, sm_scale)
        return out, lse