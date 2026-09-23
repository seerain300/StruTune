import math
import torch
import triton
import triton.language as tl


@triton.jit
def _compute_scaled_logits_kernel(
    qn_ptr,               # *f32, 1D vector [H]
    qp_ptr,               # *f32, 1D vector [H]
    Kc_base_ptr,          # *f32, base pointer for ckv_cache, shape [num_tokens, CK]
    Kp_base_ptr,          # *f32, base pointer for kpe_cache, shape [num_tokens, KP]
    tok_idx_ptr,          # *i32, 1D vector [L_tokens] of token indices
    logits_ptr,           # *f32, 2D buffer [H, L_tokens]
    H: tl.constexpr,      # constexpr
    CK: tl.constexpr,     # constexpr
    KP: tl.constexpr,     # constexpr
    L_tokens: tl.constexpr,  # constexpr
    sm_scale: tl.constexpr,  # constexpr (scalar float)
):
    h = tl.program_id(0)  # head index
    t = tl.program_id(1)  # token index

    # Guard against out-of-range (though grid is exact)
    if h >= H or t >= L_tokens:
        return

    # Load qn[h] and qp[h] as scalars
    # qn[h] is a scalar load
    qn_val = tl.load(qn_ptr + h)  # scalar f32
    # qp[h] is a scalar load
    qp_val = tl.load(qp_ptr + h)  # scalar f32

    # Compute address for Kc[t, :] and Kp[t, :]
    idx = tok_idx_ptr[t]  # scalar i32
    base_kc = idx * CK
    base_kp = idx * KP

    # Vectorized loads for Kc[t, :] and Kp[t, :]
    offs = tl.arange(0, CK)
    kc = tl.load(Kc_base_ptr + base_kc + offs)  # [CK] f32
    offs2 = tl.arange(0, KP)
    kp = tl.load(Kp_base_ptr + base_kp + offs2)  # [KP] f32

    # Dot products: sum_k qn[k] * Kc[t, k] and sum_k qp[k] * Kp[t, k]
    # Broadcast qn_val to [CK] and qp_val to [KP] for elementwise mul
    dot_qn = tl.sum(kc * qn_val, axis=0)  # scalar
    dot_qp = tl.sum(kp * qp_val, axis=0)  # scalar

    # Combine and scale
    scaled = sm_scale * (dot_qn + dot_qp)

    # Store scaled logits to logits_ptr[h, t]
    # logits_ptr is a contiguous 2D buffer with row-major order (stride along L_tokens)
    row_ptr = logits_ptr + h * L_tokens
    tl.store(row_ptr + t, scaled)


@triton.jit
def _lse_kernel(
    logits_ptr,           # *f32, 2D buffer [H, L_tokens]
    lse_ptr,              # *f32, 1D buffer [H]
    H: tl.constexpr,
    L_tokens: tl.constexpr,
):
    h = tl.program_id(0)
    if h >= H:
        return

    # Compute per-head logsumexp over t in [0..L_tokens-1]
    # Initialize max and sumexp
    max_val = tl.full((), -float("inf"), tl.float32)
    sumexp = tl.full((), 0.0, tl.float32)

    t = 0
    while t < L_tokens:
        ptr = logits_ptr + h * L_tokens + t
        val = tl.load(ptr)
        # Stable update
        new_max = tl.maximum(max_val, val)
        sumexp = sumexp * tl.exp(max_val - new_max) + tl.exp(val - new_max)
        max_val = new_max
        t += 1

    lse = tl.log(sumexp) + max_val
    tl.store(lse_ptr + h, lse)


@triton.jit
def _compute_output_kernel(
    qn_ptr,               # *f32, 1D [H]
    qp_ptr,               # *f32, 1D [H]
    logits_ptr,           # *f32, 2D [H, L_tokens]
    lse_ptr,              # *f32, 1D [H]
    Kc_base_ptr,          # *f32, base for ckv_cache, shape [num_tokens, CK]
    tok_idx_ptr,          # *i32, 1D [L_tokens]
    out_ptr,              # *f32, 2D [H, CK]
    H: tl.constexpr,
    CK: tl.constexpr,
    L_tokens: tl.constexpr,
):
    h = tl.program_id(0)
    if h >= H:
        return

    # Initialize output vector to zero
    offs = tl.arange(0, CK)
    out_row_ptr = out_ptr + h * CK
    tl.store(out_row_ptr + offs, 0.0)  # store zeros for all CK

    # Accumulate over tokens
    t = 0
    while t < L_tokens:
        ptr = logits_ptr + h * L_tokens + t
        scaled = tl.load(ptr)
        lse = tl.load(lse_ptr + h)
        soft = tl.exp(scaled - lse)
        idx = tok_idx_ptr[t]
        base_kc = idx * CK
        kc = tl.load(Kc_base_ptr + base_kc + offs)  # [CK]
        # out_row_ptr is [CK]
        out_row_ptr += 0  # no-op (already at base)
        # accumulate: out[h, :] += soft * kc
        # soft is scalar; kc is vector [CK]
        tl.store(out_row_ptr + offs, tl.load(out_row_ptr + offs) + soft * kc)
        t += 1


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Cast inputs to float32 for compute
        device = q_nope.device
        # Ensure tensors are on the same device
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda

        # Prepare shapes and constants
        batch_size = q_nope.shape[0]
        H = q_nope.shape[1]  # num_qo_heads
        CK = q_nope.shape[2]  # head_dim_ckv (512)
        KP = q_pe.shape[2]    # head_dim_kpe (64)

        # Allocate output (float32) and lse (float32)
        output = torch.empty((batch_size, H, CK), dtype=torch.float32, device=device)
        lse = torch.empty((batch_size, H), dtype=torch.float32, device=device)

        # Process each batch element
        for b in range(batch_size):
            # Determine number of tokens for this batch element
            L_tokens = int(kv_indptr[b + 1].item()) - int(kv_indptr[b].item())
            # Prepare token indices slice for this batch
            tok_idx = kv_indices[kv_indptr[b]: kv_indptr[b + 1]].contiguous().to(torch.int32)

            # Reshape q_nope[b] and q_pe[b] to 1D vectors (qn, qp)
            qn = q_nope[b].reshape(-1).contiguous().to(torch.float32)  # [H]
            qp = q_pe[b].reshape(-1).contiguous().to(torch.float32)    # [H]

            # Logits buffer [H, L_tokens]
            logits = torch.empty((H, L_tokens), dtype=torch.float32, device=device)

            # Launch kernel 1: compute scaled logits
            _compute_scaled_logits_kernel[(H, L_tokens)](
                qn, qp, ckv_cache, kpe_cache, tok_idx, logits,
                H=H, CK=CK, KP=KP, L_tokens=L_tokens, sm_scale=float(sm_scale)
            )

            # Launch kernel 2: compute per-head lse
            _lse_kernel[(H,)](
                logits, lse[b],
                H=H, L_tokens=L_tokens
            )

            # Launch kernel 3: compute output per head
            output[b] = torch.empty((H, CK), dtype=torch.float32, device=device)
            _compute_output_kernel[(H,)](
                qn, qp, logits, lse[b], ckv_cache, tok_idx, output[b],
                H=H, CK=CK, L_tokens=L_tokens
            )

        # Cast output to bfloat16 to match original behavior; lse stays float32
        output_bf16 = output.to(torch.bfloat16)
        return output_bf16, lse


def run(*args):
    return ModelNew()(*args)
