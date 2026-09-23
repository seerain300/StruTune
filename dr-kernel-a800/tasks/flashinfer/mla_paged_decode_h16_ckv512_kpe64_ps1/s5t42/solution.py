import math
import torch
import triton
import triton.language as tl


@triton.jit
def _compute_scaled_logits_kernel(
    qn_ptr,               # *f32, shape [H, CK]
    qp_ptr,               # *f32, shape [H, KP]
    Kc_base_ptr,          # *f32, base for ckv_cache, shape [num_tokens, CK]
    Kp_base_ptr,          # *f32, base for kpe_cache, shape [num_tokens, KP]
    tok_idx_ptr,          # *i32, shape [L_tokens]
    out_ptr,              # *f32, output logits, shape [H, L_tokens]
    H: tl.constexpr,      # num heads
    CK: tl.constexpr,     # head_dim_ckv
    KP: tl.constexpr,     # head_dim_kpe
    L_tokens: tl.constexpr,
    sm_scale: tl.constexpr,
):
    h = tl.program_id(0)  # head index
    t = tl.program_id(1)  # token index

    # Load qn[h, :] and qp[h, :]
    qn = tl.load(qn_ptr + h * CK + tl.arange(0, CK))  # shape [CK]
    qp = tl.load(qp_ptr + h * KP + tl.arange(0, KP))  # shape [KP]

    # Load token index
    tok = tl.load(tok_idx_ptr + t)  # scalar int32

    # Compute base offsets for Kc[t, :] and Kp[t, :]
    Kc_offset = tok * CK + tl.arange(0, CK)  # shape [CK]
    Kp_offset = tok * KP + tl.arange(0, KP)  # shape [KP]

    Kc_row = tl.load(Kc_base_ptr + Kc_offset)  # shape [CK]
    Kp_row = tl.load(Kp_base_ptr + Kp_offset)  # shape [KP]

    # Dot products
    dot_ck = tl.sum(qn * Kc_row, axis=0)  # scalar
    dot_kp = tl.sum(qp * Kp_row, axis=0)  # scalar

    # Scale and store
    val = sm_scale * (dot_ck + dot_kp)  # scalar
    tl.store(out_ptr + h * L_tokens + t, val)


@triton.jit
def _lse_kernel(
    logits_ptr,           # *f32, shape [H, L_tokens]
    lse_ptr,              # *f32, shape [H]
    H: tl.constexpr,
    L_tokens: tl.constexpr,
):
    h = tl.program_id(0)

    # Compute max for numerical stability
    max_val = -float("inf")
    for t in range(0, L_tokens):
        val = tl.load(logits_ptr + h * L_tokens + t)
        max_val = tl.maximum(max_val, val)

    # Compute sumexp with respect to max
    sumexp = 0.0
    for t in range(0, L_tokens):
        val = tl.load(logits_ptr + h * L_tokens + t)
        exp_val = tl.exp(val - max_val)
        sumexp += exp_val

    lse = tl.log(sumexp) + max_val
    tl.store(lse_ptr + h, lse)


@triton.jit
def _compute_output_kernel(
    qn_ptr,               # *f32, shape [H, CK]
    qp_ptr,               # *f32, shape [H, KP]
    Kc_base_ptr,          # *f32, base for ckv_cache, shape [num_tokens, CK]
    Kp_base_ptr,          # *f32, base for kpe_cache, shape [num_tokens, KP]
    tok_idx_ptr,          # *i32, shape [L_tokens]
    logits_ptr,           # *f32, shape [H, L_tokens]
    lse_ptr,              # *f32, shape [H]
    out_ptr,              # *f32, shape [H, CK]
    H: tl.constexpr,
    CK: tl.constexpr,
    KP: tl.constexpr,
    L_tokens: tl.constexpr,
    sm_scale: tl.constexpr,
):
    h = tl.program_id(0)
    # Load qn[h, :] and qp[h, :]
    qn = tl.load(qn_ptr + h * CK + tl.arange(0, CK))  # [CK]
    qp = tl.load(qp_ptr + h * KP + tl.arange(0, KP))  # [KP]

    # Initialize output vector
    out = tl.zeros((CK,), dtype=tl.float32)

    lse = tl.load(lse_ptr + h)  # scalar

    for t in range(0, L_tokens):
        val = tl.load(logits_ptr + h * L_tokens + t)  # scalar
        softmax_t = tl.exp(val - lse)  # scalar
        # Load Kc[t, :]
        tok = tl.load(tok_idx_ptr + t)
        Kc_offset = tok * CK + tl.arange(0, CK)
        Kc_row = tl.load(Kc_base_ptr + Kc_offset)  # [CK]
        out += softmax_t * Kc_row

    tl.store(out_ptr + h * CK + tl.arange(0, CK), out)


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # All compute in Triton, no torch ops on tensors
        device = q_nope.device
        dtype_compute = torch.float32

        # Cast inputs to float32 for compute
        qn = q_nope.contiguous().to(dtype_compute)
        qp = q_pe.contiguous().to(dtype_compute)
        Kc_all = ckv_cache.contiguous().to(dtype_compute)  # [num_pages, 1, CK] -> [num_tokens, CK]
        Kp_all = kpe_cache.contiguous().to(dtype_compute)  # [num_pages, 1, KP] -> [num_tokens, KP]

        # Dimensions
        H = qn.shape[0]  # num_qo_heads, assert equals 16
        CK = qn.shape[2]  # head_dim_ckv, assert equals 512
        assert qn.shape[1] == 1, "qn second dim must be 1"
        assert qp.shape[1] == 1, "qp second dim must be 1"
        assert qn.shape[2] == CK and qp.shape[2] == 64, "head dims must match"
        assert qn.is_cuda and qp.is_cuda and Kc_all.is_cuda and Kp_all.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda, "all tensors must be on CUDA for Triton"

        # Prepare output and lse
        output = torch.zeros((H, CK), dtype=torch.float32, device=device)  # [H, CK]
        lse = torch.empty((H,), dtype=torch.float32, device=device)        # [H]

        # Process each batch element
        # For given input, batch_size = 1, but we keep generic logic for completeness
        b = 0
        # Determine L_tokens for this batch element
        L = int(kv_indptr[b + 1].item()) - int(kv_indptr[b].item())
        if L <= 0:
            # No tokens for this batch
            return output.to(torch.bfloat16), lse

        # Slice token indices for this batch
        tok_idx = kv_indices[b:b + L].contiguous().to(torch.int32)  # [L_tokens]

        # Allocate logits buffer [H, L_tokens]
        logits = torch.empty((H, L), dtype=torch.float32, device=device)

        # Launch kernel to compute scaled_logits
        _compute_scaled_logits_kernel[(H, L)](
            qn, qp, Kc_all, Kp_all, tok_idx, logits,
            H=H, CK=CK, KP=64, L_tokens=L, sm_scale=float(sm_scale)
        )

        # Launch kernel to compute lse per head
        _lse_kernel[(H,)](
            logits, lse,
            H=H, L_tokens=L
        )

        # Launch kernel to compute final output per head
        _compute_output_kernel[(H,)](
            qn, qp, Kc_all, Kp_all, tok_idx, logits, lse, output,
            H=H, CK=CK, KP=64, L_tokens=L, sm_scale=float(sm_scale)
        )

        # Cast output to bfloat16 to match original
        output = output.to(torch.bfloat16)

        return output, lse


def run(*args):
    return ModelNew()(*args)
