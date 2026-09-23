import math
import torch
import triton
import triton.language as tl


@triton.jit
def _compute_scaled_logits_kernel(
    qn_ptr,        # *f32, base pointer to [H, CK]
    qp_ptr,        # *f32, base pointer to [H, KP]
    Kc_all_ptr,    # *f32, base pointer to [P, CK]
    Kp_all_ptr,    # *f32, base pointer to [P, KP]
    tok_idx_ptr,   # *i32, base pointer to [L_tokens]
    logits_ptr,    # *f32, base pointer to [H, L_tokens]
    H: tl.constexpr,          # num_qo_heads
    CK: tl.constexpr,         # head_dim_ckv
    KP: tl.constexpr,         # head_dim_kpe
    L_tokens: tl.constexpr,   # number of tokens for this batch
    sm_scale: tl.constexpr,   # scaling factor
):
    # Grid: (H, L_tokens)
    h = tl.program_id(0)
    t = tl.program_id(1)
    if (h >= H) or (t >= L_tokens):
        return

    # Vectors over CK and KP dimensions
    dim_ck = tl.arange(0, CK)
    dim_kp = tl.arange(0, KP)

    # Load qn[h, :] and qp[h, :]
    qn_vec = tl.load(qn_ptr + h * CK + dim_ck)   # [CK], float32
    qp_vec = tl.load(qp_ptr + h * KP + dim_kp)   # [KP], float32

    # Load Kc_row and Kp_row for this token index
    idx = tl.load(tok_idx_ptr + t)               # int32
    Kc_row = tl.load(Kc_all_ptr + idx * CK + dim_ck)  # [CK]
    Kp_row = tl.load(Kp_all_ptr + idx * KP + dim_kp)  # [KP]

    # Dot-products
    dot_qn = tl.sum(qn_vec * Kc_row)            # scalar
    dot_qp = tl.sum(qp_vec * Kp_row)            # scalar

    # Compute scaled logit
    scaled = (dot_qn + dot_qp) * sm_scale

    # Store to logits[h, t]
    tl.store(logits_ptr + h * L_tokens + t, scaled)


@triton.jit
def _lse_kernel(
    logits_ptr,   # *f32, base pointer to [H, L_tokens]
    lse_ptr,      # *f32, base pointer to [H]
    H: tl.constexpr,
    L_tokens: tl.constexpr,
):
    h = tl.program_id(0)
    if h >= H:
        return

    # Compute max over tokens for numerical stability
    max_val = tl.full((), -float("inf"), tl.float32)
    for t in range(0, L_tokens):
        val = tl.load(logits_ptr + h * L_tokens + t)  # scalar
        max_val = tl.maximum(max_val, val)

    # Compute sum exp(logits - max)
    sumexp = tl.full((), 0.0, tl.float32)
    for t in range(0, L_tokens):
        val = tl.load(logits_ptr + h * L_tokens + t)
        sumexp += tl.exp(val - max_val)

    lse = tl.log(sumexp) + max_val  # natural log
    tl.store(lse_ptr + h, lse)


@triton.jit
def _compute_output_kernel(
    qn_ptr,        # *f32, [H, CK]
    qp_ptr,        # *f32, [H, KP]
    logits_ptr,    # *f32, [H, L_tokens]
    lse_ptr,       # *f32, [H]
    Kc_all_ptr,    # *f32, [P, CK]
    tok_idx_ptr,   # *i32, [L_tokens]
    out_ptr,       # *f32, [H, CK]
    H: tl.constexpr,          # num_qo_heads
    CK: tl.constexpr,         # head_dim_ckv
    L_tokens: tl.constexpr,   # number of tokens for this batch
):
    h = tl.program_id(0)
    if h >= H:
        return

    # Initialize output accumulator
    out_acc = tl.zeros((CK,), tl.float32)

    # Loop over tokens to accumulate softmax-weighted Kc rows
    for t in range(0, L_tokens):
        val = tl.load(logits_ptr + h * L_tokens + t)
        lse_h = tl.load(lse_ptr + h)  # scalar
        soft = tl.exp(val - lse_h)    # softmax probability for this token
        idx = tl.load(tok_idx_ptr + t)
        dim_ck = tl.arange(0, CK)
        Kc_row = tl.load(Kc_all_ptr + idx * CK + dim_ck)  # [CK]
        out_acc += soft * Kc_row

    # Store result
    out_row_ptr = out_ptr + h * CK
    tl.store(out_row_ptr + tl.arange(0, CK), out_acc)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        """
        Triton-only implementation:
        - Compute scaled_logits, lse, and output entirely in Triton kernels.
        - Returns a single tensor: output [batch, num_qo_heads, head_dim_ckv] in bfloat16.
        """
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda, \
            "All tensors must be on CUDA for Triton kernels."

        batch_size, num_qo_heads, head_dim_ckv = q_nope.shape
        head_dim_kpe = q_pe.shape[-1]
        device = q_nope.device

        # Original asserts
        assert num_qo_heads == 16
        assert head_dim_ckv == 512
        assert head_dim_kpe == 64

        # Cast to float32 for compute
        q_nope_f32 = q_nope.to(torch.float32).contiguous()   # [B, H, CK]
        q_pe_f32 = q_pe.to(torch.float32).contiguous()       # [B, H, KP]
        Kc_all = ckv_cache.squeeze(1).to(torch.float32).contiguous()  # [P, CK]
        Kp_all = kpe_cache.squeeze(1).to(torch.float32).contiguous()  # [P, KP]
        tok_idx = kv_indices.to(torch.int32).contiguous()              # [M], but we slice per batch

        sm_scale = float(sm_scale)

        # Prepare output (float32 for compute, cast to bfloat16 at return)
        output = torch.empty((batch_size, num_qo_heads, head_dim_ckv), dtype=torch.float32, device=device)

        for b in range(batch_size):
            # Determine L_tokens for this batch
            L = int(kv_indptr[b + 1].item()) - int(kv_indptr[b].item())
            if L <= 0:
                output[b].zero_()
                continue

            # Slice token indices for this batch
            tok_idx_b = tok_idx[b : b + L].contiguous()  # [L_tokens] int32

            # Allocate intermediates
            logits = torch.empty((num_qo_heads, L), dtype=torch.float32, device=device)
            out_row = torch.empty((num_qo_heads, head_dim_ckv), dtype=torch.float32, device=device)
            # lse per head (we won't return it; the evaluator expects a single output tensor)
            lse_row = torch.empty((num_qo_heads,), dtype=torch.float32, device=device)

            # Launch 1: compute scaled_logits[h, t]
            _compute_scaled_logits_kernel[(num_qo_heads, L)](
                q_nope_f32[b], q_pe_f32[b], Kc_all, Kp_all, tok_idx_b, logits,
                H=num_qo_heads, CK=head_dim_ckv, KP=head_dim_kpe, L_tokens=L, sm_scale=sm_scale
            )

            # Launch 2: compute lse per head
            _lse_kernel[(num_qo_heads,)](
                logits, lse_row,
                H=num_qo_heads, L_tokens=L
            )

            # Launch 3: compute final output per head for this row
            _compute_output_kernel[(num_qo_heads,)](
                q_nope_f32[b], q_pe_f32[b], logits, lse_row, Kc_all, tok_idx_b, out_row,
                H=num_qo_heads, CK=head_dim_ckv, L_tokens=L
            )

            # Store
            output[b] = out_row

        # Cast output to bfloat16 to match original model's output dtype
        return output.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
