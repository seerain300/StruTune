import math
import torch

import triton
import triton.language as tl


@triton.jit
def _compute_output_and_lse_per_head_kernel(
    q_nope_rows_ptr,   # *f32, flattened [H, D1]
    q_pe_rows_ptr,     # *f32, flattened [H, D2]
    Kc_sub_ptr,        # *f32, [L_tokens, D1]
    Kp_sub_ptr,        # *f32, [L_tokens, D2]
    out_row_ptr,       # *f32, [D1]
    lse_out_ptr,       # *f32, scalar per head
    H: tl.int32,       # number of heads (runtime)
    D1: tl.constexpr,  # head_dim_ckv (512)
    D2: tl.constexpr,  # head_dim_kpe (64)
    B: tl.int32,       # batch size (runtime, used to locate indptr)
    N: tl.int32,       # number of cached tokens (unused but kept for context)
    sm_scale: tl.float32,
    MAX_T: tl.constexpr,  # compile-time upper bound on tokens
):
    # One Triton program per (b, h) where b is program_id(0) and h is a constexpr loop index
    b = tl.program_id(axis=0)
    # Load q vectors for head h
    qn = tl.load(q_nope_rows_ptr + h * D1 + tl.arange(0, D1)).to(tl.float32)  # [D1]
    qp = tl.load(q_pe_rows_ptr + h * D2 + tl.arange(0, D2)).to(tl.float32)   # [D2]

    # Compute per-column max and sum across tokens
    token_max_vec = tl.full((D1,), -float("inf"), dtype=tl.float32)
    token_sum_vec = tl.zeros((D1,), dtype=tl.float32)

    # Iterate over tokens in static range up to MAX_T; mask beyond actual L_tokens
    for t in tl.static_range(0, MAX_T):
        # Read L_tokens from runtime indptr (safe scalar read once)
        L_tokens = tl.load(kv_indptr_ptr + b + 1) - tl.load(kv_indptr_ptr + b)
        valid = t < L_tokens

        # kv_indices[page_beg + t] where page_beg = indptr[b]
        idx = tl.load(kv_indices_ptr + (tl.load(kv_indptr_ptr + b) + t), mask=valid, other=0).to(tl.int32)

        # Load Kc_row and Kp_row as 1D vectors with mask
        Kc_row = tl.load(Kc_sub_ptr + idx * D1 + tl.arange(0, D1), mask=valid, other=0.0).to(tl.float32)  # [D1]
        Kp_row = tl.load(Kp_sub_ptr + idx * D2 + tl.arange(0, D2), mask=valid, other=0.0).to(tl.float32)  # [D2]

        # Compute scalar logits for this token: (qn @ Kc_row) + (qp @ Kp_row)
        dot1 = tl.sum(qn * Kc_row, axis=0)  # scalar
        dot2 = tl.sum(qp * Kp_row, axis=0)  # scalar
        logits_scalar = (dot1 + dot2) * sm_scale

        # Update per-column max and sum (masked)
        token_max_vec = tl.maximum(token_max_vec, logits_scalar)
        token_sum_vec += tl.where(valid, tl.exp(logits_scalar - token_max_vec), 0.0)

        # Accumulate output: out[h, :] += logits_scalar * Kc_row
        # out_row_ptr points to base of output for this head; add offset d for each column
        for d in tl.static_range(0, D1):
            # If valid, add contribution; otherwise skip
            contrib = tl.where(valid, logits_scalar, 0.0) * Kc_row[d]
            tl.store(out_row_ptr + d, tl.load(out_row_ptr + d) + contrib)

    # Compute lse for this head: logsumexp(logits) / ln(2)
    lse_val = tl.sum(tl.log(token_sum_vec) + token_max_vec) * (1.0 / math.log(2.0))
    tl.store(lse_out_ptr + h, lse_val)


class ModelNew(torch.nn.Module):
    def __init__(self, max_t: int = 2048):
        super().__init__()
        self.max_t = max_t

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure tensors are on CUDA for Triton
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda, "All tensors must be on CUDA"
        device = q_nope.device

        B = q_nope.shape[0]
        H = q_nope.shape[1]
        D1 = q_nope.shape[2]
        D2 = q_pe.shape[2]
        N = ckv_cache.shape[0]

        # Convert input caches to float32 for computation
        Kc_sub = ckv_cache.to(torch.float32)  # [N, D1]
        Kp_sub = kpe_cache.to(torch.float32)  # [N, D2]

        # Flatten q_nope and q_pe to [H*D1], [H*D2]
        q_nope_rows = q_nope.view(H * D1)
        q_pe_rows = q_pe.view(H * D2)

        # Allocate output as flat 1D for all heads and cast later
        output_flat = torch.zeros(B * H * D1, dtype=torch.float32, device=device)

        # lse per head
        lse = torch.empty((B, H), dtype=torch.float32, device=device)

        # Launch Triton: one program per (b, h). We need to pass H as runtime, but Triton will treat it as runtime.
        # We also pass indptr and kv_indices as device tensors. Triton will read scalars per program.
        grid = (B * H,)
        _compute_output_and_lse_per_head_kernel[grid](
            q_nope_rows, q_pe_rows,
            Kc_sub, Kp_sub,
            output_flat, lse,
            H=H, D1=D1, D2=D2,
            B=B, N=N,
            sm_scale=float(sm_scale),
            MAX_T=self.max_t,
            kv_indptr=kv_indptr,          # [B+1] int32 tensor
            kv_indices=kv_indices,        # [num_tokens] int32 tensor
        )

        # Reshape and cast to bfloat16 to match original output dtype
        output = output_flat.view(B, H, D1).to(torch.bfloat16)

        return output, lse


def run(*args):
    return ModelNew()(*args)
