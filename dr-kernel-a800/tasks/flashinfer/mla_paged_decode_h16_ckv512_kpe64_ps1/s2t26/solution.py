import math
import torch
import triton
import triton.language as tl


# Triton kernel: per-batch element b and per-head h, compute output vector and lse across tokens.
# Assumptions:
# - q_nope_ptr points to a flattened tensor of shape [B*H*D1], dtype bfloat16
# - q_pe_ptr   points to a flattened tensor of shape [B*H*D2], dtype bfloat16
# - ckv_cache_ptr points to a flattened tensor of shape [N*D1], dtype bfloat16
# - kpe_cache_ptr points to a flattened tensor of shape [N*D2], dtype bfloat16
# - kv_indptr: [B+1], int32, cumulative counts per batch
# - kv_indices: [L_tokens], int32, token indices per batch
# - out_ptr: [B*H*D1], float32, to store accumulated output
# - lse_ptr: [B*H], float32, to store lse per (b, h)
@triton.jit
def _compute_out_and_lse_kernel(
    q_nope_ptr,            # *bfloat16
    q_pe_ptr,              # *bfloat16
    ckv_cache_ptr,         # *bfloat16
    kpe_cache_ptr,         # *bfloat16
    kv_indptr,             # *int32
    kv_indices,            # *int32
    lse_ptr,               # *float32 (B*H)
    out_ptr,               # *float32 (B*H*D1)
    B: tl.constexpr,       # batch size (constexpr for grid)
    H: tl.constexpr,       # num heads
    D1: tl.constexpr,      # head_dim_ckv
    D2: tl.constexpr,      # head_dim_kpe
    sm_scale: tl.constexpr,# scaling factor
    MAX_T: tl.constexpr,   # maximum tokens per batch element (constexpr)
):
    # One program per batch element b
    b = tl.program_id(0)

    # Compute L_tokens for this batch element (runtime, but used in masks)
    len_indptr = tl.load(kv_indptr + b + 1)  # b+1 exists due to len_indptr shape [B+1]
    L_tokens = len_indptr - tl.load(kv_indptr + b)

    # Precompute base for kv_indices
    page_beg = tl.load(kv_indptr + b)
    # We will iterate over heads h
    for h in tl.static_range(0, H):
        # Vector token_max and sum for lse (per-column)
        token_max_vec = tl.full((D1,), -float("inf"), dtype=tl.float32)
        token_sum_vec = tl.zeros((D1,), dtype=tl.float32)

        # Load qn and qp for this head (cast to float32)
        # q_nope_ptr is flattened: index = b*H*D1 + h*D1 + d
        qn = tl.load(q_nope_ptr + b * H * D1 + h * D1 + tl.arange(0, D1)).to(tl.float32)  # [D1]
        qp = tl.load(q_pe_ptr   + b * H * D2 + h * D2 + tl.arange(0, D2)).to(tl.float32)  # [D2]

        # Iterate tokens statically up to MAX_T (masked by t < L_tokens)
        for t in tl.static_range(0, MAX_T):
            valid = t < L_tokens
            # idx = kv_indices[page_beg + t]
            idx = tl.load(kv_indices + (page_beg + t))
            # Load Kc_row and Kp_row as float32
            Kc_row = tl.load(ckv_cache_ptr + idx * D1 + tl.arange(0, D1), mask=valid, other=0.0).to(tl.float32)  # [D1]
            Kp_row = tl.load(kpe_cache_ptr + idx * D2 + tl.arange(0, D2), mask=valid, other=0.0).to(tl.float32)  # [D2]

            # Compute logits (scalar) for this token
            dot1 = tl.sum(qn * Kc_row, axis=0)
            dot2 = tl.sum(qp * Kp_row, axis=0)
            logits = (dot1 + dot2) * sm_scale  # scalar

            # Accumulate output: out[b, h, :] += logits * Kc_row
            # out_ptr is flattened: index = (b*H + h) * D1 + d
            for d in tl.static_range(0, D1):
                out_elem = tl.load(out_ptr + (b * H + h) * D1 + d, mask=valid, other=0.0) + (logits * Kc_row[d])
                tl.store(out_ptr + (b * H + h) * D1 + d, out_elem, mask=valid)

            # LSE update: per-column max and sum
            token_max_vec = tl.maximum(token_max_vec, tl.where(valid, logits, token_max_vec))
            token_sum_vec += tl.where(valid, tl.exp(logits - token_max_vec), 0.0)

        # Compute lse for this (b, h): lse = max + log(sum) / ln(2)
        # Store to lse_ptr at linear index b*H + h
        max_val = tl.max(token_max_vec)
        sum_val = tl.sum(token_sum_vec, axis=0)
        lse_val = max_val + tl.log(sum_val) / 1.4426950408889634  # 1/ln(2)
        tl.store(lse_ptr + b * H + h, lse_val)


class ModelNew(torch.nn.Module):
    def __init__(self, max_t: int = 2048):
        super().__init__()
        # MAX_T controls the static loop upper bound; keep >= max expected L_tokens
        self.max_t = max_t

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        """
        Triton-only implementation:
        - q_nope: [B, H, D1], bfloat16, device CUDA
        - q_pe:   [B, H, D2], bfloat16, device CUDA
        - ckv_cache: [N, 1, D1], bfloat16, device CUDA
        - kpe_cache: [N, 1, D2], bfloat16, device CUDA
        - kv_indptr: [B+1], int32, device CUDA
        - kv_indices: [L_tokens], int32, device CUDA
        - sm_scale: float (Python scalar)

        Returns:
        - output: [B, H, D1], bfloat16
        - lse: [B, H], float32
        """
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda, "All tensors must be on CUDA"
        device = q_nope.device
        B = q_nope.shape[0]
        H = q_nope.shape[1]
        D1 = q_nope.shape[2]
        D2 = q_pe.shape[2]
        N = ckv_cache.shape[0]

        # Ensure dtypes: original inputs are bfloat16, cast to bfloat16 pointers in Triton
        q_nope = q_nope.to(torch.bfloat16)
        q_pe   = q_pe.to(torch.bfloat16)
        ckv_cache = ckv_cache.to(torch.bfloat16)
        kpe_cache = kpe_cache.to(torch.bfloat16)

        # Make sure tensors are contiguous
        q_nope = q_nope.contiguous()
        q_pe = q_pe.contiguous()
        ckv_cache = ckv_cache.contiguous()
        kpe_cache = kpe_cache.contiguous()
        kv_indptr = kv_indptr.to(torch.int32).contiguous()
        kv_indices = kv_indices.to(torch.int32).contiguous()

        # Allocate output buffer (float32 for compute)
        out_flat = torch.empty(B * H * D1, dtype=torch.float32, device=device)
        lse = torch.empty((B, H), dtype=torch.float32, device=device)

        # Launch Triton kernel: one program per batch element
        grid = (B,)
        _compute_out_and_lse_kernel[grid](
            q_nope.view(-1),                         # [B*H*D1], bfloat16
            q_pe.view(-1),                           # [B*H*D2], bfloat16
            ckv_cache.view(-1),                     # [N*D1], bfloat16
            kpe_cache.view(-1),                     # [N*D2], bfloat16
            kv_indptr,                              # [B+1], int32
            kv_indices,                             # [L_tokens], int32
            lse,                                    # [B, H], float32
            out_flat,                               # [B*H*D1], float32
            H=H, D1=D1, D2=D2, sm_scale=float(sm_scale), MAX_T=self.max_t, B=B
        )

        # Reshape and cast output to bfloat16 to match original
        output = out_flat.view(B, H, D1).to(torch.bfloat16)
        return output, lse


def run(*args):
    return ModelNew()(*args)
