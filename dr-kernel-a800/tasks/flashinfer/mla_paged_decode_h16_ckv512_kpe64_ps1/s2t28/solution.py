import math
import torch
import triton
import triton.language as tl


@triton.jit
def _compute_out_and_lse_per_bh(
    q_nope_ptr,            # *bfloat16, flattened [B*H*D1]
    q_pe_ptr,              # *bfloat16, flattened [B*H*D2]
    ckv_cache_ptr,         # *bfloat16, flattened [N*D1]
    kpe_cache_ptr,         # *bfloat16, flattened [N*D2]
    kv_indptr,             # *int32, [B+1]
    kv_indices,            # *int32, [L_tokens]
    lse_ptr,               # *float32, flattened [B*H]
    out_ptr,               # *float32, flattened [B*H*D1]
    H: tl.constexpr,       # num heads
    D1: tl.constexpr,      # head_dim_ckv
    D2: tl.constexpr,      # head_dim_kpe
    L_tokens: tl.int32,    # runtime scalar, number of tokens for this batch element
    sm_scale: tl.float32,  # scaling factor
):
    # One Triton program per (b, h)
    b = tl.program_id(0)
    h = tl.program_id(1)

    # Base offsets
    base_qn = h * D1
    base_qp = h * D2

    # Load qn and qp for this head (vector of length D1/D2)
    qn = tl.load(q_nope_ptr + base_qn + tl.arange(0, D1)).to(tl.float32)  # [D1]
    qp = tl.load(q_pe_ptr + base_qp + tl.arange(0, D2)).to(tl.float32)   # [D2]

    # Initialize per-column lse accumulators (broadcast scalar across D1)
    token_max_vec = tl.full((D1,), -float("inf"), dtype=tl.float32)
    token_sum_vec = tl.zeros((D1,), dtype=tl.float32)

    # Output row accumulator
    out_row = tl.zeros((D1,), dtype=tl.float32)

    # Iterate over tokens exactly L_tokens times
    for t in range(0, L_tokens):
        # Compute index into kv_indptr
        # We don't need current token index explicitly; we fetch from kv_indices using t offset
        # But we need the token id for this batch: token_id = kv_indices[ kv_indptr[b] + t ]
        # However, in the original code, per-batch tokens start from kv_indptr[b] to kv_indptr[b+1]
        # So token id for this batch is kv_indices[ kv_indptr[b] + t ].
        # Load base and compute idx
        # idx = kv_indices[ kv_indptr[b] + t ]
        # Note: Triton supports scalar arithmetic and tl.load with scalar + tl.arange for 1D
        idx = tl.load(kv_indices + (tl.load(kv_indptr + b) + t))

        # Load Kc_row and Kp_row (1D vectors of length D1 and D2)
        Kc_row = tl.load(ckv_cache_ptr + idx * D1 + tl.arange(0, D1)).to(tl.float32)  # [D1]
        Kp_row = tl.load(kpe_cache_ptr + idx * D2 + tl.arange(0, D2)).to(tl.float32)  # [D2]

        # Compute logits scalar for this token
        dot1 = tl.sum(qn * Kc_row, axis=0)           # scalar
        dot2 = tl.sum(qp * Kp_row, axis=0)           # scalar
        logits_scalar = (dot1 + dot2) * sm_scale     # scalar

        # Update per-column max and sum for lse
        token_max_vec = tl.maximum(token_max_vec, logits_scalar)
        token_sum_vec += tl.exp(logits_scalar - token_max_vec)

        # Accumulate output row: out[b, h, :] += logits * Kc_row
        out_row += logits_scalar * Kc_row

    # Compute lse: logsumexp with base 2
    lse_val = tl.log(token_sum_vec) + token_max_vec
    lse_val = lse_val / math.log(2.0)  # pass sm_scale is float32; host provides 1.0

    # Store lse and output row
    # lse_ptr is flattened [B, H]
    lse_offset = b * H + h
    tl.store(lse_ptr + lse_offset, lse_val)

    # out_ptr is flattened [B*H*D1]
    out_offset = b * (H * D1) + h * D1
    tl.store(out_ptr + out_offset + tl.arange(0, D1), out_row)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure CUDA tensors
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda, "All tensors must be on CUDA"
        device = q_nope.device

        # Shapes
        B = q_nope.shape[0]
        H = q_nope.shape[1]
        D1 = q_nope.shape[2]
        D2 = q_pe.shape[2]

        # Compute L_tokens per batch element (runtime)
        # kv_indptr: [B+1], cumulative counts
        # For each batch b, L_tokens = kv_indptr[b+1] - kv_indptr[b]
        # Ensure dtype is int32 for Triton load arithmetic
        len_indptr = kv_indptr.shape[0]
        assert len_indptr == B + 1, "kv_indptr must have length B+1"
        L_tokens_list = []
        for b in range(B):
            L = tl.load(kv_indptr + b + 1) - tl.load(kv_indptr + b)
            L_tokens_list.append(int(L.item()))
        # We can use the first L_tokens to set grid; Triton handles per-program runtime loop.

        # Flatten inputs for pointer arithmetic
        q_nope_flat = q_nope.view(-1)        # [B*H*D1], bfloat16
        q_pe_flat = q_pe.view(-1)            # [B*H*D2], bfloat16
        ckv_flat = ckv_cache.view(-1)        # [N*D1], bfloat16
        kpe_flat = kpe_cache.view(-1)        # [N*D2], bfloat16

        # Output and lse buffers
        out_flat = torch.empty(B * H * D1, dtype=torch.float32, device=device)  # float32 for compute
        lse_flat = torch.empty(B * H, dtype=torch.float32, device=device)

        # Launch grid: one program per (b, h)
        grid = (B, H)
        _compute_out_and_lse_per_bh[grid](
            q_nope_flat, q_pe_flat, ckv_flat, kpe_flat,
            kv_indptr, kv_indices,
            lse_flat, out_flat,
            H=H, D1=D1, D2=D2, L_tokens=L_tokens_list[0] if len(L_tokens_list) > 0 else 0,
            sm_scale=float(sm_scale),
        )

        # Reshape and cast output to bfloat16 to match original
        output = out_flat.view(B, H, D1).to(torch.bfloat16)
        lse = lse_flat.view(B, H)

        return output, lse


def run(*args):
    return ModelNew()(*args)
