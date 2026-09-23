import math
import torch
import triton
import triton.language as tl


# Triton kernel: concatenate [B, T, H] and [B, I, H] into [B, T+I, H] along sequence dim.
@triton.jit
def _concat_sequences_kernel(
    e_ptr, h_ptr, o_ptr,
    B, T, I, H,
    e_stride_b, e_stride_t, e_stride_h,
    h_stride_b, h_stride_i, h_stride_h,
    o_stride_b, o_stride_l, o_stride_h,
    BLOCK_L: tl.constexpr,
):
    b = tl.program_id(0)
    # Compute base pointers for this batch
    e_base = e_ptr + b * e_stride_b
    h_base = h_ptr + b * h_stride_b
    o_base = o_ptr + b * o_stride_b

    # Loop over the concatenated sequence length in tiles of BLOCK_L
    total = T + I
    for l0 in range(0, total, BLOCK_L):
        offs = l0 + tl.arange(0, BLOCK_L)
        mask = offs < total

        # Determine how many come from encoder_hidden_states vs hidden_states
        enc_mask = offs < T
        hid_mask = offs >= T

        # Prepare H-vector of indices
        h_idx = tl.arange(0, H)

        # Load encoder rows
        e_ptrs = e_base + offs[:, None] * e_stride_t + h_idx[None, :] * e_stride_h
        enc_vals = tl.load(e_ptrs, mask=mask[:, None], other=0.0)

        # Load hidden rows
        h_ptrs = h_base + (offs - T)[:, None] * h_stride_i + h_idx[None, :] * h_stride_h
        hid_vals = tl.load(h_ptrs, mask=hid_mask[:, None], other=0.0)

        # Choose per-l: enc if enc_mask, else hid
        # Construct select mask for 2D: enc_mask[:, None] matches H dim
        vals = tl.where(enc_mask[:, None], enc_vals, hid_vals)

        # Store to output
        o_ptrs = o_base + offs[:, None] * o_stride_l + h_idx[None, :] * o_stride_h
        tl.store(o_ptrs, vals, mask=mask[:, None])


# Triton kernel: split C [B*(T+I), H] into processed_encoder [B, T, H] and processed_hidden [B, I, H]
# We map m -> (b, l) where m = b*(T+I) + l, then copy to the appropriate output.
@triton.jit
def _split_streams_kernel(
    C_ptr,
    out_e_ptr, out_h_ptr,
    M, T, I, H,
    C_stride_m, C_stride_n,
    out_e_stride_b, out_e_stride_t, out_e_stride_h,
    out_h_stride_b, out_h_stride_i, out_h_stride_h,
    BLOCK_H: tl.constexpr,
):
    # We'll use a 1D grid over batches
    b = tl.program_id(0)

    # Loop over sequence segments
    for t0 in range(0, T, BLOCK_H):
        offs_t = t0 + tl.arange(0, BLOCK_H)
        mask_t = offs_t < T
        # Base pointer for batch b in C
        C_batch = C_ptr + b * (T + I) * C_stride_m
        # Load encoder rows
        C_ptrs_e = C_batch + offs_t[:, None] * C_stride_m + tl.arange(0, H)[None, :] * C_stride_n
        C_vals_e = tl.load(C_ptrs_e, mask=mask_t[:, None], other=0.0)

        # Store to processed_encoder
        out_e_base = out_e_ptr + b * out_e_stride_b
        out_e_ptrs = out_e_base + offs_t[:, None] * out_e_stride_t + tl.arange(0, H)[None, :] * out_e_stride_h
        tl.store(out_e_ptrs, C_vals_e, mask=mask_t[:, None])

    for i0 in range(0, I, BLOCK_H):
        offs_i = i0 + tl.arange(0, BLOCK_H)
        mask_i = offs_i < I
        # Base pointer for batch b in C at image segment
        C_batch = C_ptr + b * (T + I) * C_stride_m
        # Load image rows (offset T in the flattened sequence)
        C_ptrs_h = C_batch + (offs_i[:, None] + T) * C_stride_m + tl.arange(0, H)[None, :] * C_stride_n
        C_vals_h = tl.load(C_ptrs_h, mask=mask_i[:, None], other=0.0)

        # Store to processed_hidden
        out_h_base = out_h_ptr + b * out_h_stride_b
        out_h_ptrs = out_h_base + offs_i[:, None] * out_h_stride_i + tl.arange(0, H)[None, :] * out_h_stride_h
        tl.store(out_h_ptrs, C_vals_h, mask=mask_i[:, None])


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        Triton-optimized version that:
          - Concatenates sequences in Triton
          - Performs GEMM via torch (robust and fast)
          - Splits results in Triton
        """
        assert hidden_states.dim() == 3 and encoder_hidden_states.dim() == 3, "Inputs must be [B, seq, H]"
        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        H = hidden_states.shape[2]

        # Ensure all inputs are on same device and dtype
        device = hidden_states.device
        dtype = hidden_states.dtype
        # Allocate concatenated tensor [B, T+I, H]
        out_cat = torch.empty((B, T + I, H), dtype=dtype, device=device)

        # Launch concatenation kernel: one program per batch
        _concat_sequences_kernel[(B,)](
            encoder_hidden_states, hidden_states, out_cat,
            B, T, I, H,
            *encoder_hidden_states.stride(), *hidden_states.stride(), *out_cat.stride(),
            BLOCK_L=256,
            num_warps=1,
            num_stages=1,
        )

        # GEMM via torch: A = out_cat [B, T+I, H], W = process_weight [H, H], result C [B, T+I, H]
        # Compute C = A @ W.T (no bias)
        # Note: torch.matmul handles broadcasting W.T correctly.
        C = torch.matmul(out_cat, process_weight.t())

        # Allocate outputs
        processed_encoder = torch.empty((B, T, H), dtype=dtype, device=device)
        processed_hidden = torch.empty((B, I, H), dtype=dtype, device=device)

        # Launch split kernel: one program per batch
        _split_streams_kernel[(B,)](
            C,
            processed_encoder, processed_hidden,
            B*(T + I), T, I, H,
            *C.stride(),
            *processed_encoder.stride(), *processed_hidden.stride(),
            BLOCK_H=128,
            num_warps=1,
            num_stages=1,
        )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
