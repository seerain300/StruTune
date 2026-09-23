import torch
import triton
import triton.language as tl


@triton.jit
def _concatenate_sequences_kernel(
    src1_ptr, src2_ptr, dst_ptr,
    B, T, I, H,
    stride_s1_b, stride_s1_t, stride_s1_h,
    stride_s2_b, stride_s2_i, stride_s2_h,
    stride_d_b, stride_d_l, stride_d_h,
    BLOCK_L: tl.constexpr,
):
    # One program per batch
    b = tl.program_id(0)
    # Offsets along the sequence dimension
    l_offsets = tl.arange(0, BLOCK_L)
    # Loop over combined length
    for l in range(0, T + I, BLOCK_L):
        idx = l + l_offsets
        mask = idx < (T + I)

        # Determine source: first T positions from encoder_hidden, remaining I from hidden
        is_encoder = idx < T

        # Base pointers for current batch
        src1_base = b * stride_s1_b
        src2_base = b * stride_s2_b
        dst_base = b * stride_d_b

        # Compute source pointers (choose based on mask)
        src1_ptrs = src1_ptr + src1_base + is_encoder.to(tl.int64) * stride_s1_t + idx.to(tl.int64) * stride_s1_h
        src2_ptrs = src2_ptr + src2_base + (idx - T).to(tl.int64) * stride_s2_i * (1 - is_encoder.to(tl.int64).to(tl.int64)) + idx.to(tl.int64) * stride_s2_h
        # dst pointers
        dst_ptrs = dst_ptr + dst_base + idx.to(tl.int64) * stride_d_l + l_offsets.to(tl.int64) * stride_d_h

        # Masked loads: only load from the chosen source
        # Triton doesn't support vectorized where on pointers, so do two masked loads and combine
        src1_vals = tl.load(src1_ptrs, mask=is_encoder & mask, other=0.0)
        src2_vals = tl.load(src2_ptrs, mask=(~is_encoder) & mask, other=0.0)
        out_vals = src1_vals + src2_vals
        tl.store(dst_ptrs, out_vals, mask=mask)


@triton.jit
def _split_rows_kernel(
    src_ptr, dst1_ptr, dst2_ptr,
    B, T, I, H,
    stride_s_b, stride_s_l, stride_s_h,
    stride_d1_b, stride_d1_t, stride_d1_h,
    stride_d2_b, stride_d2_i, stride_d2_h,
    BLOCK_L: tl.constexpr,
):
    # One program per batch
    b = tl.program_id(0)
    # Encode l index for encoder and hidden parts
    for l in range(0, T, BLOCK_L):
        idx_t = l + tl.arange(0, BLOCK_L)
        mask_t = idx_t < T
        src_ptrs_t = src_ptr + b * stride_s_b + idx_t.to(tl.int64) * stride_s_l
        dst1_ptrs = dst1_ptr + b * stride_d1_b + idx_t.to(tl.int64) * stride_d1_t
        # H dimension is contiguous in both, so we can copy H columns directly
        # Loop over H (assuming H fits in BLOCK_H or do elementwise copy)
        # To keep it simple and robust, copy one column at a time using a loop:
        # We'll implement per-element copy with a small BLOCK_H for the inner loop.
        for j in range(0, H):
            src_col = tl.load(src_ptrs_t + j * stride_s_h, mask=mask_t, other=0.0)
            tl.store(dst1_ptrs + j * stride_d1_h, src_col, mask=mask_t)

    for l in range(0, I, BLOCK_L):
        idx_i = l + tl.arange(0, BLOCK_L)
        mask_i = idx_i < I
        src_ptrs_i = src_ptr + b * stride_s_b + (T + idx_i).to(tl.int64) * stride_s_l
        dst2_ptrs = dst2_ptr + b * stride_d2_b + idx_i.to(tl.int64) * stride_d2_i
        for j in range(0, H):
            src_col = tl.load(src_ptrs_i + j * stride_s_h, mask=mask_i, other=0.0)
            tl.store(dst2_ptrs + j * stride_d2_h, src_col, mask=mask_i)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        hidden_states: [B, I, H]
        encoder_hidden_states: [B, T, H]
        process_weight: [H, H]
        Returns: (processed_encoder: [B, T, H], processed_hidden: [B, I, H])
        """
        # Ensure inputs are on the same device and dtype
        assert hidden_states.device == encoder_hidden_states.device == process_weight.device, "All tensors must be on the same device"
        B, I, H = hidden_states.shape
        B2, T, H2 = encoder_hidden_states.shape
        assert B == B2 and H == H2, "Batch size and hidden_dim must match across inputs"
        H_w = process_weight.shape[0]
        assert process_weight.shape[1] == H_w, "process_weight must be [H, H]"
        assert hidden_states.dtype == encoder_hidden_states.dtype == process_weight.dtype, "All tensors must share dtype"

        # Allocate output for concatenation: [B, T+I, H]
        out_cat = torch.empty((B, T + I, H), dtype=hidden_states.dtype, device=hidden_states.device)

        # Launch concat kernel: one program per batch
        _concatenate_sequences_kernel[(B,)](
            encoder_hidden_states, hidden_states, out_cat,
            B, T, I, H,
            *encoder_hidden_states.stride(),
            *hidden_states.stride(),
            *out_cat.stride(),
            BLOCK_L=256,  # sequence tile; moderate value to cover typical T+I
            num_warps=1,
            num_stages=1,
        )

        # Batched GEMM: concatenated [M, H] @ [H, H] -> [M, H]
        # M = B * (T + I)
        M = B * (T + I)
        A = out_cat  # [B, T+I, H]
        W = process_weight  # [H, H]
        # Flatten A to [M, H] by viewing (row-major)
        A_view = A.view(M, H)
        # Compute processed
        processed = A_view @ W  # [M, H]

        # Reshape back to [B, T+I, H]
        processed = processed.view(B, T + I, H)

        # Allocate outputs
        processed_encoder = torch.empty((B, T, H), dtype=processed.dtype, device=processed.device)
        processed_hidden = torch.empty((B, I, H), dtype=processed.dtype, device=processed.device)

        # Launch split kernel: one program per batch
        _split_rows_kernel[(B,)](
            processed,
            processed_encoder,
            processed_hidden,
            B, T, I, H,
            *processed.stride(),
            *processed_encoder.stride(),
            *processed_hidden.stride(),
            BLOCK_L=256,
            num_warps=1,
            num_stages=1,
        )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
