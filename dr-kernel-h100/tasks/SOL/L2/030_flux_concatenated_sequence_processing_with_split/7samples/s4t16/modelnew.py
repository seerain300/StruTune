import torch
import triton
import triton.language as tl

# Simple 1D Triton kernel to copy a slice from a contiguous [B, L, H] tensor into an output [B, len_slice, H] tensor.
@triton.jit
def slice_copy_kernel(
    src_ptr,       # *f32 [B, L, H]
    dst_ptr,       # *f32 [B, len_slice, H]
    B: tl.int32,
    L: tl.int32,
    H: tl.int32,
    len_slice: tl.int32,
    # strides in elements
    src_stride_b, src_stride_l, src_stride_h,
    dst_stride_b, dst_stride_l, dst_stride_h,
    BLOCK_H: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_l = tl.program_id(1)  # output sequence index in [0, len_slice)
    # Compute H tile
    for h_off in range(0, H, BLOCK_H):
        offs_h = h_off + tl.arange(0, BLOCK_H)
        mask_h = offs_h < H
        # Compute source and destination offsets
        src_offset = pid_b * src_stride_b + pid_l * src_stride_l + offs_h * src_stride_h
        dst_offset = pid_b * dst_stride_b + pid_l * dst_stride_l + offs_h * dst_stride_h
        # Load and store
        vals = tl.load(src_ptr + src_offset, mask=mask_h, other=0.0)
        tl.store(dst_ptr + dst_offset, vals, mask=mask_h)

# Triton kernel to zero a contiguous [B, L, H] tensor slice: dst[:, :0, :] = 0
@triton.jit
def zero_slice_kernel(
    dst_ptr,       # *f32 [B, L, H]
    B: tl.int32,
    H: tl.int32,
    L: tl.int32,
    len_zero: tl.int32,  # number of tokens to zero (usually T or I)
    dst_stride_b, dst_stride_l, dst_stride_h,
    BLOCK_H: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_l = tl.program_id(1)  # l in [0, len_zero)
    for h_off in range(0, H, BLOCK_H):
        offs_h = h_off + tl.arange(0, BLOCK_H)
        mask_h = offs_h < H
        dst_offset = pid_b * dst_stride_b + pid_l * dst_stride_l + offs_h * dst_stride_h
        # Store zeros
        zeros = tl.zeros((BLOCK_H,), dtype=tl.float32)
        tl.store(dst_ptr + dst_offset, zeros, mask=mask_h)

class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version: compute concatenated matmul with PyTorch, then use Triton to slice into encoder and image streams.
        Returns (processed_encoder_hidden_states, processed_hidden_states).
        """
        # Ensure on CUDA; Triton slicing kernels require CUDA tensors
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Inputs must be on CUDA for Triton execution."

        # Step 1: Concatenate along sequence dimension (PyTorch)
        text_seq_len = encoder_hidden_states.shape[1]
        img_seq_len = hidden_states.shape[1]
        total_len = text_seq_len + img_seq_len

        # Step 2: Apply linear projection (PyTorch matmul), no bias
        # Shapes: encoder_hidden_states [B, T, H], hidden_states [B, I, H], process_weight [H, H]
        # concatenated [B, T+I, H], processed [B, T+I, H]
        concatenated = torch.cat([encoder_hidden_states, hidden_states], dim=1)  # [B, T+I, H]
        processed = torch.matmul(concatenated, process_weight.t())               # [B, T+I, H], float32 accumulation

        # Step 3: Slicing using Triton
        B = processed.shape[0]
        H = processed.shape[2]
        # Allocate outputs
        processed_encoder = torch.empty((B, text_seq_len, H), dtype=processed.dtype, device=processed.device)
        processed_hidden = torch.empty((B, img_seq_len, H), dtype=processed.dtype, device=processed.device)

        # Launch Triton copy kernel for encoder slice: positions [0:text_seq_len)
        if text_seq_len > 0:
            # Grid: (B, text_seq_len)
            grid = (B, text_seq_len)
            slice_copy_kernel[grid](
                processed, processed_encoder,
                B, total_len, H, text_seq_len,
                processed.stride(0), processed.stride(1), processed.stride(2),
                processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
                BLOCK_H=64,
                num_warps=2,
                num_stages=2,
            )

        # Launch Triton copy kernel for image slice: positions [text_seq_len:text_seq_len+img_seq_len)
        if img_seq_len > 0:
            grid = (B, img_seq_len)
            slice_copy_kernel[grid](
                processed, processed_hidden,
                B, total_len, H, img_seq_len,
                processed.stride(0), processed.stride(1), processed.stride(2),
                processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
                BLOCK_H=64,
                num_warps=2,
                num_stages=2,
            )

        # If either slice is empty, processed_encoder or processed_hidden may be empty as desired.
        return processed_encoder, processed_hidden