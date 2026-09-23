import torch
import triton
import triton.language as tl

@triton.jit
def batched_matvec_kernel(
    in_ptr,         # *f16/f32 [B, N, H] where N = T + I
    weight_ptr,     # *f16/f32 [H, H]
    out_ptr,        # *f16/f32 [B, N, H]
    B: tl.int32,
    N: tl.int32,    # total sequence length = text_seq_len + img_seq_len
    H: tl.int32,
    # strides (in elements)
    in_stride_b, in_stride_n, in_stride_h,
    weight_stride_h, weight_stride_k,   # weight is [H, H] so stride_k = 1
    out_stride_b, out_stride_n, out_stride_h,
    BLOCK_H: tl.constexpr,              # tile size along hidden dim
    BLOCK_K: tl.constexpr,              # tile size along input vector length
):
    # Each program handles one (batch, token) pair and computes the full output vector of length H
    pid_b = tl.program_id(0)   # batch index
    pid_n = tl.program_id(1)   # output token index in [0, N)

    # Accumulator for the output vector for this (b, n)
    output_vec = tl.zeros((H,), dtype=tl.float32)

    # Iterate over hidden dimension in tiles
    for h_off in range(0, H, BLOCK_H):
        offs_h = h_off + tl.arange(0, BLOCK_H)
        mask_h = offs_h < H

        # For each tile of H, accumulate contributions from K dimension
        acc = tl.zeros((BLOCK_H,), dtype=tl.float32)

        # Loop over input vector length (H) in chunks of BLOCK_K
        for k_off in range(0, H, BLOCK_K):
            offs_k = k_off + tl.arange(0, BLOCK_K)
            mask_k = offs_k < H

            # Load input vector chunk x for this token n: shape [BLOCK_K]
            # Address for in_ptr: base at (b, n, 0) + offs_k * stride_h
            # in_stride_h = 1 for contiguous last dim, but we pass general stride.
            x_chunk = tl.load(
                in_ptr + pid_b * in_stride_b + pid_n * in_stride_n + offs_k * in_stride_h,
                mask=mask_k,
                other=0.0
            ).to(tl.float32)  # accumulate in float32

            # Load weight block w of shape [BLOCK_H, BLOCK_K]
            # w is [H, H]; we want rows = offs_h, cols = offs_k
            w_block = tl.load(
                weight_ptr + offs_h[:, None] * weight_stride_h + offs_k[None, :] * weight_stride_k,
                mask=mask_h[:, None] & mask_k[None, :],
                other=0.0
            ).to(tl.float32)  # accumulate in float32

            # Accumulate: acc += sum over K of w_block * x_chunk
            # Broadcast x_chunk to [1, BLOCK_K] and multiply with [BLOCK_H, BLOCK_K]
            acc += tl.sum(w_block * x_chunk[None, :], axis=1)

        # Add this H-tile's acc to the output vector
        output_vec = output_vec + acc

    # Store the output vector to out_ptr at (b, n, :)
    tl.store(out_ptr + pid_b * out_stride_b + pid_n * out_stride_n + offs_h * out_stride_h, output_vec, mask=mask_h)


def triton_matvec_concat(encoded_hidden: torch.Tensor, image_hidden: torch.Tensor, process_weight: torch.Tensor) -> torch.Tensor:
    """
    Triton-optimized batched matrix-vector product:
    Input:
      - encoded_hidden: [B, T, H]
      - image_hidden: [B, I, H]
      - process_weight: [H, H]
    Output:
      - processed: [B, T+I, H]
    We materialize concatenation in PyTorch and then compute matvec in Triton.
    """
    assert encoded_hidden.is_cuda and image_hidden.is_cuda and process_weight.is_cuda, "Triton inputs must be CUDA tensors."
    B = encoded_hidden.shape[0]
    T = encoded_hidden.shape[1]
    I = image_hidden.shape[1]
    H = encoded_hidden.shape[2]
    assert image_hidden.shape[2] == H, "hidden_dim must match between inputs."
    assert process_weight.shape[0] == H and process_weight.shape[1] == H, "process_weight must be [H, H]."

    # Materialize concatenation along sequence dimension
    concatenated = torch.cat([encoded_hidden, image_hidden], dim=1)  # [B, T+I, H]
    # Output tensor
    out = torch.empty((B, T + I, H), device=concatenated.device, dtype=torch.float32)

    # Launch Triton kernel: grid over (batch, tokens)
    grid = (B, T + I)

    # Choose tile sizes; 128 works well on many GPUs, adjust num_warps for occupancy
    BLOCK_H = 128
    BLOCK_K = 128

    batched_matvec_kernel[grid](
        concatenated, process_weight, out,
        B, T + I, H,
        concatenated.stride(0), concatenated.stride(1), concatenated.stride(2),
        process_weight.stride(0), process_weight.stride(1),
        out.stride(0), out.stride(1), out.stride(2),
        BLOCK_H=BLOCK_H, BLOCK_K=BLOCK_K,
        num_warps=4,  # adjust as needed
    )

    # Cast output back to input dtype if desired (original returns same dtype as inputs).
    # The original code uses float32 inputs; we keep float32 here.
    return out


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version of the original run function.
        Returns (processed_encoder_hidden_states, processed_image_hidden_states).
        """
        # Ensure inputs are CUDA for Triton
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Inputs must be on CUDA for Triton execution."
        total = triton_matvec_concat(encoder_hidden_states, hidden_states, process_weight)  # [B, T+I, H]
        T = encoder_hidden_states.shape[1]
        processed_encoder = total[:, :T, :]
        processed_hidden = total[:, T:, :]
        return processed_encoder, processed_hidden