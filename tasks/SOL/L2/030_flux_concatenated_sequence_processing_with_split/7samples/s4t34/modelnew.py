import torch
import triton
import triton.language as tl


@triton.jit
def concat_select_kernel(
    encoder_ptr,          # *dtype [B, T, H]
    image_ptr,            # *dtype [B, I, H]
    out_ptr,              # *dtype [B, T+I, H] temporary concatenated output
    B: tl.int32,
    T: tl.int32,
    I: tl.int32,
    H: tl.int32,
    encoder_stride_b, encoder_stride_t, encoder_stride_h,
    image_stride_b, image_stride_i, image_stride_h,
    out_stride_b, out_stride_t, out_stride_h,
    BLOCK_H: tl.constexpr,
):
    # program ids: one per (b, t)
    pid_b = tl.program_id(0)  # batch
    pid_t = tl.program_id(1)  # output token index in [0, T+I)

    # Decide source: encoder if pid_t < T, else image at index pid_t - T
    # Compute input pointer based on source
    # We'll iterate over hidden dimension H in tiles
    for h_off in range(0, H, BLOCK_H):
        offs_h = h_off + tl.arange(0, BLOCK_H)
        mask_h = offs_h < H
        if pid_t < T:
            # Load from encoder[b, pid_t, :]
            x = tl.load(encoder_ptr + pid_b * encoder_stride_b + pid_t * encoder_stride_t + offs_h * encoder_stride_h,
                        mask=mask_h, other=0.0)
        else:
            # Load from image[b, pid_t - T, :]
            idx_i = pid_t - T
            x = tl.load(image_ptr + pid_b * image_stride_b + idx_i * image_stride_i + offs_h * image_stride_h,
                        mask=mask_h, other=0.0)
        # Store into out_ptr[b, pid_t, :]
        tl.store(out_ptr + pid_b * out_stride_b + pid_t * out_stride_t + offs_h * out_stride_h, x, mask=mask_h)


@triton.jit
def matvec_weight_kernel(
    x_ptr,        # *dtype [B, T+I, H] (concatenated input vectors)
    weight_ptr,   # *dtype [H, H]
    y_ptr,        # *dtype [B, T+I, H] output
    B: tl.int32,
    T_plus_I: tl.int32,
    H: tl.int32,
    x_stride_b, x_stride_t, x_stride_h,
    weight_stride_w, weight_stride_k,  # weight is [H, H]
    y_stride_b, y_stride_t, y_stride_h,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Each program handles one (b, t)
    pid_b = tl.program_id(0)
    pid_t = tl.program_id(1)

    # Accumulator for the output vector at (b, t)
    output_vec = tl.zeros((H,), dtype=tl.float32)

    # Loop over hidden dimension H in tiles
    for h_off in range(0, H, BLOCK_H):
        offs_h = h_off + tl.arange(0, BLOCK_H)
        mask_h = offs_h < H

        # Accumulate over K in tiles
        for k_off in range(0, H, BLOCK_K):
            offs_k = k_off + tl.arange(0, BLOCK_K)
            mask_k = offs_k < H

            # Load x chunk: x[b, t, k] for k in offs_k
            x_chunk = tl.load(
                x_ptr + pid_b * x_stride_b + pid_t * x_stride_t + offs_k * x_stride_h,
                mask=mask_k, other=0.0
            )  # [BLOCK_K]

            # Load weight block W[h, k] for h in offs_h, k in offs_k: shape [BLOCK_H, BLOCK_K]
            w_block = tl.load(
                weight_ptr + offs_h[:, None] * weight_stride_w + offs_k[None, :] * weight_stride_k,
                mask=mask_h[:, None] & mask_k[None, :],
                other=0.0
            )  # [BLOCK_H, BLOCK_K]

            # Accumulate: output_vec[h] += sum_k w_block[h, k] * x_chunk[k]
            # Compute w_block * x_chunk per h
            # We'll do an explicit sum across K tile for each h in the tile
            # This is a simple per-tile reduction and avoids complex broadcasting.
            # Create a loop over BLOCK_K, which Triton will unroll.
            acc_tile = tl.zeros((BLOCK_H,), dtype=tl.float32)
            for kk in range(0, BLOCK_K):
                k_idx = offs_k[kk]
                valid_k = k_idx < H
                # Select x_chunk[kk] with mask
                x_val = x_chunk[kk] if valid_k else 0.0
                acc_tile += w_block[:, kk] * x_val
            # Now add this tile's contribution to output_vec
            output_vec += acc_tile

    # Store the output vector to y[b, t, :]
    tl.store(
        y_ptr + pid_b * y_stride_b + pid_t * y_stride_t + offs_h * y_stride_h,
        output_vec,
        mask=mask_h
    )


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version:
        - Concatenate along sequence dimension without torch.cat using a Triton kernel.
        - Compute linear projection without torch.matmul using a Triton matvec kernel.
        - Return split streams as in the original.
        """
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Inputs must be on CUDA for Triton execution."

        # Ensure contiguity and dtype: compute in float32 for stability
        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        H = hidden_states.shape[2]
        assert encoder_hidden_states.shape[2] == H
        assert process_weight.shape[0] == H and process_weight.shape[1] == H

        # Cast inputs to float32 for kernel (no torch ops in heavy path)
        encoder = encoder_hidden_states.contiguous().to(torch.float32)
        image = hidden_states.contiguous().to(torch.float32)
        weight = process_weight.contiguous().to(torch.float32)

        # Temporary concatenated output buffer [B, T+I, H] in float32
        out_concat = torch.empty((B, T + I, H), device=encoder.device, dtype=torch.float32)

        # Launch concat-select kernel: populate out_concat[b, t, :] with encoder or image as appropriate
        BLOCK_H = 64  # hidden tile size
        grid1 = (B, T + I)
        concat_select_kernel[grid1](
            encoder, image, out_concat,
            B, T, I, H,
            encoder.stride(0), encoder.stride(1), encoder.stride(2),
            image.stride(0), image.stride(1), image.stride(2),
            out_concat.stride(0), out_concat.stride(1), out_concat.stride(2),
            BLOCK_H=BLOCK_H,
            num_warps=2,  # small tiles, modest parallelism
            num_stages=2
        )

        # Output buffer for processed [B, T+I, H] in float32
        processed = torch.empty((B, T + I, H), device=encoder.device, dtype=torch.float32)

        # Launch matvec kernel: y = out_concat @ weight (no bias)
        grid2 = (B, T + I)
        matvec_weight_kernel[grid2](
            out_concat, weight, processed,
            B, T + I, H,
            out_concat.stride(0), out_concat.stride(1), out_concat.stride(2),
            weight.stride(0), weight.stride(1),
            processed.stride(0), processed.stride(1), processed.stride(2),
            BLOCK_H=BLOCK_H,
            BLOCK_K=64,
            num_warps=2,
            num_stages=2
        )

        # Split back: processed_encoder = processed[:, :T, :], processed_hidden = processed[:, T:, :]
        processed_encoder = processed[:, :T, :]
        processed_hidden = processed[:, T:, :]

        # Return outputs. They are float32. If original inputs were fp16, consider casting back to match.
        # To keep computation stable, we return float32 outputs (the original PyTorch code would return float tensors too).
        return processed_encoder, processed_hidden