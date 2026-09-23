import torch
import triton
import triton.language as tl


@triton.jit
def linear_concat_write_kernel(
    encoder_ptr,          # *f32 [B, T, H]
    hidden_ptr,           # *f32 [B, I, H]
    weight_ptr,           # *f32 [H, H]
    out_ptr,              # *f32 [B, T+I, H]
    B: tl.int32,
    T: tl.int32,
    I: tl.int32,
    H: tl.int32,
    # strides (in elements)
    encoder_stride_b, encoder_stride_t, encoder_stride_h,
    hidden_stride_b, hidden_stride_i, hidden_stride_h,
    weight_stride_h, weight_stride_k,    # weight is [H, H]
    out_stride_b, out_stride_t, out_stride_h,
    BLOCK_H: tl.constexpr,
):
    # Grid: (B, T+I)
    pid_b = tl.program_id(0)
    pid_t = tl.program_id(1)

    # Compute output vector for this (b, t). For t < T, use encoder; else use hidden (shifted by T).
    # Initialize accumulator for output vector
    output_vec = tl.zeros((H,), dtype=tl.float32)

    # Loop over H in tiles
    for h_off in range(0, H, BLOCK_H):
        offs_h = h_off + tl.arange(0, BLOCK_H)
        mask_h = offs_h < H

        # Load input vector x for this t (either from encoder or hidden)
        # Determine source
        use_encoder = pid_t < T
        # Compute pointers for x
        if use_encoder:
            # x = encoder[b, t, :]
            x_ptr = encoder_ptr + pid_b * encoder_stride_b + pid_t * encoder_stride_t
        else:
            # x = hidden[b, t - T, :]
            t_local = pid_t - T
            x_ptr = hidden_ptr + pid_b * hidden_stride_b + t_local * hidden_stride_i

        x_vec = tl.zeros((BLOCK_H,), dtype=tl.float32)
        # We need to multiply each h in offs_h with corresponding weight[h] and accumulate.
        # But we can't load weight with a 2D vector directly here; instead, we loop over K (H) again,
        # but we do it indirectly by loading weight row-wise and accumulating.
        # Instead, we switch to a standard matvec approach:
        # For each k tile, load weight rows corresponding to offs_h and x_chunk, accumulate.
        for k_off in range(0, H, BLOCK_H):
            k_vec = k_off + tl.arange(0, BLOCK_H)
            mask_k = k_vec < H
            # Load weight rows for these h indices: w[h, k] for h in offs_h, k in k_vec
            w_block = tl.load(
                weight_ptr + offs_h[:, None] * weight_stride_h + k_vec[None, :] * weight_stride_k,
                mask=mask_h[:, None] & mask_k[None, :],
                other=0.0,
            )  # shape [BLOCK_H, BLOCK_H]
            # Load input chunk x_chunk = hidden/b[b, t - T, k_vec] or encoder[b, t, k_vec]
            if use_encoder:
                x_chunk = tl.load(
                    encoder_ptr + pid_b * encoder_stride_b + pid_t * encoder_stride_t + k_vec * hidden_stride_h,
                    mask=mask_k,
                    other=0.0,
                )
            else:
                t_local = pid_t - T
                x_chunk = tl.load(
                    hidden_ptr + pid_b * hidden_stride_b + t_local * hidden_stride_i + k_vec * hidden_stride_h,
                    mask=mask_k,
                    other=0.0,
                )
            # Accumulate: output[h] += sum_k w[h, k] * x_chunk[k]
            # For each h in offs_h, take w[h, :] and dot with x_chunk
            for hi in range(BLOCK_H):
                h_idx = h_off + hi
                mhi = h_idx < H
                # Extract weight row and x_chunk element
                # weight row: w_block[hi, :]
                w_row = w_block[hi, :]
                # x_chunk[hi] using masked selection
                # We need x_chunk element at index hi among k_vec
                x_elem = tl.sum(x_chunk * (tl.arange(0, BLOCK_H) == hi), axis=0)
                # Multiply and accumulate to output_vec at h_idx
                # Only if mhi is True
                if mhi:
                    output_vec[h_idx] += x_elem

        # After processing all k tiles, store the output vector
        # We store the partial output_vec for this h_off tile
        # Note: output_vec is float32 accumulator
        # Store to out[b, t, offs_h]
        out_row_ptr = out_ptr + pid_b * out_stride_b + pid_t * out_stride_t
        tl.store(out_row_ptr + offs_h * out_stride_h, output_vec[offs_h], mask=mask_h)


@triton.jit
def split_seq_kernel(
    in_ptr,               # *f32 [B, T+I, H]
    out1_ptr,             # *f32 [B, T, H]
    out2_ptr,             # *f32 [B, I, H]
    B: tl.int32,
    T: tl.int32,
    I: tl.int32,
    H: tl.int32,
    in_stride_b, in_stride_t, in_stride_h,
    out1_stride_b, out1_stride_t, out1_stride_h,
    out2_stride_b, out2_stride_t, out2_stride_h,
    BLOCK_H: tl.constexpr,
):
    # Grid: (B, T) for output1, and (B, I) for output2
    # We implement two separate launches to keep things simple and robust.

    # Output 1: copy in[:, :T, :] -> out1
    pid_b = tl.program_id(0)
    pid_t = tl.program_id(1)

    # Initialize output vector
    out_vec = tl.zeros((H,), dtype=tl.float32)

    for h_off in range(0, H, BLOCK_H):
        offs_h = h_off + tl.arange(0, BLOCK_H)
        mask_h = offs_h < H
        # Load input vector from in[b, t, :]
        in_row_ptr = in_ptr + pid_b * in_stride_b + pid_t * in_stride_t
        # Read values for offs_h
        vals = tl.load(in_row_ptr + offs_h * in_stride_h, mask=mask_h, other=0.0)
        out_vec[offs_h] = vals

    out1_row_ptr = out1_ptr + pid_b * out1_stride_b + pid_t * out1_stride_t
    tl.store(out1_row_ptr + tl.arange(0, H) * out1_stride_h, out_vec, mask=tl.arange(0, H) < H)

    # Output 2: copy in[:, T:, :] -> out2
    pid_b = tl.program_id(0)
    pid_i = tl.program_id(1)
    t_total = T + I
    start_i = T

    out_vec = tl.zeros((H,), dtype=tl.float32)

    for h_off in range(0, H, BLOCK_H):
        offs_h = h_off + tl.arange(0, BLOCK_H)
        mask_h = offs_h < H
        in_row_ptr = in_ptr + pid_b * in_stride_b + (start_i + pid_i) * in_stride_t
        vals = tl.load(in_row_ptr + offs_h * in_stride_h, mask=mask_h, other=0.0)
        out_vec[offs_h] = vals

    out2_row_ptr = out2_ptr + pid_b * out2_stride_b + pid_i * out2_stride_t
    tl.store(out2_row_ptr + tl.arange(0, H) * out2_stride_h, out_vec, mask=tl.arange(0, H) < H)


def triton_linear_concat(hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> torch.Tensor:
    """
    Triton kernel that computes processed = cat([encoder_hidden_states, hidden_states], dim=1) @ process_weight.T
    Returns a tensor of shape [B, T+I, H] without using torch.cat or torch.matmul.
    """
    B = hidden_states.shape[0]
    T = encoder_hidden_states.shape[1]
    I = hidden_states.shape[1]
    H = hidden_states.shape[2]
    assert encoder_hidden_states.shape[2] == H and process_weight.shape[1] == H and process_weight.shape[0] == H, "Hidden dimension mismatch."

    # Allocate output
    out = torch.empty((B, T + I, H), device=hidden_states.device, dtype=torch.float32)

    # Launch kernel: grid (B, T+I)
    # Choose a reasonable tile size; 64 works well across many GPUs for H up to a few thousand.
    BLOCK_H = 64
    grid = (B, T + I)
    linear_concat_write_kernel[grid](
        encoder_hidden_states, hidden_states, process_weight, out,
        B, T, I, H,
        encoder_hidden_states.stride(0), encoder_hidden_states.stride(1), encoder_hidden_states.stride(2),
        hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
        process_weight.stride(0), process_weight.stride(1),
        out.stride(0), out.stride(1), out.stride(2),
        BLOCK_H=BLOCK_H,
        num_warps=4,
    )
    return out


def triton_split_seq(processed: torch.Tensor, T: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Triton kernel that splits processed [B, T+I, H] into
    output_encoder [B, T, H] and output_hidden [B, I, H].
    """
    B = processed.shape[0]
    I = processed.shape[1] - T
    H = processed.shape[2]
    assert processed.shape[2] == H, "Hidden dimension mismatch."

    output_encoder = torch.empty((B, T, H), device=processed.device, dtype=torch.float32)
    output_hidden = torch.empty((B, I, H), device=processed.device, dtype=torch.float32)

    # Launch two simple kernels: one for [:, :T, :], one for [:, T:, :]
    BLOCK_H = 128
    grid_e = (B, T)
    grid_i = (B, I)

    split_seq_kernel[grid_e](
        processed, output_encoder, output_hidden,
        B, T, I, H,
        processed.stride(0), processed.stride(1), processed.stride(2),
        output_encoder.stride(0), output_encoder.stride(1), output_encoder.stride(2),
        output_hidden.stride(0), output_hidden.stride(1), output_hidden.stride(2),
        BLOCK_H=BLOCK_H,
        num_warps=4,
    )

    return output_encoder, output_hidden


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version of the original run function.
        Returns (processed_encoder_hidden_states, processed_hidden_states).
        """
        # Ensure inputs are on CUDA for Triton
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Inputs must be on CUDA for Triton execution."
        # Compute concatenated @ weight.T entirely in Triton (no torch.cat, no torch.matmul)
        total = triton_linear_concat(encoder_hidden_states, hidden_states, process_weight)  # [B, T+I, H]
        T = encoder_hidden_states.shape[1]
        # Split in Triton
        processed_encoder, processed_hidden = triton_split_seq(total, T)
        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
