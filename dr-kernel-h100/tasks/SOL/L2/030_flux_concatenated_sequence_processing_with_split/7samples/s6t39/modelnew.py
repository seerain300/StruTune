import torch
import triton
import triton.language as tl


@triton.jit
def concat_linear_split_kernel(
    encoder_hidden_states, hidden_states, process_weight, processed_concat,
    B, T, I, H,
    stride_e_n, stride_e_s, stride_e_h,
    stride_h_n, stride_h_s, stride_h_h,
    stride_w_h, stride_w_k,  # weight is [H, H]; we use W^T in matmul implicitly
    stride_out_n, stride_out_s, stride_out_h,
    total_seq,
    tiles_h,
    BLOCK_K: tl.constexpr, BLOCK_H: tl.constexpr,
):
    # Grid: (B, total_seq, tiles_h)
    n = tl.program_id(0)
    s = tl.program_id(1)
    h_tile = tl.program_id(2)

    h_offsets = h_tile * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = h_offsets < H

    # Decide input source based on s: if s < T, use encoder, else use hidden at index s - T
    use_encoder = s < T
    # Pointer to input row: either encoder[n, s, :] or hidden[n, s - T, :]
    if use_encoder:
        inp_row_ptr = encoder_hidden_states + n * stride_e_n + s * stride_e_s
    else:
        inp_row_ptr = hidden_states + n * stride_h_n + (s - T) * stride_h_s

    # Accumulator for out[n, s, h_offsets]
    out_vec = tl.zeros([BLOCK_H], dtype=tl.float32)

    # Accumulate over K tiles
    for k in range(0, H, BLOCK_K):
        k_offsets = k + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < H

        # Load input vector segment of length BLOCK_K (implicitly treating as W^T)
        # Note: input is [H], process_weight is [H, H], and we multiply input_vec.T @ W
        inp_segment = tl.load(inp_row_ptr + k_offsets * stride_e_h, mask=mask_k, other=0.0)  # [BLOCK_K]

        # Load weight segment: weight[k_offsets, h_offsets]
        # weight is [H, H], strides (stride_w_h, stride_w_k)
        weight_tile = tl.load(process_weight + k_offsets[:, None] * stride_w_h + h_offsets[None, :] * stride_w_k,
                              mask=mask_k[:, None] & mask_h[None, :], other=0.0)  # [BLOCK_K, BLOCK_H]

        # Accumulate: out_vec[h_offsets] += inp_segment[k] * weight_tile[k, h_offsets]
        # This is equivalent to out_vec += sum_k inp[k] * W^T[:, h_offsets][k]
        out_vec += tl.sum(weight_tile * inp_segment[:, None], axis=0)

    # Store the accumulated out_vec to processed_concat[n, s, h_offsets]
    out_row_ptr = processed_concat + n * stride_out_n + s * stride_out_s
    tl.store(out_row_ptr + h_offsets * stride_out_h, out_vec, mask=mask_h)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        Triton-optimized version of:
            concatenated = torch.cat([encoder_hidden_states, hidden_states], dim=1)  # [B, T+I, H]
            processed = concatenated @ process_weight.t()  # [B, T+I, H]
            processed_encoder = processed[:, :T, :]
            processed_hidden = processed[:, T:, :]
        """
        # Ensure dtypes and contiguity
        if hidden_states.dtype != torch.float32:
            hidden_states = hidden_states.float()
        if encoder_hidden_states.dtype != torch.float32:
            encoder_hidden_states = encoder_hidden_states.float()
        if process_weight.dtype != torch.float32:
            process_weight = process_weight.float()

        hidden_states = hidden_states.contiguous()
        encoder_hidden_states = encoder_hidden_states.contiguous()
        process_weight = process_weight.contiguous()

        B = encoder_hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        H = encoder_hidden_states.shape[2]
        assert hidden_states.shape[2] == H
        assert process_weight.shape == (H, H)

        total_seq = T + I

        # Allocate output buffer [B, T+I, H], contiguous
        processed_concat = torch.empty((B, total_seq, H), device=hidden_states.device, dtype=hidden_states.dtype)

        # Tile sizes: use 128 for H and K by default; adjust if H < 128
        BLOCK_H = 128
        BLOCK_K = 128

        tiles_h = (H + BLOCK_H - 1) // BLOCK_H
        grid = (B, total_seq, tiles_h)

        # Launch Triton kernel
        concat_linear_split_kernel[grid](
            encoder_hidden_states, hidden_states, process_weight, processed_concat,
            B, T, I, H,
            encoder_hidden_states.stride(0), encoder_hidden_states.stride(1), encoder_hidden_states.stride(2),
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            process_weight.stride(0), process_weight.stride(1),
            processed_concat.stride(0), processed_concat.stride(1), processed_concat.stride(2),
            total_seq,
            tiles_h,
            BLOCK_K=BLOCK_K, BLOCK_H=BLOCK_H,
            num_warps=8,  # higher occupancy
            num_stages=3,  # deeper pipelining
        )

        # Split outputs
        processed_encoder = processed_concat[:, :T, :]
        processed_hidden = processed_concat[:, T:, :]

        return processed_encoder, processed_hidden