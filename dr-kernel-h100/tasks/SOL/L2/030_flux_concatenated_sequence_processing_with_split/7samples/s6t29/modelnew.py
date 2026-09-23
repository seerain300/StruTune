import torch
import triton
import triton.language as tl


@triton.jit
def concat_linear_split_kernel(
    encoder_ptr,   # [B, T, H], float32, contiguous
    hidden_ptr,    # [B, I, H], float32, contiguous
    weight_ptr,    # [H, H], float32, contiguous
    out_ptr,       # [B, T+I, H], float32, contiguous
    B: tl.int32, T: tl.int32, I: tl.int32, H: tl.int32,
    stride_e_n: tl.int32, stride_e_s: tl.int32, stride_e_h: tl.int32,
    stride_h_n: tl.int32, stride_h_s: tl.int32, stride_h_h: tl.int32,
    stride_w_h: tl.int32, stride_w_k: tl.int32,
    stride_out_n: tl.int32, stride_out_s: tl.int32, stride_out_h: tl.int32,
    total_seq: tl.int32,
    tiles_h: tl.int32,
    BLOCK_K: tl.constexpr, BLOCK_H: tl.constexpr,
):
    # Program IDs
    pid_n = tl.program_id(0)  # batch
    pid_s = tl.program_id(1)  # sequence position in [0, total_seq)
    pid_th = tl.program_id(2) # tile over hidden dim

    # Compute H offsets for this tile
    h_offsets = pid_th * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = h_offsets < H

    # Determine if this s corresponds to encoder or hidden
    is_encoder = pid_s < T
    # Compute input pointer (row) based on s
    # If encoder: input_row = encoder[pid_n, pid_s, :]
    # If hidden: input_row = hidden[pid_n, pid_s - T, :]
    # Note: We do not compute input row explicitly; instead we select which pointer to use.
    # Prepare input row pointer selection:
    in_ptr_sel = encoder_ptr if is_encoder else hidden_ptr
    s_in = pid_s if is_encoder else (pid_s - T)
    # Input strides for selected pointer
    stride_in_n = stride_e_n if is_encoder else stride_h_n
    stride_in_s = stride_e_s if is_encoder else stride_h_s
    stride_in_h = stride_e_h if is_encoder else stride_h_h

    # Accumulator for this output row
    acc = tl.zeros([BLOCK_H], dtype=tl.float32)

    # Loop over K dimension (input features) in tiles
    # Note: Weight is [H, H]; we multiply input_vec (length H) by weight.T (also length H vectors)
    for k_start in range(0, H, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < H

        # Load input vector slice: shape [BLOCK_K]
        # Pointer arithmetic for input row elements:
        # inp_ptr = in_ptr_sel + pid_n*stride_in_n + s_in*stride_in_s + k_offsets*stride_in_h
        inp_ptr = in_ptr_sel + pid_n * stride_in_n + s_in * stride_in_s + k_offsets * stride_in_h
        inp = tl.load(inp_ptr, mask=k_mask, other=0.0)

        # Load weight.T slice: shape [BLOCK_K]
        # weight_ptr has shape [H, H]; weight.T indexing for row k: weight[k, h] -> weight_ptr[k, h]
        w_ptr = weight_ptr + h_offsets[:, None] * stride_w_h + k_offsets[None, :] * stride_w_k
        w_tile = tl.load(w_ptr, mask=h_mask[:, None] & k_mask[None, :], other=0.0)

        # Accumulate: acc[h] += sum_k w_tile[h, k] * inp[k]
        # w_tile shape [BLOCK_H, BLOCK_K], inp shape [BLOCK_K]
        acc += tl.sum(w_tile * inp[None, :], axis=1)

    # Store results into out[:, s, :]
    out_ptr_row = out_ptr + pid_n * stride_out_n + pid_s * stride_out_s + h_offsets * stride_out_h
    tl.store(out_ptr_row, acc, mask=h_mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized forward that:
        - Concatenates encoder_hidden_states and hidden_states along sequence dim
        - Applies linear projection with process_weight.T
        - Splits back into encoder and hidden outputs
        """
        assert hidden_states.dim() == 3 and encoder_hidden_states.dim() == 3, "Inputs must be [B, S, H]"
        assert process_weight.dim() == 2, "process_weight must be [H, H]"
        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        H = hidden_states.shape[2]

        # Enforce dtype and contiguity
        dtype = torch.float32
        if hidden_states.dtype != dtype:
            hidden_states = hidden_states.to(dtype)
        if encoder_hidden_states.dtype != dtype:
            encoder_hidden_states = encoder_hidden_states.to(dtype)
        if process_weight.dtype != dtype:
            process_weight = process_weight.to(dtype)
        hidden_states = hidden_states.contiguous()
        encoder_hidden_states = encoder_hidden_states.contiguous()
        process_weight = process_weight.contiguous()

        total_seq = T + I
        # Allocate output tensor [B, total_seq, H], float32, contiguous
        processed_concat = torch.empty((B, total_seq, H), device=hidden_states.device, dtype=dtype)

        # Extract strides
        stride_e_n, stride_e_s, stride_e_h = encoder_hidden_states.stride(0), encoder_hidden_states.stride(1), encoder_hidden_states.stride(2)
        stride_h_n, stride_h_s, stride_h_h = hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2)
        stride_w_h, stride_w_k = process_weight.stride(0), process_weight.stride(1)  # [H, H]
        stride_out_n, stride_out_s, stride_out_h = processed_concat.stride(0), processed_concat.stride(1), processed_concat.stride(2)

        tiles_h = (H + 64 - 1) // 64  # using BLOCK_H=64 for safety across various H
        grid = (B, total_seq, tiles_h)

        concat_linear_split_kernel[grid](
            encoder_hidden_states, hidden_states, process_weight, processed_concat,
            B, T, I, H,
            stride_e_n, stride_e_s, stride_e_h,
            stride_h_n, stride_h_s, stride_h_h,
            stride_w_h, stride_w_k,
            stride_out_n, stride_out_s, stride_out_h,
            total_seq,
            tiles_h,
            BLOCK_K=64, BLOCK_H=64,
            num_warps=4,
        )

        # Split outputs
        processed_encoder = processed_concat[:, :T, :]
        processed_hidden = processed_concat[:, T:, :]

        return processed_encoder, processed_hidden