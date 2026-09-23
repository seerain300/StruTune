import torch
import triton
import triton.language as tl


@triton.jit
def concat_linear_split_kernel(
    encoder_hidden_states, hidden_states, process_weight, processed_concat,
    B, T, I, H, total_seq,
    stride_e_n, stride_e_s, stride_e_h,
    stride_h_n, stride_h_s, stride_h_h,
    stride_w_h, stride_w_k,          # process_weight has shape [H, H]
    stride_out_n, stride_out_s, stride_out_h,
    tiles_h,
    BLOCK_K: tl.constexpr, BLOCK_H: tl.constexpr,
):
    # program ids: over batch, sequence positions, and H tiles
    n = tl.program_id(0)           # batch index
    s = tl.program_id(1)           # sequence position in [0, total_seq)
    tile_h = tl.program_id(2)      # tile index over hidden dimension

    # Decide which input row to use based on s: encoder if s < T, else hidden at s - T
    # Note: s is guaranteed to be in [0, total_seq), and total_seq = T + I.
    # s < T -> from encoder; otherwise from hidden at s - T.
    is_encoder = s < T

    # Compute base offsets for input rows
    # If is_encoder, base_e = n*T + s; else base_e = n*I + (s - T)
    base_e = tl.where(is_encoder, n * T + s, n * I + (s - T))

    # Accumulator for the output vector for this (n, s) over hidden tile
    acc = tl.zeros([BLOCK_H], dtype=tl.float32)

    # Precompute h_offsets for this tile
    h_offsets = tile_h * BLOCK_H + tl.arange(0, BLOCK_H)

    # Loop over K in tiles
    # K is the input feature dimension of the rows (T or I). For each k tile, load input vector input_vec.
    # Then, for each h in this tile, load W^T[h, :] (i.e., process_weight[h, :]) and accumulate dot product.
    for k_start in range(0, 1024, BLOCK_K):  # upper bound; actual loop executes only up to K
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < H  # K for encoder/hidden inputs is H

        # Load input vector input_vec = encoder[n, s, :] or hidden[n, s - T, :]
        # Pointer: base_e + stride_e_n * n + stride_e_s * s + stride_e_h * k_offsets
        input_vec = tl.load(
            encoder_hidden_states + base_e + stride_e_n * n + stride_e_s * s + stride_e_h * k_offsets,
            mask=mask_k,
            other=0.0,
        )

        # Accumulate dot product across this K tile
        # For each h in the current tile, load W^T[h, :] = process_weight[h, :] and multiply with input_vec
        # Reduce along K tile
        for hi in range(BLOCK_H):
            h_idx = h_offsets[hi]
            mask_h = h_idx < H
            w_row = tl.load(
                process_weight + h_idx * stride_w_h + k_offsets * stride_w_k,
                mask=mask_k & mask_h,
                other=0.0,
            )
            # acc[h_idx] += sum_{k in k_offsets} input_vec[k] * w_row[k]
            # Manual reduction for robustness
            acc[hi] += tl.sum(w_row * input_vec, axis=0)

    # Store the computed output vector into processed_concat[n, s, :]
    out_ptrs = processed_concat + n * stride_out_n + s * stride_out_s + h_offsets * stride_out_h
    mask_h_store = h_offsets < H
    tl.store(out_ptrs, acc, mask=mask_h_store)


def _run_triton_concat_linear(encoder_hidden_states: torch.Tensor,
                              hidden_states: torch.Tensor,
                              process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Triton implementation of:
      concatenated = torch.cat([encoder_hidden_states, hidden_states], dim=1)  # [B, T+I, H]
      processed = concatenated @ process_weight.t()                           # [B, T+I, H]
      processed_encoder = processed[:, :T, :]
      processed_hidden = processed[:, T:, :]
    Returns (processed_encoder, processed_hidden).
    """
    assert encoder_hidden_states.is_cuda and hidden_states.is_cuda and process_weight.is_cuda, "Tensors must be CUDA."
    B = encoder_hidden_states.shape[0]
    T = encoder_hidden_states.shape[1]
    I = hidden_states.shape[1]
    H = encoder_hidden_states.shape[2]
    assert hidden_states.shape[2] == H, "hidden_dim must match between encoder and image."
    assert process_weight.shape[0] == H and process_weight.shape[1] == H, "process_weight must be [H, H]."

    # Ensure float32 and contiguous
    dtype = torch.float32
    encoder = encoder_hidden_states.contiguous().to(dtype)
    hidden = hidden_states.contiguous().to(dtype)
    weight = process_weight.contiguous().to(dtype)

    total_seq = T + I

    # Allocate output
    processed_concat = torch.empty((B, total_seq, H), device=encoder.device, dtype=dtype)

    # Strides
    stride_e_n, stride_e_s, stride_e_h = encoder.stride(0), encoder.stride(1), encoder.stride(2)
    stride_h_n, stride_h_s, stride_h_h = hidden.stride(0), hidden.stride(1), hidden.stride(2)
    stride_w_h, stride_w_k = weight.stride(0), weight.stride(1)  # weight [H, H]
    stride_out_n, stride_out_s, stride_out_h = processed_concat.stride(0), processed_concat.stride(1), processed_concat.stride(2)

    tiles_h = (H + 128 - 1) // 128  # default BLOCK_H=128

    # Launch Triton kernel
    grid = (B, total_seq, tiles_h)
    concat_linear_split_kernel[grid](
        encoder, hidden, weight, processed_concat,
        B, T, I, H, total_seq,
        stride_e_n, stride_e_s, stride_e_h,
        stride_h_n, stride_h_s, stride_h_h,
        stride_w_h, stride_w_k,
        stride_out_n, stride_out_s, stride_out_h,
        tiles_h,
        BLOCK_K=128, BLOCK_H=128,
        num_warps=4,
        num_stages=2,
    )

    # Split outputs into encoder and hidden streams
    processed_encoder = processed_concat[:, :T, :]
    processed_hidden = processed_concat[:, T:, :]
    return processed_encoder, processed_hidden


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                encoder_hidden_states: torch.Tensor,
                process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return _run_triton_concat_linear(encoder_hidden_states, hidden_states, process_weight)


def run(*args):
    return ModelNew()(*args)
