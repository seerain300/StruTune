import torch
import triton
import triton.language as tl


@triton.jit
def concat_linear_split_kernel(
    encoder_hidden_states_ptr,  # *const float
    hidden_states_ptr,          # *const float
    process_weight_ptr,         # *const float  shape [H, H]
    processed_concat_ptr,       # *float        shape [B, T+I, H]
    B: tl.constexpr,            # batch size
    T: tl.constexpr,            # text_seq_len
    I: tl.constexpr,            # img_seq_len
    H: tl.constexpr,            # hidden_dim
    stride_e_n, stride_e_s, stride_e_h,   # encoder strides
    stride_h_n, stride_h_s, stride_h_h,   # hidden strides
    stride_w_h, stride_w_k,               # process_weight strides (we'll use HxH, indexing as w[k, h])
    stride_out_n, stride_out_s, stride_out_h,  # output strides
    total_seq,                    # T + I (runtime int)
    tiles_h,                      # number of H tiles
    BLOCK_K: tl.constexpr,        # tile over input features
    BLOCK_H: tl.constexpr,        # tile over output hidden features
):
    # Program IDs
    n = tl.program_id(0)          # batch
    s = tl.program_id(1)          # sequence position in [0, total_seq)
    h_tile = tl.program_id(2)     # tile index along hidden_dim

    # Compute h offsets and mask
    h_offsets = h_tile * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = h_offsets < H

    # Determine source (encoder vs hidden) based on s
    is_encoder = s < T
    # Load input vector from either encoder or hidden based on is_encoder
    # Inputs are [B, S, H] contiguous with stride_e_h = 1 (after .contiguous())
    if is_encoder:
        # src row = encoder_hidden_states[n, s, :]
        # Pointer arithmetic: base = n * stride_e_n + s * stride_e_s
        base_e = n * stride_e_n + s * stride_e_s
        x = tl.load(encoder_hidden_states_ptr + base_e + h_offsets * stride_e_h, mask=h_mask, other=0.0)
    else:
        # src row = hidden_states[n, s - T, :]
        base_h = n * stride_h_n + (s - T) * stride_h_s
        x = tl.load(hidden_states_ptr + base_h + h_offsets * stride_h_h, mask=h_mask, other=0.0)

    # Load process_weight.T as W[k, h] where weight is [H, H] (we pass weight as [H, H])
    # Accumulate over K (input features) in tiles
    acc = tl.zeros([BLOCK_H], dtype=tl.float32)
    k_start = 0
    while k_start < H:
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < H
        # weight is [H, H]; indexing w[k, h] means ptr = process_weight_ptr + k*stride_w_h + h*stride_w_k
        w = tl.load(process_weight_ptr + k_offsets[:, None] * stride_w_h + h_offsets[None, :] * stride_w_k,
                    mask=k_mask[:, None] & h_mask[None, :], other=0.0)
        # x is [BLOCK_H], w is [BLOCK_K, BLOCK_H]; multiply-accumulate along K tile
        # We'll do a simple loop over k to be safe with Triton's broadcasting (less risk of kernel failures)
        for kk in range(0, BLOCK_K):
            k_idx = k_start + kk
            k_valid = k_idx < H
            if k_valid:
                xk = x[kk]  # elementwise
                acc += w[kk, :] * xk
        k_start += BLOCK_K

    # Store the accumulated vector into processed_concat[n, s, :]
    out_ptr = processed_concat_ptr + n * stride_out_n + s * stride_out_s + h_offsets * stride_out_h
    tl.store(out_ptr, acc, mask=h_mask)


def run(hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Triton implementation of:
      concatenated = cat([encoder_hidden_states, hidden_states], dim=1)  # [B, T+I, H]
      processed = concatenated @ process_weight.T                        # [B, T+I, H]
      processed_encoder = processed[:, :T, :]
      processed_hidden = processed[:, T:, :]
    All math performed inside Triton; tensors are made contiguous and float32.
    """
    assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "All tensors must be CUDA for Triton."
    assert hidden_states.dtype == torch.float32 and encoder_hidden_states.dtype == torch.float32 and process_weight.dtype == torch.float32, "Use float32."

    B = hidden_states.shape[0]
    T = encoder_hidden_states.shape[1]
    I = hidden_states.shape[1]
    H = hidden_states.shape[2]
    total_seq = T + I

    # Ensure contiguous tensors
    e = encoder_hidden_states.contiguous()
    h = hidden_states.contiguous()
    w = process_weight.contiguous()

    # Allocate output [B, total_seq, H] (contiguous)
    processed_concat = torch.empty((B, total_seq, H), device=e.device, dtype=torch.float32)

    # Strides
    stride_e_n, stride_e_s, stride_e_h = e.stride(0), e.stride(1), e.stride(2)
    stride_h_n, stride_h_s, stride_h_h = h.stride(0), h.stride(1), h.stride(2)
    # weight is [H, H]; we index as w[k, h] so strides are:
    # For contiguous [H, H]: stride_w_h = H, stride_w_k = 1, but we rely on w.contiguous() and use element indexing
    stride_w_h, stride_w_k = w.stride(0), w.stride(1)

    stride_out_n, stride_out_s, stride_out_h = processed_concat.stride(0), processed_concat.stride(1), processed_concat.stride(2)

    # Tiling parameters
    BLOCK_H = 128
    tiles_h = (H + BLOCK_H - 1) // BLOCK_H
    BLOCK_K = 128  # tile over input features (here equals hidden features)

    # Launch Triton kernel with 3D grid
    grid = (B, total_seq, tiles_h)
    concat_linear_split_kernel[grid](
        e, h, w, processed_concat,
        B, T, I, H,
        stride_e_n, stride_e_s, stride_e_h,
        stride_h_n, stride_h_s, stride_h_h,
        stride_w_h, stride_w_k,
        stride_out_n, stride_out_s, stride_out_h,
        total_seq,
        tiles_h,
        BLOCK_K=BLOCK_K, BLOCK_H=BLOCK_H,
        num_warps=4, num_stages=2,
    )

    # Split outputs
    processed_encoder = processed_concat[:, :T, :]
    processed_hidden = processed_concat[:, T:, :]

    return processed_encoder, processed_hidden


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        return run(*args)