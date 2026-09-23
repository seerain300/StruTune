import torch
import triton
import triton.language as tl


@triton.jit
def _concat_project_split_kernel(
    encoder_ptr,        # *f32, [B, T, H]
    hidden_ptr,         # *f32, [B, I, H]
    weight_ptr,         # *f32, [H, H]
    out_encoder_ptr,    # *f32, [B, T, H]
    out_hidden_ptr,     # *f32, [B, I, H]
    B: tl.constexpr,    # batch size
    T: tl.constexpr,    # text seq len
    I: tl.constexpr,    # image seq len
    H: tl.constexpr,    # hidden dim
    # strides (in elements)
    e_b_stride, e_t_stride, e_h_stride,
    h_b_stride, h_i_stride, h_h_stride,
    oeb_b_stride, oeb_t_stride, oeb_h_stride,
    oeh_b_stride, oeh_i_stride, oeh_h_stride,
    w_h_stride0, w_h_stride1,  # weight strides for [H, H]
):
    b = tl.program_id(0)  # batch index
    l = tl.program_id(1)  # sequence position in [0, T+I)

    # Determine source: encoder or hidden stream
    is_encoder = l < T

    # Compute base offsets for the input row vectors
    # For encoder: concatenated[b, l, :]
    # For hidden: concatenated[b, l + T, :]
    if is_encoder:
        row_ptr = encoder_ptr + b * e_b_stride + l * e_h_stride
    else:
        row_ptr = hidden_ptr + b * h_b_stride + (l - T) * h_h_stride

    # Vector of H elements for this row
    h_idx = tl.arange(0, H)
    row_vec = tl.load(row_ptr + h_idx * e_h_stride if is_encoder else h_h_stride, mask=h_idx < H, other=0.0)  # other=0.0 ensures safe load

    # Accumulator for output row
    out_vec = tl.zeros((H,), dtype=tl.float32)

    # Multiply by weight.T: out_vec[j] += row_vec[i] * weight[i, j], for j in 0..H-1
    # We loop i across H in tiles (BLOCK) to keep it efficient.
    BLOCK = 64  # tile size for reduction over hidden dim
    for i_start in range(0, H, BLOCK):
        i_idx = i_start + tl.arange(0, BLOCK)
        # Load a tile of the row (i elements)
        row_tile = tl.load(row_ptr + i_idx * (e_h_stride if is_encoder else h_h_stride), mask=i_idx < H, other=0.0)
        # Load corresponding tile of weight rows W[i, :]
        # weight is [H, H], we want W[i, :] for each i in i_idx
        w_rows = tl.load(
            weight_ptr + i_idx[:, None] * w_h_stride0 + tl.arange(0, H)[None, :] * w_h_stride1,
            mask=(i_idx[:, None] < H) & (tl.arange(0, H)[None, :] < H),
            other=0.0
        )  # shape [BLOCK, H]
        # Accumulate: out_vec += sum_i row_tile[i] * w_rows[i, :]
        # Broadcast row_tile to [BLOCK, H] and multiply
        out_vec += tl.sum(w_rows * row_tile[:, None], axis=0)

    # Store to appropriate output
    if is_encoder:
        out_row_ptr = out_encoder_ptr + b * oeb_b_stride + l * oeb_h_stride
    else:
        out_row_ptr = out_hidden_ptr + b * oeh_b_stride + (l - T) * oeh_h_stride
    tl.store(out_row_ptr + h_idx * (oeb_h_stride if is_encoder else oeh_h_stride), out_vec)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        Triton-only implementation of:
          concatenated = torch.cat([encoder_hidden_states, hidden_states], dim=1)  # [B, T+I, H]
          processed = concatenated @ process_weight.T                             # [B, T+I, H]
          processed_encoder = processed[:, :T, :]
          processed_hidden   = processed[:, T:, :]
        """
        assert hidden_states.ndim == 3 and encoder_hidden_states.ndim == 3, "Inputs must be [B, L, H]"
        B, T, H = encoder_hidden_states.shape
        B2, I, H2 = hidden_states.shape
        assert B == B2 and H == H2, "Batch and hidden_dim must match for both inputs"
        assert process_weight.shape == (H, H), "process_weight must be [hidden_dim, hidden_dim]"

        device = hidden_states.device
        dtype = hidden_states.dtype  # typically float32; we compute in fp32

        # Ensure tensors are on same device and dtype; we'll do compute in fp32
        encoder = encoder_hidden_states.to(device=device, dtype=torch.float32)
        hidden = hidden_states.to(device=device, dtype=torch.float32)
        weight = process_weight.to(device=device, dtype=torch.float32)

        # Allocate outputs (fp32 for stability)
        processed_encoder = torch.empty((B, T, H), dtype=torch.float32, device=device)
        processed_hidden = torch.empty((B, I, H), dtype=torch.float32, device=device)

        # Launch Triton kernel: one program per (b, l) in concatenated sequence
        grid = (B, T + I)
        _concat_project_split_kernel[grid](
            encoder, hidden, weight,
            processed_encoder, processed_hidden,
            B, T, I, H,
            # strides
            encoder_hidden_states.stride(0), encoder_hidden_states.stride(1), encoder_hidden_states.stride(2),
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
            processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
            process_weight.stride(0), process_weight.stride(1),
            num_warps=1,
            num_stages=1,
        )

        # If original dtype is not float32, cast outputs back to original dtype
        # The original run returns float32 by default; we keep fp32 for stability
        # You can uncomment the following if you need to match original dtype exactly:
        # processed_encoder = processed_encoder.to(encoder_hidden_states.dtype)
        # processed_hidden = processed_hidden.to(hidden_states.dtype)

        return processed_encoder, processed_hidden