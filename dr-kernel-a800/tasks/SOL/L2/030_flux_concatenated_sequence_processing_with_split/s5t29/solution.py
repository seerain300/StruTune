import torch
import triton
import triton.language as tl


@triton.jit
def batched_linear_row_kernel(
    in_ptr,           # *f32, concatenated [B, T, H_in]
    weightT_ptr,      # *f32, [H_in, H_out] (process_weight.T)
    out_ptr,          # *f32, [B, T, H_out] (initialized to zeros)
    B: tl.constexpr,
    T: tl.constexpr,
    H_in: tl.constexpr,   # input hidden dim (concatenated row length)
    H_out: tl.constexpr,  # output hidden dim (process_weight shape[1])
    stride_i_b: tl.constexpr,
    stride_i_t: tl.constexpr,
    stride_i_h: tl.constexpr,  # input hidden stride
    stride_w_k: tl.constexpr,  # weight_T dim-0 stride
    stride_w_n: tl.constexpr,  # weight_T dim-1 stride
    stride_o_b: tl.constexpr,
    stride_o_t: tl.constexpr,
    stride_o_h: tl.constexpr,  # output hidden stride
    BLOCK_K: tl.constexpr,     # chunk size over H_in
):
    # Each program handles one (b, t) and accumulates the whole output row
    b = tl.program_id(0)
    t = tl.program_id(1)
    if (b >= B) or (t >= T):
        return

    # Accumulator for output row
    acc = tl.zeros([H_out], dtype=tl.float32)

    # Loop over input hidden dimension in chunks
    k0 = 0
    while k0 < H_in:
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < H_in

        # Load input chunk: in_ptr[b, t, k_offsets]
        in_ptrs = in_ptr + b * stride_i_b + t * stride_i_t + k_offsets * stride_i_h
        in_chunk = tl.load(in_ptrs, mask=mask_k, other=0.0)  # shape [BLOCK_K]

        # Load weight_T tile: [BLOCK_K, H_out] for current k chunk
        # Columns are all H_out (full output dimension per iteration)
        w_ptrs = weightT_ptr + k_offsets[:, None] * stride_w_k + tl.arange(0, H_out)[None, :] * stride_w_n
        w_tile = tl.load(w_ptrs, mask=mask_k[:, None], other=0.0)  # shape [BLOCK_K, H_out]

        # Accumulate: acc += sum_k(w_tile[k, :] * in_chunk[k])
        # Broadcast in_chunk[:, None] to [BLOCK_K, 1] and multiply with [BLOCK_K, H_out]
        acc += tl.sum(w_tile * in_chunk[:, None], axis=0)

        k0 += BLOCK_K

    # Store accumulated result into out[b, t, :]
    o_ptrs = out_ptr + b * stride_o_b + t * stride_o_t + tl.arange(0, H_out) * stride_o_h
    tl.store(o_ptrs, acc)  # out is already zeros; we store the full row


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        # Shapes: hidden_states [B, Simg, H], encoder_hidden_states [B, Stext, H], process_weight [H, H]
        B = hidden_states.shape[0]
        Stext = encoder_hidden_states.shape[1]
        Simg = hidden_states.shape[1]
        H = hidden_states.shape[2]
        Hout = process_weight.shape[1]

        # Concatenate along sequence dimension on host
        concatenated = torch.cat([encoder_hidden_states, hidden_states], dim=1)  # [B, T, H], T = Stext + Simg

        # Allocate output [B, T, Hout]
        out = torch.zeros((B, Stext + Simg, Hout), dtype=torch.float32, device=concatenated.device)

        # Ensure contiguous for predictable strides
        in_ctg = concatenated.contiguous()
        weight_T = process_weight.t().contiguous()  # [H, Hout]

        # Choose chunk size (conservative for robustness)
        BLOCK_K = 32

        # Launch Triton kernel: 2D grid over (B, T)
        grid = (B, Stext + Simg)
        batched_linear_row_kernel[grid](
            in_ctg, weight_T, out,
            B, Stext + Simg, H, Hout,
            in_ctg.stride(0), in_ctg.stride(1), in_ctg.stride(2),
            weight_T.stride(0), weight_T.stride(1),
            out.stride(0), out.stride(1), out.stride(2),
            BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        # Split back into separate streams
        processed_encoder = out[:, :Stext, :]
        processed_hidden = out[:, Stext:, :]
        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
