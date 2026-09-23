import torch
import triton
import triton.language as tl


@triton.jit
def batched_matmul_cat_with_weightT_kernel(
    encoder_ptr,  # [B, T, H], float32
    hidden_ptr,   # [B, I, H], float32
    weight_ptr,   # [H, H], float32
    out_ptr,      # [B, T+I, H], float32
    B: tl.int32, T: tl.int32, I: tl.int32, H: tl.int32,
    stride_e_n: tl.int32, stride_e_s: tl.int32, stride_e_h: tl.int32,
    stride_h_n: tl.int32, stride_h_s: tl.int32, stride_h_h: tl.int32,
    stride_w_h: tl.int32, stride_w_k: tl.int32,
    stride_out_n: tl.int32, stride_out_s: tl.int32, stride_out_h: tl.int32,
    BLOCK_K: tl.constexpr, BLOCK_H: tl.constexpr,
):
    # Each program computes one output row (n, s) where s in [0, T+I)
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)

    # Accumulator for the output vector of length H
    acc = tl.zeros((BLOCK_H,), dtype=tl.float32)

    # Determine if this sequence index belongs to encoder or hidden stream
    is_encoder = pid_s < T

    # Loop over input features K (equals H) in tiles
    k0 = 0
    while k0 < H:
        # Load input vector slice input[k0 : k0 + BLOCK_K]
        if is_encoder:
            src_ptr = encoder_ptr + pid_n * stride_e_n + pid_s * stride_e_s
            k_offsets = k0 + tl.arange(0, BLOCK_K)
            # Load with mask for tails
            input_slice = tl.load(src_ptr + k_offsets * stride_e_h, mask=k_offsets < H, other=0.0).to(tl.float32)
        else:
            src_ptr = hidden_ptr + pid_n * stride_h_n + (pid_s - T) * stride_h_s
            k_offsets = k0 + tl.arange(0, BLOCK_K)
            input_slice = tl.load(src_ptr + k_offsets * stride_h_h, mask=k_offsets < H, other=0.0).to(tl.float32)

        # Load corresponding weight slice weight[k : k+BLOCK_K, h : h+BLOCK_H]
        # We'll accumulate into acc over this K-tile
        # For each k in the tile, compute dot with weight rows
        for kk in range(BLOCK_K):
            k_idx = k0 + kk
            # Skip if k_idx >= H (mask already zeroed input, but guard weight too)
            if k_idx < H:
                # weight row at k_idx across H tile
                h_offsets = tl.arange(0, BLOCK_H)
                w_row = tl.load(
                    weight_ptr + k_idx * stride_w_h + h_offsets * stride_w_k,
                    mask=h_offsets < H,
                    other=0.0,
                ).to(tl.float32)
                # Accumulate dot: sum over K-tile element with weight row
                # We have input_slice[kk] as scalar, broadcast w_row
                acc += input_slice[kk] * w_row

        k0 += BLOCK_K

    # Store the accumulated output vector into out[n, s, :]
    out_row_ptr = out_ptr + pid_n * stride_out_n + pid_s * stride_out_s
    h_offsets = tl.arange(0, BLOCK_H)
    tl.store(out_row_ptr + h_offsets * stride_out_h, acc, mask=h_offsets < H)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                encoder_hidden_states: torch.Tensor,
                process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version:
        - Concatenates encoder_hidden_states and hidden_states along sequence dimension.
        - Applies linear projection using process_weight.T.
        - Splits back into separate encoder and image streams.

        All heavy computation is done in Triton kernels.
        """
        # Ensure dtypes and contiguity; operate in float32
        assert hidden_states.dtype == torch.float32 and encoder_hidden_states.dtype == torch.float32 and process_weight.dtype == torch.float32, \
            "This Triton implementation expects float32 tensors."

        B, I, H = hidden_states.shape
        T, H_e, _ = encoder_hidden_states.shape
        assert H_e == H, "hidden_dim must match between encoder_hidden_states and hidden_states"

        # Make inputs contiguous for predictable strides
        e = encoder_hidden_states.contiguous()
        h = hidden_states.contiguous()
        w = process_weight.contiguous()  # [H, H]

        # Allocate output [B, T+I, H]
        total_seq = T + I
        out = torch.empty((B, total_seq, H), device=h.device, dtype=torch.float32)

        # Strides (in elements)
        stride_e_n, stride_e_s, stride_e_h = e.stride(0), e.stride(1), e.stride(2)
        stride_h_n, stride_h_s, stride_h_h = h.stride(0), h.stride(1), h.stride(2)
        stride_w_h, stride_w_k = w.stride(0), w.stride(1)  # w is [H, H]
        stride_out_n, stride_out_s, stride_out_h = out.stride(0), out.stride(1), out.stride(2)

        # Launch Triton kernel over grid (B, T+I)
        # Tile sizes: H is typically 64-256; choose BLOCK_H=128 and BLOCK_K=64 for decent performance
        BLOCK_K = 64
        BLOCK_H = 128
        grid = (B, total_seq)

        batched_matmul_cat_with_weightT_kernel[grid](
            e, h, w, out,
            B, T, I, H,
            stride_e_n, stride_e_s, stride_e_h,
            stride_h_n, stride_h_s, stride_h_h,
            stride_w_h, stride_w_k,
            stride_out_n, stride_out_s, stride_out_h,
            BLOCK_K=BLOCK_K, BLOCK_H=BLOCK_H,
            num_warps=4,
        )

        # Split outputs into encoder and hidden streams
        processed_encoder = out[:, :T, :]
        processed_hidden = out[:, T:, :]

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
