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
    # Program ids: over (batch, sequence)
    pid_n = tl.program_id(0)  # batch index
    pid_s = tl.program_id(1)  # sequence index in [0, T+I)

    total_seq = T + I

    # Determine source tensor: encoder for s < T, hidden for s >= T
    is_encoder = pid_s < T

    # Base pointer for input row
    if is_encoder:
        src_ptr = encoder_ptr + pid_n * stride_e_n + pid_s * stride_e_s
    else:
        src_ptr = hidden_ptr + pid_n * stride_h_n + (pid_s - T) * stride_h_s

    # Output pointer for this row
    out_row_ptr = out_ptr + pid_n * stride_out_n + pid_s * stride_out_s

    # Accumulator for output vector [H]
    acc = tl.zeros((BLOCK_H,), dtype=tl.float32)

    # Loop over K (input features) in tiles
    k0 = 0
    while k0 < H:
        ks = k0 + tl.arange(0, BLOCK_K)
        k_mask = ks < H

        # Load input vector slice [BLOCK_K] for this (n, s)
        input_vec = tl.load(src_ptr + ks * stride_e_h, mask=k_mask, other=0.0)

        # For each output feature tile h, accumulate input_vec[k] * weight[k, h]
        h0 = 0
        while h0 < H:
            hs = h0 + tl.arange(0, BLOCK_H)
            h_mask = hs < H

            # Load weight slice [BLOCK_K, BLOCK_H]: weight[ks, hs]
            # weight is [H, H], so weight_ptr + ks[:, None] * stride_w_h + hs[None, :] * stride_w_k
            weight_block = tl.load(
                weight_ptr + ks[:, None] * stride_w_h + hs[None, :] * stride_w_k,
                mask=k_mask[:, None] & h_mask[None, :],
                other=0.0,
            )
            # Accumulate: acc += sum_k input_vec[k] * weight_block[k, :]
            # Do a row-wise reduction: sum over axis=0
            acc += tl.sum(input_vec[None, :] * weight_block, axis=0)

            h0 += BLOCK_H

        k0 += BLOCK_K

    # Store the full output vector [H] for this (n, s)
    # acc is [BLOCK_H], masked by h_mask
    h0 = 0
    while h0 < H:
        hs = h0 + tl.arange(0, BLOCK_H)
        h_mask = hs < H
        # Store acc into out_row_ptr + hs * stride_out_h
        tl.store(out_row_ptr + hs * stride_out_h, acc[hs], mask=h_mask)
        h0 += BLOCK_H


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                encoder_hidden_states: torch.Tensor,
                process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized forward:
        - Concatenates encoder_hidden_states and hidden_states along sequence dim.
        - Applies linear projection via matmul with process_weight.T in Triton.
        - Splits output back into encoder and hidden streams.
        """
        # Ensure CUDA tensors and float32
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Tensors must be CUDA"
        assert hidden_states.dtype == torch.float32 and encoder_hidden_states.dtype == torch.float32 and process_weight.dtype == torch.float32, "Use float32 tensors"

        # Make tensors contiguous for simple stride arithmetic
        e = encoder_hidden_states.contiguous()  # [B, T, H]
        h = hidden_states.contiguous()          # [B, I, H]
        w = process_weight.contiguous()         # [H, H]

        B, T, H_e = e.shape
        B2, I, H_h = h.shape
        assert B == B2 and H_e == H_h, "Batch or hidden_dim mismatch"

        # Allocate output [B, T+I, H]
        out = torch.empty((B, T + I, H_e), dtype=torch.float32, device=e.device)

        # Extract strides
        stride_e_n, stride_e_s, stride_e_h = e.stride(0), e.stride(1), e.stride(2)
        stride_h_n, stride_h_s, stride_h_h = h.stride(0), h.stride(1), h.stride(2)
        stride_w_h, stride_w_k = w.stride(0), w.stride(1)  # w is [H, H], so both strides are typically 1
        stride_out_n, stride_out_s, stride_out_h = out.stride(0), out.stride(1), out.stride(2)

        # Launch Triton kernel over (B, T+I)
        BLOCK_K = 64
        BLOCK_H = 128
        grid = (B, T + I)

        batched_matmul_cat_with_weightT_kernel[grid](
            e, h, w, out,
            B, T, I, H_e,
            stride_e_n, stride_e_s, stride_e_h,
            stride_h_n, stride_h_s, stride_h_h,
            stride_w_h, stride_w_k,
            stride_out_n, stride_out_s, stride_out_h,
            BLOCK_K=BLOCK_K, BLOCK_H=BLOCK_H,
            num_warps=4,
        )

        # Split into encoder and hidden outputs
        processed_encoder = out[:, :T, :]
        processed_hidden = out[:, T:, :]

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
