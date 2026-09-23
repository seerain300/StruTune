import torch
import triton
import triton.language as tl


@triton.jit
def cat_linear_kernel(
    encoder_ptr,   # *float32, [B, T, H]
    hidden_ptr,    # *float32, [B, I, H]
    weight_ptr,    # *float32, [H, H]
    out_ptr,       # *float32, [B, T+I, H]
    B: tl.int32, T: tl.int32, I: tl.int32, H: tl.int32,
    stride_e_n: tl.int32, stride_e_s: tl.int32, stride_e_h: tl.int32,
    stride_h_n: tl.int32, stride_h_s: tl.int32, stride_h_h: tl.int32,
    stride_w_h: tl.int32, stride_w_k: tl.int32,
    stride_out_n: tl.int32, stride_out_s: tl.int32, stride_out_h: tl.int32,
    total_seq: tl.int32,
    BLOCK_K: tl.constexpr, BLOCK_H: tl.constexpr,
):
    # Program ids: over batch and sequence positions
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)

    # Output row pointer
    out_row_ptr = out_ptr + pid_n * stride_out_n + pid_s * stride_out_s

    # Determine if this sequence pos belongs to encoder or hidden stream
    is_encoder = pid_s < T

    # Accumulator for the output vector (float32)
    acc = tl.zeros((BLOCK_H,), dtype=tl.float32)

    # Loop over input feature dimension (K) in tiles
    for k0 in range(0, H, BLOCK_K):
        # Load input vector element at positions k0 + offs
        if is_encoder:
            # encoder_hidden_states[n, s, k]
            src_ptr = encoder_ptr + pid_n * stride_e_n + pid_s * stride_e_s
            k_idx = k0 + tl.arange(0, BLOCK_K)
            mask_k = k_idx < H
            input_k = tl.load(src_ptr + k_idx * stride_e_h, mask=mask_k, other=0.0)
        else:
            # hidden_states[n, s - T, k]
            src_ptr = hidden_ptr + pid_n * stride_h_n + (pid_s - T) * stride_h_s
            k_idx = k0 + tl.arange(0, BLOCK_K)
            mask_k = k_idx < H
            input_k = tl.load(src_ptr + k_idx * stride_h_h, mask=mask_k, other=0.0)

        # For each output feature tile h0, accumulate dot product
        for h0 in range(0, H, BLOCK_H):
            h_idx = h0 + tl.arange(0, BLOCK_H)
            mask_h = h_idx < H

            # weight[k, h] for k in [k0:k0+BLOCK_K), h in [h0:h0+BLOCK_H)
            # Load weight slice as a matrix [BLOCK_K, BLOCK_H]
            # weight_ptr + k_idx[:, None]*stride_w_h + h_idx[None, :]*stride_w_k
            w_vals = tl.load(
                weight_ptr + k_idx[:, None] * stride_w_h + h_idx[None, :] * stride_w_k,
                mask=(mask_k[:, None] & mask_h[None, :]),
                other=0.0,
            )

            # input_k is [BLOCK_K], w_vals is [BLOCK_K, BLOCK_H]
            # Compute dot per h column: acc[h0:h0+BLOCK_H] += sum_k input_k[k] * w_vals[k, :]
            # Reduction over K axis
            prod = input_k[:, None] * w_vals  # [BLOCK_K, BLOCK_H]
            acc[h0 : h0 + BLOCK_H] += tl.sum(prod, axis=0)

    # Store the accumulated output vector
    tl.store(out_row_ptr + tl.arange(0, BLOCK_H) * stride_out_h, acc, mask=tl.arange(0, BLOCK_H) < H)


class ModelNew(torch.nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,     # [B, I, H]
        encoder_hidden_states: torch.Tensor,  # [B, T, H]
        process_weight: torch.Tensor,   # [H, H]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version of the original run function:
        - Concatenates encoder_hidden_states and hidden_states along the sequence dimension.
        - Applies linear projection with process_weight.T.
        - Splits back into processed_encoder and processed_hidden.
        The heavy computation is performed by Triton; no torch.matmul is used on tensors in forward.
        """
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Tensors must be CUDA for Triton."

        B, I, H = hidden_states.shape
        _, T, H2 = encoder_hidden_states.shape
        assert H == H2, "hidden_dim must match between inputs."

        # Ensure contiguous tensors
        e = encoder_hidden_states.contiguous()
        h = hidden_states.contiguous()
        w = process_weight.contiguous()

        # Allocate output [B, T+I, H]
        total_seq = T + I
        out = torch.empty((B, total_seq, H), dtype=torch.float32, device=hidden_states.device)

        # Strides
        stride_e_n, stride_e_s, stride_e_h = e.stride(0), e.stride(1), e.stride(2)
        stride_h_n, stride_h_s, stride_h_h = h.stride(0), h.stride(1), h.stride(2)
        stride_w_h, stride_w_k = w.stride(0), w.stride(1)  # w is [H, H]
        stride_out_n, stride_out_s, stride_out_h = out.stride(0), out.stride(1), out.stride(2)

        # Launch Triton kernel over (B, total_seq)
        BLOCK_K = 128
        BLOCK_H = 128
        grid = (B, total_seq)

        cat_linear_kernel[grid](
            e, h, w, out,
            B, T, I, H,
            stride_e_n, stride_e_s, stride_e_h,
            stride_h_n, stride_h_s, stride_h_h,
            stride_w_h, stride_w_k,
            stride_out_n, stride_out_s, stride_out_h,
            total_seq,
            BLOCK_K=BLOCK_K, BLOCK_H=BLOCK_H,
            num_warps=4,
        )

        # Split outputs
        processed_encoder = out[:, :T, :]
        processed_hidden = out[:, T:, :]

        return processed_encoder, processed_hidden