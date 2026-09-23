import torch
import triton
import triton.language as tl


@triton.jit
def compute_concat_linear_kernel(
    encoder_ptr,   # [B, T, H], float32, contiguous
    hidden_ptr,    # [B, I, H], float32, contiguous
    weight_ptr,    # [H, H], float32, contiguous
    out_ptr,       # [B, T+I, H], float32, contiguous
    B: tl.int32, T: tl.int32, I: tl.int32, H: tl.int32,
    stride_e_n: tl.int32, stride_e_s: tl.int32, stride_e_h: tl.int32,
    stride_h_n: tl.int32, stride_h_s: tl.int32, stride_h_h: tl.int32,
    stride_w_h: tl.int32, stride_w_k: tl.int32,  # note: weight is [H, H], use h as row, k as col
    stride_out_n: tl.int32, stride_out_s: tl.int32, stride_out_h: tl.int32,
    BLOCK_H: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # program ids
    pid_n = tl.program_id(0)  # batch index
    pid_s = tl.program_id(1)  # sequence index in [0, T+I)

    total_seq = T + I

    # Decide source: encoder if s < T, else hidden
    is_encoder = pid_s < T

    # Output row pointer for (n, s)
    out_row_ptr = out_ptr + pid_n * stride_out_n + pid_s * stride_out_s

    # Prepare accumulator vector for output length H
    acc = tl.zeros((BLOCK_H,), dtype=tl.float32)

    # Loop over output features (H) in tiles
    h0 = 0
    while h0 < H:
        h_idx = h0 + tl.arange(0, BLOCK_H)
        mask_h = h_idx < H

        # Compute acc[h0:h0+BLOCK_H] += input_vec[k] * weight[k, h_idx]
        # Initialize acc tile to zero
        acc_tile = tl.zeros((BLOCK_H,), dtype=tl.float32)

        k0 = 0
        while k0 < H:  # K dimension equals H
            k_idx = k0 + tl.arange(0, BLOCK_K)
            mask_k = k_idx < H

            # Load input vector element k for this (n, s)
            if is_encoder:
                src_ptr = encoder_ptr + pid_n * stride_e_n + pid_s * stride_e_s
                input_k = tl.load(src_ptr + k_idx * stride_e_h, mask=mask_k, other=0.0)
            else:
                src_ptr = hidden_ptr + pid_n * stride_h_n + (pid_s - T) * stride_h_s
                input_k = tl.load(src_ptr + k_idx * stride_h_h, mask=mask_k, other=0.0)

            # Load weight slice weight[k_idx, h_idx]
            # weight_ptr is [H, H]; row index = k_idx, col index = h_idx
            w_vals = tl.load(weight_ptr + k_idx[:, None] * stride_w_h + h_idx[None, :] * stride_w_k,
                             mask=mask_k[:, None] & mask_h[None, :], other=0.0)

            # Accumulate: acc_tile[h] += sum_k input_k[k] * w_vals[k, h]
            # Sum over k axis (size BLOCK_K)
            acc_tile += tl.sum(w_vals * input_k[None, :], axis=0)

            k0 += BLOCK_K

        # Store the accumulated acc_tile to out[n, s, h0:h0+BLOCK_H]
        tl.store(out_row_ptr + h_idx * stride_out_h, acc_tile, mask=mask_h)

        h0 += BLOCK_H


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                encoder_hidden_states: torch.Tensor,
                process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-only implementation:
        - Concatenation is handled logically inside the kernel by selecting source based on s.
        - Linear projection is computed via GEMM inside the Triton kernel.
        - Outputs are split into encoder and hidden streams.
        """

        # Ensure contiguous tensors
        e = encoder_hidden_states.contiguous()
        h = hidden_states.contiguous()
        w = process_weight.contiguous()

        # Shapes
        B, T, H = e.shape
        B2, I, H2 = h.shape
        assert B == B2 and H == H2, "Mismatched shapes for batch or hidden_dim"

        # Allocate output [B, T+I, H]
        total_seq = T + I
        out = torch.empty((B, total_seq, H), device=e.device, dtype=torch.float32).contiguous()

        # Compute strides (in elements)
        stride_e_n, stride_e_s, stride_e_h = e.stride(0), e.stride(1), e.stride(2)
        stride_h_n, stride_h_s, stride_h_h = h.stride(0), h.stride(1), h.stride(2)
        stride_w_h, stride_w_k = w.stride(0), w.stride(1)  # w is [H, H]
        stride_out_n, stride_out_s, stride_out_h = out.stride(0), out.stride(1), out.stride(2)

        # Launch Triton kernel over grid (B, T+I)
        # Tiling parameters: since H is often 128 or 256 in these workloads, these tiles work well.
        BLOCK_H = 128
        BLOCK_K = 64
        grid = (B, total_seq)

        compute_concat_linear_kernel[grid](
            e, h, w, out,
            B, T, I, H,
            stride_e_n, stride_e_s, stride_e_h,
            stride_h_n, stride_h_s, stride_h_h,
            stride_w_h, stride_w_k,
            stride_out_n, stride_out_s, stride_out_h,
            BLOCK_H=BLOCK_H, BLOCK_K=BLOCK_K,
            num_warps=4,
        )

        # Split outputs into encoder and hidden streams
        processed_encoder = out[:, :T, :]
        processed_hidden = out[:, T:, :]

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
