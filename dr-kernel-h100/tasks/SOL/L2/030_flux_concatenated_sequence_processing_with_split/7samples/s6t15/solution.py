import torch
import triton
import triton.language as tl


@triton.jit
def batched_matmul_cat_weightT_tileH_kernel(
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
    # 3D grid: (batch, seq_pos, h_tile)
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)  # sequence position in [0, T+I-1]
    pid_ht = tl.program_id(2)  # tile index along H

    total_seq = T + I

    # Compute H tile range
    h_start = pid_ht * BLOCK_H
    h_offsets = h_start + tl.arange(0, BLOCK_H)
    h_mask = h_offsets < H

    # Determine if this sequence pos belongs to encoder or hidden stream
    is_encoder = pid_s < T

    # Accumulator for output tile (BLOCK_H elements)
    acc = tl.zeros((BLOCK_H,), dtype=tl.float32)

    # Loop over K (input features) in tiles of BLOCK_K
    k0 = 0
    while k0 < H:
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < H

        # Load input vector elements for this (n, s)
        if is_encoder:
            # encoder_hidden_states[n, s, k]
            src_ptr = encoder_ptr + pid_n * stride_e_n + pid_s * stride_e_s
            input_vec = tl.load(
                src_ptr + k_offsets * stride_e_h,
                mask=k_mask,
                other=0.0,
            )
        else:
            # hidden_states[n, s - T, k]
            src_ptr = hidden_ptr + pid_n * stride_h_n + (pid_s - T) * stride_h_s
            input_vec = tl.load(
                src_ptr + k_offsets * stride_h_h,
                mask=k_mask,
                other=0.0,
            )

        # Load corresponding weight slice: weight[k_offsets, h_offsets]
        weight_slice_ptr = weight_ptr + k_offsets[:, None] * stride_w_h + h_offsets[None, :] * stride_w_k
        weight_mask = (k_mask[:, None]) & (h_mask[None, :])
        w_tile = tl.load(weight_slice_ptr, mask=weight_mask, other=0.0)  # shape [BLOCK_K, BLOCK_H]

        # Accumulate: acc += sum_k input_vec[k] * w_tile[k, :]
        # Triton doesn't have a direct matmul intrinsic in this context, so we reduce over K manually.
        # Cast to float32 for accumulation
        acc += tl.sum(w_tile * input_vec[:, None], axis=0)  # sum over axis=0 (K tile)

        k0 += BLOCK_K

    # Store the accumulated tile to output
    out_row_ptr = out_ptr + pid_n * stride_out_n + pid_s * stride_out_s
    tl.store(out_row_ptr + h_offsets * stride_out_h, acc, mask=h_mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version of the original run function:
        - Concatenate encoder_hidden_states and hidden_states along sequence dim
        - Apply linear projection with process_weight.T
        - Split back into encoder and hidden outputs
        """
        # Ensure inputs are on CUDA and contiguous
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Inputs must be CUDA tensors"
        e = encoder_hidden_states.contiguous()
        h = hidden_states.contiguous()
        w = process_weight.contiguous()

        B, T, H = e.shape
        _, I, H2 = h.shape
        assert H == H2, "hidden_dim must match between encoder and image inputs"
        assert w.shape == (H, H), "process_weight must have shape [hidden_dim, hidden_dim]"

        # Allocate output
        total_seq = T + I
        out = torch.empty((B, total_seq, H), dtype=torch.float32, device=hidden_states.device)

        # Strides (in elements)
        stride_e_n, stride_e_s, stride_e_h = e.stride(0), e.stride(1), e.stride(2)
        stride_h_n, stride_h_s, stride_h_h = h.stride(0), h.stride(1), h.stride(2)
        stride_w_h, stride_w_k = w.stride(0), w.stride(1)
        stride_out_n, stride_out_s, stride_out_h = out.stride(0), out.stride(1), out.stride(2)

        # Choose tile sizes; for generality, use 128 for H tile and 64 for K tile
        BLOCK_K = 64
        BLOCK_H = 128

        # Launch Triton kernel with 3D grid
        grid = (B, total_seq, triton.cdiv(H, BLOCK_H))

        batched_matmul_cat_weightT_tileH_kernel[grid](
            e, h, w, out,
            B, T, I, H,
            stride_e_n, stride_e_s, stride_e_h,
            stride_h_n, stride_h_s, stride_h_h,
            stride_w_h, stride_w_k,
            stride_out_n, stride_out_s, stride_out_h,
            BLOCK_K=BLOCK_K, BLOCK_H=BLOCK_H,
            num_warps=4,
        )

        # Split outputs
        processed_encoder = out[:, :T, :]
        processed_hidden = out[:, T:, :]

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
