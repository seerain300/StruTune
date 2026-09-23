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
    # Grid: (B, T+I). Each program computes one output row (n, s).
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)

    total_seq = T + I

    # Prepare output vector for this (n, s)
    out_row_ptr = out_ptr + pid_n * stride_out_n + pid_s * stride_out_s
    # Accumulator for output vector (float32)
    acc = tl.zeros((BLOCK_H,), dtype=tl.float32)

    # Determine if this s belongs to encoder or hidden stream
    is_encoder = pid_s < T

    # Loop over K (input features) in tiles of BLOCK_K
    k0 = 0
    while k0 < H:
        # Load input vector element k
        if is_encoder:
            # encoder_hidden_states[n, s, k]
            src_ptr = encoder_ptr + pid_n * stride_e_n + pid_s * stride_e_s
            # Each element is strided by stride_e_h
            input_k = tl.load(src_ptr + (k0 + tl.arange(0, BLOCK_K)) * stride_e_h, mask=(k0 + tl.arange(0, BLOCK_K)) < H, other=0.0)
        else:
            # hidden_states[n, s - T, k]
            src_ptr = hidden_ptr + pid_n * stride_h_n + (pid_s - T) * stride_h_s
            input_k = tl.load(src_ptr + (k0 + tl.arange(0, BLOCK_K)) * stride_h_h, mask=(k0 + tl.arange(0, BLOCK_K)) < H, other=0.0)

        # Load weight slice [k, h] for all h in tile
        h_range = tl.arange(0, BLOCK_H)
        w_ptrs = weight_ptr + (k0 + h_range) * stride_w_h  # broadcasting across k: (k0 + k_tile) rows
        # We need to broadcast input_k across BLOCK_H columns
        # For each kk in k-tile, compute acc[h] += input_k[kk] * weight[kk, h]
        for kk in range(BLOCK_K):
            k_valid = (k0 + kk) < H
            # Scale factor for this kk
            scale = input_k[kk] if k_valid else 0.0
            w_ptrs_k = w_ptrs + kk * stride_w_k  # each kk moves across rows of weight
            w_vals = tl.load(w_ptrs_k + h_range * stride_w_h, mask=h_range < H, other=0.0)
            acc += w_vals * scale

        k0 += BLOCK_K

    # Store the accumulated output vector for this (n, s)
    out_ptrs = out_row_ptr + (tl.arange(0, BLOCK_H)) * stride_out_h
    tl.store(out_ptrs, acc, mask=(tl.arange(0, BLOCK_H) < H))


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        Compute:
            concatenated = cat([encoder_hidden_states, hidden_states], dim=1)  # [B, T+I, H]
            processed = concatenated @ process_weight.T  # [B, T+I, H]
            return processed[:, :T, :], processed[:, T:, :]
        Entire computation done in Triton. No torch.matmul on tensors.
        """
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "All inputs must be CUDA tensors"
        assert hidden_states.dtype == torch.float32 and encoder_hidden_states.dtype == torch.float32 and process_weight.dtype == torch.float32, "Expected float32 inputs"
        B, I, H = hidden_states.shape
        T, H2, Hw = encoder_hidden_states.shape
        assert H == H2 == Hw, "Hidden and weight feature sizes must match"

        # Ensure contiguous tensors for simple pointer arithmetic
        e = encoder_hidden_states.contiguous()
        h = hidden_states.contiguous()
        w = process_weight.contiguous()

        total_seq = T + I
        # Allocate output [B, T+I, H] in float32
        out = torch.empty((B, total_seq, H), device=hidden_states.device, dtype=torch.float32)

        # Strides
        stride_e_n, stride_e_s, stride_e_h = e.stride(0), e.stride(1), e.stride(2)
        stride_h_n, stride_h_s, stride_h_h = h.stride(0), h.stride(1), h.stride(2)
        stride_w_h, stride_w_k = w.stride(0), w.stride(1)
        stride_out_n, stride_out_s, stride_out_h = out.stride(0), out.stride(1), out.stride(2)

        # Launch Triton kernel: grid over (batch, sequence positions)
        # Use BLOCK_K=64, BLOCK_H=128; masks handle tails.
        grid = (B, total_seq)
        batched_matmul_cat_with_weightT_kernel[grid](
            e, h, w, out,
            B, T, I, H,
            stride_e_n, stride_e_s, stride_e_h,
            stride_h_n, stride_h_s, stride_h_h,
            stride_w_h, stride_w_k,
            stride_out_n, stride_out_s, stride_out_h,
            BLOCK_K=64, BLOCK_H=128,
            num_warps=4,
        )

        # Split into encoder and hidden outputs
        processed_encoder = out[:, :T, :]
        processed_hidden = out[:, T:, :]

        return processed_encoder, processed_hidden