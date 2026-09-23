import torch
import triton
import triton.language as tl


# Kernel 1: batched matmul for input of shape [B, N, H] with weight^T of shape [H, H] -> output [B, N, H]
@triton.jit
def batched_matmul_T_with_weightT_kernel(
    input_ptr,  # *ptr to [B, N, H]
    weight_ptr,  # *ptr to [H, H] (process_weight)
    output_ptr,  # *ptr to [B, N, H]
    B: tl.constexpr, N: tl.constexpr, H: tl.constexpr,
    stride_input_n, stride_input_s, stride_input_h,
    stride_weight_k, stride_weight_h,  # weight is [H, H], we'll read rows by k and cols by h
    stride_output_n, stride_output_s, stride_output_h,
    BLOCK_K: tl.constexpr, BLOCK_H: tl.constexpr
):
    # One program per (batch, seq) pair
    pid = tl.program_id(0)
    n = pid // N
    s = pid % N

    # Bounds check (defensive)
    if n >= B:
        return

    # Initialize output row
    offs_h = tl.arange(0, BLOCK_H)
    acc = tl.zeros([BLOCK_H], dtype=tl.float32)

    # Loop over K (hidden dimension) in tiles
    for k0 in range(0, H, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < H

        # Load input vector input[n, s, :] as a BLOCK_K vector, masked
        input_vec = tl.load(
            input_ptr + n * stride_input_n + s * stride_input_s + offs_k * stride_input_h,
            mask=mask_k,
            other=0.0
        )
        # Load weight rows: weight[offs_k, offs_h] as [BLOCK_K, BLOCK_H]
        weight_rows = tl.load(
            weight_ptr + offs_k[:, None] * stride_weight_k + offs_h[None, :] * stride_weight_h,
            mask=mask_k[:, None],
            other=0.0
        )

        # Accumulate: acc += sum_k input_vec[k] * weight_rows[k, :]
        # Convert input_vec to [BLOCK_K, 1] to broadcast
        acc += tl.sum(weight_rows * input_vec[:, None], axis=0)

    # Store results, masked by offs_h < H
    mask_h = offs_h < H
    tl.store(output_ptr + n * stride_output_n + s * stride_output_s + offs_h * stride_output_h, acc, mask=mask_h)


# Kernel 2: compute the full concatenated result directly, without creating [B, T+I, H] input tensor.
# It reads rows from either encoder_hidden_states or hidden_states depending on s < T.
# output shape: [B, T+I, H], where out[n, s, :] = input[n, s, :] @ weight.T
@triton.jit
def batched_matmul_cat_with_weightT_kernel(
    encoder_ptr, hidden_ptr, weight_ptr, output_ptr,
    B: tl.constexpr, T: tl.constexpr, I: tl.constexpr, H: tl.constexpr,
    stride_encoder_n, stride_encoder_s, stride_encoder_h,
    stride_hidden_n, stride_hidden_s, stride_hidden_h,
    stride_weight_k, stride_weight_h,
    stride_out_n, stride_out_s, stride_out_h,
    BLOCK_K: tl.constexpr, BLOCK_H: tl.constexpr
):
    # One program per (batch, seq) pair
    pid = tl.program_id(0)
    s_total = T + I
    n = pid // s_total
    s = pid % s_total

    # Determine whether this s belongs to encoder (s < T) or hidden (s >= T)
    from_encoder = s < T

    # Bounds check
    if n >= B:
        return

    # Prepare offsets and accumulator
    offs_h = tl.arange(0, BLOCK_H)
    acc = tl.zeros([BLOCK_H], dtype=tl.float32)

    # Loop over K dimension
    for k0 in range(0, H, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < H

        if from_encoder:
            # Load input vector from encoder_hidden_states
            input_vec = tl.load(
                encoder_ptr + n * stride_encoder_n + s * stride_encoder_s + offs_k * stride_encoder_h,
                mask=mask_k,
                other=0.0
            )
        else:
            # Load input vector from hidden_states
            input_vec = tl.load(
                hidden_ptr + n * stride_hidden_n + (s - T) * stride_hidden_s + offs_k * stride_hidden_h,
                mask=mask_k,
                other=0.0
            )

        # Load weight rows: weight[offs_k, offs_h] as [BLOCK_K, BLOCK_H]
        weight_rows = tl.load(
            weight_ptr + offs_k[:, None] * stride_weight_k + offs_h[None, :] * stride_weight_h,
            mask=mask_k[:, None],
            other=0.0
        )

        # Accumulate
        acc += tl.sum(weight_rows * input_vec[:, None], axis=0)

    # Store
    mask_h = offs_h < H
    tl.store(output_ptr + n * stride_out_n + s * stride_out_s + offs_h * stride_out_h, acc, mask=mask_h)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        Triton-optimized version of the original run function.
        Computes:
          concatenated = cat([encoder_hidden_states, hidden_states], dim=1)  # [B, T+I, H]
          processed = concatenated @ process_weight.T                         # [B, T+I, H]
          return processed_encoder = processed[:, :T, :], processed_hidden = processed[:, T:, :]
        All core matmuls are performed by Triton kernels. No torch.matmul on tensors.
        """
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Inputs must be CUDA tensors for Triton."
        B = hidden_states.shape[0]
        H = hidden_states.shape[2]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        device = hidden_states.device

        # Ensure dtype is float32 for numeric stability (original code uses default float32)
        if process_weight.dtype != torch.float32:
            process_weight = process_weight.float()
        if hidden_states.dtype != torch.float32:
            hidden_states = hidden_states.float()
        if encoder_hidden_states.dtype != torch.float32:
            encoder_hidden_states = encoder_hidden_states.float()

        # We will compute processed_encoder and processed_hidden directly via Triton kernels.
        # processed_encoder: each row is input[n, s, :] @ process_weight.T for s in [0..T-1]
        processed_encoder = torch.empty((B, T, H), device=device, dtype=torch.float32)

        grid_encoder = (B * T,)
        batched_matmul_T_with_weightT_kernel[grid_encoder](
            encoder_hidden_states, process_weight, processed_encoder,
            B, T, H,
            encoder_hidden_states.stride(0), encoder_hidden_states.stride(1), encoder_hidden_states.stride(2),
            process_weight.stride(0), process_weight.stride(1),
            processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
            BLOCK_K=64, BLOCK_H=128,
            num_warps=4,
        )

        # processed_hidden: each row is input[n, s, :] @ process_weight.T for s in [0..I-1]
        processed_hidden = torch.empty((B, I, H), device=device, dtype=torch.float32)

        grid_hidden = (B * I,)
        batched_matmul_T_with_weightT_kernel[grid_hidden](
            hidden_states, process_weight, processed_hidden,
            B, I, H,
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            process_weight.stride(0), process_weight.stride(1),
            processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
            BLOCK_K=64, BLOCK_H=128,
            num_warps=4,
        )

        # We cannot directly create the concatenated result via Triton without some intermediate,
        # but since we've computed encoder and hidden streams separately, we return them as in the original.
        return processed_encoder, processed_hidden