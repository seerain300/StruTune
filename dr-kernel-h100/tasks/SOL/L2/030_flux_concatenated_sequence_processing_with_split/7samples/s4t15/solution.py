import torch
import triton
import triton.language as tl

# Kernel: compute out[b, t, :] = input_vec @ weight.T, where input_vec is either
# encoder_hidden_states[b, t, :] (if t < T) or hidden_states[b, t - T, :].
# Grid: (B, T+I). Each program computes one output token for one batch.
@triton.jit
def matmul_token_kernel(
    input_ptr,          # *fp32 [B, (T or I), H]
    weight_ptr,         # *fp32 [H, H]
    out_ptr,            # *fp32 [B, T+I, H]
    B: tl.int32,
    I: tl.int32,        # number of image tokens (hidden_states.shape[1])
    T: tl.int32,        # number of encoder tokens (encoder_hidden_states.shape[1])
    H: tl.int32,        # hidden_dim
    INPUT_STRIDE_B: tl.int32, INPUT_STRIDE_L: tl.int32, INPUT_STRIDE_H: tl.int32,
    WEIGHT_STRIDE_K: tl.int32, WEIGHT_STRIDE_N: tl.int32,   # weight is [N=H, K=H]
    OUT_STRIDE_B: tl.int32, OUT_STRIDE_T: tl.int32, OUT_STRIDE_H: tl.int32,
    USE_ENCODER: tl.int32,   # 1 if t < T (use encoder), else 0 (use hidden)
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_t = tl.program_id(1)
    # Determine source index based on USE_ENCODER
    src = pid_t if USE_ENCODER == 1 else (pid_t - T)
    # Initialize output vector for this token
    output_vec = tl.zeros((H,), dtype=tl.float32)

    # Loop over hidden dimension in tiles
    for h_off in range(0, H, BLOCK_H):
        offs_h = h_off + tl.arange(0, BLOCK_H)
        mask_h = offs_h < H

        # Accumulator for current H tile
        acc = tl.zeros((BLOCK_H,), dtype=tl.float32)

        # Loop over K (input hidden) in tiles
        for k_off in range(0, H, BLOCK_K):
            offs_k = k_off + tl.arange(0, BLOCK_K)
            mask_k = offs_k < H

            # Load input scalar from the chosen source: input[pid_b, src, k]
            # Note: pid_t encodes which token in the output we're computing.
            # If USE_ENCODER==1, src=pid_t (0..T-1); else src=pid_t - T (T..T+I-1).
            input_index = src * INPUT_STRIDE_L + offs_k * INPUT_STRIDE_H
            input_vec_chunk = tl.load(input_ptr + pid_b * INPUT_STRIDE_B + input_index, mask=mask_k, other=0.0)  # [BLOCK_K]
            input_vec_chunk = input_vec_chunk.to(tl.float32)  # ensure fp32

            # Load weight tile [BLOCK_H, BLOCK_K]: weight[offs_h, offs_k]
            w_index = (offs_h[:, None] * WEIGHT_STRIDE_N) + (offs_k[None, :] * WEIGHT_STRIDE_K)
            w_tile = tl.load(weight_ptr + w_index, mask=mask_h[:, None] & mask_k[None, :], other=0.0)  # [BLOCK_H, BLOCK_K]
            w_tile = w_tile.to(tl.float32)

            # Accumulate: sum over K for each H in the tile
            # Broadcast w_tile [BLOCK_H, BLOCK_K] and input_vec_chunk [BLOCK_K] -> [BLOCK_H, BLOCK_K]
            acc += tl.sum(w_tile * input_vec_chunk[None, :], axis=1)

        # Add tile contribution to output vector
        output_vec = output_vec + acc

    # Store output vector for this token
    out_index = tl.arange(0, H)
    mask_out = out_index < H
    tl.store(out_ptr + pid_b * OUT_STRIDE_B + pid_t * OUT_STRIDE_T + out_index * OUT_STRIDE_H, output_vec, mask=mask_out)


def triton_compute_split(encoder_hidden_states: torch.Tensor, hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute processed_encoder and processed_hidden using Triton without materializing concatenation.
    Returns (processed_encoder: [B, T, H], processed_hidden: [B, I, H]).
    """
    assert encoder_hidden_states.is_cuda and hidden_states.is_cuda and process_weight.is_cuda, "Inputs must be on CUDA for Triton execution."
    B = encoder_hidden_states.shape[0]
    T = encoder_hidden_states.shape[1]
    I = hidden_states.shape[1]
    H = encoder_hidden_states.shape[2]
    assert hidden_states.shape[2] == H, "hidden_states and encoder_hidden_states must have the same hidden_dim."
    assert process_weight.shape[0] == H and process_weight.shape[1] == H, "process_weight must be [H, H]."
    assert process_weight.is_contiguous(), "process_weight must be contiguous."

    # Allocate outputs
    processed_encoder = torch.empty((B, T, H), device=encoder_hidden_states.device, dtype=torch.float32)
    processed_hidden = torch.empty((B, I, H), device=hidden_states.device, dtype=torch.float32)

    # Compute strides (in elements)
    # Input for encoder path: input_ptr -> encoder_hidden_states
    # Input for hidden path: input_ptr -> hidden_states
    # We will launch two grids: one for encoder tokens (pid_t in [0, T)), one for image tokens (pid_t in [T, T+I)).
    # To do that, we call this kernel twice: once with USE_ENCODER=1 and src=pid_t, once with USE_ENCODER=0 and src=pid_t - T.
    # But Triton launch grid is fixed; instead, we launch a single grid over (B, T+I) and set USE_ENCODER accordingly.

    # Launch for encoder tokens (pid_t in [0, T))
    grid = (B, T)
    matmul_token_kernel[grid](
        encoder_hidden_states, process_weight, processed_encoder,
        B, I, T, H,
        encoder_hidden_states.stride(0), encoder_hidden_states.stride(1), encoder_hidden_states.stride(2),
        process_weight.stride(1), process_weight.stride(0),  # weight strides: (N,H) stride(0)=H, (K,H) stride(1)=1 if contiguous
        processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
        1,  # USE_ENCODER=1
        BLOCK_H=64, BLOCK_K=64, num_warps=4,
    )

    # Launch for image tokens (pid_t in [T, T+I))
    grid2 = (B, I)
    matmul_token_kernel[grid2](
        hidden_states, process_weight, processed_hidden,
        B, I, T, H,
        hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
        process_weight.stride(1), process_weight.stride(0),
        processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
        0,  # USE_ENCODER=0, src = pid_t - T
        BLOCK_H=64, BLOCK_K=64, num_warps=4,
    )

    return processed_encoder, processed_hidden


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version of the original run function.
        Returns (processed_encoder_hidden_states, processed_image_hidden_states).
        """
        # Ensure inputs are on CUDA for Triton
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Inputs must be on CUDA for Triton execution."
        # Compute both streams using the Triton kernel
        processed_encoder, processed_hidden = triton_compute_split(encoder_hidden_states, hidden_states, process_weight)
        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
