import torch
import triton
import triton.language as tl


@triton.jit
def _concat_linear_kernel(
    encoder_ptr,       # *f16/f32 [B, T, H]
    hidden_ptr,        # *f16/f32 [B, I, H]
    weight_ptr,        # *f16/f32 [H, H]
    out_ptr,           # *f16/f32 [B, T+I, H]
    B: tl.int32,
    T: tl.int32,
    I: tl.int32,
    H: tl.int32,
    # input strides (elements)
    encoder_stride_b, encoder_stride_t, encoder_stride_h,
    hidden_stride_b, hidden_stride_i, hidden_stride_h,
    # weight strides (elements)
    weight_stride_h, weight_stride_k,
    # output strides (elements)
    out_stride_b, out_stride_t, out_stride_h,
    # meta-parameters
    BLOCK_H: tl.constexpr,  # tile size along hidden dimension
    BLOCK_K: tl.constexpr,  # tile size along K (loop)
):
    # Grid: (B, T+I). Each program computes one output position (b, t).
    b = tl.program_id(0)
    t = tl.program_id(1)

    # Pointer to the input vector: from encoder if t < T, else from hidden at t - T.
    use_encoder = t < T
    # Note: Triton supports scalar control flow. We branch based on use_encoder.
    if use_encoder:
        in_ptr = encoder_ptr + b * encoder_stride_b + t * encoder_stride_t
    else:
        in_idx = t - T
        in_ptr = hidden_ptr + b * hidden_stride_b + in_idx * hidden_stride_i

    # Initialize output vector accumulator (float32)
    # We process H in tiles.
    # We'll write into out_ptr via masked stores.
    # For each H tile:
    for h_off in range(0, H, BLOCK_H):
        offs_h = h_off + tl.arange(0, BLOCK_H)
        mask_h = offs_h < H

        # Accumulator for this H tile
        acc = tl.zeros((BLOCK_H,), dtype=tl.float32)

        # Loop over K in tiles
        for k_off in range(0, H, BLOCK_K):  # H is the reduction dimension (weight columns)
            offs_k = k_off + tl.arange(0, BLOCK_K)
            mask_k = offs_k < H

            # Load input vector chunk x_chunk [BLOCK_K] in original dtype, then cast to fp32 for math
            x_chunk = tl.load(in_ptr + offs_k * 0, mask=mask_k, other=0.0)  # placeholder to infer dtype
            # We need to load x_chunk from in_ptr + offs_k * hidden_stride_h (or encoder_stride_h).
            # However, we don't have separate stride for input vector; we rely on pointer arithmetic above.
            # Triton allows to load using the pointer and offsets; since in_ptr is base pointer and offs_k
            # represents contiguous index, we can load as x_chunk = tl.load(in_ptr + offs_k, mask=mask_k, other=0.0).
            x_chunk = tl.load(in_ptr + offs_k, mask=mask_k, other=0.0).to(tl.float32)

            # Load weight block [BLOCK_H, BLOCK_K] as fp32
            w_block = tl.load(
                weight_ptr + offs_h[:, None] * weight_stride_h + offs_k[None, :] * weight_stride_k,
                mask=mask_h[:, None] & mask_k[None, :],
                other=0.0,
            ).to(tl.float32)

            # Accumulate: acc += sum over K of w_block * x_chunk
            # Implement as tl.sum over axis=1
            acc += tl.sum(w_block * x_chunk[None, :], axis=1)

        # Store the accumulated acc to output
        # out[b, t, offs_h] = acc
        out_ptrs = out_ptr + b * out_stride_b + t * out_stride_t + offs_h * out_stride_h
        tl.store(out_ptrs, acc, mask=mask_h)


def triton_concat_linear(encoder_hidden_states: torch.Tensor, hidden_states: torch.Tensor, process_weight: torch.Tensor) -> torch.Tensor:
    """
    Compute concatenated = cat([encoder_hidden_states, hidden_states], dim=1) and return processed = concatenated @ process_weight.T
    entirely with Triton (no torch.cat or torch.matmul in heavy path).
    Returns tensor of shape [batch, text_seq_len + img_seq_len, hidden_dim], float32.
    """
    assert encoder_hidden_states.is_cuda and hidden_states.is_cuda and process_weight.is_cuda, "Inputs must be on CUDA for Triton execution."
    B = encoder_hidden_states.shape[0]
    T = encoder_hidden_states.shape[1]
    I = hidden_states.shape[1]
    H = encoder_hidden_states.shape[2]
    assert hidden_states.shape[0] == B and hidden_states.shape[2] == H
    assert process_weight.shape[0] == H and process_weight.shape[1] == H, "process_weight must be [hidden_dim, hidden_dim]"

    # Allocate output tensor (compute in float32 for stability). We'll return float32.
    # If you need to match input dtype, adjust final casting in caller.
    out = torch.empty((B, T + I, H), device=encoder_hidden_states.device, dtype=torch.float32)

    # Launch kernel: one program per (b, t)
    grid = (B, T + I)
    _concat_linear_kernel[grid](
        encoder_hidden_states,
        hidden_states,
        process_weight,
        out,
        B, T, I, H,
        # input strides
        encoder_hidden_states.stride(0), encoder_hidden_states.stride(1), encoder_hidden_states.stride(2),
        hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
        # weight strides
        process_weight.stride(0), process_weight.stride(1),
        # output strides
        out.stride(0), out.stride(1), out.stride(2),
        # meta-parameters
        BLOCK_H=64,  # tile over hidden dimension
        BLOCK_K=64,  # reduction tile
        num_warps=4,
        num_stages=2,
    )
    return out


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version of the original run function.
        Returns (processed_encoder_hidden_states, processed_hidden_states).
        """
        # Ensure tensors are on CUDA for Triton execution.
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Inputs must be on CUDA for Triton execution."
        total = triton_concat_linear(encoder_hidden_states, hidden_states, process_weight)  # [B, T+I, H], float32
        T = encoder_hidden_states.shape[1]
        processed_encoder = total[:, :T, :]
        processed_hidden = total[:, T:, :]
        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
