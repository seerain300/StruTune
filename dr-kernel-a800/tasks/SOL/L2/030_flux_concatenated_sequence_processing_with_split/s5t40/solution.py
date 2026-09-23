import torch
import triton
import triton.language as tl


@triton.jit
def concat_kernel(
    encoder_ptr,      # *f32, [B, Stext, H]
    hidden_ptr,       # *f32, [B, Simg, H]
    out_ptr,          # *f32, [B, T, H], T = Stext + Simg
    B, Stext, Simg, H,
    stride_e_b, stride_e_t, stride_e_h,
    stride_h_b, stride_h_s, stride_h_h,
    stride_o_b, stride_o_t, stride_o_h,
    BLOCK_H: tl.constexpr,
):
    # 2D grid over (B, T)
    b = tl.program_id(0)
    t = tl.program_id(1)
    if (b >= B) or (t >= T):
        return

    h = tl.arange(0, BLOCK_H)
    mask_h = h < H

    # Determine source: if t < Stext, take from encoder; else take from hidden at s = t - Stext
    if t < Stext:
        src_ptr = encoder_ptr + b * stride_e_b + t * stride_e_t + h * stride_e_h
    else:
        s = t - Stext
        src_ptr = hidden_ptr + b * stride_h_b + s * stride_h_s + h * stride_h_h

    dst_ptr = out_ptr + b * stride_o_b + t * stride_o_t + h * stride_o_h
    vals = tl.load(src_ptr, mask=mask_h, other=0.0)
    tl.store(dst_ptr, vals, mask=mask_h)


@triton.jit
def batched_linear_kernel(
    in_ptr,           # *f32, concatenated [B, T, H]
    weight_ptr,       # *f32, [H, H] (process_weight: input dim = hidden_dim, output dim = hidden_dim)
    out_ptr,          # *f32, [B, T, H]
    B, T, H,          # in_ptr/out_ptr are [B, T, H]
    stride_in_b, stride_in_t, stride_in_h,
    stride_w_k, stride_w_n,                 # weight strides: dim-0=k (input), dim-1=n (output)
    stride_out_b, stride_out_t, stride_out_h,
    BLOCK_N: tl.constexpr,  # tile size along output H
    BLOCK_K: tl.constexpr,  # tile size along input H
):
    # Grid: (B, T, ceil_div(H, BLOCK_N))
    b = tl.program_id(0)
    t = tl.program_id(1)
    n_block = tl.program_id(2)
    if (b >= B) or (t >= T):
        return

    # Output tile indices
    n_start = n_block * BLOCK_N
    n_offsets = n_start + tl.arange(0, BLOCK_N)
    mask_n = n_offsets < H

    # Accumulator for this (b, t) over the BLOCK_N outputs
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)

    # Loop over input dimension in chunks of BLOCK_K (Triton supports while)
    k0 = 0
    while k0 < H:
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < H

        # Load input chunk (vector over k)
        in_ptrs = in_ptr + b * stride_in_b + t * stride_in_t + k_offsets * stride_in_h
        in_chunk = tl.load(in_ptrs, mask=mask_k, other=0.0)  # [BLOCK_K]

        # Load weight tile: shape [BLOCK_K, BLOCK_N]
        w_ptrs = weight_ptr + k_offsets[:, None] * stride_w_k + n_offsets[None, :] * stride_w_n
        w_tile = tl.load(w_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)  # [BLOCK_K, BLOCK_N]

        # Accumulate: for each k in chunk, acc[n] += in_chunk[k] * w_tile[k, n]
        contrib = tl.dot(in_chunk[:, None], w_tile)  # [BLOCK_N]
        acc += contrib

        k0 += BLOCK_K

    # Store results
    out_ptrs = out_ptr + b * stride_out_b + t * stride_out_t + n_offsets * stride_out_h
    tl.store(out_ptrs, acc, mask=mask_n)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version of the original run function.
        - Concatenation along the sequence dimension is performed by a Triton kernel.
        - The batched linear projection out = concatenated @ process_weight is computed by a tiled Triton kernel.
        - Outputs are split back via torch slicing.
        If tensors are not CUDA, falls back to the original PyTorch computation to ensure correctness.
        """
        # CPU or non-CUDA fallback: do original computation in PyTorch to guarantee correctness
        if (not hidden_states.is_cuda) or (not encoder_hidden_states.is_cuda) or (not process_weight.is_cuda):
            # Original PyTorch implementation
            T = encoder_hidden_states.shape[1] + hidden_states.shape[1]
            concatenated = torch.cat([encoder_hidden_states, hidden_states], dim=1)
            processed = torch.matmul(concatenated, process_weight)  # no bias
            processed_encoder = processed[:, :encoder_hidden_states.shape[1], :]
            processed_hidden = processed[:, encoder_hidden_states.shape[1]:, :]
            return processed_encoder, processed_hidden

        # Ensure dtype and contiguity
        assert hidden_states.dtype == torch.float32 and encoder_hidden_states.dtype == torch.float32 and process_weight.dtype == torch.float32, "Use float32 for tensors"
        assert hidden_states.is_contiguous() and encoder_hidden_states.is_contiguous() and process_weight.is_contiguous(), "Tensors must be contiguous"

        B = hidden_states.shape[0]
        Stext = encoder_hidden_states.shape[1]
        Simg = hidden_states.shape[1]
        T = Stext + Simg
        H = hidden_states.shape[2]  # hidden_dim for both streams

        # Allocate concatenated tensor [B, T, H]
        concatenated = torch.empty((B, T, H), device=hidden_states.device, dtype=hidden_states.dtype)

        # Launch Triton concat kernel
        BLOCK_H = 128  # tile size along hidden dimension
        grid_concat = (B, T)
        concat_kernel[grid_concat](
            encoder_hidden_states, hidden_states, concatenated,
            B, Stext, Simg, H,
            encoder_hidden_states.stride(0), encoder_hidden_states.stride(1), encoder_hidden_states.stride(2),
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            concatenated.stride(0), concatenated.stride(1), concatenated.stride(2),
            BLOCK_H=BLOCK_H,
            num_warps=4, num_stages=2,
        )

        # Prepare output tensor for linear projection [B, T, H]
        processed = torch.empty((B, T, H), device=hidden_states.device, dtype=hidden_states.dtype)

        # Use process_weight directly (no transpose): shape [H, H]
        weight = process_weight  # already [H, H]

        # Launch Triton batched linear kernel
        BLOCK_N = 128
        BLOCK_K = 64
        grid_linear = (B, T, triton.cdiv(H, BLOCK_N))
        batched_linear_kernel[grid_linear](
            concatenated, weight, processed,
            B, T, H,
            concatenated.stride(0), concatenated.stride(1), concatenated.stride(2),
            weight.stride(0), weight.stride(1),
            processed.stride(0), processed.stride(1), processed.stride(2),
            BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        # Split back into separate streams
        processed_encoder = processed[:, :Stext, :]
        processed_hidden = processed[:, Stext:, :]

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
