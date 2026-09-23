import torch
import triton
import triton.language as tl


@triton.jit
def concat_encoder_image_kernel(
    e_ptr, i_ptr, out_ptr,
    B: tl.int32, T: tl.int32, I: tl.int32, H: tl.int32,
    e_s0, e_s1, e_s2,  # strides for encoder
    i_s0, i_s1, i_s2,  # strides for image
    o_s0, o_s1, o_s2,  # strides for output
    BLOCK_l: tl.constexpr, BLOCK_h: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_l = tl.program_id(1)
    pid_h = tl.program_id(2)

    l = pid_l * BLOCK_l + tl.arange(0, BLOCK_l)  # sequence index in [0, T+I)
    h = pid_h * BLOCK_h + tl.arange(0, BLOCK_h)  # hidden dimension index

    mask_l = l < (T + I)
    mask_h = h < H
    mask = mask_l[:, None] & mask_h[None, :]

    # For each l: if l < T -> source is encoder; else -> source is image
    # Compute pointers for each source
    # Note: l is a vector, we broadcast to 2D [BLOCK_l, BLOCK_h]
    # For masked loads/stores, we use mask.

    # Load from encoder if l < T
    mask_enc = mask & (l[:, None] < T)
    # Pointer offsets for encoder: e_ptr + b*e_s0 + l*e_s1 + h*e_s2
    e_offsets = pid_b * e_s0 + l[:, None] * e_s1 + h[None, :] * e_s2
    e_vals = tl.load(e_ptr + e_offsets, mask=mask_enc, other=0.0)

    # Load from image if l >= T
    l_img = l - T
    mask_img = mask & (l[:, None] >= T)
    i_offsets = pid_b * i_s0 + l_img[:, None] * i_s1 + h[None, :] * i_s2
    i_vals = tl.load(i_ptr + i_offsets, mask=mask_img, other=0.0)

    # Select: where l < T, use e_vals; else, use i_vals
    # When neither mask is true, both loads are masked, and other=0 ensures safe write.
    out_vals = tl.where(l[:, None] < T, e_vals, i_vals)

    # Store to output: out_ptr + b*o_s0 + l*o_s1 + h*o_s2
    out_offsets = pid_b * o_s0 + l[:, None] * o_s1 + h[None, :] * o_s2
    tl.store(out_ptr + out_offsets, out_vals, mask=mask)


@triton.jit
def matmul_linear_kernel(
    A_ptr, W_ptr, C_ptr,
    B: tl.int32, T: tl.int32, I: tl.int32, H: tl.int32,
    A_s0, A_s1, A_s2,  # A strides: [B, T+I, H]
    W_s0, W_s1, W_s2,  # W strides: [H, H]
    C_s0, C_s1, C_s2,  # C strides: [B, T+I, H]
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Flatten M = B*(T+I)
    M = B * (T + I)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Masks for output tile
    mask_m = m < M
    mask_n = n < H
    mask_out = mask_m[:, None] & mask_n[None, :]

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Reduction over K = H
    for k in range(0, H, BLOCK_K):
        k_idx = k + tl.arange(0, BLOCK_K)
        mask_k = k_idx < H

        # For A: row index in [0, M), column index n, reduction index k_idx
        # Compute b, l from row index m:
        L = T + I
        b = m // L
        l = m % L

        # Pointer offsets for A: A[b, l, k_idx] -> b*A_s0 + l*A_s1 + k_idx*A_s2
        a_offsets = b[:, None] * A_s0 + l[:, None] * A_s1 + k_idx[None, :] * A_s2
        a_mask = mask_m[:, None] & mask_k[None, :]
        a_tile = tl.load(A_ptr + a_offsets, mask=a_mask, other=0.0)

        # For W^T: we want W[k_idx, n] since W is [H, H] and W^T is [H, H] too
        # Pointer offsets for W: W[k_idx, n] -> k_idx*W_s0 + n*W_s1
        w_offsets = k_idx[:, None] * W_s0 + n[None, :] * W_s1
        w_mask = mask_k[:, None] & mask_n[None, :]
        w_tile = tl.load(W_ptr + w_offsets, mask=w_mask, other=0.0)

        # Accumulate
        acc += tl.dot(a_tile, w_tile)

    # Store back to C with mask
    # Compute C offsets: C[b, l, n] -> b*C_s0 + l*C_s1 + n*C_s2
    # We need l for each m, so same as above
    c_offsets = b[:, None] * C_s0 + l[:, None] * C_s1 + n[None, :] * C_s2
    tl.store(C_ptr + c_offsets, acc, mask=mask_out)


@triton.jit
def _copy_rows_kernel(
    src_ptr, dst_ptr,
    B: tl.int32, COUNT: tl.int32, H: tl.int32,
    src_s0, src_s1, src_s2,
    dst_s0, dst_s1, dst_s2,
    ROW_START: tl.int32,
    BLOCK_l: tl.constexpr, BLOCK_h: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_l = tl.program_id(1)  # tile over rows to copy
    pid_h = tl.program_id(2)  # tile over hidden dim

    l = ROW_START + pid_l * BLOCK_l + tl.arange(0, BLOCK_l)
    h = pid_h * BLOCK_h + tl.arange(0, BLOCK_h)

    mask_l = l < (ROW_START + COUNT)
    mask_h = h < H
    mask = mask_l[:, None] & mask_h[None, :]

    # Compute base offsets per (b, l, h)
    src_offsets = pid_b * src_s0 + l[:, None] * src_s1 + h[None, :] * src_s2
    dst_offsets = pid_b * dst_s0 + l[:, None] * dst_s1 + h[None, :] * dst_s2

    vals = tl.load(src_ptr + src_offsets, mask=mask, other=0.0)
    tl.store(dst_ptr + dst_offsets, vals, mask=mask)


def _ceil_div(a, b):
    return (a + b - 1) // b


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized forward that performs:
          concatenated = cat([encoder_hidden_states, hidden_states], dim=1)
          processed = concatenated @ process_weight.T
          return processed_encoder = processed[:, :T, :], processed_hidden = processed[:, T:, :]
        """
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "All inputs must be CUDA tensors for Triton kernels."
        B, I, H = hidden_states.shape
        assert encoder_hidden_states.shape == (B, T, H), "encoder_hidden_states must have shape [batch, text_seq_len, hidden_dim]"
        assert process_weight.shape == (H, H), "process_weight must have shape [hidden_dim, hidden_dim]"

        device = hidden_states.device
        dtype = hidden_states.dtype
        # Ensure dtype is float32 for numerical stability; Triton kernels assume float32 math
        if dtype != torch.float32:
            hidden_states = hidden_states.float()
            encoder_hidden_states = encoder_hidden_states.float()
            process_weight = process_weight.float()

        T = encoder_hidden_states.shape[1]

        # 1) Concatenate encoder_hidden_states and hidden_states along sequence dimension
        concatenated = torch.empty((B, T + I, H), device=device, dtype=torch.float32)
        grid_concat = (B, _ceil_div(T + I, 64), _ceil_div(H, 64))
        concat_encoder_image_kernel[grid_concat](
            encoder_hidden_states, hidden_states, concatenated,
            B, T, I, H,
            encoder_hidden_states.stride(0), encoder_hidden_states.stride(1), encoder_hidden_states.stride(2),
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            concatenated.stride(0), concatenated.stride(1), concatenated.stride(2),
            BLOCK_l=64, BLOCK_h=64,
            num_warps=4, num_stages=2,
        )

        # 2) Matmul: processed = concatenated @ process_weight.T
        processed = torch.empty((B, T + I, H), device=device, dtype=torch.float32)
        grid_matmul = ( _ceil_div(B * (T + I), 128), _ceil_div(H, 128) )
        matmul_linear_kernel[grid_matmul](
            concatenated, process_weight, processed,
            B, T, I, H,
            concatenated.stride(0), concatenated.stride(1), concatenated.stride(2),
            process_weight.stride(0), process_weight.stride(1), process_weight.stride(2),
            processed.stride(0), processed.stride(1), processed.stride(2),
            BLOCK_M=128, BLOCK_N=128, BLOCK_K=64,
            num_warps=4, num_stages=2,
        )

        # 3) Split processed back into encoder and image streams
        processed_encoder = torch.empty((B, T, H), device=device, dtype=torch.float32)
        grid_copy_encoder = (B, _ceil_div(T, 64), _ceil_div(H, 64))
        _copy_rows_kernel[grid_copy_encoder](
            processed, processed_encoder,
            B, T, H,
            processed.stride(0), processed.stride(1), processed.stride(2),
            processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
            ROW_START=0,
            BLOCK_l=64, BLOCK_h=64,
            num_warps=4, num_stages=2,
        )

        processed_hidden = torch.empty((B, I, H), device=device, dtype=torch.float32)
        grid_copy_hidden = (B, _ceil_div(I, 64), _ceil_div(H, 64))
        _copy_rows_kernel[grid_copy_hidden](
            processed, processed_hidden,
            B, I, H,
            processed.stride(0), processed.stride(1), processed.stride(2),
            processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
            ROW_START=T,
            BLOCK_l=64, BLOCK_h=64,
            num_warps=4, num_stages=2,
        )

        # Cast back to original dtype if needed
        if dtype != torch.float32:
            processed_encoder = processed_encoder.to(dtype)
            processed_hidden = processed_hidden.to(dtype)

        return processed_encoder, processed_hidden