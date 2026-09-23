import torch
import triton
import triton.language as tl


@triton.jit
def cat_seq_kernel(
    encoder_ptr, hidden_ptr, out_ptr,
    B, T, I, H,
    # strides for encoder [B, T, H]
    enc_b_stride, enc_t_stride, enc_h_stride,
    # strides for hidden [B, I, H]
    hid_b_stride, hid_i_stride, hid_h_stride,
    # strides for out [B, L, H]
    out_b_stride, out_m_stride, out_h_stride,
):
    # Each program handles one (b, m) pair
    b = tl.program_id(0)
    m = tl.program_id(1)

    # If m is out of range, return (grid ensures m < T+I, but keep safe)
    if m >= (T + I):
        return

    # Decide source: encoder if m < T, else hidden at index m - T
    is_encoder = m < T

    # Vector of hidden-dim indices
    k = tl.arange(0, H)

    # Compute pointers and load
    if is_encoder:
        enc_off = b * enc_b_stride + m * enc_t_stride + k * enc_h_stride
        val = tl.load(encoder_ptr + enc_off)
    else:
        hid_off = b * hid_b_stride + (m - T) * hid_i_stride + k * hid_h_stride
        val = tl.load(hidden_ptr + hid_off)

    # Store to output at [b, m, :]
    out_off = b * out_b_stride + m * out_m_stride + k * out_h_stride
    tl.store(out_ptr + out_off, val)


@triton.jit
def batched_matmul_kernel(
    A_ptr, W_T_ptr, C_ptr,
    B, M, N, K,
    # strides for A: [B, M, K]
    A_b_stride, A_m_stride, A_k_stride,
    # strides for W_T: [K, N]
    WT_k_stride, WT_n_stride,
    # strides for C: [B, M, N]
    C_b_stride, C_m_stride, C_n_stride,
    # tile sizes
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 3D grid: (M tiles, N tiles, batch)
    m_block = tl.program_id(0)
    n_block = tl.program_id(1)
    b = tl.program_id(2)

    # Offsets for this tile
    offs_m = m_block * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = n_block * BLOCK_N + tl.arange(0, BLOCK_N)

    # Masks for bounds
    mask_m = offs_m < M
    mask_n = offs_n < N

    # Initialize accumulator in float32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension in BLOCK_K steps
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        # Load A[b, offs_m, offs_k] -> (BLOCK_M, BLOCK_K)
        A_ptrs = A_ptr + b * A_b_stride + offs_m[:, None] * A_m_stride + offs_k[None, :] * A_k_stride
        A_mask = mask_m[:, None] & mask_k[None, :]
        A_tile = tl.load(A_ptrs, mask=A_mask, other=0.0)

        # Load W_T[offs_k, offs_n] -> (BLOCK_K, BLOCK_N)
        WT_ptrs = W_T_ptr + offs_k[:, None] * WT_k_stride + offs_n[None, :] * WT_n_stride
        WT_mask = mask_k[:, None] & mask_n[None, :]
        WT_tile = tl.load(WT_ptrs, mask=WT_mask, other=0.0)

        # Accumulate
        acc += tl.dot(A_tile, WT_tile)

    # Store result to C[b, offs_m, offs_n]
    C_ptrs = C_ptr + b * C_b_stride + offs_m[:, None] * C_m_stride + offs_n[None, :] * C_n_stride
    C_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(C_ptrs, acc, mask=C_mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                encoder_hidden_states: torch.Tensor,
                process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Shapes
        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        H = hidden_states.shape[2]
        L = T + I

        # Cast inputs to float32 for robust Triton computation and ensure contiguity
        enc32 = encoder_hidden_states.contiguous().to(torch.float32)   # [B, T, H]
        hid32 = hidden_states.contiguous().to(torch.float32)           # [B, I, H]
        w32 = process_weight.contiguous().to(torch.float32)            # [H, H]

        # Output buffer for concatenation: [B, L, H], float32
        cat_out = torch.empty((B, L, H), device=enc32.device, dtype=torch.float32)

        # Launch concatenation Triton kernel: grid over (batch, sequence)
        grid_cat = (B, L)
        cat_seq_kernel[grid_cat](
            enc32, hid32, cat_out,
            B, T, I, H,
            enc32.stride(0), enc32.stride(1), enc32.stride(2),
            hid32.stride(0), hid32.stride(1), hid32.stride(2),
            cat_out.stride(0), cat_out.stride(1), cat_out.stride(2),
            num_warps=1, num_stages=2,
        )

        # Prepare W_T = process_weight.T: [H, H]
        W_T = w32.transpose(0, 1).contiguous()

        # Output buffer for processed: [B, L, H], float32
        processed32 = torch.empty((B, L, H), device=enc32.device, dtype=torch.float32)

        # Tile sizes and grid
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 32
        grid_mm = (triton.cdiv(L, BLOCK_M), triton.cdiv(H, BLOCK_N), B)

        # Launch Triton GEMM
        batched_matmul_kernel[grid_mm](
            cat_out, W_T, processed32,
            B, L, H, H,
            cat_out.stride(0), cat_out.stride(1), cat_out.stride(2),
            W_T.stride(0), W_T.stride(1),
            processed32.stride(0), processed32.stride(1), processed32.stride(2),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=3,
        )

        # Cast output back to original dtype of encoder_hidden_states
        processed_dtype = encoder_hidden_states.dtype
        processed32_out = processed32.to(processed_dtype)

        processed_encoder = processed32_out[:, :T, :]
        processed_hidden = processed32_out[:, T:, :]

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
