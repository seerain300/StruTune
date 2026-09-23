import torch
import triton
import triton.language as tl


@triton.jit
def cat_seq_kernel(encoder_ptr, hidden_ptr, out_ptr,
                    B, T, I, H,
                    encoder_stride_b, encoder_stride_t, encoder_stride_h,
                    hidden_stride_b, hidden_stride_i, hidden_stride_h,
                    out_stride_b, out_stride_m, out_stride_h):
    # Grid: (B, L)
    b = tl.program_id(0)
    m = tl.program_id(1)  # 0..L-1

    # Select source based on sequence index
    is_encoder = m < T

    # Compute offsets
    if is_encoder:
        offs_t = m
        src_ptr = encoder_ptr + b * encoder_stride_b + offs_t * encoder_stride_t
    else:
        offs_i = m - T
        src_ptr = hidden_ptr + b * hidden_stride_b + offs_i * hidden_stride_i

    # Base output offset
    out_off = b * out_stride_b + m * out_stride_m

    # Vectorize over hidden dimension
    offs_h = tl.arange(0, H)
    src_vals = tl.load(src_ptr + offs_h * encoder_stride_h if is_encoder else src_ptr + offs_h * hidden_stride_h,
                       mask=offs_h < H, other=0.0)
    tl.store(out_ptr + out_off + offs_h * out_stride_h, src_vals, mask=offs_h < H)


@triton.jit
def batched_matmul_kernel(a_ptr, b_ptr, c_ptr,
                           B, M, N, K,
                           a_stride_b, a_stride_m, a_stride_k,
                           b_stride_k, b_stride_n,
                           c_stride_b, c_stride_m, c_stride_n,
                           BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    # Grid: (tiles over M, tiles over N, batch)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    b = tl.program_id(2)

    # Compute tile coordinates
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Reduction loop over K
    k0 = 0
    while k0 < K:
        k_offsets = k0 + tl.arange(0, BLOCK_K)

        # Load A tile: [BLOCK_M, BLOCK_K]
        a_ptrs = a_ptr + b * a_stride_b + m_offsets[:, None] * a_stride_m + k_offsets[None, :] * a_stride_k
        a_mask = (m_offsets[:, None] < M) & (k_offsets[None, :] < K)
        a_tile = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # Load B tile: [BLOCK_K, BLOCK_N] where B is process_weight.T -> [K, N] with K=H, N=H
        b_ptrs = b_ptr + k_offsets[:, None] * b_stride_k + n_offsets[None, :] * b_stride_n
        b_mask = (k_offsets[:, None] < K) & (n_offsets[None, :] < N)
        b_tile = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Accumulate
        acc += tl.dot(a_tile, b_tile)

        k0 += BLOCK_K

    # Store C tile
    c_ptrs = c_ptr + b * c_stride_b + m_offsets[:, None] * c_stride_m + n_offsets[None, :] * c_stride_n
    c_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(self, encoder_hidden_states: torch.Tensor, hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        Triton-optimized version of the original run function.
        Performs concatenation and linear projection entirely in Triton kernels.
        Returns processed_encoder and processed_hidden streams.
        """
        B = encoder_hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        H = encoder_hidden_states.shape[2]
        L = T + I

        # Ensure inputs are contiguous and on CUDA
        assert encoder_hidden_states.is_cuda and hidden_states.is_cuda and process_weight.is_cuda, "Inputs must be CUDA tensors"

        # Cast to float32 for numerical robustness
        enc32 = encoder_hidden_states.contiguous().to(torch.float32)
        hid32 = hidden_states.contiguous().to(torch.float32)
        w32 = process_weight.contiguous().to(torch.float32)

        # Concatenate along sequence dimension using Triton
        cat_out = torch.empty((B, L, H), device=enc32.device, dtype=torch.float32)
        grid_cat = (B, L)
        cat_seq_kernel[grid_cat](
            enc32, hid32, cat_out,
            B, T, I, H,
            enc32.stride(0), enc32.stride(1), enc32.stride(2),
            hid32.stride(0), hid32.stride(1), hid32.stride(2),
            cat_out.stride(0), cat_out.stride(1), cat_out.stride(2),
            num_warps=1, num_stages=1,
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
            B, L, H, H,  # M=L, N=H, K=H
            cat_out.stride(0), cat_out.stride(1), cat_out.stride(2),
            W_T.stride(0), W_T.stride(1),
            processed32.stride(0), processed32.stride(1), processed32.stride(2),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=3,
        )

        # Split back into separate streams
        processed_encoder = processed32[:, :T, :]
        processed_hidden = processed32[:, T:, :]

        # Return results in float32 (Triton compute dtype)
        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
