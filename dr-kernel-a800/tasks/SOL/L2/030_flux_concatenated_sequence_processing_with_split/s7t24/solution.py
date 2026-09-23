import torch
import triton
import triton.language as tl


@triton.jit
def cat_rows_kernel(
    e_ptr,            # *const float, [B, T, H]
    i_ptr,            # *const float, [B, I, H]
    out_ptr,          # *float, [B, M, H], where M = T + I
    B: tl.constexpr,
    T: tl.constexpr,
    I: tl.constexpr,
    H: tl.constexpr,
    stride_e_b, stride_e_t, stride_e_h,
    stride_i_b, stride_i_i, stride_i_h,
    stride_out_b, stride_out_m, stride_out_h,
):
    # program ids
    b = tl.program_id(0)  # batch
    p = tl.program_id(1)  # row index in concatenated sequence

    # bounds masks
    mask_b = b < B
    mask_p = p < (T + I)

    # hidden dimension indices
    n = tl.arange(0, H)
    mask_n = n < H

    # decide source: encoder or image
    is_encoder = p < T

    # pointers
    if is_encoder:
        # load from e[b, p, :]
        e_row_ptr = e_ptr + b * stride_e_b + p * stride_e_t
        src_vals = tl.load(e_row_ptr + n * stride_e_h, mask=mask_n, other=0.0)
    else:
        # load from i[b, p - T, :]
        i_row_idx = p - T
        i_row_ptr = i_ptr + b * stride_i_b + i_row_idx * stride_i_i
        src_vals = tl.load(i_row_ptr + n * stride_i_h, mask=mask_n, other=0.0)

    # store into out[b, p, :]
    out_row_ptr = out_ptr + b * stride_out_b + p * stride_out_m
    tl.store(out_row_ptr + n * stride_out_h, src_vals, mask=mask_n)


@triton.jit
def batched_matmul_kernel(
    X_ptr,            # *const float, [B, M, H]
    W_ptr,            # *const float, [H, H]
    Y_ptr,            # *float, [B, M, H]
    B: tl.constexpr,
    M: tl.constexpr,  # M = T + I
    H: tl.constexpr,
    stride_x_b, stride_x_m, stride_x_h,
    stride_w_k, stride_w_h,
    stride_y_b, stride_y_m, stride_y_h,
    BLOCK_M: tl.constexpr,  # e.g., 64 or 128
    BLOCK_N: tl.constexpr,  # e.g., 64 or 128
    BLOCK_K: tl.constexpr,  # e.g., 32 or 64
):
    # one program per batch
    b = tl.program_id(0)
    # loop over M and N tiles
    for m0 in range(0, M, BLOCK_M):
        for n0 in range(0, H, BLOCK_N):
            # accumulator
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

            # reduction over K
            for k0 in range(0, H, BLOCK_K):
                # indices for tiles
                m_idx = m0 + tl.arange(0, BLOCK_M)
                n_idx = n0 + tl.arange(0, BLOCK_N)
                k_idx = k0 + tl.arange(0, BLOCK_K)

                # mask for M, N, K boundaries
                mask_m = m_idx < M
                mask_n = n_idx < H
                mask_k = k_idx < H

                # load X[b, m_idx, k_idx] -> shape [BLOCK_M, BLOCK_K]
                x_ptrs = X_ptr + b * stride_x_b + m_idx[:, None] * stride_x_m + k_idx[None, :] * stride_x_h
                x_tile = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)

                # load W[k_idx, n_idx] -> shape [BLOCK_K, BLOCK_N]
                w_ptrs = W_ptr + k_idx[:, None] * stride_w_k + n_idx[None, :] * stride_w_h
                w_tile = tl.load(w_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)

                # accumulate
                acc += tl.dot(x_tile, w_tile)

            # store result: Y[b, m_idx, n_idx]
            y_ptrs = Y_ptr + b * stride_y_b + m_idx[:, None] * stride_y_m + n_idx[None, :] * stride_y_h
            tl.store(y_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Ensure CUDA and float32 (Triton kernels assume float32)
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Inputs must be on CUDA device"
        hidden_states = hidden_states.contiguous().float()
        encoder_hidden_states = encoder_hidden_states.contiguous().float()
        process_weight = process_weight.contiguous().float()

        B = hidden_states.shape[0]
        I = hidden_states.shape[1]
        T = encoder_hidden_states.shape[1]
        H = hidden_states.shape[2]
        M = T + I

        # Allocate concatenated matrix [B, M, H]
        X_cat = torch.empty((B, M, H), device=hidden_states.device, dtype=torch.float32)

        # Launch concatenation kernel: grid = (B, M)
        grid_cat = (B, M)
        cat_rows_kernel[grid_cat](
            encoder_hidden_states, hidden_states, X_cat,
            B, T, I, H,
            encoder_hidden_states.stride(0), encoder_hidden_states.stride(1), encoder_hidden_states.stride(2),
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            X_cat.stride(0), X_cat.stride(1), X_cat.stride(2),
            num_warps=2, num_stages=2,
        )

        # Allocate output [B, M, H]
        Y = torch.empty((B, M, H), device=hidden_states.device, dtype=torch.float32)

        # Launch batched matmul kernel: grid = (B,)
        # Tile sizes: choose based on H
        BLOCK_M = 64 if M <= 256 else 128
        BLOCK_N = 64 if H <= 256 else 128
        BLOCK_K = 64 if H <= 256 else 128

        grid_mm = (B,)
        batched_matmul_kernel[grid_mm](
            X_cat, process_weight, Y,
            B, M, H,
            X_cat.stride(0), X_cat.stride(1), X_cat.stride(2),
            process_weight.stride(0), process_weight.stride(1),
            Y.stride(0), Y.stride(1), Y.stride(2),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=3,
        )

        # Split results into encoder and image streams
        processed_encoder = Y[:, :T, :]
        processed_hidden = Y[:, T:, :]

        # Cast back to original dtype of hidden_states for consistency
        processed_encoder = processed_encoder.to(hidden_states.dtype)
        processed_hidden = processed_hidden.to(hidden_states.dtype)

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
