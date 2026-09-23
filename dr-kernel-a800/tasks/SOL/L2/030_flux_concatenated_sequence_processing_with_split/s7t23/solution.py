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
    # program ids: each program handles one row of the concatenated sequence
    b = tl.program_id(0)    # batch index
    p = tl.program_id(1)    # concatenated row index in [0, T+I)

    # column indices across hidden dim
    n = tl.arange(0, H)

    # masks
    mask_b = b < B
    mask_p = p < (T + I)
    mask_n = n < H

    # decide source: encoder or image
    is_encoder = p < T

    # compute source pointers
    if is_encoder:
        e_row_ptr = e_ptr + b * stride_e_b + p * stride_e_t
    else:
        i_row_idx = p - T
        i_row_ptr = i_ptr + b * stride_i_b + i_row_idx * stride_i_i

    # destination pointer
    out_row_ptr = out_ptr + b * stride_out_b + p * stride_out_m

    # load and store
    if is_encoder:
        vals = tl.load(e_row_ptr + n * stride_e_h, mask=mask_n, other=0.0)
    else:
        vals = tl.load(i_row_ptr + n * stride_i_h, mask=mask_n, other=0.0)
    tl.store(out_row_ptr + n * stride_out_h, vals, mask=mask_n)


@triton.jit
def matmul_kernel(
    X_ptr,            # *const float, [B, M, H] where M = T + I
    W_ptr,            # *const float, [H, H]
    Y_ptr,            # *float, [B, M, H]
    B: tl.constexpr,
    M: tl.constexpr,  # sequence length after concat
    H: tl.constexpr,  # hidden dim
    stride_X_b, stride_X_m, stride_X_h,
    stride_W_k, stride_W_n,
    stride_Y_b, stride_Y_m, stride_Y_h,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # program ids: tile over output rows (M) and hidden dim (N)
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)

    # tile offsets
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    n_offsets = tl.arange(0, BLOCK_N)                    # [BLOCK_N]

    # masks
    mask_m = m_offsets < M
    mask_n = n_offsets < H

    # accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # loop over K dimension (hidden dims)
    for k_start in range(0, H, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)     # [BLOCK_K]
        mask_k = k_offsets < H

        # load X tile: X[pid_b, m_offsets, k_offsets] -> [BLOCK_M, BLOCK_K]
        x_ptrs = X_ptr + pid_b * stride_X_b + m_offsets[:, None] * stride_X_m + k_offsets[None, :] * stride_X_h
        x_mask = mask_m[:, None] & mask_k[None, :]
        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)

        # load W tile: W[k_offsets, n_offsets] -> [BLOCK_K, BLOCK_N]
        w_ptrs = W_ptr + k_offsets[:, None] * stride_W_k + n_offsets[None, :] * stride_W_n
        w_mask = mask_k[:, None] & mask_n[None, :]
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)

        # accumulate
        acc += tl.dot(x_tile, w_tile)  # [BLOCK_M, BLOCK_N]

    # store result Y[pid_b, m_offsets, n_offsets] -> [BLOCK_M, BLOCK_N]
    y_ptrs = Y_ptr + pid_b * stride_Y_b + m_offsets[:, None] * stride_Y_m + n_offsets[None, :] * stride_Y_h
    y_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(y_ptrs, acc, mask=y_mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-only implementation:
        - Concatenates encoder_hidden_states and hidden_states along sequence dimension using Triton.
        - Applies linear projection using Triton GEMM.
        - Splits output back into encoder and image streams.
        """
        # Ensure CUDA tensors
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "All tensors must be on CUDA"

        # Cast to float32 for robustness
        if hidden_states.dtype != torch.float32:
            hidden_states = hidden_states.float()
        if encoder_hidden_states.dtype != torch.float32:
            encoder_hidden_states = encoder_hidden_states.float()
        if process_weight.dtype != torch.float32:
            process_weight = process_weight.float()

        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        H = hidden_states.shape[2]
        M = T + I

        # Allocate concatenated input
        X_cat = torch.empty((B, M, H), device=hidden_states.device, dtype=torch.float32)

        # Launch concatenation kernel: grid = (B, M)
        grid_cat = (B, M)
        cat_rows_kernel[grid_cat](
            encoder_hidden_states, hidden_states, X_cat,
            B, T, I, H,
            encoder_hidden_states.stride(0), encoder_hidden_states.stride(1), encoder_hidden_states.stride(2),
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            X_cat.stride(0), X_cat.stride(1), X_cat.stride(2),
            num_warps=1, num_stages=1,
        )

        # Allocate output for linear projection
        Y = torch.empty((B, M, H), device=hidden_states.device, dtype=torch.float32)

        # Choose tile sizes (robust defaults; can be tuned)
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 64

        # Launch matmul kernel: grid = (B, ceil_div(M, BLOCK_M))
        grid_mm = (B, triton.cdiv(M, BLOCK_M))
        matmul_kernel[grid_mm](
            X_cat, process_weight, Y,
            B, M, H,
            X_cat.stride(0), X_cat.stride(1), X_cat.stride(2),
            process_weight.stride(0), process_weight.stride(1),
            Y.stride(0), Y.stride(1), Y.stride(2),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        # Split results into encoder and hidden streams
        processed_encoder = Y[:, :T, :]
        processed_hidden = Y[:, T:, :]

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
