import torch
import triton
import triton.language as tl


@triton.jit
def cat_rows_kernel(
    e_ptr,  # *ptr to encoder_hidden_states: [B, T, H]
    i_ptr,  # *ptr to hidden_states: [B, I, H]
    y_ptr,  # *ptr to output X_cat: [B, M, H], M = T + I
    B, T, I, H,
):
    b = tl.program_id(0)  # batch index
    p = tl.program_id(1)  # concatenated row index [0, T+I)

    # Bounds check for safety (shouldn't be needed if grid matches, but keep for robustness)
    if b >= B or p >= (T + I):
        return

    # Determine source tensor
    is_encoder = p < T
    src_b = b
    src_p = p  # if encoder
    src_i = p - T  # if image (since p ranges [0, T+I))

    # We need to iterate over H to copy each element. Use a small unrolled loop over H.
    # Create a vector of indices for H dimension.
    h_idx = tl.arange(0, H)

    if is_encoder:
        # Load row from encoder_hidden_states[b, p, :]
        e_row_ptr = e_ptr + src_b * H * T + src_p * H + h_idx
        val = tl.load(e_row_ptr, mask=h_idx < H, other=0.0)
        # Store into X_cat[b, p, :]
        y_row_ptr = y_ptr + b * (T + I) * H + p * H + h_idx
        tl.store(y_row_ptr, val, mask=h_idx < H)
    else:
        # Load row from hidden_states[b, p-T, :]
        i_row_ptr = i_ptr + src_b * H * I + src_i * H + h_idx
        val = tl.load(i_row_ptr, mask=h_idx < H, other=0.0)
        # Store into X_cat[b, p, :]
        y_row_ptr = y_ptr + b * (T + I) * H + p * H + h_idx
        tl.store(y_row_ptr, val, mask=h_idx < H)


@triton.jit
def batched_matmul_kernel(
    x_ptr,  # *ptr to X_cat[b]: [M, H], contiguous
    w_ptr,  # *ptr to weight: [H, H], contiguous
    y_ptr,  # *ptr to output Y[b]: [M, H], contiguous
    M, H,
    BLOCK_N: tl.constexpr,  # tile size for N (hidden dim)
    BLOCK_K: tl.constexpr,  # tile size for K (hidden dim)
):
    # One program per batch element; we assume grid = (B,)
    # Load meta params
    # We need to iterate rows (M) and columns (N=H). Use nested loops over tiles.
    # Accumulate in fp32
    # Note: Triton range loops require compile-time constant bounds; here M and H are passed as integers.
    # We'll use while loops to cover arbitrary sizes safely.
    # Initialize accumulator
    # We'll process N in tiles of size BLOCK_N and K in tiles of size BLOCK_K.

    # Prepare row and column indices
    row_idx = tl.arange(0, BLOCK_N)  # for M rows, but we'll use while loop to iterate across M
    col_idx = tl.arange(0, BLOCK_N)  # for N=H

    # Iterate over rows (M) in chunks
    m_start = 0
    while m_start < M:
        m_offsets = m_start + tl.arange(0, BLOCK_N)
        # Mask for valid rows
        row_valid = m_offsets < M

        # Accumulator for this tile [BLOCK_N, BLOCK_N]
        acc = tl.zeros((BLOCK_N, BLOCK_N), dtype=tl.float32)

        # Iterate over K (H) in chunks
        k_start = 0
        while k_start < H:
            k_offsets = k_start + tl.arange(0, BLOCK_K)
            k_valid = k_offsets < H

            # Load A tile: X_cat[b, m_offsets, k_offsets] -> shape [BLOCK_N, BLOCK_K]
            # Address: x_ptr + b*(M*H) + m_offsets[:, None]*H + k_offsets[None, :]
            x_tile_ptr = x_ptr + m_offsets[:, None] * H + k_offsets[None, :]
            A = tl.load(x_tile_ptr, mask=row_valid[:, None] & k_valid[None, :], other=0.0).to(tl.float32)

            # Load W tile: W[k_offsets, n_offsets] -> shape [BLOCK_K, BLOCK_N]
            W_tile_ptr = w_ptr + k_offsets[:, None] * H + col_idx[None, :]
            W = tl.load(W_tile_ptr, mask=k_valid[:, None] & (col_idx[None, :] < H), other=0.0).to(tl.float32)

            # Accumulate: acc += A @ W
            acc += tl.dot(A, W)

            k_start += BLOCK_K

        # Store results to Y: y_ptr + b*(M*H) + m_offsets[:, None]*H + col_idx[None, :]
        y_tile_ptr = y_ptr + m_offsets[:, None] * H + col_idx[None, :]
        tl.store(y_tile_ptr, acc, mask=row_valid[:, None] & (col_idx[None, :] < H))

        m_start += BLOCK_N


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        # Ensure CUDA tensors and matching dtype
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Inputs must be CUDA tensors"
        B, I, H = hidden_states.shape
        assert encoder_hidden_states.shape[0] == B and encoder_hidden_states.shape[2] == H, "encoder_hidden_states shape mismatch"
        assert process_weight.shape[0] == H and process_weight.shape[1] == H, "process_weight must be [H, H]"

        # Make inputs contiguous
        e = encoder_hidden_states.contiguous()
        i = hidden_states.contiguous()
        w = process_weight.contiguous()

        # 1) Concatenate along sequence dimension using Triton (avoid torch.cat)
        M = e.shape[1] + i.shape[1]  # T + I

        # Allocate X_cat per batch: [B, M, H], contiguous
        X_cat = torch.empty((B, M, H), device=hidden_states.device, dtype=torch.float32)

        # Launch cat_rows_kernel: grid = (B, M)
        grid_cat = (B, M)
        cat_rows_kernel[grid_cat](
            e, i, X_cat,
            B, e.shape[1], i.shape[1], H,
            num_warps=1, num_stages=1,
        )

        # 2) Batched linear projection: Y[b] = X_cat[b] @ w
        # Allocate Y per batch: [B, M, H], contiguous
        Y = torch.empty((B, M, H), device=hidden_states.device, dtype=torch.float32)

        # Launch batched_matmul_kernel: one program per batch
        grid_mm = (B,)
        # Choose tiles; for typical H up to 1024, 128 works well.
        batched_matmul_kernel[grid_mm](
            X_cat, w, Y,
            M, H,
            BLOCK_N=128, BLOCK_K=64,
            num_warps=4, num_stages=2,
        )

        # 3) Split results: processed_encoder = Y[:, :T, :], processed_hidden = Y[:, T:, :]
        # Return in original dtype
        processed_encoder = Y[:, :e.shape[1], :].to(hidden_states.dtype)
        processed_hidden = Y[:, e.shape[1]:, :].to(hidden_states.dtype)

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
