import torch
import triton
import triton.language as tl


@triton.jit
def cat_rows_kernel(
    e_ptr, i_ptr, out_ptr,
    B, T, I, H, M,
    stride_e_b, stride_e_t, stride_e_h,
    stride_i_b, stride_i_i, stride_i_h,
    stride_o_b, stride_o_m, stride_o_h,
):
    # Grid: (B, M)
    b = tl.program_id(0)
    p = tl.program_id(1)

    # bounds guard
    if b >= B or p >= M:
        return

    # Determine source tensor and row index
    is_encoder = p < T
    # src index along sequence dim
    src_idx = tl.where(is_encoder, p, p - T)

    # Row vectors
    # We use a simple vectorized load/store across H dimension
    offs_h = tl.arange(0, H)

    if is_encoder:
        src_ptr = e_ptr + b * stride_e_b + src_idx * stride_e_t + offs_h * stride_e_h
        dst_ptr = out_ptr + b * stride_o_b + p * stride_o_m + offs_h * stride_o_h
    else:
        src_ptr = i_ptr + b * stride_i_b + (src_idx) * stride_i_i + offs_h * stride_i_h
        dst_ptr = out_ptr + b * stride_o_b + p * stride_o_m + offs_h * stride_o_h

    vals = tl.load(src_ptr)
    tl.store(dst_ptr, vals)


@triton.jit
def batched_gemm_kernel(
    X_ptr, W_ptr, Y_ptr,
    M, H,
    stride_x_b, stride_x_m, stride_x_k,  # X is [M, H], but we'll pass as [B, M, H]
    stride_w_k, stride_w_n,
    stride_y_b, stride_y_m, stride_y_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # One program per batch
    b = tl.program_id(0)

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over N dimension in tiles
    n_tiles = tl.cdiv(H, BLOCK_N)
    for n_tile in range(0, n_tiles):
        n_start = n_tile * BLOCK_N
        n_offsets = n_start + tl.arange(0, BLOCK_N)

        # Loop over M dimension in tiles (we need to accumulate across M in this program)
        m_tiles = tl.cdiv(M, BLOCK_M)
        for m_tile in range(0, m_tiles):
            m_start = m_tile * BLOCK_M
            m_offsets = m_start + tl.arange(0, BLOCK_M)

            # Initialize acc for this tile
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

            # Loop over K dimension in tiles
            k_tiles = tl.cdiv(H, BLOCK_K)
            for k_tile in range(0, k_tiles):
                k_start = k_tile * BLOCK_K
                k_offsets = k_start + tl.arange(0, BLOCK_K)

                # A_tile: X[b, m_offsets, k_offsets] -> shape [BLOCK_M, BLOCK_K]
                a_ptrs = (
                    X_ptr
                    + b * stride_x_b
                    + m_offsets[:, None] * stride_x_m
                    + k_offsets[None, :] * stride_x_k
                )
                a_mask = (m_offsets[:, None] < M) & (k_offsets[None, :] < H)
                a = tl.load(a_ptrs, mask=a_mask, other=0.0)

                # B_tile: W[k_offsets, n_offsets] -> shape [BLOCK_K, BLOCK_N]
                b_ptrs = (
                    W_ptr
                    + k_offsets[:, None] * stride_w_k
                    + n_offsets[None, :] * stride_w_n
                )
                b_mask = (k_offsets[:, None] < H) & (n_offsets[None, :] < H)
                b = tl.load(b_ptrs, mask=b_mask, other=0.0)

                # acc += A @ B
                # Ensure dtype is float32
                acc += tl.dot(a.to(tl.float32), b.to(tl.float32))

            # Store acc to Y for this tile
            y_ptrs = (
                Y_ptr
                + b * stride_y_b
                + m_offsets[:, None] * stride_y_m
                + n_offsets[None, :] * stride_y_n
            )
            y_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < H)
            tl.store(y_ptrs, acc, mask=y_mask)


@triton.jit
def slice_copy_rows_kernel(
    src_ptr, dst_ptr,
    B, M, src_start, rows_to_copy, H,
    stride_src_b, stride_src_m, stride_src_n,
    stride_dst_b, stride_dst_m, stride_dst_n,
):
    # Each program copies one row of a batch
    b = tl.program_id(0)
    p = tl.program_id(1)
    if b >= B or p >= rows_to_copy:
        return
    src_row = src_start + p
    # Compute pointers
    src_row_ptr = src_ptr + b * stride_src_b + src_row * stride_src_m
    dst_row_ptr = dst_ptr + b * stride_dst_b + p * stride_dst_m
    offs = tl.arange(0, H)
    vals = tl.load(src_row_ptr + offs * stride_src_n)
    tl.store(dst_row_ptr + offs * stride_dst_n, vals)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        # Ensure on CUDA and contiguous
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "All tensors must be on CUDA device."
        hidden_states = hidden_states.contiguous()
        encoder_hidden_states = encoder_hidden_states.contiguous()
        process_weight = process_weight.contiguous()

        B = hidden_states.shape[0]
        I = hidden_states.shape[1]
        T = encoder_hidden_states.shape[1]
        H = hidden_states.shape[2]
        M = T + I

        # Allocate concatenated X_cat [B, M, H]
        X_cat = torch.empty((B, M, H), device=hidden_states.device, dtype=hidden_states.dtype)

        # Launch Triton concatenation kernel
        grid_cat = (B, M)
        cat_rows_kernel[grid_cat](
            encoder_hidden_states, hidden_states, X_cat,
            B, T, I, H, M,
            encoder_hidden_states.stride(0), encoder_hidden_states.stride(1), encoder_hidden_states.stride(2),
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            X_cat.stride(0), X_cat.stride(1), X_cat.stride(2),
            num_warps=4, num_stages=2,
        )

        # Allocate Y [B, M, H] and perform Triton GEMM: Y = X_cat @ process_weight
        Y = torch.empty((B, M, H), device=hidden_states.device, dtype=torch.float32)  # accumulate/store in fp32 for stability

        # Pass X_cat and process_weight as [B, M, H] and [H, H]
        # Note: Triton kernel expects pointers to tensors. We need to interpret X_cat as [B, M, H].
        # We will pass strides accordingly:
        # For X: [B, M, H] -> stride_x_b, stride_x_m, stride_x_k
        stride_x_b, stride_x_m, stride_x_k = X_cat.stride(0), X_cat.stride(1), X_cat.stride(2)
        stride_w_k, stride_w_n = process_weight.stride(0), process_weight.stride(1)
        # Y: [B, M, H] -> strides
        stride_y_b, stride_y_m, stride_y_n = Y.stride(0), Y.stride(1), Y.stride(2)

        # Choose tiles; H may vary across workloads. Use moderate sizes for robustness.
        BLOCK_M = 128
        BLOCK_N = 64
        BLOCK_K = 32

        grid_mm = (B,)
        batched_gemm_kernel[grid_mm](
            X_cat, process_weight, Y,
            M, H,
            stride_x_b, stride_x_m, stride_x_k,
            stride_w_k, stride_w_n,
            stride_y_b, stride_y_m, stride_y_n,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=3,
        )

        # Now split Y into encoder and hidden streams
        # Allocate outputs in float32 then cast to original dtype if needed
        processed_encoder = torch.empty((B, T, H), device=hidden_states.device, dtype=torch.float32)
        processed_hidden = torch.empty((B, I, H), device=hidden_states.device, dtype=torch.float32)

        # Launch slice copy kernels for each
        # For encoder: rows 0..T-1
        grid_copy1 = (B, T)
        slice_copy_rows_kernel[grid_copy1](
            Y, processed_encoder,
            B, M, src_start=0, rows_to_copy=T, H=H,
            stride_src_b=Y.stride(0), stride_src_m=Y.stride(1), stride_src_n=Y.stride(2),
            stride_dst_b=processed_encoder.stride(0), stride_dst_m=processed_encoder.stride(1), stride_dst_n=processed_encoder.stride(2),
            num_warps=1, num_stages=1,
        )

        # For hidden: rows T..M-1 -> rows_to_copy = I
        grid_copy2 = (B, I)
        slice_copy_rows_kernel[grid_copy2](
            Y, processed_hidden,
            B, M, src_start=T, rows_to_copy=I, H=H,
            stride_src_b=Y.stride(0), stride_src_m=Y.stride(1), stride_src_n=Y.stride(2),
            stride_dst_b=processed_hidden.stride(0), stride_dst_m=processed_hidden.stride(1), stride_dst_n=processed_hidden.stride(2),
            num_warps=1, num_stages=1,
        )

        # Cast outputs to original dtype (match PyTorch behavior)
        processed_encoder = processed_encoder.to(hidden_states.dtype)
        processed_hidden = processed_hidden.to(hidden_states.dtype)

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
