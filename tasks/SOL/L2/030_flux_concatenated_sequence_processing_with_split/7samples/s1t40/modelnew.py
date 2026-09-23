import torch
import triton
import triton.language as tl


@triton.jit
def _concatenate_streams_kernel(
    E_ptr,        # encoder_hidden_states: [B, T, D]
    H_ptr,        # hidden_states: [B, I, D]
    C_ptr,        # concatenated output: [B, T+I, D]
    B, T, I, D,
    E_b_stride, E_t_stride, E_d_stride,
    H_b_stride, H_i_stride, H_d_stride,
    C_b_stride, C_seqlen_stride, C_d_stride,
):
    # 2D grid: axis 0 over batch, axis 1 over position in [T+I]
    pid_b = tl.program_id(0)
    pid_pos = tl.program_id(1)
    t_total = T + I

    # Determine source stream and index
    stream = pid_pos // t_total  # 0 => encoder, 1 => hidden
    pos = pid_pos % t_total
    if stream == 1:
        pos = pos - T  # map position to hidden stream

    src_row_ptr = E_ptr + pid_b * E_b_stride + pos * E_t_stride
    dst_row_ptr = C_ptr + pid_b * C_b_stride + pid_pos * C_seqlen_stride

    # Copy a contiguous vector of length D
    # E/H and C are assumed contiguous along D (stride_d == 1), but we use strides for robustness
    for d in range(0, D):
        val = tl.load(src_row_ptr + d * E_d_stride)
        tl.store(dst_row_ptr + d * C_d_stride, val)


@triton.jit
def _batched_row_gemm_kernel(
    C_ptr,        # input: [B, M_total, D], where M_total = T + I
    W_ptr,        # process_weight: [D, D], right-multiplied
    Y_ptr,        # output: [B, M_total, D]
    B, M_total, D,
    C_b_stride, C_m_stride, C_d_stride,
    W_stride0, W_stride1,  # strides for W (rows, cols)
    Y_b_stride, Y_m_stride, Y_d_stride,
    BLOCK_N: tl.constexpr,  # tile along output feature dimension
    BLOCK_K: tl.constexpr,  # tile along input feature dimension
):
    # Grid: axis 0 over batch, axis 1 over tiles of M_total
    pid_b = tl.program_id(0)
    pid_m_tile = tl.program_id(1)
    m_start = pid_m_tile * BLOCK_N

    # We handle one output row m per program; m_start is the tile start, but since grid size is ceil(M_total / BLOCK_N),
    # each program computes one output row m for this batch. We'll compute m = m_start and rely on grid size to cover all rows.
    # However, grid dimension 1 is already set to M_total tiles. We need a mapping from pid to exact m.
    # To keep it simple and correct, we restructure grid to (B, M_total): each program handles one row m.
    # But Triton requires static grid. Therefore, we use a 2D grid with axis1 = M_total and compute m = m_start = pid_m_tile.
    # Note: We pass M_total as the grid second dimension, so pid_m_tile is exactly the row index m.
    m = m_start  # one program per row m

    # Output accumulator vector of length BLOCK_N (we will loop over N in steps of BLOCK_N)
    # We'll compute only one row per program and write it to Y.
    # Initialize accumulator in float32
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # Loop over K (input feature dimension) in tiles
    for k_start in range(0, D, BLOCK_K):
        # Load input vector x: C[b, m, k]
        # We'll iterate k within BLOCK_K; for each k, we accumulate into acc by multiplying with corresponding W[k, :]
        # To do BLOCK_K accumulation in one go, we load a vector x_vec of size BLOCK_K and a matrix W_tile of size [BLOCK_K, BLOCK_N],
        # then acc += sum(x_vec[:, None] * W_tile, axis=0).
        x_vec = tl.zeros((BLOCK_K,), dtype=tl.float32)
        # Load x_vec for k in [k_start, k_start + BLOCK_K)
        for kk in range(0, BLOCK_K):
            k_idx = k_start + kk
            # mask for k_idx >= D
            mask_k = k_idx < D
            x_vec[kk] = tl.load(C_ptr + pid_b * C_b_stride + m * C_m_stride + k_idx * C_d_stride, mask=mask_k, other=0.0)

        # Load W_tile: [BLOCK_K, BLOCK_N] = W[k_start + kk, n] for kk in BLOCK_K, n in [n_start, n_start+BLOCK_N)
        for kk in range(0, BLOCK_K):
            k_row = k_start + kk
            # For each kk, load all BLOCK_N columns of W and accumulate
            # We'll do this by outer product: acc[n] += x_vec[kk] * W[k_row, n]
            for n_start in range(0, D, BLOCK_N):
                n_vec = tl.arange(0, BLOCK_N)
                n_offsets = n_start + n_vec
                w_vec = tl.load(
                    W_ptr + k_row * W_stride0 + n_offsets * W_stride1,
                    mask=(n_offsets < D),
                    other=0.0,
                )
                acc = acc + x_vec[kk] * w_vec

    # Store the computed row into Y[b, m, :]
    # We write acc for all n in [0, D). We iterate in chunks of BLOCK_N and store masked.
    for n_start in range(0, D, BLOCK_N):
        n_vec = tl.arange(0, BLOCK_N)
        n_offsets = n_start + n_vec
        # Create a vector to store by selecting elements from acc
        # Triton requires explicit store; we reconstruct vector from acc
        vals = acc[n_vec]  # gather elements of acc corresponding to n_offsets; this is allowed in Triton
        tl.store(Y_ptr + pid_b * Y_b_stride + m * Y_m_stride + n_offsets * Y_d_stride, vals, mask=(n_offsets < D))


@triton.jit
def _split_two_streams_kernel(
    Y_ptr,        # [B, M_total, D] processed concatenated
    E_out_ptr,    # [B, T, D] output for encoder stream
    H_out_ptr,    # [B, I, D] output for hidden stream
    B, T, I, D,
    Y_b_stride, Y_m_stride, Y_d_stride,
    E_b_stride, E_d_stride,
    H_b_stride, H_d_stride,
):
    # 3D grid: axis 0 over batch, axis 1 over rows in [0, T), axis 2 over rows in [0, I)
    pid_b = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_i = tl.program_id(2)

    # Copy Y[b, pid_t, :] -> E_out[b, pid_t, :]
    src_row_ptr = Y_ptr + pid_b * Y_b_stride + pid_t * Y_m_stride
    dst_row_ptr = E_out_ptr + pid_b * E_b_stride + pid_t * E_d_stride  # E_d_stride should be 1 for contiguous D

    for d in range(0, D):
        val = tl.load(src_row_ptr + d * Y_d_stride)
        tl.store(dst_row_ptr + d * E_d_stride, val)

    # Copy Y[b, pid_t + T, :] -> H_out[b, pid_i, :]
    src_row_ptr2 = Y_ptr + pid_b * Y_b_stride + (pid_i + T) * Y_m_stride
    dst_row_ptr2 = H_out_ptr + pid_b * H_b_stride + pid_i * H_d_stride

    for d in range(0, D):
        val = tl.load(src_row_ptr2 + d * Y_d_stride)
        tl.store(dst_row_ptr2 + d * H_d_stride, val)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Ensure device and dtype compatibility
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "All tensors must be on CUDA"
        assert hidden_states.dtype == torch.float32 and encoder_hidden_states.dtype == torch.float32 and process_weight.dtype == torch.float32, "Use float32 tensors"

        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        D = hidden_states.shape[2]
        assert encoder_hidden_states.shape[2] == D and process_weight.shape[0] == D and process_weight.shape[1] == D, "Dimension mismatch"

        # Make inputs contiguous
        E = encoder_hidden_states.contiguous()
        H = hidden_states.contiguous()
        W = process_weight.contiguous()

        # 1) Concatenate streams into [B, T+I, D] using Triton
        M_total = T + I
        C = torch.empty((B, M_total, D), device=E.device, dtype=torch.float32)

        grid_concat = (B, M_total)
        _concatenate_streams_kernel[grid_concat](
            E, H, C,
            B, T, I, D,
            E.stride(0), E.stride(1), E.stride(2),
            H.stride(0), H.stride(1), H.stride(2),
            C.stride(0), C.stride(1), C.stride(2),
            num_warps=2, num_stages=2,
        )

        # 2) Compute processed = C @ W (no bias) using Triton batched row GEMM
        Y = torch.empty((B, M_total, D), device=E.device, dtype=torch.float32)

        # Choose tiles. For robustness and simplicity, set BLOCK_N and BLOCK_K to 64.
        BLOCK_N = 64
        BLOCK_K = 64

        grid = (B, M_total)
        _batched_row_gemm_kernel[grid](
            C, W, Y,
            B, M_total, D,
            C.stride(0), C.stride(1), C.stride(2),
            W.stride(0), W.stride(1),
            Y.stride(0), Y.stride(1), Y.stride(2),
            BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        # 3) Split back into encoder and hidden streams using Triton
        processed_encoder = torch.empty((B, T, D), device=E.device, dtype=torch.float32)
        processed_hidden = torch.empty((B, I, D), device=E.device, dtype=torch.float32)

        grid_split = (B, T, I)
        _split_two_streams_kernel[grid_split](
            Y, processed_encoder, processed_hidden,
            B, T, I, D,
            Y.stride(0), Y.stride(1), Y.stride(2),
            processed_encoder.stride(0), processed_encoder.stride(2),
            processed_hidden.stride(0), processed_hidden.stride(2),
            num_warps=2, num_stages=2,
        )

        return processed_encoder, processed_hidden