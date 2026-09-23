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
    BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)  # batch
    pid_row = tl.program_id(1)  # row in concatenated sequence
    pid_tile = tl.program_id(2)  # feature tile

    # total rows in concatenated stream
    total = T + I
    # determine source stream: 0 => encoder rows [0..T-1], 1 => hidden rows [T..T+I-1]
    stream = pid_row // total
    pos = pid_row % total
    if stream == 1:
        pos = pos - T

    # feature offsets for this tile
    d_offsets = pid_tile * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = d_offsets < D

    # source pointer
    if stream == 0:
        src = E_ptr + pid_b * E_b_stride + pos * E_t_stride + d_offsets * E_d_stride
    else:
        src = H_ptr + pid_b * H_b_stride + pos * H_i_stride + d_offsets * H_d_stride

    # destination pointer
    dst = C_ptr + pid_b * C_b_stride + pid_row * C_seqlen_stride + d_offsets * C_d_stride

    # load and store
    vals = tl.load(src, mask=mask_d, other=0.0)
    tl.store(dst, vals, mask=mask_d)


@triton.jit
def _batched_gemm_two_outputs_kernel(
    C_ptr,        # concatenated: [B, T+I, D]
    W_ptr,        # weight: [D, D]
    Y0_ptr,       # output for encoder stream: [B, T, D]
    Y1_ptr,       # output for hidden stream: [B, I, D]
    B, T, I, D,
    C_b_stride, C_seqlen_stride, C_d_stride,
    W_d0_stride, W_d1_stride,  # strides for W: [D, D]
    Y0_b_stride, Y0_t_stride, Y0_d_stride,
    Y1_b_stride, Y1_i_stride, Y1_d_stride,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # This kernel computes both Y0 (encoder) and Y1 (hidden) without using torch.
    # We iterate over rows (m) and output feature tiles (n), accumulating over hidden dim (k).
    # We launch a 1D grid over (B * 2) and compute per (batch, stream) inside.
    pid = tl.program_id(0)
    stream = pid // I  # stream: 0 for encoder, 1 for hidden
    b = pid % I  # b is in [0, B). Only valid when stream==0 or stream==1 depending on grid size; but we compute them per grid.
    # To handle grid size, we compute b = pid // 2 and stream = pid % 2, then guard with if b >= B.
    # Simpler: grid size must equal B*2; we launch with grid=(B,). So this branching isn't needed in forward,
    # but for safety we recompute using known launch.
    # Given that we launch grid=(B,), we set stream = 0 and b = program_id(0).
    b = pid
    # Since grid is (B,), we will not reuse this kernel in its general 2-stream form here.
    # For correctness, we implement per-stream kernels below instead.
    # Placeholder to ensure compilation; Triton will ignore unreachable code.
    pass


@triton.jit
def _batched_gemm_per_stream_kernel(
    C_ptr,        # concatenated: [B, T+I, D]
    W_ptr,        # weight: [D, D]
    Y_ptr,        # output: [B, M, D] where M=T for encoder, M=I for hidden
    B, T, I, D, M,  # M is either T or I
    C_b_stride, C_m_stride, C_d_stride,
    W_d0_stride, W_d1_stride,
    Y_b_stride, Y_m_stride, Y_d_stride,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # 2D grid: axis 0 over batch, axis 1 over M tiles
    pid_b = tl.program_id(0)
    pid_mtile = tl.program_id(1)

    # output row offsets for this tile
    m_offsets = pid_mtile * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = tl.arange(0, BLOCK_N)

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K (hidden_dim) in tiles
    for k0 in range(0, D, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)

        # Pointers for C[b, m, k] and W[k, n]
        C_ptrs = C_ptr + pid_b * C_b_stride + m_offsets[:, None] * C_m_stride + k_offsets[None, :] * C_d_stride
        W_ptrs = W_ptr + k_offsets[:, None] * W_d0_stride + n_offsets[None, :] * W_d1_stride

        # Masks for boundaries
        mask_cm = (m_offsets[:, None] < M) & (k_offsets[None, :] < D)
        mask_wn = (k_offsets[:, None] < D) & (n_offsets[None, :] < D)

        # Load tiles
        C_tile = tl.load(C_ptrs, mask=mask_cm, other=0.0)  # [BLOCK_M, BLOCK_K]
        W_tile = tl.load(W_ptrs, mask=mask_wn, other=0.0)  # [BLOCK_K, BLOCK_N]

        # Accumulate
        acc += tl.dot(C_tile, W_tile)  # [BLOCK_M, BLOCK_N]

    # Store results
    Y_ptrs = Y_ptr + pid_b * Y_b_stride + m_offsets[:, None] * Y_m_stride + n_offsets[None, :] * Y_d_stride
    mask_store = (m_offsets[:, None] < M) & (n_offsets[None, :] < D)
    tl.store(Y_ptrs, acc, mask=mask_store)


@triton.jit
def _split_copy_rows_kernel(
    Y_ptr,        # [B, T+I, D]
    OUT0_ptr,     # [B, T, D] (encoder)
    OUT1_ptr,     # [B, I, D] (hidden)
    B, T, I, D,
    Y_b_stride, Y_m_stride, Y_d_stride,
    OUT0_b_stride, OUT0_d_stride,
    OUT1_b_stride, OUT1_d_stride,
    BLOCK_D: tl.constexpr,
):
    # 3D grid over (batch, rows, feature tiles)
    pid_b = tl.program_id(0)
    pid_row = tl.program_id(1)
    pid_tile = tl.program_id(2)

    d_offsets = pid_tile * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = d_offsets < D

    if pid_row < T:
        src = Y_ptr + pid_b * Y_b_stride + pid_row * Y_m_stride + d_offsets * Y_d_stride
        dst = OUT0_ptr + pid_b * OUT0_b_stride + pid_row * OUT0_d_stride + d_offsets
        vals = tl.load(src, mask=mask_d, other=0.0)
        tl.store(dst, vals, mask=mask_d)
    else:
        src = Y_ptr + pid_b * Y_b_stride + pid_row * Y_m_stride + d_offsets * Y_d_stride
        dst_row = pid_row - T
        dst = OUT1_ptr + pid_b * OUT1_b_stride + dst_row * OUT1_d_stride + d_offsets
        vals = tl.load(src, mask=mask_d, other=0.0)
        tl.store(dst, vals, mask=mask_d)


# Main forward logic using Triton kernels only
class ModelNew(torch.nn.Module):
    def forward(self, encoder_hidden_states: torch.Tensor, hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        Triton-only implementation of:
          concatenated = cat(encoder_hidden_states, hidden_states, dim=1)
          processed = concatenated @ process_weight.T
          processed_encoder = processed[:, :T, :]
          processed_hidden = processed[:, T:, :]
        """
        # Ensure inputs are contiguous and float32 for predictable behavior
        E = encoder_hidden_states.contiguous().to(torch.float32)
        H = hidden_states.contiguous().to(torch.float32)
        W = process_weight.contiguous().to(torch.float32)

        B, T, D = E.shape
        I = H.shape[1]
        assert E.shape[2] == D and H.shape[2] == D and W.shape[0] == D and W.shape[1] == D, "Dimension mismatch"

        # 1) Concatenate streams in Triton: [B, T+I, D]
        C_total = T + I
        C = torch.empty((B, C_total, D), device=E.device, dtype=torch.float32)

        BLOCK_D = 128
        grid_concat = (B, C_total, (D + BLOCK_D - 1) // BLOCK_D)
        _concatenate_streams_kernel[grid_concat](
            E, H, C,
            B, T, I, D,
            E.stride(0), E.stride(1), E.stride(2),
            H.stride(0), H.stride(1), H.stride(2),
            C.stride(0), C.stride(1), C.stride(2),
            BLOCK_D=BLOCK_D,
            num_warps=4, num_stages=2,
        )

        # 2) Batched GEMM in Triton: compute Y = C @ W^T
        # We'll compute outputs for encoder and hidden separately using per-stream kernels.
        # Outputs
        processed_encoder = torch.empty((B, T, D), device=E.device, dtype=torch.float32)
        processed_hidden = torch.empty((B, I, D), device=E.device, dtype=torch.float32)

        # Launch per-stream kernels with grid over batch and M tiles
        # Choose tiles; typical values for D up to thousands
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 64

        # Enc stream (M=T)
        grid_enc = (B, (T + BLOCK_M - 1) // BLOCK_M)
        _batched_gemm_per_stream_kernel[grid_enc](
            C, W, processed_encoder,
            B, T, I, D, T,
            C.stride(0), C.stride(1), C.stride(2),
            W.stride(0), W.stride(1),
            processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        # Hidden stream (M=I)
        grid_hid = (B, (I + BLOCK_M - 1) // BLOCK_M)
        _batched_gemm_per_stream_kernel[grid_hid](
            C, W, processed_hidden,
            B, T, I, D, I,
            C.stride(0), C.stride(1), C.stride(2),
            W.stride(0), W.stride(1),
            processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        # 3) Split (already computed in per-stream GEMM; return results)
        return processed_encoder, processed_hidden