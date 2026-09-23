import torch
import triton
import triton.language as tl

@triton.jit
def _concatenate_streams_kernel(
    E_ptr,        # [B, T, D]
    H_ptr,        # [B, I, D]
    C_ptr,        # [B, T+I, D]
    B, T, I, D,
    E_b_stride, E_t_stride, E_d_stride,
    H_b_stride, H_i_stride, H_d_stride,
    C_b_stride, C_seqlen_stride, C_d_stride,
):
    # 2D grid over (batch, position in T+I)
    pid_b = tl.program_id(0)
    pid_pos = tl.program_id(1)
    t_total = T + I
    # Determine source stream and index
    stream = pid_pos // t_total
    pos = pid_pos % t_total
    if stream == 1:  # hidden stream
        pos = pos - T

    # Compute source and destination pointers
    if stream == 0:
        src_ptr = E_ptr + pid_b * E_b_stride + pos * E_t_stride
    else:
        src_ptr = H_ptr + pid_b * H_b_stride + pos * H_i_stride
    dst_ptr = C_ptr + pid_b * C_b_stride + pid_pos * C_seqlen_stride

    # Copy vector of length D (assumes D is small and loop is fine; use masks for safety)
    # Since D is typically 256/512/1024, a simple loop is fine.
    for d in range(0, D):
        val = tl.load(src_ptr + d * E_d_stride)  # E_d_stride == H_d_stride == C_d_stride == 1 for contiguous
        tl.store(dst_ptr + d * C_d_stride, val)


@triton.jit
def _gemm_concat_batched_kernel(
    C_ptr,        # [B, T+I, D] (concatenated input)
    Wt_ptr,       # [D, D] (process_weight transposed)
    Y_ptr,        # [B, T+I, D] (output processed)
    B, M, N, K,   # M = T+I, N = D, K = D
    C_b_stride, C_m_stride, C_k_stride,
    Wt_k_stride, Wt_n_stride,
    Y_b_stride, Y_m_stride, Y_n_stride,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Grid: (B, tiles over M rows)
    pid_b = tl.program_id(0)
    pid_m_tile = tl.program_id(1)
    m_start = pid_m_tile * BLOCK_M

    # Create column offsets
    offs_n = tl.arange(0, BLOCK_N)  # output columns
    offs_k = tl.arange(0, BLOCK_K)  # reduction dimension

    # Accumulator for this program (BLOCK_M x BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension in tiles
    for k_start in range(0, K, BLOCK_K):
        # Compute M indices for this program
        m_offsets = m_start + tl.arange(0, BLOCK_M)
        m_mask = m_offsets < M

        # Load C tile: [BLOCK_M, BLOCK_K]
        c_ptrs = C_ptr + pid_b * C_b_stride + m_offsets[:, None] * C_m_stride + (k_start + offs_k[None, :]) * C_k_stride
        c_mask = m_mask[:, None] & ( (k_start + offs_k[None, :]) < K )
        C_tile = tl.load(c_ptrs, mask=c_mask, other=0.0)

        # Load W^T tile: [BLOCK_K, BLOCK_N]
        wt_ptrs = Wt_ptr + (k_start + offs_k[:, None]) * Wt_k_stride + offs_n[None, :] * Wt_n_stride
        wt_mask = ( (k_start + offs_k[:, None]) < K ) & (offs_n[None, :] < N)
        WT_tile = tl.load(wt_ptrs, mask=wt_mask, other=0.0)

        # Accumulate
        acc += tl.dot(C_tile, WT_tile)

    # Write back result tile to Y
    y_ptrs = Y_ptr + pid_b * Y_b_stride + m_offsets[:, None] * Y_m_stride + offs_n[None, :] * Y_n_stride
    y_mask = m_mask[:, None] & (offs_n[None, :] < N)
    tl.store(y_ptrs, acc, mask=y_mask)


@triton.jit
def _split_streams_kernel(
    Y_ptr,        # [B, T+I, D]
    Y0_ptr,       # [B, T, D]
    Y1_ptr,       # [B, I, D]
    B, T, I, D,
    Y_b_stride, Y_m_stride, Y_d_stride,
    Y0_b_stride, Y0_d_stride,
    Y1_b_stride, Y1_d_stride,
):
    # 2D grid over (batch, rows) for each stream
    pid_b = tl.program_id(0)
    pid_row = tl.program_id(1)

    # Stream selection: 0 for encoder, 1 for hidden
    stream = pid_row // I  # either 0 or 1 depending on grid; here we pass T and I appropriately
    pos = pid_row % I

    # If stream 0 (encoder), pid_row in [0, T)
    if stream == 0:
        src_ptr = Y_ptr + pid_b * Y_b_stride + pos * Y_m_stride
    else:  # stream == 1 (hidden)
        src_ptr = Y_ptr + pid_b * Y_b_stride + (T + pos) * Y_m_stride
    # destination pointers
    if stream == 0:
        dst_ptr = Y0_ptr + pid_b * Y0_b_stride + pos * Y0_d_stride
    else:
        dst_ptr = Y1_ptr + pid_b * Y1_b_stride + pos * Y1_d_stride

    # Copy D elements
    for d in range(0, D):
        val = tl.load(src_ptr + d * Y_d_stride)
        tl.store(dst_ptr + d * Y0_d_stride, val)  # use Y0_d_stride for both, since D is same


def _triton_forward(hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # Ensure contiguity and dtype
    device = hidden_states.device
    dtype = torch.float32

    E = encoder_hidden_states.contiguous().to(dtype)
    H = hidden_states.contiguous().to(dtype)
    W = process_weight.contiguous().to(dtype)  # [D, D]

    B, T, D = E.shape
    B2, I, D2 = H.shape
    assert B == B2 and D == D2, "Batch or hidden_dim mismatch"

    # 1) Concatenate streams in Triton: [B, T+I, D]
    C_total = T + I
    C = torch.empty((B, C_total, D), device=device, dtype=dtype)
    grid_concat = (B, C_total)
    _concatenate_streams_kernel[grid_concat](
        E, H, C,
        B, T, I, D,
        E.stride(0), E.stride(1), E.stride(2),
        H.stride(0), H.stride(1), H.stride(2),
        C.stride(0), C.stride(1), C.stride(2),
        num_warps=4, num_stages=2,
    )

    # 2) Compute processed = C @ W^T using Triton GEMM: [B, T+I, D]
    Wt = W.transpose(0, 1).contiguous()  # [D, D]
    Y = torch.empty((B, C_total, D), device=device, dtype=dtype)

    BLOCK_M = 64   # rows tile
    BLOCK_N = 64   # columns tile
    BLOCK_K = 64   # reduction tile
    grid_gemm = (B, triton.cdiv(C_total, BLOCK_M))
    _gemm_concat_batched_kernel[grid_gemm](
        C, Wt, Y,
        B, C_total, D, D,
        C.stride(0), C.stride(1), C.stride(2),
        Wt.stride(0), Wt.stride(1),
        Y.stride(0), Y.stride(1), Y.stride(2),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=2,
    )

    # 3) Split back in Triton
    processed_encoder = torch.empty((B, T, D), device=device, dtype=dtype)
    processed_hidden = torch.empty((B, I, D), device=device, dtype=dtype)
    grid_split = (B, T)
    _split_streams_kernel[grid_split](
        Y, processed_encoder, processed_hidden,
        B, T, I, D,
        Y.stride(0), Y.stride(1), Y.stride(2),
        processed_encoder.stride(0), processed_encoder.stride(2),
        processed_hidden.stride(0), processed_hidden.stride(2),
        num_warps=4, num_stages=2,
    )

    return processed_encoder, processed_hidden


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return _triton_forward(hidden_states, encoder_hidden_states, process_weight)