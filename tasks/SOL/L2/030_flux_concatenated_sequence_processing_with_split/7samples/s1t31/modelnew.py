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
    C_b_stride, C_m_stride, C_d_stride,
    BLOCK_D: tl.constexpr,
):
    # 3D grid over (batch, sequence position, feature tiles)
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)  # sequence index in [0, T+I)
    pid_dn = tl.program_id(2)  # tile index along feature dimension
    m_total = T + I

    # Compute source stream: stream 0 = encoder rows, stream 1 = hidden rows
    stream = pid_m // m_total
    pos = pid_m % m_total
    if stream == 1:
        pos = pos - T

    # Feature tile range
    d_start = pid_dn * BLOCK_D
    d_offsets = d_start + tl.arange(0, BLOCK_D)
    mask_d = d_offsets < D

    # Base pointers for source and destination rows
    src_base = E_ptr + pid_b * E_b_stride + pos * E_t_stride if stream == 0 else H_ptr + pid_b * H_b_stride + pos * H_i_stride
    dst_base = C_ptr + pid_b * C_b_stride + pid_m * C_m_stride

    # Load and store feature tile
    vals = tl.load(src_base + d_offsets * E_d_stride, mask=mask_d, other=0.0)
    tl.store(dst_base + d_offsets * C_d_stride, vals, mask=mask_d)


@triton.jit
def _batched_matmul_ct_wt_kernel(
    C_ptr,        # input X: [B, M, D], M = T+I
    W_ptr,        # process_weight: [D, D], we use it as W[k, n]
    Y_ptr,        # output Y: [B, M, D]
    B, M, D,
    C_b_stride, C_m_stride, C_d_stride,
    W_k_stride, W_n_stride,  # W strides: (stride along k, stride along n)
    Y_b_stride, Y_m_stride, Y_d_stride,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Grid: (B, tiles over M, tiles over N)
    pid_b = tl.program_id(0)
    pid_tm = tl.program_id(1)  # tile over M
    pid_tn = tl.program_id(2)  # tile over N

    m_start = pid_tm * BLOCK_M
    n_start = pid_tn * BLOCK_N

    m_offsets = m_start + tl.arange(0, BLOCK_M)
    n_offsets = n_start + tl.arange(0, BLOCK_N)

    mask_m = m_offsets < M
    mask_n = n_offsets < D

    # Accumulator for this tile
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension in tiles
    for k_start in range(0, D, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < D

        # Load C_sub: [BLOCK_M, BLOCK_K] -> C[b, m, k]
        c_ptrs = C_ptr + pid_b * C_b_stride + m_offsets[:, None] * C_m_stride + k_offsets[None, :] * C_d_stride
        c_mask = mask_m[:, None] & mask_k[None, :]
        c_sub = tl.load(c_ptrs, mask=c_mask, other=0.0)

        # Load W_sub: [BLOCK_K, BLOCK_N] -> W[k, n]
        w_ptrs = W_ptr + k_offsets[:, None] * W_k_stride + n_offsets[None, :] * W_n_stride
        w_mask = mask_k[:, None] & mask_n[None, :]
        w_sub = tl.load(w_ptrs, mask=w_mask, other=0.0)

        # Accumulate
        acc += tl.dot(c_sub, w_sub)

    # Store acc to Y
    y_ptrs = Y_ptr + pid_b * Y_b_stride + m_offsets[:, None] * Y_m_stride + n_offsets[None, :] * Y_d_stride
    y_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(y_ptrs, acc, mask=y_mask)


@triton.jit
def _split_copy_kernel(
    Y_ptr,        # input Y: [B, T+I, D]
    out0_ptr,     # output0: processed_encoder [B, T, D]
    out1_ptr,     # output1: processed_hidden [B, I, D]
    B, T, I, D,
    Y_b_stride, Y_m_stride, Y_d_stride,
    out0_b_stride, out0_d_stride,
    out1_b_stride, out1_d_stride,
    BLOCK_D: tl.constexpr,
):
    # 3D grid: (batch, stream row, feature tiles)
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)  # position in [0, T+I)
    pid_dn = tl.program_id(2)  # tile along D
    m_total = T + I
    stream = pid_m // m_total
    pos = pid_m % m_total
    if stream == 1:
        pos = pos - T  # hidden stream

    d_start = pid_dn * BLOCK_D
    d_offsets = d_start + tl.arange(0, BLOCK_D)
    mask_d = d_offsets < D

    # Load from Y
    y_base = Y_ptr + pid_b * Y_b_stride + pid_m * Y_m_stride
    vals = tl.load(y_base + d_offsets * Y_d_stride, mask=mask_d, other=0.0)

    # Store to appropriate output
    if stream == 0:
        dst = out0_ptr + pid_b * out0_b_stride + pos * out0_d_stride
    else:
        dst = out1_ptr + pid_b * out1_b_stride + (pos - T) * out1_d_stride

    tl.store(dst + d_offsets * (out0_d_stride if stream == 0 else out1_d_stride), vals, mask=mask_d)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-only implementation of:
          1) Concatenate encoder_hidden_states and hidden_states along sequence dimension
          2) Apply linear projection: concatenated @ process_weight.T
          3) Split back into separate encoder and hidden streams
        Returns (processed_encoder, processed_hidden)
        """
        assert hidden_states.dim() == 3 and encoder_hidden_states.dim() == 3, "Inputs must be 3D [B, S, D]"
        assert process_weight.dim() == 2, "process_weight must be 2D [D, D]"
        assert hidden_states.shape[2] == encoder_hidden_states.shape[2] == process_weight.shape[0] == process_weight.shape[1], "D mismatch"

        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        D = hidden_states.shape[2]

        # Ensure contiguous tensors and float32 compute
        E = encoder_hidden_states.contiguous().to(torch.float32)
        H = hidden_states.contiguous().to(torch.float32)
        W = process_weight.contiguous().to(torch.float32)

        # 1) Concatenate in Triton: [B, T+I, D]
        M_total = T + I
        C = torch.empty((B, M_total, D), device=E.device, dtype=torch.float32)

        BLOCK_D = 128
        grid_concat = (B, M_total, triton.cdiv(D, BLOCK_D))
        _concatenate_streams_kernel[grid_concat](
            E, H, C,
            B, T, I, D,
            E.stride(0), E.stride(1), E.stride(2),
            H.stride(0), H.stride(1), H.stride(2),
            C.stride(0), C.stride(1), C.stride(2),
            BLOCK_D=BLOCK_D,
            num_warps=4, num_stages=2,
        )

        # 2) Compute Y = C @ W^T using Triton (no bias)
        Y = torch.empty((B, M_total, D), device=E.device, dtype=torch.float32)

        # Grid over (B, tiles over M_total, tiles over D)
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 64
        grid_gemm = (B, triton.cdiv(M_total, BLOCK_M), triton.cdiv(D, BLOCK_N))
        _batched_matmul_ct_wt_kernel[grid_gemm](
            C, W, Y,
            B, M_total, D,
            C.stride(0), C.stride(1), C.stride(2),
            W.stride(0), W.stride(1),
            Y.stride(0), Y.stride(1), Y.stride(2),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        # 3) Split outputs in Triton
        processed_encoder = torch.empty((B, T, D), device=E.device, dtype=torch.float32)
        processed_hidden = torch.empty((B, I, D), device=E.device, dtype=torch.float32)

        BLOCK_D_SPLIT = 128
        grid_split = (B, M_total, triton.cdiv(D, BLOCK_D_SPLIT))
        _split_copy_kernel[grid_split](
            Y, processed_encoder, processed_hidden,
            B, T, I, D,
            Y.stride(0), Y.stride(1), Y.stride(2),
            processed_encoder.stride(0), processed_encoder.stride(2),
            processed_hidden.stride(0), processed_hidden.stride(2),
            BLOCK_D=BLOCK_D_SPLIT,
            num_warps=4, num_stages=2,
        )

        # Cast back to original dtype if needed (original example uses float32)
        # Note: original function's inputs are float32 in typical usage; keep outputs as float32.

        return processed_encoder, processed_hidden