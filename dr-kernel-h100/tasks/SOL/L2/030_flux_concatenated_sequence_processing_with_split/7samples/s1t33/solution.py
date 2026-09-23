import torch
import triton
import triton.language as tl


# Triton kernel: concatenate encoder_hidden_states [B, T, D] and hidden_states [B, I, D]
# into concatenated [B, T+I, D], copying rows into output at positions [0:T) and [T:T+I).
@triton.jit
def _concatenate_streams_kernel(
    E_ptr,  # encoder [B, T, D]
    H_ptr,  # hidden [B, I, D]
    C_ptr,  # concatenated [B, T+I, D]
    B, T, I, D,
    E_b_stride, E_t_stride, E_d_stride,
    H_b_stride, H_i_stride, H_d_stride,
    C_b_stride, C_seq_stride, C_d_stride,
):
    pid_b = tl.program_id(0)
    pid_pos = tl.program_id(1)  # position in [0, T+I)
    t_total = T + I
    stream = pid_pos // t_total
    pos = pid_pos % t_total
    if stream == 1:
        pos = pos - T  # map hidden stream positions

    if stream == 0:
        src = E_ptr + pid_b * E_b_stride + pos * E_t_stride
    else:
        src = H_ptr + pid_b * H_b_stride + pos * H_i_stride

    dst = C_ptr + pid_b * C_b_stride + pid_pos * C_seq_stride

    # Copy vector of length D (assumed contiguous in last dim)
    for d in range(0, D):
        val = tl.load(src + d * E_d_stride)
        tl.store(dst + d * C_d_stride, val)


# Triton kernel: batched matmul without bias
# Computes Y[b, m, n] = sum_k C[b, m, k] * W[k, n], for all b in [0, B), m in [0, M), n in [0, N)
# Inputs:
#   C: [B, M, D] (concatenated stream)
#   W: [D, D] (process_weight, here we use its transposed logical access via strides)
# Outputs:
#   Y: [B, M, D] (result of C @ W^T)
# We use a 3D grid: (B, tiles over M, tiles over N).
@triton.jit
def _batched_gemm_kernel(
    C_ptr,  # [B, M, D]
    W_ptr,  # [D, D], process_weight
    Y_ptr,  # [B, M, D]
    B, M, D,
    C_b_stride, C_m_stride, C_d_stride,
    W_d_stride0, W_d_stride1,  # strides for [D, D], row and col
    Y_b_stride, Y_m_stride, Y_d_stride,
    BLOCK_M: tl.constexpr,  # tile along sequence rows
    BLOCK_N: tl.constexpr,  # tile along output columns
    BLOCK_K: tl.constexpr,  # tile along reduction dimension
):
    # Program ids
    pid_b = tl.program_id(0)  # batch
    pid_m = tl.program_id(1)  # tile along sequence rows
    pid_n = tl.program_id(2)  # tile along output columns

    # Compute tile offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Pointers for Y tile
    y_ptrs = Y_ptr + pid_b * Y_b_stride + offs_m[:, None] * Y_m_stride + offs_n[None, :] * Y_d_stride

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension (D) in tiles
    for k0 in range(0, D, BLOCK_K):
        k = k0 + offs_k  # [BLOCK_K]

        # Load C tile: shape [BLOCK_M, BLOCK_K]
        c_ptrs = C_ptr + pid_b * C_b_stride + offs_m[:, None] * C_m_stride + k[None, :] * C_d_stride
        c_mask = (offs_m[:, None] < M) & (k[None, :] < D)
        c_tile = tl.load(c_ptrs, mask=c_mask, other=0.0)

        # Load W tile: shape [BLOCK_K, BLOCK_N], W is [D, D]
        w_ptrs = W_ptr + k[:, None] * W_d_stride0 + offs_n[None, :] * W_d_stride1
        w_mask = (k[:, None] < D) & (offs_n[None, :] < D)
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)

        # Accumulate
        acc += tl.dot(c_tile, w_tile)

    # Store results
    y_mask = (offs_m[:, None] < M) & (offs_n[None, :] < D)
    tl.store(y_ptrs, acc, mask=y_mask)


# Triton kernel: split Y [B, T+I, D] into Y0 [B, T, D] and Y1 [B, I, D]
@triton.jit
def _split_kernel(
    Y_ptr,  # [B, T+I, D]
    Y0_ptr, # [B, T, D]
    Y1_ptr, # [B, I, D]
    B, T, I, D,
    Y_b_stride, Y_s_stride, Y_d_stride,
    Y0_b_stride, Y0_t_stride, Y0_d_stride,
    Y1_b_stride, Y1_i_stride, Y1_d_stride,
):
    pid_b = tl.program_id(0)
    pid_row = tl.program_id(1)  # 0..T-1 for Y0, or 0..I-1 for Y1
    stream = pid_row // (T + I)
    pos = pid_row % (T + I)
    if stream == 1:
        pos = pos - T

    # Source row in Y
    src_row = pid_b * Y_b_stride + pos * Y_s_stride

    # Destination pointers
    if stream == 0:
        dst_row = Y0_ptr + pid_b * Y0_b_stride + pid_row * Y0_t_stride
    else:
        dst_row = Y1_ptr + pid_b * Y1_b_stride + pid_row * Y1_i_stride

    # Copy vector of length D
    for d in range(0, D):
        val = tl.load(Y_ptr + src_row + d * Y_d_stride)
        tl.store(dst_row + d * (Y0_d_stride if stream == 0 else Y1_d_stride), val)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        Triton-optimized forward:
        - Concatenate encoder_hidden_states and hidden_states along sequence dimension in Triton.
        - Compute processed = concatenated @ process_weight.T in Triton (batched GEMM).
        - Split result back into separate encoder and hidden outputs in Triton.
        """
        assert hidden_states.dim() == 3 and encoder_hidden_states.dim() == 3, "Inputs must be [B, seq_len, D]"
        assert process_weight.dim() == 2, "process_weight must be [D, D]"

        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        D = hidden_states.shape[2]
        assert encoder_hidden_states.shape[0] == B and process_weight.shape[1] == D and process_weight.shape[0] == D, "Shape mismatch"

        # Ensure contiguous tensors (simplifies stride handling in Triton kernels)
        E = encoder_hidden_states.contiguous()
        H = hidden_states.contiguous()
        W = process_weight.contiguous()  # [D, D]

        # 1) Concatenate in Triton: [B, T+I, D]
        C_total = T + I
        C = torch.empty((B, C_total, D), device=E.device, dtype=torch.float32)

        grid_concat = (B, C_total)
        _concatenate_streams_kernel[grid_concat](
            E, H, C,
            B, T, I, D,
            E.stride(0), E.stride(1), E.stride(2),
            H.stride(0), H.stride(1), H.stride(2),
            C.stride(0), C.stride(1), C.stride(2),
            num_warps=4, num_stages=2,
        )

        # 2) Compute processed = C @ W^T in Triton (batched GEMM without bias)
        # Choose tiles. BLOCK sizes should divide typical D (e.g., 256, 512, 1024), but masks handle tails.
        BLOCK_M = 64  # rows per program
        BLOCK_N = 64  # output columns per program
        BLOCK_K = 64  # reduction tile

        M_total = C.shape[1]  # T + I

        processed = torch.empty((B, M_total, D), device=C.device, dtype=torch.float32)

        grid_gemm = (B, triton.cdiv(M_total, BLOCK_M), triton.cdiv(D, BLOCK_N))
        _batched_gemm_kernel[grid_gemm](
            C, W, processed,
            B, M_total, D,
            C.stride(0), C.stride(1), C.stride(2),
            W.stride(0), W.stride(1),  # W is [D, D], strides for row/col
            processed.stride(0), processed.stride(1), processed.stride(2),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        # 3) Split back into encoder and hidden parts using Triton
        processed_encoder = torch.empty((B, T, D), device=C.device, dtype=torch.float32)
        processed_hidden = torch.empty((B, I, D), device=C.device, dtype=torch.float32)

        grid_split = (B, T)  # stream id = 0 for encoder
        _split_kernel[grid_split](
            processed, processed_encoder, processed_hidden,
            B, T, I, D,
            processed.stride(0), processed.stride(1), processed.stride(2),
            processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
            processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
            num_warps=4, num_stages=2,
        )

        # If I != 0, also split hidden stream
        if I != 0:
            grid_split_hidden = (B, I)  # stream id = 1 for hidden
            _split_kernel[grid_split_hidden](
                processed, processed_encoder, processed_hidden,
                B, T, I, D,
                processed.stride(0), processed.stride(1), processed.stride(2),
                processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
                processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
                num_warps=4, num_stages=2,
            )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
