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

    total = T + I
    stream = pid_row // total
    pos = pid_row % total
    if stream == 1:
        pos = pos - T

    src_ptr = E_ptr + pid_b * E_b_stride + pos * E_t_stride
    dst_ptr = C_ptr + pid_b * C_b_stride + pid_row * C_seqlen_stride

    # copy a vector of length D in tiles
    for d in range(0, D, BLOCK_D):
        offs = d + tl.arange(0, BLOCK_D)
        mask = offs < D
        vals = tl.load(src_ptr + offs * E_d_stride, mask=mask, other=0.0)
        tl.store(dst_ptr + offs * C_d_stride, vals, mask=mask)


@triton.jit
def _batched_gemm_per_stream_kernel(
    X_ptr,        # input stream: [B, M, D] where M=T for encoder or I for hidden
    W_ptr,        # process_weight.T: [D, D]
    Y_ptr,        # output: [B, M, D]
    B, M, D,
    X_b_stride, X_m_stride, X_d_stride,
    W0_stride, W1_stride,
    Y_b_stride, Y_m_stride, Y_d_stride,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_b = tl.program_id(0)  # batch index
    pid_m = tl.program_id(1)  # row-tile index within M
    pid_n = tl.program_id(2)  # feature-tile index within D

    # tile coordinates
    m0 = pid_m * BLOCK_M
    n0 = pid_n * BLOCK_N

    # output accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # reduction over K = D
    for k0 in range(0, D, BLOCK_K):
        k_ids = k0 + tl.arange(0, BLOCK_K)

        # load X[b, m, k] -> shape (BLOCK_M, BLOCK_K)
        m_ids = m0 + tl.arange(0, BLOCK_M)
        x_ptrs = X_ptr + pid_b * X_b_stride + m_ids[:, None] * X_m_stride + k_ids[None, :] * X_d_stride
        x_mask = (m_ids[:, None] < M) & (k_ids[None, :] < D)
        X_block = tl.load(x_ptrs, mask=x_mask, other=0.0)

        # load W[k, n] -> shape (BLOCK_K, BLOCK_N)
        w_ptrs = W_ptr + k_ids[:, None] * W0_stride + (n0 + tl.arange(0, BLOCK_N))[None, :] * W1_stride
        w_mask = (k_ids[:, None] < D) & ((n0 + tl.arange(0, BLOCK_N))[None, :] < D)
        W_block = tl.load(w_ptrs, mask=w_mask, other=0.0)

        # accumulate
        acc += tl.dot(X_block, W_block)

    # write back to Y
    y_ptrs = Y_ptr + pid_b * Y_b_stride + (m0 + tl.arange(0, BLOCK_M))[:, None] * Y_m_stride + (n0 + tl.arange(0, BLOCK_N))[None, :] * Y_d_stride
    y_mask = ((m0 + tl.arange(0, BLOCK_M))[:, None] < M) & ((n0 + tl.arange(0, BLOCK_N))[None, :] < D)
    tl.store(y_ptrs, acc, mask=y_mask)


@triton.jit
def _split_copy_rows_kernel(
    Y_ptr,        # input: [B, T+I, D]
    E_out_ptr,    # output for encoder stream: [B, T, D]
    H_out_ptr,    # output for hidden stream: [B, I, D]
    B, T, I, D,
    Y_b_stride, Y_t_stride, Y_d_stride,
    E_b_stride, E_t_stride, E_d_stride,
    H_b_stride, H_t_stride, H_d_stride,
    BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)  # batch
    pid_rows = tl.program_id(1)  # row-tile index
    pid_tile = tl.program_id(2)  # feature-tile index

    # rows to copy: first T for encoder, next I for hidden
    for r in range(0, T):
        y_row = r
        e_dst = E_out_ptr + pid_b * E_b_stride + r * E_t_stride
        y_src = Y_ptr + pid_b * Y_b_stride + y_row * Y_t_stride
        for d in range(0, D, BLOCK_D):
            offs = d + tl.arange(0, BLOCK_D)
            mask = offs < D
            vals = tl.load(y_src + offs * Y_d_stride, mask=mask, other=0.0)
            tl.store(e_dst + offs * E_d_stride, vals, mask=mask)

    for r in range(0, I):
        y_row = T + r
        h_dst = H_out_ptr + pid_b * H_b_stride + r * H_t_stride
        y_src = Y_ptr + pid_b * Y_b_stride + y_row * Y_t_stride
        for d in range(0, D, BLOCK_D):
            offs = d + tl.arange(0, BLOCK_D)
            mask = offs < D
            vals = tl.load(y_src + offs * Y_d_stride, mask=mask, other=0.0)
            tl.store(h_dst + offs * H_d_stride, vals, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version:
        1) Concatenate encoder_hidden_states and hidden_states along sequence dimension
        2) Apply linear projection via Triton GEMM: [B, T+I, D] @ [D, D].T
        3) Split back into encoder and hidden streams
        Returns: (processed_encoder_hidden_states, processed_hidden_states)
        """
        # Ensure inputs are contiguous
        E = encoder_hidden_states.contiguous()  # [B, T, D]
        H = hidden_states.contiguous()          # [B, I, D]
        W = process_weight.contiguous()         # [D, D]

        B, T, D = E.shape
        I = H.shape[1]

        # 1) Triton concatenate into C: [B, T+I, D]
        C = torch.empty((B, T + I, D), device=E.device, dtype=torch.float32)
        BLOCK_D = 128
        grid_concat = (B, T + I, (D + BLOCK_D - 1) // BLOCK_D)
        _concatenate_streams_kernel[grid_concat](
            E, H, C,
            B, T, I, D,
            E.stride(0), E.stride(1), E.stride(2),
            H.stride(0), H.stride(1), H.stride(2),
            C.stride(0), C.stride(1), C.stride(2),
            BLOCK_D=BLOCK_D,
        )

        # 2) Triton GEMM: Y = C @ W.T, per stream. Compute both encoder and hidden outputs in Triton.
        # Prepare outputs
        Y_encoder = torch.empty((B, T, D), device=E.device, dtype=torch.float32)
        Y_hidden = torch.empty((B, I, D), device=E.device, dtype=torch.float32)

        # Stream 0: encoder rows [0..T)
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 64
        grid_encoder = (B, (T + BLOCK_M - 1) // BLOCK_M, (D + BLOCK_N - 1) // BLOCK_N)
        _batched_gemm_per_stream_kernel[grid_encoder](
            C, W.t(), Y_encoder,
            B, T, D,
            C.stride(0), C.stride(1), C.stride(2),
            W.stride(0), W.stride(1),
            Y_encoder.stride(0), Y_encoder.stride(1), Y_encoder.stride(2),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        # Stream 1: hidden rows [T..T+I)
        Y_hidden = torch.empty((B, I, D), device=E.device, dtype=torch.float32)  # will be filled below
        # We need to pass W.t() as well; since W is [D, D], W.t() is [D, D]
        grid_hidden = (B, (I + BLOCK_M - 1) // BLOCK_M, (D + BLOCK_N - 1) // BLOCK_N)
        _batched_gemm_per_stream_kernel[grid_hidden](
            C, W.t(), Y_hidden,
            B, I, D,
            C.stride(0), C.stride(1), C.stride(2),
            W.stride(0), W.stride(1),
            Y_hidden.stride(0), Y_hidden.stride(1), Y_hidden.stride(2),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        # 3) Triton split from the computed Y by directly copying results into final outputs
        processed_encoder = torch.empty((B, T, D), device=E.device, dtype=torch.float32)
        processed_hidden = torch.empty((B, I, D), device=E.device, dtype=torch.float32)

        BLOCK_SPLIT = 128
        grid_encoder_rows = (B, T, (D + BLOCK_SPLIT - 1) // BLOCK_SPLIT)
        grid_hidden_rows = (B, I, (D + BLOCK_SPLIT - 1) // BLOCK_SPLIT)

        # Copy Y_encoder into processed_encoder
        _split_copy_rows_kernel[grid_encoder_rows](
            Y_encoder, processed_encoder, processed_hidden,  # third arg unused, but required
            B, T, I, D,
            Y_encoder.stride(0), Y_encoder.stride(1), Y_encoder.stride(2),
            processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
            processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
            BLOCK_D=BLOCK_SPLIT,
            num_warps=4, num_stages=2,
        )

        # Copy Y_hidden into processed_hidden
        _split_copy_rows_kernel[grid_hidden_rows](
            Y_hidden, processed_encoder, processed_hidden,
            B, T, I, D,
            Y_hidden.stride(0), Y_hidden.stride(1), Y_hidden.stride(2),
            processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
            processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
            BLOCK_D=BLOCK_SPLIT,
            num_warps=4, num_stages=2,
        )

        # Return only the encoder and hidden streams (processed_encoder already filled above)
        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
