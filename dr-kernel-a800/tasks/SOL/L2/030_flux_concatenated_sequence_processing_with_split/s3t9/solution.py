import torch
import triton
import triton.language as tl


@triton.jit
def _copy_encoder_to_out(
    enc_ptr,      # *ptr to encoder_hidden_states: [B, T, D]
    out_ptr,      # *ptr to concatenated out: [B, P, D], we only write first T rows
    B: tl.constexpr,
    T: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,  # tile over T (sequence)
    BLOCK_N: tl.constexpr,  # tile over D (feature)
):
    # Grid: (B, ceil(T / BLOCK_M), ceil(D / BLOCK_N))
    b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N

    m_offsets = m_start + tl.arange(0, BLOCK_M)  # along T
    n_offsets = n_start + tl.arange(0, BLOCK_N)  # along D

    mask_m = m_offsets < T
    mask_n = n_offsets < D

    for mi in range(BLOCK_M):
        m = m_start + mi
        if m < T:
            for ni in range(BLOCK_N):
                n = n_start + ni
                if n < D:
                    val = tl.load(enc_ptr + b * T * D + m * D + n)
                    tl.store(out_ptr + b * (T + 0) * D + m * D + n, val)


@triton.jit
def _copy_img_to_out(
    hst_ptr,      # *ptr to hidden_states: [B, I, D]
    out_ptr,      # *ptr to concatenated out: [B, P, D]
    B: tl.constexpr,
    I: tl.constexpr,
    D: tl.constexpr,
    T: tl.constexpr,
    BLOCK_M: tl.constexpr,  # tile over I (image seq)
    BLOCK_N: tl.constexpr,  # tile over D (feature)
):
    # Grid: (B, ceil(I / BLOCK_M), ceil(D / BLOCK_N))
    b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N

    m_offsets = m_start + tl.arange(0, BLOCK_M)  # along I
    n_offsets = n_start + tl.arange(0, BLOCK_N)  # along D

    mask_m = m_offsets < I
    mask_n = n_offsets < D

    for mi in range(BLOCK_M):
        m = m_start + mi
        if m < I:
            for ni in range(BLOCK_N):
                n = n_start + ni
                if n < D:
                    val = tl.load(hst_ptr + b * I * D + m * D + n)
                    # write to out at offset T in sequence dimension
                    tl.store(out_ptr + b * (T + I) * D + (m + T) * D + n, val)


@triton.jit
def _project_rows_kernel(
    X_ptr,        # *ptr to concatenated input: [B, P, D]
    WT_ptr,       # *ptr to W_T: [D, D]
    Y_ptr,        # *ptr to output: [B, P, D]
    B: tl.constexpr,
    P: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,  # tile over P (sequence)
    BLOCK_N: tl.constexpr,  # tile over D (feature)
):
    # Grid: (B, ceil(P / BLOCK_M), ceil(D / BLOCK_N))
    b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N

    m_offsets = m_start + tl.arange(0, BLOCK_M)  # along P
    n_offsets = n_start + tl.arange(0, BLOCK_N)  # along D

    mask_m = m_offsets < P
    mask_n = n_offsets < D

    # Accumulator for this tile
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Reduction over K = D
    for k in range(0, D):
        # Load X tile row vector: X[b, m, k] -> [BLOCK_M, 1]
        x_ptrs = X_ptr + b * P * D + m_offsets[:, None] * D + k
        x_tile = tl.load(x_ptrs, mask=mask_m[:, None], other=0.0)

        # Load WT row: WT[k, n] -> [1, BLOCK_N]
        wt_ptrs = WT_ptr + k * D + n_offsets[None, :]
        wt_row = tl.load(wt_ptrs, mask=mask_n[None, :], other=0.0)

        # Outer product accumulate
        acc += x_tile * wt_row

    # Store results
    y_ptrs = Y_ptr + b * P * D + m_offsets[:, None] * D + n_offsets[None, :]
    tl.store(y_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


@triton.jit
def _copy_rows_to_offset_T(
    src_ptr,      # *ptr to source: [B, P, D]
    dst_ptr,      # *ptr to destination: [B, T, D]
    B: tl.constexpr,
    P: tl.constexpr,
    T: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,  # tile over T (destination rows)
    BLOCK_N: tl.constexpr,  # tile over D (feature)
):
    # Copy src[b, :T, :] -> dst[b, :, :]
    # Grid: (B, ceil(T / BLOCK_M), ceil(D / BLOCK_N))
    b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N

    m_offsets = m_start + tl.arange(0, BLOCK_M)  # along T
    n_offsets = n_start + tl.arange(0, BLOCK_N)  # along D

    mask_m = m_offsets < T
    mask_n = n_offsets < D

    for mi in range(BLOCK_M):
        m = m_start + mi
        if m < T:
            for ni in range(BLOCK_N):
                n = n_start + ni
                if n < D:
                    val = tl.load(src_ptr + b * P * D + m * D + n)
                    tl.store(dst_ptr + b * T * D + m * D + n, val)


@triton.jit
def _copy_rows_to_offset_T_img(
    src_ptr,      # *ptr to source: [B, P, D]
    dst_ptr,      # *ptr to destination: [B, I, D]
    B: tl.constexpr,
    P: tl.constexpr,
    I: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,  # tile over I (destination rows)
    BLOCK_N: tl.constexpr,  # tile over D (feature)
):
    # Copy src[b, T:, :] -> dst[b, :, :]
    # Grid: (B, ceil(I / BLOCK_M), ceil(D / BLOCK_N))
    b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N

    m_offsets = m_start + tl.arange(0, BLOCK_M)  # along I
    n_offsets = n_start + tl.arange(0, BLOCK_N)  # along D

    mask_m = m_offsets < I
    mask_n = n_offsets < D

    for mi in range(BLOCK_M):
        m = m_start + mi
        if m < I:
            for ni in range(BLOCK_N):
                n = n_start + ni
                if n < D:
                    # src row index is m + T
                    val = tl.load(src_ptr + b * P * D + (m + T) * D + n)
                    tl.store(dst_ptr + b * I * D + m * D + n, val)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                encoder_hidden_states: torch.Tensor,
                process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-only implementation:
        - Concatenate encoder_hidden_states and hidden_states along sequence dim.
        - Apply linear projection via a Triton kernel (out = X @ process_weight.T).
        - Split back into processed_encoder and processed_hidden.
        """
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, \
            "Triton kernels require CUDA tensors."
        B, T, D = encoder_hidden_states.shape
        I = hidden_states.shape[1]
        P = T + I

        # Ensure contiguity
        encoder = encoder_hidden_states.contiguous()
        hidden = hidden_states.contiguous()
        wt = process_weight.t().contiguous()  # [D, D]
        device = encoder.device
        dtype = encoder.dtype

        # 1) Concatenate into out [B, P, D] using Triton
        out = torch.empty((B, P, D), device=device, dtype=dtype)
        grid_concat = (B, (T + 128 - 1) // 128, (D + 64 - 1) // 64)
        _copy_encoder_to_out[grid_concat](
            encoder, out,
            B=B, T=T, D=D,
            BLOCK_M=128, BLOCK_N=64,
            num_warps=4, num_stages=2
        )
        grid_img = (B, (I + 128 - 1) // 128, (D + 64 - 1) // 64)
        _copy_img_to_out[grid_img](
            hidden, out,
            B=B, I=I, D=D, T=T,
            BLOCK_M=128, BLOCK_N=64,
            num_warps=4, num_stages=2
        )

        # 2) Linear projection using Triton per-element kernel: out @ wt -> processed
        processed = torch.empty((B, P, D), device=device, dtype=dtype)
        grid_proj = (B, (P + 128 - 1) // 128, (D + 64 - 1) // 64)
        _project_rows_kernel[grid_proj](
            out, wt, processed,
            B=B, P=P, D=D,
            BLOCK_M=128, BLOCK_N=64,
            num_warps=4, num_stages=2
        )

        # 3) Split via Triton copy kernels
        processed_encoder = torch.empty((B, T, D), device=device, dtype=dtype)
        grid_split1 = (B, (T + 128 - 1) // 128, (D + 64 - 1) // 64)
        _copy_rows_to_offset_T[grid_split1](
            processed, processed_encoder,
            B=B, P=P, T=T, D=D,
            BLOCK_M=128, BLOCK_N=64,
            num_warps=4, num_stages=2
        )

        processed_hidden = torch.empty((B, I, D), device=device, dtype=dtype)
        grid_split2 = (B, (I + 128 - 1) // 128, (D + 64 - 1) // 64)
        _copy_rows_to_offset_T_img[grid_split2](
            processed, processed_hidden,
            B=B, P=P, I=I, D=D,
            BLOCK_M=128, BLOCK_N=64,
            num_warps=4, num_stages=2
        )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
