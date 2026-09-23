import torch
import triton
import triton.language as tl


@triton.jit
def _copy_encoder_to_out(
    enc_ptr,      # *ptr to encoder_hidden_states: [B, T, D]
    out_ptr,      # *ptr to concatenated out: [B, P, D], we only write first T
    B: tl.constexpr,
    T: tl.constexpr,
    D: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Grid: (B, ceil(T / BLOCK_P), ceil(D / BLOCK_N))
    b = tl.program_id(0)
    pid_p = tl.program_id(1)
    pid_n = tl.program_id(2)

    p_start = pid_p * BLOCK_P
    n_start = pid_n * BLOCK_N

    p_offsets = p_start + tl.arange(0, BLOCK_P)
    n_offsets = n_start + tl.arange(0, BLOCK_N)

    mask_p = p_offsets < T
    mask_n = n_offsets < D

    # For each tile, copy elementwise
    for pi in range(BLOCK_P):
        p = p_start + pi
        if p < T:
            for ni in range(BLOCK_N):
                n = n_start + ni
                if n < D:
                    val = tl.load(enc_ptr + b * T * D + p * D + n)
                    tl.store(out_ptr + b * D * (T + 0) + p * D + n, val)


@triton.jit
def _copy_img_to_out(
    hst_ptr,      # *ptr to hidden_states: [B, I, D]
    out_ptr,      # *ptr to concatenated out: [B, P, D]
    B: tl.constexpr,
    I: tl.constexpr,
    D: tl.constexpr,
    T: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Grid: (B, ceil(I / BLOCK_P), ceil(D / BLOCK_N))
    b = tl.program_id(0)
    pid_p = tl.program_id(1)
    pid_n = tl.program_id(2)

    p_start = pid_p * BLOCK_P  # along I
    n_start = pid_n * BLOCK_N  # along D

    p_offsets = p_start + tl.arange(0, BLOCK_P)
    n_offsets = n_start + tl.arange(0, BLOCK_N)

    mask_p = p_offsets < I
    mask_n = n_offsets < D

    for pi in range(BLOCK_P):
        i = p_start + pi
        if i < I:
            for ni in range(BLOCK_N):
                n = n_start + ni
                if n < D:
                    val = tl.load(hst_ptr + b * I * D + i * D + n)
                    # out has shape [B, P, D], P = T + I; i-th index in out is at offset T + i
                    tl.store(out_ptr + b * D * (T + I) + (T + i) * D + n, val)


@triton.jit
def _gemm_per_row_kernel(
    A_ptr,        # *ptr to concatenated input: [B, P, D]
    WT_ptr,       # *ptr to process_weight.T: [D, D]
    Y_ptr,        # *ptr to output: [B, P, D]
    B: tl.constexpr,
    P: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,  # tile size along M=P
    BLOCK_N: tl.constexpr,  # tile size along N=D
):
    # Grid: (B, ceil(P/BLOCK_M), ceil(D/BLOCK_N))
    b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N

    m_offsets = m_start + tl.arange(0, BLOCK_M)  # along P
    n_offsets = n_start + tl.arange(0, BLOCK_N)  # along D

    mask_m = m_offsets < P
    mask_n = n_offsets < D

    # fp32 accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Per-row reduction over K = D
    for k in range(0, D):
        # Load A tile: A[b, m, k] -> shape [BLOCK_M, 1]
        a_ptrs = A_ptr + b * P * D + m_offsets[:, None] * D + k
        a_tile = tl.load(a_ptrs, mask=mask_m[:, None], other=0.0)

        # Load WT row: WT[k, n] -> shape [1, BLOCK_N]
        wt_ptrs = WT_ptr + k * D + n_offsets[None, :]
        wt_row = tl.load(wt_ptrs, mask=mask_n[None, :], other=0.0)

        # Accumulate
        acc += a_tile.to(tl.float32) @ wt_row.to(tl.float32)

    # Store back to Y (we'll cast as needed on host). Store in fp32 for safety.
    y_ptrs = Y_ptr + b * P * D + m_offsets[:, None] * D + n_offsets[None, :]
    tl.store(y_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


@triton.jit
def _copy_slice_to_encoder(
    y_ptr,        # *ptr to processed: [B, P, D]
    out_ptr,      # *ptr to processed_encoder: [B, T, D]
    B: tl.constexpr,
    T: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Grid: (B, ceil(T/BLOCK_M), ceil(D/BLOCK_N))
    b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N

    m_offsets = m_start + tl.arange(0, BLOCK_M)
    n_offsets = n_start + tl.arange(0, BLOCK_N)

    mask_m = m_offsets < T
    mask_n = n_offsets < D

    y_ptrs = y_ptr + b * (T + 0) * D + m_offsets[:, None] * D + n_offsets[None, :]
    out_ptrs = out_ptr + b * T * D + m_offsets[:, None] * D + n_offsets[None, :]

    vals = tl.load(y_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0)
    tl.store(out_ptrs, vals, mask=mask_m[:, None] & mask_n[None, :])


@triton.jit
def _copy_slice_to_img(
    y_ptr,        # *ptr to processed: [B, P, D]
    out_ptr,      # *ptr to processed_hidden: [B, I, D]
    B: tl.constexpr,
    T: tl.constexpr,
    I: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Grid: (B, ceil(I/BLOCK_M), ceil(D/BLOCK_N))
    b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N

    m_offsets = m_start + tl.arange(0, BLOCK_M)  # along I
    n_offsets = n_start + tl.arange(0, BLOCK_N)  # along D

    mask_m = m_offsets < I
    mask_n = n_offsets < D

    y_ptrs = y_ptr + b * (T + I) * D + (T + m_offsets[:, None]) * D + n_offsets[None, :]
    out_ptrs = out_ptr + b * I * D + m_offsets[:, None] * D + n_offsets[None, :]

    vals = tl.load(y_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0)
    tl.store(out_ptrs, vals, mask=mask_m[:, None] & mask_n[None, :])


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                encoder_hidden_states: torch.Tensor,
                process_weight: torch.Tensor):
        # Shapes
        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        D = hidden_states.shape[2]
        assert encoder_hidden_states.shape == (B, T, D), "encoder_hidden_states must be [B, T, D]"
        assert hidden_states.shape == (B, I, D), "hidden_states must be [B, I, D]"
        assert process_weight.shape == (D, D), "process_weight must be [D, D]"
        assert hidden_states.dtype == encoder_hidden_states.dtype, "dtypes must match"
        assert process_weight.dtype == hidden_states.dtype, "process_weight dtype should match hidden_states dtype"

        device = hidden_states.device
        dtype = hidden_states.dtype

        # Allocate concatenated A [B, P, D]
        P = T + I
        A = torch.empty((B, P, D), device=device, dtype=dtype)

        # Build A by copying encoder and hidden parts
        # Choose tiles
        BLOCK_P_COPY = 64
        BLOCK_N_COPY = 64

        # Copy encoder into A[:, :T, :]
        grid_enc = (B, triton.cdiv(T, BLOCK_P_COPY), triton.cdiv(D, BLOCK_N_COPY))
        _copy_encoder_to_out[grid_enc](
            encoder_hidden_states, A,
            B=B, T=T, D=D,
            BLOCK_P=BLOCK_P_COPY, BLOCK_N=BLOCK_N_COPY,
        )

        # Copy hidden into A[:, T:, :]
        grid_img = (B, triton.cdiv(I, BLOCK_P_COPY), triton.cdiv(D, BLOCK_N_COPY))
        _copy_img_to_out[grid_img](
            hidden_states, A,
            B=B, I=I, D=D, T=T,
            BLOCK_P=BLOCK_P_COPY, BLOCK_N=BLOCK_N_COPY,
        )

        # Ensure process_weight.T is contiguous [D, D]
        WT = process_weight.t().contiguous()  # [D, D]

        # Output Y [B, P, D]
        Y = torch.empty((B, P, D), device=device, dtype=torch.float32)  # compute in fp32 for stability

        # GEMM grid: (B, tiles of P, tiles of D)
        BLOCK_M_GEMM = 128  # along P
        BLOCK_N_GEMM = 128  # along D
        grid_gemm = (B, triton.cdiv(P, BLOCK_M_GEMM), triton.cdiv(D, BLOCK_N_GEMM))
        _gemm_per_row_kernel[grid_gemm](
            A, WT, Y,
            B=B, P=P, D=D,
            BLOCK_M=BLOCK_M_GEMM, BLOCK_N=BLOCK_N_GEMM,
            num_warps=4, num_stages=2,
        )

        # Split Y into encoder and hidden outputs
        processed_encoder = torch.empty((B, T, D), device=device, dtype=torch.float32)
        processed_hidden = torch.empty((B, I, D), device=device, dtype=torch.float32)

        BLOCK_M_COPY = 128
        BLOCK_N_COPY = 128

        grid_enc_out = (B, triton.cdiv(T, BLOCK_M_COPY), triton.cdiv(D, BLOCK_N_COPY))
        _copy_slice_to_encoder[grid_enc_out](
            Y, processed_encoder,
            B=B, T=T, D=D,
            BLOCK_M=BLOCK_M_COPY, BLOCK_N=BLOCK_N_COPY,
            num_warps=4, num_stages=2,
        )

        grid_img_out = (B, triton.cdiv(I, BLOCK_M_COPY), triton.cdiv(D, BLOCK_N_COPY))
        _copy_slice_to_img[grid_img_out](
            Y, processed_hidden,
            B=B, T=T, I=I, D=D,
            BLOCK_M=BLOCK_M_COPY, BLOCK_N=BLOCK_N_COPY,
            num_warps=4, num_stages=2,
        )

        # Match original return types: original returns (processed_encoder, processed_hidden)
        # Keep outputs as float32. If you need to match input dtype, you can cast back:
        # processed_encoder = processed_encoder.to(dtype)
        # processed_hidden = processed_hidden.to(dtype)
        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
