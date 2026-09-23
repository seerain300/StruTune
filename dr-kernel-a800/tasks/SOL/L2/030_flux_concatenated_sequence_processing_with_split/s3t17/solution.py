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

    for pi in range(BLOCK_P):
        p = p_start + pi
        if p < T:
            for ni in range(BLOCK_N):
                n = n_start + ni
                if n < D:
                    val = tl.load(enc_ptr + b * T * D + p * D + n)
                    tl.store(out_ptr + b * (T + 0) * D + p * D + n, val)


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

    p_start = pid_p * BLOCK_P
    n_start = pid_n * BLOCK_N

    p_offsets = p_start + tl.arange(0, BLOCK_P)
    n_offsets = n_start + tl.arange(0, BLOCK_N)

    mask_p = p_offsets < I
    mask_n = n_offsets < D

    for pi in range(BLOCK_P):
        p = p_start + pi
        if p < I:
            for ni in range(BLOCK_N):
                n = n_start + ni
                if n < D:
                    val = tl.load(hst_ptr + b * I * D + p * D + n)
                    # out is [B, T+I, D]; write at column p + T
                    tl.store(out_ptr + b * (T + I) * D + (p + T) * D + n, val)


@triton.jit
def _gemm_kernel(
    X_ptr,        # *ptr to concatenated input: [B, P, D]
    WT_ptr,       # *ptr to process_weight.T: [D, D]
    Y_ptr,        # *ptr to output: [B, P, D]
    B: tl.constexpr,
    P: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,  # tile size over P
    BLOCK_N: tl.constexpr,  # tile size over D
):
    # Grid: (B, ceil(P/BLOCK_M), ceil(D/BLOCK_N))
    b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N

    m_offsets = m_start + tl.arange(0, BLOCK_M)  # over P
    n_offsets = n_start + tl.arange(0, BLOCK_N)  # over D

    mask_m = m_offsets < P
    mask_n = n_offsets < D

    # Initialize fp32 accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Explicit K loop over hidden dimension
    for k in range(0, D):
        # Load A[:, k] -> [BLOCK_M]
        a_ptrs = X_ptr + b * P * D + m_offsets * D + k
        a_col = tl.load(a_ptrs, mask=mask_m, other=0.0)  # [BLOCK_M]
        a_col = a_col[:, None]  # [BLOCK_M, 1]

        # Load WT[k, :] -> [1, BLOCK_N]
        wt_row_ptrs = WT_ptr + k * D + n_offsets
        wt_row = tl.load(wt_row_ptrs, mask=mask_n, other=0.0)  # [BLOCK_N]
        wt_row = wt_row[None, :]  # [1, BLOCK_N]

        # Outer product accumulate
        acc += a_col * wt_row

    # Store acc to Y[b, m, n] -> [BLOCK_M, BLOCK_N]
    y_ptrs = Y_ptr + b * P * D + m_offsets[:, None] * D + n_offsets[None, :]
    tl.store(y_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


@triton.jit
def _copy_slice_to_encoder(
    src_ptr,      # *ptr to processed: [B, P, D]
    dst_ptr,      # *ptr to processed_encoder: [B, T, D]
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

    p_offsets = p_start + tl.arange(0, BLOCK_P)  # over T
    n_offsets = n_start + tl.arange(0, BLOCK_N)  # over D

    mask_p = p_offsets < T
    mask_n = n_offsets < D

    for pi in range(BLOCK_P):
        p = p_start + pi
        if p < T:
            for ni in range(BLOCK_N):
                n = n_start + ni
                if n < D:
                    val = tl.load(src_ptr + b * (T + 0) * D + p * D + n)
                    tl.store(dst_ptr + b * T * D + p * D + n, val)


@triton.jit
def _copy_slice_to_hidden(
    src_ptr,      # *ptr to processed: [B, P, D]
    dst_ptr,      # *ptr to processed_hidden: [B, I, D]
    B: tl.constexpr,
    T: tl.constexpr,
    I: tl.constexpr,
    D: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Grid: (B, ceil(I / BLOCK_P), ceil(D / BLOCK_N))
    b = tl.program_id(0)
    pid_p = tl.program_id(1)
    pid_n = tl.program_id(2)

    p_start = pid_p * BLOCK_P
    n_start = pid_n * BLOCK_N

    p_offsets = p_start + tl.arange(0, BLOCK_P)  # over I (img_seq_len)
    n_offsets = n_start + tl.arange(0, BLOCK_N)  # over D

    mask_p = p_offsets < I
    mask_n = n_offsets < D

    for pi in range(BLOCK_P):
        p = p_start + pi
        if p < I:
            for ni in range(BLOCK_N):
                n = n_start + ni
                if n < D:
                    val = tl.load(src_ptr + b * (T + I) * D + (p + T) * D + n)
                    tl.store(dst_ptr + b * I * D + p * D + n, val)


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    process_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Triton-optimized version of the original run:
    - Concatenates encoder_hidden_states and hidden_states along sequence dimension in Triton.
    - Computes processed = concatenated @ process_weight.T in Triton.
    - Splits back into separate encoder and image streams using Triton kernels.
    """
    assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Tensors must be on CUDA"
    # We implement GEMM in fp32 for robustness; inputs should be fp32 in this Triton version.
    assert hidden_states.dtype == torch.float32 and encoder_hidden_states.dtype == torch.float32 and process_weight.dtype == torch.float32, "This Triton version expects float32 tensors"

    B = hidden_states.shape[0]
    T = encoder_hidden_states.shape[1]
    I = hidden_states.shape[1]
    D = hidden_states.shape[2]
    P = T + I

    # 1) Concatenate without torch.cat
    out = torch.empty((B, P, D), device=hidden_states.device, dtype=hidden_states.dtype)

    # Launch copy kernels for encoder and image parts
    BLOCK_P = 128
    BLOCK_N = 128
    grid_encoder = (B, (T + BLOCK_P - 1) // BLOCK_P, (D + BLOCK_N - 1) // BLOCK_N)
    _copy_encoder_to_out[grid_encoder](
        encoder_hidden_states, out,
        B=B, T=T, D=D, BLOCK_P=BLOCK_P, BLOCK_N=BLOCK_N, num_warps=4, num_stages=2
    )

    grid_img = (B, (I + BLOCK_P - 1) // BLOCK_P, (D + BLOCK_N - 1) // BLOCK_N)
    _copy_img_to_out[grid_img](
        hidden_states, out,
        B=B, I=I, D=D, T=T, BLOCK_P=BLOCK_P, BLOCK_N=BLOCK_N, num_warps=4, num_stages=2
    )

    # 2) GEMM: processed = out @ process_weight.T, compute in fp32
    processed = torch.empty((B, P, D), device=hidden_states.device, dtype=torch.float32)

    BLOCK_M = 128
    BLOCK_N = 128
    grid_gemm = (B, (P + BLOCK_M - 1) // BLOCK_M, (D + BLOCK_N - 1) // BLOCK_N)
    _gemm_kernel[grid_gemm](
        out, process_weight.t(),
        processed,
        B=B, P=P, D=D,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        num_warps=4, num_stages=2
    )

    # 3) Split back into separate streams using Triton kernels
    processed_encoder = torch.empty((B, T, D), device=hidden_states.device, dtype=torch.float32)
    processed_hidden = torch.empty((B, I, D), device=hidden_states.device, dtype=torch.float32)

    grid_encoder_copy = (B, (T + BLOCK_P - 1) // BLOCK_P, (D + BLOCK_N - 1) // BLOCK_N)
    _copy_slice_to_encoder[grid_encoder_copy](
        processed, processed_encoder,
        B=B, T=T, D=D, BLOCK_P=BLOCK_P, BLOCK_N=BLOCK_N, num_warps=4, num_stages=2
    )

    grid_hidden_copy = (B, (I + BLOCK_P - 1) // BLOCK_P, (D + BLOCK_N - 1) // BLOCK_N)
    _copy_slice_to_hidden[grid_hidden_copy](
        processed, processed_hidden,
        B=B, T=T, I=I, D=D, BLOCK_P=BLOCK_P, BLOCK_N=BLOCK_N, num_warps=4, num_stages=2
    )

    return processed_encoder, processed_hidden


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # This forward uses Triton kernels only; no torch.cat, matmul, or tensor mm.
        return run(*args)


def run(*args):
    return ModelNew()(*args)
