import torch
import triton
import triton.language as tl


@triton.jit
def _gemm_reduce_k_kernel(
    X_ptr,        # *ptr to concatenated input: [B, P, D]
    WT_ptr,       # *ptr to process_weight.T: [D, D]
    Y_ptr,        # *ptr to output: [B, P, D]
    B: tl.constexpr,
    P: tl.constexpr,   # P = T + I
    D: tl.constexpr,   # hidden_dim
    BLOCK_M: tl.constexpr,  # tile size over M=P
    BLOCK_N: tl.constexpr,  # tile size over N=D
):
    # Each program handles one (b, M-tile, N-tile)
    b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N

    m_offsets = m_start + tl.arange(0, BLOCK_M)  # over P
    n_offsets = n_start + tl.arange(0, BLOCK_N)  # over D

    mask_m = m_offsets < P
    mask_n = n_offsets < D

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Reduce over K = D, one step at a time
    for k in range(0, D):
        # X tile [BM, 1]: X[b, m, k]
        x_ptrs = X_ptr + b * P * D + m_offsets[:, None] * D + k  # [BM, 1]
        x_tile = tl.load(x_ptrs, mask=mask_m[:, None], other=0.0)  # [BM, 1], fp32

        # WT row [1, BN]: WT[k, n]
        wt_ptrs = WT_ptr + k * D + n_offsets[None, :]  # [1, BN]
        wt_row = tl.load(wt_ptrs, mask=mask_n[None, :], other=0.0)  # [1, BN], fp32

        # Outer product accumulate: [BM, BN]
        acc += tl.dot(x_tile, wt_row)

    # Store result: Y[b, m, n] = acc[m, n]
    y_ptrs = Y_ptr + b * P * D + m_offsets[:, None] * D + n_offsets[None, :]
    tl.store(y_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


@triton.jit
def _copy_encoder_to_out(
    enc_ptr,      # *ptr to encoder_hidden_states: [B, T, D]
    out_ptr,      # *ptr to concatenated out: [B, P, D]
    B: tl.constexpr,
    T: tl.constexpr,
    D: tl.constexpr,
    BLOCK_P: tl.constexpr,  # tile over T (sequence)
    BLOCK_N: tl.constexpr,  # tile over D (feature)
):
    # Grid: (B, ceil(T / BLOCK_P), ceil(D / BLOCK_N))
    b = tl.program_id(0)
    pid_p = tl.program_id(1)
    pid_n = tl.program_id(2)

    p_start = pid_p * BLOCK_P
    n_start = pid_n * BLOCK_N

    p_offsets = p_start + tl.arange(0, BLOCK_P)  # along T
    n_offsets = n_start + tl.arange(0, BLOCK_N)  # along D

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
    BLOCK_P: tl.constexpr,  # tile over I (image seq)
    BLOCK_N: tl.constexpr,  # tile over D (feature)
):
    # Grid: (B, ceil(I / BLOCK_P), ceil(D / BLOCK_N))
    b = tl.program_id(0)
    pid_p = tl.program_id(1)
    pid_n = tl.program_id(2)

    p_start = pid_p * BLOCK_P
    n_start = pid_n * BLOCK_N

    p_offsets = p_start + tl.arange(0, BLOCK_P)  # along I
    n_offsets = n_start + tl.arange(0, BLOCK_N)  # along D

    mask_p = p_offsets < I
    mask_n = n_offsets < D

    for pi in range(BLOCK_P):
        p = p_start + pi
        if p < I:
            for ni in range(BLOCK_N):
                n = n_start + ni
                if n < D:
                    val = tl.load(hst_ptr + b * I * D + p * D + n)
                    # write to out at offset T + p
                    tl.store(out_ptr + b * (T + I) * D + (T + p) * D + n, val)


@triton.jit
def _copy_to_processed_encoder(
    out_ptr,      # *ptr to concatenated processed: [B, P, D]
    enc_out_ptr,  # *ptr to processed_encoder: [B, T, D]
    B: tl.constexpr,
    T: tl.constexpr,
    D: tl.constexpr,
    BLOCK_P: tl.constexpr,  # tile over T
    BLOCK_N: tl.constexpr,  # tile over D
):
    # Grid: (B, ceil(T / BLOCK_P), ceil(D / BLOCK_N))
    b = tl.program_id(0)
    pid_p = tl.program_id(1)
    pid_n = tl.program_id(2)

    p_start = pid_p * BLOCK_P
    n_start = pid_n * BLOCK_N

    p_offsets = p_start + tl.arange(0, BLOCK_P)  # along T
    n_offsets = n_start + tl.arange(0, BLOCK_N)  # along D

    mask_p = p_offsets < T
    mask_n = n_offsets < D

    for pi in range(BLOCK_P):
        p = p_start + pi
        if p < T:
            for ni in range(BLOCK_N):
                n = n_start + ni
                if n < D:
                    val = tl.load(out_ptr + b * (T + 0) * D + p * D + n)
                    tl.store(enc_out_ptr + b * T * D + p * D + n, val)


@triton.jit
def _copy_to_processed_hidden(
    out_ptr,       # *ptr to concatenated processed: [B, P, D]
    hst_out_ptr,   # *ptr to processed_hidden: [B, I, D]
    B: tl.constexpr,
    T: tl.constexpr,
    I: tl.constexpr,
    D: tl.constexpr,
    BLOCK_P: tl.constexpr,  # tile over I
    BLOCK_N: tl.constexpr,  # tile over D
):
    # Grid: (B, ceil(I / BLOCK_P), ceil(D / BLOCK_N))
    b = tl.program_id(0)
    pid_p = tl.program_id(1)
    pid_n = tl.program_id(2)

    p_start = pid_p * BLOCK_P
    n_start = pid_n * BLOCK_N

    p_offsets = p_start + tl.arange(0, BLOCK_P)  # along I
    n_offsets = n_start + tl.arange(0, BLOCK_N)  # along D

    mask_p = p_offsets < I
    mask_n = n_offsets < D

    for pi in range(BLOCK_P):
        p = p_start + pi
        if p < I:
            for ni in range(BLOCK_N):
                n = n_start + ni
                if n < D:
                    val = tl.load(out_ptr + b * (T + I) * D + (T + p) * D + n)
                    tl.store(hst_out_ptr + b * I * D + p * D + n, val)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # No parameters; all computation in Triton

    def forward(self, encoder_hidden_states: torch.Tensor,
                hidden_states: torch.Tensor,
                process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Ensure inputs are on the same device and contiguous
        device = encoder_hidden_states.device
        B = encoder_hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        D = encoder_hidden_states.shape[2]
        P = T + I

        # Make inputs contiguous
        enc = encoder_hidden_states.contiguous()
        hst = hidden_states.contiguous()
        wt_t = process_weight.t().contiguous()  # [D, D]

        # 1) Concatenate into out: [B, P, D]
        out = torch.empty((B, P, D), dtype=enc.dtype, device=device)

        # Launch copy kernels
        BLOCK_P = 64
        BLOCK_N = 64
        grid_enc = (B, triton.cdiv(T, BLOCK_P), triton.cdiv(D, BLOCK_N))
        _copy_encoder_to_out[grid_enc](
            enc, out, B, T, D, BLOCK_P, BLOCK_N,
            num_warps=4, num_stages=2
        )

        BLOCK_I = 64
        grid_img = (B, triton.cdiv(I, BLOCK_I), triton.cdiv(D, BLOCK_N))
        _copy_img_to_out[grid_img](
            hst, out, B, I, D, T, BLOCK_I, BLOCK_N,
            num_warps=4, num_stages=2
        )

        # 2) GEMM: Y = out @ wt_t  -> [B, P, D]
        Y = torch.empty((B, P, D), dtype=torch.float32, device=device)  # compute in fp32 for accuracy
        # We'll perform matmul with Triton, accumulating in fp32. Output is fp32; we'll cast later.
        # Grid over (B, tiles of P, tiles of D)
        BLOCK_M = 64
        BLOCK_N2 = 64
        grid_gemm = (B, triton.cdiv(P, BLOCK_M), triton.cdiv(D, BLOCK_N2))
        _gemm_reduce_k_kernel[grid_gemm](
            out, wt_t, Y, B, P, D, BLOCK_M, BLOCK_N2,
            num_warps=4, num_stages=2
        )

        # 3) Split into processed streams
        processed_encoder = torch.empty((B, T, D), dtype=Y.dtype, device=device)
        processed_hidden = torch.empty((B, I, D), dtype=Y.dtype, device=device)

        # Launch copy kernels for splitting
        grid_split = (B, triton.cdiv(T, BLOCK_P), triton.cdiv(D, BLOCK_N))
        _copy_to_processed_encoder[grid_split](
            Y, processed_encoder, B, T, D, BLOCK_P, BLOCK_N,
            num_warps=4, num_stages=2
        )

        grid_split_h = (B, triton.cdiv(I, BLOCK_I), triton.cdiv(D, BLOCK_N))
        _copy_to_processed_hidden[grid_split_h](
            Y, processed_hidden, B, T, I, D, BLOCK_I, BLOCK_N,
            num_warps=4, num_stages=2
        )

        # Return tensors. Note: original code used matmul in fp32 by default; Y is fp32.
        # If you need to match original dtype, you could cast to enc.dtype.
        # Here we keep fp32 for numerical stability.
        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
