import torch
import triton
import triton.language as tl


@triton.jit
def _copy_encoder_to_X(
    enc_ptr,      # *ptr to encoder_hidden_states: [B, T, D]
    X_ptr,        # *ptr to concatenated X: [B, P, D]
    B: tl.constexpr,
    T: tl.constexpr,
    D: tl.constexpr,
    P: tl.constexpr,  # P = T + I (we only write first T rows)
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

    # Copy encoder into X[:, :T, :]
    for pi in range(BLOCK_P):
        p = p_start + pi
        if p < T:
            for ni in range(BLOCK_N):
                n = n_start + ni
                if n < D:
                    val = tl.load(enc_ptr + b * T * D + p * D + n)
                    tl.store(X_ptr + b * P * D + p * D + n, val)


@triton.jit
def _copy_hidden_to_X(
    hst_ptr,      # *ptr to hidden_states: [B, I, D]
    X_ptr,        # *ptr to concatenated X: [B, P, D]
    B: tl.constexpr,
    I: tl.constexpr,
    D: tl.constexpr,
    P: tl.constexpr,  # P = T + I
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

    p_offsets = p_start + tl.arange(0, BLOCK_P)  # over I
    n_offsets = n_start + tl.arange(0, BLOCK_N)  # over D

    mask_p = p_offsets < I
    mask_n = n_offsets < D

    # Copy hidden into X[:, T:, :]
    for pi in range(BLOCK_P):
        p = p_start + pi
        if p < I:
            for ni in range(BLOCK_N):
                n = n_start + ni
                if n < D:
                    val = tl.load(hst_ptr + b * I * D + p * D + n)
                    tl.store(X_ptr + b * P * D + (T + p) * D + n, val)


@triton.jit
def _matmul_bpn_xwd(
    X_ptr,        # *ptr to X: [B, P, D]
    WT_ptr,       # *ptr to WT: [D, D]
    Y_ptr,        # *ptr to Y: [B, P, D]
    B: tl.constexpr,
    P: tl.constexpr,
    D: tl.constexpr,
    BLOCK_P: tl.constexpr,  # tile over P
    BLOCK_N: tl.constexpr,  # tile over D
    BLOCK_K: tl.constexpr,  # reduction chunk over D
):
    # Grid: (B, ceil(P / BLOCK_P), ceil(D / BLOCK_N))
    b = tl.program_id(0)
    pid_p = tl.program_id(1)
    pid_n = tl.program_id(2)

    p_start = pid_p * BLOCK_P
    n_start = pid_n * BLOCK_N

    p_offsets = p_start + tl.arange(0, BLOCK_P)  # over P
    n_offsets = n_start + tl.arange(0, BLOCK_N)  # over D

    mask_p = p_offsets < P
    mask_n = n_offsets < D

    acc = tl.zeros((BLOCK_P, BLOCK_N), dtype=tl.float32)

    # Loop over hidden dimension K in chunks
    for k0 in range(0, D, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < D

        # Load X tile [BP, BK]: X[b, p, k]
        x_ptrs = X_ptr + b * P * D + p_offsets[:, None] * D + k_offsets[None, :]
        x_tile = tl.load(x_ptrs, mask=mask_p[:, None] & mask_k[None, :], other=0.0)

        # Load WT tile [BK, BN]: WT[k, n]
        wt_ptrs = WT_ptr + k_offsets[:, None] * D + n_offsets[None, :]
        wt_tile = tl.load(wt_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)

        acc += tl.dot(x_tile, wt_tile)

    y_ptrs = Y_ptr + b * P * D + p_offsets[:, None] * D + n_offsets[None, :]
    tl.store(y_ptrs, acc, mask=mask_p[:, None] & mask_n[None, :])


@triton.jit
def _copy_Y_to_encoder(
    Y_ptr,        # *ptr to processed: [B, P, D]
    out_ptr,      # *ptr to processed_encoder: [B, T, D]
    B: tl.constexpr,
    T: tl.constexpr,
    D: tl.constexpr,
    P: tl.constexpr,  # P = T + I
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
                    val = tl.load(Y_ptr + b * P * D + p * D + n)
                    tl.store(out_ptr + b * T * D + p * D + n, val)


@triton.jit
def _copy_Y_to_hidden(
    Y_ptr,        # *ptr to processed: [B, P, D]
    out_ptr,      # *ptr to processed_hidden: [B, I, D]
    B: tl.constexpr,
    I: tl.constexpr,
    D: tl.constexpr,
    P: tl.constexpr,  # P = T + I
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
                    val = tl.load(Y_ptr + b * P * D + (T + p) * D + n)
                    tl.store(out_ptr + b * I * D + p * D + n, val)


class ModelNew(torch.nn.Module):
    def forward(self, encoder_hidden_states: torch.Tensor,
                hidden_states: torch.Tensor,
                process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version of the original run function.
        - Avoids torch.cat and torch.matmul in forward.
        - Uses Triton kernels for concatenation, GEMM, and splitting.
        Returns (processed_encoder, processed_hidden).
        """
        assert encoder_hidden_states.is_cuda and hidden_states.is_cuda and process_weight.is_cuda, \
            "All inputs must be on CUDA device for Triton kernels."

        B = encoder_hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        D = hidden_states.shape[2]
        P = T + I

        # Allocate concatenated X in fp32 for numerical stability
        X = torch.empty((B, P, D), dtype=torch.float32, device=hidden_states.device)

        # Prepare WT = process_weight.T in fp32
        WT = process_weight.t().contiguous().to(torch.float32)

        # Output Y in fp32
        Y = torch.empty((B, P, D), dtype=torch.float32, device=hidden_states.device)

        # Kernel launch parameters
        BLOCK_POS = 128  # tile over P
        BLOCK_F = 64     # tile over D
        BLOCK_K = 64     # reduction chunk

        # Build concatenated input X
        grid_encoder = (B, triton.cdiv(T, BLOCK_POS), triton.cdiv(D, BLOCK_F))
        _copy_encoder_to_X[grid_encoder](
            encoder_hidden_states.to(torch.float32), X,
            B, T, D, P,
            BLOCK_P=BLOCK_POS, BLOCK_N=BLOCK_F,
        )

        grid_hidden = (B, triton.cdiv(I, BLOCK_POS), triton.cdiv(D, BLOCK_F))
        _copy_hidden_to_X[grid_hidden](
            hidden_states.to(torch.float32), X,
            B, I, D, P, T,
            BLOCK_P=BLOCK_POS, BLOCK_N=BLOCK_F,
        )

        # GEMM: Y = X @ WT
        grid_gemm = (B, triton.cdiv(P, BLOCK_POS), triton.cdiv(D, BLOCK_F))
        _matmul_bpn_xwd[grid_gemm](
            X, WT, Y,
            B, P, D,
            BLOCK_P=BLOCK_POS, BLOCK_N=BLOCK_F, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=3,
        )

        # Split outputs
        processed_encoder = torch.empty((B, T, D), dtype=torch.float32, device=hidden_states.device)
        processed_hidden = torch.empty((B, I, D), dtype=torch.float32, device=hidden_states.device)

        grid_split_encoder = (B, triton.cdiv(T, BLOCK_POS), triton.cdiv(D, BLOCK_F))
        _copy_Y_to_encoder[grid_split_encoder](
            Y, processed_encoder,
            B, T, D, P,
            BLOCK_P=BLOCK_POS, BLOCK_N=BLOCK_F,
        )

        grid_split_hidden = (B, triton.cdiv(I, BLOCK_POS), triton.cdiv(D, BLOCK_F))
        _copy_Y_to_hidden[grid_split_hidden](
            Y, processed_hidden,
            B, I, D, P, T,
            BLOCK_P=BLOCK_POS, BLOCK_N=BLOCK_F,
        )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
