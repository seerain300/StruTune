import torch
import triton
import triton.language as tl


@triton.jit
def _copy_encoder_to_X(
    enc_ptr,      # *ptr to encoder_hidden_states: [B, T, D]
    X_ptr,        # *ptr to concatenated out: [B, P, D], we write first T rows
    B: tl.constexpr,
    T: tl.constexpr,
    D: tl.constexpr,
    P: tl.constexpr,   # P = T + I, passed for completeness
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
                    # X is [B, P, D]; for encoder part, row index is p in [0, T)
                    tl.store(X_ptr + b * P * D + p * D + n, val)


@triton.jit
def _copy_hidden_to_X(
    hst_ptr,      # *ptr to hidden_states: [B, I, D]
    X_ptr,        # *ptr to concatenated out: [B, P, D]
    B: tl.constexpr,
    I: tl.constexpr,
    D: tl.constexpr,
    P: tl.constexpr,   # P = T + I
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
            # write into X[b, T + p, :] which equals b*P*D + (T + p)*D + n
            base_row = b * P * D + (T + p) * D
            for ni in range(BLOCK_N):
                n = n_start + ni
                if n < D:
                    val = tl.load(hst_ptr + b * I * D + p * D + n)
                    tl.store(X_ptr + base_row + n, val)


@triton.jit
def _matmul_bpn_xwd(
    X_ptr,        # *ptr to X: [B, P, D]
    WT_ptr,       # *ptr to WT: [D, D] (process_weight.T)
    Y_ptr,        # *ptr to Y: [B, P, D]
    B: tl.constexpr,
    P: tl.constexpr,
    D: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Grid: (B, ceil(P / BLOCK_P), ceil(D / BLOCK_N))
    b = tl.program_id(0)
    pid_p = tl.program_id(1)
    pid_n = tl.program_id(2)

    p_start = pid_p * BLOCK_P
    n_start = pid_n * BLOCK_N

    p_offsets = p_start + tl.arange(0, BLOCK_P)
    n_offsets = n_start + tl.arange(0, BLOCK_N)

    mask_p = p_offsets < P
    mask_n = n_offsets < D

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_P, BLOCK_N), dtype=tl.float32)

    # Loop over K = D dimension (hidden size)
    # We iterate k from 0 to D-1
    for k in range(0, D):
        # Load X tile [BP, 1]: X[b, p, k]
        x_ptrs = X_ptr + b * P * D + p_offsets[:, None] * D + k
        # mask for p
        x_tile = tl.load(x_ptrs, mask=mask_p[:, None], other=0.0)  # [BP, 1], fp32

        # Load WT tile [1, BN]: WT[k, n]
        wt_ptrs = WT_ptr + k * D + n_offsets[None, :]  # [1, BN]
        wt_tile = tl.load(wt_ptrs, mask=mask_n[None, :], other=0.0)  # [1, BN]

        # Accumulate: acc += x_tile @ wt_tile
        acc += tl.dot(x_tile, wt_tile)

    # Store result tile to Y
    y_ptrs = Y_ptr + b * P * D + p_offsets[:, None] * D + n_offsets[None, :]
    tl.store(y_ptrs, acc, mask=mask_p[:, None] & mask_n[None, :])


@triton.jit
def _copy_Y_to_encoder(
    Y_ptr,        # *ptr to Y: [B, P, D]
    out_ptr,      # *ptr to processed_encoder: [B, T, D]
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
                    val = tl.load(Y_ptr + b * P * D + p * D + n)
                    tl.store(out_ptr + b * T * D + p * D + n, val)


@triton.jit
def _copy_Y_to_hidden(
    Y_ptr,        # *ptr to Y: [B, P, D]
    out_ptr,      # *ptr to processed_hidden: [B, I, D]
    B: tl.constexpr,
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
    def forward(
        self,
        encoder_hidden_states: torch.Tensor,
        hidden_states: torch.Tensor,
        process_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized forward that:
        - concatenates encoder and hidden along sequence dimension via Triton
        - performs the linear projection via Triton matmul
        - splits results into two outputs via Triton
        """
        assert encoder_hidden_states.dim() == 3 and hidden_states.dim() == 3
        B = encoder_hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        D = encoder_hidden_states.shape[2]
        P = T + I

        device = encoder_hidden_states.device

        # Prepare X: [B, P, D] in fp32
        X = torch.empty((B, P, D), dtype=torch.float32, device=device)

        # Launch concat kernels
        BLOCK_P = 64
        BLOCK_N = 64
        grid_encoder = (B, triton.cdiv(T, BLOCK_P), triton.cdiv(D, BLOCK_N))
        _copy_encoder_to_X[grid_encoder](
            encoder_hidden_states.to(torch.float32),
            X,
            B, T, D, P,
            BLOCK_P, BLOCK_N,
        )

        grid_hidden = (B, triton.cdiv(I, BLOCK_P), triton.cdiv(D, BLOCK_N))
        _copy_hidden_to_X[grid_hidden](
            hidden_states.to(torch.float32),
            X,
            B, I, D, P,
            BLOCK_P, BLOCK_N,
        )

        # WT = process_weight.T in fp32
        WT = process_weight.t().contiguous().to(torch.float32)

        # Output Y: [B, P, D] in fp32
        Y = torch.empty((B, P, D), dtype=torch.float32, device=device)

        # Launch GEMM kernel
        grid_gemm = (B, triton.cdiv(P, BLOCK_P), triton.cdiv(D, BLOCK_N))
        _matmul_bpn_xwd[grid_gemm](
            X, WT, Y,
            B, P, D,
            BLOCK_P=BLOCK_P, BLOCK_N=BLOCK_N,
            num_warps=4, num_stages=3,
        )

        # Split outputs
        processed_encoder = torch.empty((B, T, D), dtype=torch.float32, device=device)
        processed_hidden = torch.empty((B, I, D), dtype=torch.float32, device=device)

        grid_split_encoder = (B, triton.cdiv(T, BLOCK_P), triton.cdiv(D, BLOCK_N))
        _copy_Y_to_encoder[grid_split_encoder](
            Y, processed_encoder,
            B, T, D,
            BLOCK_P, BLOCK_N,
        )

        grid_split_hidden = (B, triton.cdiv(I, BLOCK_P), triton.cdiv(D, BLOCK_N))
        _copy_Y_to_hidden[grid_split_hidden](
            Y, processed_hidden,
            B, I, D,
            BLOCK_P, BLOCK_N,
        )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
