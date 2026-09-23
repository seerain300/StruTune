import torch
import triton
import triton.language as tl


@triton.jit
def _copy_encoder_to_out(
    enc_ptr,      # *ptr to encoder_hidden_states: [B, T, D]
    out_ptr,      # *ptr to concatenated out: [B, P, D], P = T + I (we only write first T)
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
    T: tl.constexpr,         # T is needed to compute P and to offset copy
    BLOCK_P: tl.constexpr,   # tile over I (sequence after T)
    BLOCK_N: tl.constexpr,   # tile over D (feature)
):
    # Grid: (B, ceil(I / BLOCK_P), ceil(D / BLOCK_N))
    b = tl.program_id(0)
    pid_p = tl.program_id(1)
    pid_n = tl.program_id(2)

    p_start = pid_p * BLOCK_P  # within [T, T+I)
    n_start = pid_n * BLOCK_N

    p_offsets = p_start + tl.arange(0, BLOCK_P)  # along I, but offset by T
    n_offsets = n_start + tl.arange(0, BLOCK_N)  # along D

    mask_p = p_offsets < I
    mask_n = n_offsets < D

    for pi in range(BLOCK_P):
        p_rel = p_start + pi
        if p_rel < I:
            p = T + p_rel
            for ni in range(BLOCK_N):
                n = n_start + ni
                if n < D:
                    val = tl.load(hst_ptr + b * I * D + p_rel * D + n)
                    tl.store(out_ptr + b * (T + I) * D + p * D + n, val)


@triton.jit
def _gemm_kernel(
    X_ptr,        # *ptr to concatenated input: [B, P, D]
    WT_ptr,       # *ptr to W_T: [D, D]
    Y_ptr,        # *ptr to output: [B, P, D]
    B: tl.constexpr,
    P: tl.constexpr,
    D: tl.constexpr,
    BLOCK_P: tl.constexpr,  # tile over P (sequence)
    BLOCK_N: tl.constexpr,  # tile over N (feature)
    BLOCK_K: tl.constexpr,  # reduction chunk over K (features)
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

    acc = tl.zeros((BLOCK_P, BLOCK_N), dtype=tl.float32)

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
def _copy_encoder_slice(
    Y_ptr,        # *ptr to processed: [B, P, D]
    out_ptr,      # *ptr to processed_encoder: [B, T, D]
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
                    val = tl.load(Y_ptr + b * (T + 0) * D + p * D + n)
                    tl.store(out_ptr + b * T * D + p * D + n, val)


@triton.jit
def _copy_img_slice(
    Y_ptr,        # *ptr to processed: [B, P, D]
    out_ptr,      # *ptr to processed_hidden: [B, I, D]
    B: tl.constexpr,
    I: tl.constexpr,
    D: tl.constexpr,
    T: tl.constexpr,         # T is needed to compute P
    BLOCK_P: tl.constexpr,   # tile over I (sequence after T)
    BLOCK_N: tl.constexpr,   # tile over D (feature)
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
        p_rel = p_start + pi
        if p_rel < I:
            p = T + p_rel
            for ni in range(BLOCK_N):
                n = n_start + ni
                if n < D:
                    val = tl.load(Y_ptr + b * (T + I) * D + p * D + n)
                    tl.store(out_ptr + b * I * D + p_rel * D + n, val)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                encoder_hidden_states: torch.Tensor,
                process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-only implementation:
          - Concatenate encoder_hidden_states and hidden_states along sequence via Triton copy kernels
          - GEMM via Triton kernel: out = concatenated @ process_weight.T
          - Split via Triton copy kernels
        """
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Triton kernels require CUDA tensors"
        B, I, D = hidden_states.shape
        B_e, T, D_e = encoder_hidden_states.shape
        assert B_e == B and D == D_e, "Batch size and hidden_dim must match across inputs"
        D_w, D_w2 = process_weight.shape
        assert D_w == D and D_w2 == D, "process_weight must be [D, D]"

        # 1) Build concatenated input [B, P, D], P = T + I
        P = T + I
        out = torch.empty((B, P, D), dtype=torch.float32, device=hidden_states.device)

        # Launch copy kernels
        BLOCK_P = 128
        BLOCK_N = 64

        # Copy encoder part: out[:, :T, :]
        grid_enc = (B, triton.cdiv(T, BLOCK_P), triton.cdiv(D, BLOCK_N))
        _copy_encoder_to_out[grid_enc](
            encoder_hidden_states.contiguous(), out, B, T, D, BLOCK_P, BLOCK_N
        )

        # Copy image part: out[:, T:, :]
        grid_img = (B, triton.cdiv(I, BLOCK_P), triton.cdiv(D, BLOCK_N))
        _copy_img_to_out[grid_img](
            hidden_states.contiguous(), out, B, I, D, T, BLOCK_P, BLOCK_N
        )

        # 2) GEMM: Y = out @ process_weight.T
        Y = torch.empty((B, P, D), dtype=torch.float32, device=hidden_states.device)
        grid_gemm = (B, triton.cdiv(P, BLOCK_P), triton.cdiv(D, BLOCK_N))
        _gemm_kernel[grid_gemm](
            out, process_weight.t().contiguous(), Y, B, P, D, BLOCK_P, BLOCK_N, BLOCK_K=64
        )

        # 3) Split results
        processed_encoder = torch.empty((B, T, D), dtype=torch.float32, device=hidden_states.device)
        grid_e = (B, triton.cdiv(T, BLOCK_P), triton.cdiv(D, BLOCK_N))
        _copy_encoder_slice[grid_e](
            Y, processed_encoder, B, T, D, BLOCK_P, BLOCK_N
        )

        processed_hidden = torch.empty((B, I, D), dtype=torch.float32, device=hidden_states.device)
        grid_i = (B, triton.cdiv(I, BLOCK_P), triton.cdiv(D, BLOCK_N))
        _copy_img_slice[grid_i](
            Y, processed_hidden, B, I, D, T, BLOCK_P, BLOCK_N
        )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
