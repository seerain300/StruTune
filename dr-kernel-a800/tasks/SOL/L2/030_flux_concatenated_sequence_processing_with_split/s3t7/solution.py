import torch
import triton
import triton.language as tl


@triton.jit
def _copy_encoder_to_out(
    enc_ptr,      # *ptr to encoder_hidden_states: [B, T, D]
    out_ptr,      # *ptr to concatenated out: [B, P, D], we write first T rows
    B: tl.constexpr,
    T: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,  # tile over T (sequence), 1 for simplicity
    BLOCK_N: tl.constexpr,  # tile over D (feature)
):
    # Grid: (B, 1, ceil(D / BLOCK_N))
    b = tl.program_id(0)
    pid_n = tl.program_id(2)

    n_start = pid_n * BLOCK_N
    n_offsets = n_start + tl.arange(0, BLOCK_N)
    mask_n = n_offsets < D

    for p in range(0, T):
        base = enc_ptr + b * T * D + p * D
        vals = tl.load(base + n_offsets, mask=mask_n, other=0.0)
        out_row_ptr = out_ptr + b * (T + 0) * D + p * D
        tl.store(out_row_ptr + n_offsets, vals, mask=mask_n)


@triton.jit
def _copy_img_to_out(
    hst_ptr,      # *ptr to hidden_states: [B, I, D]
    out_ptr,      # *ptr to concatenated out: [B, P, D]
    B: tl.constexpr,
    I: tl.constexpr,
    D: tl.constexpr,
    T: tl.constexpr,
    BLOCK_M: tl.constexpr,  # tile over I (sequence for image part)
    BLOCK_N: tl.constexpr,  # tile over D (feature)
):
    # Grid: (B, ceil(I / BLOCK_M), ceil(D / BLOCK_N))
    b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N

    m_offsets = m_start + tl.arange(0, BLOCK_M)
    n_offsets = n_start + tl.arange(0, BLOCK_N)

    mask_m = m_offsets < I
    mask_n = n_offsets < D

    for mi in range(BLOCK_M):
        m = m_start + mi
        if m < I:
            vals = tl.load(hst_ptr + b * I * D + m * D + n_offsets, mask=mask_n, other=0.0)
            out_row_ptr = out_ptr + b * (T + I) * D + (T + m) * D
            tl.store(out_row_ptr + n_offsets, vals, mask=mask_n)


@triton.jit
def _gemm_batched_kernel_block(
    A_ptr,        # *ptr to concatenated input X: [B, P, D]
    B_ptr,        # *ptr to W_T: [D, D] (process_weight.T)
    Y_ptr,        # *ptr to output: [B, P, D]
    B: tl.constexpr,
    P: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,  # tile over M = P
    BLOCK_N: tl.constexpr,  # tile over N = D
    BLOCK_K: tl.constexpr,  # reduction tile over K = D
):
    # Grid: (B, ceil(P / BLOCK_M), ceil(D / BLOCK_N))
    b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N

    m_offsets = m_start + tl.arange(0, BLOCK_M)
    n_offsets = n_start + tl.arange(0, BLOCK_N)

    mask_m = m_offsets < P
    mask_n = n_offsets < D

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, D, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < D

        # Load A tile: [BLOCK_M, BLOCK_K] -> A[b, m, k]
        a_ptrs = A_ptr + b * P * D + m_offsets[:, None] * D + k_offsets[None, :]
        a_mask = mask_m[:, None] & mask_k[None, :]
        a_tile = tl.load(a_ptrs, mask=a_mask, other=0.0)  # load as original dtype, cast to fp32 later

        # Load B tile: [BLOCK_K, BLOCK_N] -> B[k, n] = WT[k, n]
        b_ptrs = B_ptr + k_offsets[:, None] * D + n_offsets[None, :]
        b_mask = mask_k[:, None] & mask_n[None, :]
        b_tile = tl.load(b_ptrs, mask=b_mask, other=0.0)  # load as original dtype, cast to fp32 later

        # Ensure fp32 accumulation
        a_tile = a_tile.to(tl.float32)
        b_tile = b_tile.to(tl.float32)

        # Accumulate
        acc += tl.dot(a_tile, b_tile)

    # Store results: Y[b, m, n]
    y_ptrs = Y_ptr + b * P * D + m_offsets[:, None] * D + n_offsets[None, :]
    y_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(y_ptrs, acc, mask=y_mask)


@triton.jit
def _copy_rows_to_first_T(
    src_ptr,      # *ptr to Y: [B, P, D]
    dst_ptr,      # *ptr to processed_encoder: [B, T, D]
    B: tl.constexpr,
    T: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,  # tile over T
    BLOCK_N: tl.constexpr,  # tile over D
):
    # Grid: (B, ceil(T / BLOCK_M), ceil(D / BLOCK_N))
    b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N

    m_offsets = m_start + tl.arange(0, BLOCK_M)
    n_offsets = n_start + tl.arange(0, BLOCK_N)

    mask_m = m_offsets < T
    mask_n = n_offsets < D

    for mi in range(BLOCK_M):
        m = m_start + mi
        if m < T:
            vals = tl.load(src_ptr + b * P * D + m * D + n_offsets, mask=mask_n, other=0.0)
            tl.store(dst_ptr + b * T * D + m * D + n_offsets, vals, mask=mask_n)


@triton.jit
def _copy_rows_to_offset_T(
    src_ptr,      # *ptr to Y: [B, P, D]
    dst_ptr,      # *ptr to processed_hidden: [B, I, D]
    B: tl.constexpr,
    P: tl.constexpr,
    T: tl.constexpr,
    I: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,  # tile over I (second half rows)
    BLOCK_N: tl.constexpr,  # tile over D
):
    # Grid: (B, ceil(I / BLOCK_M), ceil(D / BLOCK_N))
    b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N

    m_offsets = m_start + tl.arange(0, BLOCK_M)
    n_offsets = n_start + tl.arange(0, BLOCK_N)

    mask_m = m_offsets < I
    mask_n = n_offsets < D

    for mi in range(BLOCK_M):
        m = m_start + mi
        if m < I:
            src_row = T + m  # row index in src Y: T + m
            vals = tl.load(src_ptr + b * P * D + src_row * D + n_offsets, mask=mask_n, other=0.0)
            tl.store(dst_ptr + b * I * D + m * D + n_offsets, vals, mask=mask_n)


class ModelNew(torch.nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,            # [B, I, D]
        encoder_hidden_states: torch.Tensor,    # [B, T, D]
        process_weight: torch.Tensor,           # [D, D]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-only implementation of:
          concatenated = torch.cat([encoder_hidden_states, hidden_states], dim=1)  # [B, P, D], P = T + I
          processed = concatenated @ process_weight.T                               # [B, P, D]
          processed_encoder = processed[:, :T, :]
          processed_hidden = processed[:, T:, :]
        """
        assert hidden_states.ndim == 3 and encoder_hidden_states.ndim == 3 and process_weight.ndim == 2
        B, I, D = hidden_states.shape
        B_e, T, D_e = encoder_hidden_states.shape
        assert B_e == B and D == D_e and D == process_weight.shape[0] == process_weight.shape[1]

        # Ensure contiguous
        enc = encoder_hidden_states.contiguous()
        hst = hidden_states.contiguous()
        wt = process_weight.contiguous()  # [D, D], no bias

        # 1) Concatenate into out[:, :T, :] and out[:, T:, :] without torch.cat
        P = T + I
        out = torch.empty((B, P, D), device=enc.device, dtype=enc.dtype)

        # Copy encoder part
        BLOCK_N_COPY = 128
        grid_enc = (B, 1, (D + BLOCK_N_COPY - 1) // BLOCK_N_COPY)
        _copy_encoder_to_out[grid_enc](
            enc, out,
            B=B, T=T, D=D,
            BLOCK_M=1, BLOCK_N=BLOCK_N_COPY,
            num_warps=2, num_stages=2
        )

        # Copy image part
        BLOCK_M_COPY = 64
        grid_img = (B, (I + BLOCK_M_COPY - 1) // BLOCK_M_COPY, (D + BLOCK_N_COPY - 1) // BLOCK_N_COPY)
        _copy_img_to_out[grid_img](
            hst, out,
            B=B, I=I, D=D, T=T,
            BLOCK_M=BLOCK_M_COPY, BLOCK_N=BLOCK_N_COPY,
            num_warps=2, num_stages=2
        )

        # 2) GEMM: out = out @ wt.T using Triton
        wt_t = wt.t().contiguous()  # [D, D]
        Y = torch.empty((B, P, D), device=enc.device, dtype=enc.dtype)

        # Tiling parameters for GEMM (can tune)
        BLOCK_M_GEMM = 128
        BLOCK_N_GEMM = 128
        BLOCK_K_GEMM = 64

        grid_gemm = (B, (P + BLOCK_M_GEMM - 1) // BLOCK_M_GEMM, (D + BLOCK_N_GEMM - 1) // BLOCK_N_GEMM)
        _gemm_batched_kernel_block[grid_gemm](
            out, wt_t, Y,
            B=B, P=P, D=D,
            BLOCK_M=BLOCK_M_GEMM, BLOCK_N=BLOCK_N_GEMM, BLOCK_K=BLOCK_K_GEMM,
            num_warps=4, num_stages=3
        )

        # 3) Split into two streams using Triton copy kernels
        processed_encoder = torch.empty((B, T, D), device=enc.device, dtype=enc.dtype)
        processed_hidden = torch.empty((B, I, D), device=enc.device, dtype=enc.dtype)

        BLOCK_M_SPLIT = 128
        BLOCK_N_SPLIT = 128

        grid_split1 = (B, (T + BLOCK_M_SPLIT - 1) // BLOCK_M_SPLIT, (D + BLOCK_N_SPLIT - 1) // BLOCK_N_SPLIT)
        _copy_rows_to_first_T[grid_split1](
            Y, processed_encoder,
            B=B, T=T, D=D,
            BLOCK_M=BLOCK_M_SPLIT, BLOCK_N=BLOCK_N_SPLIT,
            num_warps=4, num_stages=2
        )

        grid_split2 = (B, (I + BLOCK_M_SPLIT - 1) // BLOCK_M_SPLIT, (D + BLOCK_N_SPLIT - 1) // BLOCK_N_SPLIT)
        _copy_rows_to_offset_T[grid_split2](
            Y, processed_hidden,
            B=B, P=P, T=T, I=I, D=D,
            BLOCK_M=BLOCK_M_SPLIT, BLOCK_N=BLOCK_N_SPLIT,
            num_warps=4, num_stages=2
        )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
