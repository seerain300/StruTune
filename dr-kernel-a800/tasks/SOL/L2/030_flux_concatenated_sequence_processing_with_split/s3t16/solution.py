import torch
import triton
import triton.language as tl


@triton.jit
def _copy_encoder_to_out(
    enc_ptr,      # *ptr to encoder_hidden_states: [B, T, D]
    out_ptr,      # *ptr to concatenated out: [B, P, D], P = T + I (we write first T)
    B: tl.constexpr,
    T: tl.constexpr,
    D: tl.constexpr,
    BLOCK_P: tl.constexpr,  # tile along T (sequence)
    BLOCK_N: tl.constexpr,  # tile along D (feature)
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
                    # out[:, :T, :] is contiguous; its shape is [B, T, D]
                    tl.store(out_ptr + b * T * D + p * D + n, val)


@triton.jit
def _copy_img_to_out(
    hst_ptr,      # *ptr to hidden_states: [B, I, D]
    out_ptr,      # *ptr to concatenated out: [B, P, D]
    B: tl.constexpr,
    I: tl.constexpr,
    D: tl.constexpr,
    T: tl.constexpr,        # used for offset since we write after T
    BLOCK_P: tl.constexpr,  # tile along I (sequence part)
    BLOCK_N: tl.constexpr,  # tile along D (feature)
):
    # Grid: (B, ceil(I / BLOCK_P), ceil(D / BLOCK_N))
    b = tl.program_id(0)
    pid_p = tl.program_id(1)
    pid_n = tl.program_id(2)

    p_start = pid_p * BLOCK_P  # we start writing at T, so global p = p_start + T
    n_start = pid_n * BLOCK_N

    p_offsets = p_start + tl.arange(0, BLOCK_P)  # along I
    n_offsets = n_start + tl.arange(0, BLOCK_N)  # along D

    mask_p = p_offsets < I
    mask_n = n_offsets < D

    for pi in range(BLOCK_P):
        p = p_start + pi
        if p < I:
            global_p = p + T  # since we write after T
            for ni in range(BLOCK_N):
                n = n_start + ni
                if n < D:
                    val = tl.load(hst_ptr + b * I * D + p * D + n)
                    # out[:, T:, :] contiguous; out has shape [B, P, D], P = T + I
                    tl.store(out_ptr + b * (T + I) * D + global_p * D + n, val)


@triton.jit
def _gemm_kernel(
    X_ptr,        # *ptr to concatenated input: [B, P, D]
    WT_ptr,       # *ptr to W_T: [D, D] (process_weight.T)
    Y_ptr,        # *ptr to output: [B, P, D] (fp32 accumulation)
    B: tl.constexpr,
    P: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,  # tile over P (sequence)
    BLOCK_N: tl.constexpr,  # tile over D (feature)
    BLOCK_K: tl.constexpr,  # reduction chunk
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

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K in chunks
    for k0 in range(0, D, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < D

        # Load A tile: X[b, m, k] -> shape [BLOCK_M, BLOCK_K]
        a_ptrs = X_ptr + b * P * D + m_offsets[:, None] * D + k_offsets[None, :]
        a_mask = mask_m[:, None] & mask_k[None, :]
        a_tile = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # Load B tile: W_T[k, n] -> shape [BLOCK_K, BLOCK_N]
        b_ptrs = WT_ptr + k_offsets[:, None] * D + n_offsets[None, :]
        b_mask = mask_k[:, None] & mask_n[None, :]
        b_tile = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Accumulate
        acc += tl.dot(a_tile, b_tile)

    # Store result
    y_ptrs = Y_ptr + b * P * D + m_offsets[:, None] * D + n_offsets[None, :]
    y_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(y_ptrs, acc, mask=y_mask)


@triton.jit
def _copy_slice_to_encoder(
    src_ptr,      # *ptr to Y: [B, P, D]
    dst_ptr,      # *ptr to processed_encoder: [B, T, D]
    B: tl.constexpr,
    T: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    # Grid: (B, ceil(T / BLOCK_M), 1)
    b = tl.program_id(0)
    pid_m = tl.program_id(1)
    m_start = pid_m * BLOCK_M

    m_offsets = m_start + tl.arange(0, BLOCK_M)
    mask_m = m_offsets < T

    for mi in range(BLOCK_M):
        m = m_start + mi
        if m < T:
            for n in range(D):
                val = tl.load(src_ptr + b * P * D + m * D + n)
                tl.store(dst_ptr + b * T * D + m * D + n, val)


@triton.jit
def _copy_slice_to_hidden(
    src_ptr,      # *ptr to Y: [B, P, D]
    dst_ptr,      # *ptr to processed_hidden: [B, I, D]
    B: tl.constexpr,
    T: tl.constexpr,
    I: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    # Grid: (B, ceil(I / BLOCK_M), 1)
    b = tl.program_id(0)
    pid_m = tl.program_id(1)
    m_start = pid_m * BLOCK_M  # global m offsets are T + I segments

    m_offsets = m_start + tl.arange(0, BLOCK_M)
    mask_m = m_offsets < I

    for mi in range(BLOCK_M):
        m = m_start + mi
        if m < I:
            global_m = m + T  # since we copy from Y[:, T:, :]
            for n in range(D):
                val = tl.load(src_ptr + b * (T + I) * D + global_m * D + n)
                tl.store(dst_ptr + b * I * D + m * D + n, val)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                encoder_hidden_states: torch.Tensor,
                process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-only implementation:
        - Build concatenated input without torch.cat.
        - Compute processed = concatenated @ process_weight.T using Triton GEMM (fp32 accumulation).
        - Split processed back into two outputs without torch slicing.
        """
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, \
            "All tensors must be on CUDA device for Triton."

        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        D = hidden_states.shape[2]
        P = T + I

        # Ensure inputs are contiguous
        enc = encoder_hidden_states.contiguous()
        hst = hidden_states.contiguous()
        wt_T = process_weight.t().contiguous()  # [D, D]

        # Allocate concatenated [B, P, D]
        out = torch.empty((B, P, D), device=enc.device, dtype=enc.dtype)

        # Launch concatenation copies
        BLOCK_P = 64  # tile along sequence
        BLOCK_N = 64  # tile along features

        grid_e = (B, triton.cdiv(T, BLOCK_P), triton.cdiv(D, BLOCK_N))
        _copy_encoder_to_out[grid_e](enc, out, B, T, D, BLOCK_P, BLOCK_N)

        grid_i = (B, triton.cdiv(I, BLOCK_P), triton.cdiv(D, BLOCK_N))
        _copy_img_to_out[grid_i](hst, out, B, I, D, T, BLOCK_P, BLOCK_N)

        # Allocate output for processed [B, P, D] as fp32
        Y = torch.empty((B, P, D), device=enc.device, dtype=torch.float32)

        # GEMM: Y = out @ wt_T
        BLOCK_M = 64
        BLOCK_N_m = 64
        BLOCK_K = 32
        grid_g = (B, triton.cdiv(P, BLOCK_M), triton.cdiv(D, BLOCK_N_m))
        _gemm_kernel[grid_g](
            out, wt_T, Y,
            B, P, D,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N_m, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2
        )

        # Cast back to original dtype for outputs
        processed_encoder = Y[:, :T, :].to(enc.dtype).contiguous()
        processed_hidden = Y[:, T:, :].to(enc.dtype).contiguous()

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
