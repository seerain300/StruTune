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
    BLOCK_P: tl.constexpr,  # tile over T (sequence)
    BLOCK_N: tl.constexpr,  # tile over D (feature)
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

    # Simple elementwise copy: out[b, p, n] = enc[b, p, n]
    for pi in range(BLOCK_P):
        p = p_start + pi
        if p < T:
            for ni in range(BLOCK_N):
                n = n_start + ni
                if n < D:
                    val = tl.load(enc_ptr + b * T * D + p * D + n)
                    # out has P rows, we only write first T rows
                    tl.store(out_ptr + b * P * D + p * D + n, val)


@triton.jit
def _copy_img_to_out(
    hst_ptr,      # *ptr to hidden_states: [B, I, D]
    out_ptr,      # *ptr to concatenated out: [B, P, D], we write from T onward
    B: tl.constexpr,
    I: tl.constexpr,
    D: tl.constexpr,
    T: tl.constexpr,
    BLOCK_P: tl.constexpr,  # tile over I (sequence of hidden)
    BLOCK_N: tl.constexpr,  # tile over D (feature)
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

    # Simple elementwise copy: out[b, T + p, n] = hst[b, p, n]
    for pi in range(BLOCK_P):
        p = p_start + pi
        if p < I:
            for ni in range(BLOCK_N):
                n = n_start + ni
                if n < D:
                    val = tl.load(hst_ptr + b * I * D + p * D + n)
                    tl.store(out_ptr + b * P * D + (T + p) * D + n, val)


@triton.jit
def _gemm_kernel(
    X_ptr,        # *ptr to concatenated input: [B, P, D], float32
    WT_ptr,       # *ptr to process_weight.T: [D, D], float32
    Y_ptr,        # *ptr to output: [B, P, D], float32
    B: tl.constexpr,
    P: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,  # tile over M (P)
    BLOCK_N: tl.constexpr,  # tile over N (D)
    BLOCK_K: tl.constexpr,  # reduction chunk over K (D)
):
    # Grid: (B, ceil(P / BLOCK_M), ceil(D / BLOCK_N))
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

    # Loop over K dimension (D), accumulate
    for k0 in range(0, D, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < D

        # Load X tile: [BM, BK], X[b, m, k]
        x_ptrs = X_ptr + b * P * D + m_offsets[:, None] * D + k_offsets[None, :]
        x_tile = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)

        # Load WT tile: [BK, BN], WT[k, n]
        wt_ptrs = WT_ptr + k_offsets[:, None] * D + n_offsets[None, :]
        wt_tile = tl.load(wt_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)

        # Accumulate
        acc += tl.dot(x_tile, wt_tile)

    # Store result: Y[b, m, n]
    y_ptrs = Y_ptr + b * P * D + m_offsets[:, None] * D + n_offsets[None, :]
    tl.store(y_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


@triton.jit
def _copy_slice_to_encoder(
    src_ptr,      # *ptr to Y: [B, P, D], float32
    dst_ptr,      # *ptr to processed_encoder: [B, T, D], float32
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

    m_offsets = m_start + tl.arange(0, BLOCK_M)  # over T
    n_offsets = n_start + tl.arange(0, BLOCK_N)  # over D

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
def _copy_slice_to_img(
    src_ptr,      # *ptr to Y: [B, P, D], float32
    dst_ptr,      # *ptr to processed_hidden: [B, I, D], float32
    B: tl.constexpr,
    I: tl.constexpr,
    D: tl.constexpr,
    T: tl.constexpr,
    BLOCK_M: tl.constexpr,  # tile over I
    BLOCK_N: tl.constexpr,  # tile over D
):
    # Grid: (B, ceil(I / BLOCK_M), ceil(D / BLOCK_N))
    b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N

    m_offsets = m_start + tl.arange(0, BLOCK_M)  # over I
    n_offsets = n_start + tl.arange(0, BLOCK_N)  # over D

    mask_m = m_offsets < I
    mask_n = n_offsets < D

    for mi in range(BLOCK_M):
        m = m_start + mi
        if m < I:
            for ni in range(BLOCK_N):
                n = n_start + ni
                if n < D:
                    val = tl.load(src_ptr + b * P * D + (T + m) * D + n)
                    tl.store(dst_ptr + b * I * D + m * D + n, val)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                encoder_hidden_states: torch.Tensor,
                process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-only implementation:
        - Concatenate encoder_hidden_states and hidden_states along sequence dimension without torch.cat.
        - Compute out = concatenated @ process_weight.T using a Triton GEMM (fp32 accumulation).
        - Split outputs via Triton copy kernels.
        Returns (processed_encoder, processed_hidden) as [B, T, D] and [B, I, D], respectively.
        """
        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        D = hidden_states.shape[2]
        P = T + I

        # Ensure inputs are contiguous. Operate in float32 for GEMM.
        enc = encoder_hidden_states.contiguous().to(torch.float32)
        hst = hidden_states.contiguous().to(torch.float32)
        wt = process_weight.contiguous().to(torch.float32)  # [D, D], no bias

        # Allocate concatenated tensor [B, P, D] in float32
        out_cat = torch.empty((B, P, D), dtype=torch.float32, device=enc.device)

        # 1) Copy encoder part into out_cat[:, :T, :]
        grid_enc = (B, triton.cdiv(T, 64), triton.cdiv(D, 64))
        _copy_encoder_to_out[grid_enc](enc, out_cat, B, T, D, BLOCK_P=64, BLOCK_N=64, num_warps=4, num_stages=2)

        # 2) Copy hidden part into out_cat[:, T:, :]
        grid_img = (B, triton.cdiv(I, 64), triton.cdiv(D, 64))
        _copy_img_to_out[grid_img](hst, out_cat, B, I, D, T, BLOCK_P=64, BLOCK_N=64, num_warps=4, num_stages=2)

        # 3) GEMM: out = out_cat @ wt.T, float32
        # Note: out_cat is [B, P, D], wt.T is [D, D]. Result is [B, P, D]
        grid_gemm = (B, triton.cdiv(P, 64), triton.cdiv(D, 64))
        _gemm_kernel[grid_gemm](out_cat, wt, out_cat, B, P, D, BLOCK_M=64, BLOCK_N=64, BLOCK_K=32, num_warps=4, num_stages=2)

        # 4) Split into processed_encoder and processed_hidden (float32 outputs)
        processed_encoder = torch.empty((B, T, D), dtype=torch.float32, device=enc.device)
        processed_hidden = torch.empty((B, I, D), dtype=torch.float32, device=enc.device)

        grid_e = (B, triton.cdiv(T, 64), triton.cdiv(D, 64))
        _copy_slice_to_encoder[grid_e](out_cat, processed_encoder, B, T, D, BLOCK_M=64, BLOCK_N=64, num_warps=4, num_stages=2)

        grid_h = (B, triton.cdiv(I, 64), triton.cdiv(D, 64))
        _copy_slice_to_img[grid_h](out_cat, processed_hidden, B, I, D, T, BLOCK_M=64, BLOCK_N=64, num_warps=4, num_stages=2)

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
