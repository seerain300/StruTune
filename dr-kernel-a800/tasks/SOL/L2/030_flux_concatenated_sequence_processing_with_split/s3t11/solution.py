import torch
import triton
import triton.language as tl


@triton.jit
def _copy_encoder_to_out(
    enc_ptr,      # *ptr to [B, T, D]
    out_ptr,      # *ptr to [B, P, D]
    B: tl.constexpr,
    T: tl.constexpr,
    D: tl.constexpr,
    BLOCK_P: tl.constexpr,  # tile along T
    BLOCK_N: tl.constexpr,  # tile along D
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

    base_enc = enc_ptr + b * T * D
    base_out = out_ptr + b * (T + 0) * D  # write to first T rows

    enc_ptrs = base_enc + p_offsets[:, None] * D + n_offsets[None, :]
    tile = tl.load(enc_ptrs, mask=mask_p[:, None] & mask_n[None, :], other=0.0)
    out_ptrs = base_out + p_offsets[:, None] * D + n_offsets[None, :]
    tl.store(out_ptrs, tile, mask=mask_p[:, None] & mask_n[None, :])


@triton.jit
def _copy_img_to_out(
    hst_ptr,      # *ptr to [B, I, D]
    out_ptr,      # *ptr to [B, P, D]
    B: tl.constexpr,
    I: tl.constexpr,
    D: tl.constexpr,
    T: tl.constexpr,        # offset for encoder rows
    BLOCK_P: tl.constexpr,  # tile along I (image seq)
    BLOCK_N: tl.constexpr,  # tile along D
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

    base_hst = hst_ptr + b * I * D
    # out starts writing at row T
    base_out = out_ptr + b * (T + 0) * D + T * D

    hst_ptrs = base_hst + p_offsets[:, None] * D + n_offsets[None, :]
    tile = tl.load(hst_ptrs, mask=mask_p[:, None] & mask_n[None, :], other=0.0)
    out_ptrs = base_out + p_offsets[:, None] * D + n_offsets[None, :]
    tl.store(out_ptrs, tile, mask=mask_p[:, None] & mask_n[None, :])


@triton.jit
def _gemm_bmm_kernel(
    X_ptr,        # *ptr to X: [B, P, D]
    WT_ptr,       # *ptr to W_T: [D, D]
    Y_ptr,        # *ptr to Y: [B, P, D]
    B: tl.constexpr,
    P: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,  # tile along M=P
    BLOCK_N: tl.constexpr,  # tile along N=D
    BLOCK_K: tl.constexpr,  # reduction tile along K=D
):
    # Grid: (B, ceil(P/BLOCK_M), ceil(D/BLOCK_N))
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

        # Load A tile: X[b, m, k] -> shape [BM, BK]
        a_ptrs = X_ptr + b * P * D + m_offsets[:, None] * D + k_offsets[None, :]
        a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)

        # Load B tile: WT[k, n] -> shape [BK, BN]
        b_ptrs = WT_ptr + k_offsets[:, None] * D + n_offsets[None, :]
        b_tile = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)

        # Accumulate
        acc += tl.dot(a, b_tile)

    # Store result Y[b, m, n]
    y_ptrs = Y_ptr + b * P * D + m_offsets[:, None] * D + n_offsets[None, :]
    tl.store(y_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


@triton.jit
def _copy_slice_to_encoder(
    src_ptr,      # *ptr to processed: [B, P, D]
    dst_ptr,      # *ptr to processed_encoder: [B, T, D]
    B: tl.constexpr,
    T: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,  # tile along T
    BLOCK_N: tl.constexpr,  # tile along D
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

    base_src = src_ptr + b * P * D
    base_dst = dst_ptr + b * T * D

    src_ptrs = base_src + m_offsets[:, None] * D + n_offsets[None, :]
    tile = tl.load(src_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0)
    dst_ptrs = base_dst + m_offsets[:, None] * D + n_offsets[None, :]
    tl.store(dst_ptrs, tile, mask=mask_m[:, None] & mask_n[None, :])


@triton.jit
def _copy_slice_to_hidden(
    src_ptr,      # *ptr to processed: [B, P, D]
    dst_ptr,      # *ptr to processed_hidden: [B, I, D]
    B: tl.constexpr,
    I: tl.constexpr,
    D: tl.constexpr,
    T: tl.constexpr,        # split offset
    BLOCK_M: tl.constexpr,  # tile along I
    BLOCK_N: tl.constexpr,  # tile along D
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

    base_src = src_ptr + b * P * D + T * D  # start from T-th row
    base_dst = dst_ptr + b * I * D

    src_ptrs = base_src + m_offsets[:, None] * D + n_offsets[None, :]
    tile = tl.load(src_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0)
    dst_ptrs = base_dst + m_offsets[:, None] * D + n_offsets[None, :]
    tl.store(dst_ptrs, tile, mask=mask_m[:, None] & mask_n[None, :])


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        Triton-optimized forward that avoids torch.cat and torch.matmul in the hot path:
        1) Concatenate encoder_hidden_states and hidden_states into 'out' via Triton copy kernels.
        2) Compute processed = out @ process_weight.T via Triton GEMM kernel.
        3) Split processed into processed_encoder and processed_hidden via Triton copy kernels.
        """
        # Ensure inputs are contiguous
        encoder_hidden_states = encoder_hidden_states.contiguous()
        hidden_states = hidden_states.contiguous()
        # process_weight is [D, D]; we'll pass its transpose as [D, D]
        process_weight_T = process_weight.t().contiguous()  # [D, D]

        B = encoder_hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        D = encoder_hidden_states.shape[2]
        P = T + I

        # 1) Concatenate into out: [B, P, D]
        out = torch.empty((B, P, D), device=hidden_states.device, dtype=torch.float32)

        # Tile sizes for copies
        BLOCK_P = 64
        BLOCK_N = 64

        # Copy encoder into first T rows
        grid1 = (B, triton.cdiv(T, BLOCK_P), triton.cdiv(D, BLOCK_N))
        _copy_encoder_to_out[grid1](
            encoder_hidden_states, out,
            B=B, T=T, D=D,
            BLOCK_P=BLOCK_P, BLOCK_N=BLOCK_N,
        )

        # Copy hidden into last I rows
        grid2 = (B, triton.cdiv(I, BLOCK_P), triton.cdiv(D, BLOCK_N))
        _copy_img_to_out[grid2](
            hidden_states, out,
            B=B, I=I, D=D, T=T,
            BLOCK_P=BLOCK_P, BLOCK_N=BLOCK_N,
        )

        # 2) GEMM: processed = out @ process_weight_T, shape [B, P, D]
        processed = torch.empty((B, P, D), device=hidden_states.device, dtype=torch.float32)

        BLOCK_M = 64
        BLOCK_N2 = 64
        BLOCK_K = 64
        grid3 = (B, triton.cdiv(P, BLOCK_M), triton.cdiv(D, BLOCK_N2))
        _gemm_bmm_kernel[grid3](
            out, process_weight_T,
            processed,
            B=B, P=P, D=D,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N2, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=3,
        )

        # 3) Split outputs
        processed_encoder = torch.empty((B, T, D), device=hidden_states.device, dtype=torch.float32)
        processed_hidden = torch.empty((B, I, D), device=hidden_states.device, dtype=torch.float32)

        BLOCK_M2 = 64
        BLOCK_N3 = 64
        grid4 = (B, triton.cdiv(T, BLOCK_M2), triton.cdiv(D, BLOCK_N3))
        _copy_slice_to_encoder[grid4](
            processed, processed_encoder,
            B=B, T=T, D=D,
            BLOCK_M=BLOCK_M2, BLOCK_N=BLOCK_N3,
        )

        grid5 = (B, triton.cdiv(I, BLOCK_M2), triton.cdiv(D, BLOCK_N3))
        _copy_slice_to_hidden[grid5](
            processed, processed_hidden,
            B=B, I=I, D=D, T=T,
            BLOCK_M=BLOCK_M2, BLOCK_N=BLOCK_N3,
        )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
