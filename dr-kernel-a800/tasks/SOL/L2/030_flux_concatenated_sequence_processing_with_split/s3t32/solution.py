import torch
import triton
import triton.language as tl


@triton.jit
def _copy_encoder_to_out(
    enc_ptr,      # *ptr to encoder_hidden_states: [B, T, D]
    out_ptr,      # *ptr to concatenated out: [B, P, D], P=T+I (we only write first T)
    B: tl.constexpr,
    T: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,  # tile over T (sequence)
    BLOCK_N: tl.constexpr,  # tile over D (feature)
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
                    val = tl.load(enc_ptr + b * T * D + m * D + n)
                    # out[b, m, n] where m in [0, T)
                    tl.store(out_ptr + b * (T + 0) * D + m * D + n, val)


@triton.jit
def _copy_img_to_out(
    hst_ptr,      # *ptr to hidden_states: [B, I, D]
    out_ptr,      # *ptr to concatenated out: [B, P, D]
    B: tl.constexpr,
    I: tl.constexpr,
    D: tl.constexpr,
    T: tl.constexpr,        # P = T + I
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
        i = m_start + mi
        if i < I:
            for ni in range(BLOCK_N):
                n = n_start + ni
                if n < D:
                    val = tl.load(hst_ptr + b * I * D + i * D + n)
                    # out[b, T + i, n]
                    tl.store(out_ptr + b * (T + I) * D + (T + i) * D + n, val)


@triton.jit
def _gemm_bmm(
    In_ptr,       # *ptr to input tensor: [B, P, D]
    WT_ptr,       # *ptr to process_weight.T: [D, D]
    Out_ptr,      # *ptr to output tensor: [B, P, D]
    B: tl.constexpr,
    P: tl.constexpr,  # M = P
    D: tl.constexpr,  # N and K
    BLOCK_M: tl.constexpr,  # tile over M (unused; one program handles one (b,p))
    BLOCK_N: tl.constexpr,  # tile over N (feature)
):
    # Grid: (B, P, ceil(D / BLOCK_N))
    b = tl.program_id(0)
    p = tl.program_id(1)
    pid_n = tl.program_id(2)

    n_start = pid_n * BLOCK_N
    n_offsets = n_start + tl.arange(0, BLOCK_N)
    mask_n = n_offsets < D

    # Accumulate in fp32
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # Loop over K dimension (D) and accumulate
    for k in range(0, D):
        in_val = tl.load(In_ptr + b * P * D + p * D + k)  # input scalar
        wt_val = tl.load(WT_ptr + k * D + n_offsets)      # weight row [BLOCK_N]
        acc += in_val * wt_val  # elementwise multiply, reduce per n

    tl.store(Out_ptr + b * P * D + p * D + n_offsets, acc, mask=mask_n)


@triton.jit
def _copy_slice_encoder(
    src_ptr,      # *ptr to processed: [B, P, D]
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
            for ni in range(BLOCK_N):
                n = n_start + ni
                if n < D:
                    val = tl.load(src_ptr + b * (T + 0) * D + m * D + n)
                    tl.store(dst_ptr + b * T * D + m * D + n, val)


@triton.jit
def _copy_slice_hidden(
    src_ptr,      # *ptr to processed: [B, P, D]
    dst_ptr,      # *ptr to processed_hidden: [B, I, D]
    B: tl.constexpr,
    I: tl.constexpr,
    D: tl.constexpr,
    T: tl.constexpr,        # src region starts at column T
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
        i = m_start + mi
        if i < I:
            for ni in range(BLOCK_N):
                n = n_start + ni
                if n < D:
                    val = tl.load(src_ptr + b * (T + I) * D + (T + i) * D + n)
                    tl.store(dst_ptr + b * I * D + i * D + n, val)


class ModelNew(torch.nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        process_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Triton requires CUDA tensors
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Inputs must be CUDA tensors for Triton."

        # Ensure contiguity
        enc = encoder_hidden_states.contiguous()    # [B, T, D]
        hst = hidden_states.contiguous()           # [B, I, D]
        WT = process_weight.t().contiguous()       # [D, D]

        B = enc.shape[0]
        T = enc.shape[1]
        I = hst.shape[1]
        D = hst.shape[2]
        P = T + I

        # 1) Concatenate without torch.cat using Triton
        out = torch.empty((B, P, D), dtype=enc.dtype, device=enc.device)
        BLOCK_M = 64
        BLOCK_N = 64

        grid_enc = (B, triton.cdiv(T, BLOCK_M), triton.cdiv(D, BLOCK_N))
        grid_img = (B, triton.cdiv(I, BLOCK_M), triton.cdiv(D, BLOCK_N))

        _copy_encoder_to_out[grid_enc](
            enc, out,
            B, T, D,
            BLOCK_M, BLOCK_N,
            num_warps=4, num_stages=2
        )
        _copy_img_to_out[grid_img](
            hst, out,
            B, I, D, T,
            BLOCK_M, BLOCK_N,
            num_warps=4, num_stages=2
        )

        # 2) Linear projection using Triton GEMM (no torch.matmul)
        processed = torch.empty((B, P, D), dtype=enc.dtype, device=enc.device)
        # Launch one program per (b, p, tile_n). We use a 3D grid where pid_n tiles D.
        BLOCK_N2 = 64
        grid_gemm = (B, P, triton.cdiv(D, BLOCK_N2))
        _gemm_bmm[grid_gemm](
            out, WT, processed,
            B, P, D,
            1, BLOCK_N2,  # BLOCK_M is unused; using 1 keeps signature; BLOCK_N2 is the feature tile
            num_warps=4, num_stages=2
        )

        # 3) Split using Triton copy kernels
        processed_encoder = torch.empty((B, T, D), dtype=processed.dtype, device=processed.device)
        processed_hidden = torch.empty((B, I, D), dtype=processed.dtype, device=processed.device)

        grid_slice_enc = (B, triton.cdiv(T, BLOCK_M), triton.cdiv(D, BLOCK_N))
        _copy_slice_encoder[grid_slice_enc](
            processed, processed_encoder,
            B, T, D,
            BLOCK_M, BLOCK_N,
            num_warps=4, num_stages=2
        )

        grid_slice_hidden = (B, triton.cdiv(I, BLOCK_M), triton.cdiv(D, BLOCK_N))
        _copy_slice_hidden[grid_slice_hidden](
            processed, processed_hidden,
            B, I, D, T,
            BLOCK_M, BLOCK_N,
            num_warps=4, num_stages=2
        )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
