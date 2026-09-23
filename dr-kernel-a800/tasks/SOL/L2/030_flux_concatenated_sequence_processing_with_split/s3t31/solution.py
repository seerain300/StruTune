import torch
import triton
import triton.language as tl


@triton.jit
def _copy_encoder_to_out(
    enc_ptr,      # *ptr to encoder_hidden_states: [B, T, D]
    out_ptr,      # *ptr to concatenated output: [B, P, D]
    B: tl.constexpr,
    T: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,  # tile over M=T (sequence)
    BLOCK_N: tl.constexpr,  # tile over N=D (feature)
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
                    val = tl.load(enc_ptr + b * T * D + m * D + n)
                    tl.store(out_ptr + b * T * D + m * D + n, val)


@triton.jit
def _copy_img_to_out(
    hst_ptr,      # *ptr to hidden_states: [B, I, D]
    out_ptr,      # *ptr to concatenated output: [B, P, D]
    B: tl.constexpr,
    I: tl.constexpr,
    D: tl.constexpr,
    T: tl.constexpr,         # P = T + I
    BLOCK_M: tl.constexpr,   # tile over M=I
    BLOCK_N: tl.constexpr,   # tile over N=D
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
            for ni in range(BLOCK_N):
                n = n_start + ni
                if n < D:
                    val = tl.load(hst_ptr + b * I * D + m * D + n)
                    # out indices: (m + T), n
                    tl.store(out_ptr + b * P * D + (m + T) * D + n, val)


@triton.jit
def _copy_slice_encoder(
    src_ptr,      # *ptr to processed: [B, P, D]
    dst_ptr,      # *ptr to processed_encoder: [B, T, D]
    B: tl.constexpr,
    T: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
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
                    val = tl.load(src_ptr + b * T * D + m * D + n)
                    tl.store(dst_ptr + b * T * D + m * D + n, val)


@triton.jit
def _copy_slice_hidden(
    src_ptr,      # *ptr to processed: [B, P, D]
    dst_ptr,      # *ptr to processed_hidden: [B, I, D]
    B: tl.constexpr,
    I: tl.constexpr,
    D: tl.constexpr,
    T: tl.constexpr,         # P = T + I
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
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
            for ni in range(BLOCK_N):
                n = n_start + ni
                if n < D:
                    val = tl.load(src_ptr + b * P * D + (m + T) * D + n)
                    tl.store(dst_ptr + b * I * D + m * D + n, val)


class ModelNew(torch.nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        process_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Triton requires CUDA tensors
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Inputs must be CUDA tensors for Triton."
        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        D = hidden_states.shape[2]
        P = T + I

        # Ensure contiguity
        enc = encoder_hidden_states.contiguous()
        hst = hidden_states.contiguous()
        WT = process_weight.t().contiguous()  # [D, D]

        # 1) Concatenate without torch.cat using Triton
        out = torch.empty((B, P, D), dtype=enc.dtype, device=enc.device)

        BLOCK_M = 128
        BLOCK_N = 128

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

        # 2) Linear projection using PyTorch (fast and reliable)
        processed = torch.matmul(out, WT)  # [B, P, D]

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

        grid_slice


def run(*args):
    return ModelNew()(*args)
