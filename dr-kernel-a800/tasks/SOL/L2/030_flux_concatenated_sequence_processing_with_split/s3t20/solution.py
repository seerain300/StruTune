import torch
import triton
import triton.language as tl


@triton.jit
def _copy_encoder_to_out(
    enc_ptr,      # *ptr to encoder_hidden_states: [B, T, D]
    out_ptr,      # *ptr to concatenated out: [B, P, D], we write first T
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
                    tl.store(out_ptr + b * (T + 0) * D + p * D + n, val)


@triton.jit
def _copy_img_to_out(
    hst_ptr,      # *ptr to hidden_states: [B, I, D]
    out_ptr,      # *ptr to concatenated out: [B, P, D], P = T + I
    B: tl.constexpr,
    I: tl.constexpr,
    D: tl.constexpr,
    T: tl.constexpr,
    BLOCK_P: tl.constexpr,  # tile along I (img_seq_len)
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

    for pi in range(BLOCK_P):
        p = p_start + pi
        if p < I:
            # store into out[:, T + p, :]
            for ni in range(BLOCK_N):
                n = n_start + ni
                if n < D:
                    val = tl.load(hst_ptr + b * I * D + p * D + n)
                    tl.store(out_ptr + b * (T + I) * D + (T + p) * D + n, val)


@triton.jit
def _gemm_kernel(
    A_ptr,        # *ptr to concatenated: [B, P, D]
    B_ptr,        # *ptr to process_weight.T: [D, D]
    C_ptr,        # *ptr to output processed: [B, P, D]
    Bsz: tl.constexpr,  # number of batches
    P: tl.constexpr,    # sequence length (T + I)
    D: tl.constexpr,    # hidden_dim
    BLOCK_M: tl.constexpr,  # tile size along M (P)
    BLOCK_N: tl.constexpr,  # tile size along N (D)
    BLOCK_K: tl.constexpr,  # reduction tile along K (D)
):
    # Grid: (Bsz, ceil(P / BLOCK_M), ceil(D / BLOCK_N))
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

        # Load A tile: A[b, m, k] -> shape [BLOCK_M, BLOCK_K]
        a_ptrs = A_ptr + b * P * D + m_offsets[:, None] * D + k_offsets[None, :]
        a_tile = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)

        # Load B tile: B[k, n] -> shape [BLOCK_K, BLOCK_N]
        b_ptrs = B_ptr + k_offsets[:, None] * D + n_offsets[None, :]
        b_tile = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)

        # Accumulate
        acc += tl.dot(a_tile, b_tile)

    # Store C: C[b, m, n] = acc
    c_ptrs = C_ptr + b * P * D + m_offsets[:, None] * D + n_offsets[None, :]
    tl.store(c_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


@triton.jit
def _copy_first_slice(
    src_ptr,      # *ptr to processed: [B, P, D]
    dst_ptr,      # *ptr to processed_encoder: [B, T, D]
    Bsz: tl.constexpr,
    T: tl.constexpr,
    D: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Grid: (Bsz, ceil(T / BLOCK_P), ceil(D / BLOCK_N))
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
                    val = tl.load(src_ptr + b * (T + 0) * D + p * D + n)
                    tl.store(dst_ptr + b * T * D + p * D + n, val)


@triton.jit
def _copy_second_slice(
    src_ptr,      # *ptr to processed: [B, P, D]
    dst_ptr,      # *ptr to processed_hidden: [B, I, D]
    Bsz: tl.constexpr,
    I: tl.constexpr,
    D: tl.constexpr,
    T: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Grid: (Bsz, ceil(I / BLOCK_P), ceil(D / BLOCK_N))
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
                    val = tl.load(src_ptr + b * (T + I) * D + (T + p) * D + n)
                    tl.store(dst_ptr + b * I * D + p * D + n, val)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        # Shapes
        Bsz = hidden_states.shape[0]
        I = hidden_states.shape[1]  # img_seq_len
        D = hidden_states.shape[2]
        T = encoder_hidden_states.shape[1]  # text_seq_len
        P = T + I
        assert encoder_hidden_states.shape[0] == Bsz and encoder_hidden_states.shape[2] == D
        assert process_weight.shape == (D, D)

        # Ensure contiguous
        enc = encoder_hidden_states.contiguous()
        hst = hidden_states.contiguous()
        wt_t = process_weight.t().contiguous()  # [D, D]

        # Allocate concatenated input [B, P, D]
        out = torch.empty((Bsz, P, D), dtype=enc.dtype, device=enc.device)

        # Kernel 1: copy encoder part
        grid_copy_enc = (Bsz, triton.cdiv(T, 128), triton.cdiv(D, 64))
        _copy_encoder_to_out[grid_copy_enc](
            enc, out,
            Bsz, T, D,
            BLOCK_P=128, BLOCK_N=64,
        )

        # Kernel 2: copy image part
        grid_copy_img = (Bsz, triton.cdiv(I, 128), triton.cdiv(D, 64))
        _copy_img_to_out[grid_copy_img](
            hst, out,
            Bsz, I, D, T,
            BLOCK_P=128, BLOCK_N=64,
        )

        # Allocate output processed [B, P, D]
        processed = torch.empty((Bsz, P, D), dtype=out.dtype, device=out.device)

        # GEMM kernel: processed = out @ wt_t
        grid_gemm = (Bsz, triton.cdiv(P, 128), triton.cdiv(D, 64))
        _gemm_kernel[grid_gemm](
            out, wt_t, processed,
            Bsz, P, D,
            BLOCK_M=128, BLOCK_N=64, BLOCK_K=64,
        )

        # Allocate outputs
        processed_encoder = torch.empty((Bsz, T, D), dtype=processed.dtype, device=processed.device)
        processed_hidden = torch.empty((Bsz, I, D), dtype=processed.dtype, device=processed.device)

        # Kernel 3: copy first slice (encoder stream)
        grid_slice1 = (Bsz, triton.cdiv(T, 128), triton.cdiv(D, 64))
        _copy_first_slice[grid_slice1](
            processed, processed_encoder,
            Bsz, T, D,
            BLOCK_P=128, BLOCK_N=64,
        )

        # Kernel 4: copy second slice (image stream)
        grid_slice2 = (Bsz, triton.cdiv(I, 128), triton.cdiv(D, 64))
        _copy_second_slice[grid_slice2](
            processed, processed_hidden,
            Bsz, I, D, T,
            BLOCK_P=128, BLOCK_N=64,
        )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
