import torch
import triton
import triton.language as tl


@triton.jit
def _copy_encoder_to_out(
    enc_ptr,      # *ptr to [B, T, D]
    out_ptr,      # *ptr to [B, P, D], we write to first T rows
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
    BLOCK_P: tl.constexpr,  # tile along I
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
    base_out = out_ptr + b * (T + 0) * D + T * D  # start writing at row T

    hst_ptrs = base_hst + p_offsets[:, None] * D + n_offsets[None, :]
    tile = tl.load(hst_ptrs, mask=mask_p[:, None] & mask_n[None, :], other=0.0)
    out_ptrs = base_out + p_offsets[:, None] * D + n_offsets[None, :]
    tl.store(out_ptrs, tile, mask=mask_p[:, None] & mask_n[None, :])


@triton.jit
def _matmul_gemv_batched(
    A_ptr,        # *ptr to X: [B, M, K], here M = P, K = D
    B_ptr,        # *ptr to WT: [K, N], here WT = process_weight.T
    C_ptr,        # *ptr to Y: [B, M, N], here M = P, N = D
    B: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Grid over (B, tiles of M, tiles of N)
    b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N

    m_offsets = m_start + tl.arange(0, BLOCK_M)
    n_offsets = n_start + tl.arange(0, BLOCK_N)

    mask_m = m_offsets < M
    mask_n = n_offsets < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < K

        # Load A tile [BM, BK]: A[b, m, k]
        a_ptrs = A_ptr + b * M * K + m_offsets[:, None] * K + k_offsets[None, :]
        a_tile = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)

        # Load B tile [BK, BN]: B[k, n] == WT[k, n]
        b_ptrs = B_ptr + k_offsets[:, None] * N + n_offsets[None, :]
        b_tile = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)

        acc += tl.dot(a_tile, b_tile)

    # Store results into C[b, m, n]
    c_ptrs = C_ptr + b * M * N + m_offsets[:, None] * N + n_offsets[None, :]
    tl.store(c_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


@triton.jit
def _copy_slice_to_encoder(
    src_ptr,      # *ptr to processed: [B, P, D]
    dst_ptr,      # *ptr to processed_encoder: [B, T, D]
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

    src_base = src_ptr + b * P * D
    dst_base = dst_ptr + b * T * D

    src_ptrs = src_base + p_offsets[:, None] * D + n_offsets[None, :]
    tile = tl.load(src_ptrs, mask=mask_p[:, None] & mask_n[None, :], other=0.0)
    dst_ptrs = dst_base + p_offsets[:, None] * D + n_offsets[None, :]
    tl.store(dst_ptrs, tile, mask=mask_p[:, None] & mask_n[None, :])


@triton.jit
def _copy_slice_to_hidden(
    src_ptr,      # *ptr to processed: [B, P, D]
    dst_ptr,      # *ptr to processed_hidden: [B, I, D]
    B: tl.constexpr,
    I: tl.constexpr,
    D: tl.constexpr,
    T: tl.constexpr,        # offset to start copying from
    BLOCK_P: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Grid: (B, ceil(I / BLOCK_P), ceil(D / BLOCK_N))
    b = tl.program_id(0)
    pid_p = tl.program_id(1)
    pid_n = tl.program_id(2)

    p_start = pid_p * BLOCK_P
    n_start = pid_n * BLOCK_N

    p_offsets = p_start + tl.arange(0, BLOCK_P)  # along I (starting from T)
    n_offsets = n_start + tl.arange(0, BLOCK_N)  # along D

    mask_p = p_offsets < I
    mask_n = n_offsets < D

    src_base = src_ptr + b * P * D
    dst_base = dst_ptr + b * I * D

    src_ptrs = src_base + (p_offsets[:, None] + T) * D + n_offsets[None, :]
    tile = tl.load(src_ptrs, mask=mask_p[:, None] & mask_n[None, :], other=0.0)
    dst_ptrs = dst_base + p_offsets[:, None] * D + n_offsets[None, :]
    tl.store(dst_ptrs, tile, mask=mask_p[:, None] & mask_n[None, :])


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        # Shapes
        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        D = encoder_hidden_states.shape[2]
        P = T + I

        # Ensure contiguous and float32
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Inputs must be on CUDA"
        hidden = hidden_states.contiguous().to(torch.float32)
        enc = encoder_hidden_states.contiguous().to(torch.float32)
        wt = process_weight.contiguous().to(torch.float32)  # [D, D], need [K, N] where K=D, N=D

        # 1) Concatenate: out [B, P, D]
        out = torch.empty((B, P, D), device=hidden.device, dtype=torch.float32)

        # Launch concat kernels
        # Encode part
        grid_enc = (B, triton.cdiv(T, 64), triton.cdiv(D, 64))
        _copy_encoder_to_out[grid_enc](
            enc, out, B, T, D,
            BLOCK_P=64, BLOCK_N=64
        )
        # Image part
        grid_img = (B, triton.cdiv(I, 64), triton.cdiv(D, 64))
        _copy_img_to_out[grid_img](
            hidden, out, B, I, D, T,
            BLOCK_P=64, BLOCK_N=64
        )

        # 2) GEMM: processed = out @ wt.T, where wt.T is wt.view(D, D)
        processed = torch.empty((B, P, D), device=hidden.device, dtype=torch.float32)

        # Launch GEMM kernel
        grid_gemm = (B, triton.cdiv(P, 64), triton.cdiv(D, 64))
        _matmul_gemv_batched[grid_gemm](
            out, wt, processed, B, P, D, D,
            BLOCK_M=64, BLOCK_N=64, BLOCK_K=64,
            num_warps=4, num_stages=3
        )

        # 3) Split
        processed_encoder = torch.empty((B, T, D), device=hidden.device, dtype=torch.float32)
        processed_hidden = torch.empty((B, I, D), device=hidden.device, dtype=torch.float32)

        grid_split = (B, triton.cdiv(T, 64), triton.cdiv(D, 64))
        _copy_slice_to_encoder[grid_split](
            processed, processed_encoder, B, T, D, 64, 64
        )

        grid_split2 = (B, triton.cdiv(I, 64), triton.cdiv(D, 64))
        _copy_slice_to_hidden[grid_split2](
            processed, processed_hidden, B, I, D, T, 64, 64
        )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
