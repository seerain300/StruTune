import torch
import triton
import triton.language as tl


@triton.jit
def _copy_encoder_to_out(
    enc_ptr,      # *ptr to encoder_hidden_states: [B, T, D]
    out_ptr,      # *ptr to concatenated out: [B, P, D], we write to first T rows
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

    base_enc = enc_ptr + b * T * D
    base_out = out_ptr + b * (T + 0) * D  # write to first T rows

    enc_ptrs = base_enc + p_offsets[:, None] * D + n_offsets[None, :]
    tile = tl.load(enc_ptrs, mask=mask_p[:, None] & mask_n[None, :], other=0.0)
    out_ptrs = base_out + p_offsets[:, None] * D + n_offsets[None, :]
    tl.store(out_ptrs, tile, mask=mask_p[:, None] & mask_n[None, :])


@triton.jit
def _copy_img_to_out(
    hst_ptr,      # *ptr to hidden_states: [B, I, D]
    out_ptr,      # *ptr to concatenated out: [B, P, D]
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
def _gemm_cat_to_out(
    X_ptr,        # *ptr to concatenated tensor: [B, P, D]
    WT_ptr,       # *ptr to process_weight.T: [D, D]
    Y_ptr,        # *ptr to output processed: [B, P, D]
    B: tl.constexpr,
    P: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,  # tile along M (sequence P)
    BLOCK_N: tl.constexpr,  # tile along N (feature D)
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

    # Loop over reduction dimension K in chunks
    for k0 in range(0, D, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < D

        # Load X tile [BM, BK]: X[b, m, k]
        x_ptrs = X_ptr + b * P * D + m_offsets[:, None] * D + k_offsets[None, :]
        x_tile = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)

        # Load WT tile [BK, BN]: WT[k, n]
        wt_ptrs = WT_ptr + k_offsets[:, None] * D + n_offsets[None, :]
        wt_tile = tl.load(wt_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)

        acc += tl.dot(x_tile, wt_tile)

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

    base_src = src_ptr + b * P * D  # P is not used; src[:, :T, :]
    # src row index is limited by T, but pointer arithmetic uses m_offsets (<= T)
    src_ptrs = base_src + m_offsets[:, None] * D + n_offsets[None, :]
    tile = tl.load(src_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0)

    base_dst = dst_ptr + b * T * D
    dst_ptrs = base_dst + m_offsets[:, None] * D + n_offsets[None, :]
    tl.store(dst_ptrs, tile, mask=mask_m[:, None] & mask_n[None, :])


@triton.jit
def _copy_slice_to_hidden(
    src_ptr,      # *ptr to processed: [B, P, D]
    dst_ptr,      # *ptr to processed_hidden: [B, I, D]
    B: tl.constexpr,
    I: tl.constexpr,
    D: tl.constexpr,
    T: tl.constexpr,        # offset to start slice in src
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

    base_src = src_ptr + b * P * D + T * D  # start at P rows offset by T
    src_ptrs = base_src + m_offsets[:, None] * D + n_offsets[None, :]
    tile = tl.load(src_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0)

    base_dst = dst_ptr + b * I * D
    dst_ptrs = base_dst + m_offsets[:, None] * D + n_offsets[None, :]
    tl.store(dst_ptrs, tile, mask=mask_m[:, None] & mask_n[None, :])


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        Triton-only implementation of:
          concatenated = torch.cat([encoder_hidden_states, hidden_states], dim=1)
          processed = concatenated @ process_weight.T
          processed_encoder = processed[:, :text_seq_len, :]
          processed_hidden = processed[:, text_seq_len:, :]
        """
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "All tensors must be CUDA tensors"
        assert hidden_states.dtype in (torch.float32, torch.float16, torch.bfloat16) and process_weight.dtype in (torch.float32, torch.float16, torch.bfloat16), "Unsupported dtype"

        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        D = hidden_states.shape[2]
        P = T + I

        # Ensure inputs are contiguous
        enc = encoder_hidden_states.contiguous()
        hst = hidden_states.contiguous()
        wt_T = process_weight.t().contiguous()  # [D, D]
        # Allocate output concatenation
        out = torch.empty((B, P, D), device=hidden_states.device, dtype=hidden_states.dtype)
        # Concatenation: copy encoder and hidden into out
        # Choose tile sizes for copy kernels
        BLOCK_P = 64
        BLOCK_N = 64
        _copy_encoder_to_out[(B, triton.cdiv(T, BLOCK_P), triton.cdiv(D, BLOCK_N))](
            enc, out, B, T, D, BLOCK_P, BLOCK_N
        )
        _copy_img_to_out[(B, triton.cdiv(I, BLOCK_P), triton.cdiv(D, BLOCK_N))](
            hst, out, B, I, D, T, BLOCK_P, BLOCK_N
        )

        # Allocate processed output
        processed = torch.empty((B, P, D), device=hidden_states.device, dtype=hidden_states.dtype)

        # GEMM: processed = out @ wt_T
        # Launch Triton GEMM kernel
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 64
        grid = (B, triton.cdiv(P, BLOCK_M), triton.cdiv(D, BLOCK_N))
        _gemm_cat_to_out[grid](
            out, wt_T, processed, B, P, D, BLOCK_M, BLOCK_N, BLOCK_K,
            num_warps=4, num_stages=3
        )

        # Split into two outputs
        processed_encoder = torch.empty((B, T, D), device=hidden_states.device, dtype=hidden_states.dtype)
        processed_hidden = torch.empty((B, I, D), device=hidden_states.device, dtype=hidden_states.dtype)
        # Choose tile sizes for slice copy kernels
        BLOCK_M = 64
        BLOCK_N = 64
        _copy_slice_to_encoder[(B, triton.cdiv(T, BLOCK_M), triton.cdiv(D, BLOCK_N))](
            processed, processed_encoder, B, T, D, BLOCK_M, BLOCK_N
        )
        _copy_slice_to_hidden[(B, triton.cdiv(I, BLOCK_M), triton.cdiv(D, BLOCK_N))](
            processed, processed_hidden, B, I, D, T, BLOCK_M, BLOCK_N
        )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
