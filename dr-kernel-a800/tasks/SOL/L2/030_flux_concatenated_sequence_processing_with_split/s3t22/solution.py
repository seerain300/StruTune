import torch
import triton
import triton.language as tl


@triton.jit
def _concat_sequences_kernel(
    enc_ptr,      # *ptr to encoder_hidden_states: [B, T, D]
    hst_ptr,      # *ptr to hidden_states: [B, I, D]
    out_ptr,      # *ptr to concatenated output: [B, P, D]
    B: tl.constexpr,
    T: tl.constexpr,
    I: tl.constexpr,
    D: tl.constexpr,
    BLOCK_P: tl.constexpr,  # tile size along T (sequence)
    BLOCK_N: tl.constexpr,  # tile size along D (feature)
):
    # Grid: (B, ceil(T/BLOCK_P), ceil(D/BLOCK_N))
    b = tl.program_id(0)
    pid_p = tl.program_id(1)
    pid_n = tl.program_id(2)

    p_start = pid_p * BLOCK_P
    n_start = pid_n * BLOCK_N

    p_offsets = p_start + tl.arange(0, BLOCK_P)  # along T
    n_offsets = n_start + tl.arange(0, BLOCK_N)  # along D

    mask_p = p_offsets < T
    mask_n = n_offsets < D

    # First, copy encoder part: out[b, p, n] for p in [0, T), n in [0, D)
    for pi in range(BLOCK_P):
        p = p_start + pi
        if p < T:
            for ni in range(BLOCK_N):
                n = n_start + ni
                if n < D:
                    val = tl.load(enc_ptr + b * T * D + p * D + n)
                    tl.store(out_ptr + b * (T + I) * D + p * D + n, val)

    # Then, copy image part: out[b, p, n] for p in [T, T+I), n in [0, D)
    for pi in range(BLOCK_P):
        p = p_start + tl.arange(0, BLOCK_P)  # vector
        # We need to shift by T: p_img = p_start + tl.arange(0, BLOCK_P) + T
        p_img = p_start + tl.arange(0, BLOCK_P) + T
        # Select only valid p_img < (T + I)
        mask_p_img = (p_img < (T + I))
        # Load from hidden_states
        hst_ptrs = hst_ptr + b * I * D + (p_img - T)[:, None] * D + n_offsets[None, :]
        mask = mask_p_img[:, None] & mask_n[None, :]
        hst_tile = tl.load(hst_ptrs, mask=mask, other=0.0)
        # Store to out at positions [b, p_img, n]
        out_ptrs = out_ptr + b * (T + I) * D + p_img[:, None] * D + n_offsets[None, :]
        tl.store(out_ptrs, hst_tile, mask=mask)


@triton.jit
def _gemm_kernel(
    X_ptr,        # *ptr to concatenated input: [B, P, D]
    WT_ptr,       # *ptr to process_weight.T: [D, D]
    Y_ptr,        # *ptr to output: [B, P, D]
    B: tl.constexpr,
    P: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,  # along P (sequence)
    BLOCK_N: tl.constexpr,  # along D (feature)
    BLOCK_K: tl.constexpr,  # reduction chunk
):
    # Grid: (B, ceil(P/BLOCK_M), ceil(D/BLOCK_N))
    b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N

    m_offsets = m_start + tl.arange(0, BLOCK_M)  # along P
    n_offsets = n_start + tl.arange(0, BLOCK_N)  # along D

    mask_m = m_offsets < P
    mask_n = n_offsets < D

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Reduction over K
    for k0 in range(0, D, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < D

        # Load X tile [BM, BK]: X[b, m, k]
        x_ptrs = X_ptr + b * P * D + m_offsets[:, None] * D + k_offsets[None, :]
        x_tile = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)

        # Load WT tile [BK, BN]: WT[k, n]
        wt_ptrs = WT_ptr + k_offsets[:, None] * D + n_offsets[None, :]
        wt_tile = tl.load(wt_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)

        # Accumulate
        acc += tl.dot(x_tile, wt_tile)

    # Store result Y[b, m, n]
    y_ptrs = Y_ptr + b * P * D + m_offsets[:, None] * D + n_offsets[None, :]
    tl.store(y_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


@triton.jit
def _copy_slice_kernel_encoder(
    src_ptr,      # *ptr to Y: [B, P, D]
    dst_ptr,      # *ptr to processed_encoder: [B, T, D]
    B: tl.constexpr,
    T: tl.constexpr,
    D: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Grid: (B, ceil(T/BLOCK_P), ceil(D/BLOCK_N))
    b = tl.program_id(0)
    pid_p = tl.program_id(1)
    pid_n = tl.program_id(2)

    p_start = pid_p * BLOCK_P
    n_start = pid_n * BLOCK_N

    p_offsets = p_start + tl.arange(0, BLOCK_P)  # along T
    n_offsets = n_start + tl.arange(0, BLOCK_N)  # along D

    mask_p = p_offsets < T
    mask_n = n_offsets < D

    src_ptrs = src_ptr + b * P * D + p_offsets[:, None] * D + n_offsets[None, :]
    dst_ptrs = dst_ptr + b * T * D + p_offsets[:, None] * D + n_offsets[None, :]
    vals = tl.load(src_ptrs, mask=mask_p[:, None] & mask_n[None, :], other=0.0)
    tl.store(dst_ptrs, vals, mask=mask_p[:, None] & mask_n[None, :])


@triton.jit
def _copy_slice_kernel_img(
    src_ptr,      # *ptr to Y: [B, P, D]
    dst_ptr,      # *ptr to processed_hidden: [B, I, D]
    B: tl.constexpr,
    I: tl.constexpr,
    D: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Grid: (B, ceil(I/BLOCK_P), ceil(D/BLOCK_N))
    b = tl.program_id(0)
    pid_p = tl.program_id(1)
    pid_n = tl.program_id(2)

    p_start = pid_p * BLOCK_P
    n_start = pid_n * BLOCK_N

    p_offsets = p_start + tl.arange(0, BLOCK_P)  # along I
    n_offsets = n_start + tl.arange(0, BLOCK_N)  # along D

    mask_p = p_offsets < I
    mask_n = n_offsets < D

    # src indices correspond to T + I + p
    src_ptrs = src_ptr + b * P * D + (p_offsets[:, None] + T) * D + n_offsets[None, :]
    dst_ptrs = dst_ptr + b * I * D + p_offsets[:, None] * D + n_offsets[None, :]
    vals = tl.load(src_ptrs, mask=mask_p[:, None] & mask_n[None, :], other=0.0)
    tl.store(dst_ptrs, vals, mask=mask_p[:, None] & mask_n[None, :])


class ModelNew(torch.nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        process_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized forward:
        - Concatenate encoder_hidden_states and hidden_states along sequence dim using Triton.
        - Apply linear projection using Triton GEMM.
        - Split back into two streams using Triton slice-copy kernels.
        """
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Tensors must be on CUDA for Triton kernels"
        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        D = hidden_states.shape[2]
        P = T + I

        # Ensure contiguous
        enc = encoder_hidden_states.contiguous()
        hst = hidden_states.contiguous()
        wt = process_weight.transpose(0, 1).contiguous()  # process_weight.T: [D, D]
        # We'll compute GEMM in fp32
        # Allocate concatenated input [B, P, D] in fp32 for safety
        out = torch.empty((B, P, D), device=hidden_states.device, dtype=torch.float32)
        # Triton concat
        grid_concat = (B, triton.cdiv(T, 64), triton.cdiv(D, 64))
        _concat_sequences_kernel[grid_concat](
            enc, hst, out,
            B, T, I, D,
            BLOCK_P=64, BLOCK_N=64,
            num_warps=4, num_stages=2,
        )

        # Allocate output processed [B, P, D] in fp32
        y = torch.empty((B, P, D), device=hidden_states.device, dtype=torch.float32)
        # Triton GEMM
        grid_gemm = (B, triton.cdiv(P, 64), triton.cdiv(D, 64))
        _gemm_kernel[grid_gemm](
            out, wt, y,
            B, P, D,
            BLOCK_M=64, BLOCK_N=64, BLOCK_K=64,
            num_warps=4, num_stages=2,
        )

        # Prepare outputs (convert back to original dtype if needed)
        # Note: original run returns tensors with the same dtype as inputs; here we assume fp32 for robustness.
        processed_encoder = torch.empty((B, T, D), device=hidden_states.device, dtype=torch.float32)
        processed_hidden = torch.empty((B, I, D), device=hidden_states.device, dtype=torch.float32)

        grid_split_e = (B, triton.cdiv(T, 64), triton.cdiv(D, 64))
        _copy_slice_kernel_encoder[grid_split_e](
            y, processed_encoder,
            B, T, D,
            BLOCK_P=64, BLOCK_N=64,
            num_warps=4, num_stages=2,
        )

        grid_split_h = (B, triton.cdiv(I, 64), triton.cdiv(D, 64))
        _copy_slice_kernel_img[grid_split_h](
            y, processed_hidden,
            B, I, D,
            BLOCK_P=64, BLOCK_N=64,
            num_warps=4, num_stages=2,
        )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
