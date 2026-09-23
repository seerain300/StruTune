import torch

# Triton import guarded for safety
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


@triton.jit
def _concatenate_seq_kernel(
    enc_ptr,      # *ptr to encoder_hidden_states: [B, T, D]
    hst_ptr,      # *ptr to hidden_states: [B, I, D]
    out_ptr,      # *ptr to concatenated output: [B, P, D]
    B: tl.constexpr,
    T: tl.constexpr,
    I: tl.constexpr,
    D: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    b = tl.program_id(0)
    pid_p = tl.program_id(1)
    p_start = pid_p * BLOCK_P
    p_offsets = p_start + tl.arange(0, BLOCK_P)
    P = T + I

    mask_p = p_offsets < P
    mask_enc = (b < B) & (p_offsets < T) & mask_p
    mask_hst = (b < B) & (p_offsets >= T) & (p_offsets < P) & mask_p

    enc_row_ptr = enc_ptr + b * T * D + p_offsets * D
    hst_row_ptr = hst_ptr + b * I * D + (p_offsets - T) * D
    out_row_ptr = out_ptr + b * P * D + p_offsets * D

    # Load encoder for p < T
    enc_vals = tl.load(enc_row_ptr, mask=mask_enc, other=0.0)
    tl.store(out_row_ptr, enc_vals, mask=mask_p & mask_enc)

    # Load hidden for p >= T
    hst_vals = tl.load(hst_row_ptr, mask=mask_hst, other=0.0)
    tl.store(out_row_ptr, hst_vals, mask=mask_p & mask_hst)


@triton.jit
def _gemm_kernel(
    X_ptr,        # *ptr to X: [B, P, D]
    WT_ptr,       # *ptr to W_T: [D, D]
    Y_ptr,        # *ptr to Y: [B, P, D]
    B: tl.constexpr,
    P: tl.constexpr,
    D: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    b = tl.program_id(0)
    pid_p = tl.program_id(1)
    pid_n = tl.program_id(2)

    p_start = pid_p * BLOCK_P
    n_start = pid_n * BLOCK_N

    p_offsets = p_start + tl.arange(0, BLOCK_P)
    n_offsets = n_start + tl.arange(0, BLOCK_N)

    mask_p = p_offsets < P
    mask_n = n_offsets < D

    acc = tl.zeros((BLOCK_P, BLOCK_N), dtype=tl.float32)

    # Reduction over K in chunks
    for k0 in range(0, D, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < D

        # Load X tile [BP, BK]
        x_ptrs = X_ptr + b * P * D + p_offsets[:, None] * D + k_offsets[None, :]
        x_tile = tl.load(x_ptrs, mask=mask_p[:, None] & mask_k[None, :], other=0.0)

        # Load W_T tile [BK, BN]
        wt_ptrs = WT_ptr + k_offsets[:, None] * D + n_offsets[None, :]
        wt_tile = tl.load(wt_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)

        # Accumulate
        acc += tl.dot(x_tile, wt_tile)

    # Store Y tile
    y_ptrs = Y_ptr + b * P * D + p_offsets[:, None] * D + n_offsets[None, :]
    tl.store(y_ptrs, acc, mask=mask_p[:, None] & mask_n[None, :])


@triton.jit
def _copy_slice_kernel(
    src_ptr,      # *ptr to Y: [B, P, D]
    dst_ptr,      # *ptr to processed_encoder: [B, T, D]
    B: tl.constexpr,
    T: tl.constexpr,
    D: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    b = tl.program_id(0)
    pid_p = tl.program_id(1)
    p_start = pid_p * BLOCK_P
    p_offsets = p_start + tl.arange(0, BLOCK_P)
    mask_p = p_offsets < T

    src_row_ptr = src_ptr + b * P * D + p_offsets * D
    dst_row_ptr = dst_ptr + b * T * D + p_offsets * D

    vals = tl.load(src_row_ptr, mask=mask_p, other=0.0)
    tl.store(dst_row_ptr, vals, mask=mask_p)


@triton.jit
def _copy_slice_kernel_img(
    src_ptr,      # *ptr to Y: [B, P, D]
    dst_ptr,      # *ptr to processed_hidden: [B, I, D]
    B: tl.constexpr,
    T: tl.constexpr,
    I: tl.constexpr,
    D: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    b = tl.program_id(0)
    pid_p = tl.program_id(1)
    p_start = pid_p * BLOCK_P
    p_offsets = p_start + tl.arange(0, BLOCK_P)
    mask_p = (p_offsets >= T) & (p_offsets < (T + I)) & (p_offsets < (T + I))

    src_row_ptr = src_ptr + b * P * D + p_offsets * D
    dst_row_ptr = dst_ptr + b * I * D + (p_offsets - T) * D

    vals = tl.load(src_row_ptr, mask=mask_p, other=0.0)
    tl.store(dst_row_ptr, vals, mask=mask_p)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        Triton-optimized forward:
        - Concatenates along sequence via Triton (_concatenate_seq_kernel).
        - Performs batched matmul via Triton (_gemm_kernel): Y = concatenated @ process_weight.T.
        - Splits into encoder and hidden outputs via Triton copy kernels.
        Returns:
          processed_encoder: [B, T, D]
          processed_hidden: [B, I, D]
        """
        assert TRITON_AVAILABLE, "Triton is not available."

        # Shapes
        B, T, D = encoder_hidden_states.shape
        I = hidden_states.shape[1]
        P = T + I

        # Ensure contiguous
        enc = encoder_hidden_states.contiguous()
        hst = hidden_states.contiguous()
        W = process_weight.contiguous()  # [D, D]

        # 1) Concatenate along sequence via Triton: out [B, P, D]
        out = torch.empty((B, P, D), device=enc.device, dtype=enc.dtype)
        BLOCK_P = 128
        grid_concat = (B, triton.cdiv(P, BLOCK_P))
        _concatenate_seq_kernel[grid_concat](
            enc, hst, out,
            B=B, T=T, I=I, D=D,
            BLOCK_P=BLOCK_P,
        )

        # 2) GEMM via Triton: Y = out @ W_T, where W_T = W.T
        W_T = W.t().contiguous()  # [D, D]
        Y = torch.empty((B, P, D), device=enc.device, dtype=enc.dtype)

        BLOCK_P_GEMM = 128
        BLOCK_N = 64
        BLOCK_K = 32
        grid_gemm = (B, triton.cdiv(P, BLOCK_P_GEMM), triton.cdiv(D, BLOCK_N))
        _gemm_kernel[grid_gemm](
            out, W_T, Y,
            B=B, P=P, D=D,
            BLOCK_P=BLOCK_P_GEMM, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        )

        # 3) Split via Triton copy kernels
        processed_encoder = torch.empty((B, T, D), device=enc.device, dtype=enc.dtype)
        processed_hidden = torch.empty((B, I, D), device=enc.device, dtype=enc.dtype)

        grid_copy = (B, triton.cdiv(T, BLOCK_P))
        _copy_slice_kernel[grid_copy](
            Y, processed_encoder,
            B=B, T=T, D=D,
            BLOCK_P=BLOCK_P,
        )

        grid_copy_img = (B, triton.cdiv(P, BLOCK_P))
        _copy_slice_kernel_img[grid_copy_img](
            Y, processed_hidden,
            B=B, T=T, I=I, D=D,
            BLOCK_P=BLOCK_P,
        )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
