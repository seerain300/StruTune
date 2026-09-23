import torch
import triton
import triton.language as tl


@triton.jit
def _copy_encoder_to_out(
    enc_ptr,      # *ptr to encoder_hidden_states: [B, T, D]
    out_ptr,      # *ptr to concatenated out: [B, P, D], write first T
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
    out_ptr,      # *ptr to concatenated out: [B, P, D]
    B: tl.constexpr,
    I: tl.constexpr,
    D: tl.constexpr,
    T: tl.constexpr,
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

    for pi in range(BLOCK_P):
        p = p_start + pi
        if p < I:
            for ni in range(BLOCK_N):
                n = n_start + ni
                if n < D:
                    val = tl.load(hst_ptr + b * I * D + p * D + n)
                    tl.store(out_ptr + b * (T + I) * D + (T + p) * D + n, val)


@triton.jit
def _gemm_out_kernel(
    Out_ptr,      # *ptr to concatenated out: [B, P, D], float32
    WT_ptr,       # *ptr to process_weight.T: [D, D], float32
    Y_ptr,        # *ptr to processed: [B, P, D], float32
    B: tl.constexpr,
    P: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,  # tile over M=P
    BLOCK_N: tl.constexpr,  # tile over N=D
    BLOCK_K: tl.constexpr,  # reduction tile
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

        # A tile: Out[b, m, k] -> [BM, BK]
        a_ptrs = Out_ptr + b * P * D + m_offsets[:, None] * D + k_offsets[None, :]
        a_vals = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)  # [BM, BK], fp32

        # B tile: WT[k, n] -> [BK, BN]
        wt_ptrs = WT_ptr + k_offsets[:, None] * D + n_offsets[None, :]
        wt_vals = tl.load(wt_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)  # [BK, BN], fp32

        # Accumulate
        acc += tl.dot(a_vals, wt_vals)

    # Store result
    y_ptrs = Y_ptr + b * P * D + m_offsets[:, None] * D + n_offsets[None, :]
    tl.store(y_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


@triton.jit
def _copy_slice_encoder(
    processed_ptr,  # *ptr to processed: [B, P, D], float32
    out_ptr,        # *ptr to processed_encoder: [B, T, D], float32
    B: tl.constexpr,
    T: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,  # tile over T
    BLOCK_N: tl.constexpr,  # tile over D
):
    b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N

    m_offsets = m_start + tl.arange(0, BLOCK_M)  # along T
    n_offsets = n_start + tl.arange(0, BLOCK_N)  # along D

    mask_m = m_offsets < T
    mask_n = n_offsets < D

    src_ptrs = processed_ptr + b * P * D + m_offsets[:, None] * D + n_offsets[None, :]
    vals = tl.load(src_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0)
    dst_ptrs = out_ptr + b * T * D + m_offsets[:, None] * D + n_offsets[None, :]
    tl.store(dst_ptrs, vals, mask=mask_m[:, None] & mask_n[None, :])


@triton.jit
def _copy_slice_img(
    processed_ptr,  # *ptr to processed: [B, P, D], float32
    out_ptr,        # *ptr to processed_hidden: [B, I, D], float32
    B: tl.constexpr,
    I: tl.constexpr,
    D: tl.constexpr,
    T: tl.constexpr,
    BLOCK_M: tl.constexpr,  # tile over I
    BLOCK_N: tl.constexpr,  # tile over D
):
    b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N

    m_offsets = m_start + tl.arange(0, BLOCK_M)  # along I
    n_offsets = n_start + tl.arange(0, BLOCK_N)  # along D

    mask_m = m_offsets < I
    mask_n = n_offsets < D

    src_ptrs = processed_ptr + b * P * D + (T + m_offsets[:, None]) * D + n_offsets[None, :]
    vals = tl.load(src_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0)
    dst_ptrs = out_ptr + b * I * D + m_offsets[:, None] * D + n_offsets[None, :]
    tl.store(dst_ptrs, vals, mask=mask_m[:, None] & mask_n[None, :])


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Ensure contiguous tensors for predictable strides
        encoder_hidden_states = encoder_hidden_states.contiguous()
        hidden_states = hidden_states.contiguous()
        process_weight = process_weight.contiguous()

        B, T, D = encoder_hidden_states.shape
        B2, I, D2 = hidden_states.shape
        assert B == B2 and D == D2, "Batch and feature dims must match"
        P = T + I

        # 1) Concatenate in Triton: out[:, :T, :] = encoder; out[:, T:, :] = hidden
        out = torch.empty((B, P, D), device=encoder_hidden_states.device, dtype=torch.float32)

        # Launch copy encoder kernel
        BLOCK_P = 64
        BLOCK_N = 64
        grid_concat_enc = (B, triton.cdiv(T, BLOCK_P), triton.cdiv(D, BLOCK_N))
        _copy_encoder_to_out[grid_concat_enc](
            encoder_hidden_states, out,
            B=B, T=T, D=D,
            BLOCK_P=BLOCK_P, BLOCK_N=BLOCK_N,
            num_warps=4, num_stages=2
        )
        # Launch copy hidden kernel
        grid_concat_img = (B, triton.cdiv(I, BLOCK_P), triton.cdiv(D, BLOCK_N))
        _copy_img_to_out[grid_concat_img](
            hidden_states, out,
            B=B, I=I, D=D, T=T,
            BLOCK_P=BLOCK_P, BLOCK_N=BLOCK_N,
            num_warps=4, num_stages=2
        )

        # 2) GEMM in Triton: processed = out @ process_weight.T (compute in fp32)
        WT = process_weight.t().contiguous()  # [D, D], float32
        processed = torch.empty((B, P, D), device=out.device, dtype=torch.float32)

        BLOCK_M = 128
        BLOCK_N = 64
        BLOCK_K = 64
        grid_gemm = (B, triton.cdiv(P, BLOCK_M), triton.cdiv(D, BLOCK_N))
        _gemm_out_kernel[grid_gemm](
            out, WT, processed,
            B=B, P=P, D=D,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2
        )

        # 3) Split in Triton
        processed_encoder = torch.empty((B, T, D), device=processed.device, dtype=torch.float32)
        grid_enc = (B, triton.cdiv(T, BLOCK_M), triton.cdiv(D, BLOCK_N))
        _copy_slice_encoder[grid_enc](
            processed, processed_encoder,
            B=B, T=T, D=D,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            num_warps=4, num_stages=2
        )

        processed_hidden = torch.empty((B, I, D), device=processed.device, dtype=torch.float32)
        grid_img = (B, triton.cdiv(I, BLOCK_M), triton.cdiv(D, BLOCK_N))
        _copy_slice_img[grid_img](
            processed, processed_hidden,
            B=B, I=I, D=D, T=T,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            num_warps=4, num_stages=2
        )

        # Cast outputs back to original dtypes for consistency
        processed_encoder = processed_encoder.to(encoder_hidden_states.dtype)
        processed_hidden = processed_hidden.to(hidden_states.dtype)

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
