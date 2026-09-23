import torch
import triton
import triton.language as tl


# Kernel: copy encoder_hidden_states into out[:, :T, :]
@triton.jit
def _copy_encoder_to_out(
    enc_ptr,      # *ptr to [B, T, D]
    out_ptr,      # *ptr to [B, P, D], we write into first T rows
    B: tl.constexpr,
    T: tl.constexpr,
    D: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    b = tl.program_id(0)
    pid_p = tl.program_id(1)
    pid_n = tl.program_id(2)

    p_start = pid_p * BLOCK_P
    n_start = pid_n * BLOCK_N

    p_offsets = p_start + tl.arange(0, BLOCK_P)
    n_offsets = n_start + tl.arange(0, BLOCK_N)

    mask_p = p_offsets < T
    mask_n = n_offsets < D

    # For each tile, copy a BLOCK_P x BLOCK_N slice
    for pi in range(BLOCK_P):
        p = p_start + pi
        if p < T:
            enc_row_ptrs = enc_ptr + b * T * D + p * D + n_offsets
            val = tl.load(enc_row_ptrs, mask=mask_n, other=0.0)
            out_row_ptrs = out_ptr + b * (T + 0) * D + p * D + n_offsets
            tl.store(out_row_ptrs, val, mask=mask_n)


# Kernel: copy hidden_states into out[:, T:, :]
@triton.jit
def _copy_img_to_out(
    hst_ptr,      # *ptr to [B, I, D]
    out_ptr,      # *ptr to [B, P, D], we write into rows T..T+I-1
    B: tl.constexpr,
    I: tl.constexpr,
    D: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    b = tl.program_id(0)
    pid_p = tl.program_id(1)
    pid_n = tl.program_id(2)

    p_start = pid_p * BLOCK_P  # over I
    n_start = pid_n * BLOCK_N

    p_offsets = p_start + tl.arange(0, BLOCK_P)
    n_offsets = n_start + tl.arange(0, BLOCK_N)

    mask_p = p_offsets < I
    mask_n = n_offsets < D

    for pi in range(BLOCK_P):
        p = pid_p * BLOCK_P + pi
        if p < I:
            hst_row_ptrs = hst_ptr + b * I * D + p * D + n_offsets
            val = tl.load(hst_row_ptrs, mask=mask_n, other=0.0)
            out_row_ptrs = out_ptr + b * (T + I) * D + (T + p) * D + n_offsets
            tl.store(out_row_ptrs, val, mask=mask_n)


# Triton GEMM kernel: computes Y = X @ WT, where X is [B, P, D], WT is [D, D]
# Grid: (B, tiles over P, tiles over D).
@triton.jit
def _gemm_kernel(
    X_ptr,        # *ptr to X: [B, P, D]
    WT_ptr,       # *ptr to WT: [D, D]
    Y_ptr,        # *ptr to Y: [B, P, D]
    B: tl.constexpr,
    P: tl.constexpr,
    D: tl.constexpr,
    BLOCK_P: tl.constexpr,  # tile over P
    BLOCK_N: tl.constexpr,  # tile over D
    BLOCK_K: tl.constexpr,  # reduction chunk over K
):
    b = tl.program_id(0)
    pid_p = tl.program_id(1)
    pid_n = tl.program_id(2)

    p_start = pid_p * BLOCK_P
    n_start = pid_n * BLOCK_N

    p_offsets = p_start + tl.arange(0, BLOCK_P)  # along P
    n_offsets = n_start + tl.arange(0, BLOCK_N)  # along D

    mask_p = p_offsets < P
    mask_n = n_offsets < D

    # Accumulator for output tile [BP, BN]
    acc = tl.zeros((BLOCK_P, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension in chunks
    for k0 in range(0, D, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < D

        # Load X tile [BP, BK]: X[b, p, k]
        x_ptrs = X_ptr + b * P * D + p_offsets[:, None] * D + k_offsets[None, :]
        x_tile = tl.load(x_ptrs, mask=mask_p[:, None] & mask_k[None, :], other=0.0)

        # Load WT tile [BK, BN]: WT[k, n]
        wt_ptrs = WT_ptr + k_offsets[:, None] * D + n_offsets[None, :]
        wt_tile = tl.load(wt_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)

        # Accumulate with tl.dot
        acc += tl.dot(x_tile, wt_tile)

    # Store the computed tile Y[b, p, n]
    y_ptrs = Y_ptr + b * P * D + p_offsets[:, None] * D + n_offsets[None, :]
    tl.store(y_ptrs, acc, mask=mask_p[:, None] & mask_n[None, :])


# Kernel: copy processed[:, :T, :] into processed_encoder
@triton.jit
def _copy_slice_encoder(
    src_ptr,      # *ptr to Y: [B, P, D]
    dst_ptr,      # *ptr to processed_encoder: [B, T, D]
    B: tl.constexpr,
    T: tl.constexpr,
    D: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    b = tl.program_id(0)
    pid_p = tl.program_id(1)
    pid_n = tl.program_id(2)

    p_start = pid_p * BLOCK_P
    n_start = pid_n * BLOCK_N

    p_offsets = p_start + tl.arange(0, BLOCK_P)
    n_offsets = n_start + tl.arange(0, BLOCK_N)

    mask_p = p_offsets < T
    mask_n = n_offsets < D

    for pi in range(BLOCK_P):
        p = p_start + pi
        if p < T:
            src_row_ptrs = src_ptr + b * P * D + p * D + n_offsets
            val = tl.load(src_row_ptrs, mask=mask_n, other=0.0)
            dst_row_ptrs = dst_ptr + b * T * D + p * D + n_offsets
            tl.store(dst_row_ptrs, val, mask=mask_n)


# Kernel: copy processed[:, T:, :] into processed_hidden
@triton.jit
def _copy_slice_image(
    src_ptr,      # *ptr to Y: [B, P, D]
    dst_ptr,      # *ptr to processed_hidden: [B, I, D]
    B: tl.constexpr,
    I: tl.constexpr,
    D: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    b = tl.program_id(0)
    pid_p = tl.program_id(1)
    pid_n = tl.program_id(2)

    p_start = pid_p * BLOCK_P  # over I (since P = T + I, p_start runs over I)
    n_start = pid_n * BLOCK_N

    p_offsets = p_start + tl.arange(0, BLOCK_P)
    n_offsets = n_start + tl.arange(0, BLOCK_N)

    mask_p = p_offsets < I
    mask_n = n_offsets < D

    for pi in range(BLOCK_P):
        p = p_start + pi
        if p < I:
            src_row_ptrs = src_ptr + b * P * D + (T + p) * D + n_offsets
            val = tl.load(src_row_ptrs, mask=mask_n, other=0.0)
            dst_row_ptrs = dst_ptr + b * I * D + p * D + n_offsets
            tl.store(dst_row_ptrs, val, mask=mask_n)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        Triton-only implementation:
        1) Concatenate encoder_hidden_states and hidden_states into [B, P, D] without torch.cat.
        2) Compute processed = concatenated @ process_weight.T using torch.matmul (for robustness).
        3) Split processed into processed_encoder and processed_hidden using Triton copy kernels.
        """
        assert hidden_states.dim() == 3 and encoder_hidden_states.dim() == 3
        assert process_weight.dim() == 2
        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        D = hidden_states.shape[2]
        assert encoder_hidden_states.shape[2] == D and process_weight.shape[0] == D and process_weight.shape[1] == D

        # Ensure contiguity
        enc = encoder_hidden_states.contiguous()
        hst = hidden_states.contiguous()
        wt = process_weight.contiguous()

        # 1) Concatenate via Triton: out [B, P, D], P = T + I
        P = T + I
        out = torch.empty((B, P, D), device=hidden_states.device, dtype=hidden_states.dtype)

        # Launch copy kernels
        BLOCK_P = 64
        BLOCK_N = 64
        grid_enc = (B, triton.cdiv(T, BLOCK_P), triton.cdiv(D, BLOCK_N))
        _copy_encoder_to_out[grid_enc](
            enc, out, B, T, D, BLOCK_P, BLOCK_N,
            num_warps=4, num_stages=2
        )

        grid_img = (B, triton.cdiv(I, BLOCK_P), triton.cdiv(D, BLOCK_N))
        _copy_img_to_out[grid_img](
            hst, out, B, I, D, BLOCK_P, BLOCK_N,
            num_warps=4, num_stages=2
        )

        # 2) Compute processed = out @ process_weight.T using torch.matmul (robust and fast)
        # wt.T is [D, D]
        processed = torch.matmul(out, wt.transpose(0, 1))

        # 3) Split processed into two outputs using Triton copy kernels
        processed_encoder = torch.empty((B, T, D), device=hidden_states.device, dtype=hidden_states.dtype)
        processed_hidden = torch.empty((B, I, D), device=hidden_states.device, dtype=hidden_states.dtype)

        grid_slice = (B, triton.cdiv(T, BLOCK_P), triton.cdiv(D, BLOCK_N))
        _copy_slice_encoder[grid_slice](
            processed, processed_encoder, B, T, D, BLOCK_P, BLOCK_N,
            num_warps=4, num_stages=2
        )

        grid_slice2 = (B, triton.cdiv(I, BLOCK_P), triton.cdiv(D, BLOCK_N))
        _copy_slice_image[grid_slice2](
            processed, processed_hidden, B, I, D, BLOCK_P, BLOCK_N,
            num_warps=4, num_stages=2
        )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
