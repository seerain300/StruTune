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
    T: tl.constexpr,        # offset for encoder rows (we write starting at row T)
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
    base_out = out_ptr + b * (T + 0) * D + T * D  # write starting at row T

    hst_ptrs = base_hst + p_offsets[:, None] * D + n_offsets[None, :]
    tile = tl.load(hst_ptrs, mask=mask_p[:, None] & mask_n[None, :], other=0.0)
    out_ptrs = base_out + p_offsets[:, None] * D + n_offsets[None, :]
    tl.store(out_ptrs, tile, mask=mask_p[:, None] & mask_n[None, :])


@triton.jit
def _gemm_kernel(
    X_ptr,        # *ptr to out: [B, P, D], where P = T + I
    WT_ptr,       # *ptr to W_T: [D, D] (process_weight transposed)
    Y_ptr,        # *ptr to processed: [B, P, D]
    B: tl.constexpr,
    P: tl.constexpr,
    D: tl.constexpr,
    BLOCK_P: tl.constexpr,  # tile along P
    BLOCK_N: tl.constexpr,  # tile along N (feature)
    BLOCK_K: tl.constexpr,  # reduction chunk along K
):
    # Grid: (B, ceil(P / BLOCK_P), ceil(D / BLOCK_N))
    b = tl.program_id(0)
    pid_p = tl.program_id(1)
    pid_n = tl.program_id(2)

    p_start = pid_p * BLOCK_P
    n_start = pid_n * BLOCK_N

    p_offsets = p_start + tl.arange(0, BLOCK_P)  # along P
    n_offsets = n_start + tl.arange(0, BLOCK_N)  # along D

    mask_p = p_offsets < P
    mask_n = n_offsets < D

    acc = tl.zeros((BLOCK_P, BLOCK_N), dtype=tl.float32)

    # Loop over K (feature dimension) in chunks
    for k0 in range(0, D, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < D

        # Load X tile [BP, BK]: X[b, p, k]
        x_ptrs = X_ptr + b * P * D + p_offsets[:, None] * D + k_offsets[None, :]
        x_tile = tl.load(x_ptrs, mask=mask_p[:, None] & mask_k[None, :], other=0.0)

        # Load WT tile [BK, BN]: WT[k, n]
        wt_ptrs = WT_ptr + k_offsets[:, None] * D + n_offsets[None, :]
        wt_tile = tl.load(wt_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)

        acc += tl.dot(x_tile, wt_tile)

    y_ptrs = Y_ptr + b * P * D + p_offsets[:, None] * D + n_offsets[None, :]
    tl.store(y_ptrs, acc, mask=mask_p[:, None] & mask_n[None, :])


@triton.jit
def _copy_slice_to_encoder(
    src_ptr,      # *ptr to Y: [B, P, D]
    dst_ptr,      # *ptr to processed_encoder: [B, T, D]
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

    src_row_base = src_ptr + b * P * D
    dst_row_base = dst_ptr + b * T * D

    src_ptrs = src_row_base + p_offsets[:, None] * D + n_offsets[None, :]
    dst_ptrs = dst_row_base + p_offsets[:, None] * D + n_offsets[None, :]

    tile = tl.load(src_ptrs, mask=mask_p[:, None] & mask_n[None, :], other=0.0)
    tl.store(dst_ptrs, tile, mask=mask_p[:, None] & mask_n[None, :])


@triton.jit
def _copy_slice_to_hidden(
    src_ptr,      # *ptr to Y: [B, P, D]
    dst_ptr,      # *ptr to processed_hidden: [B, I, D]
    B: tl.constexpr,
    I: tl.constexpr,
    D: tl.constexpr,
    T: tl.constexpr,        # offset in src to start at row T
    BLOCK_P: tl.constexpr,  # tile along I
    BLOCK_N: tl.constexpr,  # tile along D
):
    # Grid: (B, ceil(I / BLOCK_P), ceil(D / BLOCK_N))
    b = tl.program_id(0)
    pid_p = tl.program_id(1)
    pid_n = tl.program_id(2)

    p_start = pid_p * BLOCK_P
    n_start = pid_n * BLOCK_N

    p_offsets = p_start + tl.arange(0, BLOCK_P)
    n_offsets = n_start + tl.arange(0, BLOCK_N)

    mask_p = p_offsets < I
    mask_n = n_offsets < D

    src_row_base = src_ptr + b * P * D + T * D  # start reading at row T
    dst_row_base = dst_ptr + b * I * D

    src_ptrs = src_row_base + p_offsets[:, None] * D + n_offsets[None, :]
    dst_ptrs = dst_row_base + p_offsets[:, None] * D + n_offsets[None, :]

    tile = tl.load(src_ptrs, mask=mask_p[:, None] & mask_n[None, :], other=0.0)
    tl.store(dst_ptrs, tile, mask=mask_p[:, None] & mask_n[None, :])


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized forward:
          1) Concatenate encoder_hidden_states and hidden_states along sequence dimension without torch.cat.
          2) Compute processed = concatenated @ process_weight.T using a Triton GEMM kernel.
          3) Split into processed_encoder and processed_hidden using Triton copy kernels.

        Returns:
          processed_encoder: [B, T, D]
          processed_hidden: [B, I, D]
        """
        assert hidden_states.dim() == 3 and encoder_hidden_states.dim() == 3, "inputs must be [B, *, D]"
        assert process_weight.shape[0] == process_weight.shape[1], "process_weight must be square [D, D]"

        B, T, D = encoder_hidden_states.shape
        I = hidden_states.shape[1]
        P = T + I

        # Ensure contiguous inputs
        enc = encoder_hidden_states.contiguous()
        hst = hidden_states.contiguous()
        wt_t = process_weight.t().contiguous()  # [D, D]

        # 1) Concatenate into out [B, P, D] using Triton
        out = torch.empty((B, P, D), device=enc.device, dtype=enc.dtype)

        # Launch copy_encoder_to_out
        grid_encoder = (B, triton.cdiv(T, 64), triton.cdiv(D, 64))
        _copy_encoder_to_out[grid_encoder](enc, out, B, T, D, BLOCK_P=64, BLOCK_N=64, num_warps=4, num_stages=3)

        # Launch copy_img_to_out
        grid_img = (B, triton.cdiv(I, 64), triton.cdiv(D, 64))
        _copy_img_to_out[grid_img](hst, out, B, I, D, T, BLOCK_P=64, BLOCK_N=64, num_warps=4, num_stages=3)

        # 2) GEMM: out [B, P, D] @ wt_t [D, D] -> processed [B, P, D], Triton
        processed = torch.empty((B, P, D), device=enc.device, dtype=enc.dtype)

        grid_gemm = (B, triton.cdiv(P, 64), triton.cdiv(D, 64))
        _gemm_kernel[grid_gemm](out, wt_t, processed, B, P, D, BLOCK_P=64, BLOCK_N=64, BLOCK_K=64, num_warps=4, num_stages=3)

        # 3) Split via Triton copy kernels
        processed_encoder = torch.empty((B, T, D), device=enc.device, dtype=enc.dtype)
        grid_slice_enc = (B, triton.cdiv(T, 64), triton.cdiv(D, 64))
        _copy_slice_to_encoder[grid_slice_enc](processed, processed_encoder, B, T, D, BLOCK_P=64, BLOCK_N=64, num_warps=4, num_stages=3)

        processed_hidden = torch.empty((B, I, D), device=enc.device, dtype=enc.dtype)
        grid_slice_hid = (B, triton.cdiv(I, 64), triton.cdiv(D, 64))
        _copy_slice_to_hidden[grid_slice_hid](processed, processed_hidden, B, I, D, T, BLOCK_P=64, BLOCK_N=64, num_warps=4, num_stages=3)

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
