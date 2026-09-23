import torch
import triton
import triton.language as tl


@triton.jit
def concat_rows_to_A_kernel(
    enc_ptr,       # *const T, shape [B, T, H]
    img_ptr,       # *const T, shape [B, I, H]
    A_ptr,         # *T, shape [M, H], M = B*(T+I)
    B, T, I, H,    # int32 dimensions
    stride_b_e, stride_t_e, stride_h_e,  # enc strides
    stride_b_i, stride_i_i, stride_h_i,  # img strides
    stride_m_a, stride_h_a,               # A strides
    BLOCK_M: tl.constexpr, BLOCK_H: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    M_total = B * (T + I)
    mask_m = m < M_total

    # Compute batch and seq index
    b = m // (T + I)
    seq = m % (T + I)

    # Determine source tensor: encoder if seq < T, else image at offset seq - T
    use_enc = seq < T

    # Compute source offsets and load
    enc_offsets = b * stride_b_e + seq * stride_t_e + tl.arange(0, BLOCK_H) * stride_h_e
    vals = tl.load(enc_ptr + enc_offsets, mask=mask_m, other=0.0)

    # Store to A[m, :]
    A_offsets = m[:, None] * stride_m_a + tl.arange(0, BLOCK_H)[None, :] * stride_h_a
    tl.store(A_ptr + A_offsets, vals[:, None], mask=mask_m[:, None])


@triton.jit
def batched_matmul_kernel(
    A_ptr,  # *const float32, shape [M, H], M = B*(T+I)
    B_ptr,  # *const float32, shape [H, H] (process_weight.T)
    C_ptr,  # *float32, shape [M, H]
    M, H,   # int32
    stride_am, stride_ah,        # A strides
    stride_bh, stride_bb,        # B strides (B is [H, H])
    stride_cm, stride_ch,        # C strides
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Each program computes a tile C[offs_m, offs_n]
    offs_m = tl.program_id(axis=0) * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.program_id(axis=1) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < H

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension (H)
    for k in range(0, H, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        mask_k = offs_k < H

        # Load A tile: [BLOCK_M, BLOCK_K]
        A_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ah
        A_tile = tl.load(A_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)

        # Load B tile as [BLOCK_K, BLOCK_N]; B is [H, H]
        B_ptrs = B_ptr + offs_k[:, None] * stride_bh + offs_n[None, :] * stride_bb
        B_tile = tl.load(B_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)

        # Accumulate
        acc += tl.dot(A_tile, B_tile)

    # Store result tile
    C_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_ch
    tl.store(C_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


@triton.jit
def copy_rows_encoder_kernel(
    C_ptr,               # *const float32, shape [M, H]
    out_ptr,             # *float32, shape [B, T, H]
    B, T, I, H,          # dims (I, H not used but kept for signature)
    stride_m_c, stride_h_c,        # C strides
    stride_b_o, stride_t_o, stride_h_o,  # output strides
    b,                   # batch index
    BLOCK_M: tl.constexpr, BLOCK_H: tl.constexpr,
):
    # Copy rows [0 : B*T) of C into out[b, :, :]
    num_rows = B * T
    for m_start in range(0, num_rows, BLOCK_M):
        m = m_start + tl.arange(0, BLOCK_M)
        mask_m = m < (B * T)
        C_ptrs = C_ptr + m[:, None] * stride_m_c + tl.arange(0, BLOCK_H)[None, :] * stride_h_c
        t_idx = m % T
        out_ptrs = out_ptr + b * stride_b_o + t_idx[:, None] * stride_t_o + tl.arange(0, BLOCK_H)[None, :] * stride_h_o
        for h_start in range(0, H, BLOCK_H):
            h = h_start + tl.arange(0, BLOCK_H)
            mask_h = h < H
            vals = tl.load(C_ptrs, mask=mask_m[:, None] & mask_h[None, :], other=0.0)
            tl.store(out_ptrs, vals, mask=mask_m[:, None] & mask_h[None, :])


@triton.jit
def copy_rows_hidden_kernel(
    C_ptr,               # *const float32, shape [M, H]
    out_ptr,             # *float32, shape [B, I, H]
    B, T, I, H,          # dims
    stride_m_c, stride_h_c,        # C strides
    stride_b_o, stride_i_o, stride_h_o,  # output strides
    b,                   # batch index
    BLOCK_M: tl.constexpr, BLOCK_H: tl.constexpr,
):
    # Copy rows [B*T : B*(T+I)) of C into out[b, :, :]
    start = B * T
    end = B * (T + I)
    num_rows = end - start
    for m_start in range(0, num_rows, BLOCK_M):
        m = start + m_start + tl.arange(0, BLOCK_M)
        mask_m = m < end
        C_ptrs = C_ptr + m[:, None] * stride_m_c + tl.arange(0, BLOCK_H)[None, :] * stride_h_c
        dest_m = m - start
        i_idx = dest_m % I
        out_ptrs = out_ptr + b * stride_b_o + i_idx[:, None] * stride_i_o + tl.arange(0, BLOCK_H)[None, :] * stride_h_o
        for h_start in range(0, H, BLOCK_H):
            h = h_start + tl.arange(0, BLOCK_H)
            mask_h = h < H
            vals = tl.load(C_ptrs, mask=mask_m[:, None] & mask_h[None, :], other=0.0)
            tl.store(out_ptrs, vals, mask=mask_m[:, None] & mask_h[None, :])


def _run_concatenate_rows_to_A(enc: torch.Tensor, img: torch.Tensor, A: torch.Tensor):
    B, T, H = enc.shape
    I = img.shape[1]
    M = B * (T + I)
    grid = (triton.cdiv(M, 128),)
    concat_rows_to_A_kernel[grid](
        enc, img, A,
        B, T, I, H,
        enc.stride(0), enc.stride(1), enc.stride(2),
        img.stride(0), img.stride(1), img.stride(2),
        A.stride(0), A.stride(1),
        num_warps=4, num_stages=2,
    )


def _run_matmul(A: torch.Tensor, B_t: torch.Tensor, C: torch.Tensor):
    M, H = A.shape
    # Ensure dtypes are float32 for robust Triton kernel
    A_ = A.to(torch.float32)
    B_ = B_t.to(torch.float32)
    # Launch a 2D grid over M and H tiles
    BLOCK_M = 128
    BLOCK_N = 64
    BLOCK_K = 64
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(H, BLOCK_N))
    batched_matmul_kernel[grid](
        A_, B_, C,
        M, H,
        A_.stride(0), A_.stride(1),
        B_.stride(0), B_.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M, BLOCK_N, BLOCK_K,
        num_warps=4, num_stages=2,
    )


def _run_copy_encoder(C: torch.Tensor, processed_encoder: torch.Tensor, B: int, T: int):
    M_total = B * (T + 0)  # first B*T rows
    grid = (triton.cdiv(B * T, 128),)
    copy_rows_encoder_kernel[grid](
        C, processed_encoder,
        B, T, 0, 0,  # I and H not used here; kernel only needs B, T for mask
        C.stride(0), C.stride(1),
        processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
        0,  # start batch
        num_warps=4, num_stages=2,
    )


def _run_copy_hidden(C: torch.Tensor, processed_hidden: torch.Tensor, B: int, T: int, I: int):
    M_total = B * (T + I)
    grid = (triton.cdiv(B * I, 128),)
    copy_rows_hidden_kernel[grid](
        C, processed_hidden,
        B, T, I, 0,  # H not used here; kernel uses C's column stride
        C.stride(0), C.stride(1),
        processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
        0,  # start batch
        num_warps=4, num_stages=2,
    )


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-only implementation:
        1) Concatenate along sequence dimension using Triton
        2) Matmul using Triton
        3) Split into encoder and hidden outputs using Triton per-batch copy kernels
        """
        # Ensure inputs are contiguous
        enc = encoder_hidden_states.contiguous()
        img = hidden_states.contiguous()
        weight = process_weight.contiguous()  # [H, H]

        B, T, H = enc.shape
        I = img.shape[1]
        M = B * (T + I)

        # 1) Concatenate rows into A [M, H]
        A = torch.empty((M, H), dtype=enc.dtype, device=enc.device)
        _run_concatenate_rows_to_A(enc, img, A)

        # 2) Compute C = A @ process_weight.T in Triton
        A_f32 = A.to(torch.float32)
        weight_T = weight.t().contiguous()  # [H, H]
        C = torch.empty((M, H), dtype=torch.float32, device=enc.device)
        _run_matmul(A_f32, weight_T, C)

        # 3) Split C into processed_encoder and processed_hidden
        processed_encoder = torch.empty((B, T, H), dtype=torch.float32, device=enc.device)
        processed_hidden = torch.empty((B, I, H), dtype=torch.float32, device=enc.device)
        _run_copy_encoder(C, processed_encoder, B, T)
        _run_copy_hidden(C, processed_hidden, B, T, I)

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
