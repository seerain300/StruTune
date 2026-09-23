import torch
import triton
import triton.language as tl


@triton.jit
def matmul_batched_kernel(
    A_ptr,     # pointer to A: [B, S, H], S = T + I
    B_ptr,     # pointer to B: [H, H] (process_weight.T)
    C_ptr,     # pointer to C: [M, H], M = B*S
    B,         # batch size
    S,         # sequence length (T + I)
    H,         # hidden_dim
    stride_b_a, stride_s_a, stride_h_a,  # strides for A
    stride_h_b, stride_k_b,              # strides for B (H x H): rows, cols
    stride_m_c, stride_h_c,              # strides for C
    BLOCK_M: tl.constexpr,               # tile size for rows
    BLOCK_N: tl.constexpr,               # tile size for cols
    BLOCK_K: tl.constexpr,               # reduction tile
):
    # grid over (rows m, cols n)
    pid_m = tl.program_id(0)  # over rows m in [0, M)
    pid_n = tl.program_id(1)  # over cols n in [0, H)

    M = B * S

    # offsets for rows and columns
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < H

    # initialize accumulator (compute in fp32)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # loop over K dimension (hidden_dim)
    for k0 in range(0, H, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < H

        # For each output row offs_m, determine batch b and seq index s
        # m in [0, M): b = m // S, s = m % S
        b = offs_m // S
        s = offs_m % S

        # Load A tile: A[b, s, k] -> A_ptr + b*stride_b_a + s*stride_s_a + k*stride_h_a
        a_row_ptrs = A_ptr + b[:, None] * stride_b_a + s[:, None] * stride_s_a + offs_k[None, :] * stride_h_a
        a_tile = tl.load(a_row_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0).to(tl.float32)

        # Load B tile: B[k, n] -> B_ptr + k*stride_h_b + n*stride_k_b
        b_tile_ptrs = B_ptr + offs_k[:, None] * stride_h_b + offs_n[None, :] * stride_k_b
        b_tile = tl.load(b_tile_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0).to(tl.float32)

        # Accumulate
        acc += tl.dot(a_tile, b_tile)

    # Write back to C: C[m, n] -> C_ptr + m*stride_m_c + n*stride_h_c
    c_ptrs = C_ptr + offs_m[:, None] * stride_m_c + offs_n[None, :] * stride_h_c
    tl.store(c_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


@triton.jit
def copy_rows_encoder_kernel(
    C_ptr,     # [M, H], M = B*(T+I)
    E_ptr,     # [B, T, H]
    B, T, I, H,
    stride_m_c, stride_h_c,        # strides for C
    e_stride_b, e_stride_t, e_stride_h,  # strides for E
    BLOCK_M: tl.constexpr,            # rows per block
    BLOCK_N: tl.constexpr,            # cols per block
):
    pid_m = tl.program_id(0)  # over rows in [0, B*T)
    pid_n = tl.program_id(1)  # over cols in [0, H)

    M = B * (T + I)
    start_row = 0
    end_row = B * T

    offs_m = start_row + pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < end_row
    mask_n = offs_n < H

    c_ptrs = C_ptr + offs_m[:, None] * stride_m_c + offs_n[None, :] * stride_h_c
    # Map m in [0, B*T) to batch b = m // T and seq t = m % T
    b = offs_m // T
    t = offs_m % T
    e_ptrs = E_ptr + b[:, None] * e_stride_b + t[:, None] * e_stride_t + offs_n[None, :] * e_stride_h
    vals = tl.load(c_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0)
    tl.store(e_ptrs, vals, mask=mask_m[:, None] & mask_n[None, :])


@triton.jit
def copy_rows_hidden_kernel(
    C_ptr,     # [M, H]
    H_ptr,     # [B, I, H]
    B, T, I, H,
    stride_m_c, stride_h_c,        # strides for C
    h_stride_b, h_stride_i, h_stride_h,  # strides for H
    BLOCK_M: tl.constexpr,            # rows per block
    BLOCK_N: tl.constexpr,            # cols per block
):
    pid_m = tl.program_id(0)  # over rows in [B*T, M)
    pid_n = tl.program_id(1)  # over cols in [0, H)

    M = B * (T + I)
    start_row = B * T
    end_row = M

    offs_m = start_row + pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < end_row
    mask_n = offs_n < H

    c_ptrs = C_ptr + offs_m[:, None] * stride_m_c + offs_n[None, :] * stride_h_c

    # Map m in [B*T, M): b = (m - B*T) // I, seq = (m - B*T) % I
    rel = offs_m - start_row
    b = rel // I
    seq = rel % I

    h_ptrs = H_ptr + b[:, None] * h_stride_b + seq[:, None] * h_stride_i + offs_n[None, :] * h_stride_h
    vals = tl.load(c_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0)
    tl.store(h_ptrs, vals, mask=mask_m[:, None] & mask_n[None, :])


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-only implementation:
        - Concatenate along sequence dimension using torch.cat (data movement only).
        - Compute matmul C = A @ process_weight.T using a Triton kernel.
        - Split C into processed_encoder and processed_hidden using Triton copy kernels.
        """
        # Shapes
        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        H = hidden_states.shape[2]
        assert encoder_hidden_states.shape == (B, T, H), "encoder_hidden_states shape must be [batch, text_seq_len, hidden_dim]"
        assert hidden_states.shape == (B, I, H), "hidden_states shape must be [batch, img_seq_len, hidden_dim]"
        assert process_weight.shape == (H, H), "process_weight shape must be [hidden_dim, hidden_dim]"

        # Ensure contiguity for predictable strides
        enc = encoder_hidden_states.contiguous()  # [B, T, H]
        img = hidden_states.contiguous()         # [B, I, H]
        weight = process_weight.contiguous()     # [H, H]

        # Concatenate along sequence dimension to form A [B, T+I, H]
        A = torch.cat([enc, img], dim=1)  # [B, S, H], S = T + I

        # Allocate C [M, H], M = B*S
        S = T + I
        M = B * S
        C = torch.empty((M, H), dtype=torch.float32, device=hidden_states.device)  # compute in fp32

        # Launch Triton matmul kernel
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 32

        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(H, BLOCK_N))
        matmul_batched_kernel[grid](
            A, weight.t(), C,
            B, S, H,
            A.stride(0), A.stride(1), A.stride(2),
            weight.t().stride(0), weight.t().stride(1),
            C.stride(0), C.stride(1),
            num_warps=4, num_stages=2,
        )

        # Allocate outputs (fp32 as per compute)
        processed_encoder = torch.empty((B, T, H), dtype=torch.float32, device=hidden_states.device)
        processed_hidden = torch.empty((B, I, H), dtype=torch.float32, device=hidden_states.device)

        # Launch per-batch copy kernels:
        # 1) Copy first B*T rows of C into processed_encoder
        grid_encoder = (triton.cdiv(B * T, BLOCK_M), triton.cdiv(H, BLOCK_N))
        copy_rows_encoder_kernel[grid_encoder](
            C, processed_encoder,
            B, T, I, H,
            C.stride(0), C.stride(1),
            processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
            num_warps=4, num_stages=2,
        )

        # 2) Copy remaining rows [B*T, M) into processed_hidden
        grid_hidden = (triton.cdiv(M - B * T, BLOCK_M), triton.cdiv(H, BLOCK_N))
        copy_rows_hidden_kernel[grid_hidden](
            C, processed_hidden,
            B, T, I, H,
            C.stride(0), C.stride(1),
            processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
            num_warps=4, num_stages=2,
        )

        # If original inputs are fp16/bf16 and you need to match their dtype, cast outputs:
        # processed_encoder = processed_encoder.to(encoder_hidden_states.dtype)
        # processed_hidden = processed_hidden.to(hidden_states.dtype)

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
