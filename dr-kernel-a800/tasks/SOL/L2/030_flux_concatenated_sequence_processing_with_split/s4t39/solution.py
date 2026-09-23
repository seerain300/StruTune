import torch
import triton
import triton.language as tl


@triton.jit
def concat_rows_to_A_kernel(
    enc_ptr,       # *const float, [B, T, H]
    img_ptr,       # *const float, [B, I, H]
    A_ptr,         # *float,        [M, H]
    B, T, I, H,    # int32
    stride_b_e, stride_t_e, stride_h_e,  # enc strides
    stride_b_i, stride_i_i, stride_h_i,  # img strides
    stride_m_a, stride_h_a,               # A strides
    M_total,                               # int32, M = B*(T+I)
    BLOCK_M: tl.constexpr = 128,
):
    pid = tl.program_id(0)
    m_start = pid * BLOCK_M
    offs_m = m_start + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M_total

    # Compute batch and sequence indices from offs_m
    b_idx = offs_m // (T + I)
    s_idx = offs_m % (T + I)

    # Determine source tensor: encoder if s < T, else image at s - T
    is_encoder = s_idx < T

    # Compute source row pointers
    enc_row_ptrs = enc_ptr + b_idx * stride_b_e + s_idx * stride_t_e + tl.arange(0, H) * stride_h_e
    img_row_ptrs = img_ptr + b_idx * stride_b_i + (s_idx - T) * stride_i_i + tl.arange(0, H) * stride_h_i

    src_row_ptrs = tl.where(is_encoder, enc_row_ptrs, img_row_ptrs)

    # A has shape [M, H]; write A[m, :] = source row
    A_row_ptrs = A_ptr + offs_m[:, None] * stride_m_a + tl.arange(0, H)[None, :] * stride_h_a
    vals = tl.load(src_row_ptrs, mask=mask_m[:, None], other=0.0)
    tl.store(A_row_ptrs, vals, mask=mask_m[:, None])


@triton.jit
def batched_matmul_kernel(
    A_ptr,         # [M, H], row-major
    B_ptr,         # [H, H], process_weight.T, row-major
    C_ptr,         # [M, H], output, row-major
    M, H,                  # int32
    stride_m_a, stride_h_a,   # A strides
    stride_h_b, stride_bh_b,  # B strides (H, H)
    stride_m_c, stride_h_c,   # C strides
    BLOCK_M: tl.constexpr = 128,
    BLOCK_N: tl.constexpr = 128,
    BLOCK_K: tl.constexpr = 32,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N
    offs_m = m_start + tl.arange(0, BLOCK_M)
    offs_n = n_start + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, H, BLOCK_K):
        k = k0 + offs_k  # [BLOCK_K]

        # A tile: [BLOCK_M, BLOCK_K]
        A_ptrs = A_ptr + offs_m[:, None] * stride_m_a + k[None, :] * stride_h_a
        a_mask = (offs_m[:, None] < M) & (k[None, :] < H)
        A_tile = tl.load(A_ptrs, mask=a_mask, other=0.0)

        # B tile: [BLOCK_K, BLOCK_N]
        B_ptrs = B_ptr + k[:, None] * stride_h_b + offs_n[None, :] * stride_bh_b
        b_mask = (k[:, None] < H) & (offs_n[None, :] < H)
        B_tile = tl.load(B_ptrs, mask=b_mask, other=0.0)

        acc += tl.dot(A_tile, B_tile)

    # Write C tile
    C_ptrs = C_ptr + offs_m[:, None] * stride_m_c + offs_n[None, :] * stride_h_c
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < H)
    tl.store(C_ptrs, acc, mask=c_mask)


@triton.jit
def copy_rows_encoder_kernel(
    C_ptr,           # [M, H], row-major
    out_ptr,         # [B, T, H], row-major (processed_encoder)
    B, T, I, H,      # dims
    stride_m_c, stride_h_c,     # C strides
    stride_b_o, stride_t_o, stride_h_o,  # out strides
    start_row: tl.constexpr,    # start row in C for this batch (0)
    BLOCK_M: tl.constexpr = 128,
    BLOCK_N: tl.constexpr = 128,
):
    pid_b = tl.program_id(0)  # batch index
    pid_m = tl.program_id(1)  # tile over T
    m_start = pid_m * BLOCK_M
    offs_m = m_start + tl.arange(0, BLOCK_M)  # rows in [0, T)
    offs_n = tl.arange(0, BLOCK_N)            # cols [0, H)

    mask_m = (offs_m < T) & (pid_b < B)
    src_row = start_row + offs_m

    # C row pointers
    C_ptrs = C_ptr + src_row[:, None] * stride_m_c + offs_n[None, :] * stride_h_c
    mask = mask_m[:, None] & (offs_n[None, :] < H)
    vals = tl.load(C_ptrs, mask=mask, other=0.0)

    # out pointers: out[b, s, :] with s = offs_m
    out_ptrs = out_ptr + pid_b * stride_b_o + offs_m[:, None] * stride_t_o + offs_n[None, :] * stride_h_o
    tl.store(out_ptrs, vals, mask=mask)


@triton.jit
def copy_rows_hidden_kernel(
    C_ptr,           # [M, H], row-major
    out_ptr,         # [B, I, H], row-major (processed_hidden)
    B, T, I, H,      # dims
    stride_m_c, stride_h_c,     # C strides
    stride_b_o, stride_i_o, stride_h_o,  # out strides
    start_row: tl.constexpr,    # start row in C for this batch (B*T)
    BLOCK_M: tl.constexpr = 128,
    BLOCK_N: tl.constexpr = 128,
):
    pid_b = tl.program_id(0)  # batch index
    pid_m = tl.program_id(1)  # tile over I
    m_start = pid_m * BLOCK_M
    offs_m = m_start + tl.arange(0, BLOCK_M)  # rows in [0, I)
    offs_n = tl.arange(0, BLOCK_N)            # cols [0, H)

    mask_m = (offs_m < I) & (pid_b < B)
    src_row = start_row + offs_m

    C_ptrs = C_ptr + src_row[:, None] * stride_m_c + offs_n[None, :] * stride_h_c
    mask = mask_m[:, None] & (offs_n[None, :] < H)
    vals = tl.load(C_ptrs, mask=mask, other=0.0)

    out_ptrs = out_ptr + pid_b * stride_b_o + offs_m[:, None] * stride_i_o + offs_n[None, :] * stride_h_o
    tl.store(out_ptrs, vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-only implementation:
        - Concatenate encoder_hidden_states and hidden_states into A [M, H] via Triton kernel.
        - Compute C = A @ process_weight.T via Triton GEMM.
        - Split C into processed_encoder [B, T, H] and processed_hidden [B, I, H] via Triton copy kernels.

        Args:
            hidden_states: [B, I, H]
            encoder_hidden_states: [B, T, H]
            process_weight: [H, H] (no bias)
        Returns:
            processed_encoder: [B, T, H]
            processed_hidden: [B, I, H]
        """
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Tensors must be on CUDA for Triton."
        B, T, H = encoder_hidden_states.shape
        assert hidden_states.shape[0] == B, "Batch size must match between encoder_hidden_states and hidden_states."
        assert hidden_states.shape[2] == H and encoder_hidden_states.shape[2] == H, "Hidden dim must match."
        I = hidden_states.shape[1]
        M = B * (T + I)

        # Make inputs contiguous for predictable strides
        enc = encoder_hidden_states.contiguous()
        img = hidden_states.contiguous()
        weight_T = process_weight.contiguous()  # [H, H]

        # Allocate A [M, H] and C [M, H]
        A = torch.empty((M, H), dtype=torch.float32, device=enc.device)
        C = torch.empty((M, H), dtype=torch.float32, device=enc.device)

        # Launch concatenation kernel
        BLOCK_M = 128
        grid_concat = (triton.cdiv(M, BLOCK_M),)
        concat_rows_to_A_kernel[grid_concat](
            enc, img, A,
            B, T, I, H,
            enc.stride(0), enc.stride(1), enc.stride(2),
            img.stride(0), img.stride(1), img.stride(2),
            A.stride(0), A.stride(1),
            M,  # M_total
            BLOCK_M=BLOCK_M,
        )

        # Launch matmul kernel
        grid_mm = (triton.cdiv(M, 128), triton.cdiv(H, 128))
        batched_matmul_kernel[grid_mm](
            A, weight_T,
            C,
            M, H,
            A.stride(0), A.stride(1),
            weight_T.stride(0), weight_T.stride(1),
            C.stride(0), C.stride(1),
            BLOCK_M=128, BLOCK_N=128, BLOCK_K=32,
            num_warps=4, num_stages=2,
        )

        # Split: allocate outputs and launch copy kernels
        processed_encoder = torch.empty((B, T, H), dtype=torch.float32, device=enc.device)
        processed_hidden = torch.empty((B, I, H), dtype=torch.float32, device=enc.device)

        # Copy encoder rows [0, B*T)
        grid_copy_e = (B, triton.cdiv(T, 128))
        copy_rows_encoder_kernel[grid_copy_e](
            C, processed_encoder,
            B, T, I, H,
            C.stride(0), C.stride(1),
            processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
            start_row=0,
            BLOCK_M=128, BLOCK_N=128,
            num_warps=4, num_stages=2,
        )

        # Copy hidden rows [B*T, M)
        start_row_hidden = B * T
        grid_copy_h = (B, triton.cdiv(I, 128))
        copy_rows_hidden_kernel[grid_copy_h](
            C, processed_hidden,
            B, T, I, H,
            C.stride(0), C.stride(1),
            processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
            start_row=start_row_hidden,
            BLOCK_M=128, BLOCK_N=128,
            num_warps=4, num_stages=2,
        )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
