import torch
import triton
import triton.language as tl


@triton.jit
def concat_rows_to_A_kernel(
    enc_ptr,       # [B, T, H]
    img_ptr,       # [B, I, H]
    A_ptr,         # [M, H], M = B*(T+I)
    B, T, I, H,    # dimensions
    stride_b_e, stride_t_e, stride_h_e,  # enc strides
    stride_b_i, stride_i_i, stride_h_i,  # img strides
    stride_m_a, stride_h_a,               # A strides
    BLOCK_M: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    # Each program copies a tile of rows into A
    pid = tl.program_id(axis=0)
    m_start = pid * BLOCK_M
    m = m_start + tl.arange(0, BLOCK_M)
    mask_m = m < (B * (T + I))

    # Compute batch and sequence index for each m
    b = m // (T + I)            # batch index
    seq = m % (T + I)           # sequence index in concatenated stream

    # Determine source tensor: first T rows from encoder, remaining from image
    is_encoder = seq < T

    # Compute source row indices within the source tensor
    src_row_e = seq            # valid when is_encoder is True
    src_row_i = seq - T        # valid when is_encoder is False

    # Compute pointers for loads
    enc_ptrs = enc_ptr + b[:, None] * stride_b_e + src_row_e[:, None] * stride_t_e + tl.arange(0, BLOCK_H)[None, :] * stride_h_e
    img_ptrs = img_ptr + b[:, None] * stride_b_i + src_row_i[:, None] * stride_i_i + tl.arange(0, BLOCK_H)[None, :] * stride_h_i

    # Load with masks: if not encoder, source row_i might be negative; ensure masked correctly
    mask_load_e = mask_m[:, None] & (is_encoder[:, None])
    mask_load_i = mask_m[:, None] & (~is_encoder[:, None])
    vals_e = tl.load(enc_ptrs, mask=mask_load_e, other=0.0)
    vals_i = tl.load(img_ptrs, mask=mask_load_i, other=0.0)
    vals = tl.where(is_encoder[:, None], vals_e, vals_i)

    # Store into A[m, :]
    A_ptrs = A_ptr + m[:, None] * stride_m_a + tl.arange(0, BLOCK_H)[None, :] * stride_h_a
    mask_h = tl.arange(0, BLOCK_H) < H
    tl.store(A_ptrs, vals, mask=mask_m[:, None] & mask_h[None, :])


@triton.jit
def batched_matmul_kernel(
    A_ptr,          # [M, H], M = B*(T+I)
    B_ptr,          # [H, H] (process_weight.T)
    C_ptr,          # [M, H]
    M, H,           # dims
    stride_m_a, stride_h_a,
    stride_h_b, stride_h2_b,  # B strides
    stride_m_c, stride_h_c,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Each program computes a tile C[m0:m0+BLOCK_M, n0:n0+BLOCK_N]
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    m0 = pid_m * BLOCK_M
    n0 = pid_n * BLOCK_N

    m = m0 + tl.arange(0, BLOCK_M)
    n = n0 + tl.arange(0, BLOCK_N)
    mask_m = m < M
    mask_n = n < H

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k0 in range(0, H, BLOCK_K):
        k = k0 + tl.arange(0, BLOCK_K)
        mask_k = k < H

        # Load A tile [BLOCK_M, BLOCK_K]
        A_ptrs = A_ptr + m[:, None] * stride_m_a + k[None, :] * stride_h_a
        a = tl.load(A_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        a = a.to(tl.float32)

        # Load B^T tile [BLOCK_K, BLOCK_N]
        # B^T is [H, H]: we want B[k, n] -> [BLOCK_K, BLOCK_N]
        B_ptrs = B_ptr + k[:, None] * stride_h_b + n[None, :] * stride_h2_b
        b = tl.load(B_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        b = b.to(tl.float32)

        acc += tl.dot(a, b)

    # Store result into C
    C_ptrs = C_ptr + m[:, None] * stride_m_c + n[None, :] * stride_h_c
    tl.store(C_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


@triton.jit
def copy_rows_encoder_kernel(
    C_ptr,             # [M, H]
    out_ptr,           # [B, T, H]
    B, T, I, H,        # dims
    stride_m_c, stride_h_c,          # C strides
    stride_b_o, stride_t_o, stride_h_o,  # output strides
    b,                  # batch index
    BLOCK_M: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    # Copy rows [0 : b*T) of C into out[b, :, :]
    num_rows = b * T
    for m_start in range(0, num_rows, BLOCK_M):
        m = m_start + tl.arange(0, BLOCK_M)
        mask_m = m < (b * T)
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
    C_ptr,             # [M, H]
    out_ptr,           # [B, I, H]
    B, T, I, H,        # dims
    stride_m_c, stride_h_c,          # C strides
    stride_b_o, stride_i_o, stride_h_o,  # output strides
    b,                  # batch index
    BLOCK_M: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    # Copy rows [b*T : (b+1)*T + b*I) of C into out[b, :, :]
    start = B * T
    end = B * (T + I)
    for m_start in range(start, end, BLOCK_M):
        m = m_start + tl.arange(0, BLOCK_M)
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


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        Triton-only implementation:
        1) Concatenate along sequence dim into A [B*(T+I), H] with Triton.
        2) Compute C = A @ process_weight.T with a Triton matmul kernel.
        3) Split C into processed_encoder [B, T, H] and processed_hidden [B, I, H] using Triton copy kernels.
        """
        # Ensure inputs are contiguous
        assert hidden_states.is_contiguous(), "hidden_states must be contiguous"
        assert encoder_hidden_states.is_contiguous(), "encoder_hidden_states must be contiguous"
        assert process_weight.is_contiguous(), "process_weight must be contiguous"

        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        H = hidden_states.shape[2]

        # 1) Concatenate into A using Triton
        M = B * (T + I)
        A = torch.empty((M, H), dtype=torch.float32, device=hidden_states.device)

        # Strides
        stride_b_e, stride_t_e, stride_h_e = encoder_hidden_states.stride()
        stride_b_i, stride_i_i, stride_h_i = hidden_states.stride()
        stride_m_a, stride_h_a = A.stride()

        # Launch concat kernel
        BLOCK_M = 128
        BLOCK_H = 64
        grid_concat = (triton.cdiv(M, BLOCK_M),)
        concat_rows_to_A_kernel[grid_concat](
            encoder_hidden_states, hidden_states, A,
            B, T, I, H,
            stride_b_e, stride_t_e, stride_h_e,
            stride_b_i, stride_i_i, stride_h_i,
            stride_m_a, stride_h_a,
            BLOCK_M=BLOCK_M, BLOCK_H=BLOCK_H,
            num_warps=4, num_stages=2,
        )

        # 2) Compute C = A @ process_weight.T using Triton matmul
        # Process weight is [H, H], transpose is [H, H] already
        B_T = process_weight.t()  # [H, H]
        C = torch.empty((M, H), dtype=torch.float32, device=hidden_states.device)

        stride_m_a_cm, stride_h_a_cm = A.stride()
        stride_h_b, stride_h2_b = B_T.stride()
        stride_m_c, stride_h_c = C.stride()

        BLOCK_M_M = 128
        BLOCK_N_M = 64
        BLOCK_K_M = 64
        grid_mm = (triton.cdiv(M, BLOCK_M_M), triton.cdiv(H, BLOCK_N_M))
        batched_matmul_kernel[grid_mm](
            A, B_T, C,
            M, H,
            stride_m_a_cm, stride_h_a_cm,
            stride_h_b, stride_h2_b,
            stride_m_c, stride_h_c,
            BLOCK_M=BLOCK_M_M, BLOCK_N=BLOCK_N_M, BLOCK_K=BLOCK_K_M,
            num_warps=4, num_stages=2,
        )

        # 3) Split C into processed_encoder and processed_hidden using Triton copy kernels
        processed_encoder = torch.empty((B, T, H), dtype=torch.float32, device=hidden_states.device)
        processed_hidden = torch.empty((B, I, H), dtype=torch.float32, device=hidden_states.device)

        # Copy per-batch encoder rows
        for b in range(B):
            grid_e = (triton.cdiv(b * T, BLOCK_M),)
            copy_rows_encoder_kernel[grid_e](
                C, processed_encoder[b],
                B, T, I, H,
                C.stride(0), C.stride(1),
                processed_encoder[b].stride(0), processed_encoder[b].stride(1), processed_encoder[b].stride(2),
                b,
                BLOCK_M=BLOCK_M, BLOCK_H=BLOCK_H,
                num_warps=4, num_stages=2,
            )

            # Copy per-batch hidden rows
            start = B * T
            end = B * (T + I)
            grid_h = (triton.cdiv(end - start, BLOCK_M),)
            copy_rows_hidden_kernel[grid_h](
                C, processed_hidden[b],
                B, T, I, H,
                C.stride(0), C.stride(1),
                processed_hidden[b].stride(0), processed_hidden[b].stride(1), processed_hidden[b].stride(2),
                b,
                BLOCK_M=BLOCK_M, BLOCK_H=BLOCK_H,
                num_warps=4, num_stages=2,
            )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
