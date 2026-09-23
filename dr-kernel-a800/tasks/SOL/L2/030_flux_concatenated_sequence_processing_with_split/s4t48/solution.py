import torch
import triton
import triton.language as tl


@triton.jit
def concat_rows_to_A_kernel(
    enc_ptr,       # *const float32, [B, T, H]
    img_ptr,       # *const float32, [B, I, H]
    A_ptr,         # *float32, [M, H], M = B*(T+I)
    B, T, I, H,    # int32 dims
    stride_b_e, stride_t_e, stride_h_e,  # enc strides
    stride_b_i, stride_i_i, stride_h_i,  # img strides
    stride_m_a, stride_h_a,               # A strides
    BLOCK_M: tl.constexpr, BLOCK_H: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    M_total = B * (T + I)
    mask_m = m < M_total

    # Determine batch and sequence index
    b = m // (T + I)
    seq = m % (T + I)
    use_enc = seq < T

    # Load source vector for each m
    h = tl.arange(0, BLOCK_H)
    mask_h = h < H
    if use_enc:
        enc_offsets = b[:, None] * stride_b_e + seq[:, None] * stride_t_e + h[None, :] * stride_h_e
        vals = tl.load(enc_ptr + enc_offsets, mask=mask_m[:, None] & mask_h[None, :], other=0.0)
    else:
        # image part, seq - T is the correct index
        img_seq = seq - T
        img_offsets = b[:, None] * stride_b_i + img_seq[:, None] * stride_i_i + h[None, :] * stride_h_i
        vals = tl.load(img_ptr + img_offsets, mask=mask_m[:, None] & mask_h[None, :], other=0.0)

    # Store into A[m, :]
    A_offsets = m[:, None] * stride_m_a + h[None, :] * stride_h_a
    tl.store(A_ptr + A_offsets, vals, mask=mask_m[:, None] & mask_h[None, :])


@triton.jit
def batched_matmul_rowwise_kernel(
    A_ptr,       # *const float32, [M, H]
    Bt_ptr,      # *const float32, [H, H] (process_weight.T)
    C_ptr,       # *float32, [M, H]
    M, H,        # int32
    stride_m_a, stride_h_a,    # A strides
    stride_k_bt, stride_h_bt,  # Bt strides (rows along H, cols along H)
    stride_m_c, stride_h_c,    # C strides
    BLOCK_K: tl.constexpr,
):
    # One program per output row
    m = tl.program_id(axis=0)
    if m >= M:
        return
    # Initialize accumulator (float32)
    acc = tl.zeros((H,), dtype=tl.float32)

    # Iterate over reduction dimension in tiles
    for k in range(0, H, BLOCK_K):
        k_vec = k + tl.arange(0, BLOCK_K)
        mask_k = k_vec < H

        # Load A[m, k:k+BLOCK_K]
        A_offsets = m * stride_m_a + k_vec * stride_h_a
        a_vec = tl.load(A_ptr + A_offsets, mask=mask_k, other=0.0).to(tl.float32)

        # Load Bt[k:k+BLOCK_K, :]
        B_offsets = k_vec[:, None] * stride_k_bt + tl.arange(0, BLOCK_K)[None, :] * stride_h_bt
        # Note: Bt is [H, H], here we load a BLOCK_K x BLOCK_K tile but only need first BLOCK_K columns.
        b_block = tl.load(Bt_ptr + B_offsets, mask=mask_k[:, None] & (tl.arange(0, BLOCK_K)[None, :] < H), other=0.0)
        # Extract the first BLOCK_K columns
        b_vec = b_block[:, 0].to(tl.float32)

        # Accumulate
        acc += tl.sum(a_vec * b_vec, axis=0)

    # Store result C[m, :]
    C_offsets = m * stride_m_c + tl.arange(0, H) * stride_h_c
    tl.store(C_ptr + C_offsets, acc)


@triton.jit
def copy_rows_encoder_kernel(
    C_ptr,               # *const float32, [M, H]
    out_ptr,             # *float32, [B, T, H]
    B, T, I, H,          # dims
    stride_m_c, stride_h_c,        # C strides
    stride_b_o, stride_t_o, stride_h_o,  # output strides
    b,                   # batch index
    BLOCK_M: tl.constexpr, BLOCK_H: tl.constexpr,
):
    # Copy rows [0 : B*T) of C into out[b, :, :]
    num_rows = B * T
    row_start = 0
    for m_start in range(0, num_rows, BLOCK_M):
        m = row_start + m_start + tl.arange(0, BLOCK_M)
        mask_m = m < (B * T)
        t_idx = m % T
        C_offsets = m[:, None] * stride_m_c + tl.arange(0, BLOCK_H)[None, :] * stride_h_c
        out_offsets = b * stride_b_o + t_idx[:, None] * stride_t_o + tl.arange(0, BLOCK_H)[None, :] * stride_h_o
        vals = tl.load(C_ptr + C_offsets, mask=mask_m[:, None], other=0.0)
        tl.store(out_ptr + out_offsets, vals, mask=mask_m[:, None])


@triton.jit
def copy_rows_hidden_kernel(
    C_ptr,               # *const float32, [M, H]
    out_ptr,             # *float32, [B, I, H]
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
    row_start = start
    for m_start in range(0, num_rows, BLOCK_M):
        m = row_start + m_start + tl.arange(0, BLOCK_M)
        mask_m = m < end
        i_idx = (m - start) % I
        C_offsets = m[:, None] * stride_m_c + tl.arange(0, BLOCK_H)[None, :] * stride_h_c
        out_offsets = b * stride_b_o + i_idx[:, None] * stride_i_o + tl.arange(0, BLOCK_H)[None, :] * stride_h_o
        vals = tl.load(C_ptr + C_offsets, mask=mask_m[:, None], other=0.0)
        tl.store(out_ptr + out_offsets, vals, mask=mask_m[:, None])


def _choose_blocks(M: int, H: int):
    # Simple block choices
    BLOCK_M = 128 if M >= 128 else 64
    BLOCK_H = 64 if H >= 64 else 32
    BLOCK_K = 64 if H >= 64 else 32
    return BLOCK_M, BLOCK_H, BLOCK_K


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                encoder_hidden_states: torch.Tensor,
                process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-only implementation:
        - Build A [B*(T+I), H] via concatenation Triton kernel
        - Compute C = A @ process_weight.T via Triton rowwise matmul kernel
        - Split C into processed_encoder and processed_hidden via Triton copy kernels
        """
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "All inputs must be CUDA tensors"
        assert hidden_states.dtype == torch.float32 and encoder_hidden_states.dtype == torch.float32 and process_weight.dtype == torch.float32, "Expected float32 tensors"

        # Ensure contiguity
        enc = encoder_hidden_states.contiguous()
        img = hidden_states.contiguous()
        B, T, H = enc.shape
        I = img.shape[1]

        # 1) Concatenate rows into A [M, H], M = B*(T+I)
        M = B * (T + I)
        A = torch.empty((M, H), dtype=enc.dtype, device=enc.device)

        BLOCK_M, BLOCK_H = 128, 64
        grid = (triton.cdiv(M, BLOCK_M),)
        concat_rows_to_A_kernel[grid](
            enc, img, A,
            B, T, I, H,
            enc.stride(0), enc.stride(1), enc.stride(2),
            img.stride(0), img.stride(1), img.stride(2),
            A.stride(0), A.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_H=BLOCK_H,
            num_warps=4, num_stages=2,
        )

        # 2) Compute C = A @ process_weight.T via Triton rowwise matmul
        C = torch.empty((M, H), dtype=enc.dtype, device=enc.device)
        # process_weight.T is [H, H]
        weight_T = process_weight.t().contiguous()

        grid_rows = (M,)
        BLOCK_K = 64
        batched_matmul_rowwise_kernel[grid_rows](
            A, weight_T, C,
            M, H,
            A.stride(0), A.stride(1),
            weight_T.stride(0), weight_T.stride(1),
            C.stride(0), C.stride(1),
            BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        # 3) Split C into processed_encoder and processed_hidden
        processed_encoder = torch.empty((B, T, H), dtype=C.dtype, device=C.device)
        processed_hidden = torch.empty((B, I, H), dtype=C.dtype, device=C.device)

        for b in range(B):
            # encoder part: rows [0 : B*T)
            grid_e = (triton.cdiv(B * T, 128),)
            copy_rows_encoder_kernel[grid_e](
                C, processed_encoder[b],
                B, T, I, H,
                C.stride(0), C.stride(1),
                processed_encoder[b].stride(0), processed_encoder[b].stride(1), processed_encoder[b].stride(2),
                b,
                BLOCK_M=128, BLOCK_H=64,
                num_warps=4, num_stages=2,
            )
            # hidden part: rows [B*T : B*(T+I))
            grid_h = (triton.cdiv(B * I, 128),)
            copy_rows_hidden_kernel[grid_h](
                C, processed_hidden[b],
                B, T, I, H,
                C.stride(0), C.stride(1),
                processed_hidden[b].stride(0), processed_hidden[b].stride(1), processed_hidden[b].stride(2),
                b,
                BLOCK_M=128, BLOCK_H=64,
                num_warps=4, num_stages=2,
            )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
