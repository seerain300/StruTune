import torch
import triton
import triton.language as tl


@triton.jit
def concat_to_A_rows_kernel(
    enc_ptr,       # *ptr to [B, T, H]
    img_ptr,       # *ptr to [B, I, H]
    A_ptr,         # *ptr to [M_total, H], M_total = B*(T+I)
    B, T, I, H,
    stride_b_e, stride_t_e, stride_h_e,
    stride_b_i, stride_i_i, stride_h_i,
    stride_m_a, stride_h_a,  # A strides: row (m), col (h)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # 2D grid: rows and columns
    pid_m = tl.program_id(axis=0)  # tile along M_total
    pid_n = tl.program_id(axis=1)  # tile along H

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # output rows
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # output columns (H)

    M_total = B * (T + I)
    mask_m = offs_m < M_total

    # Compute batch and seq for each output row
    b_vec = offs_m // (T + I)
    s_vec = offs_m % (T + I)

    # Build source pointers
    # If s_vec < T: from encoder, else from image
    enc_mask = s_vec < T
    src_row_ptrs = tl.where(enc_mask,
                            enc_ptr + b_vec[:, None] * stride_b_e + s_vec[:, None] * stride_t_e,
                            img_ptr + b_vec[:, None] * stride_b_i + (s_vec[:, None] - T) * stride_i_i)

    # Load tiles from source
    a_tile = tl.load(src_row_ptrs + offs_n[None, :] * stride_h_e, mask=mask_m[:, None], other=0)

    # Store into A
    A_ptrs = A_ptr + offs_m[:, None] * stride_m_a + offs_n[None, :] * stride_h_a
    tl.store(A_ptrs, a_tile, mask=mask_m[:, None])


@triton.jit
def matmul_kernel(
    A_ptr,           # *ptr to [M, H], M = B*(T+I)
    BT_ptr,          # *ptr to [H, H] (process_weight.T)
    C_ptr,           # *ptr to [M, H], output
    M, H,
    A_stride_r, A_stride_c,
    BT_stride_h, BT_stride_k,  # BT is [H, H]
    C_stride_r, C_stride_c,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)  # along rows of C (M)
    pid_n = tl.program_id(axis=1)  # along columns of C (H)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)    # [BLOCK_M]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)    # [BLOCK_N]

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over reduction dimension K (H)
    for k in range(0, H, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)  # [BLOCK_K]

        # A tile: [BLOCK_M, BLOCK_K] from A[offs_m, offs_k]
        A_ptrs = A_ptr + offs_m[:, None] * A_stride_r + offs_k[None, :] * A_stride_c
        A_mask = (offs_m[:, None] < M) & (offs_k[None, :] < H)
        a = tl.load(A_ptrs, mask=A_mask, other=0)

        # BT tile: [BLOCK_K, BLOCK_N] from BT[offs_k, offs_n]
        BT_ptrs = BT_ptr + offs_k[:, None] * BT_stride_h + offs_n[None, :] * BT_stride_k
        BT_mask = (offs_k[:, None] < H) & (offs_n[None, :] < H)
        b = tl.load(BT_ptrs, mask=BT_mask, other=0)

        # Accumulate
        acc += tl.dot(a, b)

    # Write back
    C_ptrs = C_ptr + offs_m[:, None] * C_stride_r + offs_n[None, :] * C_stride_c
    C_mask = (offs_m[:, None] < M) & (offs_n[None, :] < H)
    tl.store(C_ptrs, acc, mask=C_mask)


@triton.jit
def copy_rows_segment_kernel(
    src_ptr,         # *ptr to [M_total, H]
    dst_ptr,         # *ptr to [B, S_out, H]
    M_total, B, S_out, H,  # S_out = T or I
    src_stride_r, src_stride_c,
    dst_stride_b, dst_stride_s, dst_stride_c,
    start_src_row,   # for encoder it's 0, for hidden it's B*T
    BLOCK_M: tl.constexpr,  # tile in rows
    BLOCK_N: tl.constexpr,  # tile in H
):
    # 2D grid over rows in segment and H
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = start_src_row + pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # source rows within this segment
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    total_rows = B * S_out
    mask_m = offs_m < total_rows

    # Compute batch and sequence indices for each source row
    b_vec = offs_m // S_out
    s_vec = offs_m % S_out

    # Source pointers: src_ptr + offs_m * src_stride_r
    src_row_ptrs = src_ptr + offs_m[:, None] * src_stride_r + offs_n[None, :] * src_stride_c
    a_tile = tl.load(src_row_ptrs, mask=mask_m[:, None], other=0)

    # Destination pointers: dst_ptr[b, s, :]
    dst_row_ptrs = dst_ptr + b_vec[:, None] * dst_stride_b + s_vec[:, None] * dst_stride_s + offs_n[None, :] * dst_stride_c
    tl.store(dst_row_ptrs, a_tile, mask=mask_m[:, None])


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        hidden_states: [B, I, H]
        encoder_hidden_states: [B, T, H]
        process_weight: [H, H]
        Returns: (processed_encoder: [B, T, H], processed_hidden: [B, I, H])
        """
        # Ensure contiguous tensors
        enc = encoder_hidden_states.contiguous()
        img = hidden_states.contiguous()
        wt = process_weight.contiguous()

        B, T, H = enc.shape
        _, I, _ = img.shape

        # 1) Build concatenated A [M_total, H] where M_total = B*(T+I) using Triton
        M_total = B * (T + I)
        A = torch.empty((M_total, H), dtype=enc.dtype, device=enc.device)

        BLOCK_M = 128
        BLOCK_N = 64
        grid_concat = (triton.cdiv(M_total, BLOCK_M), triton.cdiv(H, BLOCK_N))
        concat_to_A_rows_kernel[grid_concat](
            enc, img, A,
            B, T, I, H,
            enc.stride(0), enc.stride(1), enc.stride(2),
            img.stride(0), img.stride(1), img.stride(2),
            A.stride(0), A.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            num_warps=4, num_stages=2,
        )

        # 2) Matmul: C = A @ process_weight.T using Triton (no torch.matmul)
        BT = wt.transpose(0, 1).contiguous()  # [H, H]
        C = torch.empty((M_total, H), dtype=A.dtype, device=A.device)

        BLOCK_M_G = 64
        BLOCK_N_G = 64
        BLOCK_K_G = 32
        grid_gemm = (triton.cdiv(M_total, BLOCK_M_G), triton.cdiv(H, BLOCK_N_G))
        matmul_kernel[grid_gemm](
            A, BT, C,
            M_total, H,
            A.stride(0), A.stride(1),
            BT.stride(0), BT.stride(1),
            C.stride(0), C.stride(1),
            BLOCK_M=BLOCK_M_G, BLOCK_N=BLOCK_N_G, BLOCK_K=BLOCK_K_G,
            num_warps=4, num_stages=2,
        )

        # 3) Split outputs via Triton: processed_encoder [B, T, H] and processed_hidden [B, I, H]
        processed_encoder = torch.empty((B, T, H), dtype=C.dtype, device=C.device)
        processed_hidden = torch.empty((B, I, H), dtype=C.dtype, device=C.device)

        # Copy encoder rows: rows 0 to B*T
        grid_copy_e = (triton.cdiv(B * T, 64), triton.cdiv(H, 64))
        copy_rows_segment_kernel[grid_copy_e](
            C, processed_encoder,
            B * T, B, T, H,
            C.stride(0), C.stride(1),
            processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
            start_src_row=0,
            BLOCK_M=64, BLOCK_N=64,
            num_warps=2, num_stages=2,
        )

        # Copy hidden rows: rows B*T to (B*T + B*I)
        grid_copy_h = (triton.cdiv(B * I, 64), triton.cdiv(H, 64))
        copy_rows_segment_kernel[grid_copy_h](
            C, processed_hidden,
            B * I, B, I, H,
            C.stride(0), C.stride(1),
            processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
            start_src_row=B * T,
            BLOCK_M=64, BLOCK_N=64,
            num_warps=2, num_stages=2,
        )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
