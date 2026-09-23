import torch
import triton
import triton.language as tl


@triton.jit
def concat_rows_to_A_kernel(
    enc_ptr,   # *ptr to [B, T, H]
    img_ptr,   # *ptr to [B, I, H]
    A_ptr,     # *ptr to [M, H], M = B*(T+I)
    B, T, I, H,
    stride_b_e, stride_t_e, stride_h_e,  # strides for enc
    stride_b_i, stride_i_i, stride_h_i,  # strides for img
    stride_m_a, stride_h_a,               # strides for A
    BLOCK_H: tl.constexpr,
):
    # One program per output row m in [0, M)
    m = tl.program_id(0)
    S = T + I
    b = m // S
    s = m % S
    is_encoder = s < T

    # Vector of columns to load/store
    offs_k = tl.arange(0, BLOCK_H)

    # Base pointer for chosen source tensor
    if is_encoder:
        base = enc_ptr + b * stride_b_e + s * stride_t_e
    else:
        base = img_ptr + b * stride_b_i + (s - T) * stride_i_i

    # Store to A[m, :]
    A_row_ptr = A_ptr + m * stride_m_a
    for k0 in range(0, H, BLOCK_H):
        k = k0 + offs_k
        mask_k = k < H
        vals = tl.load(base + k * stride_h_e, mask=mask_k, other=0.0)
        tl.store(A_row_ptr + k * stride_h_a, vals, mask=mask_k)


@triton.jit
def batched_matmul_kernel(
    A_ptr,   # [M, H], input rows
    B_ptr,   # [H, H], process_weight.T
    C_ptr,   # [M, H], output
    M, H,
    stride_am, stride_ak,  # A strides (row, col)
    stride_bh, stride_bb,  # B strides (row, col)
    stride_cm, stride_cn,  # C strides (row, col)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # 2D grid over output tiles
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Reduction over K=H
    for k0 in range(0, H, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)

        # Load A tile [BLOCK_M, BLOCK_K]
        a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < H)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # Load B tile [BLOCK_K, BLOCK_N]
        b_ptrs = B_ptr + offs_k[:, None] * stride_bh + offs_n[None, :] * stride_bb
        b_mask = (offs_k[:, None] < H) & (offs_n[None, :] < H)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Accumulate
        acc += tl.dot(a, b)

    # Store to C
    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < H)
    tl.store(c_ptrs, acc, mask=c_mask)


@triton.jit
def copy_rows_to_single_b_kernel(
    C_ptr,      # [M, H]
    out_ptr,    # [B, N, H] per batch
    B, T, I, H, # dims
    start_row,  # start row in C for this output (b*T for encoder, b*I+B*T for hidden)
    stride_cm, stride_cn,              # C strides
    out_stride_b, out_stride_seq, out_stride_h,  # out strides
    N,                                   # N = T for encoder, I for hidden
    BLOCK_N: tl.constexpr,
):
    # Grid: (batch, tiles over N)
    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # For this batch, copy rows m in [start_row, start_row + N) into out[pid_b, offs_n, :]
    for r in range(0, N):
        m = start_row + r
        c_row_ptr = C_ptr + m * stride_cm + offs_n * stride_cn
        out_row_ptr = out_ptr + pid_b * out_stride_b + r * out_stride_seq + offs_n * out_stride_h
        mask = offs_n < N
        vals = tl.load(c_row_ptr, mask=mask, other=0.0)
        tl.store(out_row_ptr, vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        Triton-only implementation of:
          concatenated = cat([encoder_hidden_states, hidden_states], dim=1)
          processed = concatenated @ process_weight.T
          processed_encoder = processed[:B*T]
          processed_hidden   = processed[B*T:]
        """
        # Ensure contiguity
        enc = encoder_hidden_states.contiguous()
        img = hidden_states.contiguous()
        W = process_weight.contiguous()  # [H, H]
        B = enc.shape[0]
        T = enc.shape[1]
        I = img.shape[1]
        H = enc.shape[2]
        M = B * (T + I)

        # 1) Concatenate into A [M, H] using Triton
        A = torch.empty((M, H), dtype=enc.dtype, device=enc.device)

        # Strides
        stride_b_e, stride_t_e, stride_h_e = enc.stride(0), enc.stride(1), enc.stride(2)
        stride_b_i, stride_i_i, stride_h_i = img.stride(0), img.stride(1), img.stride(2)
        stride_m_a, stride_h_a = A.stride(0), A.stride(1)

        # Launch concat kernel: one program per row m
        grid_concat = (M,)
        concat_rows_to_A_kernel[grid_concat](
            enc, img, A,
            B, T, I, H,
            stride_b_e, stride_t_e, stride_h_e,
            stride_b_i, stride_i_i, stride_h_i,
            stride_m_a, stride_h_a,
            num_warps=4, num_stages=2,
        )

        # 2) Compute C = A @ W.T using Triton batched matmul
        C = torch.empty((M, H), dtype=A.dtype, device=A.device)

        stride_am, stride_ak = A.stride(0), A.stride(1)
        stride_bh, stride_bb = W.stride(0), W.stride(1)
        stride_cm, stride_cn = C.stride(0), C.stride(1)

        # Tile sizes: good defaults; can be tuned further
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 32

        grid_matmul = (triton.cdiv(M, BLOCK_M), triton.cdiv(H, BLOCK_N))
        batched_matmul_kernel[grid_matmul](
            A, W, C,
            M, H,
            stride_am, stride_ak,
            stride_bh, stride_bb,
            stride_cm, stride_cn,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        # 3) Allocate per-batch outputs
        processed_encoder = [torch.empty((T, H), dtype=C.dtype, device=C.device) for _ in range(B)]
        processed_hidden = [torch.empty((I, H), dtype=C.dtype, device=C.device) for _ in range(B)]

        # 4) Split C into two outputs using Triton kernels
        # a) Encoder: rows 0..B*T-1
        for b in range(B):
            start_row = b * T
            out = processed_encoder[b]
            # Grid over batch and column tiles
            grid = (1, triton.cdiv(T, 64))
            copy_rows_to_single_b_kernel[grid](
                C, out,
                B, T, I, H,
                start_row,
                C.stride(0), C.stride(1),
                out.stride(0), out.stride(1), out.stride(2),
                T,
                BLOCK_N=64,
                num_warps=4, num_stages=2,
            )

        # b) Hidden: rows B*T..M-1
        for b in range(B):
            start_row = b * I + B * T
            out = processed_hidden[b]
            grid = (1, triton.cdiv(I, 64))
            copy_rows_to_single_b_kernel[grid](
                C, out,
                B, T, I, H,
                start_row,
                C.stride(0), C.stride(1),
                out.stride(0), out.stride(1), out.stride(2),
                I,
                BLOCK_N=64,
                num_warps=4, num_stages=2,
            )

        # Stack per-batch outputs into [B, T, H] and [B, I, H]
        processed_encoder = torch.stack(processed_encoder, dim=0)  # [B, T, H]
        processed_hidden = torch.stack(processed_hidden, dim=0)    # [B, I, H]

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
