import torch
import triton
import triton.language as tl


@triton.jit
def concat_rows_to_A_kernel(
    enc_ptr,       # [B, T, H]
    img_ptr,       # [B, I, H]
    A_ptr,         # [M, H], M = B*(T+I)
    B, T, I, H,    # dimensions
    stride_b_e, stride_t_e, stride_h_e,  # strides for encoder
    stride_b_i, stride_i_i, stride_h_i,  # strides for image
    stride_m_a, stride_h_a,               # strides for A (row-major: [M, H])
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # One program per output row m in [0, M)
    m = tl.program_id(0)

    # Compute which batch and sequence this row corresponds to
    S = T + I
    b = m // S
    seq = m % S

    # Determine which input tensor to read from
    is_img = seq >= T
    # Compute source row index within the selected tensor
    src_row = seq - (0 if is_img else T)

    # We'll copy a full row of length H into A[m, :].
    offs_n = tl.arange(0, BLOCK_N)
    mask_n = offs_n < H

    if is_img:
        ptr = img_ptr + b * stride_b_i + src_row * stride_i_i
    else:
        ptr = enc_ptr + b * stride_b_e + src_row * stride_t_e

    vals = tl.load(ptr + offs_n * stride_h, mask=mask_n, other=0.0)

    # Store into A[m, :]
    A_row_ptr = A_ptr + m * stride_m_a
    tl.store(A_row_ptr + offs_n * stride_h_a, vals, mask=mask_n)


@triton.jit
def batched_matmul_kernel(
    A_ptr,     # [M, H], row-major
    BT_ptr,    # [H, H], process_weight.T (row-major)
    C_ptr,     # [M, H] output, row-major
    M, H,
    stride_m_a, stride_h_a,
    stride_h_bt, stride_n_bt,
    stride_m_c, stride_h_c,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # 2D launch: grid over output tiles (m, n)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N

    # Accumulator for this tile
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension in tiles
    for k_start in range(0, H, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < H

        # Load A tile: shape [BLOCK_M, BLOCK_K]
        a_ptrs = A_ptr + m_start * stride_m_a + k_offsets[None, :] * stride_h_a
        a = tl.load(a_ptrs, mask=(tl.arange(0, BLOCK_M)[:, None] < M) & (k_mask[None, :]), other=0.0)

        # Load B^T tile: shape [BLOCK_K, BLOCK_N]
        bt_ptrs = BT_ptr + k_offsets[:, None] * stride_h_bt + n_start * stride_n_bt
        bt = tl.load(bt_ptrs, mask=(k_mask[:, None]) & (tl.arange(0, BLOCK_N)[None, :] < H), other=0.0)

        # Accumulate
        acc += tl.dot(a.to(tl.float32), bt.to(tl.float32))

    # Write back to C
    c_ptrs = C_ptr + (m_start * stride_m_c) + (n_start + tl.arange(0, BLOCK_N)) * stride_h_c
    out_mask = (tl.arange(0, BLOCK_M)[:, None] < M) & (n_start + tl.arange(0, BLOCK_N)[None, :] < H)
    tl.store(c_ptrs, acc, mask=out_mask)


@triton.jit
def copy_rows_to_output_kernel(
    C_ptr,        # [M, H], source
    Out_ptr,      # [B, len, H], destination
    B, T, I, H,   # len = T or I
    stride_m_c, stride_h_c,
    out_stride_b, out_stride_len, out_stride_h,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # This kernel copies rows from C into Out per batch.
    # For processed_encoder: len = T, rows 0..B*T-1
    # For processed_hidden: len = I, rows B*T..M-1
    pid = tl.program_id(0)
    # Each program handles a tile over the output rows
    start_row = pid * BLOCK_M
    end_row = start_row + BLOCK_M

    # Vectorize over columns
    cols = tl.arange(0, BLOCK_N)
    col_mask = cols < H

    # We need to compute for each row m in [start_row, end_row) its corresponding output row in Out.
    # For encoder: m maps to b = m // T, len_idx = m % T
    # For hidden:  m maps to b = (m - B*T) // I, len_idx = (m - B*T) % I
    # However, since the caller will set grid accordingly, we handle general len here.

    # Loop over rows in this tile
    for i in range(0, BLOCK_M):
        m = start_row + i
        row_mask = m < (B * (T + I))  # guard against out-of-range; not necessary if grid is exact

        # Compute b and len_idx based on len passed from host
        if tl.constexpr(len == T):
            b = m // T
            len_idx = m % T
        else:
            # len == I
            m_img = m - (B * T)  # shift for hidden
            b = m_img // I
            len_idx = m_img % I

        # Compute source pointers in C
        c_row_ptr = C_ptr + m * stride_m_c
        # Destination pointer: Out[b, len_idx, :]
        out_row_ptr = Out_ptr + b * out_stride_b + len_idx * out_stride_len

        vals = tl.load(c_row_ptr + cols * stride_h_c, mask=col_mask, other=0.0)
        tl.store(out_row_ptr + cols * out_stride_h, vals, mask=col_mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        # Ensure inputs are contiguous and same dtype
        enc = encoder_hidden_states.contiguous()
        img = hidden_states.contiguous()
        weight = process_weight.contiguous()

        B, T, H = enc.shape
        I = img.shape[1]
        S = T + I
        M = B * S

        # Allocate A: [M, H] and C: [M, H] (C will be result of matmul)
        A = torch.empty((M, H), dtype=enc.dtype, device=enc.device)
        C = torch.empty((M, H), dtype=enc.dtype, device=enc.device)

        # Prepare weight^T: [H, H] (row-major)
        weightT = weight.transpose(0, 1).contiguous()  # [H, H]

        # Step 1: Concatenate rows from encoder_hidden_states and hidden_states into A
        grid_concat = (M,)
        concat_rows_to_A_kernel[grid_concat](
            enc, img, A,
            B, T, I, H,
            enc.stride(0), enc.stride(1), enc.stride(2),
            img.stride(0), img.stride(1), img.stride(2),
            A.stride(0), A.stride(1),
            BLOCK_M=128,  # one program per row, vectorize over H
            BLOCK_N=128,
            num_warps=4, num_stages=2,
        )

        # Step 2: Compute C = A @ weightT using Triton batched matmul
        # Tiling parameters (can be tuned): BLOCK_M=64, BLOCK_N=64, BLOCK_K=32
        grid_matmul = (triton.cdiv(M, 64), triton.cdiv(H, 64))
        batched_matmul_kernel[grid_matmul](
            A, weightT, C,
            M, H,
            A.stride(0), A.stride(1),
            weightT.stride(0), weightT.stride(1),
            C.stride(0), C.stride(1),
            BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
            num_warps=4, num_stages=3,
        )

        # Step 3: Split C into processed_encoder [B, T, H] and processed_hidden [B, I, H]
        # For encoder: rows 0..B*T-1
        processed_encoder = torch.empty((B, T, H), dtype=enc.dtype, device=enc.device)
        grid_copy_e = (triton.cdiv(B * T, 64),)
        copy_rows_to_output_kernel[grid_copy_e](
            C, processed_encoder,
            B, T, I, H,
            C.stride(0), C.stride(1),
            processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
            len==T,  # constexpr path: T
            BLOCK_M=64, BLOCK_N=128,
            num_warps=4, num_stages=2,
        )

        # For hidden: rows B*T..M-1
        processed_hidden = torch.empty((B, I, H), dtype=enc.dtype, device=enc.device)
        grid_copy_h = (triton.cdiv((B * I), 64),)
        copy_rows_to_output_kernel[grid_copy_h](
            C, processed_hidden,
            B, T, I, H,
            C.stride(0), C.stride(1),
            processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
            len==I,  # constexpr path: I
            BLOCK_M=64, BLOCK_N=128,
            num_warps=4, num_stages=2,
        )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
