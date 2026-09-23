import torch
import triton
import triton.language as tl


@triton.jit
def concat_rows_to_A_kernel(
    enc_ptr,       # [B, T, H]
    img_ptr,       # [B, I, H]
    A_ptr,         # [M, H]
    B, T, I, H,    # int32
    stride_b_e, stride_t_e, stride_h_e,  # enc strides
    stride_b_i, stride_i_i, stride_h_i,  # img strides
    stride_m_a, stride_h_a,               # A strides
    M_total,                                # int32, M = B*(T+I)
    BLOCK_M: tl.constexpr = 128,
):
    pid = tl.program_id(0)
    m_start = pid * BLOCK_M
    offs_m = m_start + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M_total

    # Compute batch and sequence indices
    b_idx = offs_m // (T + I)
    s_idx = offs_m % (T + I)

    # Determine source tensor: encoder if s < T, else image at s - T
    is_encoder = s_idx < T

    enc_row = b_idx * stride_b_e + s_idx * stride_t_e
    img_row = b_idx * stride_b_i + (s_idx - T) * stride_i_i

    src_row_ptrs = tl.where(is_encoder, enc_ptr + enc_row, img_ptr + img_row)
    src_mask = mask_m[:, None]  # exact row masks

    # Load source row into A
    A_row_ptrs = A_ptr + offs_m[:, None] * stride_m_a + tl.arange(0, H)[None, :] * stride_h_a
    A_tile = tl.load(src_row_ptrs, mask=src_mask & (tl.arange(0, H)[None, :] < H), other=0.0)

    # Store to A
    A_store_ptrs = A_ptr + offs_m[:, None] * stride_m_a + tl.arange(0, H)[None, :] * stride_h_a
    tl.store(A_store_ptrs, A_tile, mask=mask_m[:, None])


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
        # Load A_tile: [BLOCK_M, BLOCK_K]
        A_ptrs = A_ptr + offs_m[:, None] * stride_m_a + k[None, :] * stride_h_a
        a_mask = (offs_m[:, None] < M) & (k[None, :] < H)
        A_tile = tl.load(A_ptrs, mask=a_mask, other=0.0)

        # Load B_tile: [BLOCK_K, BLOCK_N]
        B_ptrs = B_ptr + k[:, None] * stride_h_b + offs_n[None, :] * stride_bh_b
        b_mask = (k[:, None] < H) & (offs_n[None, :] < H)
        B_tile = tl.load(B_ptrs, mask=b_mask, other=0.0)

        # Accumulate
        acc += tl.dot(A_tile, B_tile)

    # Store C tile
    C_ptrs = C_ptr + offs_m[:, None] * stride_m_c + offs_n[None, :] * stride_h_c
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < H)
    tl.store(C_ptrs, acc, mask=c_mask)


@triton.jit
def copy_rows_encoder_kernel(
    C_ptr,           # [M, H]
    out_ptr,         # [B, T, H]
    B, T, I, H,      # dims
    stride_m_c, stride_h_c,     # C strides
    stride_b_o, stride_t_o, stride_h_o,  # out strides
    BLOCK_M: tl.constexpr = 128,
    BLOCK_N: tl.constexpr = 128,
):
    pid_b = tl.program_id(0)  # batch index
    pid_m = tl.program_id(1)  # tile over T
    m_start = pid_m * BLOCK_M
    offs_m = m_start + tl.arange(0, BLOCK_M)  # rows in [0, T)
    offs_n = tl.arange(0, BLOCK_N)            # cols [0, H)

    mask_m = (offs_m < T) & (pid_b < B)
    # source rows in C start at batch*T and go for T rows
    src_row = pid_b * T + offs_m

    C_row_ptrs = C_ptr + src_row[:, None] * stride_m_c + offs_n[None, :] * stride_h_c
    mask = mask_m[:, None] & (offs_n[None, :] < H)

    out_row = pid_b * T + offs_m
    out_ptrs = out_ptr + out_row[:, None] * stride_b_o + offs_n[None, :] * stride_t_o * stride_h_o

    vals = tl.load(C_row_ptrs, mask=mask, other=0.0)
    tl.store(out_ptrs, vals, mask=mask)


@triton.jit
def copy_rows_hidden_kernel(
    C_ptr,           # [M, H]
    out_ptr,         # [B, I, H]
    B, T, I, H,      # dims
    stride_m_c, stride_h_c,     # C strides
    stride_b_o, stride_i_o, stride_h_o,  # out strides
    BLOCK_M: tl.constexpr = 128,
    BLOCK_N: tl.constexpr = 128,
):
    pid_b = tl.program_id(0)  # batch index
    pid_m = tl.program_id(1)  # tile over I
    m_start = pid_m * BLOCK_M
    offs_m = m_start + tl.arange(0, BLOCK_M)  # rows in [B*T, B*T + I)
    offs_n = tl.arange(0, BLOCK_N)            # cols [0, H)

    M_total = B * (T + I)
    mask_m = (offs_m < M_total) & ((offs_m >= B * T) & (offs_m < B * T + I)) & (pid_b < B)

    C_row_ptrs = C_ptr + offs_m[:, None] * stride_m_c + offs_n[None, :] * stride_h_c
    mask = mask_m[:, None] & (offs_n[None, :] < H)

    out_row = offs_m - B * T
    out_ptrs = out_ptr + pid_b * stride_b_o + out_row[:, None] * stride_i_o + offs_n[None, :] * stride_h_o

    vals = tl.load(C_row_ptrs, mask=mask, other=0.0)
    tl.store(out_ptrs, vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        Triton-only implementation:
        - Concatenate encoder_hidden_states and hidden_states along sequence dimension without torch.cat.
        - Compute processed = concatenated @ process_weight.T without torch.matmul.
        - Split processed into processed_encoder [B, T, H] and processed_hidden [B, I, H] using Triton copy kernels.
        """
        # Extract shapes
        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        H = hidden_states.shape[2]
        assert encoder_hidden_states.shape[2] == H, "encoder_hidden_states hidden_dim must match hidden_states"

        device = hidden_states.device
        dtype = torch.float32  # align with typical float32 model

        # Prepare concatenated A [M, H], M = B * (T + I)
        M = B * (T + I)
        A = torch.empty((M, H), dtype=dtype, device=device)
        enc = encoder_hidden_states.contiguous().to(dtype)
        img = hidden_states.contiguous().to(dtype)

        # Launch concat kernel: grid over tiles of M
        BLOCK_M_A = 128
        grid_concat = (triton.cdiv(M, BLOCK_M_A),)
        concat_rows_to_A_kernel[grid_concat](
            enc, img, A,
            B, T, I, H,
            enc.stride(0), enc.stride(1), enc.stride(2),
            img.stride(0), img.stride(1), img.stride(2),
            A.stride(0), A.stride(1),
            M,
            num_warps=4, num_stages=2,
        )

        # Prepare B = process_weight.T [H, H]
        Bt = process_weight.t().contiguous().to(dtype)  # [H, H]
        # Output C [M, H]
        C = torch.empty((M, H), dtype=dtype, device=device)

        # Launch matmul kernel: grid over tiles of M and N
        BLOCK_M_M = 128
        BLOCK_N_M = 128
        BLOCK_K_M = 32
        grid_matmul = (triton.cdiv(M, BLOCK_M_M), triton.cdiv(H, BLOCK_N_M))
        batched_matmul_kernel[grid_matmul](
            A, Bt, C,
            M, H,
            A.stride(0), A.stride(1),
            Bt.stride(0), Bt.stride(1),
            C.stride(0), C.stride(1),
            BLOCK_M=BLOCK_M_M, BLOCK_N=BLOCK_N_M, BLOCK_K=BLOCK_K_M,
            num_warps=4, num_stages=2,
        )

        # Allocate outputs and launch copy kernels
        processed_encoder = torch.empty((B, T, H), dtype=dtype, device=device)
        processed_hidden = torch.empty((B, I, H), dtype=dtype, device=device)

        # Encoder copy: rows [0, B*T)
        grid_copy_e = (B, triton.cdiv(T, BLOCK_M_A))
        copy_rows_encoder_kernel[grid_copy_e](
            C, processed_encoder,
            B, T, I, H,
            C.stride(0), C.stride(1),
            processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
            num_warps=4, num_stages=2,
        )

        # Hidden copy: rows [B*T, M)
        grid_copy_h = (B, triton.cdiv(I, BLOCK_M_A))
        copy_rows_hidden_kernel[grid_copy_h](
            C, processed_hidden,
            B, T, I, H,
            C.stride(0), C.stride(1),
            processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
            num_warps=4, num_stages=2,
        )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
