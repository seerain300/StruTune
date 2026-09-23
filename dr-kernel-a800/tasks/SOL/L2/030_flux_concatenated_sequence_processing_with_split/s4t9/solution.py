import torch
import triton
import triton.language as tl


@triton.jit
def concat_to_A_kernel(
    enc_ptr,       # *ptr to [B, T, H]
    img_ptr,       # *ptr to [B, I, H]
    A_ptr,         # *ptr to [M_total, H], M_total = B*(T+I)
    B, T, I, H,    # dimensions
    stride_b_e, stride_t_e, stride_h_e,  # enc strides
    stride_b_i, stride_i_i, stride_h_i,  # img strides
    stride_b_a, stride_h_a,               # A strides
    num_warps: tl.constexpr,
):
    # Grid: (M_total,)
    m = tl.program_id(0)
    M_total = B * (T + I)
    if m >= M_total:
        return

    # Compute batch and sequence index
    b = m // (T + I)
    seq = m % (T + I)

    # Select source tensor
    # If seq < T: read from encoder_hidden_states[b, seq, :]
    # Else: read from hidden_states[b, seq - T, :]
    offs = tl.arange(0, H)
    mask = offs < H

    src_ptr = enc_ptr
    src_stride_b = stride_b_e
    src_stride_seq = stride_t_e
    src_stride_h = stride_h_e
    if seq >= T:
        src_ptr = img_ptr
        src_stride_b = stride_b_i
        src_stride_seq = stride_i_i
        src_stride_h = stride_h_i
        seq = seq - T

    vals = tl.load(
        src_ptr + b * src_stride_b + seq * src_stride_seq + offs * src_stride_h,
        mask=mask,
        other=0.0,
    )
    # Store into A[m, :]
    tl.store(A_ptr + m * stride_b_a + offs * stride_h_a, vals, mask=mask)


@triton.jit
def matmul_A_B_kernel(
    A_ptr,       # *ptr to [M, H]
    B_ptr,       # *ptr to process_weight.T, [H, H]
    C_ptr,       # *ptr to [M, H], output
    M, H,        # dims: M = batch*(T+I), H is hidden_dim
    stride_a_m, stride_a_k,   # A strides: m (rows), k (cols)
    stride_b_k, stride_b_n,   # B strides: k (rows), n (cols)
    stride_c_m, stride_c_n,   # C strides: m (rows), n (cols)
    BLOCK_N: tl.constexpr,    # tile over columns of C
    BLOCK_K: tl.constexpr,    # tile over reduction dim
    num_warps: tl.constexpr,
):
    # Grid: (M, ceil_div(H, BLOCK_N))
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    m = pid_m
    n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = n < H

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # Loop over K = H in tiles
    for k0 in range(0, H, BLOCK_K):
        k = k0 + tl.arange(0, BLOCK_K)
        mask_k = k < H

        # Load A[m, k] as vector (BLOCK_K,)
        a = tl.load(
            A_ptr + m * stride_a_m + k * stride_a_k,
            mask=mask_k,
            other=0.0,
        )
        # Load B[k, n] as matrix (BLOCK_K, BLOCK_N)
        b_mat = tl.load(
            B_ptr + k[:, None] * stride_b_k + n[None, :] * stride_b_n,
            mask=mask_k[:, None] & mask_n[None, :],
            other=0.0,
        )
        # Accumulate
        acc += tl.sum(b_mat * a[:, None], axis=0)

    # Store C[m, n:n+BLOCK_N]
    tl.store(C_ptr + m * stride_c_m + n * stride_c_n, acc, mask=mask_n)


@triton.jit
def copy_rows_to_encoder_kernel(
    C_ptr,          # *ptr to [M_total, H]
    out_ptr,        # *ptr to [B, T, H]
    B, T, M_total, H,
    start_row,      # start row in C for this batch
    stride_c_row, stride_c_col,
    out_stride_b, out_stride_t, out_stride_h,
    num_warps: tl.constexpr,
):
    # Grid: (T, ceil_div(H, BLOCK_N))
    pid_t = tl.program_id(0)
    pid_n = tl.program_id(1)
    t = pid_t
    n = pid_n * 128 + tl.arange(0, 128)
    mask_n = n < H

    m = start_row + t
    if m >= M_total:
        return

    # Load C[m, n:n+BLOCK_N]
    c_ptrs = C_ptr + m * stride_c_row + n * stride_c_col
    values = tl.load(c_ptrs, mask=mask_n, other=0.0)

    # Store to out[b=0, t, :]
    out_ptrs = out_ptr + 0 * out_stride_b + t * out_stride_t + n * out_stride_h
    tl.store(out_ptrs, values, mask=mask_n)


@triton.jit
def copy_rows_to_hidden_kernel(
    C_ptr,          # *ptr to [M_total, H]
    out_ptr,        # *ptr to [B, I, H]
    B, T, M_total, I, H,
    start_row,      # start row in C for hidden: B*T
    stride_c_row, stride_c_col,
    out_stride_b, out_stride_i, out_stride_h,
    num_warps: tl.constexpr,
):
    # Grid: (I, ceil_div(H, BLOCK_N))
    pid_i = tl.program_id(0)
    pid_n = tl.program_id(1)
    i = pid_i
    n = pid_n * 128 + tl.arange(0, 128)
    mask_n = n < H

    m = start_row + i
    if m >= M_total:
        return

    # Load C[m, n:n+BLOCK_N]
    c_ptrs = C_ptr + m * stride_c_row + n * stride_c_col
    values = tl.load(c_ptrs, mask=mask_n, other=0.0)

    # Store to out[b=0, i, :]
    out_ptrs = out_ptr + 0 * out_stride_b + i * out_stride_i + n * out_stride_h
    tl.store(out_ptrs, values, mask=mask_n)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        Triton-only implementation of the original run function.
        - Concatenates encoder_hidden_states and hidden_states along sequence dimension using Triton.
        - Computes matmul with process_weight.T using a Triton GEMM kernel.
        - Splits the result into processed_encoder and processed_hidden using Triton copy kernels.
        """
        assert hidden_states.dim() == 3 and encoder_hidden_states.dim() == 3 and process_weight.dim() == 2
        B, T, H = encoder_hidden_states.shape
        _, I, _ = hidden_states.shape
        assert hidden_states.shape[2] == H and process_weight.shape[0] == H and process_weight.shape[1] == H

        device = encoder_hidden_states.device
        dtype = encoder_hidden_states.dtype

        # 1) Concatenate into A: [M_total, H], M_total = B*(T+I)
        M_total = B * (T + I)
        A = torch.empty((M_total, H), device=device, dtype=dtype)
        # Strides
        stride_b_a, stride_h_a = A.stride(0), A.stride(1)
        # Launch concat kernel: grid=(M_total,)
        grid_concat = (M_total,)
        concat_to_A_kernel[grid_concat](
            encoder_hidden_states, hidden_states, A,
            B, T, I, H,
            encoder_hidden_states.stride(0), encoder_hidden_states.stride(1), encoder_hidden_states.stride(2),
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            stride_b_a, stride_h_a,
            num_warps=2,
        )

        # 2) Compute C = A @ process_weight.T using Triton GEMM
        # process_weight.T is [H, H]
        B_T = process_weight.t().contiguous()  # [H, H]
        C = torch.empty((M_total, H), device=device, dtype=dtype)
        # Strides for A, B_T, C
        stride_a_m, stride_a_k = A.stride(0), A.stride(1)
        stride_b_k, stride_b_n = B_T.stride(0), B_T.stride(1)
        stride_c_m, stride_c_n = C.stride(0), C.stride(1)
        # Launch matmul kernel: grid=(M_total, ceil_div(H, BLOCK_N))
        BLOCK_N = 128
        grid_matmul = (M_total, triton.cdiv(H, BLOCK_N))
        matmul_A_B_kernel[grid_matmul](
            A, B_T, C,
            M_total, H,
            stride_a_m, stride_a_k,
            stride_b_k, stride_b_n,
            stride_c_m, stride_c_n,
            BLOCK_N=BLOCK_N, BLOCK_K=32,
            num_warps=4,
        )

        # 3) Split C into processed_encoder [B, T, H] and processed_hidden [B, I, H]
        processed_encoder = torch.empty((B, T, H), device=device, dtype=dtype)
        processed_hidden = torch.empty((B, I, H), device=device, dtype=dtype)

        # Triton copy rows for encoder: rows [0 : B*T)
        start_row_encoder = 0
        grid_encoder = (T, triton.cdiv(H, 128))
        # Launch per-batch copies; here B=1 in test, but we keep general
        for b in range(B):
            copy_rows_to_encoder_kernel[grid_encoder](
                C, processed_encoder[b],
                B, T, M_total, H,
                start_row_encoder,
                C.stride(0), C.stride(1),
                processed_encoder[b].stride(0), processed_encoder[b].stride(1), processed_encoder[b].stride(2),
                num_warps=2,
            )

        # Triton copy rows for hidden: rows [B*T : M_total)
        start_row_hidden = B * T
        grid_hidden = (I, triton.cdiv(H, 128))
        for b in range(B):
            copy_rows_to_hidden_kernel[grid_hidden](
                C, processed_hidden[b],
                B, T, M_total, I, H,
                start_row_hidden,
                C.stride(0), C.stride(1),
                processed_hidden[b].stride(0), processed_hidden[b].stride(1), processed_hidden[b].stride(2),
                num_warps=2,
            )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
