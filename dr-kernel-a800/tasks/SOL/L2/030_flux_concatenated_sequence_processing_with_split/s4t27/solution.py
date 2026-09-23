import torch
import triton
import triton.language as tl


@triton.jit
def concat_to_A_rows_kernel(
    enc_ptr,        # *fp32, [B, T, H]
    img_ptr,        # *fp32, [B, I, H]
    A_ptr,          # *fp32, [M, H], M = B*(T+I)
    B: tl.int32, T: tl.int32, I: tl.int32, H: tl.int32,
    stride_b_e, stride_t_e, stride_h_e,    # enc strides
    stride_b_i, stride_i_i, stride_h_i,    # img strides
    stride_b_a, stride_h_a,                 # A strides
    BLOCK_M: tl.constexpr,  # tile over M rows
    BLOCK_H: tl.constexpr,  # tile over H columns
):
    # Grid: (cdiv(M, BLOCK_M), cdiv(H, BLOCK_H))
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)

    m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    Hdim = H
    M_total = B * (T + I)
    valid_m = m < M_total

    # Compute batch b and sequence index s
    b = m // (T + I)                    # [BLOCK_M]
    s = m % (T + I)                     # [BLOCK_M]

    # Determine source tensor: encoder if s < T else image
    use_enc = s < T                     # [BLOCK_M]

    # Column tile
    n = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)  # [BLOCK_H]
    valid_n = n < Hdim

    # Pointers for A[m, n]
    A_ptrs = A_ptr + m[:, None] * stride_b_a + n[None, :] * stride_h_a  # [BM, BH]

    # Enc pointers
    enc_ptrs = enc_ptr + b[:, None] * stride_b_e + s[:, None] * stride_t_e + n[None, :] * stride_h_e
    # Img pointers (shift s by -T for image stream)
    img_ptrs = img_ptr + b[:, None] * stride_b_i + (s[:, None] - T) * stride_i_i + n[None, :] * stride_h_i

    # Masks
    enc_mask = valid_m[:, None] & use_enc[:, None] & valid_n[None, :]
    img_mask = valid_m[:, None] & (~use_enc)[:, None] & valid_n[None, :]

    # Load and store
    vals = tl.load(enc_ptrs, mask=enc_mask, other=0.0)
    vals += tl.load(img_ptrs, mask=img_mask, other=0.0)
    tl.store(A_ptrs, vals, mask=valid_m[:, None] & valid_n[None, :])


@triton.jit
def matmul_batched_kernel(
    A_ptr,           # *fp32, [M, H]
    B_ptr,           # *fp32, [H, H] (process_weight.T)
    C_ptr,           # *fp32, [M, H]
    M: tl.int32, H: tl.int32,
    stride_a_m, stride_a_h,     # A strides
    stride_b_h, stride_b_k,     # B strides (B is [H, H], so second dim is also H)
    stride_c_m, stride_c_h,     # C strides
    BLOCK_M: tl.constexpr,  # tile over M rows
    BLOCK_N: tl.constexpr,  # tile over output H columns
    BLOCK_K: tl.constexpr,  # tile over reduction H
):
    # Grid: (cdiv(M, BLOCK_M), cdiv(H, BLOCK_N))
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]
    valid_m = m < M
    valid_n = n < H

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, H, BLOCK_K):
        k = k0 + tl.arange(0, BLOCK_K)  # [BLOCK_K]
        valid_k = k < H

        # Load A_tile [BM, BK]: A[m, k]
        A_ptrs = A_ptr + m[:, None] * stride_a_m + k[None, :] * stride_a_h
        A_mask = valid_m[:, None] & valid_k[None, :]
        A_tile = tl.load(A_ptrs, mask=A_mask, other=0.0)

        # Load B_tile [BK, BN]: B[k, n] where B is [H, H]
        B_ptrs = B_ptr + k[:, None] * stride_b_h + n[None, :] * stride_b_k
        B_mask = valid_k[:, None] & valid_n[None, :]
        B_tile = tl.load(B_ptrs, mask=B_mask, other=0.0)

        # Accumulate
        acc += tl.dot(A_tile, B_tile)

    # Store C[m, n]
    C_ptrs = C_ptr + m[:, None] * stride_c_m + n[None, :] * stride_c_h
    C_mask = valid_m[:, None] & valid_n[None, :]
    tl.store(C_ptrs, acc, mask=C_mask)


@triton.jit
def copy_rows_encoder_kernel(
    C_ptr,           # *fp32, [M, H], M = B*(T+I)
    out_ptr,         # *fp32, [B, T, H]
    B: tl.int32, T: tl.int32, H: tl.int32,
    stride_c_m, stride_c_h,
    stride_out_b, stride_out_t, stride_out_h,
    BLOCK_N: tl.constexpr,
):
    # Grid: (B, cdiv(H, BLOCK_N))
    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Rows to copy: 0 .. B*T - 1
    start_row = 0
    end_row = B * T - 1

    # Column tile
    n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    valid_n = n < H

    # Iterate rows in this batch
    for r in range(0, B * T):
        src_row = start_row + r
        b = src_row // T
        t = src_row % T

        C_row_ptr = C_ptr + src_row * stride_c_m
        out_row_ptr = out_ptr + b * stride_out_b + t * stride_out_t

        for n0 in range(0, H, BLOCK_N):
            n_vec = n0 + tl.arange(0, BLOCK_N)
            mask = (n_vec < H) & valid_n
            vals = tl.load(C_row_ptr + (n_vec * stride_c_h), mask=mask, other=0.0)
            tl.store(out_row_ptr + (n_vec * stride_out_h), vals, mask=mask)


@triton.jit
def copy_rows_hidden_kernel(
    C_ptr,           # *fp32, [M, H], M = B*(T+I)
    out_ptr,         # *fp32, [B, I, H]
    B: tl.int32, T: tl.int32, I: tl.int32, H: tl.int32,
    stride_c_m, stride_c_h,
    stride_out_b, stride_out_i, stride_out_h,
    BLOCK_N: tl.constexpr,
):
    # Grid: (B, cdiv(H, BLOCK_N))
    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Rows to copy: B*T .. M-1
    start_row = B * T
    end_row = B * (T + I) - 1

    # Column tile
    n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    valid_n = n < H

    # Iterate rows in this batch
    for r in range(start_row, end_row + 1):
        b = r // (T + I)
        s = r % (T + I)
        i = s - T

        C_row_ptr = C_ptr + r * stride_c_m
        out_row_ptr = out_ptr + b * stride_out_b + i * stride_out_i

        for n0 in range(0, H, BLOCK_N):
            n_vec = n0 + tl.arange(0, BLOCK_N)
            mask = (n_vec < H) & valid_n
            vals = tl.load(C_row_ptr + (n_vec * stride_c_h), mask=mask, other=0.0)
            tl.store(out_row_ptr + (n_vec * stride_out_h), vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Shapes
        B = encoder_hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        H = encoder_hidden_states.shape[2]
        assert hidden_states.shape[2] == H, "hidden_dim must match between encoder_hidden_states and hidden_states"

        # Ensure contiguous inputs
        enc = encoder_hidden_states.contiguous()
        img = hidden_states.contiguous()
        weight_t = process_weight.t().contiguous()  # [H, H]

        # Allocate A [M, H] and C [M, H], M = B*(T+I)
        M = B * (T + I)
        A = torch.empty((M, H), dtype=torch.float32, device=enc.device)
        C = torch.empty((M, H), dtype=torch.float32, device=enc.device)

        # Launch concatenation kernel to build A
        BLOCK_M = 128
        BLOCK_H = 64
        grid_concat = (triton.cdiv(M, BLOCK_M), triton.cdiv(H, BLOCK_H))
        concat_to_A_rows_kernel[grid_concat](
            enc, img, A,
            B, T, I, H,
            enc.stride(0), enc.stride(1), enc.stride(2),
            img.stride(0), img.stride(1), img.stride(2),
            A.stride(0), A.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_H=BLOCK_H,
            num_warps=4, num_stages=2,
        )

        # Launch matmul kernel
        BLOCK_Mm = 64
        BLOCK_Nn = 64
        BLOCK_Kk = 64
        grid_matmul = (triton.cdiv(M, BLOCK_Mm), triton.cdiv(H, BLOCK_Nn))
        matmul_batched_kernel[grid_matmul](
            A, weight_t, C,
            M, H,
            A.stride(0), A.stride(1),
            weight_t.stride(0), weight_t.stride(1),
            C.stride(0), C.stride(1),
            BLOCK_M=BLOCK_Mm, BLOCK_N=BLOCK_Nn, BLOCK_K=BLOCK_Kk,
            num_warps=4, num_stages=2,
        )

        # Allocate outputs
        processed_encoder = torch.empty((B, T, H), dtype=torch.float32, device=enc.device)
        processed_hidden = torch.empty((B, I, H), dtype=torch.float32, device=enc.device)

        # Launch per-batch copy kernels
        BLOCK_Nn = 64
        grid_encoder = (B, triton.cdiv(H, BLOCK_Nn))
        copy_rows_encoder_kernel[grid_encoder](
            C, processed_encoder,
            B, T, H,
            C.stride(0), C.stride(1),
            processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
            BLOCK_N=BLOCK_Nn,
            num_warps=4, num_stages=2,
        )

        grid_hidden = (B, triton.cdiv(H, BLOCK_Nn))
        copy_rows_hidden_kernel[grid_hidden](
            C, processed_hidden,
            B, T, I, H,
            C.stride(0), C.stride(1),
            processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
            BLOCK_N=BLOCK_Nn,
            num_warps=4, num_stages=2,
        )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
