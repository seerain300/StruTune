import torch
import triton
import triton.language as tl


@triton.jit
def copy_rows_to_A_BTI_kernel(
    src_ptr,        # *T, [B, S, H], S = T + I
    A_ptr,          # *T, [B, S, H]
    B, S, H,        # int32
    stride_b_s, stride_s_s, stride_h_s,   # strides for src
    stride_b_a, stride_s_a, stride_h_a,   # strides for A
    BLOCK_S: tl.constexpr,
):
    # program ids: batch and sequence tile
    b = tl.program_id(0)
    s_block = tl.program_id(1)
    s_start = s_block * BLOCK_S

    offs_s = s_start + tl.arange(0, BLOCK_S)
    mask_s = offs_s < S

    h_range = tl.arange(0, H)
    # For each column h in H, copy row-wise
    for h in range(H):
        src_row_ptr = src_ptr + b * stride_b_s + offs_s * stride_s_s + h * stride_h_s
        A_row_ptr = A_ptr + b * stride_b_a + offs_s * stride_s_a + h * stride_h_a
        # load from src and store to A
        vals = tl.load(src_row_ptr, mask=mask_s, other=0.0)
        tl.store(A_row_ptr, vals, mask=mask_s)


@triton.jit
def matmul_A_BT_C_kernel(
    A_ptr,          # *T, [M, H], M = B*(T+I)
    BT_ptr,         # *T, [H, H]
    C_ptr,          # *T, [M, H]
    M, H,           # int32
    stride_am, stride_ah,
    stride_bth, stride_btk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D launch: over tiles of output rows (M) and cols (N=H)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # rows of A
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # cols of BT and C

    mask_m = offs_m < M
    mask_n = offs_n < H

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Reduction over K=H
    for k0 in range(0, H, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < H

        # A submatrix: [BLOCK_M, BLOCK_K]
        a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ah)
        a_mask = mask_m[:, None] & mask_k[None, :]
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # BT submatrix: [BLOCK_K, BLOCK_N], BT is [H, H]
        bt_ptrs = BT_ptr + (offs_k[:, None] * stride_bth + offs_n[None, :] * stride_btk)
        bt_mask = mask_k[:, None] & mask_n[None, :]
        bt = tl.load(bt_ptrs, mask=bt_mask, other=0.0)

        # acc += a @ bt
        acc += tl.dot(a, bt)

    # Store result to C
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(c_ptrs, acc, mask=c_mask)


@triton.jit
def copy_rows_to_encoder_out_kernel(
    C_ptr,          # *T, [M, H], M = B*(T+I)
    out_ptr,        # *T, [B, T, H]
    M, B, T, H,     # int32
    stride_cm, stride_cn,
    out_stride_b, out_stride_t, out_stride_h,
    start_row: tl.constexpr,  # we'll pass B*T
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    b = tl.program_id(0)
    m_start = start_row + b * T  # row m in C for batch b, first T rows

    offs_m = m_start + tl.arange(0, BLOCK_M)  # rows to copy from C
    offs_n = tl.arange(0, BLOCK_N)            # columns

    mask_m = offs_m < (start_row + b * T + T)
    mask_n = offs_n < H

    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    out_ptrs = out_ptr + (b * out_stride_b + offs_m[:, None] * 0 + offs_n[None, :] * out_stride_h)  # offs_m doesn't exist, fixed b
    # To map, we want out[b, offs_t, h] where offs_t = offs_m - start_row - b*T
    # Let's recompute offs_t per element:
    offs_t = offs_m - start_row - b * T
    out_ptrs = out_ptr + b * out_stride_b + offs_t[:, None] * out_stride_t + offs_n[None, :] * out_stride_h

    c_mask = mask_m[:, None] & mask_n[None, :]
    vals = tl.load(c_ptrs, mask=c_mask, other=0.0)
    tl.store(out_ptrs, vals, mask=c_mask)


@triton.jit
def copy_rows_to_hidden_out_kernel(
    C_ptr,          # *T, [M, H]
    out_ptr,        # *T, [B, I, H]
    M, B, I, H,     # int32
    stride_cm, stride_cn,
    out_stride_b, out_stride_i, out_stride_h,
    start_row: tl.constexpr,  # we'll pass B*T
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    b = tl.program_id(0)
    m_start = start_row + b * I  # row m in C for batch b, next I rows

    offs_m = m_start + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    mask_m = offs_m < (start_row + b * I + I)
    mask_n = offs_n < H

    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    # out indices: b, offs_i = offs_m - start_row - b*I, h
    offs_i = offs_m - start_row - b * I
    out_ptrs = out_ptr + b * out_stride_b + offs_i[:, None] * out_stride_i + offs_n[None, :] * out_stride_h

    c_mask = mask_m[:, None] & mask_n[None, :]
    vals = tl.load(c_ptrs, mask=c_mask, other=0.0)
    tl.store(out_ptrs, vals, mask=c_mask)


# Entry point: ModelNew
class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Ensure dtype float32 for stability and correctness in Triton
        B = hidden_states.shape[0]
        I = hidden_states.shape[1]
        T = encoder_hidden_states.shape[1]
        H = hidden_states.shape[2]
        assert encoder_hidden_states.shape == (B, T, H), "encoder_hidden_states shape must be [B, T, H]"
        assert hidden_states.shape == (B, I, H), "hidden_states shape must be [B, I, H]"
        assert process_weight.shape == (H, H), "process_weight shape must be [H, H]"

        # Build A [B, T+I, H] using Triton kernel: copy rows from encoder and hidden
        S = T + I
        A = torch.empty((B, S, H), dtype=torch.float32, device=hidden_states.device)

        # Copy encoder rows into A[:, :T, :]
        grid_enc = (B, triton.cdiv(T, 64))
        copy_rows_to_A_BTI_kernel[grid_enc](
            encoder_hidden_states, A,
            B, T, H,
            encoder_hidden_states.stride(0), encoder_hidden_states.stride(1), encoder_hidden_states.stride(2),
            A.stride(0), A.stride(1), A.stride(2),
            BLOCK_S=64,
            num_warps=4, num_stages=2,
        )

        # Copy image rows into A[:, T:, :]
        grid_img = (B, triton.cdiv(I, 64))
        # We need a temporary tensor for hidden to copy, or directly compute pointers: A[:, T:, :] area
        # We'll copy hidden directly into A's last I rows using src offset T
        copy_rows_to_A_BTI_kernel[grid_img](
            hidden_states, A,
            B, I, H,
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            A.stride(0), A.stride(1), A.stride(2),
            BLOCK_S=64,
            start_seq=T,  # start copying into A[:, T:, :]
            num_warps=4, num_stages=2,
        )

        # Prepare BT = process_weight.T contiguous in float32
        BT = process_weight.t().contiguous().to(torch.float32)

        # Compute C = A @ BT, A is [B*S, H]
        M = B * S
        C = torch.empty((M, H), dtype=torch.float32, device=A.device)

        # Launch GEMM Triton kernel
        BLOCK_M = 128
        BLOCK_N = 128
        BLOCK_K = 32
        grid_gemm = (triton.cdiv(M, BLOCK_M), triton.cdiv(H, BLOCK_N))
        matmul_A_BT_C_kernel[grid_gemm](
            A, BT, C,
            M, H,
            A.stride(0), A.stride(1),
            BT.stride(0), BT.stride(1),
            C.stride(0), C.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        # Split C into outputs:
        processed_encoder = torch.empty((B, T, H), dtype=torch.float32, device=C.device)
        processed_hidden = torch.empty((B, I, H), dtype=torch.float32, device=C.device)

        # Copy first B*T rows of C into processed_encoder
        start_row_encoder = 0  # rows 0..(B*T-1)
        copy_rows_to_encoder_out_kernel[(B,)](
            C, processed_encoder,
            M, B, T, H,
            C.stride(0), C.stride(1),
            processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
            start_row=start_row_encoder,
            BLOCK_M=128, BLOCK_N=128,
            num_warps=4, num_stages=2,
        )

        # Copy next B*I rows of C into processed_hidden
        start_row_hidden = B * T  # rows (B*T)..(B*T + B*I - 1)
        copy_rows_to_hidden_out_kernel[(B,)](
            C, processed_hidden,
            M, B, I, H,
            C.stride(0), C.stride(1),
            processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
            start_row=start_row_hidden,
            BLOCK_M=128, BLOCK_N=128,
            num_warps=4, num_stages=2,
        )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
