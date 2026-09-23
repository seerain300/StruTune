import torch
import triton
import triton.language as tl


@triton.jit
def concat_to_A_BTI_kernel(
    enc_ptr,       # *T, [B, T, H]
    img_ptr,       # *T, [B, I, H]
    A_ptr,         # *T, [B, S, H], S = T + I
    B, T, I, H,    # int32
    stride_b_e, stride_t_e, stride_h_e,
    stride_b_i, stride_i_i, stride_h_i,
    stride_b_a, stride_t_a, stride_h_a,
    S,              # int32 = T + I
    BLOCK_T: tl.constexpr,
):
    b = tl.program_id(0)
    s = tl.program_id(1)
    if (b < 0) or (s < 0) or (b >= B) or (s >= S):
        return
    # If s < T: read from encoder, else from image
    if s < T:
        ptr = enc_ptr + b * stride_b_e + s * stride_t_e
    else:
        ptr = img_ptr + b * stride_b_i + (s - T) * stride_i_i

    # Write to A[b, s, :]
    A_row_ptr = A_ptr + b * stride_b_a + s * stride_t_a
    # Copy H elements
    for h in range(0, H):
        val = tl.load(ptr + h * stride_h_e)
        tl.store(A_row_ptr + h * stride_h_a, val)


@triton.jit
def matmul_per_row_kernel(
    A_ptr,       # *float, [M, H], M = B*S
    BT_ptr,      # *float, [H, H] = process_weight.T (contiguous)
    C_ptr,       # *float, [M, H], output
    M, H,        # int32
    stride_m_a, stride_h_a,
    stride_h_b, stride_h_bt,
    stride_m_c, stride_h_c,
    BLOCK_H: tl.constexpr,
):
    m = tl.program_id(0)
    if m < 0 or m >= M:
        return
    # Compute dot(A[m, :], BT[:, n]) for n in [0..H)
    acc = tl.zeros((), dtype=tl.float32)
    for n in range(0, H, BLOCK_H):
        offs_n = n + tl.arange(0, BLOCK_H)
        # Load A[m, offs_n]
        A_row_ptr = A_ptr + m * stride_m_a
        A_vals = tl.load(A_row_ptr + offs_n * stride_h_a, mask=offs_n < H, other=0.0)
        # Load BT[offs_n, :]
        BT_row_ptr = BT_ptr + offs_n * stride_h_b  # BT is [H, H], contiguous
        BT_vals = tl.load(BT_row_ptr + offs_n * stride_h_bt, mask=offs_n < H, other=0.0)
        # Accumulate in float32
        A_vals = A_vals.to(tl.float32)
        BT_vals = BT_vals.to(tl.float32)
        acc += tl.sum(A_vals * BT_vals, axis=0)
    # Store result to C[m, n] for n in [0..H)
    for n in range(0, H):
        tl.store(C_ptr + m * stride_m_c + n * stride_h_c, acc)


@triton.jit
def copy_rows_to_encoder_kernel(
    C_ptr,             # *float, [M, H]
    processed_e_ptr,   # *float, [B, T, H]
    M, B, T, H,
    stride_m_c, stride_h_c,
    stride_b_e, stride_t_e, stride_h_e,
    start_row: tl.constexpr,  # start at 0
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Each program copies one row m to processed_e[b, s, :] where b = m // T, s = m % T
    m = tl.program_id(0)
    if (m < 0) or (m >= start_row) or (m >= start_row + B * T):
        return
    b = m // T
    s = m % T
    src_row_ptr = C_ptr + m * stride_m_c
    dest_row_ptr = processed_e_ptr + b * stride_b_e + s * stride_t_e
    for h in range(0, H):
        val = tl.load(src_row_ptr + h * stride_h_c)
        tl.store(dest_row_ptr + h * stride_h_e, val)


@triton.jit
def copy_rows_to_hidden_kernel(
    C_ptr,             # *float, [M, H]
    processed_h_ptr,   # *float, [B, I, H]
    M, B, I, H,
    stride_m_c, stride_h_c,
    stride_b_h, stride_i_h, stride_h_h,
    start_row: tl.constexpr,  # start at B*T
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Each program copies one row m to processed_h[b, s - B*T, :] where b = (m - start_row) // I, s = (m - start_row) % I
    m = tl.program_id(0)
    if (m < 0) or (m >= start_row) or (m >= start_row + B * I):
        return
    b = (m - start_row) // I
    s = (m - start_row) % I
    src_row_ptr = C_ptr + m * stride_m_c
    dest_row_ptr = processed_h_ptr + b * stride_b_h + s * stride_i_h
    for h in range(0, H):
        val = tl.load(src_row_ptr + h * stride_h_c)
        tl.store(dest_row_ptr + h * stride_h_h, val)


class ModelNew(torch.nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        process_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-only implementation:
        - Concatenates encoder_hidden_states and hidden_states along sequence dim in Triton.
        - Performs linear projection via Triton per-row matmul.
        - Splits the result into encoder and hidden outputs via Triton.
        """
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Triton kernels require CUDA tensors"
        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        H = hidden_states.shape[2]
        S = T + I
        M = B * S

        # Ensure contiguity for predictable strides
        enc = encoder_hidden_states.contiguous()
        hid = hidden_states.contiguous()
        weight = process_weight.contiguous()  # [H, H]

        # 1) Concatenate into A [B, S, H] via Triton
        A = torch.empty((B, S, H), dtype=torch.float32, device=enc.device)
        grid_concat = (B, triton.cdiv(S, 128))
        concat_to_A_BTI_kernel[grid_concat](
            enc, hid, A,
            B, T, I, H,
            enc.stride(0), enc.stride(1), enc.stride(2),
            hid.stride(0), hid.stride(1), hid.stride(2),
            A.stride(0), A.stride(1), A.stride(2),
            S,
            BLOCK_T=128,
            num_warps=4, num_stages=2,
        )

        # 2) Compute processed = A @ process_weight.T using Triton per-row matmul (float32)
        BT = weight.t().contiguous()  # [H, H]
        C = torch.empty((M, H), dtype=torch.float32, device=A.device)

        grid_rows = (M,)
        matmul_per_row_kernel[grid_rows](
            A, BT, C,
            M, H,
            A.stride(0), A.stride(1),
            BT.stride(0), BT.stride(1),
            C.stride(0), C.stride(1),
            BLOCK_H=64,
            num_warps=2, num_stages=2,
        )

        # 3) Split C into processed_encoder and processed_hidden via Triton
        processed_encoder = torch.empty((B, T, H), dtype=torch.float32, device=C.device)
        processed_hidden = torch.empty((B, I, H), dtype=torch.float32, device=C.device)

        # Copy encoder rows: rows [0..B*T)
        start_row_e = 0
        grid_copy_e = (B * T,)
        copy_rows_to_encoder_kernel[grid_copy_e](
            C, processed_encoder,
            M, B, T, H,
            C.stride(0), C.stride(1),
            processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
            start_row=start_row_e,
            BLOCK_M=64, BLOCK_N=64,
            num_warps=2, num_stages=2,
        )

        # Copy hidden rows: rows [B*T..M)
        start_row_h = B * T
        grid_copy_h = (B * I,)
        copy_rows_to_hidden_kernel[grid_copy_h](
            C, processed_hidden,
            M, B, I, H,
            C.stride(0), C.stride(1),
            processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
            start_row=start_row_h,
            BLOCK_M=64, BLOCK_N=64,
            num_warps=2, num_stages=2,
        )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
