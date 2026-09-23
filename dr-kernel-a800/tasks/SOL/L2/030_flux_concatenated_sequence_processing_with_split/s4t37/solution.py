import torch
import triton
import triton.language as tl


@triton.jit
def concat_rows_to_A_kernel(
    enc_ptr,       # [B, T, H]
    img_ptr,       # [B, I, H]
    A_ptr,         # [M, H], M = B*(T+I)
    B, T, I, H,    # dims
    stride_b_e, stride_t_e, stride_h_e,  # enc strides
    stride_b_i, stride_i_i, stride_h_i,  # img strides
    stride_m_a, stride_h_a,              # A strides
    BLOCK_M: tl.constexpr = 128,
    BLOCK_N: tl.constexpr = 128,
):
    # Each program handles a tile of output rows
    pid = tl.program_id(0)
    m_start = pid * BLOCK_M
    m_offsets = m_start + tl.arange(0, BLOCK_M)
    n_offsets = tl.arange(0, BLOCK_N)

    M_total = B * (T + I)

    # Mask for valid rows
    mask_rows = m_offsets < M_total

    # Compute (batch, seq) for each m
    b = m_offsets // (T + I)
    s = m_offsets % (T + I)

    # Determine source tensor: encoder if s < T, else image at s - T
    src_e = s < T  # boolean per element
    # Compute base pointers for enc and img
    enc_row_ptrs = enc_ptr + b[:, None] * stride_b_e + s[:, None] * stride_t_e + n_offsets[None, :] * stride_h_e
    enc_mask = mask_rows[:, None] & (n_offsets[None, :] < H)
    val_e = tl.load(enc_row_ptrs, mask=enc_mask, other=0.0)

    # For images, compute seq index relative to image start
    # Note: s - T gives the index within hidden_states for rows where s >= T
    # Masks guard loads.
    img_row_ptrs = img_ptr + b[:, None] * stride_b_i + (s[:, None] - T) * stride_i_i + n_offsets[None, :] * stride_h_i
    img_mask = mask_rows[:, None] & (n_offsets[None, :] < H)
    val_i = tl.load(img_row_ptrs, mask=img_mask, other=0.0)

    # Select: encoder if s < T else image
    # Triton doesn't support tl.where with arbitrary selection; use arithmetic.
    # Convert bool to 0/1
    e_mask = src_e[:, None].to(val_e.dtype)
    i_mask = (1.0 - e_mask)  # where not encoder
    val = val_e * e_mask + val_i * i_mask

    # Store to A[m, :]
    A_ptrs = A_ptr + m_offsets[:, None] * stride_m_a + n_offsets[None, :] * stride_h_a
    tl.store(A_ptrs, val, mask=mask_rows[:, None] & (n_offsets[None, :] < H))


@triton.jit
def batched_matmul_kernel(
    A_ptr,   # [M, H], row-major
    B_ptr,   # [H, H], row-major (process_weight.T)
    C_ptr,   # [M, H], row-major
    M, H,    # sizes: M = B*(T+I), H = hidden_dim
    stride_m_a, stride_h_a,
    stride_h_b, stride_h_bh,  # B strides: [H, H]
    stride_m_c, stride_h_c,
    BLOCK_M: tl.constexpr = 128,
    BLOCK_N: tl.constexpr = 128,
    BLOCK_K: tl.constexpr = 32,
):
    # 2D grid over output rows (M) and columns (H)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over reduction dimension K
    for k in range(0, H, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)

        # A tile: [BLOCK_M, BLOCK_K]
        A_ptrs = A_ptr + offs_m[:, None] * stride_m_a + offs_k[None, :] * stride_h_a
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < H)
        A_tile = tl.load(A_ptrs, mask=a_mask, other=0.0)

        # B tile: [BLOCK_K, BLOCK_N] from B [H, H]
        B_ptrs = B_ptr + offs_k[:, None] * stride_h_b + offs_n[None, :] * stride_h_bh
        b_mask = (offs_k[:, None] < H) & (offs_n[None, :] < H)
        B_tile = tl.load(B_ptrs, mask=b_mask, other=0.0)

        # FMA accumulate
        acc += tl.dot(A_tile, B_tile)

    # Store results
    C_ptrs = C_ptr + offs_m[:, None] * stride_m_c + offs_n[None, :] * stride_h_c
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < H)
    tl.store(C_ptrs, acc, mask=c_mask)


@triton.jit
def copy_rows_encoder_kernel(
    C_ptr,           # [M, H], row-major
    out_ptr,         # [B, T, H], row-major (processed_encoder)
    B, T, I, H,      # dims
    stride_m_c, stride_h_c,     # C strides
    stride_b_o, stride_t_o, stride_h_o,  # out strides
    start_row: tl.constexpr,    # start row in C for this batch (0)
    BLOCK_M: tl.constexpr = 128,
    BLOCK_N: tl.constexpr = 128,
):
    pid_b = tl.program_id(0)  # batch index
    pid_m = tl.program_id(1)  # tile over T
    m_start = pid_m * BLOCK_M
    offs_m = m_start + tl.arange(0, BLOCK_M)  # rows in [0, T)
    offs_n = tl.arange(0, BLOCK_N)            # cols [0, H)

    mask_m = (offs_m < T) & (pid_b < B)
    src_row = start_row + offs_m

    # Pointers to C rows
    C_row_ptrs = C_ptr + src_row[:, None] * stride_m_c + offs_n[None, :] * stride_h_c
    mask = mask_m[:, None] & (offs_n[None, :] < H)

    # Pointers to out [b, s, :]
    out_row = pid_b * T + offs_m
    out_ptrs = out_ptr + out_row[:, None] * stride_b_o + offs_n[None, :] * stride_t_o * stride_h_o

    vals = tl.load(C_row_ptrs, mask=mask, other=0.0)
    tl.store(out_ptrs, vals, mask=mask)


@triton.jit
def copy_rows_hidden_kernel(
    C_ptr,           # [M, H], row-major
    out_ptr,         # [B, I, H], row-major (processed_hidden)
    B, T, I, H,      # dims
    stride_m_c, stride_h_c,     # C strides
    stride_b_o, stride_i_o, stride_h_o,  # out strides
    start_row: tl.constexpr,    # start row in C for this batch (B*T)
    BLOCK_M: tl.constexpr = 128,
    BLOCK_N: tl.constexpr = 128,
):
    pid_b = tl.program_id(0)  # batch index
    pid_m = tl.program_id(1)  # tile over I
    m_start = pid_m * BLOCK_M
    offs_m = m_start + tl.arange(0, BLOCK_M)  # rows in [B*T, B*T + I)
    offs_n = tl.arange(0, BLOCK_N)            # cols [0, H)

    # Total rows M = B*(T+I)
    M_total = B * (T + I)

    mask_m = (offs_m < I) & (pid_b < B)
    src_row = start_row + offs_m

    C_row_ptrs = C_ptr + src_row[:, None] * stride_m_c + offs_n[None, :] * stride_h_c
    mask = mask_m[:, None] & (offs_n[None, :] < H)

    out_row = pid_b * I + offs_m
    out_ptrs = out_ptr + out_row[:, None] * stride_b_o + offs_n[None, :] * stride_i_o * stride_h_o

    vals = tl.load(C_row_ptrs, mask=mask, other=0.0)
    tl.store(out_ptrs, vals, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states: torch.Tensor,
                encoder_hidden_states: torch.Tensor,
                process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-only implementation:
        - Concatenate encoder_hidden_states and hidden_states into A via Triton.
        - Compute processed = A @ process_weight.T via Triton GEMM.
        - Split processed into two outputs via Triton copy kernels.
        """
        # Ensure CUDA tensors (Triton requires GPU)
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, \
            "All inputs must be CUDA tensors for Triton execution."

        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        H = hidden_states.shape[2]
        # Validate dims
        assert encoder_hidden_states.shape[2] == H and process_weight.shape[0] == H and process_weight.shape[1] == H, \
            "Dimension mismatch: hidden_dim must match across inputs."

        # 1) Build A [M, H] via Triton concatenation
        M = B * (T + I)
        A = torch.empty((M, H), dtype=torch.float32, device=hidden_states.device)

        grid_concat = (triton.cdiv(M, 128),)
        concat_rows_to_A_kernel[grid_concat](
            encoder_hidden_states, hidden_states, A,
            B, T, I, H,
            encoder_hidden_states.stride(0), encoder_hidden_states.stride(1), encoder_hidden_states.stride(2),
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            A.stride(0), A.stride(1),
            num_warps=4, num_stages=2,
        )

        # 2) Compute C = A @ process_weight.T via Triton GEMM
        C = torch.empty((M, H), dtype=torch.float32, device=hidden_states.device)

        grid_matmul = (triton.cdiv(M, 128), triton.cdiv(H, 128))
        # process_weight.T is [H, H], default row-major contiguous
        batched_matmul_kernel[grid_matmul](
            A, process_weight.t(), C,
            M, H,
            A.stride(0), A.stride(1),
            process_weight.t().stride(0), process_weight.t().stride(1),
            C.stride(0), C.stride(1),
            num_warps=4, num_stages=2,
        )

        # 3) Split into processed_encoder and processed_hidden via Triton copies
        processed_encoder = torch.empty((B, T, H), dtype=torch.float32, device=hidden_states.device)
        processed_hidden = torch.empty((B, I, H), dtype=torch.float32, device=hidden_states.device)

        # Copy encoder rows: rows [0, B*T)
        grid_enc = (B, triton.cdiv(T, 128))
        copy_rows_encoder_kernel[grid_enc](
            C, processed_encoder,
            B, T, I, H,
            C.stride(0), C.stride(1),
            processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
            start_row=0,
            num_warps=4, num_stages=2,
        )

        # Copy hidden rows: rows [B*T, M)
        grid_hid = (B, triton.cdiv(I, 128))
        copy_rows_hidden_kernel[grid_hid](
            C, processed_hidden,
            B, T, I, H,
            C.stride(0), C.stride(1),
            processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
            start_row=B * T,
            num_warps=4, num_stages=2,
        )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
