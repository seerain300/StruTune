import torch
import triton
import triton.language as tl


@triton.jit
def concat_rows_to_A_kernel(
    enc_ptr,       # pointer to [B, T, H]
    img_ptr,       # pointer to [B, I, H]
    A_ptr,         # pointer to [M, H], M = B*(T+I)
    B, T, I, H,    # dimensions
    stride_b_e, stride_t_e, stride_h_e,  # strides for enc
    stride_b_i, stride_i_i, stride_h_i,  # strides for img
    stride_m_a, stride_h_a,               # strides for A (row-major: [M, H])
    BLOCK_M: tl.constexpr = 128,         # number of rows per program
):
    # program id over rows
    pid = tl.program_id(0)
    m_start = pid * BLOCK_M
    offs_m = m_start + tl.arange(0, BLOCK_M)  # rows in [0, M)
    mask_m = offs_m < (B * (T + I))

    # compute batch and seq index for each row
    b_idx = offs_m // (T + I)          # shape [BLOCK_M]
    s_idx = offs_m % (T + I)           # shape [BLOCK_M]
    is_encoder = s_idx < T             # shape [BLOCK_M]

    # columns
    offs_n = tl.arange(0, H)           # columns [0, H)

    # Build pointers for enc and img, using masks
    # enc: [b_idx, s_idx, offs_n]
    enc_ptrs = (
        enc_ptr
        + b_idx[:, None] * stride_b_e
        + s_idx[:, None] * stride_t_e
        + offs_n[None, :] * stride_h_e
    )
    enc_mask = mask_m[:, None] & (offs_n[None, :] < H)

    # img: [b_idx, s_idx - T, offs_n]
    img_ptrs = (
        img_ptr
        + b_idx[:, None] * stride_b_i
        + (s_idx[:, None] - T) * stride_i_i
        + offs_n[None, :] * stride_h_i
    )
    img_mask = mask_m[:, None] & (offs_n[None, :] < H) & is_encoder[:, None]

    # Load from enc or img based on mask; use 0 for masked elements
    # Since we have two masks (enc_mask and img_mask), we will load both with their masks
    # and then select based on is_encoder. Triton doesn't support dynamic pointer choice per lane,
    # so we create a combined mask and load from enc by default and overwrite where img_mask is True.
    combined_mask = enc_mask | (img_mask & is_encoder[:, None])
    vals = tl.zeros((BLOCK_M, H), dtype=tl.float32)

    # Load enc rows (masked by enc_mask)
    vals = tl.load(enc_ptrs, mask=enc_mask, other=0.0)

    # Overwrite with img rows where is_encoder is True
    # We cannot selectively load here without building a separate pointer tensor, so we do a masked overwrite.
    # First, create a tensor that is zero, then add valid img loads where mask holds.
    # However, Triton doesn't support masked updates like this in a single line across pointers. To keep correctness,
    # we perform two loads with disjoint masks and use tl.where on a third mask, but Triton pointer arithmetic requires actual tensors.
    # Therefore, we instead build a branchless selection by summing enc and scaled img where mask holds.
    # To make it simple and correct: we do two tl.load calls guarded by masks:
    # For lanes where is_encoder is False, we want vals = enc. For lanes where is_encoder is True, we want vals = img.
    # Triton does not allow conditional tl.load based on scalar per-lane, so we perform a two-phase approach by
    # loading enc, then conditionally overwriting selected lanes via masked stores would be complex.
    # Given complexity, we switch to a simpler: for each row, we load from enc for all rows and from img only for encoder rows,
    # but since Triton cannot choose which pointer to load from per-lane, we do a second load into a temporary and use tl.where
    # only by computing both loads into separate tensors. Triton allows us to pass two tensors and combine via masks.
    # However, Triton’s tl.load requires pointer tensors; combining via tl.where needs precomputed vals. To keep it simple
    # and robust, we load enc and leave img rows as zeros for non-encoder lanes, then in the forward we can make H small
    # or rely on H being typical. To be safe, we instead use a single load with combined_mask: we set other=0 for non-encoder lanes,
    # and then load enc for all, and for encoder lanes, we set to the loaded img values by loading into a separate tensor and
    # updating vals where mask holds. But Triton doesn't support per-lane assignment from different pointer tensors in this way.
    #
    # To avoid this limitation, we redesign the concat kernel to do two passes per tile: first store enc rows; then store img rows
    # at the corresponding offset. That requires two kernel launches, which is okay because concat is not the main bottleneck.
    # Here we will implement the two-pass approach: first enc, then img. We'll handle this by splitting the work into two calls
    # in the forward. However, to keep this single kernel, we instead store enc for all rows, and for rows that are encoder, we
    # overwrite the A entry with the corresponding enc row; for hidden rows, A stays at 0. Since the forward allocates A and
    # we want to fill it correctly, we can't rely on A being zero-initialized; better is to do two kernels: one for enc, one for
    # hidden. Given strict requirements, we provide two Triton kernels below for concat: one for encoder rows and one for
    # image rows. This avoids the tricky per-lane pointer selection.

    # For simplicity and correctness, we remove this kernel from forward and instead launch two kernels:
    # concat_encoder_rows, concat_img_rows. The forward now contains these two Triton kernels.

@triton.jit
def concat_encoder_rows_to_A_kernel(
    enc_ptr, A_ptr,
    B, T, I, H,
    stride_b_e, stride_t_e, stride_h_e,
    stride_m_a, stride_h_a,
    BLOCK_M: tl.constexpr = 128,
):
    pid = tl.program_id(0)
    m_start = pid * BLOCK_M
    M = B * T
    offs_m = m_start + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M
    b_idx = offs_m // T
    s_idx = offs_m % T
    offs_n = tl.arange(0, H)

    enc_ptrs = (
        enc_ptr
        + b_idx[:, None] * stride_b_e
        + s_idx[:, None] * stride_t_e
        + offs_n[None, :] * stride_h_e
    )
    mask = mask_m[:, None] & (offs_n[None, :] < H)
    vals = tl.load(enc_ptrs, mask=mask, other=0.0)
    A_ptrs = A_ptr + offs_m[:, None] * stride_m_a + offs_n[None, :] * stride_h_a
    tl.store(A_ptrs, vals, mask=mask)

@triton.jit
def concat_img_rows_to_A_kernel(
    img_ptr, A_ptr,
    B, T, I, H,
    stride_b_i, stride_i_i, stride_h_i,
    stride_m_a, stride_h_a,
    BLOCK_M: tl.constexpr = 128,
):
    pid = tl.program_id(0)
    m_start = pid * BLOCK_M
    M_total = B * (T + I)
    M_enc = B * T
    m_img = m_start + tl.arange(0, BLOCK_M)
    mask_m = m_img < (M_total - M_enc)
    # For image rows, the global m index is m_img + M_enc
    b_idx = (m_img + M_enc) // (T + I)  # equals b_idx for m_img
    s_idx = (m_img + M_enc) % (T + I)   # s_idx in [T, T+I-1]
    offs_n = tl.arange(0, H)
    img_ptrs = (
        img_ptr
        + b_idx[:, None] * stride_b_i
        + (s_idx[:, None] - T) * stride_i_i
        + offs_n[None, :] * stride_h_i
    )
    mask = mask_m[:, None] & (offs_n[None, :] < H)
    vals = tl.load(img_ptrs, mask=mask, other=0.0)
    A_ptrs = A_ptr + (m_img[:, None] + M_enc) * stride_m_a + offs_n[None, :] * stride_h_a
    tl.store(A_ptrs, vals, mask=mask)

@triton.jit
def batched_matmul_kernel(
    A_ptr,      # [M, H], row-major
    B_ptr,      # [H, H], row-major (process_weight.T)
    C_ptr,      # [M, H], row-major
    M, H,       # dimensions
    stride_m_a, stride_h_a,   # strides for A
    stride_h_b, stride_h_bh,  # strides for B (row-major: [H, H])
    stride_m_c, stride_h_c,   # strides for C
    BLOCK_M: tl.constexpr = 128,
    BLOCK_N: tl.constexpr = 64,
    BLOCK_K: tl.constexpr = 32,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    m0 = pid_m * BLOCK_M
    n0 = pid_n * BLOCK_N

    offs_m = m0 + tl.arange(0, BLOCK_M)
    offs_n = n0 + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, H, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        # Load A tile: [BLOCK_M, BLOCK_K]
        A_ptrs = A_ptr + offs_m[:, None] * stride_m_a + offs_k[None, :] * stride_h_a
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < H)
        A_tile = tl.load(A_ptrs, mask=a_mask, other=0.0)

        # Load B tile as [BLOCK_K, BLOCK_N], since B is [H, H]
        B_ptrs = B_ptr + offs_k[:, None] * stride_h_b + offs_n[None, :] * stride_h_bh
        b_mask = (offs_k[:, None] < H) & (offs_n[None, :] < H)
        B_tile = tl.load(B_ptrs, mask=b_mask, other=0.0)

        # Accumulate
        acc += tl.dot(A_tile, B_tile)

    # Write back C tile
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
    start_row: tl.constexpr,    # start row in C for this batch
    BLOCK_M: tl.constexpr = 128,
    BLOCK_N: tl.constexpr = 64,
):
    pid_b = tl.program_id(0)  # batch index
    pid_m = tl.program_id(1)  # tile over T
    m_start = pid_m * BLOCK_M
    offs_m = m_start + tl.arange(0, BLOCK_M)  # rows in [0, T)
    offs_n = tl.arange(0, BLOCK_N)            # cols [0, H)

    # mask for rows within T and M bounds
    mask_m = (offs_m < T) & (pid_b < B)
    # compute source row in C: global row = start_row + offs_m
    src_row = start_row + offs_m

    # Pointers
    C_row_ptrs = C_ptr + src_row[:, None] * stride_m_c + offs_n[None, :] * stride_h_c
    mask = mask_m[:, None] & (offs_n[None, :] < H)

    out_row = pid_b * T + offs_m
    out_ptrs = out_ptr + out_row[:, None] * stride_b_o + offs_n[None, :] * stride_h_o

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
    BLOCK_N: tl.constexpr = 64,
):
    pid_b = tl.program_id(0)  # batch index
    pid_m = tl.program_id(1)  # tile over I
    m_start = pid_m * BLOCK_M
    offs_m = m_start + tl.arange(0, BLOCK_M)  # rows in [0, I)
    offs_n = tl.arange(0, BLOCK_N)            # cols [0, H)

    # mask for rows within I and batch
    mask_m = (offs_m < I) & (pid_b < B)
    # compute source row in C: global row = start_row + offs_m
    src_row = start_row + offs_m

    # Pointers
    C_row_ptrs = C_ptr + src_row[:, None] * stride_m_c + offs_n[None, :] * stride_h_c
    mask = mask_m[:, None] & (offs_n[None, :] < H)

    out_row = pid_b * I + offs_m
    out_ptrs = out_ptr + out_row[:, None] * stride_b_o + offs_n[None, :] * stride_i_o * stride_h_o  # Note: stride_h_o should be stride over hidden dim, but typically hidden dim is last so stride is 1? We pass correct stride from .stride(2).

    vals = tl.load(C_row_ptrs, mask=mask, other=0.0)
    tl.store(out_ptrs, vals, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        # Ensure all inputs are float32 and contiguous
        B, T, H = encoder_hidden_states.shape
        _, I, _ = hidden_states.shape
        # process_weight is [H, H]
        assert process_weight.shape == (H, H), "process_weight must have shape [hidden_dim, hidden_dim]"
        device = hidden_states.device
        assert hidden_states.device == encoder_hidden_states.device == process_weight.device, "All tensors must be on the same device"

        # Allocate A [M, H] and C [M, H], row-major
        M = B * (T + I)
        A = torch.empty((M, H), dtype=torch.float32, device=device)
        C = torch.empty((M, H), dtype=torch.float32, device=device)

        # Launch concat kernels: first encoder rows, then image rows
        BLOCK_M = 128
        grid_enc = (triton.cdiv(B * T, BLOCK_M),)
        concat_encoder_rows_to_A_kernel[grid_enc](
            encoder_hidden_states, A,
            B, T, I, H,
            encoder_hidden_states.stride(0), encoder_hidden_states.stride(1), encoder_hidden_states.stride(2),
            A.stride(0), A.stride(1),
            BLOCK_M=BLOCK_M,
            num_warps=4, num_stages=2,
        )

        grid_img = (triton.cdiv(B * I, BLOCK_M),)
        concat_img_rows_to_A_kernel[grid_img](
            hidden_states, A,
            B, T, I, H,
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            A.stride(0), A.stride(1),
            BLOCK_M=BLOCK_M,
            num_warps=4, num_stages=2,
        )

        # Matmul: C = A @ process_weight.T
        # B_ptr is process_weight.T, row-major [H, H]
        B_ptr = process_weight.t()
        # Launch GEMM
        BLOCK_M2 = 128
        BLOCK_N2 = 64
        BLOCK_K2 = 32
        grid_matmul = (triton.cdiv(M, BLOCK_M2), triton.cdiv(H, BLOCK_N2))
        batched_matmul_kernel[grid_matmul](
            A, B_ptr, C,
            M, H,
            A.stride(0), A.stride(1),
            B_ptr.stride(0), B_ptr.stride(1),
            C.stride(0), C.stride(1),
            BLOCK_M=BLOCK_M2, BLOCK_N=BLOCK_N2, BLOCK_K=BLOCK_K2,
            num_warps=4, num_stages=2,
        )

        # Split outputs
        processed_encoder = torch.empty((B, T, H), dtype=torch.float32, device=device)
        processed_hidden = torch.empty((B, I, H), dtype=torch.float32, device=device)

        # Copy encoder rows: rows [0 : B*T) → processed_encoder
        grid_encoder = (B, triton.cdiv(T, BLOCK_M))
        start_row_encoder = 0
        copy_rows_encoder_kernel[grid_encoder](
            C, processed_encoder,
            B, T, I, H,
            C.stride(0), C.stride(1),
            processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
            start_row_encoder,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            num_warps=4, num_stages=2,
        )

        # Copy hidden rows: rows [B*T : M) → processed_hidden
        grid_hidden = (B, triton.cdiv(I, BLOCK_M))
        start_row_hidden = B * T
        copy_rows_hidden_kernel[grid_hidden](
            C, processed_hidden,
            B, T, I, H,
            C.stride(0), C.stride(1),
            processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
            start_row_hidden,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            num_warps=4, num_stages=2,
        )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
