import torch
import triton
import triton.language as tl


@triton.jit
def concat_seqs_kernel(
    ehs_ptr,   # *encoder_hidden_states [B, L_txt, D]
    hs_ptr,    # *hidden_states [B, L_img, D]
    dst_ptr,   # *output concatenated [B, L_txt + L_img, D]
    B: tl.int32,
    L_txt: tl.int32,
    L_img: tl.int32,
    D: tl.int32,
    ehs_stride_b: tl.int32, ehs_stride_s: tl.int32, ehs_stride_d: tl.int32,
    hs_stride_b: tl.int32, hs_stride_s: tl.int32, hs_stride_d: tl.int32,
    dst_stride_b: tl.int32, dst_stride_s: tl.int32, dst_stride_d: tl.int32,
    BLOCK_M: tl.constexpr,  # tile along sequence (rows)
    BLOCK_N: tl.constexpr,  # tile along feature (cols)
):
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    total_seq = L_txt + L_img

    # Tile offsets
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # sequence positions
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # feature positions

    # Bounds masks
    m_mask = m_offsets < total_seq
    n_mask = n_offsets < D
    mask = m_mask[:, None] & n_mask[None, :]

    # Destination pointers
    dst_ptrs = dst_ptr + pid_b * dst_stride_b + m_offsets[:, None] * dst_stride_s + n_offsets[None, :] * dst_stride_d

    # For each m, if m < L_txt -> load from encoder; else load from hidden at offset m - L_txt
    e_mask = (m_offsets[None, :] < L_txt) & mask
    h_mask = (~e_mask) & mask

    # Compute source pointers
    src_e_ptrs = ehs_ptr + pid_b * ehs_stride_b + m_offsets[:, None] * ehs_stride_s + n_offsets[None, :] * ehs_stride_d
    src_h_ptrs = hs_ptr + pid_b * hs_stride_b + (m_offsets[:, None] - L_txt) * hs_stride_s + n_offsets[None, :] * hs_stride_d

    # Load with masks; other=0.0 for out-of-range
    e_vals = tl.load(src_e_ptrs, mask=e_mask, other=0.0)
    h_vals = tl.load(src_h_ptrs, mask=h_mask, other=0.0)

    out_vals = tl.where(e_mask, e_vals, tl.where(h_mask, h_vals, 0.0))
    tl.store(dst_ptrs, out_vals, mask=mask)


@triton.jit
def batched_matmul_bsmk_dn_kernel(
    A_ptr,    # [B, M, K] concatenated
    W_ptr,    # [K, N] = process_weight.T
    C_ptr,    # [B, M, N] output
    B: tl.int32,
    M: tl.int32,
    N: tl.int32,
    K: tl.int32,
    A_stride_b: tl.int32, A_stride_m: tl.int32, A_stride_k: tl.int32,
    W_stride_k: tl.int32, W_stride_n: tl.int32,
    C_stride_b: tl.int32, C_stride_m: tl.int32, C_stride_n: tl.int32,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K in chunks
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)  # [BLOCK_K]

        # Pointers for A[b, m, k] and W[k, n]
        A_ptrs = A_ptr + pid_b * A_stride_b + m_offsets[:, None] * A_stride_m + k_offsets[None, :] * A_stride_k
        W_ptrs = W_ptr + k_offsets[:, None] * W_stride_k + n_offsets[None, :] * W_stride_n

        # Masks
        a_mask = (m_offsets[:, None] < M) & (k_offsets[None, :] < K)
        w_mask = (k_offsets[:, None] < K) & (n_offsets[None, :] < N)

        # Load and cast to fp32 for accumulation
        A_tile = tl.load(A_ptrs, mask=a_mask, other=0.0).to(tl.float32)  # [BLOCK_M, BLOCK_K]
        W_tile = tl.load(W_ptrs, mask=w_mask, other=0.0).to(tl.float32)  # [BLOCK_K, BLOCK_N]

        # Accumulate
        acc += tl.dot(A_tile, W_tile)  # [BLOCK_M, BLOCK_N]

    # Store result
    C_ptrs = C_ptr + pid_b * C_stride_b + m_offsets[:, None] * C_stride_m + n_offsets[None, :] * C_stride_n
    C_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    tl.store(C_ptrs, acc, mask=C_mask)


@triton.jit
def copy_slice_kernel(
    src_ptr,    # *[B, M, N]
    dst_ptr,    # *[B, L, N], L is the slice length
    B: tl.int32,
    M: tl.int32,  # total rows
    L: tl.int32,  # rows to copy
    N: tl.int32,
    src_stride_b: tl.int32, src_stride_m: tl.int32, src_stride_n: tl.int32,
    dst_stride_b: tl.int32, dst_stride_s: tl.int32, dst_stride_n: tl.int32,
    BLOCK_M: tl.constexpr,  # tile along rows
    BLOCK_N: tl.constexpr,  # tile along cols
):
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]

    # Masks for bounds
    m_mask = m_offsets < L
    n_mask = n_offsets < N
    mask = m_mask[:, None] & n_mask[None, :]

    # Source pointers: we copy rows 0..L-1
    src_ptrs = src_ptr + pid_b * src_stride_b + m_offsets[:, None] * src_stride_m + n_offsets[None, :] * src_stride_n

    # Destination pointers: offset by 0 for the first L rows
    dst_ptrs = dst_ptr + pid_b * dst_stride_b + m_offsets[:, None] * dst_stride_s + n_offsets[None, :] * dst_stride_n

    # Load and store
    vals = tl.load(src_ptrs, mask=mask, other=0.0)
    tl.store(dst_ptrs, vals, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-only forward:
        1) Concatenate encoder_hidden_states and hidden_states along sequence dimension (Triton).
        2) Compute batched matmul processed = concatenated @ process_weight.T (Triton).
        3) Split outputs into processed_encoder and processed_hidden (Triton copy kernels).
        Returns: (processed_encoder [B, L_txt, D], processed_hidden [B, L_img, D])
        """
        # Ensure inputs are CUDA and contiguous
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "All inputs must be CUDA tensors for Triton."
        B = hidden_states.shape[0]
        L_txt = encoder_hidden_states.shape[1]
        L_img = hidden_states.shape[1]
        D = hidden_states.shape[2]

        # Make inputs contiguous (important for correct stride-based indexing)
        ehs = encoder_hidden_states.contiguous()
        hs = hidden_states.contiguous()
        # process_weight is [D, D]; we need W = process_weight.T, also contiguous
        W_t = process_weight.t().contiguous()  # [D, D]

        # 1) Concatenate sequences in Triton: dst [B, L_txt + L_img, D]
        total_seq = L_txt + L_img
        dst_concat = torch.empty((B, total_seq, D), device=hs.device, dtype=hs.dtype)

        BLOCK_M = 64
        BLOCK_N = 64
        grid_concat = (B, triton.cdiv(total_seq, BLOCK_M), triton.cdiv(D, BLOCK_N))
        concat_seqs_kernel[grid_concat](
            ehs, hs, dst_concat,
            B, L_txt, L_img, D,
            ehs.stride(0), ehs.stride(1), ehs.stride(2),
            hs.stride(0), hs.stride(1), hs.stride(2),
            dst_concat.stride(0), dst_concat.stride(1), dst_concat.stride(2),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        )

        # 2) Batched matmul in Triton: processed = A @ W, A [B, M, K], W [K, N], C [B, M, N]
        M = total_seq  # L_txt + L_img
        K = D          # hidden_dim
        N = D          # output dim equals hidden_dim

        processed = torch.empty((B, M, N), device=hs.device, dtype=hs.dtype)

        # Strides
        A = dst_concat  # A is [B, M, K]
        A_stride_b, A_stride_m, A_stride_k = A.stride(0), A.stride(1), A.stride(2)
        W = W_t  # [K, N]
        W_stride_k, W_stride_n = W.stride(0), W.stride(1)
        C_stride_b, C_stride_m, C_stride_n = processed.stride(0), processed.stride(1), processed.stride(2)

        BLOCK_M_M = 64
        BLOCK_N_N = 64
        BLOCK_K_K = 64
        grid_matmul = (B, triton.cdiv(M, BLOCK_M_M), triton.cdiv(N, BLOCK_N_N))
        batched_matmul_bsmk_dn_kernel[grid_matmul](
            A, W, processed,
            B, M, N, K,
            A_stride_b, A_stride_m, A_stride_k,
            W_stride_k, W_stride_n,
            C_stride_b, C_stride_m, C_stride_n,
            BLOCK_M=BLOCK_M_M,
            BLOCK_N=BLOCK_N_N,
            BLOCK_K=BLOCK_K_K,
            num_warps=4,
            num_stages=2,
        )

        # 3) Split outputs in Triton
        # processed_encoder = processed[:, :L_txt, :]
        # processed_hidden = processed[:, L_txt:, :]
        processed_encoder = torch.empty((B, L_txt, D), device=hs.device, dtype=hs.dtype)
        grid_copy_encoder = (B, triton.cdiv(L_txt, BLOCK_M), triton.cdiv(D, BLOCK_N))
        copy_slice_kernel[grid_copy_encoder](
            processed, processed_encoder,
            B, M, L_txt, N,
            processed.stride(0), processed.stride(1), processed.stride(2),
            processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        )

        processed_hidden = torch.empty((B, L_img, D), device=hs.device, dtype=hs.dtype)
        # Copy rows [L_txt, L_txt + L_img)
        src_rows = L_txt  # start row for hidden part
        L_rows = L_img     # number of rows to copy
        grid_copy_hidden = (B, triton.cdiv(L_rows, BLOCK_M), triton.cdiv(D, BLOCK_N))
        copy_slice_kernel[grid_copy_hidden](
            processed, processed_hidden,
            B, M, L_rows, N,
            processed.stride(0), processed.stride(1), processed.stride(2),
            processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            src_row_offset=src_rows,  # the kernel itself handles the slice; set here for clarity
        )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
