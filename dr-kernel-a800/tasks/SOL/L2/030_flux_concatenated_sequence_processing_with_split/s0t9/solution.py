import triton
import triton.language as tl


@triton.jit
def concat_seqs_kernel(
    ehs_ptr,   # *encoder_hidden_states [B, L_txt, D]
    hs_ptr,    # *hidden_states [B, L_img, D]
    dst_ptr,   # *dst concatenated [B, L_txt + L_img, D]
    B: tl.int32,
    L_txt: tl.int32,
    L_img: tl.int32,
    D: tl.int32,
    ehs_stride_b: tl.int32, ehs_stride_s: tl.int32, ehs_stride_d: tl.int32,
    hs_stride_b: tl.int32, hs_stride_s: tl.int32, hs_stride_d: tl.int32,
    dst_stride_b: tl.int32, dst_stride_s: tl.int32, dst_stride_d: tl.int32,
    BLOCK_M: tl.constexpr,  # tile along sequence
    BLOCK_N: tl.constexpr,  # tile along feature
):
    # Grid: (B, ceil((L_txt + L_img)/BLOCK_M), ceil(D/BLOCK_N))
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    total_seq = L_txt + L_img

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Masks for bounds
    m_mask = m_offsets < total_seq
    n_mask = n_offsets < D
    mask = m_mask[:, None] & n_mask[None, :]

    # Destination pointers for this tile
    dst_ptrs = dst_ptr + pid_b * dst_stride_b + m_offsets[:, None] * dst_stride_s + n_offsets[None, :] * dst_stride_d

    # Determine source for each m: if m < L_txt -> ehs; else -> hs
    src_e_ptrs = ehs_ptr + pid_b * ehs_stride_b + m_offsets[:, None] * ehs_stride_s + n_offsets[None, :] * ehs_stride_d
    src_h_ptrs = hs_ptr + pid_b * hs_stride_b + (m_offsets[:, None] - L_txt) * hs_stride_s + n_offsets[None, :] * hs_stride_d

    # Load with masks; use masks to avoid out-of-bounds
    e_mask = (m_offsets[:, None] < L_txt) & mask
    h_mask = (~e_mask) & mask  # m_offsets >= L_txt

    e_vals = tl.load(src_e_ptrs, mask=e_mask, other=0.0)
    h_vals = tl.load(src_h_ptrs, mask=h_mask, other=0.0)

    # Combine: for m < L_txt, take e_vals; else take h_vals
    out_vals = tl.where(e_mask, e_vals, h_vals)

    tl.store(dst_ptrs, out_vals, mask=mask)


@triton.jit
def batched_matmul_bsmk_dn_kernel(
    A_ptr,    # [B, M, K] = concatenated [B, L_txt + L_img, D]
    W_ptr,    # [K, N] = process_weight.T [D, D]
    C_ptr,    # [B, M, N] output
    B: tl.int32,
    M: tl.int32,
    N: tl.int32,
    K: tl.int32,
    stride_Ab: tl.int32, stride_Am: tl.int32, stride_Ak: tl.int32,
    stride_Wk: tl.int32, stride_Wn: tl.int32,
    stride_Cb: tl.int32, stride_Cm: tl.int32, stride_Cn: tl.int32,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Grid: (B, ceil(M/BLOCK_M), ceil(N/BLOCK_N))
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)

        # A tile: [BLOCK_M, BLOCK_K]
        A_ptrs = A_ptr + pid_b * stride_Ab + m_offsets[:, None] * stride_Am + k_offsets[None, :] * stride_Ak
        A_mask = (m_offsets[:, None] < M) & (k_offsets[None, :] < K)
        A_tile = tl.load(A_ptrs, mask=A_mask, other=0.0)

        # W^T tile: [BLOCK_K, BLOCK_N], W has shape [K, N]
        Wt_ptrs = W_ptr + k_offsets[:, None] * stride_Wk + n_offsets[None, :] * stride_Wn
        Wt_mask = (k_offsets[:, None] < K) & (n_offsets[None, :] < N)
        Wt_tile = tl.load(Wt_ptrs, mask=Wt_mask, other=0.0)

        # Accumulate
        acc += tl.dot(A_tile, Wt_tile)

    # Store result
    C_ptrs = C_ptr + pid_b * stride_Cb + m_offsets[:, None] * stride_Cm + n_offsets[None, :] * stride_Cn
    C_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    tl.store(C_ptrs, acc, mask=C_mask)


@triton.jit
def copy_slice_kernel(
    src_ptr,   # [B, M, N]
    dst_ptr,   # [B, L, N], where L <= M
    B: tl.int32,
    M: tl.int32,
    L: tl.int32,
    N: tl.int32,
    stride_src_b: tl.int32, stride_src_m: tl.int32, stride_src_n: tl.int32,
    stride_dst_b: tl.int32, stride_dst_l: tl.int32, stride_dst_n: tl.int32,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Grid: (B, ceil(L/BLOCK_M), ceil(N/BLOCK_N))
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    l_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    l_mask = l_offsets < L
    n_mask = n_offsets < N
    mask = l_mask[:, None] & n_mask[None, :]

    src_ptrs = src_ptr + pid_b * stride_src_b + l_offsets[:, None] * stride_src_m + n_offsets[None, :] * stride_src_n
    dst_ptrs = dst_ptr + pid_b * stride_dst_b + l_offsets[:, None] * stride_dst_l + n_offsets[None, :] * stride_dst_n

    vals = tl.load(src_ptrs, mask=mask, other=0.0)
    tl.store(dst_ptrs, vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-only forward:
        - Concatenate encoder_hidden_states and hidden_states along sequence dimension (Triton).
        - Compute batched matmul processed = concatenated @ process_weight.T (Triton).
        - Split outputs into processed_encoder and processed_hidden (Triton copy).
        Returns: (processed_encoder [B, L_txt, D], processed_hidden [B, L_img, D])
        """
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "All inputs must be CUDA tensors for Triton."
        assert hidden_states.dtype == torch.float32 and encoder_hidden_states.dtype == torch.float32 and process_weight.dtype == torch.float32, "Use float32 tensors."

        B = hidden_states.shape[0]
        L_txt = encoder_hidden_states.shape[1]
        L_img = hidden_states.shape[1]
        D = hidden_states.shape[2]

        # Ensure contiguous for predictable strides
        ehs = encoder_hidden_states.contiguous()
        hs = hidden_states.contiguous()
        pw_T = process_weight.t().contiguous()  # [D, D], process_weight.T

        # 1) Concatenate sequences in Triton: dst [B, L_txt + L_img, D]
        total_seq = L_txt + L_img
        dst_concat = torch.empty((B, total_seq, D), device=hs.device, dtype=hs.dtype)

        BLOCK_L = 64
        BLOCK_D = 64
        grid_concat = (B, triton.cdiv(total_seq, BLOCK_L), triton.cdiv(D, BLOCK_D))
        concat_seqs_kernel[grid_concat](
            ehs, hs, dst_concat,
            B, L_txt, L_img, D,
            ehs.stride(0), ehs.stride(1), ehs.stride(2),
            hs.stride(0), hs.stride(1), hs.stride(2),
            dst_concat.stride(0), dst_concat.stride(1), dst_concat.stride(2),
            BLOCK_L=BLOCK_L, BLOCK_D=BLOCK_D,
            num_warps=4, num_stages=2,
        )

        # 2) Compute processed = dst_concat @ pw_T (Triton GEMM)
        processed = torch.empty((B, total_seq, D), device=hs.device, dtype=hs.dtype)

        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 32
        grid_matmul = (B, triton.cdiv(total_seq, BLOCK_M), triton.cdiv(D, BLOCK_N))
        batched_matmul_bsmk_dn_kernel[grid_matmul](
            dst_concat, pw_T, processed,
            B, total_seq, D, D,
            dst_concat.stride(0), dst_concat.stride(1), dst_concat.stride(2),
            pw_T.stride(0), pw_T.stride(1),
            processed.stride(0), processed.stride(1), processed.stride(2),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        # 3) Split using Triton copy:
        # processed_encoder = processed[:, :L_txt, :]
        # processed_hidden = processed[:, L_txt:, :]
        processed_encoder = torch.empty((B, L_txt, D), device=hs.device, dtype=hs.dtype)
        processed_hidden = torch.empty((B, L_img, D), device=hs.device, dtype=hs.dtype)

        # Copy first L_txt slices
        grid_copy_encoder = (B, triton.cdiv(L_txt, BLOCK_M), triton.cdiv(D, BLOCK_D))
        copy_slice_kernel[grid_copy_encoder](
            processed, processed_encoder,
            B, total_seq, L_txt, D,
            processed.stride(0), processed.stride(1), processed.stride(2),
            processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_D,
            num_warps=4, num_stages=2,
        )

        # Copy remaining L_img slices
        grid_copy_hidden = (B, triton.cdiv(L_img, BLOCK_M), triton.cdiv(D, BLOCK_D))
        # For hidden, source indices start at L_txt
        src_offset = L_txt
        copy_slice_kernel[grid_copy_hidden](
            processed, processed_hidden,
            B, total_seq, L_img, D,
            processed.stride(0), (processed.stride(1) + src_offset), processed.stride(2),
            processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_D,
            num_warps=4, num_stages=2,
        )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
