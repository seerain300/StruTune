import torch
import triton
import triton.language as tl


@triton.jit
def concat_seqs_kernel(
    ehs_ptr,   # *encoder_hidden_states [B, L_txt, D]
    hs_ptr,    # *hidden_states [B, L_img, D]
    dst_ptr,   # *dst [B, L_txt + L_img, D]
    B: tl.int32,
    L_txt: tl.int32,
    L_img: tl.int32,
    D: tl.int32,
    ehs_stride_b: tl.int32, ehs_stride_s: tl.int32, ehs_stride_d: tl.int32,
    hs_stride_b: tl.int32, hs_stride_s: tl.int32, hs_stride_d: tl.int32,
    dst_stride_b: tl.int32, dst_stride_s: tl.int32, dst_stride_d: tl.int32,
    BLOCK_M: tl.constexpr,  # tile size along sequence (rows)
    BLOCK_N: tl.constexpr,  # tile size along feature (cols)
):
    # Grid: (B, ceil((L_txt + L_img)/BLOCK_M), ceil(D/BLOCK_N))
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    total_seq = L_txt + L_img

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    m_mask = m_offsets < total_seq
    n_mask = n_offsets < D
    mask = m_mask[:, None] & n_mask[None, :]

    # Destination pointers
    dst_ptrs = dst_ptr + pid_b * dst_stride_b + m_offsets[:, None] * dst_stride_s + n_offsets[None, :] * dst_stride_d

    # For each m, if m < L_txt -> load from encoder_hidden_states; else load from hidden_states at offset m - L_txt
    e_mask = (m_offsets[None, :] < L_txt) & mask
    h_mask = (~e_mask) & mask

    src_e_ptrs = ehs_ptr + pid_b * ehs_stride_b + m_offsets[:, None] * ehs_stride_s + n_offsets[None, :] * ehs_stride_d
    src_h_ptrs = hs_ptr + pid_b * hs_stride_b + (m_offsets[:, None] - L_txt) * hs_stride_s + n_offsets[None, :] * hs_stride_d

    e_vals = tl.load(src_e_ptrs, mask=e_mask, other=0.0)
    h_vals = tl.load(src_h_ptrs, mask=h_mask, other=0.0)

    out_vals = tl.where(e_mask, e_vals, tl.where(h_mask, h_vals, 0.0))

    tl.store(dst_ptrs, out_vals, mask=mask)


@triton.jit
def batched_matmul_bsmk_dn_kernel(
    A_ptr,    # [B, M, K] = concatenated [B, L_txt + L_img, D]
    W_ptr,    # [K, N] = process_weight.T [D, D]
    C_ptr,    # [B, M, N]
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

    m_mask = m_offsets < M
    n_mask = n_offsets < N
    mask_mn = m_mask[:, None] & n_mask[None, :]

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K in chunks
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < K

        # Load A tile: [BLOCK_M, BLOCK_K]
        A_ptrs = A_ptr + pid_b * stride_Ab + m_offsets[:, None] * stride_Am + k_offsets[None, :] * stride_Ak
        A_mask = mask_mn & (k_mask[None, :])
        A_tile = tl.load(A_ptrs, mask=A_mask, other=0.0)  # A_tile is [BLOCK_M, BLOCK_K]

        # Load W^T tile: [BLOCK_K, BLOCK_N] (W is [K, N])
        W_ptrs = W_ptr + k_offsets[:, None] * stride_Wk + n_offsets[None, :] * stride_Wn
        W_mask = k_mask[:, None] & n_mask[None, :]
        Wt_tile = tl.load(W_ptrs, mask=W_mask, other=0.0)  # Wt_tile is [BLOCK_K, BLOCK_N]

        # Accumulate
        acc += tl.dot(A_tile, Wt_tile)

    # Store result
    C_ptrs = C_ptr + pid_b * stride_Cb + m_offsets[:, None] * stride_Cm + n_offsets[None, :] * stride_Cn
    tl.store(C_ptrs, acc, mask=mask_mn)


@triton.jit
def copy_slice_kernel(
    src_ptr,   # *[B, M, N]
    dst_ptr,   # *[B, M_sub, N]
    B: tl.int32,
    M: tl.int32,
    M_sub: tl.int32,
    N: tl.int32,
    src_stride_b: tl.int32, src_stride_m: tl.int32, src_stride_n: tl.int32,
    dst_stride_b: tl.int32, dst_stride_m: tl.int32, dst_stride_n: tl.int32,
    BLOCK_M: tl.constexpr,  # tile over M_sub
    BLOCK_N: tl.constexpr,  # tile over N
):
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    m_mask = m_offsets < M_sub
    n_mask = n_offsets < N
    mask = m_mask[:, None] & n_mask[None, :]

    src_ptrs = src_ptr + pid_b * src_stride_b + m_offsets[:, None] * src_stride_m + n_offsets[None, :] * src_stride_n
    dst_ptrs = dst_ptr + pid_b * dst_stride_b + m_offsets[:, None] * dst_stride_m + n_offsets[None, :] * dst_stride_n

    vals = tl.load(src_ptrs, mask=mask, other=0.0)
    tl.store(dst_ptrs, vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-only forward:
        - Concatenate encoder_hidden_states and hidden_states along sequence dimension in Triton.
        - Compute processed = concatenated @ process_weight.T in Triton (batched matmul).
        - Split processed into processed_encoder [B, L_txt, D] and processed_hidden [B, L_img, D] using Triton copy kernels.
        Returns: (processed_encoder, processed_hidden)
        """
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "All inputs must be CUDA tensors for Triton."

        B = hidden_states.shape[0]
        L_txt = encoder_hidden_states.shape[1]
        L_img = hidden_states.shape[1]
        D = hidden_states.shape[2]

        # Ensure contiguity for simple stride arithmetic
        ehs = encoder_hidden_states.contiguous()
        hs = hidden_states.contiguous()
        pw_T = process_weight.t().contiguous()  # process_weight.T -> [D, D]

        total_seq = L_txt + L_img

        # 1) Concatenate sequences in Triton: dst [B, total_seq, D]
        dst_concat = torch.empty((B, total_seq, D), device=hs.device, dtype=hs.dtype)

        BLOCK_L = 64  # tile size along sequence
        BLOCK_D = 64  # tile size along feature
        grid_concat = (B, triton.cdiv(total_seq, BLOCK_L), triton.cdiv(D, BLOCK_D))
        concat_seqs_kernel[grid_concat](
            ehs, hs, dst_concat,
            B, L_txt, L_img, D,
            ehs.stride(0), ehs.stride(1), ehs.stride(2),
            hs.stride(0), hs.stride(1), hs.stride(2),
            dst_concat.stride(0), dst_concat.stride(1), dst_concat.stride(2),
            BLOCK_M=BLOCK_L, BLOCK_N=BLOCK_D,
        )

        # 2) Batched matmul in Triton: C = dst_concat @ pw_T
        # A: [B, M, K] = [B, total_seq, D], W: [K, N] = [D, D]
        C = torch.empty((B, total_seq, D), device=hs.device, dtype=hs.dtype)

        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 64
        grid_matmul = (B, triton.cdiv(total_seq, BLOCK_M), triton.cdiv(D, BLOCK_N))
        batched_matmul_bsmk_dn_kernel[grid_matmul](
            dst_concat, pw_T, C,
            B, total_seq, D, D,  # M, N, K all = D
            dst_concat.stride(0), dst_concat.stride(1), dst_concat.stride(2),
            pw_T.stride(0), pw_T.stride(1),
            C.stride(0), C.stride(1), C.stride(2),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        )

        # 3) Split using Triton copy kernels
        # processed_encoder = C[:, :L_txt, :]
        processed_encoder = torch.empty((B, L_txt, D), device=hs.device, dtype=hs.dtype)
        grid_copy_e = (B, triton.cdiv(L_txt, BLOCK_M), triton.cdiv(D, BLOCK_N))
        copy_slice_kernel[grid_copy_e](
            C, processed_encoder,
            B, total_seq, L_txt, D,
            C.stride(0), C.stride(1), C.stride(2),
            processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        )

        # processed_hidden = C[:, L_txt:, :]
        processed_hidden = torch.empty((B, L_img, D), device=hs.device, dtype=hs.dtype)
        grid_copy_h = (B, triton.cdiv(L_img, BLOCK_M), triton.cdiv(D, BLOCK_N))
        copy_slice_kernel[grid_copy_h](
            C, processed_hidden,
            B, total_seq, L_img, D,
            C.stride(0), C.stride(1), C.stride(2),
            processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
