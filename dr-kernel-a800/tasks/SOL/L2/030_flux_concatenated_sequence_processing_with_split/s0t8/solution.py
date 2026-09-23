import torch
import triton
import triton.language as tl


@triton.jit
def concat_seqs_kernel(
    ehs_ptr,  # *encoder_hidden_states [B, L_txt, D]
    hs_ptr,   # *hidden_states [B, L_img, D]
    dst_ptr,  # *output concatenated [B, L_txt + L_img, D]
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

    # Compute source masks: upper half from hidden, lower from encoder
    in_hidden = m_offsets[None, :] >= L_txt  # shape [1, BLOCK_M], broadcast to [BLOCK_M, BLOCK_N]
    in_encoder = ~in_hidden

    # Prepare pointers for both sources
    # encoder: [b, m, n]
    src_e_ptrs = ehs_ptr + pid_b * ehs_stride_b + m_offsets[:, None] * ehs_stride_s + n_offsets[None, :] * ehs_stride_d
    # hidden: [b, m - L_txt, n] but only for in_hidden rows; ensure masked loads for out-of-range rows
    src_h_m = m_offsets[:, None] - L_txt  # values are negative for in_encoder rows; masked loads will prevent OOB
    src_h_ptrs = hs_ptr + pid_b * hs_stride_b + src_h_m * hs_stride_s + n_offsets[None, :] * hs_stride_d

    # Safe loads: mask both loads by in_encoder/in_hidden and by out-of-bounds
    e_mask = mask & in_encoder
    h_mask = mask & in_hidden

    e_vals = tl.load(src_e_ptrs, mask=e_mask, other=0.0)
    h_vals = tl.load(src_h_ptrs, mask=h_mask, other=0.0)

    # Select values: for in_hidden rows use h_vals, else e_vals
    out_vals = tl.where(in_hidden[None, :], h_vals, e_vals)  # broadcast in_hidden to [BLOCK_M, BLOCK_N]

    tl.store(dst_ptrs, out_vals, mask=mask)


@triton.jit
def batched_matmul_bsmk_dn_kernel(
    A_ptr,    # [B, M, K] = concatenated [B, L_txt + L_img, D]
    W_ptr,    # [K, N] = process_weight.T [D, D]
    C_ptr,    # [B, M, N] output
    B: tl.int32,
    M: tl.int32,  # sequence length after concat
    N: tl.int32,  # hidden_dim (output dim)
    K: tl.int32,  # hidden_dim (input/output feature dim)
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
    out_mask = m_mask[:, None] & n_mask[None, :]

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < K

        # Load A tile: [BLOCK_M, BLOCK_K]
        A_ptrs = A_ptr + pid_b * stride_Ab + m_offsets[:, None] * stride_Am + k_offsets[None, :] * stride_Ak
        A_tile = tl.load(A_ptrs, mask=(out_mask & k_mask[None, :]), other=0.0)

        # Load W^T tile: [BLOCK_K, BLOCK_N] using W[k, n]
        Wt_ptrs = W_ptr + k_offsets[:, None] * stride_Wk + n_offsets[None, :] * stride_Wn
        Wt_tile = tl.load(Wt_ptrs, mask=(k_mask[:, None] & n_mask[None, :]), other=0.0)

        # Accumulate
        acc += tl.dot(A_tile, Wt_tile)

    # Store result
    C_ptrs = C_ptr + pid_b * stride_Cb + m_offsets[:, None] * stride_Cm + n_offsets[None, :] * stride_Cn
    tl.store(C_ptrs, acc, mask=out_mask)


@triton.jit
def copy_slice_kernel(
    src_ptr,  # [B, M, N]
    dst_ptr,  # [B, L, N], where L <= M
    B: tl.int32,
    M: tl.int32,  # length of src along first L dimension
    L: tl.int32,  # destination length
    N: tl.int32,
    stride_src_b: tl.int32, stride_src_m: tl.int32, stride_src_n: tl.int32,
    stride_dst_b: tl.int32, stride_dst_l: tl.int32, stride_dst_n: tl.int32,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Copy src[:, :L, :] into dst
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    m_mask = m_offsets < L
    n_mask = n_offsets < N
    mask = m_mask[:, None] & n_mask[None, :]

    src_ptrs = src_ptr + pid_b * stride_src_b + m_offsets[:, None] * stride_src_m + n_offsets[None, :] * stride_src_n
    dst_ptrs = dst_ptr + pid_b * stride_dst_b + m_offsets[:, None] * stride_dst_l + n_offsets[None, :] * stride_dst_n

    vals = tl.load(src_ptrs, mask=mask, other=0.0)
    tl.store(dst_ptrs, vals, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-only forward:
        1) Concatenate encoder_hidden_states and hidden_states along sequence dimension (Triton).
        2) Compute processed = concatenated @ process_weight.T using Triton batched matmul (float32).
        3) Split processed into processed_encoder and processed_hidden using Triton copy kernels.
        Returns: (processed_encoder [B, L_txt, D], processed_hidden [B, L_img, D])
        """
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "All inputs must be CUDA tensors for Triton."
        B = hidden_states.shape[0]
        L_txt = encoder_hidden_states.shape[1]
        L_img = hidden_states.shape[1]
        D = hidden_states.shape[2]

        # Ensure contiguous for predictable strides
        ehs = encoder_hidden_states.contiguous()
        hs = hidden_states.contiguous()
        # process_weight is [D, D]; compute W^T = [D, D] for matmul A [B, M, D] @ W^T [D, D]
        # We pass process_weight.T directly to Triton as a contiguous tensor.
        W_T = process_weight.t().contiguous()

        total_seq = L_txt + L_img

        # 1) Concatenate sequences in Triton: dst [B, total_seq, D]
        dst_concat = torch.empty((B, total_seq, D), device=hs.device, dtype=hs.dtype)

        # Tile sizes for concat; choose moderate blocks to cover typical dims
        BLOCK_L = 128
        BLOCK_D = 128
        grid_concat = (B, triton.cdiv(total_seq, BLOCK_L), triton.cdiv(D, BLOCK_D))
        concat_seqs_kernel[grid_concat](
            ehs, hs, dst_concat,
            B, L_txt, L_img, D,
            ehs.stride(0), ehs.stride(1), ehs.stride(2),
            hs.stride(0), hs.stride(1), hs.stride(2),
            dst_concat.stride(0), dst_concat.stride(1), dst_concat.stride(2),
            BLOCK_L=BLOCK_L, BLOCK_D=BLOCK_D,
        )

        # 2) Batched matmul in Triton: processed = dst_concat @ W_T
        # dst_concat: [B, M=total_seq, K=D], W_T: [K=D, N=D]
        processed = torch.empty((B, total_seq, D), device=hs.device, dtype=torch.float32)

        BLOCK_M = 128
        BLOCK_N = 128
        BLOCK_K = 32
        grid_matmul = (B, triton.cdiv(total_seq, BLOCK_M), triton.cdiv(D, BLOCK_N))
        batched_matmul_bsmk_dn_kernel[grid_matmul](
            dst_concat, W_T, processed,
            B, total_seq, D, D,
            dst_concat.stride(0), dst_concat.stride(1), dst_concat.stride(2),
            W_T.stride(0), W_T.stride(1),
            processed.stride(0), processed.stride(1), processed.stride(2),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        )

        # 3) Split using Triton copy kernels
        # processed_encoder: [:, :L_txt, :]
        processed_encoder = torch.empty((B, L_txt, D), device=hs.device, dtype=torch.float32)
        grid_copy_e = (B, triton.cdiv(L_txt, BLOCK_M), triton.cdiv(D, BLOCK_N))
        copy_slice_kernel[grid_copy_e](
            processed, processed_encoder,
            B, total_seq, L_txt, D,
            processed.stride(0), processed.stride(1), processed.stride(2),
            processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        )

        # processed_hidden: [:, L_txt:, :]
        L = L_txt  # start row index for hidden part
        processed_hidden = torch.empty((B, L_img, D), device=hs.device, dtype=torch.float32)
        grid_copy_h = (B, triton.cdiv(L_img, BLOCK_M), triton.cdiv(D, BLOCK_N))
        copy_slice_kernel[grid_copy_h](
            processed, processed_hidden,
            B, total_seq, L_img, D,
            processed.stride(0), processed.stride(1), processed.stride(2),
            processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
