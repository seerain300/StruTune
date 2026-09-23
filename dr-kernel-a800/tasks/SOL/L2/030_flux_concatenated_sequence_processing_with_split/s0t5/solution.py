import torch
import triton
import triton.language as tl


# Triton kernel: concatenate encoder_hidden_states and hidden_states along sequence dim (T)
# dst[:, :L_txt, :] = encoder_hidden_states
# dst[:, L_txt:, :] = hidden_states
@triton.jit
def concat_seqs_kernel(
    src1_ptr,  # *ptr to encoder_hidden_states [B, L_txt, D]
    src2_ptr,  # *ptr to hidden_states [B, L_img, D]
    dst_ptr,   # *ptr to dst [B, L_txt + L_img, D]
    B: tl.int32,
    L_txt: tl.int32,
    L_img: tl.int32,
    D: tl.int32,
    stride_src1_b: tl.int32, stride_src1_t: tl.int32, stride_src1_d: tl.int32,
    stride_src2_b: tl.int32, stride_src2_t: tl.int32, stride_src2_d: tl.int32,
    stride_dst_b: tl.int32, stride_dst_t: tl.int32, stride_dst_d: tl.int32,
    BLOCK_T: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_d = tl.program_id(2)

    T_total = L_txt + L_img
    t_offsets = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    d_offsets = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)

    t_mask = t_offsets < T_total
    d_mask = d_offsets < D
    mask = t_mask[:, None] & d_mask[None, :]

    # For each t in the tile, decide source
    for t_i in range(BLOCK_T):
        t = t_offsets[t_i]
        if t < L_txt:
            a_ptr = src1_ptr + pid_b * stride_src1_b + t * stride_src1_t + d_offsets[None, :] * stride_src1_d
        else:
            t_src = t - L_txt
            a_ptr = src2_ptr + pid_b * stride_src2_b + t_src * stride_src2_t + d_offsets[None, :] * stride_src2_d

        vals = tl.load(a_ptr, mask=mask[t_i], other=0.0)  # [BLOCK_D]
        dst_ptrs = dst_ptr + pid_b * stride_dst_b + t * stride_dst_t + d_offsets[None, :] * stride_dst_d
        tl.store(dst_ptrs, vals, mask=d_mask)


# Triton kernel: batched matmul for C[b, m, n] = sum_k A[b, m, k] * W[k, n]
# A is [B, M, K], W is [K, N], C is [B, M, N]
# Compute in fp32 for numerical stability.
@triton.jit
def batched_matmul_kernel(
    A_ptr, W_ptr, C_ptr,
    B: tl.int32, M: tl.int32, N: tl.int32, K: tl.int32,
    stride_Ab: tl.int32, stride_Am: tl.int32, stride_Ak: tl.int32,
    stride_Wk: tl.int32, stride_Wn: tl.int32,
    stride_Cb: tl.int32, stride_Cm: tl.int32, stride_Cn: tl.int32,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K in chunks
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)

        # Load A tile: [BLOCK_M, BLOCK_K]
        a_ptrs = A_ptr + pid_b * stride_Ab + m_offsets[:, None] * stride_Am + k_offsets[None, :] * stride_Ak
        a_mask = (m_offsets[:, None] < M) & (k_offsets[None, :] < K)
        A_tile = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # Load W tile: [BLOCK_K, BLOCK_N] where W has shape [K, N]
        w_ptrs = W_ptr + k_offsets[:, None] * stride_Wk + n_offsets[None, :] * stride_Wn
        w_mask = (k_offsets[:, None] < K) & (n_offsets[None, :] < N)
        W_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)

        # Accumulate
        acc += tl.dot(A_tile, W_tile)

    # Store result (fp32)
    c_ptrs = C_ptr + pid_b * stride_Cb + m_offsets[:, None] * stride_Cm + n_offsets[None, :] * stride_Cn
    c_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


# Triton kernel: copy processed[:, :L_txt, :] -> processed_encoder
@triton.jit
def copy_slice_kernel(
    src_ptr, dst_ptr,
    B: tl.int32, L: tl.int32, D: tl.int32,
    stride_src_b: tl.int32, stride_src_l: tl.int32, stride_src_d: tl.int32,
    stride_dst_b: tl.int32, stride_dst_l: tl.int32, stride_dst_d: tl.int32,
    BLOCK_L: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_l = tl.program_id(1)
    pid_d = tl.program_id(2)

    l_offsets = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
    d_offsets = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)

    l_mask = l_offsets < L
    d_mask = d_offsets < D
    mask = l_mask[:, None] & d_mask[None, :]

    src_ptrs = src_ptr + pid_b * stride_src_b + l_offsets[:, None] * stride_src_l + d_offsets[None, :] * stride_src_d
    dst_ptrs = dst_ptr + pid_b * stride_dst_b + l_offsets[:, None] * stride_dst_l + d_offsets[None, :] * stride_dst_d

    vals = tl.load(src_ptrs, mask=mask, other=0.0)
    tl.store(dst_ptrs, vals, mask=mask)


# Triton kernel: copy processed[:, L_txt:, :] -> processed_hidden
@triton.jit
def copy_slice_kernel2(
    src_ptr, dst_ptr,
    B: tl.int32, M: tl.int32, D: tl.int32, L_txt: tl.int32,
    stride_src_b: tl.int32, stride_src_l: tl.int32, stride_src_d: tl.int32,
    stride_dst_b: tl.int32, stride_dst_l: tl.int32, stride_dst_d: tl.int32,
    BLOCK_L: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_l = tl.program_id(1)
    pid_d = tl.program_id(2)

    l_offsets = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)  # indices in [L_txt, M)
    d_offsets = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)

    l_mask = l_offsets < (M - L_txt)
    d_mask = d_offsets < D
    mask = l_mask[:, None] & d_mask[None, :]

    src_ptrs = src_ptr + pid_b * stride_src_b + (l_offsets[:, None] + L_txt) * stride_src_l + d_offsets[None, :] * stride_src_d
    dst_ptrs = dst_ptr + pid_b * stride_dst_b + l_offsets[:, None] * stride_dst_l + d_offsets[None, :] * stride_dst_d

    vals = tl.load(src_ptrs, mask=mask, other=0.0)
    tl.store(dst_ptrs, vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-only forward:
        - Concatenate encoder_hidden_states and hidden_states along sequence dim (Triton).
        - Compute processed = concatenated @ process_weight.T using Triton batched matmul (fp32).
        - Split processed into encoder and image parts using Triton copy kernels.
        Returns: (processed_encoder [B, L_txt, D], processed_hidden [B, L_img, D])
        """
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "All inputs must be CUDA tensors for Triton."
        B = hidden_states.shape[0]
        L_txt = encoder_hidden_states.shape[1]
        L_img = hidden_states.shape[1]
        D = hidden_states.shape[2]

        # Ensure contiguous tensors for predictable strides
        ehs = encoder_hidden_states.contiguous()
        hs = hidden_states.contiguous()
        pw_T = process_weight.t().contiguous()  # [D, D], process_weight.T

        # 1) Concatenate sequences in Triton: dst [B, L_txt + L_img, D]
        total_seq = L_txt + L_img
        dst_concat = torch.empty((B, total_seq, D), device=hs.device, dtype=hs.dtype)

        BLOCK_T = 128
        BLOCK_D = 128
        grid_concat = (B, triton.cdiv(total_seq, BLOCK_T), triton.cdiv(D, BLOCK_D))
        concat_seqs_kernel[grid_concat](
            ehs, hs, dst_concat,
            B, L_txt, L_img, D,
            ehs.stride(0), ehs.stride(1), ehs.stride(2),
            hs.stride(0), hs.stride(1), hs.stride(2),
            dst_concat.stride(0), dst_concat.stride(1), dst_concat.stride(2),
            BLOCK_T=BLOCK_T, BLOCK_D=BLOCK_D,
        )

        # 2) Compute processed = dst_concat @ process_weight.T in fp32 using Triton matmul
        A = dst_concat
        W = pw_T  # [D, D]
        M = total_seq
        K = D
        N = D

        processed = torch.empty((B, M, N), device=A.device, dtype=torch.float32)  # fp32 accumulation/output

        # Choose tiles; these are conservative defaults that work across a range of sizes
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 32

        grid_matmul = (B, triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        batched_matmul_kernel[grid_matmul](
            A, W, processed,
            B, M, N, K,
            A.stride(0), A.stride(1), A.stride(2),
            W.stride(0), W.stride(1),
            processed.stride(0), processed.stride(1), processed.stride(2),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        )

        # 3) Split outputs using Triton copy kernels
        processed_encoder = torch.empty((B, L_txt, D), device=processed.device, dtype=processed.dtype)
        processed_hidden = torch.empty((B, L_img, D), device=processed.device, dtype=processed.dtype)

        BLOCK_L = 64
        BLOCK_D = 128

        # Copy processed[:, :L_txt, :] -> processed_encoder
        grid_copy1 = (B, triton.cdiv(L_txt, BLOCK_L), triton.cdiv(D, BLOCK_D))
        copy_slice_kernel[grid_copy1](
            processed, processed_encoder,
            B, L_txt, D,
            processed.stride(0), processed.stride(1), processed.stride(2),
            processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
            BLOCK_L=BLOCK_L, BLOCK_D=BLOCK_D,
        )

        # Copy processed[:, L_txt:, :] -> processed_hidden
        grid_copy2 = (B, triton.cdiv(L_img, BLOCK_L), triton.cdiv(D, BLOCK_D))
        copy_slice_kernel2[grid_copy2](
            processed, processed_hidden,
            B, M, D, L_txt,
            processed.stride(0), processed.stride(1), processed.stride(2),
            processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
            BLOCK_L=BLOCK_L, BLOCK_D=BLOCK_D,
        )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
