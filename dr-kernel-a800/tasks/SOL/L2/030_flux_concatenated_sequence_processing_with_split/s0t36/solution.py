import torch
import triton
import triton.language as tl

# Triton kernel: concatenate along sequence dimension
# Destination: dst[b, t, d] = src1[b, t, d] if t < L_txt else src2[b, t - L_txt, d]
@triton.jit
def concat_seq_kernel(
    src1, src2, dst,
    B, L_txt, L_img, D,
    src1_stride_b, src1_stride_s, src1_stride_d,
    src2_stride_b, src2_stride_s, src2_stride_d,
    dst_stride_b, dst_stride_s, dst_stride_d,
):
    pid_b = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_d = tl.program_id(2)

    b = pid_b
    t = pid_t
    d = pid_d

    # Only operate when t is within [0, L_txt + L_img)
    if (t < (L_txt + L_img)) and (d < D) and (b < B):
        if t < L_txt:
            val = tl.load(
                src1 + b * src1_stride_b + t * src1_stride_s + d * src1_stride_d
            )
        else:
            val = tl.load(
                src2 + b * src2_stride_b + (t - L_txt) * src2_stride_s + d * src2_stride_d
            )
        tl.store(dst + b * dst_stride_b + t * dst_stride_s + d * dst_stride_d, val)


# Triton kernel: split along sequence dimension back into two tensors
# From processed of shape [B, M, D], write:
# processed_encoder[b, t, d] = processed[b, t, d] for t in [0, L_txt)
# processed_hidden[b, t, d] = processed[b, t + L_txt, d] for t in [0, L_img)
@triton.jit
def split_seq_kernel(
    processed, processed_encoder, processed_hidden,
    B, L_txt, L_img, D,
    proc_stride_b, proc_stride_s, proc_stride_d,
    enc_stride_b, enc_stride_s, enc_stride_d,
    hid_stride_b, hid_stride_s, hid_stride_d,
):
    pid_b = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_d = tl.program_id(2)

    b = pid_b
    t = pid_t
    d = pid_d

    if (b < B) and (t < L_txt) and (d < D):
        val = tl.load(
            processed + b * proc_stride_b + t * proc_stride_s + d * proc_stride_d
        )
        tl.store(
            processed_encoder + b * enc_stride_b + t * enc_stride_s + d * enc_stride_d,
            val
        )
    if (b < B) and (t < L_img) and (d < D):
        val = tl.load(
            processed + b * proc_stride_b + (t + L_txt) * proc_stride_s + d * proc_stride_d
        )
        tl.store(
            processed_hidden + b * hid_stride_b + t * hid_stride_s + d * hid_stride_d,
            val
        )


# Triton kernel: perform GEMM for concatenated A [B, M, K] and W [K, N] to produce C [B, M, N]
# Here K = concatenated feature dim (D), N = process_weight dim (D), M = L_txt + L_img.
@triton.jit
def triton_gemm_cat_linear_kernel(
    A_ptr, W_ptr, C_ptr,
    B, M, K, N,
    A_stride_b, A_stride_m, A_stride_k,
    W_stride_k, W_stride_n,
    C_stride_b, C_stride_m, C_stride_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Program ids: grid = (B, ceil_div(M, BLOCK_M), ceil_div(N, BLOCK_N))
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    b = pid_b
    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N

    m_offsets = m_start + tl.arange(0, BLOCK_M)  # vector of M indices for this tile
    n_offsets = n_start + tl.arange(0, BLOCK_N)  # vector of N indices for this tile

    # Accumulator for this tile
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension in blocks
    k_start = 0
    while k_start < K:
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        # Masks for partial tiles
        mask_m = m_offsets < M
        mask_n = n_offsets < N
        mask_k = k_offsets < K

        # Load A tile: A[b, m_offsets, k_offsets] -> shape (BLOCK_M, BLOCK_K)
        a_ptrs = A_ptr + b * A_stride_b + m_offsets[:, None] * A_stride_m + k_offsets[None, :] * A_stride_k
        a_vals = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0).to(tl.float32)

        # Load W tile: W[k_offsets, n_offsets] -> shape (BLOCK_K, BLOCK_N)
        w_ptrs = W_ptr + k_offsets[:, None] * W_stride_k + n_offsets[None, :] * W_stride_n
        w_vals = tl.load(w_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0).to(tl.float32)

        # Fused multiply-add: acc += a_vals @ w_vals
        # Triton supports elementwise operations; we reduce over axis=1
        # Note: a_vals shape (BM,BK), w_vals shape (BK,BN) -> (BM,BN)
        acc += tl.sum(a_vals[:, :, None] * w_vals[None, :, :], axis=1)

        k_start += BLOCK_K

    # Store the result tile
    c_ptrs = C_ptr + b * C_stride_b + m_offsets[:, None] * C_stride_m + n_offsets[None, :] * C_stride_n
    mask_m_n = mask_m[:, None] & mask_n[None, :]
    tl.store(c_ptrs, acc, mask=mask_m_n)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        Triton-optimized version:
        - Concatenate encoder_hidden_states and hidden_states along sequence dim (Triton).
        - Apply linear projection using Triton GEMM (matmul).
        - Split back into separate streams (Triton).
        Returns (processed_encoder, processed_hidden) with shapes [B, text_seq_len, D] and [B, img_seq_len, D].
        """
        assert hidden_states.dim() == 3 and encoder_hidden_states.dim() == 3, "Inputs must be 3D tensors"
        B = hidden_states.shape[0]
        L_img = hidden_states.shape[1]
        L_txt = encoder_hidden_states.shape[1]
        D = hidden_states.shape[2]
        M = L_txt + L_img

        device = hidden_states.device
        assert device.type == 'cuda', "Triton kernels require CUDA tensors"

        # Concatenate along sequence dimension (Triton)
        concatenated = torch.empty((B, M, D), dtype=hidden_states.dtype, device=device)

        grid_concat = (B, M, D)
        concat_seq_kernel[grid_concat](
            encoder_hidden_states, hidden_states, concatenated,
            B, L_txt, L_img, D,
            encoder_hidden_states.stride(0), encoder_hidden_states.stride(1), encoder_hidden_states.stride(2),
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            concatenated.stride(0), concatenated.stride(1), concatenated.stride(2),
            num_warps=1, num_stages=1
        )

        # Prepare weight W as [D, D]; index as W[k, n] directly
        W = process_weight  # [D, D]

        # Allocate output C for GEMM, compute in float32 for robustness
        C = torch.empty((B, M, D), dtype=torch.float32, device=device)

        # Launch Triton GEMM: grid over (B, tiles of M, tiles of N)
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 64
        grid_gemm = (B, triton.cdiv(M, BLOCK_M), triton.cdiv(D, BLOCK_N))
        triton_gemm_cat_linear_kernel[grid_gemm](
            concatenated, W, C,
            B, M, D, D,  # K=D, N=D
            concatenated.stride(0), concatenated.stride(1), concatenated.stride(2),
            W.stride(0), W.stride(1),  # W is [D, D]
            C.stride(0), C.stride(1), C.stride(2),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2
        )

        # Split outputs (Triton)
        processed_encoder = torch.empty((B, L_txt, D), dtype=C.dtype, device=device)
        processed_hidden = torch.empty((B, L_img, D), dtype=C.dtype, device=device)

        grid_split = (B, L_txt, D)
        split_seq_kernel[grid_split](
            C, processed_encoder, processed_hidden,
            B, L_txt, L_img, D,
            C.stride(0), C.stride(1), C.stride(2),
            processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
            processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
            num_warps=1, num_stages=1
        )

        # Cast outputs to match input dtype (original tensors are typically float32)
        processed_encoder = processed_encoder.to(hidden_states.dtype)
        processed_hidden = processed_hidden.to(hidden_states.dtype)

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
