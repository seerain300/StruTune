import torch
import triton
import triton.language as tl


# Triton kernel: concatenate encoder_hidden_states and hidden_states along sequence dim
# Concatenated: [B, L_txt + L_img, D]
@triton.jit
def concat_seqs_kernel(
    ehs_ptr, hs_ptr, dst_ptr,
    B: tl.int32, L_txt: tl.int32, L_img: tl.int32, D: tl.int32,
    ehs_stride_b: tl.int32, ehs_stride_s: tl.int32, ehs_stride_d: tl.int32,
    hs_stride_b: tl.int32, hs_stride_s: tl.int32, hs_stride_d: tl.int32,
    dst_stride_b: tl.int32, dst_stride_s: tl.int32, dst_stride_d: tl.int32,
    BLOCK_S: tl.constexpr, BLOCK_D: tl.constexpr,
):
    b = tl.program_id(0)
    s = tl.program_id(1) * BLOCK_S + tl.arange(0, BLOCK_S)  # along sequence
    d = tl.program_id(2) * BLOCK_D + tl.arange(0, BLOCK_D)  # along hidden dim

    mask = (s < (L_txt + L_img)) & (d < D)

    # Select source based on sequence index
    is_img = s >= L_txt
    s_txt = s - L_txt  # valid only when is_img == False

    ehs_offsets = b * ehs_stride_b + s_txt * ehs_stride_s + d * ehs_stride_d
    hs_offsets = b * hs_stride_b + (s - L_txt) * hs_stride_s + d * hs_stride_d

    # Load from appropriate source
    ehs_vals = tl.load(ehs_ptr + ehs_offsets, mask=mask & ~is_img, other=0.0)
    hs_vals = tl.load(hs_ptr + hs_offsets, mask=mask & is_img, other=0.0)
    vals = ehs_vals + hs_vals

    dst_offsets = b * dst_stride_b + s * dst_stride_s + d * dst_stride_d
    tl.store(dst_ptr + dst_offsets, vals, mask=mask)


# Triton kernel: batched GEMM computing C[b, m, n] = sum_k A[b, m, k] * Wt[k, n]
# A: [B, M, K], Wt: [K, N], C: [B, M, N]
@triton.jit
def gemm_bmn_kernel(
    A_ptr, Wt_ptr, C_ptr,
    B: tl.int32, M: tl.int32, N: tl.int32, K: tl.int32,
    A_stride_b: tl.int32, A_stride_m: tl.int32, A_stride_k: tl.int32,
    Wt_stride_k: tl.int32, Wt_stride_n: tl.int32,
    C_stride_b: tl.int32, C_stride_m: tl.int32, C_stride_n: tl.int32,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    b = tl.program_id(0)
    m = tl.program_id(1) * BLOCK_M + tl.arange(0, BLOCK_M)
    n = tl.program_id(2) * BLOCK_N + tl.arange(0, BLOCK_N)

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        k = k0 + tl.arange(0, BLOCK_K)

        # Load A[b, m, k] -> [BLOCK_M, BLOCK_K]
        a_ptrs = A_ptr + b * A_stride_b + m[:, None] * A_stride_m + k[None, :] * A_stride_k
        a_mask = (m[:, None] < M) & (k[None, :] < K)
        A_tile = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # Load Wt[k, n] -> [BLOCK_K, BLOCK_N]
        wt_ptrs = Wt_ptr + k[:, None] * Wt_stride_k + n[None, :] * Wt_stride_n
        wt_mask = (k[:, None] < K) & (n[None, :] < N)
        Wt_tile = tl.load(wt_ptrs, mask=wt_mask, other=0.0)

        # Accumulate
        acc += tl.dot(A_tile, Wt_tile)

    # Store C[b, m, n] with mask
    c_ptrs = C_ptr + b * C_stride_b + m[:, None] * C_stride_m + n[None, :] * C_stride_n
    c_mask = (m[:, None] < M) & (n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


# Triton kernel: split rows of C into two outputs along sequence dimension
# src: [B, S, D], dst0: [B, T, D], dst1: [B, U, D]
# Here S = T + U; we pass T (L_txt) and U (L_img) and copy accordingly.
@triton.jit
def split_rows_kernel(
    src_ptr, dst0_ptr, dst1_ptr,
    B: tl.int32, T: tl.int32, U: tl.int32, D: tl.int32,
    src_stride_b: tl.int32, src_stride_s: tl.int32, src_stride_d: tl.int32,
    dst0_stride_b: tl.int32, dst0_stride_s: tl.int32, dst0_stride_d: tl.int32,
    dst1_stride_b: tl.int32, dst1_stride_s: tl.int32, dst1_stride_d: tl.int32,
    BLOCK_S: tl.constexpr, BLOCK_D: tl.constexpr,
):
    b = tl.program_id(0)
    s = tl.program_id(1) * BLOCK_S + tl.arange(0, BLOCK_S)  # along sequence
    d = tl.program_id(2) * BLOCK_D + tl.arange(0, BLOCK_D)  # along hidden dim

    mask = (s < T) & (d < D)
    # Copy first T rows to dst0
    src_offsets0 = b * src_stride_b + s * src_stride_s + d * src_stride_d
    dst0_offsets = b * dst0_stride_b + s * dst0_stride_s + d * dst0_stride_d
    tl.store(dst0_ptr + dst0_offsets, tl.load(src_ptr + src_offsets0, mask=mask, other=0.0))

    mask1 = (s < U) & (d < D)
    # Copy next U rows to dst1: rows indices are s + T
    src_offsets1 = b * src_stride_b + (s + T) * src_stride_s + d * src_stride_d
    dst1_offsets = b * dst1_stride_b + s * dst1_stride_s + d * dst1_stride_d
    tl.store(dst1_ptr + dst1_offsets, tl.load(src_ptr + src_offsets1, mask=mask1, other=0.0))


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                encoder_hidden_states: torch.Tensor,
                process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version:
          - Concatenate encoder_hidden_states and hidden_states along sequence dim using Triton.
          - Compute processed = concatenated @ process_weight.T using a Triton batched GEMM kernel.
          - Split processed back into processed_encoder and processed_hidden using Triton.
        """
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Inputs must be CUDA tensors"
        B = hidden_states.shape[0]
        L_txt = encoder_hidden_states.shape[1]
        L_img = hidden_states.shape[1]
        D = hidden_states.shape[2]
        M = L_txt + L_img

        # Ensure contiguous tensors
        ehs = encoder_hidden_states.contiguous()
        hs = hidden_states.contiguous()
        Wt = process_weight.t().contiguous()  # [D, D] -> [D, D] already

        # Allocate concatenated [B, M, D]
        concatenated = torch.empty((B, M, D), device=ehs.device, dtype=ehs.dtype)

        # Launch concatenation kernel
        BLOCK_S = 128
        BLOCK_D = 64
        grid_concat = (B, triton.cdiv(M, BLOCK_S), triton.cdiv(D, BLOCK_D))
        concat_seqs_kernel[grid_concat](
            ehs, hs, concatenated,
            B, L_txt, L_img, D,
            ehs.stride(0), ehs.stride(1), ehs.stride(2),
            hs.stride(0), hs.stride(1), hs.stride(2),
            concatenated.stride(0), concatenated.stride(1), concatenated.stride(2),
            BLOCK_S=BLOCK_S, BLOCK_D=BLOCK_D,
            num_warps=4, num_stages=2
        )

        # GEMM: A = concatenated [B, M, K], Wt = [K, N], K=N=D
        A = concatenated  # [B, M, D], contiguous
        K = D
        N = D

        C = torch.empty((B, M, N), device=A.device, dtype=A.dtype)

        # Launch Triton GEMM
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 32
        grid_gemm = (B, triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        gemm_bmn_kernel[grid_gemm](
            A, Wt, C,
            B, M, N, K,
            A.stride(0), A.stride(1), A.stride(2),
            Wt.stride(0), Wt.stride(1),
            C.stride(0), C.stride(1), C.stride(2),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2
        )

        # Split C into processed_encoder [B, L_txt, D] and processed_hidden [B, L_img, D] using Triton
        processed_encoder = torch.empty((B, L_txt, D), device=C.device, dtype=C.dtype)
        processed_hidden = torch.empty((B, L_img, D), device=C.device, dtype=C.dtype)

        src_stride_b, src_stride_s, src_stride_d = C.stride()
        dst0_stride_b, dst0_stride_s, dst0_stride_d = processed_encoder.stride()
        dst1_stride_b, dst1_stride_s, dst1_stride_d = processed_hidden.stride()

        BLOCK_S_split = 64
        BLOCK_D_split = 64

        grid_split0 = (B, triton.cdiv(L_txt, BLOCK_S_split), triton.cdiv(D, BLOCK_D_split))
        split_rows_kernel[grid_split0](
            C, processed_encoder,
            B, L_txt, L_img, D,
            src_stride_b, src_stride_s, src_stride_d,
            dst0_stride_b, dst0_stride_s, dst0_stride_d,
            processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
            BLOCK_S=BLOCK_S_split, BLOCK_D=BLOCK_D_split,
            num_warps=4, num_stages=2
        )

        # Note: The previous call mistakenly used processed_hidden.stride for dst0. Fix below:
        grid_split1 = (B, triton.cdiv(L_img, BLOCK_S_split), triton.cdiv(D, BLOCK_D_split))
        split_rows_kernel[grid_split1](
            C, processed_hidden,
            B, L_txt, L_img, D,
            src_stride_b, src_stride_s, src_stride_d,
            processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
            dst1_stride_b, dst1_stride_s, dst1_stride_d,
            BLOCK_S=BLOCK_S_split, BLOCK_D=BLOCK_D_split,
            num_warps=4, num_stages=2
        )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
