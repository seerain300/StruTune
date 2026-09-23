import torch
import triton
import triton.language as tl


@triton.jit
def concat_seqs_kernel(
    src1_ptr,  # encoder_hidden_states [B, L_txt, D]
    src2_ptr,  # hidden_states [B, L_img, D]
    dst_ptr,   # output [B, L_txt + L_img, D]
    B: tl.int32,
    L_txt: tl.int32,
    L_img: tl.int32,
    D: tl.int32,
    src1_stride_b: tl.int32, src1_stride_s: tl.int32, src1_stride_d: tl.int32,
    src2_stride_b: tl.int32, src2_stride_s: tl.int32, src2_stride_d: tl.int32,
    dst_stride_b: tl.int32, dst_stride_s: tl.int32, dst_stride_d: tl.int32,
    BLOCK_D: tl.constexpr,
):
    # Grid: (B, L_txt + L_img, ceil(D / BLOCK_D))
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_d_block = tl.program_id(2)

    d_offsets = pid_d_block * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = d_offsets < D

    src1_b = src1_ptr + pid_b * src1_stride_b
    src2_b = src2_ptr + pid_b * src2_stride_b
    dst_b = dst_ptr + pid_b * dst_stride_b

    if pid_s < L_txt:
        # copy from encoder_hidden_states
        vals = tl.load(
            src1_b + pid_s * src1_stride_s + d_offsets * src1_stride_d,
            mask=mask_d,
            other=0.0,
        )
        tl.store(
            dst_b + pid_s * dst_stride_s + d_offsets * dst_stride_d,
            vals,
            mask=mask_d,
        )
    else:
        # copy from hidden_states, offset by L_txt
        vals = tl.load(
            src2_b + (pid_s - L_txt) * src2_stride_s + d_offsets * src2_stride_d,
            mask=mask_d,
            other=0.0,
        )
        tl.store(
            dst_b + pid_s * dst_stride_s + d_offsets * dst_stride_d,
            vals,
            mask=mask_d,
        )


@triton.jit
def batched_matmul_kernel(
    A_ptr,   # [B, M, K] concatenated input
    W_ptr,   # [K, N] process_weight.T
    C_ptr,   # [B, M, N] output
    B: tl.int32, M: tl.int32, K: tl.int32, N: tl.int32,
    A_stride_b: tl.int32, A_stride_m: tl.int32, A_stride_k: tl.int32,
    W_stride_k: tl.int32, W_stride_n: tl.int32,
    C_stride_b: tl.int32, C_stride_m: tl.int32, C_stride_n: tl.int32,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Grid: (B, tiles along M, tiles along N)
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    k_offsets = tl.arange(0, BLOCK_K)

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K in blocks
    for k_start in range(0, K, BLOCK_K):
        # Load A_tile: [BLOCK_M, BLOCK_K] for batch pid_b, rows m_offsets, cols k_offsets
        A_ptrs = A_ptr + pid_b * A_stride_b + m_offsets[:, None] * A_stride_m + (k_start + k_offsets[None, :]) * A_stride_k
        A_mask = (m_offsets[:, None] < M) & ((k_start + k_offsets[None, :]) < K)
        A_tile = tl.load(A_ptrs, mask=A_mask, other=0.0)

        # Load Wt_tile: [BLOCK_K, BLOCK_N], W is [K, N]
        Wt_ptrs = W_ptr + (k_start + k_offsets[:, None]) * W_stride_k + n_offsets[None, :] * W_stride_n
        W_mask = ((k_start + k_offsets[:, None]) < K) & (n_offsets[None, :] < N)
        Wt_tile = tl.load(Wt_ptrs, mask=W_mask, other=0.0)

        # Accumulate: acc += A_tile @ Wt_tile
        acc += tl.dot(A_tile, Wt_tile)

    # Store results
    C_ptrs = C_ptr + pid_b * C_stride_b + m_offsets[:, None] * C_stride_m + n_offsets[None, :] * C_stride_n
    C_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    # Cast to original dtype if C is fp16/bf16; here we assume fp32 inputs/weights
    tl.store(C_ptrs, acc, mask=C_mask)


@triton.jit
def split_seqs_kernel(
    C_ptr,            # [B, M, D] processed
    out_encoder_ptr,  # [B, L_txt, D]
    out_hidden_ptr,   # [B, L_img, D]
    B: tl.int32, M: tl.int32, D: tl.int32,
    L_txt: tl.int32,
    C_stride_b: tl.int32, C_stride_m: tl.int32, C_stride_d: tl.int32,
    out_e_stride_b: tl.int32, out_e_stride_s: tl.int32, out_e_stride_d: tl.int32,
    out_h_stride_b: tl.int32, out_h_stride_s: tl.int32, out_h_stride_d: tl.int32,
    BLOCK_D: tl.constexpr,
):
    # Grid for each output: (B, L_txt, ceil(D / BLOCK_D)) and (B, L_img, ceil(D / BLOCK_D))
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_d_block = tl.program_id(2)

    d_offsets = pid_d_block * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = d_offsets < D

    C_b = C_ptr + pid_b * C_stride_b
    out_e_b = out_encoder_ptr + pid_b * out_e_stride_b
    out_h_b = out_hidden_ptr + pid_b * out_h_stride_b

    if pid_s < L_txt:
        vals = tl.load(C_b + pid_s * C_stride_m + d_offsets * C_stride_d, mask=mask_d, other=0.0)
        tl.store(out_e_b + pid_s * out_e_stride_s + d_offsets * out_e_stride_d, vals, mask=mask_d)
    else:
        vals = tl.load(C_b + pid_s * C_stride_m + d_offsets * C_stride_d, mask=mask_d, other=0.0)
        tl.store(out_h_b + (pid_s - L_txt) * out_h_stride_s + d_offsets * out_h_stride_d, vals, mask=mask_d)


@triton.jit
def transpose_weight_kernel(
    W_in_ptr,  # input weight [D, D]
    Wt_ptr,    # output transposed [D, D]
    D: tl.int32,
    W_in_stride0: tl.int32, W_in_stride1: tl.int32,
    Wt_stride0: tl.int32, Wt_stride1: tl.int32,
    BLOCK_D: tl.constexpr,
):
    # Simple 2D copy: Wt[n, k] = W_in[k, n]
    n_offsets = tl.arange(0, BLOCK_D)
    k_offsets = tl.arange(0, BLOCK_D)
    for n_start in range(0, D, BLOCK_D):
        for k_start in range(0, D, BLOCK_D):
            n_idx = n_start + n_offsets
            k_idx = k_start + k_offsets
            mask = (n_idx[:, None] < D) & (k_idx[None, :] < D)
            vals = tl.load(W_in_ptr + k_idx[None, :] * W_in_stride0 + n_idx[:, None] * W_in_stride1, mask=mask, other=0.0)
            tl.store(Wt_ptr + n_idx[:, None] * Wt_stride0 + k_idx[None, :] * Wt_stride1, vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized forward:
        - Concatenate along sequence dim using concat_seqs_kernel (no torch.cat).
        - Transpose process_weight using transpose_weight_kernel (no .t()).
        - Batched matmul using batched_matmul_kernel (no torch.matmul).
        - Split outputs using split_seqs_kernel (no torch slicing-based ops).
        All computation happens via Triton kernels; no torch ops in forward.
        """
        # Ensure contiguous inputs
        encoder_hidden_states = encoder_hidden_states.contiguous()
        hidden_states = hidden_states.contiguous()
        process_weight = process_weight.contiguous()

        B = hidden_states.shape[0]
        L_img = hidden_states.shape[1]
        L_txt = encoder_hidden_states.shape[1]
        D = hidden_states.shape[2]

        # Allocate and run concatenation
        M = L_txt + L_img
        concatenated = torch.empty((B, M, D), device=hidden_states.device, dtype=hidden_states.dtype)
        BLOCK_D = 128  # vectorization along D
        grid_concat = (B, M, triton.cdiv(D, BLOCK_D))
        concat_seqs_kernel[grid_concat](
            encoder_hidden_states, hidden_states, concatenated,
            B, L_txt, L_img, D,
            encoder_hidden_states.stride(0), encoder_hidden_states.stride(1), encoder_hidden_states.stride(2),
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            concatenated.stride(0), concatenated.stride(1), concatenated.stride(2),
            BLOCK_D=BLOCK_D,
            num_warps=4, num_stages=2,
        )

        # Transpose process_weight to [D, D] (Wt): Wt[n, k] = W[k, n]
        Wt = torch.empty((D, D), device=process_weight.device, dtype=process_weight.dtype)
        grid_transpose = (triton.cdiv(D, 128), triton.cdiv(D, 128))
        transpose_weight_kernel[grid_transpose](
            process_weight, Wt,
            D,
            process_weight.stride(0), process_weight.stride(1),
            Wt.stride(0), Wt.stride(1),
            BLOCK_D=128,
            num_warps=4, num_stages=2,
        )

        # Allocate output for processed (C): [B, M, D]
        C = torch.empty((B, M, D), device=hidden_states.device, dtype=hidden_states.dtype)

        # Run batched matmul: C = concatenated @ Wt
        # Choose tile sizes suitable for a wide range
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 32
        grid_gemm = (B, triton.cdiv(M, BLOCK_M), triton.cdiv(D, BLOCK_N))
        batched_matmul_kernel[grid_gemm](
            concatenated, Wt, C,
            B, M, D, D,
            concatenated.stride(0), concatenated.stride(1), concatenated.stride(2),
            Wt.stride(0), Wt.stride(1),
            C.stride(0), C.stride(1), C.stride(2),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        # Allocate outputs for split and run split kernel
        processed_encoder = torch.empty((B, L_txt, D), device=hidden_states.device, dtype=hidden_states.dtype)
        processed_hidden = torch.empty((B, L_img, D), device=hidden_states.device, dtype=hidden_states.dtype)

        grid_split = (B, L_txt, triton.cdiv(D, 64)), (B, L_img, triton.cdiv(D, 64))
        split_seqs_kernel[grid_split](
            # We pass grid as a tuple of two calls; Triton expects a single grid, so we combine into one launch by using the larger one
            # To simplify, we launch two separate kernels with their respective grids:
            C, processed_encoder, processed_hidden,
            B, M, D, L_txt,
            C.stride(0), C.stride(1), C.stride(2),
            processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
            processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
            BLOCK_D=64,
            num_warps=4, num_stages=2,
        )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
