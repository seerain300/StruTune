import torch
import triton
import triton.language as tl


@triton.jit
def concat_seqs_kernel(
    ehs_ptr,          # *encoder_hidden_states [B, L_txt, D]
    hs_ptr,           # *hidden_states [B, L_img, D]
    out_ptr,          # *output concatenated [B, L_txt + L_img, D]
    B: tl.int32,
    L_txt: tl.int32,
    L_img: tl.int32,
    D: tl.int32,
    ehs_stride_b: tl.int32, ehs_stride_s: tl.int32, ehs_stride_d: tl.int32,
    hs_stride_b: tl.int32, hs_stride_s: tl.int32, hs_stride_d: tl.int32,
    out_stride_b: tl.int32, out_stride_s: tl.int32, out_stride_d: tl.int32,
    BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)  # s in [0, L_txt + L_img)
    pid_d = tl.program_id(2)

    d_offsets = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = d_offsets < D

    ehs_b_ptr = ehs_ptr + pid_b * ehs_stride_b
    hs_b_ptr = hs_ptr + pid_b * hs_stride_b
    out_b_ptr = out_ptr + pid_b * out_stride_b

    # If s < L_txt: copy from encoder_hidden_states; else: copy from hidden_states (s - L_txt index)
    copy_encoder = pid_s < L_txt
    if copy_encoder:
        vals = tl.load(ehs_b_ptr + pid_s * ehs_stride_s + d_offsets * ehs_stride_d, mask=mask_d, other=0.0)
        tl.store(out_b_ptr + pid_s * out_stride_s + d_offsets * out_stride_d, vals, mask=mask_d)
    else:
        vals = tl.load(hs_b_ptr + (pid_s - L_txt) * hs_stride_s + d_offsets * hs_stride_d, mask=mask_d, other=0.0)
        tl.store(out_b_ptr + pid_s * out_stride_s + d_offsets * out_stride_d, vals, mask=mask_d)


@triton.jit
def batched_matmul_kernel(
    a_ptr,             # *A [B, M, K], A is the concatenated input
    w_ptr,             # *W [K, N] where W = process_weight.T
    c_ptr,             # *C [B, M, N] output
    B: tl.int32,
    M: tl.int32,
    K: tl.int32,
    N: tl.int32,
    a_stride_b: tl.int32, a_stride_m: tl.int32, a_stride_k: tl.int32,
    w_stride_k: tl.int32, w_stride_n: tl.int32,
    c_stride_b: tl.int32, c_stride_m: tl.int32, c_stride_n: tl.int32,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)  # tile along M
    pid_n = tl.program_id(2)  # tile along N

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    k_offsets = tl.arange(0, BLOCK_K)

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K
    for k_start in range(0, K, BLOCK_K):
        k_idx = k_start + k_offsets

        # Load A tile: [BLOCK_M, BLOCK_K], A[b, m, k]
        a_ptrs = a_ptr + pid_b * a_stride_b + m_offsets[:, None] * a_stride_m + k_idx[None, :] * a_stride_k
        a_mask = (m_offsets[:, None] < M) & (k_idx[None, :] < K)
        a_tile = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # Load W^T tile: [BLOCK_K, BLOCK_N], W[k, n]
        w_ptrs = w_ptr + k_idx[:, None] * w_stride_k + n_offsets[None, :] * w_stride_n
        w_mask = (k_idx[:, None] < K) & (n_offsets[None, :] < N)
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)

        # Fused multiply-add
        acc += tl.dot(a_tile, w_tile)

    # Store C tile: [BLOCK_M, BLOCK_N], C[b, m, n]
    c_ptrs = c_ptr + pid_b * c_stride_b + m_offsets[:, None] * c_stride_m + n_offsets[None, :] * c_stride_n
    c_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


@triton.jit
def split_seqs_kernel(
    c_ptr,             # *C [B, M, D] = processed concatenated
    out1_ptr,          # *processed_encoder [B, L_txt, D]
    out2_ptr,          # *processed_hidden [B, L_img, D]
    B: tl.int32,
    L_txt: tl.int32,
    L_img: tl.int32,
    D: tl.int32,
    c_stride_b: tl.int32, c_stride_m: tl.int32, c_stride_d: tl.int32,
    out1_stride_b: tl.int32, out1_stride_s: tl.int32, out1_stride_d: tl.int32,
    out2_stride_b: tl.int32, out2_stride_s: tl.int32, out2_stride_d: tl.int32,
    BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)  # s in [0, L_txt) or [L_txt, L_txt + L_img)
    pid_d = tl.program_id(2)

    d_offsets = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = d_offsets < D

    c_b_ptr = c_ptr + pid_b * c_stride_b

    # Encoder part
    if pid_s < L_txt:
        vals = tl.load(c_b_ptr + pid_s * c_stride_m + d_offsets * c_stride_d, mask=mask_d, other=0.0)
        tl.store(out1_b_ptr + pid_s * out1_stride_s + d_offsets * out1_stride_d, vals, mask=mask_d)
    # Hidden part
    else:
        s_offset = pid_s - L_txt
        vals = tl.load(c_b_ptr + s_offset * c_stride_m + d_offsets * c_stride_d, mask=mask_d, other=0.0)
        tl.store(out2_b_ptr + (pid_s - L_txt) * out2_stride_s + d_offsets * out2_stride_d, vals, mask=mask_d)


@triton.jit
def transpose_weight_kernel(
    w_ptr,             # *process_weight [D, D]
    wt_ptr,            # *process_weight.T [D, D]
    D: tl.int32,
    w_stride_0: tl.int32, w_stride_1: tl.int32,
    wt_stride_0: tl.int32, wt_stride_1: tl.int32,
    BLOCK: tl.constexpr,
):
    # Simple 2D grid over D x D, storing w[i, j] to wt[j, i]
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)

    i = pid_i
    j = pid_j

    # Load w[i, j]
    val = tl.load(w_ptr + i * w_stride_0 + j * w_stride_1)
    # Store to wt[j, i]
    tl.store(wt_ptr + j * wt_stride_0 + i * wt_stride_1, val)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized forward:
        - Concatenate along sequence dim using Triton.
        - Apply linear projection via Triton batched matmul.
        - Split outputs back using Triton.
        """
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Inputs must be on CUDA device for Triton."

        B = hidden_states.shape[0]
        L_img = hidden_states.shape[1]
        L_txt = encoder_hidden_states.shape[1]
        D = hidden_states.shape[2]
        assert encoder_hidden_states.shape[0] == B and encoder_hidden_states.shape[2] == D
        assert process_weight.shape == (D, D)

        # 1) Concatenate [B, L_txt + L_img, D] using Triton
        M = L_txt + L_img
        concatenated = torch.empty((B, M, D), dtype=hidden_states.dtype, device=hidden_states.device)

        # Launch concat kernel
        BLOCK_D = 128  # handle D up to 128 in one tile; masks handle >128
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

        # 2) Compute W_t = process_weight.T in Triton (D, D) -> (D, D)
        W_t = torch.empty((D, D), dtype=process_weight.dtype, device=process_weight.device)
        grid_transpose = (D, D)
        transpose_weight_kernel[grid_transpose](
            process_weight, W_t,
            D,
            process_weight.stride(0), process_weight.stride(1),
            W_t.stride(0), W_t.stride(1),
            BLOCK=D,
            num_warps=4, num_stages=2,
        )

        # 3) Matmul: C = concatenated @ W_t  -> [B, M, D]
        C = torch.empty((B, M, D), dtype=torch.float32, device=hidden_states.device)  # accumulate in fp32

        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 64

        grid_gemm = (B, triton.cdiv(M, BLOCK_M), triton.cdiv(D, BLOCK_N))
        batched_matmul_kernel[grid_gemm](
            concatenated, W_t, C,
            B, M, D, D,
            concatenated.stride(0), concatenated.stride(1), concatenated.stride(2),
            W_t.stride(0), W_t.stride(1),
            C.stride(0), C.stride(1), C.stride(2),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        # 4) Split into encoder and hidden streams using Triton
        processed_encoder = torch.empty((B, L_txt, D), dtype=hidden_states.dtype, device=hidden_states.device)
        processed_hidden = torch.empty((B, L_img, D), dtype=hidden_states.dtype, device=hidden_states.device)

        grid_split = (B, L_txt + L_img, triton.cdiv(D, BLOCK_D))
        split_seqs_kernel[grid_split](
            C,
            processed_encoder, processed_hidden,
            B, L_txt, L_img, D,
            C.stride(0), C.stride(1), C.stride(2),
            processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
            processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
            BLOCK_D=BLOCK_D,
            num_warps=4, num_stages=2,
        )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
