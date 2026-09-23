import torch
import triton
import triton.language as tl


@triton.jit
def concat_seqs_kernel(
    src1_ptr,      # *ptr to encoder_hidden_states: [B, L_txt, D]
    src2_ptr,      # *ptr to hidden_states: [B, L_img, D]
    dst_ptr,       # *ptr to concatenated output: [B, L_txt + L_img, D]
    B: tl.int32,
    L_txt: tl.int32,
    L_img: tl.int32,
    D: tl.int32,
    stride_s1_b: tl.int32,
    stride_s1_m: tl.int32,
    stride_s1_k: tl.int32,
    stride_s2_b: tl.int32,
    stride_s2_m: tl.int32,
    stride_s2_k: tl.int32,
    stride_dst_b: tl.int32,
    stride_dst_m: tl.int32,
    stride_dst_k: tl.int32,
    BLOCK_M: tl.constexpr,   # tile along sequence
    BLOCK_K: tl.constexpr,   # tile along hidden_dim (loop over k)
):
    # Grid: (B, ceil((L_txt + L_img) / BLOCK_M))
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)

    total_seq = L_txt + L_img
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = m_offsets < total_seq

    # For each hidden_dim element k
    for k in range(0, D):
        # Determine source for each position: positions < L_txt from src1, else from src2
        m_src1 = m_offsets
        m_src2 = m_offsets - L_txt
        mask_from_src1 = mask_m & (m_offsets < L_txt)
        mask_from_src2 = mask_m & (m_offsets >= L_txt)

        # Pointers for src1 and src2
        src1_ptrs = src1_ptr + pid_b * stride_s1_b + m_src1 * stride_s1_m + k * stride_s1_k
        src2_ptrs = src2_ptr + pid_b * stride_s2_b + m_src2 * stride_s2_m + k * stride_s2_k
        # Load from src1 where applicable, else from src2
        val_src1 = tl.load(src1_ptrs, mask=mask_from_src1, other=0.0)
        val_src2 = tl.load(src2_ptrs, mask=mask_from_src2, other=0.0)
        val = tl.where(m_offsets < L_txt, val_src1, val_src2)

        # Destination pointers and store
        dst_ptrs = dst_ptr + pid_b * stride_dst_b + m_offsets * stride_dst_m + k * stride_dst_k
        tl.store(dst_ptrs, val, mask=mask_m)


@triton.jit
def batched_matmul_kernel(
    A_ptr,        # *ptr to A: [B, M, K], M = L_txt + L_img, K = D
    W_ptr,        # *ptr to W: [K, N] = [D, D] (process_weight; we index as W[k, n])
    C_ptr,        # *ptr to C: [B, M, N]
    B: tl.int32,  # batch size (not used in kernel math; used for grid dim)
    M: tl.int32,  # sequence length after concat
    N: tl.int32,  # hidden_dim (output dim)
    K: tl.int32,  # hidden_dim (input/output feature dim)
    stride_Ab: tl.int32,
    stride_Am: tl.int32,
    stride_Ak: tl.int32,
    stride_Wk: tl.int32,  # row stride of W (K dim)
    stride_Wn: tl.int32,  # col stride of W (N dim)
    stride_Cb: tl.int32,
    stride_Cm: tl.int32,
    stride_Cn: tl.int32,
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

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K in chunks
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)

        # Load A tile: [BLOCK_M, BLOCK_K]
        a_ptrs = A_ptr + pid_b * stride_Ab + m_offsets[:, None] * stride_Am + k_offsets[None, :] * stride_Ak
        a_mask = (m_offsets[:, None] < M) & (k_offsets[None, :] < K)
        A_tile = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # Load W^T tile: [BLOCK_K, BLOCK_N] using W[k, n]
        w_ptrs = W_ptr + k_offsets[:, None] * stride_Wk + n_offsets[None, :] * stride_Wn
        w_mask = (k_offsets[:, None] < K) & (n_offsets[None, :] < N)
        Wt_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)

        # Accumulate
        acc += tl.dot(A_tile, Wt_tile)

    # Store result
    c_ptrs = C_ptr + pid_b * stride_Cb + m_offsets[:, None] * stride_Cm + n_offsets[None, :] * stride_Cn
    c_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        Triton-optimized forward:
          - Concatenates encoder_hidden_states and hidden_states along sequence dimension via Triton.
          - Applies linear projection via Triton GEMM.
          - Splits the result into separate streams.

        Args:
            hidden_states: [B, L_img, D]
            encoder_hidden_states: [B, L_txt, D]
            process_weight: [D, D]
        Returns:
            processed_encoder: [B, L_txt, D]
            processed_hidden: [B, L_img, D]
        """
        # Shapes
        B = hidden_states.shape[0]
        L_img = hidden_states.shape[1]
        L_txt = encoder_hidden_states.shape[1]
        D = hidden_states.shape[2]
        assert D == encoder_hidden_states.shape[2], "hidden_dim must match between inputs"
        assert process_weight.shape == (D, D), f"process_weight must have shape [hidden_dim, hidden_dim], got {process_weight.shape}"

        # Ensure contiguity
        encoder_hidden_states = encoder_hidden_states.contiguous()
        hidden_states = hidden_states.contiguous()
        process_weight = process_weight.contiguous()

        # 1) Concatenate sequences using Triton
        total_seq = L_txt + L_img
        concatenated = torch.empty((B, total_seq, D), device=encoder_hidden_states.device, dtype=encoder_hidden_states.dtype)

        BLOCK_M = 128  # sequence tile
        # Launch kernel over (batch, tiles of sequence)
        grid_concat = (B, triton.cdiv(total_seq, BLOCK_M))
        concat_seqs_kernel[grid_concat](
            encoder_hidden_states, hidden_states, concatenated,
            B, L_txt, L_img, D,
            encoder_hidden_states.stride(0), encoder_hidden_states.stride(1), encoder_hidden_states.stride(2),
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            concatenated.stride(0), concatenated.stride(1), concatenated.stride(2),
            BLOCK_M=BLOCK_M, BLOCK_K=1,  # BLOCK_K is not used here; we loop over D in the kernel
            num_warps=4, num_stages=2,
        )

        # 2) Compute processed = concatenated @ process_weight.T via Triton GEMM
        processed = torch.empty((B, total_seq, D), device=concatenated.device, dtype=concatenated.dtype)

        BLOCK_M_M = 64
        BLOCK_N_N = 64
        BLOCK_K_K = 32
        grid_gemm = (B, triton.cdiv(total_seq, BLOCK_M_M), triton.cdiv(D, BLOCK_N_N))
        batched_matmul_kernel[grid_gemm](
            concatenated, process_weight, processed,
            B, total_seq, D, D,
            concatenated.stride(0), concatenated.stride(1), concatenated.stride(2),
            process_weight.stride(0), process_weight.stride(1),
            processed.stride(0), processed.stride(1), processed.stride(2),
            BLOCK_M=BLOCK_M_M, BLOCK_N=BLOCK_N_N, BLOCK_K=BLOCK_K_K,
            num_warps=4, num_stages=2,
        )

        # 3) Split back
        processed_encoder = processed[:, :L_txt, :]
        processed_hidden = processed[:, L_txt:, :]
        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
