import torch
import triton
import triton.language as tl


@triton.jit
def cat_seq_kernel(
    encoder_ptr, hidden_ptr, out_ptr,
    B, T, I, H,
    stride_e_b, stride_e_t, stride_e_h,
    stride_h_b, stride_h_i, stride_h_h,
    stride_o_b, stride_o_l, stride_o_h,
    BLOCK_N: tl.constexpr,
):
    """
    Concatenate along sequence dimension:
    out[b, l, :] = encoder[b, l, :] if l < T else hidden[b, l - T, :]
    Input shapes:
      encoder: [B, T, H]
      hidden: [B, I, H]
    Output shape:
      out: [B, T+I, H]
    Grid: (B, ceil((T+I)/BLOCK_N))
    """
    b = tl.program_id(0)
    n = tl.program_id(1)
    offs_n = n * BLOCK_N + tl.arange(0, BLOCK_N)
    L = T + I
    mask_n = offs_n < L

    # Base pointers for batch
    e_b_ptr = encoder_ptr + b * stride_e_b
    h_b_ptr = hidden_ptr + b * stride_h_b
    o_b_ptr = out_ptr + b * stride_o_b

    # For each hidden dimension k, copy from encoder for l < T or from hidden for l >= T
    for k in range(0, H):
        # Compute encoder contributions for l < T
        l_mask = mask_n < T
        enc_ptrs = e_b_ptr + l_mask[:, None] * stride_e_t + k * stride_e_h
        enc_vals = tl.load(enc_ptrs, mask=l_mask & mask_n, other=0.0)  # [BLOCK_N]

        # Compute hidden contributions for l >= T
        h_mask = mask_n >= T
        h_l = offs_n - T
        h_ptrs = h_b_ptr + h_mask[:, None] * stride_h_i + k * stride_h_h
        h_vals = tl.load(h_ptrs, mask=h_mask & mask_n, other=0.0)  # [BLOCK_N]

        # Select based on l
        sel = mask_n[:, None] < T  # broadcast across k
        vals = tl.where(sel, enc_vals[None, :], h_vals[None, :])  # [BLOCK_N, 1], effectively [BLOCK_N]

        # Store into output
        out_ptrs = o_b_ptr + mask_n[:, None] * stride_o_l + k * stride_o_h
        tl.store(out_ptrs, vals, mask=mask_n[:, None])


@triton.jit
def batched_matmul_kernel_3d(
    A_ptr, Wt_ptr, C_ptr,
    B, M, N, K,
    stride_a_b, stride_a_m, stride_a_k,
    stride_w_k, stride_w_n,
    stride_c_b, stride_c_m, stride_c_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """
    Compute C[b, m, n] = sum_k A[b, m, k] * Wt[k, n]
    A: [B, M, K] (concatenated sequences), row-major along k
    Wt: [K, N] (process_weight.T), row-major along n
    C: [B, M, N]
    Launch grid: (B, ceil(M/BLOCK_M), ceil(N/BLOCK_N))
    """
    b = tl.program_id(0)
    m_block = tl.program_id(1)
    n_block = tl.program_id(2)

    offs_m = m_block * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = n_block * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = offs_m < M
    n_mask = offs_n < N

    # Accumulator in fp32 for numerical stability
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Reduction over K in tiles
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K

        # Load A tile [BLOCK_M, BLOCK_K]
        a_ptrs = A_ptr + b * stride_a_b + offs_m[:, None] * stride_a_m + offs_k[None, :] * stride_a_k
        a = tl.load(a_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)

        # Load Wt tile [BLOCK_K, BLOCK_N]
        w_ptrs = Wt_ptr + offs_k[:, None] * stride_w_k + offs_n[None, :] * stride_w_n
        w = tl.load(w_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)

        # Accumulate
        acc += tl.dot(a, w)

    # Store result
    c_ptrs = C_ptr + b * stride_c_b + offs_m[:, None] * stride_c_m + offs_n[None, :] * stride_c_n
    tl.store(c_ptrs, acc, mask=m_mask[:, None] & n_mask[None, :])


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                encoder_hidden_states: torch.Tensor,
                process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version:
        1) Concatenate encoder_hidden_states and hidden_states along sequence dimension in Triton.
        2) Apply linear projection (A_cat @ process_weight.T) in Triton.
        3) Split back into separate encoder and image streams.
        Returns:
          processed_encoder: [B, T, H]
          processed_hidden: [B, I, H]
        """
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, \
            "All tensors must be on CUDA for Triton kernels."
        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        H = hidden_states.shape[2]
        assert encoder_hidden_states.shape[2] == H, "Hidden dims must match."
        assert process_weight.shape[0] == H and process_weight.shape[1] == H, "process_weight must be [H, H]."

        # Ensure contiguity and dtype
        encoder = encoder_hidden_states.contiguous()
        hidden = hidden_states.contiguous()
        W = process_weight.contiguous()

        # Concatenate along sequence dimension using Triton
        L = T + I
        A_cat = torch.empty((B, L, H), device=hidden.device, dtype=hidden.dtype)

        BLOCK_N = 128  # tile size over sequence length
        grid_cat = (B, triton.cdiv(L, BLOCK_N))
        cat_seq_kernel[grid_cat](
            encoder, hidden, A_cat,
            B, T, I, H,
            encoder.stride(0), encoder.stride(1), encoder.stride(2),
            hidden.stride(0), hidden.stride(1), hidden.stride(2),
            A_cat.stride(0), A_cat.stride(1), A_cat.stride(2),
            BLOCK_N=BLOCK_N,
            num_warps=4, num_stages=2,
        )

        # Transpose process_weight to [H, H] for Triton (Wt[k, n] = process_weight[n, k])
        Wt = process_weight.t().contiguous()  # [H, H], same dtype as process_weight

        # Allocate output for processed
        processed = torch.empty((B, L, H), device=hidden.device, dtype=A_cat.dtype)

        # Launch batched matmul kernel: C[b, m, n] = sum_k A_cat[b


def run(*args):
    return ModelNew()(*args)
