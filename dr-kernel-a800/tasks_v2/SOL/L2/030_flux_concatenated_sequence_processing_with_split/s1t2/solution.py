import torch
import triton
import triton.language as tl


@triton.jit
def _concat_encoder_image_kernel(
    A_ptr, B_ptr, C_ptr,
    B, T, P, K,
    stride_A_b, stride_A_t, stride_A_k,
    stride_B_b, stride_B_p, stride_B_k,
    stride_C_b, stride_C_l, stride_C_k,
    BLOCK_T: tl.constexpr, BLOCK_P: tl.constexpr,
):
    """
    Concatenate along sequence dimension:
      - A: encoder_hidden_states [B, T, K]
      - B: hidden_states [B, P, K]
      - C: concatenated [B, T+P, K]
    Each program handles a tile over batch and sequence.
    """
    pid_b = tl.program_id(0)  # batch tile
    pid_seq = tl.program_id(1)  # combined (t/p) tile

    b_offsets = pid_b * BLOCK_T + tl.arange(0, BLOCK_T)  # [BLOCK_T]
    total_L = T + P
    seq_offsets = pid_seq * BLOCK_P + tl.arange(0, BLOCK_P)  # [BLOCK_P]

    # For each position in seq_offsets, decide whether it comes from A (encoder) or B (image)
    for i in range(0, BLOCK_P):
        l_idx = seq_offsets[i]  # global sequence index
        # Validity masks
        valid_l = l_idx < total_L
        # Determine source
        a_mask = valid_l & (l_idx < T)
        b_mask = valid_l & (l_idx >= T)

        # Initialize output pointer for this (b, l)
        c_ptrs = C_ptr + b_offsets[:, None] * stride_C_b + l_idx[None, :] * stride_C_l + tl.arange(0, K) * stride_C_k  # broadcast over K
        # Compose values: load from A if a_mask else 0, load from B if b_mask else 0, then store
        a_ptrs = A_ptr + b_offsets[:, None] * stride_A_b + (l_idx if a_mask else 0) * stride_A_t + tl.arange(0, K) * stride_A_k
        b_ptrs = B_ptr + b_offsets[:, None] * stride_B_b + (l_idx - T if b_mask else 0) * stride_B_p + tl.arange(0, K) * stride_B_k

        a_mask2 = (b_offsets[:, None] < B) & a_mask
        b_mask2 = (b_offsets[:, None] < B) & b_mask

        a_vals = tl.load(a_ptrs, mask=a_mask2, other=0.0)
        b_vals = tl.load(b_ptrs, mask=b_mask2, other=0.0)

        val = a_vals + b_vals
        tl.store(c_ptrs, val, mask=(b_offsets[:, None] < B) & valid_l[None, :])


@triton.jit
def _triton_matmul_kernel(
    A_ptr, W_ptr, C_ptr,
    M, N, K,
    stride_A_m, stride_A_k,
    stride_W_k, stride_W_n,
    stride_C_m, stride_C_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """
    Compute C = A @ W, where:
      - A is [M, K] (we feed Acat flattened to [M, K]).
      - W is [K, N] = process_weight.T contiguous.
      - C is [M, N].
    Each Triton program computes a [BLOCK_M x BLOCK_N] tile of C, reducing over K.
    """
    pid_m = tl.program_id(0)  # tile id along M
    pid_n = tl.program_id(1)  # tile id along N

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # [BLOCK_M]
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)   # [BLOCK_N]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)  # [BLOCK_K]

        # Load A tile: [BLOCK_M, BLOCK_K]
        a_ptrs = A_ptr + m_offsets[:, None] * stride_A_m + k_offsets[None, :] * stride_A_k
        a_mask = (m_offsets[:, None] < M) & (k_offsets[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # Load W tile: [BLOCK_K, BLOCK_N]
        w_ptrs = W_ptr + k_offsets[:, None] * stride_W_k + n_offsets[None, :] * stride_W_n
        w_mask = (k_offsets[:, None] < K) & (n_offsets[None, :] < N)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)

        # Accumulate
        acc += tl.dot(a, w)

    # Store results to C
    c_ptrs = C_ptr + m_offsets[:, None] * stride_C_m + n_offsets[None, :] * stride_C_n
    c_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


@triton.jit
def _split_encoder_kernel(
    C_ptr, out_ptr,
    B, T, N, K,
    stride_C_b, stride_C_l, stride_C_k,
    stride_out_b, stride_out_t, stride_out_k,
    BLOCK_B: tl.constexpr, BLOCK_T: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """
    Split C [B, T+P, N] into out [B, T, N] (first T rows).
    """
    pid_b = tl.program_id(0)
    pid_t = tl.program_id(1)

    b_offsets = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    t_offsets = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)

    for k0 in range(0, N, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)

        out_tile = tl.zeros((BLOCK_B, BLOCK_T, BLOCK_K), dtype=tl.float32)

        for t0 in range(0, BLOCK_T):
            t_idx = t_offsets[t0]
            c_ptrs = C_ptr + b_offsets[:, None] * stride_C_b + t_idx * stride_C_l + k_offsets[None, :] * stride_C_k
            mask = (b_offsets[:, None] < B) & (t_idx < T) & (k_offsets[None, :] < N)
            c_vals = tl.load(c_ptrs, mask=mask, other=0.0)
            out_tile[:, t0, :] = c_vals

        out_ptrs = out_ptr + b_offsets[:, None] * stride_out_b + t_offsets[None, :] * stride_out_t + k_offsets[None, :] * stride_out_k
        store_mask = (b_offsets[:, None] < B) & (t_offsets[None, :] < T) & (k_offsets[None, :] < N)
        tl.store(out_ptrs, out_tile, mask=store_mask)


@triton.jit
def _split_image_kernel(
    C_ptr, out_ptr,
    B, T, P, N, K,
    stride_C_b, stride_C_l, stride_C_k,
    stride_out_b, stride_out_p, stride_out_k,
    BLOCK_B: tl.constexpr, BLOCK_P: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """
    Split C [B, T+P, N] into out [B, P, N] (next P rows).
    """
    pid_b = tl.program_id(0)
    pid_p = tl.program_id(1)

    b_offsets = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    p_offsets = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)

    for k0 in range(0, N, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)

        out_tile = tl.zeros((BLOCK_B, BLOCK_P, BLOCK_K), dtype=tl.float32)

        for p0 in range(0, BLOCK_P):
            p_idx = p_offsets[p0]
            c_ptrs = C_ptr + b_offsets[:, None] * stride_C_b + (p_idx + T) * stride_C_l + k_offsets[None, :] * stride_C_k
            mask = (b_offsets[:, None] < B) & (p_idx < P) & (k_offsets[None, :] < N)
            c_vals = tl.load(c_ptrs, mask=mask, other=0.0)
            out_tile[:, p0, :] = c_vals

        out_ptrs = out_ptr + b_offsets[:, None] * stride_out_b + p_offsets[None, :] * stride_out_p + k_offsets[None, :] * stride_out_k
        store_mask = (b_offsets[:, None] < B) & (p_offsets[None, :] < P) & (k_offsets[None, :] < N)
        tl.store(out_ptrs, out_tile, mask=store_mask)


def _run_triton_concat_and_matmul_and_split(
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    process_weight: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Full Triton path:
      - Concatenate encoder_hidden_states and hidden_states along sequence dim using Triton
      - Compute matmul with process_weight.T using Triton
      - Split result into (processed_encoder, processed_hidden) using Triton
    Returns: (processed_encoder [B, T, K], processed_hidden [B, P, K])
    """
    assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Tensors must be CUDA for Triton."
    assert hidden_states.ndim == 3 and encoder_hidden_states.ndim == 3, "Inputs must be 3D [B, len, K]."
    assert process_weight.ndim == 2, "process_weight must be 2D [K, K]."

    B = hidden_states.shape[0]
    T = encoder_hidden_states.shape[1]  # text_seq_len
    P = hidden_states.shape[1]          # img_seq_len
    K = hidden_states.shape[2]          # hidden_dim

    # Ensure contiguity
    A = encoder_hidden_states.contiguous()   # [B, T, K]
    Bsrc = hidden_states.contiguous()        # [B, P, K]
    W = process_weight.contiguous()          # [K, K]

    # 1) Concatenate with Triton: Ccat [B, T+P, K]
    total_L = T + P
    Ccat = torch.empty((B, total_L, K), device=hidden_states.device, dtype=torch.float32)

    BLOCK_T = 64
    BLOCK_P = 128
    grid_concat = (triton.cdiv(B, BLOCK_T), triton.cdiv(total_L, BLOCK_P))
    _concat_encoder_image_kernel[grid_concat](
        A, Bsrc, Ccat,
        B, T, P, K,
        A.stride(0), A.stride(1), A.stride(2),
        Bsrc.stride(0), Bsrc.stride(1), Bsrc.stride(2),
        Ccat.stride(0), Ccat.stride(1), Ccat.stride(2),
        BLOCK_T=BLOCK_T, BLOCK_P=BLOCK_P,
    )

    # 2) Matmul in Triton: C = Ccat @ W.T
    # Flatten A to [M, K], W_T = [K, K]
    M = B * total_L
    Aflat = Ccat.reshape(M, K).contiguous()  # [M, K]
    Wt = W.t().contiguous()                  # [K, K]
    Cmat = torch.empty((M, K), device=hidden_states.device, dtype=torch.float32)

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 64
    grid_mm = (triton.cdiv(M, BLOCK_M), triton.cdiv(K, BLOCK_N))
    _triton_matmul_kernel[grid_mm](
        Aflat, Wt, Cmat,
        M, K, K,
        Aflat.stride(0), Aflat.stride(1),
        Wt.stride(0), Wt.stride(1),
        Cmat.stride(0), Cmat.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=2
    )

    # Reshape to [B, T+P, K]
    C3 = Cmat.reshape(B, total_L, K)

    # 3) Split with Triton
    processed_encoder = torch.empty((B, T, K), device=hidden_states.device, dtype=torch.float32)
    processed_hidden = torch.empty((B, P, K), device=hidden_states.device, dtype=torch.float32)

    # Tuning for splits
    BLOCK_B = 64
    BLOCK_T_split = 64
    BLOCK_P_split = 128
    BLOCK_K_split = 64

    # Encoder split
    grid_e = (triton.cdiv(B, BLOCK_B), triton.cdiv(T, BLOCK_T_split))
    _split_encoder_kernel[grid_e](
        C3, processed_encoder,
        B, T, T, K,
        C3.stride(0), C3.stride(1), C3.stride(2),
        processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
        BLOCK_B=BLOCK_B, BLOCK_T=BLOCK_T_split, BLOCK_K=BLOCK_K_split,
        num_warps=4, num_stages=2
    )

    # Image split
    grid_i = (triton.cdiv(B, BLOCK_B), triton.cdiv(P, BLOCK_P_split))
    _split_image_kernel[grid_i](
        C3, processed_hidden,
        B, T, P, K,
        C3.stride(0), C3.stride(1), C3.stride(2),
        processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
        BLOCK_B=BLOCK_B, BLOCK_P=BLOCK_P_split, BLOCK_K=BLOCK_K_split,
        num_warps=4, num_stages=2
    )

    return processed_encoder, processed_hidden


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                encoder_hidden_states: torch.Tensor,
                process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-only implementation:
          - Concatenate (encoder, image) via Triton kernel
          - Linear projection via Triton matmul kernel
          - Split into encoder and image outputs via Triton kernels
        """
        if not (hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda):
            raise RuntimeError("ModelNew expects CUDA tensors for Triton execution.")
        return _run_triton_concat_and_matmul_and_split(hidden_states, encoder_hidden_states, process_weight)


def run(*args):
    return ModelNew()(*args)
