import torch
import triton
import triton.language as tl


@triton.jit
def _concat_sequences_kernel(
    A_ptr, B_ptr, C_ptr,
    B, T, P, K,
    stride_A_b, stride_A_t, stride_A_k,
    stride_B_b, stride_B_p, stride_B_k,
    stride_C_b, stride_C_l, stride_C_k,
    BLOCK_B: tl.constexpr, BLOCK_L: tl.constexpr, BLOCK_K: tl.constexpr
):
    """
    Concatenates two tensors along the sequence dimension into a single output.
    A: [B, T, K], B: [B, P, K] -> C: [B, T+P, K]
    Each program handles a tile over (batch, sequence).
    """
    pid_b = tl.program_id(0)  # batch tile id
    pid_l = tl.program_id(1)  # sequence tile id

    b_offsets = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    l_offsets = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)

    total_L = T + P

    # Iterate over K dimension tiles
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < K

        # For each l in the tile, decide source: A (encoder) or B (image) depending on l < T
        for l0 in range(0, BLOCK_L):
            l_idx = l_offsets[l0]
            mask_l = l_idx < total_L

            # Determine which input to copy
            a_valid = mask_l & (l_idx < T)
            b_valid = mask_l & (l_idx >= T)

            # Load from A if valid, else 0
            a_ptrs = A_ptr + b_offsets[:, None] * stride_A_b + l_idx * stride_A_t + k_offsets[None, :] * stride_A_k
            a_mask = (b_offsets[:, None] < B) & a_valid & mask_k[None, :]
            a_vals = tl.load(a_ptrs, mask=a_mask, other=0.0)

            # Load from B if valid, else 0
            b_ptrs = B_ptr + b_offsets[:, None] * stride_B_b + (l_idx - T) * stride_B_p + k_offsets[None, :] * stride_B_k
            b_mask = (b_offsets[:, None] < B) & b_valid & mask_k[None, :]
            b_vals = tl.load(b_ptrs, mask=b_mask, other=0.0)

            # Sum contributions and store
            val = a_vals + b_vals  # shape (BLOCK_B, BLOCK_K)
            c_ptrs = C_ptr + b_offsets[:, None] * stride_C_b + l_idx[None, :] * stride_C_l + k_offsets[None, :] * stride_C_k
            c_mask = (b_offsets[:, None] < B) & mask_l[None, :] & mask_k[None, :]
            tl.store(c_ptrs, val, mask=c_mask)


@triton.jit
def _split_sequences_kernel(
    C_ptr, out1_ptr, out2_ptr,
    B, T, P, K,
    stride_C_b, stride_C_l, stride_C_k,
    stride_out1_b, stride_out1_t, stride_out1_k,
    stride_out2_b, stride_out2_p, stride_out2_k,
    BLOCK_B: tl.constexpr, BLOCK_L: tl.constexpr, BLOCK_K: tl.constexpr
):
    """
    Splits C of shape [B, T+P, K] into:
      - out1: [B, T, K] (first T rows)
      - out2: [B, P, K] (next P rows)
    """
    # Grid is 3D: (batch tiles, T tiles, P tiles)
    pid_b = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_p = tl.program_id(2)

    b_offsets = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)

    # Process out1 (encoder part)
    t_offsets = pid_t * BLOCK_L + tl.arange(0, BLOCK_L)
    mask_b = b_offsets < B
    mask_t = t_offsets < T

    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < K

        out1_tile = tl.zeros((BLOCK_B, BLOCK_L, BLOCK_K), dtype=tl.float32)

        for t0 in range(0, BLOCK_L):
            t_idx = t_offsets[t0]
            c_ptrs = C_ptr + b_offsets[:, None] * stride_C_b + t_idx * stride_C_l + k_offsets[None, :] * stride_C_k
            c_mask = mask_b[:, None] & mask_t[t0] & mask_k[None, :]
            c_vals = tl.load(c_ptrs, mask=c_mask, other=0.0)
            out1_tile[:, t0, :] = c_vals

        out1_ptrs = out1_ptr + b_offsets[:, None] * stride_out1_b + t_offsets[None, :] * stride_out1_t + k_offsets[None, :] * stride_out1_k
        store_mask = mask_b[:, None] & mask_t[None, :] & mask_k[None, :]
        tl.store(out1_ptrs, out1_tile, mask=store_mask)

    # Process out2 (image part)
    p_offsets = pid_p * BLOCK_L + tl.arange(0, BLOCK_L)
    mask_p = p_offsets < P

    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < K

        out2_tile = tl.zeros((BLOCK_B, BLOCK_L, BLOCK_K), dtype=tl.float32)

        for p0 in range(0, BLOCK_L):
            p_idx = p_offsets[p0]
            c_ptrs = C_ptr + b_offsets[:, None] * stride_C_b + (p_idx + T) * stride_C_l + k_offsets[None, :] * stride_C_k
            c_mask = mask_b[:, None] & mask_p[p0] & mask_k[None, :]
            c_vals = tl.load(c_ptrs, mask=c_mask, other=0.0)
            out2_tile[:, p0, :] = c_vals

        out2_ptrs = out2_ptr + b_offsets[:, None] * stride_out2_b + p_offsets[None, :] * stride_out2_p + k_offsets[None, :] * stride_out2_k
        store_mask = mask_b[:, None] & mask_p[None, :] & mask_k[None, :]
        tl.store(out2_ptrs, out2_tile, mask=store_mask)


def _concat_and_split_triton(
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    process_weight: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Helper that:
      - Concatenates encoder_hidden_states and hidden_states (Triton)
      - Computes matmul with process_weight.T (PyTorch)
      - Splits result into (processed_encoder, processed_hidden) (Triton)
    Returns: (processed_encoder [B, T, K], processed_hidden [B, P, K])
    """
    assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Tensors must be on CUDA for Triton"

    B = hidden_states.shape[0]
    T = encoder_hidden_states.shape[1]  # text_seq_len
    P = hidden_states.shape[1]          # img_seq_len
    K = hidden_states.shape[2]          # hidden_dim

    # Ensure contiguous
    A = encoder_hidden_states.contiguous()   # [B, T, K]
    Bsrc = hidden_states.contiguous()        # [B, P, K]
    W = process_weight.contiguous()          # [K, K]

    # Allocate concatenated output [B, T+P, K]
    total_L = T + P
    C = torch.empty((B, total_L, K), device=hidden_states.device, dtype=torch.float32)

    # Launch Triton concat kernel
    BLOCK_B = 32
    BLOCK_L = 64
    BLOCK_K = 64
    grid = (triton.cdiv(B, BLOCK_B), triton.cdiv(total_L, BLOCK_L))
    _concat_sequences_kernel[grid](
        A, Bsrc, C,
        B, T, P, K,
        A.stride(0), A.stride(1), A.stride(2),
        Bsrc.stride(0), Bsrc.stride(1), Bsrc.stride(2),
        C.stride(0), C.stride(1), C.stride(2),
        BLOCK_B=BLOCK_B, BLOCK_L=BLOCK_L, BLOCK_K=BLOCK_K,
    )

    # GEMM: (B, T+P, K) @ (K, K) -> (B, T+P, K)
    processed = torch.matmul(C, W.t())

    # Allocate outputs
    processed_encoder = torch.empty((B, T, K), device=hidden_states.device, dtype=torch.float32)
    processed_hidden = torch.empty((B, P, K), device=hidden_states.device, dtype=torch.float32)

    # Launch Triton split kernel
    grid1 = (triton.cdiv(B, BLOCK_B), triton.cdiv(T, BLOCK_L), triton.cdiv(P, BLOCK_L))
    _split_sequences_kernel[grid1](
        processed, processed_encoder, processed_hidden,
        B, T, P, K,
        processed.stride(0), processed.stride(1), processed.stride(2),
        processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
        processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
        BLOCK_B=BLOCK_B, BLOCK_L=BLOCK_L, BLOCK_K=BLOCK_K,
    )

    return processed_encoder, processed_hidden


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                encoder_hidden_states: torch.Tensor,
                process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Use Triton for data movement; keep GEMM in PyTorch for performance.
        if not (hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda):
            raise RuntimeError("ModelNew expects CUDA tensors for Triton execution.")
        return _concat_and_split_triton(hidden_states, encoder_hidden_states, process_weight)


def run(*args):
    return ModelNew()(*args)
