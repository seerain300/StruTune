import torch
import triton
import triton.language as tl


@triton.jit
def _cat_seq_kernel(
    encoder_ptr,  # * [B, T, H]
    hidden_ptr,   # * [B, I, H]
    out_ptr,      # * [B, L, H]
    B: tl.constexpr, T: tl.constexpr, I: tl.constexpr, H: tl.constexpr,
    encoder_stride_b, encoder_stride_t, encoder_stride_h,
    hidden_stride_b, hidden_stride_i, hidden_stride_h,
    out_stride_b, out_stride_l, out_stride_h,
    BLOCK_L: tl.constexpr,
):
    # Each program handles one (batch, sequence) index
    b = tl.program_id(0)
    m = tl.program_id(1)  # 0..L-1

    # If m < T: use encoder[b, m, :]
    # Else: use hidden[b, m - T, :]
    k_range = tl.arange(0, H)

    if m < T:
        enc_ptrs = encoder_ptr + b * encoder_stride_b + m * encoder_stride_t + k_range * encoder_stride_h
    else:
        hid_ptrs = hidden_ptr + b * hidden_stride_b + (m - T) * hidden_stride_i + k_range * hidden_stride_h

    # Load source vector
    vals = tl.load(enc_ptrs if m < T else hid_ptrs)

    # Store into output
    out_ptrs = out_ptr + b * out_stride_b + m * out_stride_l + k_range * out_stride_h
    tl.store(out_ptrs, vals)


@triton.jit
def _batched_matmul_kernel(
    A_ptr,  # * [B, L, K]
    B_ptr,  # * [K, N]
    C_ptr,  # * [B, L, N]
    B, L, K, N,
    A_stride_b, A_stride_l, A_stride_k,
    B_stride_k, B_stride_n,
    C_stride_b, C_stride_l, C_stride_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Grid dims: (B, ceil_div(L, BLOCK_M), ceil_div(N, BLOCK_N))
    b = tl.program_id(0)
    m_block = tl.program_id(1)
    n_block = tl.program_id(2)

    m_offsets = m_block * BLOCK_M + tl.arange(0, BLOCK_M)  # sequence positions
    n_offsets = n_block * BLOCK_N + tl.arange(0, BLOCK_N)  # output columns

    m_mask = m_offsets < L
    n_mask = n_offsets < N

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Reduce over K in chunks
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < K

        # A tile: [BLOCK_M, BLOCK_K] -> A[b, m, k]
        a_ptrs = A_ptr + b * A_stride_b + m_offsets[:, None] * A_stride_l + k_offsets[None, :] * A_stride_k
        a_mask = m_mask[:, None] & k_mask[None, :]
        a_tile = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # B tile: [BLOCK_K, BLOCK_N] -> B[k, n] = W_T[k, n]
        b_ptrs = B_ptr + k_offsets[:, None] * B_stride_k + n_offsets[None, :] * B_stride_n
        b_mask = k_mask[:, None] & n_mask[None, :]
        b_tile = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Accumulate in fp32
        acc += tl.dot(a_tile.to(tl.float32), b_tile.to(tl.float32))

    # Store result to C (dtype of C_ptr will dictate cast)
    c_ptrs = C_ptr + b * C_stride_b + m_offsets[:, None] * C_stride_l + n_offsets[None, :] * C_stride_n
    store_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(c_ptrs, acc, mask=store_mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version:
          - Concatenation along sequence dim implemented in Triton (cat_seq_kernel).
          - Batched GEMM implemented in Triton (batched_matmul_kernel).
          - Returns (processed_encoder, processed_hidden) split along sequence dim.
        """
        # Inputs
        encoder = encoder_hidden_states  # [B, T, H]
        hidden = hidden_states            # [B, I, H]
        W = process_weight                # [H, H]

        # Shapes
        B = encoder.shape[0]
        T = encoder.shape[1]
        I = hidden.shape[1]
        H = encoder.shape[2]
        assert hidden.shape[2] == H
        assert W.shape[0] == H and W.shape[1] == H

        device = encoder.device

        # Allocate concatenated output [B, L, H], same dtype as input (typical fp32 in eval)
        A_cat = torch.empty((B, T + I, H), device=device, dtype=encoder.dtype)

        # Launch concat kernel: grid over (batch, sequence positions)
        grid_cat = (B, T + I)
        _cat_seq_kernel[grid_cat](
            encoder, hidden, A_cat,
            B=B, T=T, I=I, H=H,
            encoder_stride_b=encoder.stride(0), encoder_stride_t=encoder.stride(1), encoder_stride_h=encoder.stride(2),
            hidden_stride_b=hidden.stride(0), hidden_stride_i=hidden.stride(1), hidden_stride_h=hidden.stride(2),
            out_stride_b=A_cat.stride(0), out_stride_l=A_cat.stride(1), out_stride_h=A_cat.stride(2),
            BLOCK_L=1,
            num_warps=1, num_stages=1,
        )

        # Prepare W_T = process_weight.T (no bias), shape [H, H]
        W_T = W.t().contiguous()  # [H, H]

        # Allocate processed output [B, L, H]
        processed = torch.empty((B, T + I, H), device=device, dtype=encoder.dtype)

        # Choose tiling parameters (robust defaults)
        BLOCK_M = 128  # tile over sequence positions
        BLOCK_N = 128  # tile over output columns (same as hidden_dim)
        BLOCK_K = 64   # tile over reduction dimension (hidden_dim)

        # Grid over batch, sequence tiles, and output tiles
        grid_mm = (B, triton.cdiv(T + I, BLOCK_M), triton.cdiv(H, BLOCK_N))

        # Run Triton batched GEMM: processed = A_cat @ W_T
        _batched_matmul_kernel[grid_mm](
            A_cat, W_T, processed,
            B=B, L=T+I, K=H, N=H,
            A_stride_b=A_cat.stride(0), A_stride_l=A_cat.stride(1), A_stride_k=A_cat.stride(2),
            B_stride_k=W_T.stride(0), B_stride_n=W_T.stride(1),
            C_stride_b=processed.stride(0), C_stride_l=processed.stride(1), C_stride_n=processed.stride(2),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=3,
        )

        # Split processed into encoder and hidden parts
        processed_encoder = processed[:, :T, :]
        processed_hidden = processed[:, T:, :]

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
