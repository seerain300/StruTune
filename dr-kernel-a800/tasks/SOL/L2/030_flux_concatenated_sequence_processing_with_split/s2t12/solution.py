import torch
import triton
import triton.language as tl


@triton.jit
def concat_seq_kernel(
    encoder_ptr, hidden_ptr, out_ptr,
    B, T, I, H,
    # strides
    enc_bs, enc_ts, enc_hs,
    hid_bs, hid_ts, hid_hs,
    out_bs, out_ts, out_hs,
):
    # Grid: (batch, sequence)
    b = tl.program_id(0)
    m = tl.program_id(1)

    # Bounds check on batch
    if b >= B:
        return

    # Compute source index and load from appropriate tensor
    if m < T:
        # encoder[b, m, :]
        enc_off = b * enc_bs + m * enc_ts
        # vector of H columns
        cols = tl.arange(0, H)
        vals = tl.load(encoder_ptr + enc_off + cols * enc_hs)
        # store to out[b, m, :]
        out_off = b * out_bs + m * out_ts
        tl.store(out_ptr + out_off + cols * out_hs, vals)
    else:
        # hidden[b, m - T, :]
        idx = m - T
        hid_off = b * hid_bs + idx * hid_ts
        cols = tl.arange(0, H)
        vals = tl.load(hidden_ptr + hid_off + cols * hid_hs)
        # store to out[b, m, :]
        out_off = b * out_bs + m * out_ts
        tl.store(out_ptr + out_off + cols * out_hs, vals)


@triton.jit
def batched_matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    B_sz, M_sz, K_sz, N_sz,
    # strides
    A_bs, A_ms, A_ks,
    B_rs, B_cs,  # B_ptr is [K, N] where we pass B_rs = W_T.stride(0) = 1, B_cs = W_T.stride(1) = H
    C_bs, C_ms, C_ns,
    # tiling
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Grid: (m_tiles, n_tiles, b)
    m_tile = tl.program_id(0)
    n_tile = tl.program_id(1)
    b = tl.program_id(2)

    # Output tile indices
    offs_m = m_tile * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = n_tile * BLOCK_N + tl.arange(0, BLOCK_N)

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Reduction over K in chunks
    for k0 in range(0, K_sz, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)

        # Pointers for A[b, m, k] and B[k, n]
        A_ptrs = A_ptr + b * A_bs + offs_m[:, None] * A_ms + offs_k[None, :] * A_ks
        B_ptrs = B_ptr + offs_k[:, None] * B_rs + offs_n[None, :] * B_cs

        # Masks for loads
        A_mask = (offs_m[:, None] < M_sz) & (offs_k[None, :] < K_sz)
        B_mask = (offs_k[:, None] < K_sz) & (offs_n[None, :] < N_sz)

        # Load A and B as fp32
        A_vals = tl.load(A_ptrs, mask=A_mask, other=0.0)
        B_vals = tl.load(B_ptrs, mask=B_mask, other=0.0)

        # Accumulate
        acc += tl.dot(A_vals.to(tl.float32), B_vals.to(tl.float32))

    # Write back to C[b, m, n]
    C_ptrs = C_ptr + b * C_bs + offs_m[:, None] * C_ms + offs_n[None, :] * C_ns
    C_mask = (offs_m[:, None] < M_sz) & (offs_n[None, :] < N_sz)
    # Cast accumulator to float32 (C is float32); store
    tl.store(C_ptrs, acc, mask=C_mask)


class ModelNew(torch.nn.Module):
    def forward(self, encoder_hidden_states: torch.Tensor,
                hidden_states: torch.Tensor,
                process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version:
          1) Concatenate encoder_hidden_states and hidden_states along sequence dimension in Triton
          2) Apply linear projection (A_cat @ process_weight.T) in Triton
          3) Split back into separate encoder and image streams
        Returns:
          (processed_encoder: [B, T, H], processed_hidden: [B, I, H])
        """
        # Extract shapes
        B = encoder_hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        H = encoder_hidden_states.shape[2]
        L = T + I

        # Prepare inputs for Triton: cast to float32 and make contiguous
        enc32 = encoder_hidden_states.contiguous().to(torch.float32)
        hid32 = hidden_states.contiguous().to(torch.float32)
        w32 = process_weight.contiguous().to(torch.float32)

        # Allocate output for concatenation [B, L, H]
        A_cat32 = torch.empty((B, L, H), device=enc32.device, dtype=torch.float32)

        # Launch Triton concatenation kernel
        grid_concat = (B, L)
        concat_seq_kernel[grid_concat](
            enc32, hid32, A_cat32,
            B, T, I, H,
            enc32.stride(0), enc32.stride(1), enc32.stride(2),
            hid32.stride(0), hid32.stride(1), hid32.stride(2),
            A_cat32.stride(0), A_cat32.stride(1), A_cat32.stride(2),
            num_warps=4, num_stages=2,
        )

        # Prepare W_T = process_weight.T: [H, H]
        W_T = w32.transpose(0, 1).contiguous()

        # Output buffer for processed: [B, L, H], float32
        processed32 = torch.empty((B, L, H), device=enc32.device, dtype=torch.float32)

        # Tile sizes and grid for GEMM
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 32
        grid_mm = (triton.cdiv(L, BLOCK_M), triton.cdiv(H, BLOCK_N), B)

        # Launch Triton GEMM: A_cat32 [B, L, H], W_T [H, H], C [B, L, H]
        batched_matmul_kernel[grid_mm](
            A_cat32, W_T, processed32,
            B, L, H, H,  # M=L, K=H, N=H
            A_cat32.stride(0), A_cat32.stride(1), A_cat32.stride(2),
            W_T.stride(0), W_T.stride(1),  # B_rs=1 (contiguous along N), B_cs=H
            processed32.stride(0), processed32.stride(1), processed32.stride(2),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=3,
        )

        # Cast output back to original dtype of encoder_hidden_states
        processed_dtype = encoder_hidden_states.dtype
        processed32_out = processed32.to(processed_dtype)

        processed_encoder = processed32_out[:, :T, :]
        processed_hidden = processed32_out[:, T:, :]

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
