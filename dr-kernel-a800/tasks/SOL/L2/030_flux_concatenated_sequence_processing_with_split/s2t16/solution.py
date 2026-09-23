import torch
import triton
import triton.language as tl


@triton.jit
def cat_seq_kernel(
    encoder_ptr, hidden_ptr, out_ptr,
    B, T, I, H,
    enc_s0, enc_s1, enc_s2,
    hid_s0, hid_s1, hid_s2,
    out_s0, out_s1, out_s2,
    BLOCK_N: tl.constexpr,
):
    # Grid: (B, ceil_div(L, BLOCK_N))
    b = tl.program_id(0)
    tile = tl.program_id(1)
    m = tile * BLOCK_N + tl.arange(0, BLOCK_N)
    L = T + I

    # Mask for valid sequence positions
    m_mask = m < L

    # Compute source indices
    # If m < T, use encoder; else use hidden at offset m - T
    use_encoder = m < T

    # Compute pointer offsets
    # For A_cat[b, m, k], k in [0..H-1]
    k = tl.arange(0, H)
    # Build masks per k (always valid since k<H, but keep for clarity)
    k_mask = k < H

    # Load from encoder where applicable
    enc_offsets = b * enc_s0 + m[:, None] * enc_s1 + k[None, :] * enc_s2
    enc_mask = m_mask[:, None] & k_mask[None, :] & use_encoder[:, None]
    A_block = tl.load(encoder_ptr + enc_offsets, mask=enc_mask, other=0.0)

    # Load from hidden where applicable
    hid_offsets = b * hid_s0 + (m - T)[:, None] * hid_s1 + k[None, :] * hid_s2
    hid_mask = m_mask[:, None] & k_mask[None, :] & (~use_encoder)[:, None]
    B_block = tl.load(hidden_ptr + hid_offsets, mask=hid_mask, other=0.0)

    # Combine
    AB = A_block + B_block

    # Store to output A_cat[b, m, :]
    out_offsets = b * out_s0 + m[:, None] * out_s1 + k[None, :] * out_s2
    out_mask = m_mask[:, None] & k_mask[None, :]
    tl.store(out_ptr + out_offsets, AB, mask=out_mask)


@triton.jit
def batched_matmul_kernel_3d(
    A_ptr, B_ptr, C_ptr,
    Batches, M, N, K,
    A_s0, A_s1, A_s2,
    B_s0, B_s1, B_s2,
    C_s0, C_s1, C_s2,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 3D grid: (batch, tiles over M, tiles over N)
    batch = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    k = tl.arange(0, BLOCK_K)

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension in BLOCK_K steps
    for k_start in range(0, K, BLOCK_K):
        # Pointers for A tile: A[batch, m, k_start:k_start+BLOCK_K]
        A_offsets = batch * A_s0 + m[:, None] * A_s1 + (k_start + k)[None, :] * A_s2
        A_mask = (m[:, None] < M) & ((k_start + k)[None, :] < K)
        A_sub = tl.load(A_ptr + A_offsets, mask=A_mask, other=0.0)

        # Pointers for B tile: B[k_start:k_start+BLOCK_K, n] which is [K_tile, N_tile]
        B_offsets = k_start * B_s0 + k[:, None] * B_s1 + n[None, :] * B_s2
        B_mask = ((k_start + k)[:, None] < K) & (n[None, :] < N)
        B_sub = tl.load(B_ptr + B_offsets, mask=B_mask, other=0.0)

        # Accumulate
        acc += tl.dot(A_sub, B_sub)

    # Store result to C[batch, m, n]
    C_offsets = batch * C_s0 + m[:, None] * C_s1 + n[None, :] * C_s2
    C_mask = (m[:, None] < M) & (n[None, :] < N)
    # Store as float32; C is allocated as float32
    tl.store(C_ptr + C_offsets, acc, mask=C_mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version:
        - Concatenate encoder_hidden_states and hidden_states along sequence dimension in Triton.
        - Apply linear projection (A_cat @ process_weight.T) in Triton.
        - Split back into separate encoder and image streams.
        Returns (processed_encoder, processed_hidden) with shapes [B, T, H] and [B, I, H].
        """
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, \
            "All inputs must be CUDA tensors for Triton kernels."
        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        H = hidden_states.shape[2]
        assert encoder_hidden_states.shape[2] == H, "Hidden dims must match."
        assert process_weight.shape[0] == H and process_weight.shape[1] == H, "process_weight must be [H, H]."

        # Ensure contiguity
        encoder = encoder_hidden_states.contiguous()
        hidden = hidden_states.contiguous()
        W = process_weight.contiguous()

        # Concatenate along sequence dimension: [B, L, H], L = T + I
        L = T + I
        A_cat = torch.empty((B, L, H), device=hidden.device, dtype=hidden.dtype)

        BLOCK_N = 128
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

        # Transpose process_weight to [H, H] for Triton matmul (Wt[k, n] = process_weight[n, k])
        # Keep dtype consistent with inputs
        Wt = process_weight.t().contiguous()

        # Output tensor for processed: [B, L, H], float32 accumulation is fine; match input dtype
        # Since A_cat is float32 by default (typical), we allocate as float32; if inputs are fp16, Triton will cast appropriately.
        processed = torch.empty((B, L, H), device=hidden.device, dtype=torch.float32)

        # Launch batched matmul kernel: C[b, m, n] = sum_k A_cat[b, m, k] * Wt[k, n]
        BLOCK_M = 64
        BLOCK_N_mat = 64
        BLOCK_K = 64
        grid_mat = (B, triton.cdiv(L, BLOCK_M), triton.cdiv(H, BLOCK_N_mat))
        batched_matmul_kernel_3d[grid_mat](
            A_cat, Wt, processed,
            B, L, H, H,  # M=L, N=H, K=H
            A_cat.stride(0), A_cat.stride(1), A_cat.stride(2),
            Wt.stride(0), Wt.stride(1), Wt.stride(2),
            processed.stride(0), processed.stride(1), processed.stride(2),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N_mat, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        # Split processed back into separate streams
        processed_encoder = processed[:, :T, :]
        processed_hidden = processed[:, T:, :]

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
