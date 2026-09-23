import torch
import triton
import triton.language as tl


@triton.jit
def cat_seq_kernel(
    encoder_ptr, hidden_ptr, out_ptr,
    B, T, I, H,
    encoder_stride_b, encoder_stride_t, encoder_stride_h,
    hidden_stride_b, hidden_stride_i, hidden_stride_h,
    out_stride_b, out_stride_m, out_stride_h,
    BLOCK_M: tl.constexpr,  # tile size over sequence length (L)
    BLOCK_K: tl.constexpr,  # tile size over hidden dim (H)
):
    # Grid: (batch, tiles over sequence length, tiles over hidden dim)
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_k = tl.program_id(2)

    L = T + I

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    k_offsets = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)

    mask_m = m_offsets < L
    mask_k = k_offsets < H

    # Determine source: first T rows come from encoder, remaining I rows from hidden
    is_encoder = m_offsets < T

    # Base pointers for this batch
    encoder_base = encoder_ptr + pid_b * encoder_stride_b
    hidden_base = hidden_ptr + pid_b * hidden_stride_b
    out_base = out_ptr + pid_b * out_stride_b

    # Load from encoder or hidden, depending on is_encoder
    # For encoder rows: out[b, m, k] = encoder[b, m, k]
    # For hidden rows: out[b, m, k] = hidden[b, m - T, k]
    if is_encoder:
        # When is_encoder is True, m_offsets < T, so m_offsets are valid
        encoder_addrs = encoder_base + m_offsets[:, None] * encoder_stride_t + k_offsets[None, :] * encoder_stride_h
        vals = tl.load(encoder_addrs, mask=mask_m[:, None] & mask_k[None, :], other=0)
    else:
        hidden_m = m_offsets - T
        # Since is_encoder excludes m_offsets >= T, hidden_m will be < I, but we still guard
        hidden_addrs = hidden_base + hidden_m[:, None] * hidden_stride_i + k_offsets[None, :] * hidden_stride_h
        vals = tl.load(hidden_addrs, mask=mask_m[:, None] & mask_k[None, :], other=0)

    # Store into out[b, m, k]
    out_addrs = out_base + m_offsets[:, None] * out_stride_m + k_offsets[None, :] * out_stride_h
    tl.store(out_addrs, vals, mask=mask_m[:, None] & mask_k[None, :])


@triton.jit
def batched_matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    A_stride_m, A_stride_k,
    B_stride_k, B_stride_n,
    C_stride_m, C_stride_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Grid: (tiles over M, tiles over N). Since we call per-batch, we launch once for a given A_ptr/B_ptr/C_ptr.
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)

        a_ptrs = A_ptr + m_offsets[:, None] * A_stride_m + k_offsets[None, :] * A_stride_k
        b_ptrs = B_ptr + k_offsets[:, None] * B_stride_k + n_offsets[None, :] * B_stride_n

        a_mask = (m_offsets[:, None] < M) & (k_offsets[None, :] < K)
        b_mask = (k_offsets[:, None] < K) & (n_offsets[None, :] < N)

        a = tl.load(a_ptrs, mask=a_mask, other=0).to(tl.float32)
        b = tl.load(b_ptrs, mask=b_mask, other=0).to(tl.float32)

        acc += tl.dot(a, b)

    c_ptrs = C_ptr + m_offsets[:, None] * C_stride_m + n_offsets[None, :] * C_stride_n
    c_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                encoder_hidden_states: torch.Tensor,
                process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version of the original run:
        1) Concatenate encoder_hidden_states and hidden_states along sequence dimension using Triton.
        2) Apply linear projection (A_cat @ process_weight.T) using Triton (via tl.dot).
        3) Split back into processed_encoder and processed_hidden.

        Returns:
            processed_encoder: [B, T, H]
            processed_hidden: [B, I, H]
        """
        # Ensure tensors are on CUDA and contiguous
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, \
            "All tensors must be on CUDA for Triton kernels."
        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        H = hidden_states.shape[2]
        assert encoder_hidden_states.shape[2] == H, "Hidden dims must match."
        assert process_weight.shape[0] == H and process_weight.shape[1] == H, "process_weight must be [H, H]."

        # Make contiguous
        encoder = encoder_hidden_states.contiguous()
        hidden = hidden_states.contiguous()
        W = process_weight.contiguous()

        device = hidden.device
        dtype = hidden.dtype

        # 1) Concatenate along sequence dimension using Triton
        L = T + I
        A_cat = torch.empty((B, L, H), device=device, dtype=dtype)

        BLOCK_M = 128  # tile over sequence length
        BLOCK_K = 64   # tile over hidden dim
        grid_cat = (B, triton.cdiv(L, BLOCK_M), triton.cdiv(H, BLOCK_K))
        cat_seq_kernel[grid_cat](
            encoder, hidden, A_cat,
            B, T, I, H,
            encoder.stride(0), encoder.stride(1), encoder.stride(2),
            hidden.stride(0), hidden.stride(1), hidden.stride(2),
            A_cat.stride(0), A_cat.stride(1), A_cat.stride(2),
            BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        # 2) Linear projection: A_cat @ process_weight.T using Triton matmul
        # Transpose process_weight to [H, H] for Triton (Wt[k, n] = process_weight[n, k])
        Wt = process_weight.t().contiguous()  # [H, H]

        # Allocate output processed [B, L, H] with float32 accumulation
        processed = torch.empty((B, L, H), device=device, dtype=torch.float32)

        # For each batch, compute A_cat[b] @ Wt -> [L, H]
        for b in range(B):
            A_b = A_cat[b]  # [L, H]
            C_b = processed[b]  # [L, H], float32
            # Choose tile sizes; H is often moderate (e.g., 64, 128, 256). We set 128x128x64.
            BLOCK_M_mm = 128
            BLOCK_N_mm = 128
            BLOCK_K_mm = 64
            grid_mm = (triton.cdiv(L, BLOCK_M_mm), triton.cdiv(H, BLOCK_N_mm))
            batched_matmul_kernel[grid_mm](
                A_b, Wt, C_b,
                L, H, H,  # M=L, N=H, K=H
                A_b.stride(0), A_b.stride(1),  # [L, H]
                Wt.stride(0), Wt.stride(1),    # [H, H]
                C_b.stride(0), C_b.stride(1),  # [L, H]
                BLOCK_M=BLOCK_M_mm, BLOCK_N=BLOCK_N_mm, BLOCK_K=BLOCK_K_mm,
                num_warps=4, num_stages=2,
            )

        # Cast back to original dtype for final outputs
        processed = processed.to(dtype)

        # 3) Split back into separate streams
        processed_encoder = processed[:, :T, :]
        processed_hidden = processed[:, T:, :]

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
