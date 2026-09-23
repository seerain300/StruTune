import torch
import triton
import triton.language as tl


@triton.jit
def cat_seq_kernel(
    encoder_ptr, hidden_ptr, out_ptr,
    B, T, I, H,
    encoder_stride_b, encoder_stride_t, encoder_stride_h,
    hidden_stride_b, hidden_stride_i, hidden_stride_h,
    out_stride_b, out_stride_l, out_stride_h,
):
    # Grid: (B, L, H)
    b = tl.program_id(0)
    l = tl.program_id(1)
    h = tl.program_id(2)

    # Bounds: we assume grid matches, but keep guards simple
    if l < (T + I):
        if l < T:
            src = encoder_ptr + b * encoder_stride_b + l * encoder_stride_t + h * encoder_stride_h
        else:
            src = hidden_ptr + b * hidden_stride_b + (l - T) * hidden_stride_i + h * hidden_stride_h
        dst = out_ptr + b * out_stride_b + l * out_stride_l + h * out_stride_h
        val = tl.load(src)
        tl.store(dst, val)


@triton.jit
def batched_matmul_kernel_fp32_accum(
    A_ptr, B_ptr, C_ptr,
    B, M, N, K,
    A_stride_b, A_stride_m, A_stride_k,   # A is [B, M, K] here = [B, L, H]
    B_stride_k, B_stride_n,               # B_ptr is weight.T [K, N] i.e., [H, H]
    C_stride_b, C_stride_m, C_stride_n,   # C is [B, M, N] = [B, L, H]
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 3D grid: (B, tiles over M, tiles over N)
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N

    # Initialize accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K (reduction) dimension in tiles
    for k0 in range(0, K, BLOCK_K):
        k_start = k0

        # Compute tile offsets
        m_offsets = m_start + tl.arange(0, BLOCK_M)
        n_offsets = n_start + tl.arange(0, BLOCK_N)
        k_offsets = k_start + tl.arange(0, BLOCK_K)

        # Masks for bounds
        m_mask = m_offsets < M
        n_mask = n_offsets < N

        # Load A tile: shape [BLOCK_M, BLOCK_K], cast to fp32 for accumulation
        A_tile_ptr = A_ptr + pid_b * A_stride_b + m_offsets[:, None] * A_stride_m + k_offsets[None, :] * A_stride_k
        A_mask = m_mask[:, None] & (k_offsets[None, :] < K)
        A_tile = tl.load(A_tile_ptr, mask=A_mask, other=0.0).to(tl.float32)

        # Load B tile: shape [BLOCK_K, BLOCK_N] from weight.T [K, N], cast to fp32
        B_tile_ptr = B_ptr + k_offsets[:, None] * B_stride_k + n_offsets[None, :] * B_stride_n
        B_mask = (k_offsets[:, None] < K) & n_mask[None, :]
        B_tile = tl.load(B_tile_ptr, mask=B_mask, other=0.0).to(tl.float32)

        # Accumulate
        acc += tl.dot(A_tile, B_tile)

    # Write back to C in original dtype (assumed to match A's last-dim dtype); here we store fp32
    # Note: If C is not fp32, you may need to cast; in this task, inputs are typically fp32.
    C_tile_ptr = C_ptr + pid_b * C_stride_b + m_offsets[:, None] * C_stride_m + n_offsets[None, :] * C_stride_n
    C_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(C_tile_ptr, acc, mask=C_mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        Triton-optimized forward:
        - Concatenate along sequence dimension in Triton
        - Apply linear projection (matmul) in Triton using fp32 accumulation
        - Split back into separate encoder and image streams
        Returns: (processed_encoder_hidden_states, processed_hidden_states)
        """
        # Ensure contiguity
        encoder = encoder_hidden_states.contiguous()
        hidden = hidden_states.contiguous()
        weight = process_weight.contiguous()

        B = encoder.shape[0]
        T = encoder.shape[1]
        I = hidden.shape[1]
        H = encoder.shape[2]  # hidden_dim
        L = T + I

        # 1) Triton concatenation into out_cat [B, L, H]
        out_cat = torch.empty((B, L, H), device=encoder.device, dtype=encoder.dtype)
        grid_cat = (B, L, H)
        cat_seq_kernel[grid_cat](
            encoder, hidden, out_cat,
            B, T, I, H,
            encoder.stride(0), encoder.stride(1), encoder.stride(2),
            hidden.stride(0), hidden.stride(1), hidden.stride(2),
            out_cat.stride(0), out_cat.stride(1), out_cat.stride(2),
            num_warps=1, num_stages=1
        )

        # 2) Triton matmul: processed = out_cat @ weight.T using fp32 accumulation
        # Prepare weight.T
        WT = weight.t().contiguous()  # shape [H, H], dtype same as weight
        processed = torch.empty((B, L, H), device=encoder.device, dtype=encoder.dtype)  # keep same dtype as inputs

        # Strides for A=[B, L, H] and B=WT=[H, H]
        A_stride_b = out_cat.stride(0)
        A_stride_m = out_cat.stride(1)  # along L
        A_stride_k = out_cat.stride(2)  # along H (reduction dim)

        # B strides for weight.T [K=H, N=H]
        B_stride_k = WT.stride(0)  # along H (rows of WT)
        B_stride_n = WT.stride(1)  # along H (cols of WT)

        C_stride_b = processed.stride(0)
        C_stride_m = processed.stride(1)  # along L
        C_stride_n = processed.stride(2)  # along H

        # Choose tiling for robust performance across wide dims
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 64

        grid_m = triton.cdiv(L, BLOCK_M)  # tiles over sequence length
        grid_n = triton.cdiv(H, BLOCK_N)  # tiles over hidden dim

        batched_matmul_kernel_fp32_accum[(B, grid_m, grid_n)](
            out_cat, WT, processed,
            B, L, H, H,  # M=L, N=H, K=H
            A_stride_b, A_stride_m, A_stride_k,
            B_stride_k, B_stride_n,
            C_stride_b, C_stride_m, C_stride_n,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2
        )

        # 3) Split back into separate streams
        processed_encoder = processed[:, :T, :]
        processed_hidden = processed[:, T:, :]

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
