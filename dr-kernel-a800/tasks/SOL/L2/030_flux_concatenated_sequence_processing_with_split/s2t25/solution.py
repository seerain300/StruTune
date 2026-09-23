import torch
import triton
import triton.language as tl


@triton.jit
def _batched_matmul_kernel(
    A_ptr, WT_ptr, C_ptr,
    B, L, H,
    A_stride_b, A_stride_m, A_stride_k,
    WT_stride_k, WT_stride_n,
    C_stride_b, C_stride_m, C_stride_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 3D launch: (batch, tiles along L (sequence), tiles along H (output features))
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    m0 = pid_m * BLOCK_M
    n0 = pid_n * BLOCK_N

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Reduction over K (hidden_dim)
    for k0 in range(0, H, BLOCK_K):
        # A tile: [BLOCK_M, BLOCK_K] at indices (m in [m0, m0+BLOCK_M), k in [k0, k0+BLOCK_K))
        A_tile_ptr = A_ptr + pid_b * A_stride_b + (m0 + tl.arange(0, BLOCK_M)) * A_stride_m + (k0 + tl.arange(0, BLOCK_K)) * A_stride_k
        m_idx = m0 + tl.arange(0, BLOCK_M)
        k_idx = k0 + tl.arange(0, BLOCK_K)
        A_mask = (m_idx[:, None] < L) & (k_idx[None, :] < H)
        A_tile = tl.load(A_tile_ptr, mask=A_mask, other=0.0)

        # WT tile: [BLOCK_K, BLOCK_N] at indices (k in [k0, k0+BLOCK_K), n in [n0, n0+BLOCK_N))
        WT_tile_ptr = WT_ptr + (k0 + tl.arange(0, BLOCK_K)) * WT_stride_k + (n0 + tl.arange(0, BLOCK_N)) * WT_stride_n
        k2 = k0 + tl.arange(0, BLOCK_K)
        n_idx = n0 + tl.arange(0, BLOCK_N)
        WT_mask = (k2[:, None] < H) & (n_idx[None, :] < H)
        WT_tile = tl.load(WT_tile_ptr, mask=WT_mask, other=0.0)

        # Accumulate in fp32
        acc += tl.dot(A_tile.to(tl.float32), WT_tile.to(tl.float32))

    # Store result to C at (b, m in [m0, m0+BLOCK_M), n in [n0, n0+BLOCK_N))
    C_tile_ptr = C_ptr + pid_b * C_stride_b + (m0 + tl.arange(0, BLOCK_M)) * C_stride_m + (n0 + tl.arange(0, BLOCK_N)) * C_stride_n
    out_mask = (m0 + tl.arange(0, BLOCK_M)) < L
    out_mask = out_mask[:, None]  # broadcast over N
    tl.store(C_tile_ptr, acc, mask=out_mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        hidden_states: [B, I, H]
        encoder_hidden_states: [B, T, H]
        process_weight: [H, H]
        Returns: (processed_encoder: [B, T, H], processed_hidden: [B, I, H])
        """
        assert hidden_states.dim() == 3 and encoder_hidden_states.dim() == 3, "Inputs must be [B, *, H]"
        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        H = hidden_states.shape[2]
        assert encoder_hidden_states.shape[2] == H, "Hidden dims must match"
        assert process_weight.shape[0] == H and process_weight.shape[1] == H, "process_weight must be [H, H]"

        device = hidden_states.device
        dtype = hidden_states.dtype

        # 1) Concatenate along sequence dimension: [B, L, H], L = T + I
        # Data movement; PyTorch implementation for robustness.
        L = T + I
        out_cat = torch.empty((B, L, H), device=device, dtype=dtype)
        out_cat[:, :T, :] = encoder_hidden_states
        out_cat[:, T:, :] = hidden_states

        # 2) Transpose process_weight to [H, H]
        WT = process_weight.t().contiguous()  # [H, H]

        # 3) Output buffer [B, L, H]
        processed = torch.empty((B, L, H), device=device, dtype=dtype)

        # 4) Launch Triton GEMM: processed = out_cat @ WT
        A = out_cat  # [B, L, H]
        WT_ = WT     # [H, H]

        # Strides
        A_stride_b, A_stride_m, A_stride_k = A.stride(0), A.stride(1), A.stride(2)
        WT_stride_k, WT_stride_n = WT_.stride(0), WT_.stride(1)
        C_stride_b, C_stride_m, C_stride_n = processed.stride(0), processed.stride(1), processed.stride(2)

        # Tile sizes (defaults good across the given workloads)
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 32

        grid = (B, triton.cdiv(L, BLOCK_M), triton.cdiv(H, BLOCK_N))
        _batched_matmul_kernel[grid](
            A, WT_, processed,
            B, L, H,
            A_stride_b, A_stride_m, A_stride_k,
            WT_stride_k, WT_stride_n,
            C_stride_b, C_stride_m, C_stride_n,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=3,
        )

        # 5) Split back into encoder and hidden streams
        processed_encoder = processed[:, :T, :]
        processed_hidden = processed[:, T:, :]

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
