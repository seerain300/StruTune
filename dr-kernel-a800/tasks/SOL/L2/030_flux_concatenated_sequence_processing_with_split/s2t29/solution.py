import torch
import triton
import triton.language as tl


@triton.jit
def _concatenate_seq_dim_kernel(
    E_ptr, H_ptr, O_ptr,
    B, T, I, H,
    stride_E_b, stride_E_t, stride_E_h,
    stride_H_b, stride_H_i, stride_H_h,
    stride_O_b, stride_O_l, stride_O_h,
    BLOCK_L: tl.constexpr, BLOCK_H: tl.constexpr,
):
    # Program IDs
    pid_b = tl.program_id(0)
    pid_l = tl.program_id(1)
    pid_h = tl.program_id(2)

    # Tile offsets
    offs_l = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)  # sequence positions (0..T+I-1)
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)  # hidden positions (0..H-1)

    # Masks
    L = T + I
    mask_l = offs_l < L
    mask_h = offs_h < H

    # For each element in this tile, decide source based on offs_l
    for i in range(BLOCK_L):
        m = offs_l[i]              # sequence index (0..L-1)
        valid_l = m < L
        for j in range(BLOCK_H):
            h = offs_h[j]          # hidden index (0..H-1)
            valid_h = h < H
            # Compute output pointer: O[b, m, h]
            O_addr = O_ptr + pid_b * stride_O_b + m * stride_O_l + h * stride_O_h
            # If m < T -> take from encoder, else take from hidden (m - T)
            E_addr = E_ptr + pid_b * stride_E_b + m * stride_E_t + h * stride_E_h
            H_addr = H_ptr + pid_b * stride_H_b + (m - T) * stride_H_i + h * stride_H_h
            # Select source based on valid_l
            val = tl.where(valid_l & valid_h, tl.where(m < T, tl.load(E_addr), tl.load(H_addr)), 0.0)
            tl.store(O_addr, val, mask=valid_l & valid_h)


@triton.jit
def _batched_gemm_kernel(
    A_ptr, WT_ptr, C_ptr,
    B, M, N, K,
    stride_A_b, stride_A_m, stride_A_k,
    stride_WT_k, stride_WT_n,
    stride_C_b, stride_C_m, stride_C_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Program IDs for batch, M-tiles, N-tiles
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    # Tile offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]

    # Masks for output tile bounds
    mask_m = offs_m < M
    mask_n = offs_n < N

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Reduction over K
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)  # [BLOCK_K]
        mask_k = offs_k < K

        # Load A tile: [BLOCK_M, BLOCK_K]
        A_ptrs = A_ptr + pid_b * stride_A_b + offs_m[:, None] * stride_A_m + offs_k[None, :] * stride_A_k
        A_tile = tl.load(A_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)

        # Load WT tile: [BLOCK_K, BLOCK_N]
        WT_ptrs = WT_ptr + offs_k[:, None] * stride_WT_k + offs_n[None, :] * stride_WT_n
        WT_tile = tl.load(WT_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)

        # Accumulate in fp32
        acc += tl.dot(A_tile.to(tl.float32), WT_tile.to(tl.float32))

    # Store results
    C_ptrs = C_ptr + pid_b * stride_C_b + offs_m[:, None] * stride_C_m + offs_n[None, :] * stride_C_n
    tl.store(C_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        hidden_states: torch.Tensor,                 # [B, I, H]
        encoder_hidden_states: torch.Tensor,        # [B, T, H]
        process_weight: torch.Tensor,               # [H, H]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized forward:
          - Concatenate encoder_hidden_states and hidden_states along sequence dimension in Triton.
          - Apply linear projection (matmul with process_weight.T) in Triton.
          - Split results back into separate streams.
        """
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Inputs must be on CUDA"

        # Ensure contiguous for predictable strides
        E = encoder_hidden_states.contiguous()  # [B, T, H]
        H = hidden_states.contiguous()          # [B, I, H]
        WT = process_weight.t().contiguous()    # [H, H]

        B, T, H_E = E.shape
        _, I, H_H = H.shape
        assert H_E == H_H, "Hidden dimension mismatch between encoder and hidden"
        H = H_E  # common H
        WT_K, WT_N = WT.shape
        assert WT_K == H and WT_N == H, "process_weight must be [H, H]"

        # 1) Concatenate along sequence dim in Triton: out_cat [B, L, H], L = T + I
        L = T + I
        out_cat = torch.empty((B, L, H), device=E.device, dtype=E.dtype)

        # Strides
        stride_E_b, stride_E_t, stride_E_h = E.stride(0), E.stride(1), E.stride(2)
        stride_H_b, stride_H_i, stride_H_h = H.stride(0), H.stride(1), H.stride(2)
        stride_O_b, stride_O_l, stride_O_h = out_cat.stride(0), out_cat.stride(1), out_cat.stride(2)

        # Tile sizes: small blocks to cover edge cases robustly
        BLOCK_L = 128
        BLOCK_H = 64

        grid_concat = (B, triton.cdiv(L, BLOCK_L), triton.cdiv(H, BLOCK_H))
        _concatenate_seq_dim_kernel[grid_concat](
            E, H, out_cat,
            B, T, I, H,
            stride_E_b, stride_E_t, stride_E_h,
            stride_H_b, stride_H_i, stride_H_h,
            stride_O_b, stride_O_l, stride_O_h,
            BLOCK_L=BLOCK_L, BLOCK_H=BLOCK_H,
            num_warps=4, num_stages=3,
        )

        # 2) Triton GEMM: processed = out_cat @ WT, shape [B, L, H]
        processed = torch.empty((B, L, H), device=E.device, dtype=torch.float32)

        # Strides
        stride_A_b, stride_A_m, stride_A_k = out_cat.stride(0), out_cat.stride(1), out_cat.stride(2)
        stride_WT_k, stride_WT_n = WT.stride(0), WT.stride(1)
        stride_C_b, stride_C_m, stride_C_n = processed.stride(0), processed.stride(1), processed.stride(2)

        # Tile sizes for GEMM
        BLOCK_M = 128
        BLOCK_N = 64
        BLOCK_K = 32

        grid_gemm = (B, triton.cdiv(L, BLOCK_M), triton.cdiv(H, BLOCK_N))
        _batched_gemm_kernel[grid_gemm](
            out_cat, WT, processed,
            B, L, H, H,  # M=L, N=H, K=H
            stride_A_b, stride_A_m, stride_A_k,
            stride_WT_k, stride_WT_n,
            stride_C_b, stride_C_m, stride_C_n,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=3,
        )

        # 3) Split streams
        processed_encoder = processed[:, :T, :]
        processed_hidden = processed[:, T:, :]

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
