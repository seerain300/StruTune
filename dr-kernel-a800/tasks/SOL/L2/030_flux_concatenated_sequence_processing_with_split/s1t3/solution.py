import torch
import triton
import triton.language as tl


@triton.jit
def _matmul_flat_kernel(
    A_ptr,  # *f32, [M, K], row-major
    WT_ptr, # *f32, [K, K], row-major (process_weight.T)
    C_ptr,  # *f32, [M, K], row-major
    M, N, K,
    stride_am, stride_ak,
    stride_wtk, stride_wtk2,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """
    Compute C = A @ WT, where:
      A: [M, K], M = B*(T+P), K = hidden_dim
      WT: [K, K], WT = process_weight.T
      C: [M, K]
    Tiling over (M, K) with reduction over K.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)         # rows in M
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)         # cols in K (output)
    offs_k = tl.arange(0, BLOCK_K)                           # reduction chunk

    # Tile pointers for A and C
    A_tile_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)  # [BLOCK_M, BLOCK_K]
    C_tile_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)  # [BLOCK_M, BLOCK_N]

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over reduction dimension K
    for k in range(0, K, BLOCK_K):
        # WT tile: WT is [K, K], so row index is (k + offs_k), col index is offs_n
        WT_ptrs = WT_ptr + ((k + offs_k)[:, None] * stride_wtk + offs_n[None, :] * stride_wtk2)  # [BLOCK_K, BLOCK_N]
        # Masked loads
        a = tl.load(A_tile_ptrs, mask=(offs_m[:, None] < M) & ((k + offs_k)[None, :] < K), other=0.0)
        wt = tl.load(WT_ptrs, mask=((k + offs_k)[:, None] < K), other=0.0)
        # Accumulate
        acc += tl.dot(a, wt)

        # Advance A tile pointers along K
        A_tile_ptrs += BLOCK_K * stride_ak

    # Store result
    tl.store(C_tile_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                encoder_hidden_states: torch.Tensor,
                process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-only compute for the GEMM:
          - Concatenate encoder and hidden states along sequence dimension using torch.cat (data movement).
          - Perform linear projection (Acat @ process_weight.T) via Triton matmul kernel.
          - Split outputs back into encoder and hidden streams using PyTorch slicing.
        """
        # Ensure tensors are on CUDA device
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, \
            "ModelNew requires CUDA tensors"

        # Make tensors contiguous and float32
        E = encoder_hidden_states.contiguous()
        H = hidden_states.contiguous()
        W = process_weight.contiguous()

        # 1) Concatenate along sequence dim: [B, T+P, K]
        total_L = E.shape[1] + H.shape[1]
        Acat = torch.cat([E, H], dim=1).contiguous()  # [B, T+P, K]

        B = Acat.shape[0]
        T = E.shape[1]
        P = H.shape[1]
        K = Acat.shape[2]
        M = B * total_L

        # 2) Flatten Acat to [M, K] and prepare WT = W.T [K, K]
        A_flat = Acat.view(M, K).contiguous()
        WT = W.t().contiguous()  # [K, K]

        # Allocate output flat
        C_flat = torch.empty((M, K), dtype=torch.float32, device=A_flat.device)

        # 3) Launch Triton matmul kernel
        BLOCK_M = 128
        BLOCK_N = 64
        BLOCK_K = 32
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(K, BLOCK_N))
        _matmul_flat_kernel[grid](
            A_flat, WT, C_flat,
            M, K, K,
            A_flat.stride(0), A_flat.stride(1),
            WT.stride(0), WT.stride(1),
            C_flat.stride(0), C_flat.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2
        )

        # 4) Reshape to [B, T+P, K]
        C = C_flat.view(B, total_L, K)

        # 5) Split outputs
        processed_encoder = C[:, :T, :]
        processed_hidden = C[:, T:, :]

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
