import torch
import triton
import triton.language as tl


@triton.jit
def _matmul_seq_to_dim_kernel(
    X_ptr,         # [B, M, K] input, e.g., encoder_hidden_states
    W_ptr,         # [K, N] process_weight, K=input hidden_dim, N=output hidden_dim
    Y_ptr,         # [B, M, N] output
    B: tl.constexpr,  # batch size
    M: tl.constexpr,  # sequence length (rows of X)
    N: tl.constexpr,  # output hidden_dim
    K: tl.constexpr,  # input hidden_dim
    sX_b: tl.constexpr,
    sX_m: tl.constexpr,
    sX_k: tl.constexpr,
    sW0: tl.constexpr,  # stride along K for W (rows)
    sW1: tl.constexpr,  # stride along N for W (cols)
    sY_b: tl.constexpr,
    sY_m: tl.constexpr,
    sY_n: tl.constexpr,
    BLOCK_M: tl.constexpr,  # tile size along M
    BLOCK_N: tl.constexpr,  # tile size along N (output dim)
    BLOCK_K: tl.constexpr,  # tile size along K
    NUM_WARPS: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    # 2D grid: (tiles along M, batch)
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    # Offsets for this tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)

        # Load X tile: shape (BLOCK_M, BLOCK_K)
        x_ptrs = X_ptr + pid_b * sX_b + offs_m[:, None] * sX_m + offs_k[None, :] * sX_k
        x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)

        # Load W tile: shape (BLOCK_K, BLOCK_N)
        w_ptrs = W_ptr + offs_k[:, None] * sW0 + offs_n[None, :] * sW1
        w_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)

        # Accumulate: (BLOCK_M, BLOCK_K) @ (BLOCK_K, BLOCK_N) -> (BLOCK_M, BLOCK_N)
        # Cast to fp32 for accumulation stability, then store
        acc += tl.dot(x_tile.to(tl.float32), w_tile.to(tl.float32))

    # Write back Y
    y_ptrs = Y_ptr + pid_b * sY_b + offs_m[:, None] * sY_m + offs_n[None, :] * sY_n
    y_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(y_ptrs, acc, mask=y_mask)


@triton.jit
def _matmul_img_to_dim_kernel(
    X_ptr,         # [B, I, K] input, e.g., hidden_states
    W_ptr,         # [K, N] process_weight
    Y_ptr,         # [B, I, N] output
    B: tl.constexpr,
    I: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    sX_b: tl.constexpr,
    sX_i: tl.constexpr,
    sX_k: tl.constexpr,
    sW0: tl.constexpr,
    sW1: tl.constexpr,
    sY_b: tl.constexpr,
    sY_i: tl.constexpr,
    sY_n: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_WARPS: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    # 2D grid: (tiles along I, batch)
    pid_i = tl.program_id(0)
    pid_b = tl.program_id(1)
    offs_i = pid_i * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)

        x_ptrs = X_ptr + pid_b * sX_b + offs_i[:, None] * sX_i + offs_k[None, :] * sX_k
        x_mask = (offs_i[:, None] < I) & (offs_k[None, :] < K)
        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)

        w_ptrs = W_ptr + offs_k[:, None] * sW0 + offs_n[None, :] * sW1
        w_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)

        acc += tl.dot(x_tile.to(tl.float32), w_tile.to(tl.float32))

    y_ptrs = Y_ptr + pid_b * sY_b + offs_i[:, None] * sY_i + offs_n[None, :] * sY_n
    y_mask = (offs_i[:, None] < I) & (offs_n[None, :] < N)
    tl.store(y_ptrs, acc, mask=y_mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                encoder_hidden_states: torch.Tensor,
                process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version of the original run function.
        Computes:
          processed_encoder = encoder_hidden_states @ process_weight.T
          processed_hidden   = hidden_states        @ process_weight.T
        without concatenating the sequences.
        """
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, \
            "All inputs must be CUDA tensors for Triton execution."

        # Make inputs contiguous for predictable strides
        ehs = encoder_hidden_states.contiguous()
        hs = hidden_states.contiguous()
        w = process_weight.contiguous()  # [D, D]

        B = ehs.shape[0]
        D = ehs.shape[2]  # input hidden_dim
        N = w.shape[1]    # output hidden_dim (same as w.shape[1] == D if square)

        # We assume process_weight is square [D, D], no bias.
        # Output tensors
        processed_encoder = torch.empty((B, ehs.shape[1], N), device=ehs.device, dtype=ehs.dtype)
        processed_hidden = torch.empty((B, hs.shape[1], N), device=hs.device, dtype=hs.dtype)

        # Choose tile sizes. Reasonable defaults for many sizes.
        # These can be tuned; we pick moderate sizes to balance occupancy and memory bandwidth.
        BLOCK_M = 128
        BLOCK_N = 64
        BLOCK_K = 64
        NUM_WARPS = 4
        NUM_STAGES = 2

        # Launch kernel for encoder_hidden_states @ W.T -> [B, T, D]
        grid_e = (triton.cdiv(ehs.shape[1], BLOCK_M), B)
        _matmul_seq_to_dim_kernel[grid_e](
            ehs, w, processed_encoder,
            B, ehs.shape[1], N, D,
            ehs.stride(0), ehs.stride(1), ehs.stride(2),
            w.stride(0), w.stride(1),
            processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            NUM_WARPS=NUM_WARPS, NUM_STAGES=NUM_STAGES,
        )

        # Launch kernel for hidden_states @ W.T -> [B, I, D]
        grid_h = (triton.cdiv(hs.shape[1], BLOCK_M), B)
        _matmul_img_to_dim_kernel[grid_h](
            hs, w, processed_hidden,
            B, hs.shape[1], N, D,
            hs.stride(0), hs.stride(1), hs.stride(2),
            w.stride(0), w.stride(1),
            processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            NUM_WARPS=NUM_WARPS, NUM_STAGES=NUM_STAGES,
        )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
