import torch
import triton
import triton.language as tl


@triton.jit
def batched_matmul_kernel(
    C_ptr,          # *fp32, output [B, M, N] where N=K
    A_ptr,          # *fp32, input [B, M, K]
    W_ptr,          # *fp32, weight [K, N] (note: N may be K, here N=K)
    B: tl.constexpr, # batch size
    M: tl.constexpr, # total sequence length = T + I
    K: tl.constexpr, # hidden_dim
    N: tl.constexpr, # output dim (here N=K)
    stride_cb, stride_cm, stride_cn,   # strides for C
    stride_ab, stride_am, stride_ak,   # strides for A
    stride_wk, stride_wn,              # strides for W
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Program ids
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    # Tile indices
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # rows (batch * M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # cols (N)
    offs_k = tl.arange(0, BLOCK_K)

    # Pointers for output C[b, m, n]
    C_offsets = (
        pid_b * stride_cb
        + offs_m[:, None] * stride_cm
        + offs_n[None, :] * stride_cn
    )

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K
    for k in range(0, K, BLOCK_K):
        k_idx = k + offs_k

        # Load A tiles: A[b, m, k]
        A_ptrs = (
            pid_b * stride_ab
            + offs_m[:, None] * stride_am
            + k_idx[None, :] * stride_ak
        )
        a_mask = (offs_m[:, None] < M) & (k_idx[None, :] < K)
        A_tile = tl.load(A_ptrs, mask=a_mask, other=0.0)  # [BM, BK]

        # Load W tiles: W[k, n]
        W_ptrs = (
            k_idx[:, None] * stride_wk
            + offs_n[None, :] * stride_wn
        )
        w_mask = (k_idx[:, None] < K) & (offs_n[None, :] < N)
        W_tile = tl.load(W_ptrs, mask=w_mask, other=0.0)  # [BK, BN]

        # Accumulate
        acc += tl.dot(A_tile, W_tile)

    # Store results
    C_ptrs = C_offsets
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(C_ptrs, acc, mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,          # [B, I, K]
        encoder_hidden_states: torch.Tensor,  # [B, T, K]
        process_weight: torch.Tensor,         # [K, K]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version of the original run function:
        - Concatenate along sequence dimension (torch.cat, lightweight).
        - Apply linear projection via Triton GEMM.
        - Split back into encoder and hidden outputs.
        """
        # Shapes
        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        K = hidden_states.shape[2]
        N = K  # process_weight is [K, K], so output dim equals hidden_dim

        # Concatenate along sequence dimension: A [B, T+I, K]
        # Use torch.cat for correctness; this is not heavy computation.
        A = torch.cat([encoder_hidden_states, hidden_states], dim=1)  # [B, M, K], M=T+I

        # Ensure all tensors are on CUDA and contiguous
        assert A.is_cuda and process_weight.is_cuda, "Inputs must be CUDA tensors for Triton kernels."

        # Work in float32 inside Triton kernels
        A = A.contiguous().to(torch.float32)
        process_weight_T = process_weight.t().contiguous().to(torch.float32)  # [K, K]

        # Allocate output C: [B, M, N], N=K
        C = torch.empty((B, T + I, K), dtype=torch.float32, device=A.device)

        # Launch Triton GEMM kernel
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 64
        grid = (B, triton.cdiv(T + I, BLOCK_M), triton.cdiv(K, BLOCK_N))
        batched_matmul_kernel[grid](
            C, A, process_weight_T,
            B, T + I, K, K,
            C.stride(0), C.stride(1), C.stride(2),
            A.stride(0), A.stride(1), A.stride(2),
            process_weight_T.stride(0), process_weight_T.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        # Split outputs back
        processed_encoder = C[:, :T, :]  # [B, T, K]
        processed_hidden = C[:, T:, :]   # [B, I, K]

        # Cast back to original dtypes
        processed_encoder = processed_encoder.to(encoder_hidden_states.dtype)
        processed_hidden = processed_hidden.to(hidden_states.dtype)

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
