import torch
import triton
import triton.language as tl


@triton.jit
def batched_matmul_kernel(
    C_ptr,  # *fp32, output [B, M, K]
    A_ptr,  # *fp32, input [B, M, K] (concatenated)
    W_ptr,  # *fp32, weight [K, K] (process_weight.T)
    B: tl.constexpr,  # batch size
    M: tl.constexpr,  # total sequence length = T + I
    K: tl.constexpr,  # hidden dimension
    stride_ab, stride_am, stride_ak,   # strides for A: (B, M, K)
    stride_wk, stride_wn,              # strides for W: (K, K)
    stride_cb, stride_cm, stride_cn,   # strides for C: (B, M, K)
    BLOCK_M: tl.constexpr,             # tile size along M
    BLOCK_N: tl.constexpr,             # tile size along K (output N)
    BLOCK_K: tl.constexpr,             # tile size along K loop
    num_warps: tl.constexpr,
    num_stages: tl.constexpr,
):
    # Grid: (B, ceil_div(M, BLOCK_M), ceil_div(K, BLOCK_N))
    b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k_start in range(0, K, BLOCK_K):
        k_idx = k_start + offs_k  # [BLOCK_K]

        # Pointers for A tile: A[b, m, k]
        a_ptrs = A_ptr + b * stride_ab + offs_m[:, None] * stride_am + k_idx[None, :] * stride_ak
        a_mask = (offs_m[:, None] < M) & (k_idx[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)  # [BLOCK_M, BLOCK_K]

        # Pointers for W tile: W[k, n]
        w_ptrs = W_ptr + k_idx[:, None] * stride_wk + offs_n[None, :] * stride_wn
        w_mask = (k_idx[:, None] < K) & (offs_n[None, :] < K)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)  # [BLOCK_K, BLOCK_N]

        # Accumulate
        acc += tl.dot(a, w)

    # Store results to C[b, m, n]
    c_ptrs = C_ptr + b * stride_cb + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < K)
    tl.store(c_ptrs, acc, mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,           # [B, I, K]
        encoder_hidden_states: torch.Tensor,   # [B, T, K]
        process_weight: torch.Tensor,          # [K, K]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version:
        - Concatenate [B, T, K] and [B, I, K] along sequence dim to [B, T+I, K]
        - Compute C = concatenated @ process_weight.T using Triton GEMM
        - Split C back into [B, T, K] and [B, I, K]
        """
        # Ensure tensors are on CUDA and contiguous
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Triton requires CUDA tensors"
        device = hidden_states.device

        # 1) Concatenate along sequence dimension: [B, T+I, K]
        # Keep tensors in their original dtype for concat; we will cast for matmul
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        M = T + I
        K = encoder_hidden_states.shape[2]
        assert hidden_states.shape[2] == K and encoder_hidden_states.shape[2] == K and process_weight.shape[0] == K and process_weight.shape[1] == K, "Dimension mismatch"

        # Make contiguous
        x1 = encoder_hidden_states.contiguous()
        x2 = hidden_states.contiguous()
        # Concatenate on host using torch for robustness
        A = torch.cat([x1, x2], dim=1)  # [B, M, K]

        # 2) Prepare weight as [K, K] (process_weight.T should be [K, K])
        # Ensure float32 for stable GEMM and explicit strides
        W = process_weight  # [K, K]
        if W.dtype != torch.float32:
            W = W.float()
        W = W.contiguous()

        # 3) Allocate output C [B, M, K] in float32
        C = torch.empty((A.shape[0], A.shape[1], A.shape[2]), dtype=torch.float32, device=device)

        # 4) Launch Triton GEMM kernel
        # Choose conservative tile sizes; adjust as needed for performance later
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 64
        grid = (A.shape[0], triton.cdiv(M, BLOCK_M), triton.cdiv(K, BLOCK_N))
        batched_matmul_kernel[grid](
            C, A, W,
            B=A.shape[0], M=M, K=K,
            stride_ab=A.stride(0), stride_am=A.stride(1), stride_ak=A.stride(2),
            stride_wk=W.stride(0), stride_wn=W.stride(1),
            stride_cb=C.stride(0), stride_cm=C.stride(1), stride_cn=C.stride(2),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        # 5) Split outputs back into encoder and hidden streams
        processed_encoder = C[:, :T, :]  # [B, T, K]
        processed_hidden = C[:, T:, :]   # [B, I, K]

        # Cast back to original dtypes to match original behavior
        processed_encoder = processed_encoder.to(encoder_hidden_states.dtype)
        processed_hidden = processed_hidden.to(hidden_states.dtype)

        return processed_encoder, processed_hidden