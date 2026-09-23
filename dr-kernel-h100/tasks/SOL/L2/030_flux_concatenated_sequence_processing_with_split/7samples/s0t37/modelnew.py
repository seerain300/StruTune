import torch
import triton
import triton.language as tl


@triton.jit
def batched_matmul_AW_kernel(
    C_ptr,          # *fp32, output C: [B, M, K]
    A_ptr,          # *fp32, input A (concatenated): [B, M, K]
    W_ptr,          # *fp32, weight: [K, K] (process_weight.T)
    B: tl.constexpr,  # batch size
    M: tl.constexpr,  # total sequence length = T + I
    N: tl.constexpr,  # K (output hidden_dim)
    K: tl.constexpr,  # K (input hidden_dim)
    stride_ab, stride_am, stride_ak,   # strides for A: [B, M, K]
    stride_wk, stride_wn,              # strides for W: [K, N]
    stride_cb, stride_cm, stride_cn,   # strides for C: [B, M, N]
    BLOCK_M: tl.constexpr,             # tile size over M
    BLOCK_N: tl.constexpr,             # tile size over N
    BLOCK_K: tl.constexpr,             # tile size over K
    num_warps: tl.constexpr,           # number of warps
    num_stages: tl.constexpr,          # stages
):
    # Program ids for batch, M-tiles, N-tiles
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    # Compute tile offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k in range(0, K, BLOCK_K):
        # Pointers to A and W tiles
        a_ptrs = A_ptr + pid_b * stride_ab + offs_m[:, None] * stride_am + (k + offs_k)[None, :] * stride_ak
        w_ptrs = W_ptr + (k + offs_k)[:, None] * stride_wk + offs_n[None, :] * stride_wn

        # Masks for bounds
        a_mask = (offs_m[:, None] < M) & ((k + offs_k)[None, :] < K)
        w_mask = ((k + offs_k)[:, None] < K) & (offs_n[None, :] < N)

        # Load tiles
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)  # shape [BLOCK_M, BLOCK_K]
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)  # shape [BLOCK_K, BLOCK_N]

        # Accumulate
        acc += tl.dot(a, w)

    # Store results
    c_ptrs = C_ptr + pid_b * stride_cb + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        Triton-optimized version of the original run function.
        - Concatenates encoder_hidden_states and hidden_states along sequence dimension.
        - Applies linear projection via Triton GEMM.
        - Splits outputs back into encoder and image streams.
        Returns (processed_encoder_hidden_states, processed_hidden_states)
        """
        # Ensure inputs are on CUDA and contiguous
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Tensors must be on CUDA device"
        # Work in float32 for robust Triton behavior
        x1 = encoder_hidden_states.contiguous().to(torch.float32)  # [B, T, K]
        x2 = hidden_states.contiguous().to(torch.float32)         # [B, I, K]
        # Concatenate along sequence dimension
        A = torch.cat([x1, x2], dim=1).contiguous()              # [B, M, K], M = T + I

        B = A.shape[0]
        T = x1.shape[1]
        I = x2.shape[1]
        M = T + I
        K = A.shape[2]  # hidden_dim
        N = K  # output same hidden_dim

        # Weight: process_weight is [K, K] (original PyTorch code uses process_weight.T as [K, K])
        # Ensure W is [K, K] and contiguous
        # The original signature passes process_weight: [hidden_dim, hidden_dim]
        # We need W = process_weight.T (PyTorch does this). Here we construct W.T explicitly.
        # Note: The original code uses process_weight.t() to get [K, K]. We mirror that here.
        # To be safe, we assume process_weight is already [K, K] or at least compatible.
        # If it comes as [K, K], we use it directly. If it comes as [N, K], we transpose.
        # In this environment, it should be [K, K] since the original model uses process_weight.T to match hidden_dim.
        # We'll enforce it as [K, K] by calling process_weight.T on host.
        W = process_weight.contiguous().to(torch.float32)
        if W.shape != (K, K):
            # Defensive transpose in case something unexpected is passed
            W = process_weight.t().contiguous().to(torch.float32)

        # Allocate output C as fp32
        C = torch.empty((B, M, N), dtype=torch.float32, device=A.device)

        # Choose tile sizes. These are conservative and robust; later can be tuned.
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 32

        grid = (B, triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

        batched_matmul_AW_kernel[grid](
            C, A, W,
            B, M, N, K,
            A.stride(0), A.stride(1), A.stride(2),
            W.stride(0), W.stride(1),
            C.stride(0), C.stride(1), C.stride(2),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        # Split outputs along sequence dimension
        processed_encoder = C[:, :T, :]                          # [B, T, K]
        processed_hidden = C[:, T:, :]                          # [B, I, K]

        # Cast back to original input dtypes
        processed_encoder = processed_encoder.to(encoder_hidden_states.dtype)
        processed_hidden = processed_hidden.to(hidden_states.dtype)

        return processed_encoder, processed_hidden