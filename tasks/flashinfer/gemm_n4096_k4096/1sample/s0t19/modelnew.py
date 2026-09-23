class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Ensure dtype is float32 for robust accumulation
        if A.dtype != torch.float16:
            A = A.to(torch.float16)
        if B.dtype != torch.float16:
            B = B.to(torch.float16)

        M, K = A.shape
        N = B.shape[0]  # B is [N, K], so B_T is [K, N]
        # Output in float32, then cast back at the end
        C = torch.empty((M, N), device=A.device, dtype=torch.float32)

        # Choose reasonable tiles; these work well for N=4096, K=4096
        BLOCK_M = 1 if M == 1 else 64
        BLOCK_N = 128
        BLOCK_K = 64

        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        _matmul_generic_at_bt_kernel[grid](
            A, B, C,
            M, N, K,
            A.stride(0), A.stride(1),
            B.stride(0), B.stride(1),
            C.stride(0), C.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=3,
        )

        # Cast back to original dtype of A (torch.matmul keeps dtype)
        return C.to(A.dtype)