import torch
import triton
import triton.language as tl

# Triton kernel: compute C = A @ B.T
# A: [M, K], B: [K, N], output C: [M, N]
# We interpret B.T as [N, K] with strides (B.stride(1), B.stride(0)) for correct indexing.
@triton.jit
def matmul_transB_kernel_3d(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,        # A strides: row (m), col (k)
    stride_bn, stride_bk,        # B strides for B_T: B_T[n, k] => strides (B.stride(1), B.stride(0))
    stride_cm, stride_cn,        # C strides
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 3D launch grid over tiles of M, N, and K
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)

    # Compute tile offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K in chunks; this loop must be compile-time unrolled since BLOCK_K is constexpr
    # Triton requires tensor offsets; do not use bare Python variables in pointer arithmetic.
    # For each k-block:
    k0 = 0
    while k0 < K:
        # Compute current K offsets for this program
        # offs_k is vector [BLOCK_K]; use it for pointer arithmetic
        # Load A_sub = A[offs_m, offs_k] -> shape [BLOCK_M, BLOCK_K]
        A_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        A_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        A_sub = tl.load(A_ptrs, mask=A_mask, other=0.0)  # load A in native dtype (fp16), masked

        # Load B_sub_T = B_T[offs_n, offs_k] -> shape [BLOCK_N, BLOCK_K], where B_T[n, k] = B[n, k]
        B_ptrs = B_ptr + (offs_n[:, None] * stride_bn + offs_k[None, :] * stride_bk)
        B_mask = (offs_n[:, None] < N) & (offs_k[None, :] < K)
        B_sub_T = tl.load(B_ptrs, mask=B_mask, other=0.0)  # masked load

        # Cast to fp32 for dot-product accumulation
        A_sub_f32 = A_sub.to(tl.float32)
        B_sub_T_f32 = B_sub_T.to(tl.float32)

        # Compute partial product: A_sub [M,K] dot B_sub_T^T [K,N] => [M,N] partial for this tile
        # We need B_sub_T as [K,N], so transpose it
        # acc += tl.dot(A_sub_f32, tl.trans(B_sub_T_f32))
        acc += tl.dot(A_sub_f32, tl.trans(B_sub_T_f32))

        k0 += BLOCK_K

    # Store result back to C, casting to fp16
    C_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    C_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    # Store fp32 acc; Triton will cast to C dtype if needed. Ensure C is fp16.
    tl.store(C_ptrs, acc, mask=C_mask)


def triton_matmul_transB(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    # A: [M, K], B: [K, N]; return C: [M, N]
    assert A.is_cuda and B.is_cuda, "A and B must be on CUDA for Triton execution"
    M, K = A.shape
    Kb, N = B.shape
    assert K == Kb, f"Incompatible shapes: A is [M,{K}], B is [{Kb},{N}]"
    # Ensure contiguous for simpler stride usage
    A = A.contiguous()
    B = B.contiguous()

    # Output tensor (fp16 to match PyTorch default in provided get_inputs)
    C = torch.empty((M, N), device=A.device, dtype=torch.float16)

    # Strides in elements
    stride_am = A.stride(0)
    stride_ak = A.stride(1)
    # For B_T we use B strides: B_T[n, k] = B[n, k] with strides (B.stride(1), B.stride(0))
    stride_bn = B.stride(1)  # stride along N of B
    stride_bk = B.stride(0)  # stride along K of B
    stride_cm = C.stride(0)
    stride_cn = C.stride(1)

    # Tile sizes (balanced for fp16 GEMM)
    BLOCK_M = 64
    BLOCK_N = 128
    BLOCK_K = 64

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N), triton.cdiv(K, BLOCK_K))

    matmul_transB_kernel_3d[grid](
        A, B, C,
        M, N, K,
        stride_am, stride_ak,
        stride_bn, stride_bk,
        stride_cm, stride_cn,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=2,
    )
    return C


class ModelNew(torch.nn.Module):
    def forward(self, A, B):
        # Ensure tensors are on CUDA for Triton execution
        if not A.is_cuda:
            A = A.cuda(non_blocking=True)
        if not B.is_cuda:
            B = B.cuda(non_blocking=True)
        return triton_matmul_transB(A, B)


def run(*args):
    return ModelNew()(*args)
