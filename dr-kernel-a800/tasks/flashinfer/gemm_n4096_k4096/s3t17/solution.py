import torch
import triton
import triton.language as tl


@triton.jit
def matmul_at_bT_kernel(
    A, BT, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,  # BT is B.T contiguous; strides are (stride_bn=N, stride_bk=1) for contiguous BT
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D tiling over output C[m, n]
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K in chunks
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)

        # A tile: A[m, k]
        a_ptrs = A + (offs_m[:, None] * stride_am) + (offs_k[None, :] * stride_ak)
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0).to(tl.float32)

        # BT tile: BT[k, n] (B.T contiguous has shape [K, N])
        bt_ptrs = BT + (offs_k[:, None] * stride_bn) + (offs_n[None, :] * stride_bk)
        bt_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        bt = tl.load(bt_ptrs, mask=bt_mask, other=0.0).to(tl.float32)

        acc += tl.dot(a, bt)

    c_ptrs = C + (offs_m[:, None] * stride_cm) + (offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)  # Triton will cast to C's dtype if needed


@triton.jit
def row_matmul_kernel(
    A_row, BT_flat, C_row,
    M, N, K,  # M is 1 here, but passed for generality
    stride_am, stride_ak,        # A_row strides: [M, K] but M==1, so stride_am=0, stride_ak=1 in contiguous case
    stride_bt,                   # BT_flat stride: typically 1 if BT contiguous
    stride_cm,                   # C_row stride over columns
    BLOCK_N: tl.constexpr,       # columns tile
    BLOCK_K: tl.constexpr,       # K tile
):
    # Single-row output: C_row[0, n]
    offs_n = tl.arange(0, BLOCK_N)
    pid_n = tl.program_id(0)
    n_start = pid_n * BLOCK_N

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # Iterate over K in chunks
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        # A_row[0, k] load; since M==1, A_row pointer is base + k*stride_ak
        a_ptrs = A_row + offs_k * stride_ak
        a_mask = offs_k < K
        a = tl.load(a_ptrs, mask=a_mask, other=0.0).to(tl.float32)

        # BT_flat[k * N + n] = B.T[k, n]
        bt_ptrs = BT_flat + (offs_k[:, None] * N) + (n_start + offs_n[None, :])
        bt_mask = (offs_k[:, None] < K) & ((n_start + offs_n[None, :]) < N)
        bt = tl.load(bt_ptrs, mask=bt_mask, other=0.0).to(tl.float32)

        # Accumulate dot per column
        acc += tl.sum(bt * a[:, None], axis=0)

    # Store to C_row[0, n]
    c_ptrs = C_row + (n_start + offs_n) * stride_cm
    c_mask = (n_start + offs_n) < N
    tl.store(c_ptrs, acc, mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(self, A, B):
        # C = A @ B.T, A: [M, K], B: [N, K], output C: [M, N]
        assert A.is_cuda and B.is_cuda, "Inputs must be CUDA tensors for Triton execution."
        M, K = A.shape
        N, K_b = B.shape
        assert K == K_b, "B's second dimension must match A's second dimension."

        # Prepare BT = B.T contiguous for simpler addressing
        BT = B.transpose(0, 1).contiguous()  # shape [K, N]

        # Output tensor in float16 to match original setup
        out = torch.empty((M, N), device=A.device, dtype=torch.float16)

        # Specialized fast path for M == 1
        if M == 1:
            A_row = A[0]  # [K], float16
            BT_flat = BT.view(-1).contiguous()  # [K*N], float16
            C_row = out[0]  # [N], float16

            # Choose tiling for columns and K
            BLOCK_N = 256
            BLOCK_K = 64

            grid = (triton.cdiv(N, BLOCK_N),)

            row_matmul_kernel[grid](
                A_row, BT_flat, C_row,
                M, N, K,
                A_row.stride(0), A_row.stride(1),  # for M==1 contiguous row, stride(1)==1, stride(0)==K
                N,                                           # stride for BT_flat elements when viewing [K, N] flattened
                C_row.stride(0),                             # typically 1
                BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                num_warps=2, num_stages=3,
            )
            return out

        # General 2D kernel path for M > 1
        BLOCK_M = 128
        BLOCK_N = 128
        BLOCK_K = 64

        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

        matmul_at_bT_kernel[grid](
            A, BT, out,
            M, N, K,
            A.stride(0), A.stride(1),
            BT.stride(0), BT.stride(1),
            out.stride(0), out.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=8, num_stages=4,
        )

        return out


def run(*args):
    return ModelNew()(*args)
