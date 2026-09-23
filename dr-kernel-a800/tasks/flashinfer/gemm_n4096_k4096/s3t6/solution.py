import torch
import triton
import triton.language as tl


@triton.jit
def row_matmul_at_bT_kernel(
    A_ptr, BT_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,       # A strides: A is (M, K)
    stride_bk, stride_bn,       # BT strides: BT is (K, N)
    stride_cm, stride_cn,       # C strides: C is (M, N)
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # One program per output row m
    m = tl.program_id(0)

    # If m >= M, exit (masking in store ensures correctness, but we can early return)
    if m >= M:
        return

    # Loop over N in tiles
    offs_n_base = tl.arange(0, BLOCK_N)  # N tile offset template
    for n0 in range(0, N, BLOCK_N):
        # Accumulator for this N tile: vector of size BLOCK_N in fp32
        acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

        # Loop over K in chunks
        offs_k = tl.arange(0, BLOCK_K)
        for k0 in range(0, K, BLOCK_K):
            # Load A row slice: vector of length BLOCK_K
            A_row_ptrs = A_ptr + (m * stride_am + (k0 + offs_k) * stride_ak)
            a_mask = (m < M) & (k0 + offs_k < K)
            a1 = tl.load(A_row_ptrs, mask=a_mask, other=0.0)  # shape (BLOCK_K,)

            # Load BT tile: shape (BLOCK_K, BLOCK_N), columns n0:n0+BLOCK_N
            BT_ptrs = BT_ptr + ((k0 + offs_k)[:, None] * stride_bk + (n0 + offs_n_base[None, :]) * stride_bn)
            bt_mask = (k0 + offs_k)[:, None] < K & (n0 + offs_n_base[None, :]) < N
            bt = tl.load(BT_ptrs, mask=bt_mask, other=0.0)  # shape (BLOCK_K, BLOCK_N)

            # Accumulate: acc += sum_k a1[k] * bt[k, :]
            # Equivalent to a1[:, None] * bt and reduce along axis 0.
            acc += tl.sum(a1[:, None] * bt, axis=0)

        # Store results for this N tile
        C_ptrs = C_ptr + (m * stride_cm + (n0 + offs_n_base) * stride_cn)
        out_mask = (m < M) & (n0 + offs_n_base < N)
        tl.store(C_ptrs, acc, mask=out_mask)


@triton.jit
def matmul_at_bT_kernel_2d(
    A_ptr, BT_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,       # A strides: A is (M, K)
    stride_bk, stride_bn,       # BT strides: BT is (K, N)
    stride_cm, stride_cn,       # C strides: C is (M, N)
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D program ids over tiles of output C
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Tile offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]
    offs_k = tl.arange(0, BLOCK_K)                    # [BLOCK_K]

    # Accumulator in fp32 for numeric stability
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K in chunks of BLOCK_K
    for k0 in range(0, K, BLOCK_K):
        # Load A tile: shape (BLOCK_M, BLOCK_K)
        A_ptrs = A_ptr + (offs_m[:, None] * stride_am + (k0 + offs_k[None, :]) * stride_ak)
        a_mask = (offs_m[:, None] < M) & (k0 + offs_k[None, :] < K)
        a = tl.load(A_ptrs, mask=a_mask, other=0.0)

        # Load BT tile: shape (BLOCK_K, BLOCK_N), BT is (K, N)
        BT_ptrs = BT_ptr + ((k0 + offs_k[:, None]) * stride_bk + offs_n[None, :] * stride_bn)
        b_mask = (k0 + offs_k[:, None] < K) & (offs_n[None, :] < N)
        bt = tl.load(BT_ptrs, mask=b_mask, other=0.0)

        # Accumulate in fp32
        acc += tl.dot(a, bt)

    # Store result C[m, n] = acc
    C_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(C_ptrs, acc, mask=out_mask)


def _choose_row_config(N, K):
    # Heuristic for row-wise kernel
    # Use moderate BLOCK_N and BLOCK_K to balance vector width and shared memory.
    # Larger N benefits from larger BLOCK_N; BLOCK_K 64 is a good default for FP16.
    BLOCK_N = 128 if N >= 1024 else 64
    BLOCK_K = 64
    num_warps = 4
    num_stages = 3
    return BLOCK_N, BLOCK_K, num_warps, num_stages


def _choose_2d_config(M, N, K):
    # Heuristic for 2D GEMM kernel
    # Conservative tiles to ensure resource usage stays within limits.
    # Increase tile if dimensions are large; keep BLOCK_K modest.
    if N >= 1024:
        BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 32
        num_warps, num_stages = 4, 3
    else:
        BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
        num_warps, num_stages = 4, 2
    return BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages


class ModelNew(torch.nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Ensure contiguous inputs
        A = A.contiguous()
        B = B.contiguous()
        # BT = B.T with shape (K, N)
        BT = B.transpose(0, 1).contiguous()

        M, K_a = A.shape
        K_b, N = BT.shape  # BT is (K, N)
        assert K_a == K_b, f"Incompatible shapes: A is (M={M}, K_a={K_a}), B.T should have K_b={M}, got {K_b}"

        # Output tensor, keep float16 as in original
        C = torch.empty((M, N), dtype=torch.float16, device=A.device)

        # Strides
        stride_am, stride_ak = A.stride()
        stride_bk, stride_bn = BT.stride()
        stride_cm, stride_cn = C.stride()

        # Choose strategy:
        # - If M is small, use row-wise kernel (one program per row).
        # - Otherwise, use 2D kernel.
        if M <= 32:
            BLOCK_N, BLOCK_K, num_warps, num_stages = _choose_row_config(N, K_a)
            grid = (M,)
            row_matmul_at_bT_kernel[grid](
                A, BT, C,
                M, N, K_a,
                stride_am, stride_ak,
                stride_bk, stride_bn,
                stride_cm, stride_cn,
                BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                num_warps=num_warps, num_stages=num_stages,
            )
        else:
            BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages = _choose_2d_config(M, N, K_a)
            grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
            matmul_at_bT_kernel_2d[grid](
                A, BT, C,
                M, N, K_a,
                stride_am, stride_ak,
                stride_bk, stride_bn,
                stride_cm, stride_cn,
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                num_warps=num_warps, num_stages=num_stages,
            )

        return C


def run(*args):
    return ModelNew()(*args)
