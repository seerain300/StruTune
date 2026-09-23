import math
import torch
import triton
import triton.language as tl


# Triton LayerNorm kernel: per-row normalization across features (F=1536), affine (w,b), store bfloat16.
@triton.jit
def layernorm_rows_kernel(
    x_ptr,           # *const bfloat16, input [num_rows, features]
    y_ptr,           # *bfloat16, output [num_rows, features]
    ln_weight_ptr,   # *const float32, [features]
    ln_bias_ptr,     # *const float32, [features]
    num_rows: tl.constexpr,  # int
    features: tl.constexpr,  # int, must be 1536 here
    eps: tl.float32,
    BLOCK: tl.constexpr,     # tile size for reduction
):
    row = tl.program_id(0)
    if row >= num_rows:
        return

    # Two-pass: compute mean and variance in fp32
    sum_fp32 = 0.0
    sumsq_fp32 = 0.0

    for offs in range(0, features, BLOCK):
        idx = offs + tl.arange(0, BLOCK)
        mask = idx < features
        x = tl.load(x_ptr + row * features + idx, mask=mask, other=0.0).to(tl.float32)
        sum_fp32 += tl.sum(x, axis=0)
        sumsq_fp32 += tl.sum(x * x, axis=0)

    mean = sum_fp32 / features
    var = sumsq_fp32 / features - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Second pass: normalize and affine
    for offs in range(0, features, BLOCK):
        idx = offs + tl.arange(0, BLOCK)
        mask = idx < features
        x = tl.load(x_ptr + row * features + idx, mask=mask, other=0.0).to(tl.float32)
        norm = (x - mean) * inv_std
        w = tl.load(ln_weight_ptr + idx, mask=mask, other=1.0).to(tl.float32)
        b = tl.load(ln_bias_ptr + idx, mask=mask, other=0.0).to(tl.float32)
        y = norm * w + b
        tl.store(y_ptr + row * features + idx, y.to(tl.bfloat16), mask=mask)


# Triton GEMM kernel: A[M, K], B[K, N], output C[M, N] in fp32
@triton.jit
def gemm_kernel(
    A_ptr, B_ptr, C_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    stride_am, stride_ak,   # strides for A
    stride_bk, stride_bn,   # strides for B
    stride_cm, stride_cn,   # strides for C
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K
    for k in range(0, K, BLOCK_K):
        k_ids = k + offs_k
        a = tl.load(
            A_ptr + offs_m[:, None] * stride_am + k_ids[None, :] * stride_ak,
            mask=(offs_m[:, None] < M) & (k_ids[None, :] < K),
            other=0.0
        ).to(tl.float32)
        b = tl.load(
            B_ptr + k_ids[:, None] * stride_bk + offs_n[None, :] * stride_bn,
            mask=(k_ids[:, None] < K) & (offs_n[None, :] < N),
            other=0.0
        ).to(tl.float32)
        acc += tl.dot(a, b)

    # Store
    tl.store(
        C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N)
    )


# Triton GELU (erf approximation) elementwise: input X[M], output Y[M] in fp32
@triton.jit
def gelu_erf_kernel(
    X_ptr, Y_ptr,
    M: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < M

    x = tl.load(X_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    # erf approximation (Abramowitz & Stegun 7.1.26)
    # erf(x) ~ sign(x) * (1 - poly(t) * exp(-x^2)), t = 1 / (1 + p*|x|)
    p = 0.3275911
    sign = tl.where(x < 0, -1.0, 1.0)
    ax = tl.abs(x)
    t = 1.0 / (1.0 + p * ax)
    # poly = (((((a5*t + a4)*t + a3)*t + a2)*t + a1)*t)
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    poly = (((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t)
    erf_x = sign * (1.0 - poly * tl.exp(-ax * ax))
    gelu = 0.5 * x * (1.0 + erf_x)
    tl.store(Y_ptr + offs, gelu, mask=mask)


# Helper to launch layernorm kernel
def triton_layer_norm(hidden: torch.Tensor, ln_weight: torch.Tensor, ln_bias: torch.Tensor, eps: float) -> torch.Tensor:
    assert hidden.is_cuda, "Triton requires CUDA tensors."
    num_rows, features = hidden.shape
    assert features == 1536, "LayerNorm is defined over 1536 features."
    hidden_contig = hidden.contiguous()
    ln_weight_fp32 = ln_weight.to(torch.float32).contiguous()
    ln_bias_fp32 = ln_bias.to(torch.float32).contiguous()
    out = torch.empty_like(hidden_contig, dtype=torch.bfloat16, device=hidden_contig.device)
    # Choose BLOCK for features reduction
    BLOCK = 256
    grid = (num_rows,)
    layernorm_rows_kernel[grid](
        hidden_contig, out, ln_weight_fp32, ln_bias_fp32,
        num_rows=num_rows, features=features, eps=eps,
        BLOCK=BLOCK,
        num_warps=4, num_stages=2
    )
    return out


# Helper to launch GEMM kernel
def triton_gemm(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.is_cuda and B.is_cuda
    M, K = A.shape
    K2, N = B.shape
    assert K == K2, "Shape mismatch for GEMM."
    A_fp32 = A.contiguous().to(torch.float32)
    B_fp32 = B.contiguous().to(torch.float32)
    C = torch.empty((M, N), dtype=torch.float32, device=A_fp32.device)

    # Tile sizes tuned for these matrices; adjust if needed
    BLOCK_M = 64
    BLOCK_N = 128
    BLOCK_K = 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    gemm_kernel[grid](
        A_fp32, B_fp32, C,
        M, N, K,
        A_fp32.stride(0), A_fp32.stride(1),
        B_fp32.stride(0), B_fp32.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=3
    )
    return C


# Helper to launch GELU elementwise kernel
def triton_gelu(X: torch.Tensor) -> torch.Tensor:
    assert X.is_cuda
    M = X.numel()
    X_contig = X.contiguous().to(torch.float32)
    Y = torch.empty_like(X_contig, dtype=torch.float32, device=X_contig.device)
    BLOCK = 1024
    grid = (triton.cdiv(M, BLOCK),)
    gelu_erf_kernel[grid](X_contig, Y, M, BLOCK, num_warps=4, num_stages=2)
    return Y


class ModelNew(torch.nn.Module):
    def forward(
        self,
        hidden: torch.Tensor,        # [num_patches, 1536], bfloat16
        grid_thw: torch.Tensor,      # [num_grids, 3], int64
        ln_weight: torch.Tensor,     # [1536], bfloat16
        ln_bias: torch.Tensor,       # [1536], bfloat16
        fc1_weight: torch.Tensor,    # [6144, 1536], bfloat16
        fc1_bias: torch.Tensor,      # [6144], bfloat16 (not used in original heavy comp; ignored here)
        fc2_weight: torch.Tensor,    # [3584, 6144], bfloat16
        fc2_bias: torch.Tensor,      # [3584], bfloat16 (ignored)
        eps: float,
    ):
        # Step 1: Triton LayerNorm across each row (1536 features)
        hidden_norm = triton_layer_norm(hidden, ln_weight, ln_bias, eps)

        # Step 2: Spatial permutation and reshape via PyTorch (metadata-only)
        # We reproduce the original permutation logic per grid:
        # hidden_norm has shape [num_patches, 1536], grid_thw has shape [num_grids, 3] = (T, H, W).
        # The code in the original function loops over grids and constructs:
        # hidden_norm[offset:offset+T*H*W].view(T, H//2, 2, W//2, 2, 1536).permute(0,1,3,2,4,5).reshape(T*(H//2)*(W//2), 4*1536)
        # We will recompute this for all grids.
        num_patches = hidden_norm.shape[0]
        num_grids = grid_thw.shape[0]
        patches_per_grid = num_patches // num_grids  # integer division
        # Running offset
        offset = 0
        shuffled_patches = []
        for i in range(num_grids):
            T = int(grid_thw[i, 0].item())
            H = int(grid_thw[i, 1].item())
            W = int(grid_thw[i, 2].item())
            num_this = T * H * W
            patches = hidden_norm[offset:offset + num_this]
            # Ensure contiguous
            patches = patches.contiguous()
            # Reshape to (T, H//2, 2, W//2, 2, 1536)
            H2 = H // 2
            W2 = W // 2
            patches = patches.view(T, H2, 2, W2, 2, 1536)
            # Permute to (T, H//2, W//2, 2, 2, 1536)
            patches = patches.permute(0, 1, 3, 2, 4, 5)
            # Flatten spatial merge groups: (T * H//2 * W//2, 2*2*1536) = (num_patches_this, 12288)
            patches = patches.reshape(T * H2 * W2, 12288)
            shuffled_patches.append(patches)
            offset += num_this

        # Concatenate all grids (PyTorch allowed, metadata-only)
        hidden_shuffled = torch.cat(shuffled_patches, dim=0)  # [num_merged_patches, 12288]
        assert hidden_shuffled.shape[0] == hidden_norm.shape[0], "Total patches mismatch after shuffle."

        # Step 3: First Linear via Triton GEMM: A[M, K] @ Bt[K, N] -> C[M, N]
        # Note: fc1_weight is [6144, 1536], hidden_shuffled is [M, K=12288], output C[M, 6144]
        M = hidden_shuffled.shape[0]
        K = hidden_shuffled.shape[1]
        N = fc1_weight.shape[0]  # 6144
        # B is fc1_weight^T: [1536, 6144]
        B1 = fc1_weight.t().contiguous()  # [K=1536, N=6144], bfloat16
        C1 = triton_gemm(hidden_shuffled.to(torch.float32), B1.to(torch.float32))  # fp32

        # Step 4: GELU activation via Triton elementwise
        C1_gelu = triton_gelu(C1)  # fp32

        # Step 5: Second Linear via Triton GEMM: A[M, K=6144] @ Bt[K, N=3584] -> [M, 3584]
        K2 = C1_gelu.shape[1]  # 6144
        N2 = fc2_weight.shape[0]  # 3584
        B2 = fc2_weight.t().contiguous()  # [6144, 3584]
        output = triton_gemm(C1_gelu, B2.to(torch.float32))  # fp32

        return output


def run(*args):
    return ModelNew()(*args)
