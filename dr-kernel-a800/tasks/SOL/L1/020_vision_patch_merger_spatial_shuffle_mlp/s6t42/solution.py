import torch
import triton
import triton.language as tl

# Triton kernel: LayerNorm + affine over last dim for each row
@triton.jit
def _layer_norm_affine_kernel(
    x_ptr,        # *bf16, shape (M, K)
    ln_w_ptr,     # *bf16, shape (K,)
    ln_b_ptr,     # *bf16, shape (K,)
    y_ptr,        # *bf16, shape (M, K)
    M: tl.int32,  # number of rows
    K: tl.int32,  # hidden size
    eps: tl.float32,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    # First pass: compute sum and sum of squares over K
    sum_val = tl.zeros((), dtype=tl.float32)
    sum_sq = tl.zeros((), dtype=tl.float32)
    for k in range(0, K, BLOCK):
        idx = k + tl.arange(0, BLOCK)
        mask = idx < K
        x = tl.load(x_ptr + row * K + idx, mask=mask, other=0.0)
        x = x.to(tl.float32)
        sum_val += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)
    mean = sum_val / K
    var = sum_sq / K - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Second pass: normalize and apply affine, store BF16
    for k in range(0, K, BLOCK):
        idx = k + tl.arange(0, BLOCK)
        mask = idx < K
        x = tl.load(x_ptr + row * K + idx, mask=mask, other=0.0)
        x = x.to(tl.float32)
        w = tl.load(ln_w_ptr + idx, mask=mask, other=1.0).to(tl.float32)
        b = tl.load(ln_b_ptr + idx, mask=mask, other=0.0).to(tl.float32)
        y = (x - mean) * inv_std
        y = y * w + b
        # store as BF16
        tl.store(y_ptr + row * K + idx, y.to(tl.bfloat16), mask=mask)


# Triton kernel: pack normalized hidden into expanded feature dimension
# Input: ln_out (M_out, K), Output: packed (M_out, 4*K)
@triton.jit
def _pack_2x2_kernel(
    src_ptr,      # *bf16, shape (M_out * 4, K)
    dst_ptr,      # *bf16, shape (M_out, 4*K)
    M_out: tl.int32,
    K: tl.int32,
):
    out_row = tl.program_id(0)
    # base input row corresponding to this output row: idx = out_row * 4
    base_in_row = out_row * 4
    total_cols = 4 * K

    # write each of the 4 segments
    # segment s corresponds to input rows: base_in_row + s*2, base_in_row + s*2 + 1
    # columns are 0:K for first two, K:2K for third, 2K:3K for fourth
    # We don't use blocks; we loop over K (simple and safe). Note: Triton does not allow dynamic Python loops
    # over K in the kernel, so we implement with a small unrolled approach using s in static range.
    # We'll do 4 separate stores: s=0,1,2,3.

    # s = 0: rows r0=base_in_row, r1=base_in_row+1, cols 0:K
    r0 = base_in_row
    r1 = base_in_row + 1
    col0 = tl.arange(0, K)
    tl.store(dst_ptr + out_row * total_cols + (0 * K) + col0, tl.load(src_ptr + r0 * K + col0), mask=(col0 < K))
    tl.store(dst_ptr + out_row * total_cols + (1 * K) + col0, tl.load(src_ptr + r1 * K + col0), mask=(col0 < K))

    # s = 2: rows r0=base_in_row+2, r1=base_in_row+3, cols 2K:3K
    r2 = base_in_row + 2
    r3 = base_in_row + 3
    col2K = tl.arange(0, K) + 2 * K
    # mask for col2K is always valid since 2*K < 4*K and K > 0
    tl.store(dst_ptr + out_row * total_cols + (2 * K) + col2K - 2 * K, tl.load(src_ptr + r2 * K + col0), mask=(col0 < K))
    tl.store(dst_ptr + out_row * total_cols + (3 * K) + col2K - 2 * K, tl.load(src_ptr + r3 * K + col0), mask=(col0 < K))


# Triton kernel: GEMM + bias, C[M, N] = A[M, K] @ W[K, N]^T + bias[N]
@triton.jit
def _gemm_bias_kernel(
    A_ptr,        # *bf16, shape (M, K)
    W_ptr,        # *bf16, shape (N, K) where we access as B[n, k] = W_ptr[n*K + k]
    B_ptr,        # *bf16 bias, shape (N,)
    C_ptr,        # *bf16, output (M, N)
    M: tl.int32,
    N: tl.int32,
    K: tl.int32,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K in chunks
    for k0 in range(0, K, BLOCK_K):
        k = k0 + offs_k  # [BLOCK_K]
        # Load A_tile: [BLOCK_M, BLOCK_K]
        a_ptrs = A_ptr + offs_m[:, None] * K + k[None, :]
        a_mask = (offs_m[:, None] < M) & (k[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0).to(tl.float32)

        # Load W_tile as [BLOCK_K, BLOCK_N] by indexing W_ptr as B[n, k] = W_ptr[n*K + k]
        # offs_n: [BLOCK_N], k[None, :]: [1, BLOCK_K]
        b_ptrs = W_ptr + offs_n[None, :] * K + k[:, None]
        b_mask = (offs_n[None, :] < N) & (k[:, None] < K)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0).to(tl.float32)

        # Accumulate
        acc += tl.dot(a, b)

    # Add bias
    bias = tl.load(B_ptr + offs_n, mask=(offs_n < N), other=0.0).to(tl.float32)
    acc += bias[None, :]

    # Store to C in BF16
    c_ptrs = C_ptr + offs_m[:, None] * N + offs_n[None, :]
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.bfloat16), mask=c_mask)


# Triton kernel: elementwise GELU (tanh approximation) on X[M, N], store Y[M, N] (BF16)
@triton.jit
def _gelu_tanh_kernel(
    X_ptr,        # *bf16, shape (M, N)
    Y_ptr,        # *bf16, shape (M, N)
    M: tl.int32,
    N: tl.int32,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # but we only use m index here, not block
    # Triton expects grid = (M, ceil_div(N, BLOCK_N)); pid_n iterates over column tiles
    m_idx = pid_m
    n_start = pid_n * BLOCK_N
    offs_n = n_start + tl.arange(0, BLOCK_N)

    mask = (m_idx < M) & (offs_n < N)

    x = tl.load(X_ptr + m_idx * N + offs_n, mask=mask, other=0.0).to(tl.float32)
    # tanh approximation: gelu(x) = 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))
    c0 = 0.7978845608028654  # sqrt(2/pi)
    c1 = 0.044715
    x3 = x * x * x
    inner = c0 * (x + c1 * x3)
    t = tl.tanh(inner)
    y = 0.5 * x * (1.0 + t)
    tl.store(Y_ptr + m_idx * N + offs_n, y.to(tl.bfloat16), mask=mask)


# ModelNew entry point (Triton-only forward)
class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # No parameters needed; Triton kernels will use provided tensors.

    def forward(self, hidden: torch.Tensor,
                grid_thw: torch.Tensor,
                ln_weight: torch.Tensor,
                ln_bias: torch.Tensor,
                fc1_weight: torch.Tensor,
                fc1_bias: torch.Tensor,
                fc2_weight: torch.Tensor,
                fc2_bias: torch.Tensor,
                eps: float):
        device = hidden.device
        num_patches = hidden.shape[0]
        hidden_size = hidden.shape[1]  # 1536
        M_out = num_patches // 4  # output rows = num_patches / 4 (packing 2x2)

        # 1) LayerNorm + affine over last dim, per-row Triton kernel
        ln_out = torch.empty_like(hidden, dtype=torch.bfloat16, device=device)
        BLOCK_ln = 256
        grid_ln = (num_patches,)
        _layer_norm_affine_kernel[grid_ln](
            hidden, ln_weight, ln_bias, ln_out,
            M=num_patches, K=hidden_size, eps=eps,
            BLOCK=BLOCK_ln, num_warps=4, num_stages=2
        )

        # 2) Pack: reshape normalized hidden to (M_out, 4*hidden_size) via Triton kernel
        # Note: grid_thw is not used in packing; provided generator ensures num_patches % 4 == 0
        packed = torch.empty((M_out, 4 * hidden_size), dtype=torch.bfloat16, device=device)
        grid_pack = (M_out,)
        _pack_2x2_kernel[grid_pack](
            ln_out, packed,
            M_out=M_out, K=hidden_size
        )

        # 3) FC1: packed (M_out, 6144) @ fc1_weight (6144, 6144)^T + fc1_bias
        M_merged = M_out
        K1 = packed.shape[1]  # 4*hidden_size = 6144
        N1 = fc1_weight.shape[0]  # 6144
        fc1_out = torch.empty((M_merged, N1), dtype=torch.bfloat16, device=device)

        BLOCK_M1, BLOCK_N1, BLOCK_K1 = 128, 128, 64
        grid_fc1 = (triton.cdiv(M_merged, BLOCK_M1), triton.cdiv(N1, BLOCK_N1))
        _gemm_bias_kernel[grid_fc1](
            packed, fc1_weight, fc1_bias, fc1_out,
            M=M_merged, N=N1, K=K1,
            BLOCK_M=BLOCK_M1, BLOCK_N=BLOCK_N1, BLOCK_K=BLOCK_K1,
            num_warps=4, num_stages=2
        )

        # 4) GELU activation: Triton elementwise kernel on fc1_out
        # GELU on (M_merged, N1) = (M_merged, 6144)
        gelu_out = torch.empty_like(fc1_out, dtype=torch.bfloat16, device=device)
        BLOCK_N_gelu = 256
        grid_gelu = (M_merged, triton.cdiv(N1, BLOCK_N_gelu))
        # Note: gelu kernel expects input in FP32, but we read bf16, cast to fp32, compute, store bf16.
        # We'll pass fc1_out and gelu_out as pointers; Triton will cast loads to fp32.
        # Implement GELU: using tanh approximation in-kernel.
        # However, Triton doesn't have a GELU intrinsic; we implement it here. For simplicity, we can also
        # compute GELU with torch after, but to stay Triton-only, we inline the GELU computation:
        # But we already have _gelu_tanh_kernel defined; we'll use it.
        # We need to ensure gelu_out is bf16 and input is bf16. Triton will load bf16, cast to fp32 internally.

        # Launch GELU on fc1_out -> gelu_out
        _gelu_tanh_kernel[grid_gelu](
            fc1_out, gelu_out,
            M=M_merged, N=N1, BLOCK_N=BLOCK_N_gelu
        )

        # 5) FC2: gelu_out (M_merged, 6144) @ fc2_weight (3584, 6144)^T + fc2_bias
        N2 = fc2_weight.shape[0]  # 3584
        output = torch.empty((M_merged, N2), dtype=torch.bfloat16, device=device)

        BLOCK_M2, BLOCK_N2, BLOCK_K2 = 128, 128, 64
        grid_fc2 = (triton.cdiv(M_merged, BLOCK_M2), triton.cdiv(N2, BLOCK_N2))
        _gemm_bias_kernel[grid_fc2](
            gelu_out, fc2_weight, fc2_bias, output,
            M=M_merged, N=N2, K=N1,  # K is N from fc1_out
            BLOCK_M=BLOCK_M2, BLOCK_N=BLOCK_N2, BLOCK_K=BLOCK_K2,
            num_warps=4, num_stages=2
        )

        return output


def run(*args):
    return ModelNew()(*args)
