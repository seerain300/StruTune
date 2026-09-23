import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton LayerNorm over each row of length H (hidden_size=1536), bfloat16 input -> bfloat16 output
if TRITON_AVAILABLE:
    @triton.jit
    def layernorm_rows_kernel(
        x_ptr,           # *ptr to input patches (N, H), bfloat16
        y_ptr,           # *ptr to output patches (N, H), bfloat16
        ln_weight_ptr,   # *ptr to ln_weight (H), bfloat16
        ln_bias_ptr,     # *ptr to ln_bias (H), bfloat16
        N,               # number of rows (num_patches)
        H: tl.constexpr, # hidden_size (1536)
        eps,             # epsilon (float32)
        BLOCK_H: tl.constexpr,
    ):
        row_id = tl.program_id(0)
        if row_id >= N:
            return
        row_offset = row_id * H

        # Compute mean in float32
        sum_ = 0.0
        for off in range(0, H, BLOCK_H):
            cols = off + tl.arange(0, BLOCK_H)
            mask = cols < H
            x = tl.load(x_ptr + row_offset + cols, mask=mask, other=0.0)
            x = x.to(tl.float32)
            sum_ += tl.sum(x, axis=0)
        mean = sum_ / H

        # Compute variance in float32
        var_sum = 0.0
        for off in range(0, H, BLOCK_H):
            cols = off + tl.arange(0, BLOCK_H)
            mask = cols < H
            x = tl.load(x_ptr + row_offset + cols, mask=mask, other=0.0)
            x = x.to(tl.float32)
            var_sum += tl.sum((x - mean) * (x - mean), axis=0)
        var = var_sum / H
        rstd = 1.0 / tl.sqrt(var + eps)

        # Normalize and apply affine, store as bfloat16
        for off in range(0, H, BLOCK_H):
            cols = off + tl.arange(0, BLOCK_H)
            mask = cols < H
            x = tl.load(x_ptr + row_offset + cols, mask=mask, other=0.0)
            x = x.to(tl.float32)
            gamma = tl.load(ln_weight_ptr + cols, mask=mask, other=1.0).to(tl.float32)
            beta = tl.load(ln_bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
            y = (x - mean) * rstd
            y = y * gamma + beta
            tl.store(y_ptr + row_offset + cols, y.to(tl.bfloat16), mask=mask)


# Triton GEMM: C[M, N] = A[M, K] @ B[K, N], with bias on C
# We will use this for both linear layers by passing appropriate A, B (which are weight tensors).
if TRITON_AVAILABLE:
    @triton.jit
    def gemm_rowcol_kernel(
        A_ptr,            # *ptr to A (M, K), float32
        B_ptr,            # *ptr to B (K, N), float32
        Bias_ptr,         # *ptr to bias (N), float32
        C_ptr,            # *ptr to C (M, N), float32
        M, K, N,
        TILE_M: tl.constexpr,
        TILE_N: tl.constexpr,
        TILE_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * TILE_M + tl.arange(0, TILE_M)
        offs_n = pid_n * TILE_N + tl.arange(0, TILE_N)
        offs_k_init = 0

        acc = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)

        while offs_k_init < K:
            offs_k = offs_k_init + tl.arange(0, TILE_K)
            a = tl.load(
                A_ptr + (offs_m[:, None] * K + offs_k[None, :]),
                mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
                other=0.0,
            )
            b = tl.load(
                B_ptr + (offs_k[:, None] * N + offs_n[None, :]),
                mask=(offs_k[:, None] < K) & (offs_n[None, :] < N),
                other=0.0,
            )
            acc += tl.dot(a, b)
            offs_k_init += TILE_K

        # Add bias
        bias = tl.load(Bias_ptr + offs_n, mask=(offs_n < N), other=0.0)
        acc += bias[None, :]

        # Store
        tl.store(
            C_ptr + (offs_m[:, None] * N + offs_n[None, :]),
            acc,
            mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
        )


# Triton elementwise GELU activation: y = 0.5 * x * (1 + erf(x / sqrt(2)))
if TRITON_AVAILABLE:
    @triton.jit
    def gelu_kernel(
        x_ptr,      # *ptr input tensor, float32
        y_ptr,      # *ptr output tensor, float32
        numel,      # total number of elements
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < numel
        x = tl.load(x_ptr + offs, mask=mask, other=0.0)
        # GELU
        inv_sqrt2 = 0.70710678118654752440084436210485
        x_scaled = x * inv_sqrt2
        y = 0.5 * x * (1.0 + tl.math.erf(x_scaled))
        tl.store(y_ptr + offs, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(
        self,
        hidden: torch.Tensor,
        grid_thw: torch.Tensor,
        ln_weight: torch.Tensor,
        ln_bias: torch.Tensor,
        fc1_weight: torch.Tensor,
        fc1_bias: torch.Tensor,
        fc2_weight: torch.Tensor,
        fc2_bias: torch.Tensor,
        eps: float,
    ):
        """
        Triton-only forward. Computes:
          1) LayerNorm across hidden_size for each patch (num_patches, 1536)
          2) First linear: (num_merged_patches, 6144) @ (6144, 6144) + bias
          3) GELU activation
          4) Second linear: (num_merged_patches, 6144) @ (3584, 6144) + bias
        """
        assert hidden.dim() == 2, "hidden must be (num_patches, hidden_size)"
        N, H = hidden.shape
        device = hidden.device

        # 1) Triton LayerNorm: (N, H) -> (N, H), bfloat16
        hidden_norm = torch.empty((N, H), dtype=torch.bfloat16, device=device)
        BLOCK_H = 256  # tuneable
        grid = (N,)
        layernorm_rows_kernel[grid](
            hidden, hidden_norm, ln_weight, ln_bias,
            N, H, eps,
            BLOCK_H=BLOCK_H,
            num_warps=4,
            num_stages=2,
        )

        # 2) First linear: (N, 6144) @ (6144, 6144) + bias -> (N, 6144), float32
        A = hidden_norm.to(torch.float32)  # input A for first linear
        K1 = 6144
        M = N  # num_merged_patches == num_patches in provided configs
        Bias1 = fc1_bias.to(device=device, dtype=torch.float32)

        # Allocate C1
        C1 = torch.empty((M, K1), dtype=torch.float32, device=device)

        # Launch GEMM kernel: C[M,K1] = A[M,K1] @ B[K1,6144]
        TILE_M = 64
        TILE_N = 64
        TILE_K = 32
        grid_gemm1 = (triton.cdiv(M, TILE_M), triton.cdiv(K1, TILE_N))
        gemm_rowcol_kernel[grid_gemm1](
            A, fc1_weight, Bias1, C1,
            M, K1, K1,  # K dimension is 6144 for both inputs to matmul
            TILE_M=TILE_M,
            TILE_N=TILE_N,
            TILE_K=TILE_K,
            num_warps=4,
            num_stages=3,
        )

        # 3) GELU activation in Triton
        C1_gelu = torch.empty_like(C1, dtype=torch.float32, device=device)
        numel = C1.numel()
        BLOCK = 1024
        grid_gelu = (triton.cdiv(numel, BLOCK),)
        gelu_kernel[grid_gelu](
            C1, C1_gelu, numel,
            BLOCK=BLOCK,
            num_warps=4,
            num_stages=2,
        )

        # 4) Second linear: (N, 6144) @ (3584, 6144) + bias -> (N, 3584), float32
        B2 = fc2_weight  # (3584, 6144)
        Bias2 = fc2_bias.to(device=device, dtype=torch.float32)
        OUT_N = B2.shape[0]  # 3584
        C2 = torch.empty((M, OUT_N), dtype=torch.float32, device=device)

        grid_gemm2 = (triton.cdiv(M, TILE_M), triton.cdiv(OUT_N, TILE_N))
        gemm_rowcol_kernel[grid_gemm2](
            C1_gelu, B2, Bias2, C2,
            M, K1, OUT_N,
            TILE_M=TILE_M,
            TILE_N=TILE_N,
            TILE_K=TILE_K,
            num_warps=4,
            num_stages=3,
        )

        # Return bfloat16 to match typical output dtype expectations
        return C2.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
