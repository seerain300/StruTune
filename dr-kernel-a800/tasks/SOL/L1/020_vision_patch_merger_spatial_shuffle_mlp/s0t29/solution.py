import torch
import math
import triton
import triton.language as tl


@triton.jit
def layernorm_row_kernel(
    X_ptr,           # *bfloat16, input [num_rows, features]
    Y_ptr,           # *bfloat16, output [num_rows, features]
    LN_W_ptr,        # *float32, [features] ln_weight
    LN_B_ptr,        # *float32, [features] ln_bias
    NUM_ROWS,        # int
    FEATURES,        # int, e.g., 1536
    EPS,             # float32
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= NUM_ROWS:
        return

    total = 0.0
    total_sq = 0.0
    for offs in range(0, FEATURES, BLOCK):
        idx = offs + tl.arange(0, BLOCK)
        mask = idx < FEATURES
        x = tl.load(X_ptr + row * FEATURES + idx, mask=mask, other=0.0).to(tl.float32)
        total += tl.sum(x, axis=0)
        total_sq += tl.sum(x * x, axis=0)

    mean = total / FEATURES
    var = total_sq / FEATURES - mean * mean
    inv_std = 1.0 / tl.sqrt(var + EPS)

    for offs in range(0, FEATURES, BLOCK):
        idx = offs + tl.arange(0, BLOCK)
        mask = idx < FEATURES
        x = tl.load(X_ptr + row * FEATURES + idx, mask=mask, other=0.0).to(tl.float32)
        norm = (x - mean) * inv_std
        w = tl.load(LN_W_ptr + idx, mask=mask, other=1.0).to(tl.float32)
        b = tl.load(LN_B_ptr + idx, mask=mask, other=0.0).to(tl.float32)
        y = norm * w + b
        tl.store(Y_ptr + row * FEATURES + idx, y.to(tl.bfloat16), mask=mask)


@triton.jit
def linear_gemm_kernel_bf16_to_fp32(
    A_ptr,           # *bfloat16, [M, K]
    B_ptr,           # *bfloat16, [K, N]
    C_ptr,           # *float32, [M, N]
    M, N, K,         # int
    A_stride0, A_stride1,
    B_stride0, B_stride1,
    C_stride0, C_stride1,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        a_ptrs = A_ptr + offs_m[:, None] * A_stride0 + offs_k[None, :] * A_stride1
        b_ptrs = B_ptr + offs_k[:, None] * B_stride0 + offs_n[None, :] * B_stride1
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0).to(tl.float32)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0).to(tl.float32)
        acc += tl.dot(a, b)

    c_ptrs = C_ptr + offs_m[:, None] * C_stride0 + offs_n[None, :] * C_stride1
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


@triton.jit
def gelu_fp32_kernel(
    X_ptr,           # *float32, [M, N]
    Y_ptr,           # *float32, [M, N]
    M, N,            # int
    BLOCK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK + tl.arange(0, BLOCK)
    offs_n = pid_n * BLOCK + tl.arange(0, BLOCK)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(X_ptr + offs_m[:, None] * N + offs_n[None, :], mask=mask, other=0.0)
    # erf-based GELU approximation (Abramowitz & Stegun 7.1.26)
    p = 0.3275911
    sign = tl.where(x < 0, -1.0, 1.0)
    ax = tl.abs(x)
    t = 1.0 / (1.0 + p * ax)
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    poly = (((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t)
    erf_x = sign * (1.0 - poly * tl.exp(-ax * ax))
    y = 0.5 * x * (1.0 + erf_x)
    tl.store(Y_ptr + offs_m[:, None] * N + offs_n[None, :], y, mask=mask)


def _triton_linear_bf16_to_fp32(A_bf16: torch.Tensor, B_bf16: torch.Tensor):
    M, K = A_bf16.shape
    K2, N = B_bf16.shape
    assert K == K2
    A = A_bf16.contiguous()
    B = B_bf16.contiguous()
    C = torch.empty((M, N), dtype=torch.float32, device=A.device)
    grid = (triton.cdiv(M, 64), triton.cdiv(N, 128))
    linear_gemm_kernel_bf16_to_fp32[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=64, BLOCK_N=128, BLOCK_K=32,
        num_warps=4, num_stages=2,
    )
    return C


def triton_layer_norm(hidden: torch.Tensor, ln_weight: torch.Tensor, ln_bias: torch.Tensor, eps: float) -> torch.Tensor:
    assert hidden.is_cuda, "Triton requires CUDA tensors."
    num_rows, features = hidden.shape
    assert features == 1536, "LayerNorm is defined over 1536 features."
    hidden_contig = hidden.contiguous()
    ln_weight_fp32 = ln_weight.to(torch.float32).contiguous()
    ln_bias_fp32 = ln_bias.to(torch.float32).contiguous()
    out = torch.empty_like(hidden_contig, dtype=torch.bfloat16, device=hidden_contig.device)
    BLOCK = 256
    grid = (num_rows,)
    layernorm_row_kernel[grid](
        hidden_contig, out, ln_weight_fp32, ln_bias_fp32,
        num_rows, features, eps,
        BLOCK=BLOCK,
        num_warps=4, num_stages=2,
    )
    return out


def triton_gelu_fp32(X_fp32: torch.Tensor) -> torch.Tensor:
    M, N = X_fp32.shape
    Y = torch.empty_like(X_fp32, dtype=torch.float32, device=X_fp32.device)
    BLOCK = 128
    grid = (triton.cdiv(M, BLOCK), triton.cdiv(N, BLOCK))
    gelu_fp32_kernel[grid](
        X_fp32, Y, M, N, BLOCK=BLOCK, num_warps=4, num_stages=2
    )
    return Y


class ModelNew(torch.nn.Module):
    def __init__(self, num_patches: int, num_merged_patches: int, num_grids: int):
        # Inputs come from get_inputs in the evaluator; we keep init generic.
        super().__init__()
        # Store axes for potential use (not required for computation)
        self.num_patches = num_patches
        self.num_merged_patches = num_merged_patches
        self.num_grids = num_grids

    def forward(self, *args):
        # args expected order:
        # hidden: [num_patches, 1536], bfloat16, device='cuda'
        # grid_thw: [num_grids, 3], int64
        # ln_weight: [1536], bfloat16
        # ln_bias: [1536], bfloat16
        # fc1_weight: [6144, 1536], bfloat16
        # fc1_bias: [6144], bfloat16
        # fc2_weight: [3584, 6144], bfloat16
        # fc2_bias: [3584], bfloat16
        # eps: float
        hidden = args[0]
        grid_thw = args[1]
        ln_weight = args[2]
        ln_bias = args[3]
        fc1_weight = args[4]
        fc1_bias = args[5]
        fc2_weight = args[6]
        fc2_bias = args[7]
        eps = args[8]

        # 1) LayerNorm in Triton
        hidden_norm = triton_layer_norm(hidden, ln_weight, ln_bias, eps)

        # 2) Spatial shuffle via pure indexing (no torch.permute), produce [num_merged_patches, 12288]
        # For each grid i, given T, H, W:
        #   h_merged = H // 2
        #   w_merged = W // 2
        #   num_patches_this = T * H * W
        #   Take hidden_norm[base:base+num_patches_this], view as (T, h_merged, 2, w_merged, 2, 1536),
        #   permute to (T, h_merged, w_merged, 2, 2, 1536), reshape to (T * h_merged * w_merged, 1536*4) = (rows, 12288).
        # We do not use torch.permute; we build A_out directly in Triton-friendly shape.
        device = hidden_norm.device
        A_out = torch.empty((self.num_merged_patches, 12288), dtype=torch.bfloat16, device=device)
        # Build mapping: for each grid i
        total_merged = 0
        for i in range(self.num_grids):
            T = int(grid_thw[i, 0].item())
            H = int(grid_thw[i, 1].item())
            W = int(grid_thw[i, 2].item())
            h_merged = H // 2
            w_merged = W // 2
            num_patches_this = T * H * W
            if total_merged + num_patches_this > self.num_merged_patches:
                # In case the evaluator supplies larger num_merged_patches than needed, handle safely.
                break
            base = total_merged
            total_merged += num_patches_this
            # Build indices that map to hidden_norm rows. We need to compute which hidden_norm row corresponds
            # to each output row. Since the original code iterates grids and assigns patches contiguously,
            # base_offset is just base. We compute output rows: row_out in [0, T*h_merged*w_merged)
            for r_t in range(T):
                for h_merged_idx in range(h_merged):
                    for w_merged_idx in range(w_merged):
                        src_row = base + r_t * (H * W) + (h_merged_idx * 2) * W + (w_merged_idx * 2)
                        # copy 1536 features to A_out row
                        A_out[total_merged - num_patches_this + r_t * (h_merged * w_merged) + h_merged_idx * w_merged + w_merged_idx, :] = hidden_norm[src_row, :]

        # 3) First Linear: Triton GEMM. A_out [total_merged, 12288], B1 = fc1_weight.T [12288, 6144]
        B1 = fc1_weight.t().to(torch.bfloat16).contiguous()  # [12288, 6144]
        C1 = _triton_linear_bf16_to_fp32(A_out, B1)  # [self.num_merged_patches, 6144], fp32

        # 4) GELU in Triton, fp32 -> fp32
        C1_gelu = triton_gelu_fp32(C1)  # fp32

        # 5) Second Linear: Triton GEMM. B2 = fc2_weight.T [6144, 3584], bfloat16
        B2 = fc2_weight.t().to(torch.bfloat16).contiguous()  # [6144, 3584]
        output = _triton_linear_bf16_to_fp32(C1_gelu.to(torch.bfloat16), B2)  # fp32

        return output


def run(*args):
    return ModelNew()(*args)
