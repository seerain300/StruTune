import torch
import math
import triton
import triton.language as tl

# Triton kernel: per-row layer normalization over 1536 features.
# Input: x [num_rows, 1536], outputs normalized values with affine.
@triton.jit
def layernorm_row_kernel(
    x_ptr,            # *const float (we'll pass bfloat16 pointer; load as fp32)
    y_ptr,            # *float (store bfloat16)
    ln_weight_ptr,    # *const float (shape [1536])
    ln_bias_ptr,      # *const float (shape [1536])
    num_rows,         # int
    features,         # int (1536)
    eps,              # float
    stride_x_row,     # int
    stride_x_col,     # int
    stride_y_row,     # int
    stride_y_col,     # int
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= num_rows:
        return

    # First pass: compute sum and sum of squares in fp32
    s = 0.0
    ss = 0.0
    for off in range(0, features, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        mask = cols < features
        x = tl.load(x_ptr + row * stride_x_row + cols * stride_x_col, mask=mask, other=0.0)
        x = x.to(tl.float32)
        s += tl.sum(x, axis=0)
        ss += tl.sum(x * x, axis=0)

    mean = s / features
    var = ss / features - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    # Second pass: normalize and write bfloat16
    for off in range(0, features, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        mask = cols < features
        x = tl.load(x_ptr + row * stride_x_row + cols * stride_x_col, mask=mask, other=0.0)
        x = x.to(tl.float32)
        norm = (x - mean) * rstd
        w = tl.load(ln_weight_ptr + cols, mask=mask, other=1.0).to(tl.float32)
        b = tl.load(ln_bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        y = norm * w + b
        # Store as bfloat16
        tl.store(y_ptr + row * stride_y_row + cols * stride_y_col, y.to(tl.bfloat16), mask=mask)


# Triton matmul kernel: C[M, N] = A[M, K] @ B[K, N]
# A is input hidden (fp32), B is weight (bf16), C is output (fp32).
@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Program id for 2D launch grid
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    # accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        # Load A tile: [BLOCK_M, BLOCK_K]
        a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)

        # Load B tile: [BLOCK_K, BLOCK_N]
        b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)

        # acc += a @ b
        acc += tl.dot(a, b)

    # Write back C
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # No parameters; we'll get inputs via get_inputs and run as in the original

    def forward(self, *args):
        # args should be: hidden, grid_thw, ln_weight, ln_bias, fc1_weight, fc1_bias, fc2_weight, fc2_bias, eps
        hidden, grid_thw, ln_weight, ln_bias, fc1_weight, fc1_bias, fc2_weight, fc2_bias, eps = args

        # Step 1: Layer normalization using Triton (per vector across 1536 features)
        # Ensure inputs are on the same device and contiguous
        device = hidden.device
        hidden = hidden.contiguous()
        ln_weight = ln_weight.contiguous()
        ln_bias = ln_bias.contiguous()

        # We'll perform LN in fp32 for stability
        hidden_fp32 = hidden.to(torch.float32)
        hidden_ln = torch.empty_like(hidden_fp32, dtype=torch.float32, device=device)

        num_rows, features = hidden_fp32.shape
        assert features == 1536, "LayerNorm must be across 1536 features"

        # Launch Triton kernel: one program per row
        grid = (num_rows,)
        layernorm_row_kernel[grid](
            hidden_fp32, hidden_ln,
            ln_weight, ln_bias,
            num_rows, features,
            float(eps),
            hidden_fp32.stride(0), hidden_fp32.stride(1),
            hidden_ln.stride(0), hidden_ln.stride(1),
            BLOCK=1024,
            num_warps=4, num_stages=2,
        )

        # Step 2: Spatial shuffle (PyTorch view ops, no compute)
        # We follow the original code: compute t,h,w per grid, reshape, permute, and flatten.
        # Note: hidden_ln still has shape [num_patches, 1536]; grid_thw is int64 tensor (num_grids, 3).
        # The original code performs reshape/permute/flattening; we replicate that here.
        offset = 0
        shuffled_patches = []
        for i in range(grid_thw.shape[0]):
            t = int(grid_thw[i, 0].item())
            h = int(grid_thw[i, 1].item())
            w = int(grid_thw[i, 2].item())
            num_patches_this = t * h * w
            patches = hidden_ln[offset:offset + num_patches_this]  # [num_patches_this, 1536], fp32

            h_merged = h // 2
            w_merged = w // 2
            patches = patches.view(t, h_merged, 2, w_merged, 2, 1536)
            patches = patches.permute(0, 1, 3, 2, 4, 5).reshape(t * h_merged * w_merged, 8 * 1536)
            # patches now has shape [num_patches_this_after, 12288], fp32

            shuffled_patches.append(patches)
            offset += num_patches_this
        hidden_shuffled = torch.cat(shuffled_patches, dim=0)  # [num_merged_patches, 12288], fp32

        # Step 3: First linear (Triton GEMM), then GELU (PyTorch), then second linear (Triton GEMM).
        # First linear: hidden_shuffled (M, K) @ fc1_weight^T (K, out_features) -> (M, out_features)
        M = hidden_shuffled.shape[0]
        K = hidden_shuffled.shape[1]
        out_features1 = fc1_weight.shape[0]  # 6144

        # For Triton matmul, we need B of shape [K, out_features] which is fc1_weight transposed.
        # We'll create a contiguous B of shape [K, 6144] by transposing fc1_weight.
        B1 = fc1_weight.t().contiguous()  # [12288, 6144], bf16
        # C1 will be fp32
        C1 = torch.empty((M, out_features1), dtype=torch.float32, device=device)

        # Launch Triton matmul: grid over (M, out_features1)
        grid_matmul = (triton.cdiv(M, 64), triton.cdiv(out_features1, 64))
        matmul_kernel[grid_matmul](
            hidden_shuffled, B1, C1,
            M, out_features1, K,
            hidden_shuffled.stride(0), hidden_shuffled.stride(1),
            B1.stride(0), B1.stride(1),
            C1.stride(0), C1.stride(1),
            BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
            num_warps=4, num_stages=2,
        )

        # GELU activation (exact, erf-based) in PyTorch
        C1_gelu = torch.nn.functional.gelu(C1)

        # Second linear: C1_gelu (M, 6144) @ fc2_weight^T (6144, out_hidden_size) -> (M, 3584)
        out_features2 = fc2_weight.shape[0]  # 3584
        B2 = fc2_weight.t().contiguous()     # [6144, 3584], bf16
        C2 = torch.empty((M, out_features2), dtype=torch.float32, device=device)

        grid_matmul2 = (triton.cdiv(M, 64), triton.cdiv(out_features2, 64))
        matmul_kernel[grid_matmul2](
            C1_gelu, B2, C2,
            M, out_features2, 6144,
            C1_gelu.stride(0), C1_gelu.stride(1),
            B2.stride(0), B2.stride(1),
            C2.stride(0), C2.stride(1),
            BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
            num_warps=4, num_stages=2,
        )

        # Output should be bfloat16 to match original
        return C2.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
