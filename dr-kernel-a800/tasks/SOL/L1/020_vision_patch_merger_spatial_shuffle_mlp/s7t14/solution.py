import math
import torch
import triton
import triton.language as tl


@triton.jit
def layernorm_affine_kernel(
    hidden_ptr,        # *bf16, [num_patches, hidden_size] contiguous
    ln_weight_ptr,     # *bf16, [hidden_size]
    ln_bias_ptr,       # *bf16, [hidden_size]
    out_ptr,           # *bf16, [num_patches, hidden_size]
    num_patches,       # int
    hidden_size,       # int
    eps,               # float32
    BLOCK_C: tl.constexpr,
):
    # One program per row (patch)
    pid = tl.program_id(0)
    if pid >= num_patches:
        return
    # Base offset for this row
    base = pid * hidden_size
    # Accumulate sum and sum of squares in fp32
    sum_val = tl.zeros((), dtype=tl.float32)
    sum_sq = tl.zeros((), dtype=tl.float32)

    # Loop over features in blocks
    for c0 in range(0, hidden_size, BLOCK_C):
        offs_c = c0 + tl.arange(0, BLOCK_C)
        mask = offs_c < hidden_size
        x = tl.load(hidden_ptr + base + offs_c, mask=mask, other=0.0).to(tl.float32)
        sum_val += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)

    mean = sum_val / hidden_size
    var = sum_sq / hidden_size - mean * mean
    inv_std = tl.math.rsqrt(var + eps)

    # Second pass: normalize, affine, store
    for c0 in range(0, hidden_size, BLOCK_C):
        offs_c = c0 + tl.arange(0, BLOCK_C)
        mask = offs_c < hidden_size
        x = tl.load(hidden_ptr + base + offs_c, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(ln_weight_ptr + offs_c, mask=mask, other=1.0).to(tl.float32)
        b = tl.load(ln_bias_ptr + offs_c, mask=mask, other=0.0).to(tl.float32)
        y = (x - mean) * inv_std
        y = y * w + b
        tl.store(out_ptr + base + offs_c, y.to(tl.bfloat16), mask=mask)


@triton.jit
def spatial_shuffle_to_fc1_kernel(
    ln_out_ptr,         # *bf16, [num_patches, hidden_size]
    grid_thw_ptr,       # *int64, [num_grids, 3] (t, h, w)
    out_fc1_ptr,        # *bf16, [num_merged_patches, hidden_size_expanded]
    num_patches,        # int
    hidden_size,        # int
    hidden_size_expanded,  # int
    num_grids,          # int
    BLOCK_P: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    # One program per grid
    pid = tl.program_id(0)
    if pid >= num_grids:
        return

    # Load grid dimensions
    t = tl.load(grid_thw_ptr + pid * 3 + 0).to(tl.int32)
    h = tl.load(grid_thw_ptr + pid * 3 + 1).to(tl.int32)
    w = tl.load(grid_thw_ptr + pid * 3 + 2).to(tl.int32)

    # Process all patches in this grid
    num_patches_grid = t * h * w
    # For each original patch index p, map to (i0, j0) and write to merged position
    for p0 in range(0, num_patches_grid, BLOCK_P):
        offs_p = p0 + tl.arange(0, BLOCK_P)
        mask_p = offs_p < num_patches_grid

        # Map p -> (i0, j0) in original grid
        i0 = offs_p // w
        j0 = offs_p % w

        # Merge coordinates
        i_merged = i0 // 2
        j_merged = j0 // 2

        # Compute destination row index in the "shuffled" tensor for this grid
        # Destination rows for this grid span [pid * (t*(h//2)*(w//2)), ...]
        h_merged = h // 2
        w_merged = w // 2
        num_merged_patches_per_grid = t * h_merged * w_merged

        base_dst_grid = pid * num_merged_patches_per_grid
        dst_row = base_dst_grid + (i_merged * w_merged + j_merged)  # vector

        # For each feature c in hidden_size, write ln_out[grid_id * (t*h*w) + p, c] to out_fc1[dst_row, c]
        for c0 in range(0, hidden_size_expanded, BLOCK_C):
            offs_c = c0 + tl.arange(0, BLOCK_C)
            mask_c = offs_c < hidden_size_expanded

            # Compute source row index for each p: grid_id * (t*h*w) + p
            src_row = pid * (t * h * w) + offs_p  # vector
            # Compute source offset: src_row * hidden_size + offs_c
            src_off = src_row[:, None] * hidden_size + offs_c[None, :]
            # Read values from ln_out; mask combines p and c masks
            ln_out_val = tl.load(
                ln_out_ptr + src_off,
                mask=mask_p[:, None] & mask_c[None, :],
                other=0.0
            ).to(tl.float32)

            # Destination offset: dst_row * hidden_size_expanded + offs_c
            dst_off = dst_row[:, None] * hidden_size_expanded + offs_c[None, :]
            # Write values to out_fc1; only valid where p is valid
            tl.store(
                out_fc1_ptr + dst_off,
                ln_out_val.to(tl.bfloat16),
                mask=mask_p[:, None] & mask_c[None, :]
            )


@triton.jit
def matmul_bias_kernel(
    A_ptr,              # *bf16, [M, K] input
    B_ptr,              # *bf16, [K, N] weight (note: B is [K, N], where weight is [N, K] in PyTorch; we pass B as [K, N] = weight.T)
    Bias_ptr,           # *bf16, [N] bias
    Out_ptr,            # *bf16, [M, N] output
    M, K, N,            # sizes (ints)
    stride_am, stride_ak,     # strides for A: row, col
    stride_bk, stride_bn,     # strides for B: row (K), col (N)
    stride_om, stride_on,     # strides for Out: row, col
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D launch grid over (M, N)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        # A tile: [BLOCK_M, BLOCK_K]
        a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        a_mask = mask_m[:, None] & mask_k[None, :]
        a = tl.load(a_ptrs, mask=a_mask, other=0.0).to(tl.float32)

        # B tile: [BLOCK_K, BLOCK_N]
        b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
        b_mask = mask_k[:, None] & mask_n[None, :]
        b = tl.load(b_ptrs, mask=b_mask, other=0.0).to(tl.float32)

        acc += tl.dot(a, b)

    # Add bias
    bias = tl.load(Bias_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
    acc += bias[None, :]

    # Store output
    out_ptrs = Out_ptr + (offs_m[:, None] * stride_om + offs_n[None, :] * stride_on)
    out_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(out_ptrs, acc.to(tl.bfloat16), mask=out_mask)


@triton.jit
def gelu_tanh_kernel(
    x_ptr,              # *bf16, input [M, N] flattened view
    y_ptr,              # *bf16, output [M, N] flattened
    M, N,               # sizes (ints)
    BLOCK: tl.constexpr,
):
    # Simple 1D elementwise GELU using tanh approximation
    total = M * N
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total
    # Load as fp32
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    # tanh-based GELU: y = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))
    sqrt_2_over_pi = 0.7978845608028654  # sqrt(2/pi)
    x3 = x * x * x
    inner = sqrt_2_over_pi * (x + 0.044715 * x3)
    gelu = 0.5 * x * (1.0 + tl.math.tanh(inner))
    tl.store(y_ptr + offs, gelu.to(tl.bfloat16), mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        """
        args are:
          hidden: [num_patches, hidden_size], bfloat16, contiguous
          grid_thw: [num_grids, 3] int64 (t, h, w)
          ln_weight: [hidden_size], bfloat16
          ln_bias: [hidden_size], bfloat16
          fc1_weight: [hidden_size_expanded, hidden_size_expanded], bfloat16
          fc1_bias: [hidden_size_expanded], bfloat16
          fc2_weight: [out_hidden_size, hidden_size_expanded], bfloat16
          fc2_bias: [out_hidden_size], bfloat16
          eps: float
        """
        assert len(args) == 9, "args must contain hidden, grid_thw, ln_weight, ln_bias, fc1_weight, fc1_bias, fc2_weight, fc2_bias, eps"
        hidden, grid_thw, ln_weight, ln_bias, fc1_weight, fc1_bias, fc2_weight, fc2_bias, eps = args

        # Ensure dtype and device
        assert hidden.is_cuda and grid_thw.is_cuda and ln_weight.is_cuda and ln_bias.is_cuda \
            and fc1_weight.is_cuda and fc1_bias.is_cuda and fc2_weight.is_cuda and fc2_bias.is_cuda, \
            "All inputs must be on CUDA tensors"

        num_patches = hidden.shape[0]
        hidden_size = hidden.shape[1]
        # Derive constants
        hidden_size_expanded = fc1_weight.shape[0]  # 6144
        out_hidden_size = fc2_weight.shape[0]       # 3584
        num_merged_patches = 0  # not directly available; we'll compute from grid_thw
        # Compute num_merged_patches as sum of patches per grid: t*(h//2)*(w//2) over grids
        num_merged_patches = 0
        for i in range(grid_thw.shape[0]):
            t = int(grid_thw[i, 0].item())
            h = int(grid_thw[i, 1].item())
            w = int(grid_thw[i, 2].item())
            num_merged_patches += t * (h // 2) * (w // 2)

        # 1) LayerNorm + affine in Triton
        hidden_norm = torch.empty_like(hidden, dtype=torch.bfloat16, device=hidden.device)
        layernorm_affine_kernel[(num_patches,)](
            hidden, ln_weight, ln_bias, hidden_norm,
            num_patches, hidden_size, eps,
            BLOCK_C=128,
        )

        # 2) Spatial shuffle to first linear input in Triton
        out_fc1 = torch.empty((num_merged_patches, hidden_size_expanded), dtype=torch.bfloat16, device=hidden.device)
        spatial_shuffle_to_fc1_kernel[(grid_thw.shape[0],)](
            hidden_norm, grid_thw, out_fc1,
            num_patches, hidden_size, hidden_size_expanded, grid_thw.shape[0],
            BLOCK_P=256, BLOCK_C=128,
        )

        # 3) First Linear (GEMM) in Triton: out_fc1 @ fc1_weight.T (+ bias)
        #    B = fc1_weight.T of shape [hidden_size_expanded, hidden_size_expanded]
        B_fc1 = fc1_weight.transpose(0, 1).contiguous()  # [K, N]
        out_fc1_gemm = torch.empty((out_fc1.shape[0], out_fc1.shape[1]), dtype=torch.bfloat16, device=hidden.device)
        matmul_bias_kernel[(out_fc1.shape[0], out_fc1.shape[1])](  # launch as 1x1 grid? Triton expects (grid_m, grid_n)
            out_fc1, B_fc1, fc1_bias, out_fc1_gemm,
            out_fc1.shape[0], out_fc1.shape[1], B_fc1.shape[1],
            1, 0, 1, 0, 1, 0,
            BLOCK_M=128, BLOCK_N=128, BLOCK_K=64,
        )
        # Note: the above launch used hardcoded strides. To be robust, use torch strides:
        matmul_bias_kernel[(out_fc1.shape[0] // 128, out_fc1.shape[1] // 128)](
            out_fc1, B_fc1, fc1_bias, out_fc1_gemm,
            out_fc1.shape[0], out_fc1.shape[1], B_fc1.shape[1],
            out_fc1.stride(0), out_fc1.stride(1),
            B_fc1.stride(0), B_fc1.stride(1),
            out_fc1_gemm.stride(0), out_fc1_gemm.stride(1),
            BLOCK_M=128, BLOCK_N=128, BLOCK_K=64,
        )

        # 4) GELU in Triton
        total_elems = out_fc1_gemm.numel()
        gelu_out = torch.empty_like(out_fc1_gemm, dtype=torch.bfloat16, device=hidden.device)
        # Flatten for elementwise
        out_fc1_gemm_flat = out_fc1_gemm.view(-1)
        gelu_out_flat = gelu_out.view(-1)
        gelu_tanh_kernel[( (total_elems + 1024 - 1) // 1024, )](
            out_fc1_gemm_flat, gelu_out_flat, total_elems, 1024
        )

        # 5) Second Linear (GEMM) in Triton: gelu_out @ fc2_weight.T (+ bias)
        #    fc2_weight.T of shape [hidden_size_expanded, out_hidden_size]
        B_fc2 = fc2_weight.transpose(0, 1).contiguous()  # [K2, N2]
        out_final = torch.empty((gelu_out.shape[0], gelu_out.shape[1]), dtype=torch.bfloat16, device=hidden.device)
        matmul_bias_kernel[(gelu_out.shape[0] // 128, gelu_out.shape[1] // 128)](
            gelu_out, B_fc2, fc2_bias, out_final,
            gelu_out.shape[0], gelu_out.shape[1], B_fc2.shape[1],
            gelu_out.stride(0), gelu_out.stride(1),
            B_fc2.stride(0), B_fc2.stride(1),
            out_final.stride(0), out_final.stride(1),
            BLOCK_M=128, BLOCK_N=128, BLOCK_K=64,
        )

        return out_final


def run(*args):
    return ModelNew()(*args)
