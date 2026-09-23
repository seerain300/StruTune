import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: LayerNorm per row of hidden [N, C], output bfloat16
# hidden_ptr: *bf16, [N, C]
# ln_weight_ptr, ln_bias_ptr: *bf16, [C]
# out_ptr: *bf16, [N, C]
# N: int, C: int, eps: float
@triton.jit
def layernorm_affine_kernel(
    hidden_ptr, ln_weight_ptr, ln_bias_ptr, out_ptr,
    N, C, eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= N:
        return
    # compute mean and variance in float32
    sum_x = 0.0
    sum_x2 = 0.0
    c = 0
    while c < C:
        offs = c + tl.arange(0, BLOCK)
        mask = offs < C
        x = tl.load(hidden_ptr + row * C + offs, mask=mask, other=0.0).to(tl.float32)
        sum_x += tl.sum(x, axis=0)
        sum_x2 += tl.sum(x * x, axis=0)
        c += BLOCK

    mean = sum_x / C
    var = sum_x2 / C - mean * mean
    rstd = tl.rsqrt(var + eps)

    # normalize and affine
    c = 0
    while c < C:
        offs = c + tl.arange(0, BLOCK)
        mask = offs < C
        x = tl.load(hidden_ptr + row * C + offs, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(ln_weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(ln_bias_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        y = (x - mean) * rstd
        y = y * w + b
        tl.store(out_ptr + row * C + offs, y.to(tl.bfloat16), mask=mask)
        c += BLOCK


# Triton kernel: per-grid spatial shuffle (T, H, W, C -> output [T*H/2*W/2, 4*C])
# in_ptr: *bf16, [T*H*W, C] (this is a contiguous flattened view of one grid's hidden_norm)
# ln_out_ptr: *bf16, output [T*H_new*W_new, 4*C], where H_new=H//2, W_new=W//2
# T, H, W, C, eps are not needed here; we assume merge_size=2 as in the original code.
@triton.jit
def spatial_shuffle_per_grid_kernel(
    in_ptr, out_ptr,
    T, H, W, C,
    out_rows,  # T * (H//2) * (W//2)
    BLOCK_M: tl.constexpr,  # number of output rows per program
    BLOCK_N: tl.constexpr,  # features per program (4*C)
):
    pid_m = tl.program_id(0)  # over output rows
    pid_n = tl.program_id(1)  # over feature blocks

    # Each program computes a BLOCK_N-wide slice over features for a BLOCK_M-wide slice over output rows.
    # We'll vectorize across features (pid_n) and loop over output rows (pid_m).
    # For each output row j in [pid_m * BLOCK_M, ...,], determine (t, h2, w2) and copy the 2x2 patch features.
    # out_rows = T * (H//2) * (W//2)
    # output ordering: for each (t, h2, w2), we write rows corresponding to (h, w) in original:
    # original h = 2*h2 + [0,1], w = 2*w2 + [0,1], and fuse features for each (h,w) into 4*C.

    # Precompute constants
    H2 = H // 2
    W2 = W // 2

    # We iterate over output rows; for each row j, map to (t, h2, w2):
    # j = t * (H2 * W2) + h2 * W2 + w2
    # But we want vectorization. Instead, we compute for a fixed (h2, w2) across batch t and across t.
    # We'll use a grid: pid_m over output rows. For each output row j, compute t = j // (H2*W2),
    # rem = j % (H2*W2), h2 = rem // W2, w2 = rem % W2.
    # Then for that (t,h2,w2), we need to gather 4*C features from the 2x2 original positions:
    # pos00: (2*h2, 2*w2), pos01: (2*h2, 2*w2+1), pos10: (2*h2+1, 2*w2), pos11: (2*h2+1, 2*w2+1)
    # Feature index is C vector. We will write y[k] for k in [0, 4*C) where:
    # y[0:C-1] = pos00 features, y[C:2*C-1] = pos01, y[2*C:3*C-1] = pos10, y[3*C:4*C-1] = pos11.

    # feature block
    c_block = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    # We'll process one output row per iteration inside this kernel to keep indexing simple.
    # Launch grid should cover all output rows. For simplicity, BLOCK_M=1. This kernel is not vectorized
    # across output rows in a single program, so we set BLOCK_M=1 to avoid complex indexing.
    j = pid_m  # only one j per program
    if j >= out_rows:
        return

    # map j to (t, h2, w2)
    HW2 = H2 * W2
    t = j // HW2
    rem = j % HW2
    h2 = rem // W2
    w2 = rem % W2

    # original positions
    h0 = 2 * h2
    w0 = 2 * w2
    h1 = h0 + 1
    w1 = w0 + 1

    # For each original (h,w), we copy C features into the fused 4*C vector at positions determined by c_block.
    # We'll compute in chunks of BLOCK_N across c_block, but since 4*C=6144, we can set BLOCK_N=6144 to cover all features.

    # We need to loop over c = 0..C-1 in chunks to handle larger C. However, Triton doesn't support dynamic
    # loop bounds with unknown C, so we set BLOCK_N=6144 to cover all features for this task.
    # We'll implement the feature copy assuming BLOCK_N >= 4*C. If BLOCK_N < 4*C, we can either set BLOCK_N=6144
    # or fallback to PyTorch; here we set BLOCK_N=6144 to match the given workload (C=1536, 4*C=6144).

    # To keep the kernel general, we'll implement feature copying in a loop over c in chunks of 1024.
    # But Triton requires compile-time BLOCK_N, so we set BLOCK_N=6144.

    # Feature copying: write y[k] = in[t, h, w, c_block]
    # We will manually construct y per k segment [0:C), [C:2C), [2C:3C), [3C:4C)
    # We'll use c_block shifted appropriately for each of the four positions.

    # Segment 0: c in [0, C)
    c0 = c_block  # 0..BLOCK_N-1, but we only need first C. We'll mask using c0 < C.
    mask0 = c0 < C
    # feature indices for pos00
    ptr00 = in_ptr + (t * (H * W) + (h0 * W + w0)) * C + c0
    # load and store into out[j, 0:C)
    y00 = tl.load(ptr00, mask=mask0, other=0.0).to(tl.bfloat16)
    tl.store(out_ptr + j * (4 * C) + c0, y00, mask=mask0)

    # feature indices for pos01
    ptr01 = in_ptr + (t * (H * W) + (h0 * W + w1)) * C + c0
    y01 = tl.load(ptr01, mask=mask0, other=0.0).to(tl.bfloat16)
    tl.store(out_ptr + j * (4 * C) + C + c0, y01, mask=mask0)

    # feature indices for pos10
    ptr10 = in_ptr + (t * (H * W) + ((h0 + 1) * W + w0)) * C + c0
    y10 = tl.load(ptr10, mask=mask0, other=0.0).to(tl.bfloat16)
    tl.store(out_ptr + j * (4 * C) + 2 * C + c0, y10, mask=mask0)

    # feature indices for pos11
    ptr11 = in_ptr + (t * (H * W) + ((h0 + 1) * W + w1)) * C + c0
    y11 = tl.load(ptr11, mask=mask0, other=0.0).to(tl.bfloat16)
    tl.store(out_ptr + j * (4 * C) + 3 * C + c0, y11, mask=mask0)


# Triton kernel: matmul without bias (A[M, K] @ W[K, N], outputs C[M, N] in fp32)
# A_ptr: *bf16, [M, K]
# W_ptr: *bf16, [K, N]
# C_ptr: *bf32, [M, N]
@triton.jit
def triton_matmul_nobias(
    A_ptr, W_ptr, C_ptr,
    M, K, N,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        a = tl.load(A_ptr + (offs_m[:, None] * K) + offs_k[None, :],
                    mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
                    other=0.0).to(tl.float32)
        b = tl.load(W_ptr + (offs_k[:, None] * N) + offs_n[None, :],
                    mask=(offs_k[:, None] < K) & (offs_n[None, :] < N),
                    other=0.0).to(tl.float32)
        acc += tl.dot(a, b)

    tl.store(C_ptr + (offs_m[:, None] * N) + offs_n[None, :],
             acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


# Triton elementwise GELU on fp32 input, store fp32
@triton.jit
def gelu_kernel(
    x_ptr, y_ptr,
    M, N,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptr + offs_m[:, None] * N + offs_n[None, :], mask=mask, other=0.0).to(tl.float32)
    # GELU: 0.5 * x * (1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))
    c = 0.044715
    sqrt_2_over_pi = 0.7978845608028654
    x3 = x * x * x
    y = 0.5 * x * (1.0 + tl.tanh(sqrt_2_over_pi * (x + c * x3)))
    tl.store(y_ptr + offs_m[:, None] * N + offs_n[None, :], y, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # fixed constants
        self.hidden_size = 1536
        self.hidden_size_expanded = 6144
        self.out_hidden_size = 3584
        self.eps = 1e-6
        # merge_size is 2 as in the original code
        self.merge_size = 2

    def forward(self, hidden: torch.Tensor,
                grid_thw: torch.Tensor,
                ln_weight: torch.Tensor, ln_bias: torch.Tensor,
                fc1_weight: torch.Tensor, fc1_bias: torch.Tensor,
                fc2_weight: torch.Tensor, fc2_bias: torch.Tensor):
        """
        hidden: [num_patches, hidden_size] (bfloat16)
        grid_thw: [num_grids, 3], int64, per-grid (T, H, W)
        ln_weight, ln_bias: [hidden_size], bfloat16
        fc1_weight: [hidden_size_expanded, hidden_size_expanded], bfloat16
        fc1_bias: [hidden_size_expanded], bfloat16
        fc2_weight: [out_hidden_size, hidden_size_expanded], bfloat16
        fc2_bias: [out_hidden_size], bfloat16
        """
        # Ensure on CUDA
        device = hidden.device
        assert device.type == "cuda", "ModelNew requires CUDA device"
        hidden = hidden.contiguous()
        ln_weight = ln_weight.contiguous()
        ln_bias = ln_bias.contiguous()
        fc1_weight = fc1_weight.contiguous()
        fc1_bias = fc1_bias.contiguous()
        fc2_weight = fc2_weight.contiguous()
        fc2_bias = fc2_bias.contiguous()
        grid_thw = grid_thw.to(torch.int32).contiguous()

        N = hidden.shape[0]
        C = self.hidden_size

        # 1) LayerNorm + affine
        hidden_norm = torch.empty_like(hidden)  # bfloat16 output
        # Launch Triton LN kernel
        # BLOCK chosen as 256 for balance; loop covers C=1536
        BLOCK = 256
        grid = (N,)
        layernorm_affine_kernel[grid](hidden, ln_weight, ln_bias, hidden_norm, N, C, self.eps, BLOCK=BLOCK)

        # 2) Spatial shuffle per grid
        num_grids = grid_thw.shape[0]
        # We need to reconstruct the per-grid outputs using Triton. The original helper constructs:
        # For each grid, take a subset of hidden_norm corresponding to the grid's num_patches_this and perform:
        # reshape to (T, H, W, C), permute, and flatten to [T*(H//2)*(W//2), 4*C].
        # We must compute total processed so far to slice hidden_norm. We'll iterate grids and compute offsets.

        # Compute total patches accounted so far for each grid to slice hidden_norm
        offset = 0
        per_grid_outputs = []
        for i in range(num_grids):
            T = int(grid_thw[i, 0].item())
            H = int(grid_thw[i, 1].item())
            W = int(grid_thw[i, 2].item())
            # num_patches_this = T * H * W
            num_patches_this = T * H * W

            # Slice hidden_norm: indices are offset .. offset + num_patches_this - 1
            in_slice = hidden_norm[offset:offset + num_patches_this]
            # in_slice shape [num_patches_this, C], make contiguous
            in_slice = in_slice.contiguous()

            # Output shape [T * (H//2) * (W//2), 4*C] = [rows, 6144]
            H2 = H // self.merge_size
            W2 = W // self.merge_size
            rows = T * H2 * W2
            out = torch.empty((rows, self.hidden_size * 4), dtype=torch.bfloat16, device=device)

            # Launch Triton spatial_shuffle_per_grid kernel once per grid; since rows may vary, we set grid = (rows, 1)
            # We choose BLOCK_N = 4*C = 6144 to cover all features; pid_m iterates rows; pid_n is 1.
            grid_shuffle = (rows, 1)
            spatial_shuffle_per_grid_kernel[grid_shuffle](
                in_slice, out,
                T, H, W, C, rows,
                BLOCK_M=1, BLOCK_N=6144
            )
            per_grid_outputs.append(out)
            offset += num_patches_this

        # Concatenate per-grid outputs along rows: shape [num_merged_patches, 4*C]
        hidden_shuffled = torch.cat(per_grid_outputs, dim=0)  # [num_merged_patches, 6144], bfloat16

        # 3) fc1: linear (M=6144 input -> 6144 output), bias add, then GELU
        M = hidden_shuffled.shape[0]
        K = self.hidden_size_expanded  # 6144
        N1 = K  # output features (6144)
        # Allocate fp32 output for matmul
        fc1_out = torch.empty((M, N1), dtype=torch.float32, device=device)
        # Launch Triton matmul without bias
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 64
        grid_fc1 = (M // BLOCK_M + (M % BLOCK_M != 0), N1 // BLOCK_N + (N1 % BLOCK_N != 0))
        triton_matmul_nobias[grid_fc1](hidden_shuffled.to(torch.bfloat16), fc1_weight.to(torch.bfloat16), fc1_out,
                                       M, K, N1, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K)

        # Add fc1 bias (fp32)
        fc1_bias_fp32 = fc1_bias.to(torch.float32)
        fc1_out = fc1_out + fc1_bias_fp32  # broadcast over rows

        # GELU activation using Triton
        fc1_out_gelu = torch.empty_like(fc1_out, dtype=torch.float32)
        grid_gelu = (M // BLOCK_M + (M % BLOCK_M != 0), N1 // BLOCK_N + (N1 % BLOCK_N != 0))
        gelu_kernel[grid_gelu](fc1_out, fc1_out_gelu, M, N1, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)

        # 4) fc2: linear (M input -> out_hidden_size=3584 output), no bias in Triton, add bias afterwards
        N2 = self.out_hidden_size  # 3584
        fc2_out = torch.empty((M, N2), dtype=torch.float32, device=device)
        grid_fc2 = (M // BLOCK_M + (M % BLOCK_M != 0), N2 // BLOCK_N + (N2 % BLOCK_N != 0))
        triton_matmul_nobias[grid_fc2](fc1_out_gelu, fc2_weight.to(torch.bfloat16).to(torch.float32), fc2_out,
                                       M, N1, N2, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K)

        # Add fc2 bias (fp32)
        fc2_bias_fp32 = fc2_bias.to(torch.float32)
        fc2_out = fc2_out + fc2_bias_fp32

        # Return as bfloat16 to match original final dtype
        output = fc2_out.to(torch.bfloat16)

        return output


def run(*args):
    return ModelNew()(*args)
