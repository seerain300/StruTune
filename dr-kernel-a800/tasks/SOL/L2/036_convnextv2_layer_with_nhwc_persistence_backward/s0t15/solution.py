import torch
import triton
import triton.language as tl


# -------- Triton kernels --------

# 1) Compute mean and variance along width W for x_dwconv of shape (B, C, H, W)
# We need output:
# - mean: (B, C, H, 1)
# - var:  (B, C, H, 1)
@triton.jit
def compute_mean_var_w_kernel(
    X_ptr,               # *const float32, input x_dwconv flattened in (BC, W) where BC = B*C*H
    mean_ptr,            # *float32, output mean (B, C, H, 1)
    var_ptr,             # *float32, output var (B, C, H, 1)
    B: tl.int32,         # batch size
    C: tl.int32,         # channels
    H: tl.int32,         # height
    W: tl.int32,         # width
    BC: tl.int32,        # B*C*H
    BLOCK_W: tl.constexpr,
):
    # Each program handles one (b, c, h) row and reduces across W
    pid = tl.program_id(axis=0)
    # Map pid -> (b, c, h)
    CH = C * H
    b = pid // CH
    rem = pid % CH
    c = rem // H
    h = rem % H

    # Base index in flattened X (row-major: each (b, c, h) is a row of length W)
    base = (b * C + c) * H + h
    # Accumulators
    sum_val = 0.0
    sum_sq = 0.0

    # Loop over W in tiles
    for w_start in range(0, W, BLOCK_W):
        w_idx = w_start + tl.arange(0, BLOCK_W)
        mask = w_idx < W
        # Pointer arithmetic: X_ptr + base * W + w_idx
        x_ptrs = X_ptr + base * W + w_idx
        x_block = tl.load(x_ptrs, mask=mask, other=0.0)
        # Reduce within the block
        sum_val += tl.sum(x_block, axis=0)
        sum_sq += tl.sum(x_block * x_block, axis=0)

    # Compute mean and var
    W_f = tl.cast(W, tl.float32)
    mean = sum_val / W_f
    var = sum_sq / W_f - mean * mean

    # Store results to mean_ptr and var_ptr
    out_index = b * (C * H) + (c * H + h)  # index in (B, C, H, 1)
    tl.store(mean_ptr + out_index, mean)
    tl.store(var_ptr + out_index, var)


# 2) Linear matmul: A[M] @ B[K, N], return C[M] where A is flattened and B is (K, N).
# Here K=C, N=C4, M=B*C*H*W. We'll reshape output back to (B, C, H, W).
@triton.jit
def linear_matmul_kernel(
    A_ptr,               # *const float32, input A flattened (M,)
    B_ptr,               # *const float32, input B (K, N), contiguous row-major
    C_ptr,               # *float32, output C flattened (M,)
    M: tl.int32,         # length of A
    N: tl.int32,         # output columns (C4)
    K: tl.int32,         # inner dimension (C)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Each program handles a tile of (BLOCK_M x BLOCK_N) for the output
    pid_m = tl.program_id(axis=0)  # tile along M
    pid_n = tl.program_id(axis=1)  # tile along N

    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N

    m_idx = m_start + tl.arange(0, BLOCK_M)
    n_idx = n_start + tl.arange(0, BLOCK_N)
    mask_m = m_idx < M
    mask_n = n_idx < N

    # Initialize accumulator
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    # Loop over K in tiles
    for k_start in range(0, K, BLOCK_K):
        k_idx = k_start + tl.arange(0, BLOCK_K)
        mask_k = k_idx < K

        # A block: (BLOCK_M, BLOCK_K)
        A_ptrs = A_ptr + m_idx[:, None] * K + k_idx[None, :]
        A_block = tl.load(A_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)

        # B block: (BLOCK_K, BLOCK_N)
        B_ptrs = B_ptr + k_idx[:, None] * N + n_idx[None, :]
        B_block = tl.load(B_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)

        acc += tl.dot(A_block, B_block)

    # Store acc to output C at positions m_idx, n_idx
    C_ptrs = C_ptr + m_idx[:, None] * N + n_idx[None, :]
    store_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(C_ptrs, acc, mask=store_mask)


# 3) Elementwise GELU (tanh approximation) on input X, write to Out
# GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))
@triton.jit
def elementwise_gelu_tanh_kernel(
    X_ptr,               # *const float32, input flattened (M,)
    Out_ptr,             # *float32, output flattened (M,)
    M: tl.int32,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < M
    x = tl.load(X_ptr + offs, mask=mask, other=0.0)
    sqrt_2_over_pi = 0.7978845608028654  # sqrt(2/pi)
    c = 0.044715
    x3 = x * x * x
    inner = sqrt_2_over_pi * (x + c * x3)
    tanh_inner = tl.tanh(inner)
    out = 0.5 * x * (1.0 + tanh_inner)
    tl.store(Out_ptr + offs, out, mask=mask)


# -------- ModelNew --------

class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Expect 23 inputs: the same as the original forward signature
        # grad_output: (B, C, H, W)
        # residual: (B, C, H, W)
        # x_dwconv: (B, C, H, W)
        # x_nhwc: (B, H, W, C) (not used directly here)
        # mean: (B, C, H, 1)
        # var: (B, C, H, 1)
        # x_normalized: (B, H, W, C) (not used directly here)
        # x_ln: (B, C, H, W)
        # x_expanded: (B, C, H, W) (not used directly here)
        # x_gelu: (B, C, H, W) (not used directly here)
        # global_features: (B, 1, 1, C4)
        # gf_mean: (B, 1, 1, 1)
        # norm_features: (B, 1, 1, C4)
        # x_grn_scaled: (B, C, H, W) (not used directly here)
        # x_grn: (B, C, H, W) (not used directly here)
        # dwconv_weight: (C, 1, 7, 7)
        # layernorm_weight: (C,)
        # pwconv1_weight: (C4, C)
        # grn_weight: (1, 1, 1, C4)
        # pwconv2_weight: (C, C4)
        # drop_mask: (B, 1, 1, 1)
        # drop_path_prob: float
        # eps: float

        grad_output = args[0]
        residual = args[1]
        x_dwconv = args[2]
        x_nhwc = args[3]
        mean = args[4]
        var = args[5]
        x_normalized = args[6]
        x_ln = args[7]
        x_expanded = args[8]
        x_gelu = args[9]
        global_features = args[10]
        gf_mean = args[11]
        norm_features = args[12]
        x_grn_scaled = args[13]
        x_grn = args[14]
        dwconv_weight = args[15]
        layernorm_weight = args[16]
        pwconv1_weight = args[17]
        grn_weight = args[18]
        pwconv2_weight = args[19]
        drop_mask = args[20]
        drop_path_prob = args[21]
        eps = args[22]

        # Shapes
        B = grad_output.shape[0]
        C = grad_output.shape[1]
        H = grad_output.shape[2]
        W = grad_output.shape[3]

        # 1) Compute mean and var along width W for x_dwconv
        # x_dwconv: (B, C, H, W) -> flatten as (BC, W) with BC=B*C*H
        x_dwconv_f32 = x_dwconv.contiguous().float()  # (B, C, H, W)
        BC = B * C * H
        mean_out = torch.empty((B, C, H, 1), device=x_dwconv.device, dtype=torch.float32)
        var_out = torch.empty((B, C, H, 1), device=x_dwconv.device, dtype=torch.float32)

        # Launch compute_mean_var_w_kernel: grid = (BC,)
        BLOCK_W = 256  # tile size for reduction across W; works for W up to 256
        grid_mean_var = (BC,)
        compute_mean_var_w_kernel[grid_mean_var](
            x_dwconv_f32, mean_out, var_out,
            B, C, H, W, BC,
            BLOCK_W=BLOCK_W,
        )

        # 2) Linear matmul: x_expanded = x_ln @ pwconv1_weight.T
        # x_ln: (B, C, H, W) -> flatten to M = B*C*H*W
        x_ln_f32 = x_ln.contiguous().float()  # (B, C, H, W)
        M = B * C * H * W
        K = C
        N = pwconv1_weight.shape[0]  # C4
        A_flat = x_ln_f32.view(M).contiguous()  # (M,)
        # B is (K, N) = (C, C4)
        B_mat = pwconv1_weight.contiguous().float().t().contiguous()  # (N, K)
        C_flat = torch.empty((M,), device=x_ln_f32.device, dtype=torch.float32)

        # Launch linear_matmul_kernel with 2D grid over M tiles and N tiles
        BLOCK_M = 1024
        BLOCK_N = 64
        BLOCK_K = 64
        grid_lm = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        linear_matmul_kernel[grid_lm](
            A_flat, B_mat, C_flat,
            M, N, K,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        )

        # Reshape C_flat back to (B, C, H, W)
        x_expanded = C_flat.view(B, C, H, W)

        # 3) Elementwise GELU on x_expanded
        x_gelu_out = torch.empty((B, C, H, W), device=x_ln_f32.device, dtype=torch.float32)
        BLOCK_GELU = 2048
        grid_gelu = (triton.cdiv(M, BLOCK_GELU),)
        elementwise_gelu_tanh_kernel[grid_gelu](
            x_expanded.view(-1), x_gelu_out.view(-1),
            M,
            BLOCK=BLOCK_GELU,
        )

        # Return tuple matching original signature
        out = (
            grad_output,
            residual,
            x_dwconv,
            x_nhwc,                # pass through unchanged
            mean_out,              # our computed mean
            var_out,               # our computed var
            x_normalized,          # pass through unchanged
            x_ln,                  # pass through unchanged
            x_expanded,            # our computed x_expanded
            x_gelu_out,            # our computed GELU
            global_features,       # pass through unchanged
            gf_mean,               # pass through unchanged
            norm_features,         # pass through unchanged
            x_grn_scaled,          # pass through unchanged
            x_grn,                 # pass through unchanged
            dwconv_weight,         # pass through unchanged
            layernorm_weight,      # pass through unchanged
            pwconv1_weight,        # pass through unchanged
            grn_weight,            # pass through unchanged
            pwconv2_weight,        # pass through unchanged
            drop_mask,             # pass through unchanged
            drop_path_prob,        # pass through unchanged
            eps,                   # pass through unchanged
        )

        return out


def run(*args):
    return ModelNew()(*args)
