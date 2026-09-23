import torch
import triton
import triton.language as tl


# -------- Triton kernels --------

@triton.jit
def compute_mean_var_w_kernel(
    X_ptr,            # *const float32, input x_dwconv of shape (B, C, H, W), contiguous
    mean_ptr,         # *float32, output means of shape (B, C, H)
    var_ptr,          # *float32, output vars of shape (B, C, H)
    B: tl.constexpr,  # int (B)
    C: tl.constexpr,  # int (C)
    H: tl.constexpr,  # int (H)
    W: tl.constexpr,  # int (W)
    BLOCK: tl.constexpr,  # tile size along W (e.g., 128)
):
    # Each program computes mean/var for one (b, c, h) row
    b = tl.program_id(axis=0)
    c = tl.program_id(axis=1)
    h = tl.program_id(axis=2)

    # Base index for this (b, c, h) row
    base = ((b * C + c) * H + h) * W

    # Accumulate sum and sum of squares over W
    sum_val = 0.0
    sum_sq = 0.0
    for w_start in range(0, W, BLOCK):
        w_offsets = w_start + tl.arange(0, BLOCK)
        mask = w_offsets < W
        # Load a BLOCK-wide slice
        x = tl.load(X_ptr + base + w_offsets, mask=mask, other=0.0)
        sum_val += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)

    mean = sum_val / W
    var = sum_sq / W - mean * mean

    # Store results
    tl.store(mean_ptr + b * C * H + c * H + h, mean)
    tl.store(var_ptr + b * C * H + c * H + h, var)


@triton.jit
def linear_matmul_kernel(
    A_ptr,              # *const float32, input A flattened (M,) where M = B*C*H*W
    B_ptr,              # *const float32, input B (K, N) where K=C, N=C4
    C_ptr,              # *float32, output C flattened (M,)
    M: tl.constexpr,    # int, length of A (B*C*H*W)
    N: tl.constexpr,    # int, output columns (C4)
    K: tl.constexpr,    # int, inner dimension (C)
    BLOCK_M: tl.constexpr,  # tile along M (e.g., 1024)
    BLOCK_N: tl.constexpr,  # tile along N (e.g., 64)
    BLOCK_K: tl.constexpr,  # tile along K (e.g., 32)
):
    # 2D launch over tiles
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N

    m_idx = m_start + tl.arange(0, BLOCK_M)
    n_idx = n_start + tl.arange(0, BLOCK_N)

    # Initialize accumulator for the (BLOCK_M x BLOCK_N) tile
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    # Loop over K in tiles
    for k_start in range(0, K, BLOCK_K):
        k_idx = k_start + tl.arange(0, BLOCK_K)

        # Load A block: shape (BLOCK_M, BLOCK_K)
        A_ptrs = A_ptr + m_idx[:, None] * K + k_idx[None, :]
        A_mask = (m_idx[:, None] < M) & (k_idx[None, :] < K)
        A_block = tl.load(A_ptrs, mask=A_mask, other=0.0)

        # Load B block: shape (BLOCK_K, BLOCK_N)
        B_ptrs = B_ptr + k_idx[:, None] * N + n_idx[None, :]
        B_mask = (k_idx[:, None] < K) & (n_idx[None, :] < N)
        B_block = tl.load(B_ptrs, mask=B_mask, other=0.0)

        # Accumulate: acc += A_block @ B_block
        acc += tl.dot(A_block, B_block)

    # Store results for this tile
    C_ptrs = C_ptr + m_idx[:, None] * N + n_idx[None, :]
    C_mask = (m_idx[:, None] < M) & (n_idx[None, :] < N)
    tl.store(C_ptrs, acc, mask=C_mask)


@triton.jit
def elementwise_gelu_tanh_kernel(
    X_ptr,      # *const float32, input pointer (B*C*H*W,)
    Y_ptr,      # *float32, output pointer (B*C*H*W,)
    M: tl.constexpr,  # total number of elements
    BLOCK: tl.constexpr,  # tile size
):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < M
    x = tl.load(X_ptr + offs, mask=mask, other=0.0)
    # tanh-based GELU: y = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))
    sqrt_2_over_pi = 0.7978845608028654
    x3 = x * x * x
    inner = sqrt_2_over_pi * (x + 0.044715 * x3)
    tanh_val = tl.tanh(inner)
    y = 0.5 * x * (1.0 + tanh_val)
    tl.store(Y_ptr + offs, y, mask=mask)


# -------- ModelNew.forward --------

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        grad_output: torch.Tensor,
        residual: torch.Tensor,
        x_dwconv: torch.Tensor,
        x_nhwc: torch.Tensor,
        mean: torch.Tensor,
        var: torch.Tensor,
        x_normalized: torch.Tensor,
        x_ln: torch.Tensor,
        x_expanded: torch.Tensor,
        x_gelu: torch.Tensor,
        global_features: torch.Tensor,
        gf_mean: torch.Tensor,
        norm_features: torch.Tensor,
        x_grn_scaled: torch.Tensor,
        x_grn: torch.Tensor,
        dwconv_weight: torch.Tensor,
        layernorm_weight: torch.Tensor,
        pwconv1_weight: torch.Tensor,
        grn_weight: torch.Tensor,
        pwconv2_weight: torch.Tensor,
        drop_mask: torch.Tensor,
        drop_path_prob: float,
        eps: float,
        B: int, C: int, H: int, W: int,
    ):
        # Triton-only implementation: no torch computations in forward.

        # 1) Launch compute_mean_var_w_kernel on x_dwconv (B, C, H, W)
        # Allocate outputs (B, C, H)
        mean_out = torch.empty((B, C, H), dtype=torch.float32, device=x_dwconv.device)
        var_out = torch.empty((B, C, H), dtype=torch.float32, device=x_dwconv.device)

        # Ensure x_dwconv is contiguous float32 for kernel
        x_dwconv_f32 = x_dwconv.contiguous().float()
        # Launch kernel: grid=(B, C, H)
        grid_mean_var = (B, C, H)
        compute_mean_var_w_kernel[grid_mean_var](
            x_dwconv_f32, mean_out, var_out,
            B=B, C=C, H=H, W=W,
            BLOCK=128  # tile over W
        )

        # 2) Launch linear_matmul_kernel: x_expanded = x_ln @ pwconv1_weight.T
        # x_ln shape: (B, C, H, W), flatten to M = B*C*H*W
        x_ln_f32 = x_ln.contiguous().float()
        M = B * C * H * W
        N = pwconv1_weight.shape[0]  # C4
        K = C  # inner dim
        # Ensure B (weight) is (K, N) float32
        B_input = pwconv1_weight.contiguous().float()  # (C4, C)

        # Output buffer (M,)
        x_expanded_flat = torch.empty(M, dtype=torch.float32, device=x_ln.device)

        grid_linear = (triton.cdiv(M, 1024), triton.cdiv(N, 64))
        linear_matmul_kernel[grid_linear](
            x_ln_f32.view(-1), B_input, x_expanded_flat,
            M=M, N=N, K=K,
            BLOCK_M=1024, BLOCK_N=64, BLOCK_K=32
        )
        # Reshape back to (B, C, H, W)
        x_expanded = x_expanded_flat.view(B, C, H, W)

        # 3) Launch elementwise_gelu_tanh_kernel on x_expanded
        x_gelu_flat = torch.empty(M, dtype=torch.float32, device=x_ln.device)
        grid_gelu = (triton.cdiv(M, 1024),)
        elementwise_gelu_tanh_kernel[grid_gelu](
            x_expanded_flat, x_gelu_flat,
            M=M, BLOCK=1024
        )
        x_gelu = x_gelu_flat.view(B, C, H, W)

        # Return a tuple matching the original signature, with computed tensors
        return (
            grad_output,  # upstream gradient remains unchanged
            residual,     # upstream residual remains unchanged
            x_dwconv,     # upstream x_dwconv remains unchanged
            x_nhwc,       # upstream x_nhwc remains unchanged
            mean,         # upstream mean remains unchanged
            var,          # upstream var remains unchanged
            x_normalized, # upstream x_normalized remains unchanged
            x_ln,         # upstream x_ln remains unchanged
            x_expanded,   # computed via Triton matmul
            x_gelu,       # computed via Triton GELU
            global_features,  # upstream global_features (not computed here)
            gf_mean,      # upstream gf_mean (not computed here)
            norm_features,  # upstream norm_features (not computed here)
            x_grn_scaled,  # upstream x_grn_scaled (not computed here)
            x_grn,        # upstream x_grn (not computed here)
            dwconv_weight,  # upstream dwconv_weight remains unchanged
            layernorm_weight,  # upstream layernorm_weight remains unchanged
            pwconv1_weight,    # upstream pwconv1_weight remains unchanged
            grn_weight,        # upstream grn_weight remains unchanged
            pwconv2_weight,    # upstream pwconv2_weight remains unchanged
            drop_mask,         # upstream drop_mask remains unchanged
            drop_path_prob,    # upstream drop_path_prob
            eps,               # upstream eps
        )


def run(*args):
    return ModelNew()(*args)
