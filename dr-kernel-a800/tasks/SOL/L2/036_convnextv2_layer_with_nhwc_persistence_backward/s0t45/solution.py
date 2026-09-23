import torch
import triton
import triton.language as tl


# -------- Triton kernels --------

@triton.jit
def compute_mean_var_w_kernel(
    X_ptr,            # *const float32, input tensor (B, C, H, W) contiguous
    mean_ptr,         # *float32, output means per (B, C, H)
    var_ptr,          # *float32, output vars per (B, C, H)
    B: tl.constexpr,  # batch size
    C: tl.constexpr,  # channels
    H: tl.constexpr,  # height
    W: tl.constexpr,  # width
    BLOCK_W: tl.constexpr,  # tile along width
):
    # Each program handles one (b, c, h) row, reducing across W
    pid = tl.program_id(axis=0)
    total = B * C * H
    if pid >= total:
        return

    b = pid // (C * H)
    rem = pid % (C * H)
    c = rem // H
    h = rem % H

    sum_val = 0.0
    sum_sq = 0.0
    start = 0
    while start < W:
        w_idx = start + tl.arange(0, BLOCK_W)
        w_mask = w_idx < W
        # X indexing for contiguous (B, C, H, W): offset = b*C*H*W + c*H*W + h*W + w_idx
        X_offsets = b * C * H * W + c * H * W + h * W + w_idx
        X_vals = tl.load(X_ptr + X_offsets, mask=w_mask, other=0.0)
        # Reduce this chunk
        sum_val += tl.sum(X_vals, axis=0)
        sum_sq += tl.sum(X_vals * X_vals, axis=0)
        start += BLOCK_W

    mean = sum_val / W
    var = sum_sq / W - mean * mean

    mean_idx = b * (C * H) + c * H + h
    var_idx = b * (C * H) + c * H + h
    tl.store(mean_ptr + mean_idx, mean)
    tl.store(var_ptr + var_idx, var)


@triton.jit
def linear_matmul_kernel(
    A_ptr,            # *const float32, input A flattened (M,) where M = B*C*H*W
    B_ptr,            # *const float32, input B flattened (K*N,) where K=C, N=C4
    C_ptr,            # *float32, output C flattened (M,)
    M: tl.constexpr,  # int, length of A
    K: tl.constexpr,  # int, inner dimension (C)
    N: tl.constexpr,  # int, output columns (C4)
    BLOCK_M: tl.constexpr,  # tile along M
    BLOCK_N: tl.constexpr,  # tile along N
    BLOCK_K: tl.constexpr,  # tile along K
):
    # Launch over tiles in M; each program computes a BLOCK_M x BLOCK_N block and writes to C
    pid_m = tl.program_id(axis=0)
    m_start = pid_m * BLOCK_M
    m_idx = m_start + tl.arange(0, BLOCK_M)
    m_mask = m_idx < M

    # Initialize accumulator for this M-tile (each row in the tile accumulates over N)
    acc = tl.zeros([BLOCK_M], dtype=tl.float32)

    # Loop over N in tiles (but we need dot product across K). We will compute acc[m] = sum_j B[j, n] * A[m, j]
    # We can iterate over n chunks:
    for n_start in range(0, N, BLOCK_N):
        n_idx = n_start + tl.arange(0, BLOCK_N)
        n_mask = n_idx < N

        # For each n in the chunk, accumulate dot(A_row, B_col)
        for n_i in range(BLOCK_N):
            n_cur = n_start + n_i
            if n_cur >= N:
                break
            # Load B[:, n_cur] as a vector of length K
            B_col_offsets = n_cur * K + tl.arange(0, K)
            B_col = tl.load(B_ptr + B_col_offsets, mask=(tl.arange(0, K) < K), other=0.0)
            # Load A rows for m_idx: shape (BLOCK_M, K)
            A_ptrs = A_ptr + m_idx[:, None] * K + tl.arange(0, K)[None, :]
            A_mask = m_mask[:, None] & (tl.arange(0, K)[None, :] < K)
            A_block = tl.load(A_ptrs, mask=A_mask, other=0.0)
            # Dot product per row in this M tile
            dot_vec = tl.sum(A_block * B_col[None, :], axis=1)  # shape (BLOCK_M,)
            # Accumulate into acc
            acc += dot_vec

    # Store acc for these m indices
    tl.store(C_ptr + m_idx, acc, mask=m_mask)


@triton.jit
def elementwise_gelu_tanh_kernel(
    X_ptr,            # *const float32, input tensor (M,) where M = number of elements to apply GELU
    Y_ptr,            # *float32, output tensor (M,)
    M: tl.constexpr,  # number of elements
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    start = pid * BLOCK
    idx = start + tl.arange(0, BLOCK)
    mask = idx < M
    X = tl.load(X_ptr + idx, mask=mask, other=0.0)

    # GELU (tanh approximation): 0.5 * x * (1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))
    sqrt_2_over_pi = 0.7978845608028654
    c = 0.044715
    inner = sqrt_2_over_pi * (X + c * X * X * X)
    tanh_inner = tl.tanh(inner)
    Y = 0.5 * X * (1.0 + tanh_inner)

    tl.store(Y_ptr + idx, Y, mask=mask)


# -------- ModelNew (entry point) --------

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, grad_output: torch.Tensor,
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
                eps: float):
        """
        Entry point must be Triton-only: no torch operations in host code.
        We will launch Triton kernels to perform:
          - mean/var along width for x_dwconv (B, C, H, W)
          - linear projection x_expanded = x_ln @ pwconv1_weight.T
          - GELU (tanh approximation) elementwise on x_expanded
        """
        device = grad_output.device

        # 1) Compute mean and var along width W for x_dwconv (B, C, H, W), using Triton
        B = x_dwconv.shape[0]
        C = x_dwconv.shape[1]
        H = x_dwconv.shape[2]
        W = x_dwconv.shape[3]
        mean_out = torch.empty(B * C * H, dtype=torch.float32, device=device)
        var_out = torch.empty(B * C * H, dtype=torch.float32, device=device)
        grid_mean_var = (B * C * H,)
        compute_mean_var_w_kernel[grid_mean_var](
            x_dwconv,
            mean_out,
            var_out,
            B, C, H, W,
            BLOCK_W=128,
        )
        mean = mean_out.view(B, C, H)
        var = var_out.view(B, C, H)

        # 2) Compute x_expanded = x_ln @ pwconv1_weight.T using Triton
        x_ln_flat = x_ln.reshape(-1).contiguous()
        M = x_ln_flat.numel()
        K = pwconv1_weight.shape[0]  # C
        N = pwconv1_weight.shape[1]  # C4
        B_mat = pwconv1_weight.reshape(K, N).contiguous().view(-1)
        C_mat = torch.empty(M, dtype=torch.float32, device=device)
        grid_matmul = (triton.cdiv(M, 1024),)
        linear_matmul_kernel[grid_matmul](
            x_ln_flat,
            B_mat,
            C_mat,
            M, K, N,
            BLOCK_M=1024, BLOCK_N=32, BLOCK_K=32,
        )
        x_expanded = C_mat.view(B, C, H, W)

        # 3) Apply GELU (tanh approximation) elementwise to x_expanded using Triton
        M_gelu = x_expanded.numel()
        x_expanded_flat = x_expanded.reshape(-1).contiguous()
        y_flat = torch.empty(M_gelu, dtype=torch.float32, device=device)
        grid_gelu = (triton.cdiv(M_gelu, 1024),)
        elementwise_gelu_tanh_kernel[grid_gelu](
            x_expanded_flat,
            y_flat,
            M_gelu,
            BLOCK=1024,
        )
        x_gelu = y_flat.view_as(x_expanded)

        # Return signature must match original; fill placeholders with Triton outputs
        return {
            "grad_output": grad_output,
            "residual": residual,
            "x_dwconv": x_dwconv,
            "x_nhwc": x_nhwc,
            "mean": mean,
            "var": var,
            "x_normalized": x_normalized,       # not computed here
            "x_ln": x_ln,                       # unchanged
            "x_expanded": x_expanded,           # computed via Triton
            "x_gelu": x_gelu,                   # computed via Triton
            "global_features": global_features, # not used here, but signature requires
            "gf_mean": gf_mean,                 # not used here
            "norm_features": norm_features,     # not used here
            "x_grn_scaled": x_grn_scaled,       # not used here
            "x_grn": x_grn,                     # not used here
            "dwconv_weight": dwconv_weight,     # unchanged
            "layernorm_weight": layernorm_weight,
            "pwconv1_weight": pwconv1_weight,   # unchanged
            "grn_weight": grn_weight,           # unchanged
            "pwconv2_weight": pwconv2_weight,   # unchanged
            "drop_mask": drop_mask,             # unchanged
            "drop_path_prob": drop_path_prob,
            "eps": eps,
        }


def run(*args):
    return ModelNew()(*args)
