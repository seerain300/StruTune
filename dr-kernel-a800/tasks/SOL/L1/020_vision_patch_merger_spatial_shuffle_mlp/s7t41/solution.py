import math
import torch
import triton
import triton.language as tl


@triton.jit
def layernorm_affine_kernel(
    hidden_ptr,       # *bf16, [num_patches, hidden_size]
    ln_weight_ptr,    # *bf16, [hidden_size]
    ln_bias_ptr,      # *bf16, [hidden_size]
    out_ptr,          # *bf16, [num_patches, hidden_size]
    num_patches: tl.constexpr,   # int
    hidden_size: tl.constexpr,   # int
    eps: tl.constexpr,           # float
    BLOCK_C: tl.constexpr,       # int
):
    # One program per row (patch)
    row = tl.program_id(0)  # r in [0, num_patches)
    # Compute mean in fp32
    sum_val = 0.0
    sum_sq = 0.0
    c0 = 0
    while c0 < hidden_size:
        offs = c0 + tl.arange(0, BLOCK_C)
        mask = offs < hidden_size
        vals = tl.load(hidden_ptr + row * hidden_size + offs, mask=mask, other=0.0)
        vals_f32 = vals.to(tl.float32)
        sum_val += tl.sum(vals_f32, axis=0)
        sum_sq += tl.sum(vals_f32 * vals_f32, axis=0)
        c0 += BLOCK_C

    mean = sum_val / hidden_size
    var = sum_sq / hidden_size - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Normalize and affine, then store
    c0 = 0
    while c0 < hidden_size:
        offs = c0 + tl.arange(0, BLOCK_C)
        mask = offs < hidden_size
        x = tl.load(hidden_ptr + row * hidden_size + offs, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(ln_weight_ptr + offs, mask=mask, other=1.0).to(tl.float32)
        b = tl.load(ln_bias_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        y = (x - mean) * inv_std
        y = y * w + b
        y = y.to(tl.bfloat16)
        tl.store(out_ptr + row * hidden_size + offs, y, mask=mask)
        c0 += BLOCK_C


@triton.jit
def fill_X_fc1_from_ln_v2(
    ln_out_ptr,       # *bf16, [num_patches, hidden_size]
    grid_thw_ptr,     # *int64, [num_grids, 3] -> T, H, W
    X_fc1_ptr,        # *bf16, [num_merged_patches, hidden_size_expanded]
    num_patches: tl.constexpr,   # int
    num_grids: tl.constexpr,     # int
    hidden_size: tl.constexpr,   # int
    h_merged: tl.constexpr,      # int
    w_merged: tl.constexpr,      # int
    merge_size: tl.constexpr,    # int (2)
    BLOCK_M: tl.constexpr,       # int (tile over merged patches)
    BLOCK_N: tl.constexpr,       # int (tile over features)
):
    # Each program handles a tile of merged patches [pid_m * BLOCK_M : (pid_m+1) * BLOCK_M]
    pid_m = tl.program_id(0)
    start_merged = pid_m * BLOCK_M
    merged_per_grid = h_merged * w_merged
    patches_per_grid = h_merged * w_merged * 2 * 2  # 2x2 merge => 4 original patches per merged
    num_merged = num_grids * merged_per_grid

    merged_idx = start_merged + tl.arange(0, BLOCK_M)
    valid_m = merged_idx < num_merged

    # Initialize pointers for X_fc1 tile
    X_tile_ptr = X_fc1_ptr + merged_idx[:, None] * hidden_size * 2 * 2

    # For each grid
    for g in range(num_grids):
        t = tl.load(grid_thw_ptr + g * 3 + 0).to(tl.int32)
        h = tl.load(grid_thw_ptr + g * 3 + 1).to(tl.int32)
        w = tl.load(grid_thw_ptr + g * 3 + 2).to(tl.int32)

        base = g * merged_per_grid + start_merged
        merged_idx_g = base + tl.arange(0, BLOCK_M)
        valid_g = merged_idx_g < (g + 1) * merged_per_grid  # last grid may have fewer rows

        # Map merged (i_m, j_m) to original (i0, j0) coordinates
        i_m = merged_idx_g // w_merged
        j_m = merged_idx_g % w_merged
        i0 = i_m // 2
        j0 = j_m // 2

        # Flattened original patch index p in [0, t * h * w)
        p = i0 * w + j0  # [BLOCK_M]
        valid_p = valid_g & (p < t * h * w)

        # Compute the feature index c = hidden_size * 4 (since 2x2 = 4)
        c = tl.arange(0, BLOCK_N)
        valid_n = c < hidden_size * 4

        # We need to write ln_out[p, c] into X_fc1[merged_idx_g, c]
        # Note: p is per merged position; we use 2D broadcasting for stores
        # We compute source addresses for ln_out: ln_out_ptr + p * hidden_size + c
        # Since p may be out of range for last grid, we set output to 0 for invalids
        # However, we need to align p to ln_out rows; ln_out has num_patches rows,
        # but p is derived from merged positions. We rely on the host to ensure p < num_patches,
        # which is true by construction. So we proceed with masked loads.
        # We'll form a 2D pointer for ln_out: pointer depends on p and c
        # To do so, we use broadcasting: create a 2D grid [BLOCK_M, BLOCK_N] for p and c.
        # We'll construct p broadcasted: p[:, None], and c broadcasted: c[None, :].
        p_broadcast = p[:, None]
        c_broadcast = c[None, :]

        # Compute ln_out row index: since ln_out has num_patches rows, and we generated p
        # based on original patches, we need p < num_patches. Our get_inputs returns hidden
        # of shape [num_patches], and we shuffled from it; thus p should be in valid range.
        # For safety, we mask invalid_m and invalid_n.
        valid_store = valid_g[:, None] & valid_n[None, :]

        # Load ln_out values for valid p and c
        # ln_out_ptr layout: contiguous [num_patches, hidden_size]
        # We need to ensure p < num_patches; from construction, it should be true, but
        # if not, masked load with other=0.0.
        ln_out_row_ptr = ln_out_ptr + p_broadcast * hidden_size + c_broadcast
        vals = tl.load(ln_out_row_ptr, mask=valid_store, other=0.0).to(tl.float32)
        # Store into X_fc1 at merged positions
        tl.store(X_tile_ptr, vals.to(tl.bfloat16), mask=valid_store)


@triton.jit
def matmul_bias_kernel(
    A_ptr,            # *bf16, [M, K]
    B_ptr,            # *bf16, [N, K] (we will pass B as [K, N] by viewing on host)
    bias_ptr,         # *bf16, [N]
    C_ptr,            # *bf16, [M, N]
    M: tl.constexpr,  # int
    N: tl.constexpr,  # int
    K: tl.constexpr,  # int
    stride_am, stride_ak,  # int
    stride_bk, stride_bn,  # int
    stride_cm, stride_cn,  # int
    eps: tl.constexpr,      # unused, kept for signature symmetry
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    m0 = pid_m * BLOCK_M
    n0 = pid_n * BLOCK_N

    offs_m = m0 + tl.arange(0, BLOCK_M)
    offs_n = n0 + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    k0 = 0
    while k0 < K:
        k = k0 + offs_k
        a_ptrs = A_ptr + offs_m[:, None] * stride_am + k[None, :] * stride_ak
        b_ptrs = B_ptr + k[:, None] * stride_bk + offs_n[None, :] * stride_bn

        a_mask = (offs_m[:, None] < M) & (k[None, :] < K)
        b_mask = (k[:, None] < K) & (offs_n[None, :] < N)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0).to(tl.float32)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0).to(tl.float32)

        acc += tl.dot(a, b)
        k0 += BLOCK_K

    # Add bias
    bias = tl.load(bias_ptr + offs_n, mask=(offs_n < N), other=0.0).to(tl.float32)
    acc = acc + bias[None, :]

    # Store result
    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.bfloat16), mask=c_mask)


@triton.jit
def gelu_tanh_kernel(
    X_ptr,    # *bf16, [M, N]
    Y_ptr,    # *bf16, [M, N]
    M: tl.constexpr,  # int
    N: tl.constexpr,  # int
    stride_xm, stride_xn,  # int
    stride_ym, stride_yn,  # int
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    m0 = pid_m * BLOCK_M
    n0 = pid_n * BLOCK_N

    offs_m = m0 + tl.arange(0, BLOCK_M)
    offs_n = n0 + tl.arange(0, BLOCK_N)

    x_ptrs = X_ptr + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn
    y_ptrs = Y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn

    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
    # tanh-based GELU approximation: gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))
    c = 0.7978845608028654  # sqrt(2/pi)
    x3 = x * x * x
    t = c * (x + 0.044715 * x3)
    y = 0.5 * x * (1.0 + tl.math.tanh(t))
    tl.store(y_ptrs, y.to(tl.bfloat16), mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, hidden_size: int = 1536, hidden_size_expanded: int = 6144, out_hidden_size: int = 3584, eps: float = 1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.hidden_size_expanded = hidden_size_expanded
        self.out_hidden_size = out_hidden_size
        self.eps = eps
        # Predefined constants for this task
        self.merge_size = 2

    def forward(self, hidden: torch.Tensor, grid_thw: torch.Tensor, ln_weight: torch.Tensor, ln_bias: torch.Tensor,
                fc1_weight: torch.Tensor, fc1_bias: torch.Tensor, fc2_weight: torch.Tensor, fc2_bias: torch.Tensor, eps: float):
        # Ensure contiguity and dtypes
        hidden = hidden.contiguous()
        ln_weight = ln_weight.contiguous()
        ln_bias = ln_bias.contiguous()
        fc1_weight = fc1_weight.contiguous()
        fc1_bias = fc1_bias.contiguous()
        fc2_weight = fc2_weight.contiguous()
        fc2_bias = fc2_bias.contiguous()

        num_patches = hidden.shape[0]
        hidden_size = self.hidden_size

        # 1) LayerNorm + affine in Triton
        ln_out = torch.empty_like(hidden, dtype=torch.bfloat16, device=hidden.device)
        BLOCK_C = 128
        grid_layernorm = (num_patches,)
        layernorm_affine_kernel[grid_layernorm](
            hidden, ln_weight, ln_bias, ln_out,
            num_patches=num_patches,
            hidden_size=hidden_size,
            eps=self.eps,
            BLOCK_C=BLOCK_C,
            num_warps=4,
        )

        # 2) Spatial 2x2 merge into first linear input using Triton kernel
        # We need to map original patches to merged positions: for each grid,
        # original (T, H, W) -> merged (T, H//2, W//2) with 2x2 grouping.
        # Create X_fc1 of shape [num_merged_patches, hidden_size_expanded]
        # Compute merged dims
        merged_per_grid = (grid_thw[:, 1] // self.merge_size) * (grid_thw[:, 2] // self.merge_size)
        num_merged = grid_thw.shape[0] * int(merged_per_grid.item())

        X_fc1 = torch.empty((num_merged, self.hidden_size_expanded), dtype=torch.bfloat16, device=hidden.device)

        # Launch Triton fill kernel: note grid_thw is int64 on device; we pass as is.
        BLOCK_M = 128
        BLOCK_N = 128
        grid_fill = (triton.cdiv(num_merged, BLOCK_M),)
        fill_X_fc1_from_ln_v2[grid_fill](
            ln_out,
            grid_thw,
            X_fc1,
            num_patches=num_patches,
            num_grids=grid_thw.shape[0],
            hidden_size=hidden_size,
            h_merged=int((grid_thw[:, 1] // self.merge_size).min().item()),
            w_merged=int((grid_thw[:, 2] // self.merge_size).min().item()),
            merge_size=self.merge_size,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            num_warps=4,
        )

        # 3) First linear: A = X_fc1 [num_merged, hidden_size_expanded], B = fc1_weight.T [hidden_size_expanded, hidden_size_expanded]
        # Use Triton GEMM with bias
        M = X_fc1.shape[0]  # num_merged_patches
        K = X_fc1.shape[1]  # hidden_size_expanded
        N1 = fc1_weight.shape[0]  # hidden_size_expanded

        C1 = torch.empty((M, N1), dtype=torch.bfloat16, device=hidden.device)
        # We pass B as fc1_weight.T (row-major [N1, K]) for simplicity; kernel expects [K, N] view.
        # Prepare B as [K, N1] by transposing and ensuring contiguous.
        B1 = fc1_weight.t().contiguous()  # [hidden_size_expanded, hidden_size_expanded]
        bias1 = fc1_bias.contiguous()

        BLOCK_M1 = 64
        BLOCK_N1 = 64
        BLOCK_K1 = 32
        grid_gemm1 = (triton.cdiv(M, BLOCK_M1), triton.cdiv(N1, BLOCK_N1))
        matmul_bias_kernel[grid_gemm1](
            X_fc1, B1, bias1, C1,
            M=M, N=N1, K=K,
            stride_am=X_fc1.stride(0), stride_ak=X_fc1.stride(1),
            stride_bk=B1.stride(0), stride_bn=B1.stride(1),
            stride_cm=C1.stride(0), stride_cn=C1.stride(1),
            eps=self.eps,
            BLOCK_M=BLOCK_M1, BLOCK_N=BLOCK_N1, BLOCK_K=BLOCK_K1,
            num_warps=4,
        )

        # 4) GELU activation via Triton kernel
        Y1 = torch.empty_like(C1, dtype=torch.bfloat16, device=hidden.device)
        BLOCK_M2 = 128
        BLOCK_N2 = 128
        grid_gelu = (triton.cdiv(M, BLOCK_M2), triton.cdiv(N1, BLOCK_N2))
        gelu_tanh_kernel[grid_gelu](
            C1, Y1,
            M=M, N=N1,
            stride_xm=C1.stride(0), stride_xn=C1.stride(1),
            stride_ym=Y1.stride(0), stride_yn=Y1.stride(1),
            BLOCK_M=BLOCK_M2, BLOCK_N=BLOCK_N2,
            num_warps=4,
        )

        # 5) Second linear: A = Y1 [num_merged, hidden_size_expanded], B = fc2_weight.T [hidden_size_expanded, out_hidden_size]
        M2 = M
        N2 = self.out_hidden_size  # 3584
        K2 = N1  # hidden_size_expanded

        C2 = torch.empty((M2, N2), dtype=torch.bfloat16, device=hidden.device)
        B2 = fc2_weight.t().contiguous()  # [hidden_size_expanded, out_hidden_size]
        bias2 = fc2_bias.contiguous()

        BLOCK_M3 = 64
        BLOCK_N3 = 64
        BLOCK_K2 = 32
        grid_gemm2 = (triton.cdiv(M2, BLOCK_M3), triton.cdiv(N2, BLOCK_N3))
        matmul_bias_kernel[grid_gemm2](
            Y1, B2, bias2, C2,
            M=M2, N=N2, K=K2,
            stride_am=Y1.stride(0), stride_ak=Y1.stride(1),
            stride_bk=B2.stride(0), stride_bn=B2.stride(1),
            stride_cm=C2.stride(0), stride_cn=C2.stride(1),
            eps=self.eps,
            BLOCK_M=BLOCK_M3, BLOCK_N=BLOCK_N3, BLOCK_K=BLOCK_K2,
            num_warps=4,
        )

        return C2


# Helper to generate inputs (same as original), not used in ModelNew but provided for completeness
def get_inputs(axes_and_scalars: dict, device: torch.device) -> dict[str, torch.Tensor]:
    num_patches = axes_and_scalars["num_patches"]
    num_merged_patches = axes_and_scalars["num_merged_patches"]
    num_grids = axes_and_scalars["num_grids"]
    hidden_size = 1536
    hidden_size_expanded = 6144
    out_hidden_size = 3584
    merge_size = 2
    eps = 1e-6

    # Generate grid_thw such that total patches matches num_patches
    patches_per_grid = num_patches // num_grids

    # Simple configuration: T * H * W = patches_per_grid; choose T=1
    sqrt_patches = int(math.sqrt(patches_per_grid))
    h = (sqrt_patches // merge_size) * merge_size
    if h == 0:
        h = merge_size
    w = (patches_per_grid // h // merge_size) * merge_size
    if w == 0:
        w = merge_size
    t = patches_per_grid // (h * w)
    if t == 0:
        t = 1

    # Adjust to match exactly
    actual_patches_per_grid = t * h * w

    # Create grid_thw tensor
    grid_thw = torch.zeros((num_grids, 3), dtype=torch.int64, device=device)
    remaining_patches = num_patches
    for i in range(num_grids):
        if i == num_grids - 1:
            patches_for_this = remaining_patches
        else:
            patches_for_this = actual_patches_per_grid

        sqrt_p = int(math.sqrt(patches_for_this))
        h_i = (sqrt_p // merge_size) * merge_size
        if h_i == 0:
            h_i = merge_size
        w_i = (patches_for_this // h_i // merge_size) * merge_size
        if w_i == 0:
            w_i = merge_size
        t_i = patches_for_this // (h_i * w_i)
        if t_i == 0:
            t_i = 1

        grid_thw[i, 0] = t_i
        grid_thw[i, 1] = h_i
        grid_thw[i, 2] = w_i
        remaining_patches -= t_i * h_i * w_i

    hidden = torch.randn(num_patches, hidden_size, dtype=torch.bfloat16, device=device)
    ln_weight = torch.ones(hidden_size, dtype=torch.bfloat16, device=device)
    ln_bias = torch.zeros(hidden_size, dtype=torch.bfloat16, device=device)
    fc1_weight = torch.randn(hidden_size_expanded, hidden_size_expanded, dtype=torch.bfloat16, device=device) / math.sqrt(hidden_size_expanded)
    fc1_bias = torch.randn(hidden_size_expanded, dtype=torch.bfloat16, device=device)
    fc2_weight = torch.randn(out_hidden_size, hidden_size_expanded, dtype=torch.bfloat16, device=device) / math.sqrt(hidden_size_expanded)
    fc2_bias = torch.randn(out_hidden_size, dtype=torch.bfloat16, device=device)

    return {
        "hidden": hidden,
        "grid_thw": grid_thw,
        "ln_weight": ln_weight,
        "ln_bias": ln_bias,
        "fc1_weight": fc1_weight,
        "fc1_bias": fc1_bias,
        "fc2_weight": fc2_weight,
        "fc2_bias": fc2_bias,
        "eps": eps,
    }


@torch.no_grad()
def run_triton(
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
    # This function mimics the original run, but uses ModelNew.forward
    # Initialize model
    model = ModelNew().to(hidden.device)
    return model(hidden, grid_thw, ln_weight, ln_bias, fc1_weight, fc1_bias, fc2_weight, fc2_bias, eps)


def run(*args):
    return ModelNew()(*args)
