import torch
import triton
import triton.language as tl


# -------- Triton kernels --------

@triton.jit
def compute_mean_var_w_kernel(
    X_ptr,              # *const float32, input tensor flattened as [B, C, H, W]
    MEAN_ptr,           # *float32, output means flattened as [B, C, H]
    VAR_ptr,            # *float32, output vars flattened as [B, C, H]
    B: tl.constexpr,    # int
    C: tl.constexpr,    # int
    H: tl.constexpr,    # int
    W: tl.constexpr,    # int
    BLOCK_W: tl.constexpr,  # tile size along width
):
    # Each program handles one (b, c, h) row and reduces across W
    pid_b = tl.program_id(axis=0)
    pid_c = tl.program_id(axis=1)
    pid_h = tl.program_id(axis=2)

    # Base offset for this row
    base = (pid_b * C + pid_c) * H * W + pid_h * W

    # Accumulate sum and sum of squares across W
    sum_val = 0.0
    sum_sq = 0.0

    for w_start in range(0, W, BLOCK_W):
        w_idx = w_start + tl.arange(0, BLOCK_W)
        mask = w_idx < W
        x_ptrs = X_ptr + base + w_idx
        vals = tl.load(x_ptrs, mask=mask, other=0.0)
        sum_val += tl.sum(vals, axis=0)
        sum_sq += tl.sum(vals * vals, axis=0)

    mean = sum_val / W
    var = sum_sq / W - mean * mean  # variance across width

    # Store results
    out_mean_offset = pid_b * (C * H) + pid_c * H + pid_h
    out_var_offset = pid_b * (C * H) + pid_c * H + pid_h
    tl.store(MEAN_ptr + out_mean_offset, mean)
    tl.store(VAR_ptr + out_var_offset, var)


@triton.jit
def elementwise_gelu_tanh_kernel(
    X_ptr,              # *const float32, input tensor flattened [B, C, H, W]
    Y_ptr,              # *float32, output tensor flattened [B, C, H, W]
    TOTAL: tl.constexpr,    # total number of elements B*C*H*W
    BLOCK: tl.constexpr,    # vectorization chunk
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < TOTAL

    x = tl.load(X_ptr + offsets, mask=mask, other=0.0)

    # GELU tanh approximation:
    # y = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    sqrt_2_over_pi = 0.7978845608028654
    c = 0.044715
    x3 = x * x * x
    inner = sqrt_2_over_pi * (x + c * x3)
    tanh_inner = tl.tanh(inner)
    y = 0.5 * x * (1.0 + tanh_inner)

    tl.store(Y_ptr + offsets, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        """
        Launches Triton kernels and performs no torch computation.
        Expects at least one input tensor of shape (B, C, H, W) (e.g., x_dwconv).
        Returns:
          - mean across width W: (B, C, H, 1)
          - variance across width W: (B, C, H, 1)
          - GELU (tanh approximation) applied to the input: (B, C, H, W)
        """
        if len(args) == 0:
            raise RuntimeError("No input tensor provided to ModelNew.forward")

        x = args[0]
        if x.dim() != 4:
            raise RuntimeError(f"Expected input tensor of shape (B, C, H, W), got shape {tuple(x.shape)}")
        B, C, H, W = x.shape

        # Ensure float32 contiguous for Triton
        x = x.contiguous().to(torch.float32)

        # 1) Compute mean and variance along width W
        mean = torch.empty((B, C, H), dtype=torch.float32, device=x.device)
        var = torch.empty((B, C, H), dtype=torch.float32, device=x.device)

        grid_reduce = (B, C, H)
        BLOCK_W = 256  # safe tile for width reduction
        compute_mean_var_w_kernel[grid_reduce](
            x.view(-1), mean.view(-1), var.view(-1), B, C, H, W, BLOCK_W,
            num_warps=4, num_stages=2
        )

        # 2) Apply GELU (tanh approximation) elementwise
        total = B * C * H * W
        y_flat = torch.empty(total, dtype=torch.float32, device=x.device)
        BLOCK = 1024
        elementwise_gelu_tanh_kernel[(total + BLOCK - 1) // BLOCK,](
            x.view(-1), y_flat, total, BLOCK,
            num_warps=4, num_stages=2
        )
        y = y_flat.view(B, C, H, W)

        # Return results; no torch ops in forward
        return mean.unsqueeze(-1), var.unsqueeze(-1), y


def run(*args):
    return ModelNew()(*args)
