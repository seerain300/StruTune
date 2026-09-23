import torch
import triton
import triton.language as tl


@triton.jit
def _row_stats_kernel(X_ptr, Mean_ptr, Var_ptr, S: tl.constexpr, H: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    """
    One program per row. Accumulate sum and sum of squares over H tiles.
    Writes per-row mean and variance to Mean_ptr[row] and Var_ptr[row].
    """
    row = tl.program_id(0)
    # Accumulators in float32
    sum_val = 0.0
    sumsq_val = 0.0

    start = 0
    while start < H:
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < H
        # Row base index = row * H
        x = tl.load(X_ptr + row * H + offs, mask=mask, other=0.0)
        # Cast to float32 for stable accumulation
        x = x.to(tl.float32)
        sum_val += tl.sum(x, axis=0)
        sumsq_val += tl.sum(x * x, axis=0)
        start += BLOCK_SIZE

    mean = sum_val / H
    var = sumsq_val / H - mean * mean
    # Avoid negative var due to rounding
    var = tl.maximum(var, 0.0)

    tl.store(Mean_ptr + row, mean)
    tl.store(Var_ptr + row, var)


@triton.jit
def _std_rows_kernel(Var_ptr, Std_ptr, S: tl.constexpr, BLOCK: tl.constexpr):
    """
    Compute std = sqrt(var) for each row. Use BLOCK=1 as each is scalar.
    """
    row = tl.program_id(0)
    var = tl.load(Var_ptr + row)
    std = tl.sqrt(var)  # Triton supports sqrt
    tl.store(Std_ptr + row, std)


@triton.jit
def _gate_kernel(X_ptr, Y_ptr, Mean_ptr, Std_ptr, z_ptr, S: tl.constexpr, H: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    """
    One program per row. Compute threshold = mean + std * z and apply y = max(0, x - threshold).
    """
    row = tl.program_id(0)
    mean = tl.load(Mean_ptr + row)
    std = tl.load(Std_ptr + row)
    z = tl.load(z_ptr)  # scalar, already float32
    threshold = mean + std * z

    start = 0
    while start < H:
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < H
        x = tl.load(X_ptr + row * H + offs, mask=mask, other=0.0)
        x = x.to(tl.float32)
        y = x - threshold
        y = tl.maximum(y, 0.0)  # ReLU
        tl.store(Y_ptr + row * H + offs, y, mask=mask)
        start += BLOCK_SIZE


def _ndtri_approx(target_sparsity: float) -> float:
    """
    Inverse standard normal CDF approximation (Abramowitz-Stegun 7.1.26).
    Single float compute in Python; no torch here.
    """
    p = float(target_sparsity)
    # Piecewise regions
    p_low = 0.02425
    p_high = 1.0 - p_low

    # Central region
    if p <= p_low:
        # Lower region: y = sqrt(-2 ln p)
        y = ((-0.284496736 + p * (0.787692893 + p * (-0.230307633 + p * (0.073115465 + p * (-0.021414184)))) /
             (1.0 + p * (0.2316419 + p * (0.021414184)))) * (-2.0 * math.log(p)))
    elif p >= p_high:
        # Upper region
        t = 1.0 - p
        y = ((-0.284496736 + t * (0.787692893 + t * (-0.230307633 + t * (0.073115465 + t * (-0.021414184)))) /
             (1.0 + t * (0.2316419 + t * (0.021414184)))) * (-2.0 * math.log(t)))
        y = -y
    else:
        # Central region approximation
        y = p - 0.5
        r = y * y
        poly = (((((1.574090337 + r * 0.074230053) * r - 0.018729326) * r + 0.004292777) * r - 0.000518988) * r + 0.000030338)
        poly_y = y * poly
        denom = 1.0 + r * 0.140012000 + r * 0.189269600 + r * 0.258068600 + r * 0.393240000 + r
        y = poly_y / denom

    return y


def _run_triton(inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
    """
    Triton-ONLY implementation of the original run function.
    Computes per-row mean and std across the last dim, then
    applies y = max(0, x - (mean + std * _ndtri(target_sparsity))) and returns bfloat16.
    """
    assert inputs.is_cuda, "ModelNew requires CUDA tensors"
    inputs = inputs.contiguous()

    # Cast to float32 for stable math; flatten [B, L, H] -> [S, H]
    B, L, H = inputs.shape
    S = B * L
    X = inputs.to(torch.float32).view(S, H)

    # Allocate outputs
    Y = torch.empty((S, H), dtype=torch.float32, device=inputs.device)
    Mean = torch.empty(S, dtype=torch.float32, device=inputs.device)
    Var = torch.empty(S, dtype=torch.float32, device=inputs.device)
    Std = torch.empty(S, dtype=torch.float32, device=inputs.device)

    # Compute z_scalar = _ndtri(target_sparsity) as a pure Python float; no torch tensor creation here.
    z_scalar = _ndtri_approx(target_sparsity)

    # Kernel 1: per-row sum and sumsq
    BLOCK_SIZE = 1024
    _row_stats_kernel[(S,)](X, Mean, Var, S=S, H=H, BLOCK_SIZE=BLOCK_SIZE, num_warps=4)

    # Kernel 2: std per row
    _std_rows_kernel[(S,)](Var, Std, S=S, BLOCK=1, num_warps=1)

    # Kernel 3: gating with threshold = mean + std * z_scalar
    z_tensor = torch.tensor(z_scalar, dtype=torch.float32, device=inputs.device)  # minimal device tensor for threshold
    _gate_kernel[(S,)](X, Y, Mean, Std, z_tensor, S=S, H=H, BLOCK_SIZE=BLOCK_SIZE, num_warps=4)

    # Reshape and cast to bfloat16 to match original behavior
    out = Y.view(B, L, H).to(torch.bfloat16)
    return out


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Expect a single 3D input tensor [batch_size, seq_len, intermediate_size]
        if len(args) != 1:
            raise RuntimeError("ModelNew expects a single 3D input tensor [B, L, H]")
        inputs = args[0]
        return _run_triton(inputs, target_sparsity=0.1)  # default sparsity; configurable


def run(*args):
    return ModelNew()(*args)
