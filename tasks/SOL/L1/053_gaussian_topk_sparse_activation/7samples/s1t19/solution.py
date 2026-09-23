import torch
import triton
import triton.language as tl


# Kernel: compute per-row mean and population std over last dim N for X2D [rows, N]
@triton.jit
def row_stats_kernel(
    X2D_ptr,         # *f32, contiguous [rows, N]
    MEAN_ptr,        # *f32, [rows]
    STD_ptr,         # *f32, [rows]
    rows,            # int
    N,               # int
    BLOCK: tl.constexpr,
):
    row_id = tl.program_id(0)
    # Accumulate sum and sum of squares in fp32
    sum_val = 0.0
    sum_sq = 0.0

    col = 0
    while col < N:
        offs = col + tl.arange(0, BLOCK)
        mask = offs < N
        idx = row_id * N + offs
        x = tl.load(X2D_ptr + idx, mask=mask, other=0.0)
        # Reduce this chunk to scalars
        sum_val += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)
        col += BLOCK

    n_float = tl.full((), N, tl.float32)
    mean = sum_val / n_float
    var = sum_sq / n_float - mean * mean
    # population std (unbiased=False)
    std = tl.sqrt(var)
    tl.store(MEAN_ptr + row_id, mean)
    tl.store(STD_ptr + row_id, std)


# Scalar Triton kernel: inverse standard normal CDF using A&S 5.2.23
@triton.jit
def ndtri_kernel(
    P_ptr,   # *f32, [1] containing target_sparsity
    Q_ptr,   # *f32, [1] output q = sqrt(-2 ln(P)) for low/high regions
):
    p = tl.load(P_ptr)  # scalar probability
    # Compute q = sqrt(-2 ln(P)) for lower/higher region
    q = tl.sqrt(-2.0 * tl.log(p))
    tl.store(Q_ptr, q)


# Triton kernel: compute per-row threshold = mean + std * multiplier (scalar q)
@triton.jit
def threshold_vec_kernel(
    MEAN_ptr,  # *f32, [rows]
    STD_ptr,   # *f32, [rows]
    Q_ptr,     # *f32, [1] scalar multiplier
    THRESH_ptr,# *f32, [rows]
    rows,      # int
):
    row_id = tl.program_id(0)
    mean = tl.load(MEAN_ptr + row_id)
    std = tl.load(STD_ptr + row_id)
    q = tl.load(Q_ptr)
    th = mean + std * q
    tl.store(THRESH_ptr + row_id, th)


# Elementwise Triton kernel: apply ReLU(x - threshold[row]) over X2D [rows, N]
@triton.jit
def relu_threshold_kernel(
    X2D_ptr,       # *f32, [rows, N]
    THRESH_ptr,    # *f32, [rows]
    OUT2D_ptr,     # *f32, [rows, N]
    rows,          # int
    N,             # int
    BLOCK: tl.constexpr,
):
    row_id = tl.program_id(0)
    # Load threshold for this row
    th = tl.load(THRESH_ptr + row_id)
    col = 0
    while col < N:
        offs = col + tl.arange(0, BLOCK)
        mask = offs < N
        idx = row_id * N + offs
        x = tl.load(X2D_ptr + idx, mask=mask, other=0.0)
        y = x - th
        y = tl.maximum(y, 0.0)  # ReLU
        tl.store(OUT2D_ptr + idx, y, mask=mask)
        col += BLOCK


def run(inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
    # Expect inputs of shape [batch_size, seq_len, intermediate_size]
    if inputs.ndim != 3:
        raise RuntimeError("run expects a 3D tensor [batch, seq, features].")

    B, S, N = inputs.shape
    rows = B * S

    # Ensure device is CUDA; Triton requires CUDA
    device = inputs.device
    if device.type != "cuda":
        raise RuntimeError("run requires CUDA device; input tensor must be on CUDA.")

    # Compute in fp32 for numerical stability
    x = inputs.to(torch.float32)
    x2d = x.contiguous().view(rows, N)

    # 1) Compute per-row mean and std (fp32)
    mean = torch.empty(rows, device=device, dtype=torch.float32)
    std = torch.empty(rows, device=device, dtype=torch.float32)

    grid_stats = (rows,)
    row_stats_kernel[grid_stats](
        x2d, mean, std,
        rows=rows, N=N,
        BLOCK=1024,
        num_warps=8,
    )

    # 2) Compute multiplier q = sqrt(-2 ln(target_sparsity)) via Triton scalar kernel
    p = torch.full((1,), float(target_sparsity), device=device, dtype=torch.float32)
    q = torch.empty(1, device=device, dtype=torch.float32)
    ndtri_kernel[(1,)](p, q)

    # 3) Compute per-row threshold (fp32)
    threshold = torch.empty(rows, device=device, dtype=torch.float32)
    threshold_vec_kernel[grid_stats](mean, std, q, threshold, rows)

    # 4) Apply ReLU(x - threshold[row]) in Triton, write to OUT2d
    OUT2d = torch.empty((rows, N), device=device, dtype=torch.float32)
    relu_threshold_kernel[grid_stats](
        x2d, threshold, OUT2d,
        rows=rows, N=N,
        BLOCK=1024,
        num_warps=4,
    )

    # Return in bf16 to match original behavior
    return OUT2d.view(B, S, N).to(torch.bfloat16)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        if len(args) != 1:
            raise RuntimeError("ModelNew.forward expects a single input tensor.")
        return run(*args[0])


def run(*args):
    return ModelNew()(*args)
