import torch
import triton
import triton.language as tl


@triton.jit
def sum_sumsq_per_feature_kernel(
    x_ptr,          # *fp32, input flattened as [rows, L]
    sum_ptr,        # *fp32, length L
    sumsq_ptr,      # *fp32, length L
    rows,           # int, total rows = B*S (runtime scalar)
    L,              # int, feature length (runtime scalar)
    NROWS_MAX: tl.constexpr,  # int, max possible rows (compile-time constant)
    BLOCK_ROWS: tl.constexpr, # rows chunk
):
    pid = tl.program_id(0)  # feature index
    if pid >= L:
        return
    total_sum = 0.0
    total_sumsq = 0.0
    i = 0
    while i < NROWS_MAX:
        if i < rows:
            addr = i * L + pid
            val = tl.load(x_ptr + addr)  # scalar load for feature pid at row i
            total_sum += val
            total_sumsq += val * val
        i += 1
    tl.store(sum_ptr + pid, total_sum)
    tl.store(sumsq_ptr + pid, total_sumsq)


@triton.jit
def compute_mean_std_per_feature_kernel(
    sum_ptr,     # *fp32, length L
    sumsq_ptr,   # *fp32, length L
    mean_ptr,    # *fp32, length L
    std_ptr,     # *fp32, length L
    rows,        # int, runtime scalar
    L,           # int, runtime scalar
):
    pid = tl.program_id(0)  # feature index
    if pid >= L:
        return
    s = tl.load(sum_ptr + pid)
    ss = tl.load(sumsq_ptr + pid)
    mean = s / rows
    var = ss / rows - mean * mean
    var = tl.maximum(var, 0.0)
    std = tl.sqrt(var)
    tl.store(mean_ptr + pid, mean)
    tl.store(std_ptr + pid, std)


@triton.jit
def compute_threshold_per_feature_kernel(
    mean_ptr,     # *fp32, length L
    std_ptr,      # *fp32, length L
    std_multiplier,  # scalar fp32 (runtime scalar tensor)
    thr_ptr,      # *fp32, length L
    L,            # int, runtime scalar
):
    pid = tl.program_id(0)
    if pid < L:
        mean = tl.load(mean_ptr + pid)
        std = tl.load(std_ptr + pid)
        thr = mean + std * std_multiplier
        tl.store(thr_ptr + pid, thr)


@triton.jit
def sparse_relu_per_feature_kernel(
    inp_ptr,      # *fp32, flattened input [rows, L]
    thr_ptr,      # *fp32, length L
    out_ptr,      # *fp32, flattened output [rows, L]
    rows,         # int, runtime scalar
    L,            # int, runtime scalar
    BLOCK_F: tl.constexpr,  # feature block
):
    # 2D grid: pid_row over rows, pid_f over features in chunks
    pid_row = tl.program_id(0)
    pid_f = tl.program_id(1)

    f_start = pid_f * BLOCK_F
    f = f_start + tl.arange(0, BLOCK_F)
    mask_f = f < L

    thr_vec = tl.load(thr_ptr + f, mask=mask_f, other=0.0)

    # Process one row at a time for simplicity and correctness
    i = pid_row
    if i >= rows:
        return
    addr = i * L + f
    vals = tl.load(inp_ptr + addr, mask=mask_f, other=0.0)
    out_vals = tl.maximum(vals - thr_vec, 0.0)
    tl.store(out_ptr + addr, out_vals, mask=mask_f)


@triton.jit
def cast_bf16_kernel(
    in_ptr,       # *fp32, flattened [N]
    out_ptr,      # *bf16, flattened [N]
    N,            # int, total number of elements
    BLOCK: tl.constexpr,  # int, block size
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    vals = tl.load(in_ptr + offs, mask=mask, other=0.0)
    vals_bf16 = tl.cast(vals, tl.bfloat16)
    tl.store(out_ptr + offs, vals_bf16, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, target_sparsity: float, std_multiplier_tensor: torch.Tensor):
        """
        target_sparsity: scalar p in (0, 1) for adaptive threshold.
        std_multiplier_tensor: 0-dim torch.Tensor on desired device, equal to ndtri(target_sparsity),
                               passed to Triton kernel to avoid any torch ops in forward.
        """
        super().__init__()
        self.target_sparsity = float(target_sparsity)
        # Ensure std_multiplier is a 0-dim tensor on device
        if not isinstance(std_multiplier_tensor, torch.Tensor) or std_multiplier_tensor.numel() != 1:
            raise ValueError("std_multiplier_tensor must be a 0-dim torch.Tensor with 1 element.")
        self.register_buffer("std_multiplier", std_multiplier_tensor)  # keep as buffer, no grad

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs: [B, S, L], any dtype, CUDA
        B, S, L = inputs.shape
        rows = B * S

        # Flatten to [rows, L] and cast to fp32 for numerics
        inp_flat = inputs.reshape(rows, L).contiguous()
        inp_fp32 = inp_flat.to(torch.float32)

        # Buffers for per-feature stats
        sum_f = torch.empty(L, dtype=torch.float32, device=inputs.device)
        sumsq_f = torch.empty(L, dtype=torch.float32, device=inputs.device)
        mean_f = torch.empty(L, dtype=torch.float32, device=inputs.device)
        std_f = torch.empty(L, dtype=torch.float32, device=inputs.device)
        thr_f = torch.empty(L, dtype=torch.float32, device=inputs.device)

        # Launch sum/sumsq reduction per feature: grid = (L,)
        sum_sumsq_per_feature_kernel[(L,)](
            inp_fp32, sum_f, sumsq_f, rows, L, NROWS_MAX=rows, BLOCK_ROWS=1024,
            num_warps=1
        )

        # Compute mean and std per feature
        compute_mean_std_per_feature_kernel[(L,)](
            sum_f, sumsq_f, mean_f, std_f, rows, L, num_warps=1
        )

        # Compute thresholds per feature in Triton
        compute_threshold_per_feature_kernel[(L,)](
            mean_f, std_f, self.std_multiplier, thr_f, L, num_warps=1
        )

        # Apply sparse ReLU elementwise using per-feature thr
        out_fp32 = torch.empty_like(inp_fp32)
        grid_relu = (rows, (L + 128 - 1) // 128)
        sparse_relu_per_feature_kernel[grid_relu](
            inp_fp32, thr_f, out_fp32, rows, L, BLOCK_F=128, num_warps=4
        )

        # Cast to bfloat16 via Triton (must be invoked)
        N = rows * L
        out_bf16_flat = torch.empty(N, dtype=torch.bfloat16, device=inputs.device)
        cast_bf16_kernel[(triton.cdiv(N, 4096),)](
            out_fp32, out_bf16_flat, N, BLOCK=4096, num_warps=4
        )

        # Reshape to [B, S, L]
        out_bf16 = out_bf16_flat.view(B, S, L)
        return out_bf16


def run(*args):
    return ModelNew()(*args)
