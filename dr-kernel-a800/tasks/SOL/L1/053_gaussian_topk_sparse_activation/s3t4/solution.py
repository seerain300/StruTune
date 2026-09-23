import triton
import triton.language as tl


@triton.jit
def reduce_sum_sumsq_features(inp_ptr, sum_ptr, sumsq_ptr, N_ROWS: tl.constexpr, L: tl.constexpr):
    """
    For each feature j (program id 0), accumulate sum and sum of squares over all rows (N_ROWS = B*S).
    Grid: (L,)
    """
    j = tl.program_id(0)
    sum_j = 0.0
    sumsq_j = 0.0
    # Iterate over rows in chunks
    for row_start in range(0, N_ROWS, 64):
        rows = row_start + tl.arange(0, 64)
        mask = rows < N_ROWS
        offs = rows * L + j
        vals = tl.load(inp_ptr + offs, mask=mask, other=0.0)
        sum_j += tl.sum(vals, axis=0)
        sumsq_j += tl.sum(vals * vals, axis=0)
    tl.store(sum_ptr + j, sum_j)
    tl.store(sumsq_ptr + j, sumsq_j)


@triton.jit
def compute_mean_std_features(sum_ptr, sumsq_ptr, mean_ptr, std_ptr, N_ROWS: tl.constexpr, L: tl.constexpr):
    """
    Compute mean and std per feature: mean = sum / N_ROWS, std = sqrt(sumsq/N_ROWS - mean^2)
    Grid: (L,)
    """
    j = tl.program_id(0)
    sum_j = tl.load(sum_ptr + j)
    sumsq_j = tl.load(sumsq_ptr + j)
    inv_N = 1.0 / N_ROWS
    mean_j = sum_j * inv_N
    var_j = sumsq_j * inv_N - mean_j * mean_j
    var_j = tl.maximum(var_j, 0.0)  # numerical safety
    std_j = tl.sqrt(var_j)
    tl.store(mean_ptr + j, mean_j)
    tl.store(std_ptr + j, std_j)


@triton.jit
def compute_threshold_features(mean_ptr, std_ptr, std_multiplier, threshold_ptr, L: tl.constexpr):
    """
    threshold[j] = mean[j] + std[j] * std_multiplier, for j in [0, L).
    Grid: (L,)
    std_multiplier is a 0-dim tensor on device; load it once.
    """
    j = tl.program_id(0)
    mean_j = tl.load(mean_ptr + j)
    std_j = tl.load(std_ptr + j)
    mul = tl.load(std_multiplier)  # scalar load
    thr_j = mean_j + std_j * mul
    tl.store(threshold_ptr + j, thr_j)


@triton.jit
def sparse_relu_features(inp_ptr, threshold_ptr, out_ptr, N_ROWS: tl.constexpr, L: tl.constexpr):
    """
    For each feature j (program id 0), compute out[r, j] = max(inp[r, j] - threshold[j], 0)
    Grid: (L,)
    Iterate over rows in chunks and write FP32 output.
    """
    j = tl.program_id(0)
    threshold_j = tl.load(threshold_ptr + j)
    for row_start in range(0, N_ROWS, 64):
        rows = row_start + tl.arange(0, 64)
        mask = rows < N_ROWS
        offs = rows * L + j
        vals = tl.load(inp_ptr + offs, mask=mask, other=0.0)
        vals = vals - threshold_j
        vals = tl.maximum(vals, 0.0)  # ReLU
        tl.store(out_ptr + offs, vals, mask=mask)


@triton.jit
def cast_bf16_kernel(in_fp32_ptr, out_bf16_ptr, N, BLOCK: tl.constexpr):
    """
    Cast FP32 to BF16 elementwise. Must be invoked in forward.
    Grid: (ceil(N / BLOCK),)
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    x = tl.load(in_fp32_ptr + offsets, mask=mask, other=0.0)
    x_bf = tl.cast(x, tl.bfloat16)
    tl.store(out_bf16_ptr + offsets, x_bf, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, std_multiplier: torch.Tensor):
        """
        std_multiplier: a 0-dim tensor (scalar) on the target CUDA device, representing
        the inverse-normal CDF value for the target sparsity. Do not use torch ops in forward.
        """
        super().__init__()
        # Store as 0-dim tensor to avoid any torch math in forward
        assert std_multiplier.ndim == 0, "std_multiplier must be a 0-dim tensor"
        assert std_multiplier.is_cuda, "std_multiplier must be on CUDA"
        self.register_buffer("std_multiplier", std_multiplier)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Triton-only implementation:
        1) Reduce sum and sumsq per feature (Triton).
        2) Compute mean and std per feature (Triton).
        3) Compute threshold per feature (Triton), using std_multiplier.
        4) Compute sparse ReLU per feature (Triton), write FP32.
        5) Cast to BF16 via Triton kernel (forward must invoke it).
        Returns tensor of shape [B, S, L] in bfloat16.
        """
        assert inputs.is_cuda, "Inputs must be on CUDA device"
        inputs = inputs.contiguous()
        B, S, L = inputs.shape
        N_ROWS = B * S
        device = inputs.device

        inp_flat = inputs.view(-1)  # [N_ROWS * L]
        inp_fp32 = inp_flat  # keep FP32 for stable math

        # 1) Reduce sum and sumsq per feature
        sum_features = torch.empty(L, dtype=torch.float32, device=device)
        sumsq_features = torch.empty(L, dtype=torch.float32, device=device)
        grid_red = (L,)
        reduce_sum_sumsq_features[grid_red](inp_fp32, sum_features, sumsq_features, N_ROWS, L, num_warps=2)

        # 2) Compute mean and std per feature
        mean_features = torch.empty(L, dtype=torch.float32, device=device)
        std_features = torch.empty(L, dtype=torch.float32, device=device)
        compute_mean_std_features[grid_red](sum_features, sumsq_features, mean_features, std_features, N_ROWS, L, num_warps=2)

        # 3) Compute threshold per feature: mean + std * std_multiplier
        threshold = torch.empty(L, dtype=torch.float32, device=device)
        compute_threshold_features[(L,)](mean_features, std_features, self.std_multiplier, threshold, L, num_warps=2)

        # 4) Sparse ReLU per feature: out = max(inp - threshold, 0)
        out_fp32 = torch.empty(B * S * L, dtype=torch.float32, device=device)
        grid_act = (L,)
        sparse_relu_features[grid_act](inp_fp32, threshold, out_fp32, N_ROWS, L, num_warps=4)
        out_fp32 = out_fp32.view(B, S, L)

        # 5) Cast to bfloat16 via Triton kernel (forward must invoke it)
        out_bf16 = torch.empty(B * S * L, dtype=torch.bfloat16, device=device)
        grid_cast = (triton.cdiv(B * S * L, 4096),)
        cast_bf16_kernel[grid_cast](out_fp32.view(-1), out_bf16, B * S * L, BLOCK=4096, num_warps=4)

        return out_bf16.view(B, S, L)


def run(*args):
    return ModelNew()(*args)
