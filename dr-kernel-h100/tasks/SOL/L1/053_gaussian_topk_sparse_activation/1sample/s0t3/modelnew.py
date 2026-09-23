import torch
import triton
import triton.language as tl


@triton.jit
def _reduce_mean_sumsq_kernel(
    X_ptr,          # *const float32, input data
    Mean_ptr,       # *float32, output: [B*S]
    Sumsq_ptr,      # *float32, output: [B*S]
    B: tl.int32,    # batch size
    S: tl.int32,    # seq len
    K: tl.int32,    # intermediate_size (features)
    stride_b: tl.int32,  # stride for batch in elements
    stride_s: tl.int32,  # stride for seq in elements
    stride_k: tl.int32,  # stride for feature in elements (usually 1 if contiguous)
    BLOCK_K: tl.constexpr,  # chunk size for reduction
):
    pid = tl.program_id(axis=0)  # program id over rows: 0 .. B*S - 1
    # Compute (b, s) from pid
    b = pid // S
    s = pid % S

    # Base pointer for this row
    base = b * stride_b + s * stride_s

    # Accumulators in fp32
    sum_val = 0.0
    sumsq_val = 0.0

    # Loop over K in chunks of BLOCK_K
    for start in range(0, K, BLOCK_K):
        offs = start + tl.arange(0, BLOCK_K)
        mask = offs < K
        x = tl.load(X_ptr + base + offs * stride_k, mask=mask, other=0.0)
        # x is fp32
        sum_val += tl.sum(x, axis=0)
        sumsq_val += tl.sum(x * x, axis=0)

    mean = sum_val / K
    sumsq = sumsq_val / K
    var = sumsq - mean * mean  # population variance (unbiased=False)
    # Store as fp32; host will cast to desired dtype after kernel
    tl.store(Mean_ptr + pid, mean)
    tl.store(Sumsq_ptr + pid, sumsq)


@triton.jit
def _apply_threshold_relu_kernel(
    X_ptr,          # *const float32, input data
    Out_ptr,        # *float32, output data
    Mean_ptr,       # *const float32, per-row mean
    Sumsq_ptr,      # *const float32, per-row sum of squares
    B: tl.int32,    # batch size
    S: tl.int32,    # seq len
    K: tl.int32,    # intermediate_size
    stride_b: tl.int32,
    stride_s: tl.int32,
    stride_k: tl.int32,
    Z: tl.float32,  # inverse normal CDF for target sparsity (host-provided constant)
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)  # program id over rows: 0 .. B*S - 1
    b = pid // S
    s = pid % S

    base = b * stride_b + s * stride_s

    # Load per-row statistics
    mean = tl.load(Mean_ptr + pid)
    sumsq = tl.load(Sumsq_ptr + pid)
    var = sumsq - mean * mean
    # Guard against tiny negative due to fp rounding
    var = tl.maximum(var, 0.0)
    std = tl.sqrt(var)

    # Precompute threshold multiplier per row: mean + std * Z
    threshold_const = mean + std * Z

    # Apply: Out = max(X - threshold_const, 0)
    for start in range(0, K, BLOCK_K):
        offs = start + tl.arange(0, BLOCK_K)
        mask = offs < K
        x = tl.load(X_ptr + base + offs * stride_k, mask=mask, other=0.0)
        y = x - threshold_const
        # ReLU: max(y, 0)
        y = tl.maximum(y, 0.0)
        tl.store(Out_ptr + base + offs * stride_k, y, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, target_sparsity: float = 0.9):
        super().__init__()
        # Precompute inverse normal CDF for target sparsity (host scalar).
        # Use torch.distributions.normal.icdf in __init__ so forward doesn't use torch ops.
        dist = torch.distributions.normal.Normal(0, 1)
        # icdf returns a 0-dim tensor; get its Python float value
        z_score = float(dist.icdf(torch.tensor(target_sparsity)))
        self.register_buffer("z_score", torch.tensor(z_score, dtype=torch.float32), persistent=False)

        # Tunable kernel config
        self.block_k = 256
        self.num_warps_reduce = 4
        self.num_warps_element = 4

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # If not CUDA, fall back to original PyTorch implementation (ensures correctness on CPU).
        if not inputs.is_cuda:
            # Original logic: compute in fp32, apply adaptive threshold via torch ops, return bf16
            inputs_f32 = inputs.to(torch.float32)
            # Compute mean and std along last dim
            inputs_mean = inputs_f32.mean(dim=-1, keepdim=True)
            inputs_std = inputs_f32.std(dim=-1, keepdim=True, unbiased=False)
            # Use precomputed z_score scalar
            z = float(self.z_score.item())
            cutoff_threshold = inputs_mean + inputs_std * z
            sparse_output = torch.relu(inputs_f32 - cutoff_threshold)
            return sparse_output.to(torch.bfloat16)

        # CUDA path: Triton-only computation
        # Ensure dtype is float32 for computation; output will be cast to bf16
        x = inputs.to(torch.float32).contiguous()
        B, S, K = x.shape
        stride_b = x.stride(0)
        stride_s = x.stride(1)
        stride_k = x.stride(2)

        # Allocate per-row mean and sumsq buffers (fp32)
        mean_row = torch.empty(B * S, dtype=torch.float32, device=x.device)
        sumsq_row = torch.empty(B * S, dtype=torch.float32, device=x.device)

        # Launch reduction kernel: compute mean and sumsq per row
        grid_reduce = (B * S,)
        _reduce_mean_sumsq_kernel[grid_reduce](
            x,
            mean_row,
            sumsq_row,
            B, S, K,
            stride_b, stride_s, stride_k,
            BLOCK_K=self.block_k,
            num_warps=self.num_warps_reduce,
            num_stages=1,
        )

        # Allocate output buffer (fp32) and launch elementwise kernel
        out_fp32 = torch.empty_like(x)
        grid_apply = (B * S,)
        z = float(self.z_score.item())
        _apply_threshold_relu_kernel[grid_apply](
            x,
            out_fp32,
            mean_row,
            sumsq_row,
            B, S, K,
            stride_b, stride_s, stride_k,
            z,
            BLOCK_K=self.block_k,
            num_warps=self.num_warps_element,
            num_stages=1,
        )

        # Match original return dtype: bfloat16
        return out_fp32.to(torch.bfloat16)