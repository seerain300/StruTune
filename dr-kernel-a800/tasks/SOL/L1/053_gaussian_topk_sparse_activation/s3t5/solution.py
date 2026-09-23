import torch
import triton
import triton.language as tl


# Kernel: compute per-feature sum across all rows (flatten [B*S, L] => [NROWS, L]).
@triton.jit
def sum_per_feature_kernel(x_ptr, sum_ptr, NROWS, L, BLOCK: tl.constexpr):
    f = tl.program_id(0)  # feature index
    acc = 0.0
    # Iterate over rows in chunks of BLOCK
    for row_start in range(0, NROWS, BLOCK):
        rows = row_start + tl.arange(0, BLOCK)
        mask = rows < NROWS
        idx = rows * L + f
        vals = tl.load(x_ptr + idx, mask=mask, other=0.0)
        acc += tl.sum(vals, axis=0)
    tl.store(sum_ptr + f, acc)


# Kernel: compute per-feature sum of squares across all rows.
@triton.jit
def sumsq_per_feature_kernel(x_ptr, sumsq_ptr, NROWS, L, BLOCK: tl.constexpr):
    f = tl.program_id(0)  # feature index
    acc = 0.0
    for row_start in range(0, NROWS, BLOCK):
        rows = row_start + tl.arange(0, BLOCK)
        mask = rows < NROWS
        idx = rows * L + f
        vals = tl.load(x_ptr + idx, mask=mask, other=0.0)
        acc += tl.sum(vals * vals, axis=0)
    tl.store(sumsq_ptr + f, acc)


# Kernel: compute mean and std per feature from sum and sumsq.
@triton.jit
def compute_mean_std(sum_ptr, sumsq_ptr, mean_ptr, std_ptr, NROWS, L, BLOCK_F: tl.constexpr):
    f = tl.program_id(0)
    sum_f = tl.load(sum_ptr + f)
    sumsq_f = tl.load(sumsq_ptr + f)
    mean_f = sum_f / NROWS
    var_f = sumsq_f / NROWS - mean_f * mean_f
    std_f = tl.sqrt(var_f)
    tl.store(mean_ptr + f, mean_f)
    tl.store(std_ptr + f, std_f)


# Kernel: compute per-feature threshold: mean + std * std_multiplier.
@triton.jit
def compute_threshold(mean_ptr, std_ptr, std_multiplier_ptr, thr_ptr, L, BLOCK_F: tl.constexpr):
    f = tl.program_id(0)
    mean_f = tl.load(mean_ptr + f)
    std_f = tl.load(std_ptr + f)
    sm = tl.load(std_multiplier_ptr)  # scalar 0-dim tensor
    thr_f = mean_f + std_f * sm
    tl.store(thr_ptr + f, thr_f)


# Kernel: elementwise sparse ReLU with per-feature thresholds (broadcast along rows).
@triton.jit
def sparse_relu_feature_kernel(inp_ptr, thr_ptr, out_ptr, N, L, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    vals = tl.load(inp_ptr + offs, mask=mask, other=0.0)
    # feature index per element: f = offs % L
    f_idx = offs % L
    thr_vals = tl.load(thr_ptr + f_idx, mask=mask, other=0.0)
    out_vals = tl.maximum(vals - thr_vals, 0.0)
    tl.store(out_ptr + offs, out_vals, mask=mask)


# Kernel: cast FP32 to BF16. Must be invoked from forward (no torch .to).
@triton.jit
def cast_bf16_kernel(inp_fp32_ptr, out_bf16_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    vals = tl.load(inp_fp32_ptr + offs, mask=mask, other=0.0)
    vals_bf16 = vals.to(tl.bfloat16)
    tl.store(out_bf16_ptr + offs, vals_bf16, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, inputs: torch.Tensor, target_sparsity: float, std_multiplier: torch.Tensor):
        """
        inputs: [B, S, L], CUDA tensor. We compute:
          - mean and std along last dim (features), per (B,S)
          - threshold per feature: mean + std * std_multiplier
          - sparse ReLU: max(input - threshold[feature], 0)
        Return: [B, S, L] in bfloat16.
        """
        if not inputs.is_cuda:
            raise RuntimeError("ModelNew.forward expects CUDA tensors.")
        inputs = inputs.contiguous()
        B, S, L = inputs.shape
        NROWS = B * S

        # Work in FP32
        x2D = inputs.view(NROWS, L).contiguous().to(torch.float32)

        # 1) Per-feature sums and sumsqs
        sum_f = torch.empty(L, dtype=torch.float32, device=inputs.device)
        sumsq_f = torch.empty(L, dtype=torch.float32, device=inputs.device)
        BLOCK = 256
        grid_sum = (L,)
        sum_per_feature_kernel[grid_sum](x2D, sum_f, NROWS, L, BLOCK=BLOCK, num_warps=4)
        sumsq_per_feature_kernel[grid_sum](x2D, sumsq_f, NROWS, L, BLOCK=BLOCK, num_warps=4)

        # 2) Mean and std per feature
        mean_f = torch.empty(L, dtype=torch.float32, device=inputs.device)
        std_f = torch.empty(L, dtype=torch.float32, device=inputs.device)
        BLOCK_F = 256
        compute_mean_std[(grid_sum,)](sum_f, sumsq_f, mean_f, std_f, NROWS, L, BLOCK=BLOCK_F, num_warps=4)

        # 3) Per-feature threshold: mean + std * std_multiplier
        thr_f = torch.empty(L, dtype=torch.float32, device=inputs.device)
        compute_threshold[(grid_sum,)](mean_f, std_f, std_multiplier, thr_f, L, BLOCK=BLOCK_F, num_warps=4)

        # 4) Elementwise sparse ReLU with per-feature thresholds
        inp_flat = x2D.view(-1)  # [NROWS * L], FP32
        out_flat_fp32 = torch.empty(NROWS * L, dtype=torch.float32, device=inputs.device)
        BLOCK_ACT = 4096
        grid_act = (triton.cdiv(NROWS * L, BLOCK_ACT),)
        sparse_relu_feature_kernel[grid_act](inp_flat, thr_f, out_flat_fp32, NROWS * L, L, BLOCK=BLOCK_ACT, num_warps=8)

        # 5) Cast to bfloat16 via Triton (forward must invoke this kernel)
        out_bf16_flat = torch.empty(NROWS * L, dtype=torch.bfloat16, device=inputs.device)
        grid_cast = (triton.cdiv(NROWS * L, 4096),)
        cast_bf16_kernel[grid_cast](out_flat_fp32, out_bf16_flat, NROWS * L, BLOCK=4096, num_warps=4)

        # 6) Reshape back to [B, S, L]
        out = out_bf16_flat.view(B, S, L)
        return out


def run(*args):
    return ModelNew()(*args)
