import torch
import triton
import triton.language as tl


@triton.jit
def sum_per_feature_kernel(
    x_ptr,                    # *const float32, input (we will cast as needed)
    NROWS,                    # int: B * S
    L,                        # int: intermediate_size (feature length)
    sum_ptr,                  # *float32, per-feature sum
    BLOCK_ROWS: tl.constexpr  # tile of rows per iteration
):
    # one program per feature
    f = tl.program_id(0)
    acc = 0.0
    row_start = 0
    while row_start < NROWS:
        rows = row_start + tl.arange(0, BLOCK_ROWS)
        mask_rows = rows < NROWS
        idx = rows * L + f
        vals = tl.load(x_ptr + idx, mask=mask_rows, other=0.0)  # vals are raw (likely fp32)
        # accumulate scalar sum
        acc += tl.sum(vals, axis=0)
        row_start += BLOCK_ROWS
    tl.store(sum_ptr + f, acc)


@triton.jit
def sumsq_per_feature_kernel(
    x_ptr,                    # *const float32
    NROWS,                    # int
    L,                        # int
    sumsq_ptr,                # *float32
    BLOCK_ROWS: tl.constexpr
):
    f = tl.program_id(0)
    acc = 0.0
    row_start = 0
    while row_start < NROWS:
        rows = row_start + tl.arange(0, BLOCK_ROWS)
        mask_rows = rows < NROWS
        idx = rows * L + f
        vals = tl.load(x_ptr + idx, mask=mask_rows, other=0.0)
        acc += tl.sum(vals * vals, axis=0)
        row_start += BLOCK_ROWS
    tl.store(sumsq_ptr + f, acc)


@triton.jit
def compute_mean_std_per_feature(
    sum_ptr,                  # *const float32
    sumsq_ptr,                # *const float32
    NROWS,                    # int
    L,                        # int (unused in this kernel, kept for clarity)
    mean_ptr,                 # *float32
    std_ptr,                  # *float32
):
    f = tl.program_id(0)
    sum_f = tl.load(sum_ptr + f)
    sumsq_f = tl.load(sumsq_ptr + f)
    rows_f32 = tl.cast(NROWS, tl.float32)
    # mean per feature
    mean_f = sum_f / rows_f32
    # variance per feature (population)
    var_f = sumsq_f / rows_f32 - mean_f * mean_f
    # avoid tiny negative due to round-off
    var_f = tl.maximum(var_f, 0.0)
    std_f = tl.sqrt(var_f)
    tl.store(mean_ptr + f, mean_f)
    tl.store(std_ptr + f, std_f)


@triton.jit
def broadcast_threshold_per_feature(
    mean_ptr,                 # *const float32
    std_ptr,                  # *const float32
    std_multiplier,           # scalar float32 (device tensor of 0-d)
    thr_ptr,                  # *float32
    L,                        # int
):
    f = tl.program_id(0)
    mean_f = tl.load(mean_ptr + f)
    std_f = tl.load(std_ptr + f)
    mult = std_multiplier  # scalar broadcast
    thr = mean_f + std_f * mult
    tl.store(thr_ptr + f, thr)


@triton.jit
def sparse_relu_feature_kernel(
    inp_ptr,                  # *const float32
    thr_ptr,                  # *const float32, per-feature threshold
    out_ptr,                  # *float32
    NROWS,                    # int
    L,                        # int
    BLOCK_ROWS: tl.constexpr
):
    # 2D grid: (features, tiles over rows)
    f = tl.program_id(0)
    tile_id = tl.program_id(1)
    row_start = tile_id * BLOCK_ROWS
    rows = row_start + tl.arange(0, BLOCK_ROWS)
    mask_rows = rows < NROWS
    idx = rows * L + f
    vals = tl.load(inp_ptr + idx, mask=mask_rows, other=0.0)
    thr = tl.load(thr_ptr + f)  # per-feature scalar
    res = vals - thr
    res = tl.maximum(res, 0.0)  # ReLU
    tl.store(out_ptr + idx, res, mask=mask_rows)


@triton.jit
def cast_fp32_to_bf16_kernel(
    inp_ptr,                  # *const float32
    out_ptr,                  # *bfloat16
    N,                        # int total elements
    BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    vals = tl.load(inp_ptr + offs, mask=mask, other=0.0)
    vals_bf16 = vals.to(tl.bfloat16)
    tl.store(out_ptr + offs, vals_bf16, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, std_multiplier: torch.Tensor):
        super().__init__()
        # std_multiplier is a 0-d float tensor on the correct device; forward will use it directly in Triton
        self.register_buffer('std_multiplier', std_multiplier, persistent=False)

    @torch.no_grad()
    def forward(self, inputs: torch.Tensor, target_sparsity: float) -> torch.Tensor:
        # inputs: [B, S, L], dtype likely float32/float16; we will compute in fp32 in Triton
        B, S, L = inputs.shape
        NROWS = B * S

        # Ensure inputs are contiguous and cast to fp32 for robust computation
        inp = inputs.contiguous()
        inp_fp32 = inp.float()  # in-place cast; no torch ops on tensor outputs later

        # 1) Allocate per-feature sum and sumsq (fp32)
        sum_per_feature = torch.empty(L, dtype=torch.float32, device=inp.device)
        sumsq_per_feature = torch.empty(L, dtype=torch.float32, device=inp.device)

        # 2) Triton: compute per-feature sum and sumsq across all rows
        grid_sum = (L,)
        sum_per_feature_kernel[grid_sum](
            inp_fp32,
            NROWS,
            L,
            sum_per_feature,
            BLOCK_ROWS=1024,
            num_warps=4
        )

        sumsq_per_feature_kernel[grid_sum](
            inp_fp32,
            NROWS,
            L,
            sumsq_per_feature,
            BLOCK_ROWS=1024,
            num_warps=4
        )

        # 3) Triton: compute mean and std per feature
        mean_per_feature = torch.empty(L, dtype=torch.float32, device=inp.device)
        std_per_feature = torch.empty(L, dtype=torch.float32, device=inp.device)
        compute_mean_std_per_feature[(L,)](
            sum_per_feature,
            sumsq_per_feature,
            NROWS,
            L,
            mean_per_feature,
            std_per_feature,
            num_warps=1
        )

        # 4) Triton: compute per-feature threshold: mean + std * std_multiplier
        thr_per_feature = torch.empty(L, dtype=torch.float32, device=inp.device)
        broadcast_threshold_per_feature[(L,)](
            mean_per_feature,
            std_per_feature,
            self.std_multiplier,  # 0-d tensor, device scalar
            thr_per_feature,
            L,
            num_warps=1
        )

        # 5) Triton: sparse ReLU with per-feature broadcast
        # We operate on inp_fp32, write out_fp32
        out_fp32 = torch.empty(B * S * L, dtype=torch.float32, device=inp.device)
        grid_relu = (L, triton.cdiv(NROWS, 1024))
        sparse_relu_feature_kernel[grid_relu](
            inp_fp32, thr_per_feature, out_fp32, NROWS, L, BLOCK_ROWS=1024, num_warps=4
        )
        out_fp32 = out_fp32.view(B, S, L)

        # 6) Triton: cast fp32 to bfloat16 (forward must invoke this kernel, no torch .to)
        out_bf16 = torch.empty_like(inputs, dtype=torch.bfloat16, device=inputs.device)
        grid_cast = (triton.cdiv(B * S * L, 4096),)
        cast_fp32_to_bf16_kernel[grid_cast](
            out_fp32.contiguous().view(-1),
            out_bf16.view(-1),
            B * S * L,
            BLOCK=4096,
            num_warps=4
        )

        return out_bf16


def run(*args):
    return ModelNew()(*args)
