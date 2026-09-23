import torch
import triton
import triton.language as tl


@triton.jit
def reduce_sum_sumsq_per_feature(
    x_ptr,          # *float32, flattened [N = B*S*L]
    sum_ptr,        # *float32, output [L]
    sumsq_ptr,      # *float32, output [L]
    B: tl.int32, S: tl.int32, L: tl.int32,
    BLOCK_ROWS: tl.constexpr
):
    """
    For each feature f in [0, L), compute sum and sum of squares across all rows (across B*S).
    We launch one program per feature f. Inside, we loop over rows in chunks of BLOCK_ROWS.
    """
    f = tl.program_id(0)
    if f >= L:
        return
    total_sum = 0.0
    total_sumsq = 0.0
    for row_idx in range(0, B * S, BLOCK_ROWS):
        rows_off = row_idx + tl.arange(0, BLOCK_ROWS)  # [BLOCK_ROWS]
        rows_mask = rows_off < (B * S)
        # For each selected row, load its L elements at column f and accumulate
        for r in range(BLOCK_ROWS):
            row = rows_off[r]
            if row < (B * S):
                val = tl.load(x_ptr + row * L + f, mask=True, other=0.0)
                total_sum += val
                total_sumsq += val * val
    tl.store(sum_ptr + f, total_sum)
    tl.store(sumsq_ptr + f, total_sumsq)


@triton.jit
def compute_mean_std_per_feature(
    sum_ptr,        # [L] float32
    sumsq_ptr,      # [L] float32
    mean_ptr,       # [L] float32 output
    std_ptr,        # [L] float32 output
    B: tl.int32, S: tl.int32, L: tl.int32,
    std_multiplier,  # scalar float32
    BLOCK_SIZE: tl.constexpr
):
    """
    Compute mean and std per feature:
    mean[f] = sum[f] / (B*S), var[f] = sumsq[f] / (B*S) - mean[f]^2, std[f] = sqrt(var[f])
    Then compute threshold[f] = mean[f] + std[f] * std_multiplier and store both.
    """
    f = tl.program_id(0)
    if f >= L:
        return
    rows_total = B * S
    s = tl.load(sum_ptr + f)
    ss = tl.load(sumsq_ptr + f)
    mean = s / rows_total
    var = ss / rows_total - mean * mean
    var = tl.maximum(var, 0.0)  # guard against tiny negative due to FP errors
    std = tl.sqrt(var)
    thr = mean + std * std_multiplier
    tl.store(mean_ptr + f, mean)
    tl.store(std_ptr + f, std)
    tl.store(std_ptr + L, thr)  # store threshold in std_ptr + L (not std_ptr + f)


@triton.jit
def sparse_relu_per_feature(
    x_ptr,          # *float32, flattened [N = B*S*L]
    thr_ptr,        # *float32, [L] thresholds
    out_ptr,        # *float32, flattened [N]
    B: tl.int32, S: tl.int32, L: tl.int32,
    BLOCK_SIZE: tl.constexpr
):
    """
    For each row (over B*S), load the whole row, subtract the per-feature threshold, apply ReLU,
    and store. We iterate over features in chunks of BLOCK_SIZE.
    """
    row = tl.program_id(0)
    if row >= B * S:
        return
    base = row * L
    for f in range(0, L, BLOCK_SIZE):
        offs = f + tl.arange(0, BLOCK_SIZE)
        mask = offs < L
        x = tl.load(x_ptr + base + offs, mask=mask, other=0.0)
        thr = tl.load(thr_ptr + offs, mask=mask, other=0.0)
        y = x - thr
        y = tl.maximum(y, 0.0)  # ReLU
        tl.store(out_ptr + base + offs, y, mask=mask)


@triton.jit
def cast_bf16_kernel(
    inp_ptr,       # *float32, flattened [N]
    out_ptr,       # *bfloat16, flattened [N]
    N: tl.int32,
    BLOCK: tl.constexpr
):
    """
    Cast float32 to bfloat16: Triton will implicitly cast on store if out_ptr is bf16.
    We explicitly cast to bf16 via tl.store to out_ptr.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(inp_ptr + offs, mask=mask, other=0.0)
    # Store as bf16; Triton infers the destination pointer dtype
    tl.store(out_ptr + offs, x, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, std_multiplier: float):
        """
        std_multiplier is ndtri(target_sparsity), provided by the evaluator.
        """


def run(*args):
    return ModelNew()(*args)
