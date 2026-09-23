import torch
import triton
import triton.language as tl


# Triton kernel: RMSNorm per row over head_dim
# y = weight * x / sqrt(mean(x^2) + eps), applied to x with shape [B, H, L, D]
@triton.jit
def rmsnorm_row_kernel(
    x_ptr, out_ptr, norm_weight_ptr,
    B, H, L, D,
    stride_b, stride_h, stride_l, stride_d,
    eps: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # One program per row: row index is pid in [0, B*H*L)
    pid = tl.program_id(0)
    total_rows = B * H * L
    if pid >= total_rows:
        return

    # Decode (b, h, l) from pid
    b = pid // (H * L)
    rem = pid % (H * L)
    h = rem // L
    l = rem % L

    # Base offset for this row
    base = b * stride_b + h * stride_h + l * stride_l

    # Compute sum of squares across D
    sum_sq = 0.0
    for k in range(0, BLOCK):
        idx = k
        x_val = tl.load(x_ptr + base + idx * stride_d, mask=(idx < D), other=0.0)
        x_val_f32 = x_val.to(tl.float32)
        sum_sq += x_val_f32 * x_val_f32

    mean = sum_sq / D
    scale = tl.sqrt(eps + mean)
    inv_scale = 1.0 / scale  # fp32

    # Apply normalization and weight
    for k in range(0, BLOCK):
        idx = k
        x_val = tl.load(x_ptr + base + idx * stride_d, mask=(idx < D), other=0.0)
        w_val = tl.load(norm_weight_ptr + idx, mask=(idx < D), other=0.0)
        y = (x_val_f32 * inv_scale) * w_val.to(tl.float32)
        tl.store(out_ptr + base + idx * stride_d, y, mask=(idx < D))


def triton_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """
    Compute y = weight * x / sqrt(mean(x^2) + eps) per row using Triton.
    x: [B, H, L, D], dtype bfloat16 or float32
    weight: [D], dtype bfloat16 or float32
    Returns y with same shape/dtype as x.
    """
    assert x.is_cuda, "Input must be CUDA tensor for Triton kernel."
    B, H, L, D = x.shape
    assert weight.numel() == D, "weight must have length D"

    y = torch.empty_like(x)

    # Launch Triton kernel: one program per row
    grid = (B * H * L,)
    BLOCK = D  # process full head_dim; mask guards idx < D

    # Strides in elements
    stride_b = H * L * D
    stride_h = L * D
    stride_l = D
    stride_d = 1

    rmsnorm_row_kernel[grid](
        x, y, weight,
        B, H, L, D,
        stride_b, stride_h, stride_l, stride_d,
        eps,
        BLOCK=BLOCK,
    )
    return y


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # args: query, key, value, position_ids, key_cache, value_cache, cache_position, q_norm_weight, k_norm_weight, inv_freq, rms_norm_eps
        # Extract inputs; we will not use torch in host code beyond shape arithmetic and kernel launch.
        # We rely on get_inputs to provide tensors.

        # Note: The evaluator provides get_inputs separately; here we assume args are tensors as in the original.
        # However, since we cannot use torch.randn or torch.arange in host code, we must accept args as provided.

        query = args[0]
        key = args[1]
        value = args[2]
        position_ids = args[3]  # [B, L]
        key_cache = args[4]     # [B, num_kv_heads, MAX_LEN, head_dim]
        value_cache = args[5]   #


def run(*args):
    return ModelNew()(*args)
