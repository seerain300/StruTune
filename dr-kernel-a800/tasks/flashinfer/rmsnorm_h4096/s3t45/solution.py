import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Specialized fused Triton kernel for H == 4096:
# Load entire row and weight vector at once, compute sum of squares, inv_rms, and output in one pass.
@triton.jit
def fused_rms_scale_kernel_fullrow(
    hidden_ptr,         # *const float32, shape [B, 4096], row-major contiguous
    weight_ptr,         # *const float32, shape [4096]
    out_ptr,            # *float32, shape [B, 4096]
    B: tl.int32,        # number of rows
    H: tl.int32,        # H must be 4096 here
    EPS: tl.float32,    # epsilon
    BLOCK_SIZE: tl.constexpr,  # BLOCK_SIZE = 4096 (constexpr)
):
    row = tl.program_id(axis=0)
    if row >= B:
        return
    cols = tl.arange(0, BLOCK_SIZE)
    # Load entire row and weight vector
    x = tl.load(hidden_ptr + row * H + cols)
    w = tl.load(weight_ptr + cols)
    # Compute sum of squares for this row
    sum_sq = tl.sum(x * x)
    # Compute inv_rms for this row: 1 / sqrt((sum_sq / H) + EPS)
    mean = sum_sq / H
    inv_rms = 1.0 / tl.sqrt(mean + EPS)
    # Compute output and store
    y = x * inv_rms * w
    tl.store(out_ptr + row * H + cols, y)


# Generic fused Triton kernel: two passes per row (tiled), Triton-only.
@triton.jit
def fused_rms_scale_kernel_tiled(
    hidden_ptr,         # *const float32, shape [B, H], row-major contiguous
    weight_ptr,         # *const float32, shape [H]
    out_ptr,            # *float32, shape [B, H]
    B: tl.int32,        # number of rows
    H: tl.int32,        # number of columns
    EPS: tl.float32,    # epsilon
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(axis=0)
    if row >= B:
        return

    sum_sq = 0.0
    # First pass: compute sum of squares over all columns
    for col_start in range(0, H, BLOCK_SIZE):
        cols = col_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < H
        x = tl.load(hidden_ptr + row * H + cols, mask=mask, other=0.0)
        sum_sq += tl.sum(x * x, axis=0)

    # Compute inv_rms for this row: 1 / sqrt((sum_sq / H) + EPS)
    mean = sum_sq / H
    inv_rms = 1.0 / tl.sqrt(mean + EPS)

    # Second pass: compute y = x * inv_rms * weight and store
    for col_start in range(0, H, BLOCK_SIZE):
        cols = col_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < H
        x = tl.load(hidden_ptr + row * H + cols, mask=mask, other=0.0)
        w = tl.load(weight_ptr + cols, mask=mask, other=0.0)
        y = x * inv_rms * w
        tl.store(out_ptr + row * H + cols, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        # Fallback if Triton not available or tensors not on CUDA
        if (not TRITON_AVAILABLE) or (not hidden_states.is_cuda) or (not weight.is_cuda):
            # Preserve original behavior using PyTorch for correctness in fallback
            x = hidden_states.to(torch.float32)
            inv_rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-5)
            y = (x * inv_rms) * weight.to(torch.float32)
            return y.to(hidden_states.dtype)

        # Ensure float32 and contiguous inputs for Triton
        hidden_f32 = hidden_states.to(torch.float32).contiguous()
        weight_f32 = weight.to(torch.float32).contiguous()

        B, H = hidden_f32.shape
        out = torch.empty((B, H), dtype=torch.float32, device=hidden_f32.device)

        # Launch specialized fast kernel when H == 4096
        if H == 4096:
            grid = (B,)
            fused_rms_scale_kernel_fullrow[grid](
                hidden_f32,
                weight_f32,
                out,
                B,
                H,
                1e-5,
                BLOCK_SIZE=4096,
                num_warps=8,   # large vector: more warps for throughput
                num_stages=2,
            )
        else:
            # Generic tiled fused kernel
            BLOCK_SIZE = 256
            grid = (B,)
            fused_rms_scale_kernel_tiled[grid](
                hidden_f32,
                weight_f32,
                out,
                B,
                H,
                1e-5,
                BLOCK_SIZE,
                num_warps=4,
                num_stages=2,
            )

        # Cast to original dtype of hidden_states and return on original device
        out = out.to(hidden_states.dtype)
        if hidden_states.device.type != "cuda":
            out = out.to(hidden_states.device)
        return out


def run(*args):
    return ModelNew()(*args)
