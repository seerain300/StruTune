import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Fused Triton kernel:
# For each row:
# - First pass: compute sum of squares of x[row, :] in tiles.
# - Compute inv_rms = 1.0 / sqrt((sum_sq / H) + EPS).
# - Second pass: compute y = x[row, :] * inv_rms * w[:] and store to out.
@triton.jit
def fused_rms_scale_kernel(
    hidden_ptr,         # *const float, [B, H]
    weight_ptr,         # *const float, [H]
    out_ptr,            # *float, [B, H]
    H,                  # int, hidden size
    EPS,                # float
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(axis=0)  # one program per row
    sum_sq = 0.0

    # First pass: sum of squares for this row
    for col_start in range(0, H, BLOCK_SIZE):
        cols = col_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < H
        x = tl.load(hidden_ptr + row * H + cols, mask=mask, other=0.0)
        sum_sq += tl.sum(x * x, axis=0)

    # Compute inv_rms for this row
    mean = sum_sq / H
    inv_rms = 1.0 / tl.sqrt(mean + EPS)

    # Second pass: produce output y = x * inv_rms * w and store
    for col_start in range(0, H, BLOCK_SIZE):
        cols = col_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < H
        x = tl.load(hidden_ptr + row * H + cols, mask=mask, other=0.0)
        w = tl.load(weight_ptr + cols, mask=mask, other=0.0)
        y = x * inv_rms * w
        tl.store(out_ptr + row * H + cols, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states, weight):
        # Fallback to PyTorch if Triton not available or not on CUDA
        if (not TRITON_AVAILABLE) or (hidden_states.device.type != "cuda"):
            batch_size, hidden_size = hidden_states.shape
            assert hidden_size == 4096
            x = hidden_states.to(torch.float32)
            EPS = 1e-5
            inv_rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + EPS)
            y = (x * inv_rms) * weight.to(torch.float32)
            return y.to(hidden_states.dtype)

        # Ensure inputs are on CUDA and float32, contiguous
        orig_device = hidden_states.device
        x = hidden_states.contiguous().to(torch.float32)  # [B, H]
        w = weight.contiguous().to(torch.float32)        # [H]

        B, H = x.shape
        assert H == 4096, "This implementation assumes hidden_size == 4096."

        # Allocate output as float32
        out_f32 = torch.empty((B, H), dtype=torch.float32, device=x.device)

        # Choose tile size: 1024 reduces loop iterations to 4 for H=4096.
        BLOCK_SIZE = 1024

        # Launch fused kernel: one program per row
        grid = (B,)
        fused_rms_scale_kernel[grid](
            x, w, out_f32, H, 1e-5, BLOCK_SIZE, num_warps=8, num_stages=3
        )

        # Cast to original dtype and return on original device
        out = out_f32.to(hidden_states.dtype)
        if orig_device.type != "cuda":
            out = out.to(orig_device)
        return out


def run(*args):
    return ModelNew()(*args)
