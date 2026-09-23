import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Fused Triton kernel: one program per row
# - First pass: compute sum of squares of the row in tiles.
# - Compute inv_rms = 1.0 / sqrt((sumsq / H) + EPS).
# - Second pass: compute y = x * inv_rms * w and store.
@triton.jit
def fused_rms_scale_kernel(
    x_ptr,          # *float32, shape [B, H], row-major contiguous
    w_ptr,          # *float32, shape [H]
    out_ptr,        # *float32, shape [B, H]
    B, H,           # int32
    EPS,            # float32
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= B:
        return

    # Accumulate sum of squares for this row
    sumsq = 0.0
    for col_start in range(0, H, BLOCK_SIZE):
        cols = col_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < H
        x = tl.load(x_ptr + row * H + cols, mask=mask, other=0.0)
        x2 = x * x
        sumsq += tl.sum(x2, axis=0)

    # Compute inv_rms for this row: 1 / sqrt(mean + EPS)
    mean = sumsq / H
    inv = 1.0 / tl.sqrt(mean + EPS)

    # Second pass: compute output y = x * inv * w and store
    for col_start in range(0, H, BLOCK_SIZE):
        cols = col_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < H
        x = tl.load(x_ptr + row * H + cols, mask=mask, other=0.0)
        w = tl.load(w_ptr + cols, mask=mask, other=0.0)
        y = x * inv * w
        tl.store(out_ptr + row * H + cols, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states, weight):
        # Ensure shapes are as expected
        assert hidden_states.dim() == 2, "hidden_states must be 2D [B, H]"
        assert weight.dim() == 1 and weight.shape[0] == hidden_states.shape[1], "weight must be 1D [H]"

        # If inputs are on CPU and Triton is available, move to CUDA for computation
        orig_device = hidden_states.device
        if orig_device.type != "cuda" and TRITON_AVAILABLE:
            hidden = hidden_states.to("cuda")
            weight_t = weight.to("cuda")
        else:
            hidden = hidden_states
            weight_t = weight

        # Cast to float32 for computation
        hidden_f32 = hidden.contiguous().to(torch.float32)
        weight_f32 = weight_t.contiguous().to(torch.float32)

        B, H = hidden_f32.shape

        # Allocate output as float32
        out = torch.empty((B, H), dtype=torch.float32, device=hidden_f32.device)

        # Launch fused Triton kernel: one program per row
        BLOCK_SIZE = 256  # works well for H=4096 (16 tiles)
        grid = (B,)
        fused_rms_scale_kernel[grid](
            hidden_f32,
            weight_f32,
            out,
            B,
            H,
            1e-5,  # EPS
            BLOCK_SIZE,
            num_warps=4,
            num_stages=2,
        )

        # Cast to original dtype of hidden_states and return on original device
        out = out.to(hidden_states.dtype)
        if orig_device.type != "cuda":
            out = out.to(orig_device)
        return out


def run(*args):
    return ModelNew()(*args)
