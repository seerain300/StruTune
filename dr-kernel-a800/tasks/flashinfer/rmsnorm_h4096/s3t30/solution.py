import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Fused Triton kernel:
# For each row, compute inv_rms = 1 / sqrt(mean(x^2) + EPS) and then produce y[row, col] = x[row, col] * inv_rms * weight[col].
# hidden_states_ptr: float32 [B, H], row-major contiguous
# weight_ptr: float32 [H]
# out_ptr: float32 [B, H] (we will cast to original dtype after kernel execution)
@triton.jit
def fused_rms_output_kernel(hidden_states_ptr, weight_ptr, out_ptr, H: tl.constexpr, EPS: tl.float32, BLOCK_SIZE: tl.constexpr):
    row_id = tl.program_id(0)  # one program per row
    row_base = hidden_states_ptr + row_id * H
    sumsq = 0.0

    # First pass: compute sum of squares across the row
    for col_start in range(0, H, BLOCK_SIZE):
        offs = col_start + tl.arange(0, BLOCK_SIZE)
        mask = offs < H
        x = tl.load(row_base + offs, mask=mask, other=0.0)  # float32
        x2 = x * x
        sumsq += tl.sum(x2, axis=0)

    mean = sumsq / H
    inv = 1.0 / tl.sqrt(mean + EPS)

    # Second pass: compute output y = x * inv * weight
    for col_start in range(0, H, BLOCK_SIZE):
        offs = col_start + tl.arange(0, BLOCK_SIZE)
        mask = offs < H
        x = tl.load(row_base + offs, mask=mask, other=0.0)  # float32
        w = tl.load(weight_ptr + offs, mask=mask, other=0.0)  # float32
        y = x * inv * w  # float32
        tl.store(out_ptr + row_id * H + offs, y, mask=mask)   # store as float32


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states, weight):
        # If Triton is unavailable, do a PyTorch fallback for correctness.
        if not TRITON_AVAILABLE:
            batch_size, hidden_size = hidden_states.shape
            EPS = 1e-5
            x = hidden_states.to(torch.float32)
            sumsq = (x * x).sum(dim=-1, keepdim=True)  # [B, 1]
            inv_rms = 1.0 / torch.sqrt(sumsq / hidden_size + EPS)  # [B, 1]
            y = (x * inv_rms) * weight.to(torch.float32)
            return y.to(hidden_states.dtype)

        # Ensure CUDA tensors
        orig_device = hidden_states.device
        if hidden_states.device.type != "cuda":
            hidden_states = hidden_states.to("cuda")
        if weight.device.type != "cuda":
            weight = weight.to("cuda")

        # Cast to float32 for compute and make contiguous
        x = hidden_states.to(torch.float32).contiguous()  # [B, H]
        w = weight.to(torch.float32).contiguous()         # [H]
        B, H = x.shape

        # Allocate output tensor as float32
        out_f32 = torch.empty((B, H), dtype=torch.float32, device=x.device)

        # Launch fused kernel: one program per row
        grid = (B,)
        BLOCK_SIZE = 256
        EPS = 1e-5
        fused_rms_output_kernel[grid](x, w, out_f32, H, EPS, BLOCK_SIZE, num_warps=4, num_stages=2)

        # Cast to original dtype of hidden_states and return on original device
        out = out_f32.to(hidden_states.dtype)
        if orig_device.type != "cuda":
            out = out.to(orig_device)
        return out


def run(*args):
    return ModelNew()(*args)
