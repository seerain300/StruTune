import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Kernel 1: compute inv_rms per row (float32)
# hidden_states_ptr: float32 [B, H], contiguous
# inv_rms_ptr: float32 [B], contiguous
@triton.jit
def rms_kernel(hidden_states_ptr, inv_rms_ptr, H: tl.constexpr, EPS: tl.float32):
    row_id = tl.program_id(0)  # one program per row
    # Base offset for this row
    row_base = hidden_states_ptr + row_id * H
    sumsq = 0.0
    # Iterate over columns in tiles of 256
    for col_start in range(0, H, 256):
        offs = col_start + tl.arange(0, 256)
        mask = offs < H
        x = tl.load(row_base + offs, mask=mask, other=0.0)  # float32
        x2 = x * x
        sumsq += tl.sum(x2, axis=0)
    mean = sumsq / H
    inv = tl.rsqrt(mean + EPS)
    tl.store(inv_rms_ptr + row_id, inv)


# Kernel 2: produce output y[row, col] = hidden_states[row, col] * inv_rms[row] * weight[col]
# hidden_states_ptr: float32 [B, H]
# weight_ptr: float32 [H]
# inv_rms_ptr: float32 [B]
# out_ptr: dtype of hidden_states (e.g., bfloat16) [B, H]
@triton.jit
def output_kernel(hidden_states_ptr, weight_ptr, inv_rms_ptr, out_ptr, H: tl.constexpr):
    row_id = tl.program_id(0)  # one program per row
    inv = tl.load(inv_rms_ptr + row_id)  # float32 scalar
    row_base_hs = hidden_states_ptr + row_id * H
    # Output row pointer in out tensor (same dtype as hidden_states)
    # We don't have dtype info in Triton; we'll store float32 and rely on host to cast if needed.
    # To keep output dtype correct, allocate out as the desired dtype on host and let Triton write float32 values,
    # but since Triton cannot query dtype of out_ptr, we will allocate out as float32 and cast on host before returning.
    # Therefore, we will store float32 here and cast after kernel launch.
    for col_start in range(0, H, 256):
        offs = col_start + tl.arange(0, 256)
        mask = offs < H
        x = tl.load(row_base_hs + offs, mask=mask, other=0.0)  # float32
        w = tl.load(weight_ptr + offs, mask=mask, other=0.0)   # float32
        y = x * inv * w  # float32
        tl.store(out_ptr + row_id * H + offs, y, mask=mask)    # store as float32


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states, weight):
        # If Triton unavailable, fallback to original PyTorch computation
        if not TRITON_AVAILABLE:
            batch_size, hidden_size = hidden_states.shape
            EPS = 1e-5
            x = hidden_states.to(torch.float32)
            inv_rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + EPS)
            y = (x * inv_rms) * weight.to(torch.float32)
            return y.to(hidden_states.dtype)

        # Move inputs to CUDA if needed
        orig_device = hidden_states.device
        if hidden_states.device.type != "cuda":
            hidden_states = hidden_states.to("cuda")
        if weight.device.type != "cuda":
            weight = weight.to("cuda")

        # Cast to float32 for compute
        x = hidden_states.to(torch.float32).contiguous()  # [B, H]
        w = weight.to(torch.float32).contiguous()         # [H]
        B, H = x.shape

        # Allocate inv_rms vector (float32)
        inv_rms = torch.empty(B, dtype=torch.float32, device=x.device)

        # Launch kernel 1: compute inv_rms per row
        grid_rms = (B,)
        EPS = 1e-5
        rms_kernel[grid_rms](x, inv_rms, H, EPS, num_warps=4, num_stages=2)

        # Allocate output tensor as float32 (we compute/store in float32 here)
        out_f32 = torch.empty((B, H), dtype=torch.float32, device=x.device)

        # Launch kernel 2: produce output in float32
        output_kernel[grid_rms](x, w, inv_rms, out_f32, H, num_warps=4, num_stages=2)

        # Cast to original dtype of hidden_states and return on original device
        out = out_f32.to(hidden_states.dtype)
        if orig_device.type != "cuda":
            out = out.to(orig_device)
        return out


def run(*args):
    return ModelNew()(*args)
