import torch
import triton
import triton.language as tl

HIDDEN_SIZE = 4096
EPS = 1e-5

@triton.jit
def normalize_scale_row(hidden_ptr, weight_ptr, out_ptr,
                        B, H, stride_hs, stride_out, EPS):
    # One program per row
    row = tl.program_id(0)

    # Vector of column offsets
    offs = tl.arange(0, H)
    mask = offs < H  # all True for H=4096, kept for safety

    # Base pointers for this row
    hidden_row_ptr = hidden_ptr + row * stride_hs
    out_row_ptr = out_ptr + row * stride_out

    # Load the entire row (hidden_states) and cast to float32
    x = tl.load(hidden_row_ptr + offs, mask=mask, other=0.0)
    x32 = x.to(tl.float32)

    # Compute sum of squares across the row
    sumsq = tl.sum(x32 * x32, axis=0)

    # Compute inv_rms = rsqrt(mean(x^2) + EPS) where mean = sumsq / H
    inv_rms = tl.rsqrt(sumsq / H + EPS)

    # Load weight vector and cast to float32
    w = tl.load(weight_ptr + offs, mask=mask, other=0.0)
    w32 = w.to(tl.float32)

    # Compute output: y = x32 * inv_rms * w32
    y32 = x32 * inv_rms * w32

    # Store result
    tl.store(out_row_ptr + offs, y32, mask=mask)

class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor):
        # Ensure CUDA tensors and contiguous
        if not hidden_states.is_cuda or not weight.is_cuda:
            # Safe fallback to original PyTorch behavior if not on CUDA
            x = hidden_states.to(torch.float32)
            inv_rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + EPS)
            y = (x * inv_rms) * weight.to(torch.float32)
            return y.to(hidden_states.dtype)

        assert hidden_states.shape[-1] == HIDDEN_SIZE, "hidden_states must have 4096 features"
        B = hidden_states.shape[0]
        H = hidden_states.shape[-1]
        assert weight.shape[0] == H, "weight must have shape [4096]"

        # Ensure contiguity
        hidden_states = hidden_states.contiguous()
        weight = weight.contiguous()

        # Allocate FP32 output
        out = torch.empty((B, H), dtype=torch.float32, device=hidden_states.device)

        # Launch Triton kernel: one program per row
        grid = (B,)
        normalize_scale_row[grid](
            hidden_states, weight, out,
            B, H, hidden_states.stride(0), out.stride(0),
            EPS,
            num_warps=8,  # good default for 4096-wide vector
            num_stages=1,
        )

        # Cast back to original dtype to match original behavior
        return out.to(hidden_states.dtype)


def run(*args):
    return ModelNew()(*args)
