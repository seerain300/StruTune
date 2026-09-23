import torch
import triton
import triton.language as tl


@triton.jit
def _scale_row_kernel(
    x_ptr,            # *pointer to hidden_states (float32)
    weight_ptr,       # *pointer to weight (float32)
    out_ptr,          # *pointer to output (float32)
    B,                # batch size (rows)
    H,                # hidden size (columns)
    inv_rms_ptr,      # *pointer to per-row inv_rms (float32), shape [B]
    stride_x_row,     # stride for row in x (elements)
    stride_x_col,     # stride for col in x (elements)
    stride_out_row,   # stride for row in out (elements)
    stride_out_col,   # stride for col in out (elements)
):
    row = tl.program_id(0)
    if row >= B:
        return

    # Load per-row inv_rms
    inv_rms = tl.load(inv_rms_ptr + row)

    # Iterate across columns in a single iteration for H <= 8192 (common case H=4096)
    # If H is larger, Triton will still handle it, but we keep it simple and single-pass.
    cols = tl.arange(0, H)
    mask = cols < H

    # Load inputs
    x_vals = tl.load(x_ptr + row * stride_x_row + cols * stride_x_col, mask=mask, other=0.0)
    w_vals = tl.load(weight_ptr + cols, mask=mask, other=0.0)  # weight is 1D, contiguous

    # Compute output
    y = x_vals * inv_rms * w_vals

    # Store output (float32; host may cast to original dtype after)
    tl.store(out_ptr + row * stride_out_row + cols * stride_out_col, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states, weight):
        # Ensure tensors are CUDA and contiguous
        assert hidden_states.is_cuda and weight.is_cuda, "Inputs must be CUDA tensors"
        x = hidden_states.contiguous()
        w = weight.contiguous()

        # Compute in float32 for numerical stability
        x_fp32 = x.to(torch.float32)
        w_fp32 = w.to(torch.float32)

        B, H = x_fp32.shape
        assert H == 4096, "This implementation currently expects hidden_size == 4096"

        # Compute per-row inv_rms on host using PyTorch (no Triton static loop here)
        # inv_rms[i] = 1 / sqrt(mean_j(x[i, j]^2) + EPS)
        eps = 1e-5
        sumsq = (x_fp32 * x_fp32).sum(dim=-1, keepdim=True)  # [B, 1]
        mean = sumsq / H
        inv_rms = torch.rsqrt(mean + eps)  # [B]

        # Prepare output buffer (float32 compute)
        out = torch.empty((B, H), dtype=torch.float32, device=x_fp32.device)

        # Strides in elements
        stride_x_row = x_fp32.stride(0)
        stride_x_col = x_fp32.stride(1)
        stride_out_row = out.stride(0)
        stride_out_col = out.stride(1)

        # Launch one program per row; single pass over columns (no static loops)
        grid = (B,)
        _scale_row_kernel[grid](
            x_fp32, w_fp32, out,
            B, H, inv_rms,
            stride_x_row, stride_x_col,
            stride_out_row, stride_out_col,
            num_warps=4,  # reasonable default for 4096 elements per row
            num_stages=2,
        )

        # Cast to original dtype for return
        if x.dtype != torch.float32:
            out = out.to(x.dtype)
        return out


def run(*args):
    return ModelNew()(*args)
