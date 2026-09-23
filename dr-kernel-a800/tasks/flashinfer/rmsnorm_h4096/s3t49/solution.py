import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernels: all math is done inside these kernels.

if TRITON_AVAILABLE:
    @triton.jit
    def fullrow_scale_kernel(
        hidden_ptr,      # *fp32, shape [B, H], contiguous
        weight_ptr,      # *fp32, shape [H], contiguous
        out_ptr,         # *fp32, shape [B, H], contiguous
        H: tl.constexpr, # 4096
        EPS: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,  # 4096
    ):
        row = tl.program_id(0)
        row_start = row * H
        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < H

        # Load entire row of hidden states
        x = tl.load(hidden_ptr + row_start + cols, mask=mask, other=0.0)

        # Compute sum of squares for this row
        sumsq = tl.sum(x * x, axis=0)
        mean = sumsq / H
        inv_rms = 1.0 / tl.sqrt(mean + EPS)

        # Load weight vector for all columns
        w = tl.load(weight_ptr + cols, mask=mask, other=0.0)

        # Compute output: y = x * inv_rms * w
        y = x * inv_rms * w

        # Store output
        tl.store(out_ptr + row_start + cols, y, mask=mask)

    # General tiled kernel for arbitrary H (two-pass). Not used in H==4096 case.
    @triton.jit
    def tiled_scale_kernel(
        hidden_ptr,      # *fp32, shape [B, H], contiguous
        weight_ptr,      # *fp32, shape [H], contiguous
        out_ptr,         # *fp32, shape [B, H], contiguous
        B,               # int32
        H,               # int32
        EPS,             # fp32
        BLOCK_SIZE: tl.constexpr,  # 256
    ):
        row = tl.program_id(0)
        # First pass: compute sum of squares for this row
        sumsq = tl.zeros((), dtype=tl.float32)  # scalar accumulator
        for col_start in range(0, H, BLOCK_SIZE):
            cols = col_start + tl.arange(0, BLOCK_SIZE)
            mask = cols < H
            x = tl.load(hidden_ptr + row * H + cols, mask=mask, other=0.0)
            sumsq += tl.sum(x * x, axis=0)
        mean = sumsq / H
        inv_rms = 1.0 / tl.sqrt(mean + EPS)

        # Second pass: compute and store output
        for col_start in range(0, H, BLOCK_SIZE):
            cols = col_start + tl.arange(0, BLOCK_SIZE)
            mask = cols < H
            x = tl.load(hidden_ptr + row * H + cols, mask=mask, other=0.0)
            w = tl.load(weight_ptr + cols, mask=mask, other=0.0)
            y = x * inv_rms * w
            tl.store(out_ptr + row * H + cols, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states, weight):
        # Triton execution requires CUDA tensors
        assert hidden_states.is_cuda and weight.is_cuda, "Triton requires CUDA tensors"
        assert hidden_states.dim() == 2, "hidden_states must be [B, H]"
        assert weight.dim() == 1 and weight.numel() == hidden_states.shape[1], "weight must be [H]"

        B, H = hidden_states.shape

        # Cast to float32 for compute and ensure contiguous
        hidden_f32 = hidden_states.contiguous().to(torch.float32)
        weight_f32 = weight.contiguous().to(torch.float32)

        # Allocate output as float32
        out = torch.empty((B, H), device=hidden_f32.device, dtype=torch.float32)

        EPS = 1e-5

        if H == 4096:
            # Full-row specialized kernel: one program per row
            grid = (B,)
            fullrow_scale_kernel[grid](
                hidden_f32,
                weight_f32,
                out,
                H=4096,
                EPS=EPS,
                BLOCK_SIZE=4096,
                num_warps=8,
                num_stages=2,
            )
        else:
            # General tiled kernel for arbitrary H
            grid = (B,)
            BLOCK_SIZE = 256
            tiled_scale_kernel[grid](
                hidden_f32,
                weight_f32,
                out,
                B,
                H,
                EPS,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=4,
                num_stages=2,
            )

        # Cast to original dtype of hidden_states for return (semantic match with original)
        return out.to(hidden_states.dtype)


def run(*args):
    return ModelNew()(*args)
