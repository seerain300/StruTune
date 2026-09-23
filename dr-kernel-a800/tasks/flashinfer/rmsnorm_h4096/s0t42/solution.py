import torch
import triton
import triton.language as tl

# Triton kernels: reduction (sum of squares per row) and scaling
@triton.jit
def reduce_row_sumsq_kernel(x_ptr, inv_rms_ptr, B, H, EPS, BLOCK_SIZE: tl.constexpr):
    # One program per row
    row_id = tl.program_id(axis=0)
    # Guard: if grid > B, mask out
    if row_id >= B:
        return

    # Accumulator in fp32
    sumsq = 0.0
    # Loop over columns in tiles of BLOCK_SIZE
    for col_start in range(0, H, BLOCK_SIZE):
        cols = col_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < H
        # Row-major: offset = row_id * H + cols
        x = tl.load(x_ptr + row_id * H + cols, mask=mask, other=0.0)
        x_fp32 = x.to(tl.float32)
        sumsq += tl.sum(x_fp32 * x_fp32, axis=0)

    # Compute inv_rms
    mean = sumsq / H
    inv_rms = tl.rsqrt(mean + EPS)
    tl.store(inv_rms_ptr + row_id, inv_rms)


@triton.jit
def scale_row_elements_kernel(x_ptr, weight_ptr, inv_rms_ptr, out_ptr, B, H, BLOCK_SIZE: tl.constexpr):
    # One program per row
    row_id = tl.program_id(axis=0)
    if row_id >= B:
        return

    inv_rms = tl.load(inv_rms_ptr + row_id)  # scalar per row
    # Loop over columns
    for col_start in range(0, H, BLOCK_SIZE):
        cols = col_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < H

        # Load x row and weight vector
        x = tl.load(x_ptr + row_id * H + cols, mask=mask, other=0.0)
        w = tl.load(weight_ptr + cols, mask=mask, other=0.0)

        # Compute factor for this tile: inv_rms[row] * weight[j]
        factor = inv_rms * w.to(tl.float32)

        # Scale
        x_fp32 = x.to(tl.float32)
        y = x_fp32 * factor

        # Store
        tl.store(out_ptr + row_id * H + cols, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states, weight):
        # Expect hidden_states: [B, H], weight: [H]
        # The provided get_inputs uses H=4096
        x = hidden_states.contiguous()
        w = weight.contiguous()

        B, H = x.shape

        # Specialized path for H == 4096 (common case)
        if H == 4096:
            x_fp32 = x if x.dtype == torch.float32 else x.to(torch.float32)
            w_fp32 = w if w.dtype == torch.float32 else w.to(torch.float32)

            # Allocate output (fp32 compute, final cast back)
            out = torch.empty((B, H), dtype=torch.float32, device=x.device)
            inv_rms = torch.empty((B,), dtype=torch.float32, device=x.device)

            EPS = 1e-5

            # Launch reduction: one program per row
            reduce_row_sumsq_kernel[(B,)](
                x_fp32,
                inv_rms,
                B,
                H,
                EPS,
                BLOCK_SIZE=4096,
                num_warps=8,
                num_stages=3,
            )

            # Launch scaling: one program per row
            scale_row_elements_kernel[(B,)](
                x_fp32,
                w_fp32,
                inv_rms,
                out,
                B,
                H,
                BLOCK_SIZE=4096,
                num_warps=8,
                num_stages=3,
            )

            # Cast back to original dtype of hidden_states
            return out.to(hidden_states.dtype)
        else:
            # General path with masking, BLOCK_SIZE=1024
            x_fp32 = x if x.dtype == torch.float32 else x.to(torch.float32)
            w_fp32 = w if w.dtype == torch.float32 else w.to(torch.float32)

            out = torch.empty((B, H), dtype=torch.float32, device=x.device)
            inv_rms = torch.empty((B,), dtype=torch.float32, device=x.device)

            EPS = 1e-5

            reduce_row_sumsq_kernel[(B,)](
                x_fp32,
                inv_rms,
                B,
                H,
                EPS,
                BLOCK_SIZE=1024,
                num_warps=4,
                num_stages=3,
            )

            scale_row_elements_kernel[(B,)](
                x_fp32,
                w_fp32,
                inv_rms,
                out,
                B,
                H,
                BLOCK_SIZE=1024,
                num_warps=4,
                num_stages=3,
            )

            return out.to(hidden_states.dtype)


def run(*args):
    return ModelNew()(*args)
