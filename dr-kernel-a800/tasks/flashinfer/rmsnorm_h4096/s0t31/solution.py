import torch

try:
    import triton
    import triton.language as tl
except Exception:
    triton = None

# Triton kernels
if triton is not None:

    @triton.jit
    def reduce_row_sumsq(hidden_ptr, inv_rms_ptr, B, H, EPS, BLOCK_SIZE: tl.constexpr):
        # One program per row
        row_id = tl.program_id(0)
        # Guard: if row_id >= B, exit (safety)
        if row_id >= B:
            return

        # Accumulate sum of squares in fp32
        sumsq = 0.0
        offs = tl.arange(0, BLOCK_SIZE)
        # Loop over the row in tiles
        for col_start in range(0, H, BLOCK_SIZE):
            cols = col_start + offs
            mask = cols < H
            x = tl.load(hidden_ptr + row_id * H + cols, mask=mask, other=0.0)
            x = x.to(tl.float32)
            sq = x * x
            # Zero out invalid lanes
            sq = tl.where(mask, sq, 0.0)
            sumsq += tl.sum(sq, axis=0)

        mean = sumsq / H
        inv = tl.rsqrt(mean + EPS)
        # Store inv_rms[row] as fp32
        tl.store(inv_rms_ptr + row_id, inv)

    @triton.jit
    def scale_row_elements(hidden_ptr, weight_ptr, inv_rms_ptr, out_ptr, B, H, BLOCK_SIZE: tl.constexpr):
        row_id = tl.program_id(0)
        if row_id >= B:
            return

        inv = tl.load(inv_rms_ptr + row_id)  # fp32 scalar per row
        offs = tl.arange(0, BLOCK_SIZE)
        for col_start in range(0, H, BLOCK_SIZE):
            cols = col_start + offs
            mask = cols < H
            x = tl.load(hidden_ptr + row_id * H + cols, mask=mask, other=0.0)
            w = tl.load(weight_ptr + cols, mask=mask, other=0.0)
            # Compute y = x * inv * w, do compute in fp32
            x = x.to(tl.float32)
            w = w.to(tl.float32)
            y = x * inv * w
            tl.store(out_ptr + row_id * H + cols, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states, weight):
        # Fallback to PyTorch if Triton is not available or tensors are not on CUDA
        if triton is None or not hidden_states.is_cuda:
            # Original PyTorch behavior as a fallback
            batch_size, hidden_size = hidden_states.shape
            assert hidden_size == 4096
            x = hidden_states.to(torch.float32)
            inv_rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-5)
            y = (x * inv_rms) * weight.to(torch.float32)
            return y.to(hidden_states.dtype)

        # Ensure contiguous and compute in fp32
        hidden = hidden_states.contiguous()
        weight = weight.contiguous()
        B, H = hidden.shape
        assert H == 4096, "This Triton-optimized implementation expects hidden_size == 4096."

        # Allocate output buffer (compute in fp32, cast at end)
        out_fp32 = torch.empty((B, H), dtype=torch.float32, device=hidden.device)

        # Per-row inverse RMS buffer (fp32)
        inv_rms = torch.empty((B,), dtype=torch.float32, device=hidden.device)

        # Launch reduction kernel
        grid = (B,)
        reduce_row_sumsq[grid](hidden, inv_rms, B, H, 1e-5, BLOCK_SIZE=2048, num_warps=8, num_stages=2)

        # Launch scaling kernel
        scale_row_elements[grid](hidden, weight, inv_rms, out_fp32, B, H, BLOCK_SIZE=2048, num_warps=8, num_stages=2)

        # Cast back to original dtype
        return out_fp32.to(hidden_states.dtype)


def run(*args):
    return ModelNew()(*args)
