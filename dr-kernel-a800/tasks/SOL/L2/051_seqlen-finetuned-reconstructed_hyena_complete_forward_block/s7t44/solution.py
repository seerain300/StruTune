import torch
import torch.nn as nn

# Triton imports
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton LayerNorm forward kernel (FP32, per-row normalization)
# in_ptr: base pointer to input flattened as [M, D]
# w_ptr: gamma (weight) of length D
# b_ptr: beta   (bias)   of length D
# out_ptr: base pointer to output flattened as [M, D]
# M: number of rows, D: feature dimension
# eps: epsilon for LayerNorm
@triton.jit
def layernorm_fwd_kernel(in_ptr, w_ptr, b_ptr, out_ptr, M, D, eps):
    row = tl.program_id(0)
    # Guard for out-of-range programs (grid may be larger than M)
    if row >= M:
        return

    # Compute row start pointers
    in_row = in_ptr + row * D
    out_row = out_ptr + row * D

    # Pass 1: compute sum and sum of squares in FP32
    total_sum = 0.0
    total_sumsq = 0.0
    # We iterate over the row in chunks of BLOCK, but here BLOCK is >= D and masked.
    BLOCK = 1024  # must be >= D; used for vectorized operations
    cols = tl.arange(0, BLOCK)
    mask = cols < D

    x = tl.load(in_row + cols, mask=mask, other=0.0)
    x32 = x.to(tl.float32)
    total_sum += tl.sum(x32, axis=0)
    x2 = x32 * x32
    total_sumsq += tl.sum(x2, axis=0)

    mean = total_sum / D
    var = total_sumsq / D - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    # Pass 2: normalize and apply affine
    x = tl.load(in_row + cols, mask=mask, other=0.0)
    x32 = x.to(tl.float32)
    y = (x32 - mean) * rstd
    gamma = tl.load(w_ptr + cols, mask=mask, other=1.0).to(tl.float32)
    beta = tl.load(b_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = y * gamma + beta

    # Store result (cast back to original dtype)
    # Note: out_ptr is FP32 (we use FP32 throughout for accumulation)
    tl.store(out_row + cols, y, mask=mask)


def _next_power_of_two(n: int) -> int:
    # Return the next power of two >= n, capped at 1024
    v = 1
    while v < n and v < 1024:
        v <<= 1
    return v


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Expect: hidden_states, norm1_weight, norm1_bias, norm2_weight, norm2_bias
        if len(args) < 5:
            raise RuntimeError("ModelNew.forward expects at least 5 arguments: hidden_states, norm1_weight, norm1_bias, norm2_weight, norm2_bias")
        hidden = args[0]
        w1 = args[1]
        b1 = args[2]
        w2 = args[3]
        b2 = args[4]

        # Get shapes (d_model is fixed at 256 per prompt; we also infer from hidden)
        # Note: Triton kernel does not use any PyTorch tensor methods for computation.
        B, S, D = hidden.shape
        M = B * S

        # Allocate outputs as [M, D] in FP32
        # We do not use PyTorch methods like .reshape here; Triton pointer arithmetic is used below.
        # However, we still need to "conceptually" view outputs as [M, D]; Triton will get pointers directly.
        # Prepare pointers: Triton will get the data via args; we ensure w and b are contiguous.
        # Create flattened views (conceptually) for kernel: we pass pointers directly.

        # We need to flatten input logically: we will pass pointers and M, D.
        # To keep it simple and correct, we ensure inputs are contiguous and pass base pointers.
        hidden_c = hidden  # assume contiguous from caller; if not, we can call .contiguous(), but here we avoid PyTorch methods entirely.
        # Create outputs as FP32 tensors of shape [B, S, D]; we will write via Triton with per-row indexing.
        y1 = torch.empty((B, S, D), dtype=torch.float32, device=hidden.device)
        y2 = torch.empty((B, S, D), dtype=torch.float32, device=hidden.device)

        # Ensure weights/biases are contiguous (FP32)
        w1_c = w1.contiguous().to(torch.float32)
        b1_c = b1.contiguous().to(torch.float32)
        w2_c = w2.contiguous().to(torch.float32)
        b2_c = b2.contiguous().to(torch.float32)

        # Launch first LayerNorm
        BLOCK = _next_power_of_two(D)
        grid = (M,)
        eps = 1e-5
        layernorm_fwd_kernel[grid](
            hidden_c, w1_c, b1_c, y1, M, D, eps,
            BLOCK=BLOCK,
            num_warps=4,
        )

        # Launch second LayerNorm
        layernorm_fwd_kernel[grid](
            y1, w2_c, b2_c, y2, M, D, eps,
            BLOCK=BLOCK,
            num_warps=4,
        )

        return y2


def run(*args):
    return ModelNew()(*args)
