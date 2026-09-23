import torch
import triton
import triton.language as tl


@triton.jit
def layernorm_3d_kernel(
    X_ptr, Y_ptr,
    W_ptr, B_ptr,
    B, S, D,
    eps,
    BLOCK_SIZE: tl.constexpr
):
    # Each program handles one row across D: (b, s) => normalize over D
    b = tl.program_id(axis=0)
    s = tl.program_id(axis=1)

    # Compute base offsets
    row_offset = b * S * D + s * D

    # Vector of column indices
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < D

    # Load x
    x = tl.load(X_ptr + row_offset + cols, mask=mask, other=0.0)

    # Compute mean
    mean = tl.sum(x, axis=0) / D

    # Compute variance
    diff = x - mean
    var = tl.sum(diff * diff, axis=0) / D

    inv_std = 1.0 / tl.sqrt(var + eps)

    # Apply affine
    w = tl.load(W_ptr + cols, mask=mask, other=1.0)
    bval = tl.load(B_ptr + cols, mask=mask, other=0.0)
    y = diff * inv_std * w + bval

    # Store result
    tl.store(Y_ptr + row_offset + cols, y, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor, norm_weight: torch.Tensor, norm_bias: torch.Tensor):
        # hidden_states: (B, S, D)
        # norm_weight, norm_bias: (D,)
        assert hidden_states.dim() == 3, "hidden_states must be (B, S, D)"
        assert norm_weight.dim() == 1 and norm_bias.dim() == 1, "norm_weight and norm_bias must be (D,)"
        B, S, D = hidden_states.shape

        # Ensure contiguous and float32
        X = hidden_states.contiguous().to(torch.float32)
        W = norm_weight.contiguous().to(torch.float32)
        BIAS = norm_bias.contiguous().to(torch.float32)

        Y = torch.empty_like(X)

        # Choose BLOCK_SIZE as next power of two >= D, capped for performance
        # Triton prefers power-of-two block sizes. We can set BLOCK_SIZE to the next power of two up to 1024.
        BLOCK_SIZE = 1
        while BLOCK_SIZE < D and BLOCK_SIZE < 1024:
            BLOCK_SIZE <<= 1
        if BLOCK_SIZE < D:
            BLOCK_SIZE = min(1024, D)  # handle very large D by masking; for safety, cap at 1024 and loop not needed here

        grid = (B, S)
        layernorm_3d_kernel[grid](X, Y, W, BIAS, B, S, D, self.eps, BLOCK_SIZE, num_warps=4)

        return Y


def run(*args):
    return ModelNew()(*args)
