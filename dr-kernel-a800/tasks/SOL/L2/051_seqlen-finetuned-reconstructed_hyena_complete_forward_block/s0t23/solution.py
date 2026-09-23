import torch
import triton
import triton.language as tl


@triton.jit
def layernorm_2d_kernel(
    X_ptr,        # *const float, input [M, D], row-major contiguous
    Y_ptr,        # *float, output [M, D]
    W_ptr,        # *const float, weight (gamma) [D]
    B_ptr,        # *const float, bias (beta) [D]
    M,            # int, number of rows = B * S
    D,            # int, last-dim size
    EPS,          # float32, epsilon
    BLOCK_SIZE: tl.constexpr,  # tile size along D
):
    pid = tl.program_id(0)
    # Each program handles one row: row = pid
    if pid >= M:
        return

    row_start = pid * D

    # First pass: compute mean over D
    sum_ = 0.0
    for offs in range(0, D, BLOCK_SIZE):
        idx = offs + tl.arange(0, BLOCK_SIZE)
        mask = idx < D
        x = tl.load(X_ptr + row_start + idx, mask=mask, other=0.0)
        sum_ += tl.sum(x, axis=0)
    mean = sum_ / D

    # Second pass: compute variance over D
    var_sum = 0.0
    for offs in range(0, D, BLOCK_SIZE):
        idx = offs + tl.arange(0, BLOCK_SIZE)
        mask = idx < D
        x = tl.load(X_ptr + row_start + idx, mask=mask, other=0.0)
        diff = x - mean
        var_sum += tl.sum(diff * diff, axis=0)
    var = var_sum / D
    rstd = 1.0 / tl.sqrt(var + EPS)

    # Third pass: normalize, apply affine, and store
    for offs in range(0, D, BLOCK_SIZE):
        idx = offs + tl.arange(0, BLOCK_SIZE)
        mask = idx < D
        x = tl.load(X_ptr + row_start + idx, mask=mask, other=0.0)
        w = tl.load(W_ptr + idx, mask=mask, other=1.0)
        b = tl.load(B_ptr + idx, mask=mask, other=0.0)
        y = (x - mean) * rstd
        y = y * w + b
        tl.store(Y_ptr + row_start + idx, y, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states, norm1_weight, norm1_bias, layer_norm_eps):
        # hidden_states: [B, S, D], float32 on CUDA
        # norm1_weight, norm1_bias: [D], float32
        # layer_norm_eps: float
        # No torch ops allowed; only Triton kernels.
        B, S, D = hidden_states.shape
        M = B * S

        # Ensure contiguous row-major layout for kernel
        X = hidden_states.reshape(M, D).contiguous()
        Y = torch.empty_like(X)  # output tensor for normalized + affine

        # Launch Triton kernel: one program per row
        BLOCK_SIZE = 256  # tuned for D_model=256
        grid = (M,)

        layernorm_2d_kernel[grid](
            X, Y,
            norm1_weight, norm1_bias,
            M, D, layer_norm_eps,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4,
            num_stages=2,
        )

        # Reshape back to [B, S, D]
        return Y.view(B, S, D)


def run(*args):
    return ModelNew()(*args)
