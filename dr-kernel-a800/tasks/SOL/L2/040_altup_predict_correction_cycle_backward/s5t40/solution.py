import torch
import triton
import triton.language as tl


# Kernel 1: elementwise tanh over flat array
@triton.jit
def tanh_kernel(x_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    """
    Compute tanh(x) for each element in x_ptr and write to out_ptr.
    Grid: (ceil_div(N, BLOCK_SIZE),)
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    y = tl.tanh(x)
    tl.store(out_ptr + offsets, y, mask=mask)


# Kernel 2: compute sum of squares per token and produce rstd
@triton.jit
def rstd_kernel(x_ptr, B, S, H, eps, out_rstd_ptr, BLOCK_SIZE: tl.constexpr):
    """
    Grid: (B*S, ceil_div(H, BLOCK_SIZE))
    Each program computes partial sum of squares for one token over H and atomically adds into out_rstd_ptr[token].
    Note: out_rstd_ptr is actually a sum buffer; final rstd is computed on host as rsqrt(sum/H + eps).
    """
    pid_token = tl.program_id(0)
    pid_col = tl.program_id(1)
    b = pid_token // S
    s = pid_token % S
    base = b * S * H + s * H
    offsets = pid_col * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < H
    x = tl.load(x_ptr + base + offsets, mask=mask, other=0.0).to(tl.float32)
    sum_sq = tl.sum(x * x, axis=0)
    tl.atomic_add(out_rstd_ptr + pid_token, sum_sq)


# Kernel 3: broadcast-like elementwise product on flat arrays
@triton.jit
def elementwise_product_broadcast_kernel(A_ptr, B_ptr, C_ptr, N, BLOCK_SIZE: tl.constexpr):
    """
    Compute C[i] = A[i] * B[i] for i in [0, N).
    Grid: (ceil_div(N, BLOCK_SIZE),)
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    A = tl.load(A_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    B = tl.load(B_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    C = A * B
    tl.store(C_ptr + offsets, C, mask=mask)


# Kernel 4: linear row projection: out[i] = dot(x, W[i, :]) for a single row
@triton.jit
def linear_row_kernel(x_ptr, W_ptr, out_ptr, H, BLOCK_SIZE: tl.constexpr):
    """
    Grid: (H,)
    Each program computes one output element i = program_id(0): out[i] = dot(x, W[i, :])
    Iterate over H in chunks of BLOCK_SIZE.
    """
    i = tl.program_id(0)
    acc = 0.0
    for off in range(0, H, BLOCK_SIZE):
        idx = off + tl.arange(0, BLOCK_SIZE)
        mask = idx < H
        x = tl.load(x_ptr + idx, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W_ptr + i * H + idx, mask=mask, other=0.0).to(tl.float32)
        acc += tl.sum(x * w, axis=0)
    tl.store(out_ptr + i, acc)


class ModelNew(torch.nn.Module):
    def forward(self, grad_corrected: torch.Tensor,
                hidden_states: torch.Tensor,
                activated: torch.Tensor,
                prediction_coef_weight: torch.Tensor,
                correction_coef_weight: torch.Tensor,
                router_weight: torch.Tensor,
                norm_weight: torch.Tensor,
                altup_active_idx: int,
                rms_norm_eps: float):
        """
        This forward calls Triton kernels to simulate required computations.
        """
        device = grad_corrected.device
        dtype = torch.float32

        # Shapes
        B = hidden_states.shape[1]
        S = hidden_states.shape[2]
        H = hidden_states.shape[3]
        N = B * S * H

        # 1) tanh over activated: flatten to N
        act_flat = activated


def run(*args):
    return ModelNew()(*args)
