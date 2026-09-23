import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def in_proj_linear_kernel(
    X_ptr,         # *ptr to input x: shape (B, S, H), contiguous
    W_ptr,         # *ptr to in_proj_weight: shape (I, H), contiguous, I = 3*H (constexpr)
    Out_ptr,       # *ptr to output: shape (B, S, I), contiguous
    B: tl.constexpr,
    S: tl.constexpr,
    H: tl.constexpr,   # hidden size, constexpr
    I: tl.constexpr,   # out channels, I = 3*H (constexpr)
):
    # Each program handles one (b, s)
    pid = tl.program_id(axis=0)
    b = pid // S
    s = pid % S

    x_base = X_ptr + b * (S * H) + s * H
    out_base = Out_ptr + b * (S * I) + s * I

    # For each output index i, compute acc[i] = sum_h x[b, s, h] * W[i, h]
    for i in range(0, I):
        # Load weights vector W[i, :]
        w_ptrs = W_ptr + i * H + tl.arange(0, H)
        w_vals = tl.load(w_ptrs).to(tl.float32)  # H is constexpr, so tl.arange(0, H) is valid

        # Load input vector x[b, s, :]
        x_vals = tl.load(x_base + tl.arange(0, H)).to(tl.float32)

        acc = tl.sum(w_vals * x_vals, axis=0)  # scalar

        # Store result at Out[b, s, i]
        out_ptr_i = out_base + i
        tl.store(out_ptr_i, acc)


@triton.jit
def out_proj_linear_kernel(
    Y_ptr,         # *ptr to input y: shape (B, S, H), contiguous
    W_ptr,         # *ptr to out_proj_weight: shape (H, H), contiguous
    Out_ptr,       # *ptr to output: shape (B, S, H), contiguous
    B: tl.constexpr,
    S: tl.constexpr,
    H: tl.constexpr,   # constexpr
):
    # Each program handles one (b, s)
    pid = tl.program_id(axis=0)
    b = pid // S
    s = pid % S

    y_base = Y_ptr + b * (S * H) + s * H
    out_base = Out_ptr + b * (S * H) + s * H

    # For each output index h_out in 0..H-1, compute sum over h_in
    for h_out in range(0, H):
        acc = 0.0
        for h_in in range(0, H):
            # y[b, s, h_in]
            y_val = tl.load(y_base + h_in).to(tl.float32)
            # W[h_out, h_in]
            w_val = tl.load(W_ptr + h_out * H + h_in).to(tl.float32)
            acc += y_val * w_val

        # Store result at Out[b, s, h_out]
        out_ptr = out_base + h_out
        tl.store(out_ptr, acc)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor,
                in_proj_weight: torch.Tensor,
                in_proj_bias: torch.Tensor,
                conv_weight: torch.Tensor,
                conv_bias: torch.Tensor,
                out_proj_weight: torch.Tensor,
                out_proj_bias: torch.Tensor):
        # x: (B, S, H), in_proj_weight: (I, H), I=3*H, conv_weight: (H, 1, 4), out_proj_weight: (H, H)
        # We preserve original semantics: PyTorch for conv and elementwise gating; Triton for the heavy linear ops.

        B, S, H = x.shape
        I = 3 * H

        # Ensure tensors are contiguous and float32
        x_contig = x.contiguous().to(torch.float32)
        in_proj_weight_contig = in_proj_weight.contiguous().to(torch.float32)

        # 1) Triton in_proj linear: compute BCx = X @ W_in^T, shape (B, S, I)
        BCx = torch.empty((B, S, I), device=x.device, dtype=torch.float32)

        grid_in = (B * S,)
        in_proj_linear_kernel[grid_in](
            x_contig, in_proj_weight_contig, BCx,
            B, S, H, I,
            num_warps=4,
        )

        # 2) Reconstruct B, C, x_proj via slicing (PyTorch ops; no heavy compute)
        # BCx shape (B,S,I), I=3H
        B_tensor = BCx[:, :, :H]   # (B, S, H)
        C_tensor = BCx[:, :, H:2*H]  # (B, S, H)
        x_proj_tensor = BCx[:, :, 2*H:]  # (B, S, H)

        # 3) Elementwise gating: Bx = B * x_proj
        Bx = B_tensor * x_proj_tensor  # elementwise multiply, torch

        # 4) Grouped causal conv: F.conv1d(Bx, conv_weight, conv_bias, groups=H)
        #    Bx shape: (B, H, S). Causal padding of


def run(*args):
    return ModelNew()(*args)
