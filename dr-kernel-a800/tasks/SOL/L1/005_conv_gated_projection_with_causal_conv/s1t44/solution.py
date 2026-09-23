import torch
import triton
import triton.language as tl


# Triton kernel: elementwise gating Bx = B * x_proj, both are (B, H, S)
@triton.jit
def TritonGateBxKernel(
    B_ptr, X_ptr, OUT_ptr,
    Bsz, H, S,
    BLOCK_H: tl.constexpr, BLOCK_S: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_s = tl.program_id(2)

    h_offsets = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    s_offsets = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)

    mask_h = h_offsets < H
    mask_s = s_offsets < S

    # Base offset for b
    base = pid_b * (H * S)

    b_ptrs = B_ptr + base + h_offsets[None, :] * S + s_offsets[:, None]
    x_ptrs = X_ptr + base + h_offsets[None, :] * S + s_offsets[:, None]
    out_ptrs = OUT_ptr + base + h_offsets[None, :] * S + s_offsets[:, None]

    b_vals = tl.load(b_ptrs, mask=mask_h[None, :] & mask_s[:, None], other=0.0).to(tl.float32)
    x_vals = tl.load(x_ptrs, mask=mask_h[None, :] & mask_s[:, None], other=0.0).to(tl.float32)

    out_vals = b_vals * x_vals

    tl.store(out_ptrs, out_vals, mask=mask_h[None, :] & mask_s[:, None])


# Triton kernel: final linear projection y = y_lin @ W^T + bias
# y_lin: (B, H, S), W: (H, S) where W is out_proj_weight (H, H), but we use y_lin @ W^T to get (B, S, H)
@triton.jit
def TritonFinalLinearKernel(
    Y_ptr, W_ptr, BIAS_ptr, OUT_ptr,
    Bsz, H, S,
    BLOCK_H: tl.constexpr, BLOCK_S: tl.constexpr,
):
    # We compute OUT[b, s, h] = sum_{hh=0..H-1} Y[b, h, s] * W[hh, s] + bias[h]
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_h = tl.program_id(2)

    h_offsets = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)  # output hidden dimension h
    s_offsets = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)  # sequence dimension s

    mask_h = h_offsets < H
    mask_s = s_offsets < S

    acc = tl.zeros((BLOCK_H, BLOCK_S), dtype=tl.float32)

    # Y has shape (B, H, S) with contiguous layout. For fixed (b, h, s), Y[b, h, s] is at index base + h*S + s
    # We need sum over hh: Y[b, hh, s] * W[hh, s] -> W is (H, S)
    for hh in range(0, H):
        y_ptrs = Y_ptr + pid_b * (H * S) + hh * S + s_offsets  # (BLOCK_S,)
        y_vals = tl.load(y_ptrs, mask=mask_s, other=0.0).to(tl.float32)  # (BLOCK_S,)

        w_ptrs = W_ptr + hh * S + s_offsets  # (BLOCK_S,)
        w_vals = tl.load(w_ptrs, mask=mask_s, other=0.0).to(tl.float32)  # (BLOCK_S,)

        # acc[h_offsets, s_offsets] += y_vals[None, :] * w_vals[:, None]
        acc += (y_vals[None, :] * w_vals[:, None])

    bias_vals = tl.load(BIAS_ptr + h_offsets, mask=mask_h, other=0.0).to(tl.float32)  # (BLOCK_H,)
    acc = acc + bias_vals[:, None]  # broadcast over s

    # OUT is (B, S, H), contiguous
    out_ptrs = OUT_ptr + pid_b * (S * H) + s_offsets[:, None] * H + h_offsets[None, :]
    tl.store(out_ptrs, acc, mask=mask_h[None, :] & mask_s[:, None])


class ModelNew(torch.nn.Module):
    def forward(
        self,
        x: torch.Tensor,
        in_proj_weight: torch.Tensor,
        in_proj_bias: torch.Tensor,
        conv_weight: torch.Tensor,
        conv_bias: torch.Tensor,
        out_proj_weight: torch.Tensor,
        out_proj_bias: torch.Tensor,
    ):
        # Ensure contiguity
        x = x.contiguous()
        in_proj_weight = in_proj_weight.contiguous()
        in_proj_bias = in_proj_bias.contiguous()
        conv_weight = conv_weight.contiguous()
        conv_bias = conv_bias.contiguous()
        out_proj_weight = out_proj_weight.contiguous()
        out_proj_bias = out_proj_bias.contiguous()

        Bsz, S, H = x.shape

        # 1) Compute BCx = F.linear(x, in_proj_weight, in_proj_bias) -> (B, S, 3H)
        BCx = torch.nn.functional.linear(x, in_proj_weight, in_proj_bias)  # (B, S, 3H)

        # 2) Transpose to (B, 3H, S) as in the original code
        BCx_T = BCx.transpose(-1, -2)  # (B, 3H, S)

        # 3) Split into B, C, x_proj: each (B, H, S)
        B = BCx_T[:, :H, :]            # (B, H, S)
        C = BCx_T[:, H:2 * H, :]       # (B, H, S)
        x_proj = BCx_T[:, 2 * H:, :]   # (B, H, S)

        # 4) Triton: elementwise gating Bx = B * x_proj
        Bx = torch.empty((Bsz, H, S), device=x.device, dtype=x.dtype)
        TritonGateBxKernel[(Bsz, triton.cdiv(H, 64), triton.cdiv(S, 64))](
            B, x_proj, Bx,
            Bsz, H, S,
            BLOCK_H=64, BLOCK_S=64
        )

        # 5) Grouped causal conv with PyTorch for correctness: conv_out (B, H, S)
        # Causal pad left by 3
        Bx_padded = torch.nn.functional.pad(Bx, (3, 0))  # pad left 3 zeros
        conv_weight_T = conv_weight.transpose(1, 2)      # (H, 4) for conv1d
        conv_out = torch.nn.functional.conv1d(
            Bx_padded, conv_weight_T, conv_bias, groups=H, stride=1, padding=0
        )  # (B, H, S)

        # 6) Output gating: y = C * conv_out, elementwise (B, H, S)
        y_gate = C * conv_out  # (B, H, S)

        # 7) Triton: final linear projection y_final = y_gate @ out_proj_weight^T + out_proj_bias
        # out_proj_weight: (H, H); we want (B, S, H) = y_gate @ (H, H)^T where (H, H)^T is (H, H)
        # But out_proj_weight is (H, H); we can view it as (N_out=H, K=H) and do y_gate @ W^T where W is (H, H)
        # TritonFinalLinearKernel expects W as (H, S) layout; here S=H, so we can pass out_proj_weight directly.
        # Note: TritonFinalLinearKernel uses W_ptr as (H, S). In our case S=H and out_proj_weight is (H, H).
        y_final = torch.empty((Bsz, S, H), device=x.device, dtype=x.dtype)
        TritonFinalLinearKernel[(Bsz, triton.cdiv(S, 64), triton.cdiv(H, 64))](
            y_gate, out_proj_weight, out_proj_bias, y_final,
            Bsz, H, S,
            BLOCK_H=64, BLOCK_S=64
        )

        return y_final


def run(*args):
    return ModelNew()(*args)
