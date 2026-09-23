import torch
import triton
import triton.language as tl


# Triton kernel: compute linear projection for a given group M in {0,1,2}, mapping x[B, S, H] to y[B, S, H].
# We implement Y[b, s, m] = sum_i X[b, s, i] * W[M, m, i] + bias[M, m]
# Weight layout: in_proj_weight is (3H, H). For M, W[M, m, i] = in_proj_weight[M*H + m, i].
@triton.jit
def TritonLinearKernel(
    X_ptr,               # *T, shape [B, S, H]
    W_ptr,               # *T, shape [(3*M) + H] flattened; here we only read up to H entries for the current M
    BIAS_ptr,            # *T, shape [H]
    Y_ptr,               # *T, shape [B, S, H]
    B, S, H,
    M_start,             # starting group index in {0,1,2} mapping to in_proj_weight slices
    BLOCK_S: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_h = tl.program_id(2)

    # tile offsets
    s_offsets = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    h_offsets = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)

    mask_s = s_offsets < S
    mask_h = h_offsets < H

    # initialize accumulator
    acc = tl.zeros((BLOCK_S, BLOCK_H), dtype=tl.float32)

    # for each input feature i in H, accumulate X * W
    # X[b, s, i] stored contiguously with stride along s
    # weight W[M, m, i] = in_proj_weight[M*H + m, i] -> index in W_ptr is (M_start*H + m) * H + i
    for i in range(0, H):
        # load X[b, s, i]
        x_ptrs = X_ptr + pid_b * (S * H) + s_offsets * H + i
        x_vals = tl.load(x_ptrs, mask=mask_s, other=0.0)  # shape [BLOCK_S]
        # load W[M, :, i] -> shape [H]
        w_base = (M_start * H) * H  # since W_ptr is flattened but in_proj_weight has shape (3H, H)
        # W[M, m, i] index = w_base + m * H + i, m in [0..H-1]
        w_ptrs = W_ptr + w_base + h_offsets * H + i  # shape [BLOCK_H]
        w_vals = tl.load(w_ptrs, mask=mask_h, other=0.0)
        # outer product accumulate: acc += x_vals[:, None] * w_vals[None, :]
        acc += x_vals[:, None] * w_vals[None, :]

    # add bias: bias[M, m] for m in [0..H-1]
    bias_vals = tl.load(BIAS_ptr + h_offsets, mask=mask_h, other=0.0)  # [BLOCK_H]
    acc += bias_vals[None, :]

    # store to Y[b, s, m]
    y_ptrs = Y_ptr + pid_b * (S * H) + s_offsets[:, None] * H + h_offsets[None, :]
    tl.store(y_ptrs, acc, mask=mask_s[:, None] & mask_h[None, :])


# Triton kernel: element-wise gating Bx = B * x_proj over (B, S, H)
@triton.jit
def TritonGateKernel(
    B_ptr, X_ptr, OUT_ptr,
    Bsz, S, H,
    BLOCK_S: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_h = tl.program_id(2)

    s_offsets = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    h_offsets = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)

    mask_s = s_offsets < S
    mask_h = h_offsets < H

    b_ptrs = B_ptr + pid_b * (S * H) + s_offsets[:, None] * H + h_offsets[None, :]
    x_ptrs = X_ptr + pid_b * (S * H) + s_offsets[:, None] * H + h_offsets[None, :]

    b_vals = tl.load(b_ptrs, mask=mask_s[:, None] & mask_h[None, :], other=0.0)
    x_vals = tl.load(x_ptrs, mask=mask_s[:, None] & mask_h[None, :], other=0.0)

    out_vals = b_vals * x_vals

    out_ptrs = OUT_ptr + pid_b * (S * H) + s_offsets[:, None] * H + h_offsets[None, :]
    tl.store(out_ptrs, out_vals, mask=mask_s[:, None] & mask_h[None, :])


# Triton kernel: left-pad along sequence by PAD for Bx, producing Bx_pad[B, H, S + PAD]
# Inputs: Bx [B, H, S], output: out_pad [B, H, S + PAD], PAD=3
@triton.jit
def TritonPadLeftKernel(
    Bx_ptr, out_ptr,
    B, H, S, PAD,
    stride_bx_b, stride_bx_h, stride_bx_s,
    stride_ob_b, stride_ob_h, stride_ob_s,
    BLOCK_S: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_sp = tl.program_id(2)  # tiling over S_out = S + PAD

    b = pid_b
    h = pid_h
    S_out = S + PAD

    s_out_start = pid_sp * BLOCK_S
    # write zeros at the first PAD columns
    for i in range(0, PAD):
        out_ptr_pos = out_ptr + b * stride_ob_b + h * stride_ob_h + i * stride_ob_s
        tl.store(out_ptr_pos, 0.0)

    # copy from Bx[:, :, :] into out[:, :, PAD:]
    for i in range(0, BLOCK_S):
        s_in = s_out_start + i
        if s_in < S:
            bx_ptr = Bx_ptr + b * stride_bx_b + h * stride_bx_h + s_in * stride_bx_s
            val = tl.load(bx_ptr)
            out_ptr_pos = out_ptr + b * stride_ob_b + h * stride_ob_h + (s_in + PAD) * stride_ob_s
            tl.store(out_ptr_pos, val)


# Triton kernel: grouped causal 1D convolution (depthwise groups=H, kernel_size=4, causal padding 3)
# Input: Bx_pad [B, H, S+3], weight [H, 1, 4], bias [H]
# Output: conv_out [B, H, S]
@triton.jit
def TritonGroupedCausalConvKernel(
    Bx_pad_ptr,           # *T, shape [B, H, S + 3]
    Weight_ptr,           # *T, shape [H, 4], where weight[h, k]
    Bias_ptr,             # *T, shape [H]
    OUT_ptr,              # *T, shape [B, H, S]
    B, H, S, PAD,         # S_out = S + PAD
    BLOCK_S: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_s = tl.program_id(2)

    s_out_start = pid_s * BLOCK_S
    h_offsets = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = h_offsets < H

    # For each output position s in [0..S-1], compute sum over k=0..3 of Bx_pad[b, h, s + k] * weight[h, k] + bias[h]
    for i in range(0, BLOCK_S):
        s_out = s_out_start + i
        if s_out < S:
            # base pointer for this batch and sequence position for each h
            for h in range(0, BLOCK_H):
                if mask_h[h]:
                    base = Bx_pad_ptr + pid_b * (H * (S + PAD)) + h * (S + PAD) + s_out
                    acc = tl.zeros((), dtype=tl.float32)
                    # accumulate over k in {0..3}
                    for k in range(0, 4):
                        ptr_k = base + k
                        val_k = tl.load(ptr_k)
                        w_ptr = Weight_ptr + h * 4 + k
                        w_k = tl.load(w_ptr)
                        acc += val_k * w_k
                    # add bias
                    bias_k = tl.load(Bias_ptr + h)
                    acc += bias_k
                    # store to OUT[b, h, s]
                    out_ptr = OUT_ptr + pid_b * (H * S) + h * S + s_out
                    tl.store(out_ptr, acc)


# Triton kernel: final projection with bias
# Y: [B, S, H], OUT_proj: [H, H], OUT_bias: [H]
# output[B, S, H] = sum_n Y[b, s, n] * OUT_proj[n, h] + OUT_bias[h]
@triton.jit
def TritonFinalProjectionKernel(
    Y_ptr,               # *T, shape [B, S, H]
    OUT_proj_ptr,        # *T, shape [H, H]
    OUT_bias_ptr,        # *T, shape [H]
    OUT_ptr,             # *T, shape [B, S, H]
    B, S, H,
    BLOCK_S: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_h = tl.program_id(2)

    s_offsets = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    h_offsets = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)

    mask_s = s_offsets < S
    mask_h = h_offsets < H

    # accumulator per (s, h)
    acc = tl.zeros((BLOCK_S, BLOCK_H), dtype=tl.float32)

    # Y[b, s, n] dot OUT_proj[n, h] over n in [0..H-1]
    for n in range(0, H):
        y_ptrs = Y_ptr + pid_b * (S * H) + s_offsets * H + n  # [BLOCK_S]
        y_vals = tl.load(y_ptrs, mask=mask_s, other=0.0)  # [BLOCK_S]
        # OUT_proj[n, h]
        out_proj_ptrs = OUT_proj_ptr + n * H + h_offsets  # [BLOCK_H]
        out_proj_vals = tl.load(out_proj_ptrs, mask=mask_h, other=0.0)  # [BLOCK_H]
        acc += y_vals[:, None] * out_proj_vals[None, :]

    # add bias
    bias_vals = tl.load(OUT_bias_ptr + h_offsets, mask=mask_h, other=0.0)  # [BLOCK_H]
    acc += bias_vals[None, :]

    # store
    out_ptrs = OUT_ptr + pid_b * (S * H) + s_offsets[:, None] * H + h_offsets[None, :]
    tl.store(out_ptrs, acc, mask=mask_s[:, None] & mask_h[None, :])


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor,
                in_proj_weight: torch.Tensor, in_proj_bias: torch.Tensor,
                conv_weight: torch.Tensor, conv_bias: torch.Tensor,
                out_proj_weight: torch.Tensor, out_proj_bias: torch.Tensor):
        """
        Triton-only implementation of the original run function.
        All heavy computation is done by Triton kernels. PyTorch is only used for tensor
        allocations and basic shapes. We avoid any torch ops for compute.
        """

        assert x.is_cuda, "Input x must be CUDA tensor."
        assert in_proj_weight.is_cuda and in_proj_bias.is_cuda, "in_proj_weight and in_proj_bias must be CUDA."
        assert conv_weight.is_cuda and conv_bias.is_cuda, "conv_weight and conv_bias must be CUDA."
        assert out_proj_weight.is_cuda and out_proj_bias.is_cuda, "out_proj_weight and out_proj_bias must be CUDA."

        # Ensure contiguous
        x = x.contiguous()
        in_proj_weight = in_proj_weight.contiguous()
        in_proj_bias = in_proj_bias.contiguous()
        conv_weight = conv_weight.contiguous()
        conv_bias = conv_bias.contiguous()
        out_proj_weight = out_proj_weight.contiguous()
        out_proj_bias = out_proj_bias.contiguous()

        B, S, H = x.shape

        # 1) Three linear projections: M in {0,1,2} over in_proj_weight slices
        # We will compute B, C, x_proj separately.
        # Launch TritonLinearKernel three times.
        # For M=0: W_ptr points to in_proj_weight[:H, :], bias to in_proj_bias[:H]
        W0 = in_proj_weight[:H, :].contiguous()     # (H, H), flattened -> size H*H
        B_bias = in_proj_bias[:H].contiguous()      # (H,)
        B_out = torch.empty((B, S, H), dtype=x.dtype, device=x.device)

        # Grid: (B, ceil(S/BLOCK_S), ceil(H/BLOCK_H))
        BLOCK_S, BLOCK_H = 128, 32
        grid = (B, triton.cdiv(S, BLOCK_S), triton.cdiv(H, BLOCK_H))
        TritonLinearKernel[grid](
            x, W0, B_bias, B_out,
            B, S, H,
            0,
            BLOCK_S=BLOCK_S, BLOCK_H=BLOCK_H
        )

        # For M=1: C
        W1 = in_proj_weight[H:2 * H, :].contiguous()
        C_bias = in_proj_bias[H:2 * H].contiguous()
        C_out = torch.empty((B, S, H), dtype=x.dtype, device=x.device)
        TritonLinearKernel[grid](
            x, W1, C_bias, C_out,
            B, S, H,
            1,
            BLOCK_S=BLOCK_S, BLOCK_H=BLOCK_H
        )

        # For M=2: x_proj
        W2 = in_proj_weight[2 * H:3 * H, :].contiguous()
        x_proj_bias = in_proj_bias[2 * H:3 * H].contiguous()
        x_proj = torch.empty((B, S, H), dtype=x.dtype, device=x.device)
        TritonLinearKernel[grid](
            x, W2, x_proj_bias, x_proj,
            B, S, H,
            2,
            BLOCK_S=BLOCK_S, BLOCK_H=BLOCK_H
        )

        # 2) Element-wise gating: Bx = B * x_proj
        Bx = torch.empty((B, S, H), dtype=x.dtype, device=x.device)
        grid_gate = (B, triton.cdiv(S, BLOCK_S), triton.cdiv(H, BLOCK_H))
        TritonGateKernel[grid_gate](
            B_out, x_proj, Bx,
            B, S, H,
            BLOCK_S=BLOCK_S, BLOCK_H=BLOCK_H
        )

        # 3) Pad left for causal conv
        PAD = 3
        S_out = S + PAD
        Bx_padded = torch.empty((B, H, S_out), dtype=x.dtype, device=x.device)
        stride_bx_b, stride_bx_h, stride_bx_s = H * S_out, S_out, 1  # not used but define for Triton
        stride_ob_b, stride_ob_h, stride_ob_s = H * S_out, S_out, 1  # not used but define for Triton
        # Use TritonPadLeftKernel
        TritonPadLeftKernel[(B, triton.cdiv(H, BLOCK_H), triton.cdiv(S_out, BLOCK_S))](
            Bx, Bx_padded,
            B, H, S, PAD,
            stride_bx_b, stride_bx_h, stride_bx_s,
            stride_ob_b, stride_ob_h, stride_ob_s,
            BLOCK_S=BLOCK_S
        )

        # 4) Grouped causal conv: depthwise groups=H, kernel_size=4
        conv_weight_4 = conv_weight  # shape (H, 1, 4)
        conv_bias_4 = conv_bias       # shape (H)
        # Flatten weight to (H, 4) for easy access in Triton: weight[h, k]
        conv_weight_flat = conv_weight_4[:, 0, :].contiguous()  # (H, 4)
        conv_out = torch.empty((B, H, S), dtype=x.dtype, device=x.device)

        TritonGroupedCausalConvKernel[(B, triton.cdiv(H, BLOCK_H), triton.cdiv(S, BLOCK_S))](
            Bx_padded, conv_weight_flat, conv_bias_4, conv_out,
            B, H, S, PAD,
            BLOCK_S=BLOCK_S, BLOCK_H=BLOCK_H
        )

        # 5) Output gating: y = C * conv_out -> shape (B, H, S)
        y = torch.empty((B, H, S), dtype=x.dtype, device=x.device)
        grid_gate2 = (B, triton.cdiv(S, BLOCK_S), triton.cdiv(H, BLOCK_H))
        TritonGateKernel[grid_gate2](
            C_out, conv_out, y,
            B, S, H,
            BLOCK_S=BLOCK_S, BLOCK_H=BLOCK_H
        )

        # 6) Final output projection: F.linear(y, out_proj_weight, out_proj_bias) -> (B, S, H)
        # Implement with TritonFinalProjectionKernel:
        # y: (B, H, S), out_proj_weight: (H, H), out_proj_bias: (H)
        # We need output (B, S, H), i.e., per element output[B, S, H] = sum_n y[B, n, S] * out_proj_weight[n, H] + bias[H]
        # For clarity, y is (B, H, S); out_proj_weight is (H, H). We will compute per element.
        output = torch.empty((B, S, H), dtype=x.dtype, device=x.device)
        TritonFinalProjectionKernel[(B, triton.cdiv(S, BLOCK_S), triton.cdiv(H, BLOCK_H))](
            y, out_proj_weight, out_proj_bias, output,
            B, S, H,
            BLOCK_S=BLOCK_S, BLOCK_H=BLOCK_H
        )

        return output


# For completeness, keep the same run function signature as the original for evaluation harness.
@torch.no_grad()
def run(
    x: torch.Tensor,
    in_proj_weight: torch.Tensor,
    in_proj_bias: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    out_proj_weight: torch.Tensor,
    out_proj_bias: torch.Tensor,
):
    return ModelNew()(x, in_proj_weight, in_proj_bias, conv_weight, conv_bias, out_proj_weight, out_proj_bias)


# Example get_inputs() function if needed:
def get_inputs():
    # Placeholder: evaluator provides its own inputs
    batch_size = 1
    seq_len = 1024
    hidden_size = 128
    x = torch.randn(batch_size, seq_len, hidden_size, device='cuda', dtype=torch.float32)
    # Weights/biases: shapes as per original spec
    # in_proj_weight: (3*hidden_size, hidden_size) = (384, 128)
    # in_proj_bias: (3*hidden_size) = (384)
    # conv_weight: (hidden_size, 1, 4) = (128, 1, 4)
    # conv_bias: (hidden_size) = (128)
    # out_proj_weight: (hidden_size, hidden_size) = (128, 128)
    # out_proj_bias: (hidden_size) = (128)
    in_proj_weight = torch.randn(384, 128, device='cuda', dtype=torch.float32)
    in_proj_bias = torch.randn(384, device='cuda', dtype=torch.float32)
    conv_weight = torch.randn(128, 1, 4, device='cuda', dtype=torch.float32)
    conv_bias = torch.randn(128, device='cuda', dtype=torch.float32)
    out_proj_weight = torch.randn(128, 128, device='cuda', dtype=torch.float32)
    out_proj_bias = torch.randn(128, device='cuda', dtype=torch.float32)
    return x, in_proj_weight, in_proj_bias, conv_weight, conv_bias, out_proj_weight, out_proj_bias


def run(*args):
    return ModelNew()(*args)
