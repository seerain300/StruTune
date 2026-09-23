import torch
import triton
import triton.language as tl


# 1) Linear projection via Triton: out[B, S, H] = x @ W.T + bias
# x: (B, S, H) contiguous; W: (H, H) contiguous (for each of the three groups)
# bias: (H)
@triton.jit
def TritonLinearKernel(
    X_ptr, W_ptr, Bias_ptr, Out_ptr,
    B, S, H,
    BLOCK_TOTAL: tl.constexpr,
):
    pid = tl.program_id(0)
    total = B * S * H
    start = pid * BLOCK_TOTAL
    offsets = start + tl.arange(0, BLOCK_TOTAL)
    mask = offsets < total

    # map linear offsets to (b, s, h)
    h = offsets % H
    tmp = offsets // H
    s = tmp % S
    b = tmp // S

    # load x: X[b, s, h]
    x_ptrs = X_ptr + b * (S * H) + s * H + h
    x_vals = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)

    # load w: W[h, :]
    w_ptrs = W_ptr + h * H + tl.arange(0, H)
    # create a vectorized pointer for each lane with H
    # tl.load expects a 1D pointer; we need a matrix acc[BLOCK_TOTAL, H]
    # Instead, we loop over H to accumulate
    acc = tl.zeros([BLOCK_TOTAL], dtype=tl.float32)
    for i in range(0, H):
        w_val = tl.load(W_ptr + i * H + h, mask=mask, other=0.0).to(tl.float32)
        acc += x_vals * w_val

    # add bias
    bias_vals = tl.load(Bias_ptr + h, mask=mask, other=0.0).to(tl.float32)
    acc += bias_vals

    # store to Out[b, s, h]
    out_ptrs = Out_ptr + b * (S * H) + s * H + h
    tl.store(out_ptrs, acc, mask=mask)


# 2) Element-wise gating: Bx = B * x_proj
@triton.jit
def TritonGateKernel(
    B_ptr, X_ptr, OUT_ptr,
    B, S, H,
    BLOCK_TOTAL: tl.constexpr,
):
    pid = tl.program_id(0)
    total = B * S * H
    start = pid * BLOCK_TOTAL
    offsets = start + tl.arange(0, BLOCK_TOTAL)
    mask = offsets < total

    h = offsets % H
    tmp = offsets // H
    s = tmp % S
    b = tmp // S

    b_ptrs = B_ptr + b * (S * H) + s * H + h
    x_ptrs = X_ptr + b * (S * H) + s * H + h
    b_vals = tl.load(b_ptrs, mask=mask, other=0.0).to(tl.float32)
    x_vals = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
    out_vals = b_vals * x_vals

    out_ptrs = OUT_ptr + b * (S * H) + s * H + h
    tl.store(out_ptrs, out_vals, mask=mask)


# 3) Left-pad along sequence by PAD=3 for Bx
@triton.jit
def TritonPadLeftKernel(
    Bx_ptr, out_ptr,
    B, H, S, PAD,
    BLOCK_TOTAL: tl.constexpr,
):
    pid = tl.program_id(0)
    total = B * H * (S + PAD)
    start = pid * BLOCK_TOTAL
    offsets = start + tl.arange(0, BLOCK_TOTAL)
    mask = offsets < total

    h = offsets % H
    tmp = offsets // H
    s_out = tmp % (S + PAD)
    b = tmp // (S + PAD)

    # write value from Bx if s_out >= PAD else 0
    bx_ptrs = Bx_ptr + b * (H * S) + h * S + tl.where(s_out >= PAD, s_out - PAD, 0)
    # create mask for valid s_out >= PAD
    mask_bx = mask & (s_out >= PAD)
    bx_vals = tl.load(bx_ptrs, mask=mask_bx, other=0.0).to(tl.float32)

    out_ptrs = out_ptr + b * (H * (S + PAD)) + h * (S + PAD) + s_out
    tl.store(out_ptrs, bx_vals, mask=mask)


# 4) Grouped causal 1D convolution: conv_out[b, h, s] = sum_{k=0..3} Bx_pad[b, h, s+k] * conv_weight[h,k] + conv_bias[h]
@triton.jit
def TritonCausalConvKernel(
    Bx_pad_ptr, conv_weight_ptr, conv_bias_ptr, out_ptr,
    B, H, S, PAD,  # PAD=3
    BLOCK_S: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_s = tl.program_id(2)

    b = pid_b
    h = pid_h
    s_start = pid_s * BLOCK_S
    s_offsets = s_start + tl.arange(0, BLOCK_S)
    mask_s = s_offsets < S

    # accumulate over k=0..3
    acc = tl.zeros([BLOCK_S], dtype=tl.float32)
    for k in range(0, 4):
        pos = s_offsets + k  # original seq positions
        ptrs = Bx_pad_ptr + b * (H * (S + PAD)) + h * (S + PAD) + pos
        val = tl.load(ptrs, mask=mask_s, other=0.0).to(tl.float32)
        w_ptr = conv_weight_ptr + h * 4 + k  # conv_weight[h, 0, k]
        w_val = tl.load(w_ptr).to(tl.float32)
        acc += val * w_val

    bias_val = tl.load(conv_bias_ptr + h).to(tl.float32)
    acc += bias_val

    out_ptrs = out_ptr + b * (H * S) + h * S + s_offsets
    tl.store(out_ptrs, acc, mask=mask_s)


# 5) Output gating: y = C * conv_out (broadcasting conv_out[B,H,S] to (B,S,H))
@triton.jit
def TritonGateFinalKernel(
    C_ptr, ConvOut_ptr, Y_ptr,
    B, S, H,
    BLOCK_TOTAL: tl.constexpr,
):
    pid = tl.program_id(0)
    total = B * S * H
    start = pid * BLOCK_TOTAL
    offsets = start + tl.arange(0, BLOCK_TOTAL)
    mask = offsets < total

    h = offsets % H
    tmp = offsets // H
    s = tmp % S
    b = tmp // S

    # conv_out index uses (b, h, s)
    conv_ptrs = ConvOut_ptr + b * (H * S) + h * S + s
    c_ptrs = C_ptr + b * (S * H) + s * H + h

    conv_vals = tl.load(conv_ptrs, mask=mask, other=0.0).to(tl.float32)
    c_vals = tl.load(c_ptrs, mask=mask, other=0.0).to(tl.float32)
    y_vals = c_vals * conv_vals

    y_ptrs = Y_ptr + b * (S * H) + s * H + h
    tl.store(y_ptrs, y_vals, mask=mask)


# 6) Final linear projection: Y = X_final @ W.T + Bias (here X_final = Y), shapes (B,S,H)
@triton.jit
def TritonLinearFinalKernel(
    X_final_ptr, W_ptr, Bias_ptr, Out_ptr,
    B, S, H,
    BLOCK_TOTAL: tl.constexpr,
):
    pid = tl.program_id(0)
    total = B * S * H
    start = pid * BLOCK_TOTAL
    offsets = start + tl.arange(0, BLOCK_TOTAL)
    mask = offsets < total

    h = offsets % H
    tmp = offsets // H
    s = tmp % S
    b = tmp // S

    x_ptrs = X_final_ptr + b * (S * H) + s * H + h
    x_vals = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)

    acc = tl.zeros([BLOCK_TOTAL], dtype=tl.float32)
    for i in range(0, H):
        w_val = tl.load(W_ptr + i * H + h, mask=mask, other=0.0).to(tl.float32)
        acc += x_vals * w_val

    bias_vals = tl.load(Bias_ptr + h, mask=mask, other=0.0).to(tl.float32)
    acc += bias_vals

    out_ptrs = Out_ptr + b * (S * H) + s * H + h
    tl.store(out_ptrs, acc, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, x, in_proj_weight, in_proj_bias, conv_weight, conv_bias, out_proj_weight, out_proj_bias):
        # Ensure dtype and contiguity for Triton
        x = x.contiguous().to(torch.float32)
        B, S, H = x.shape

        # 1) Three linear projections for B, C, x_proj
        B_out = torch.empty((B, S, H), device=x.device, dtype=torch.float32)
        C_out = torch.empty((B, S, H), device=x.device, dtype=torch.float32)
        x_proj_out = torch.empty((B, S, H), device=x.device, dtype=torch.float32)

        # Groups: [:H,:], [H:2H,:], [2H:3H,:]
        for W, Bias, Out in [
            (in_proj_weight[:H, :], in_proj_bias[:H], B_out),
            (in_proj_weight[H:2 * H, :], in_proj_bias[H:2 * H], C_out),
            (in_proj_weight[2 * H:3 * H, :], in_proj_bias[2 * H:3 * H], x_proj_out),
        ]:
            W = W.contiguous().to(torch.float32)
            Bias = Bias.contiguous().to(torch.float32)
            BLOCK = 1024
            grid = (triton.cdiv(B * S * H, BLOCK),)
            TritonLinearKernel[grid](x, W, Bias, Out, B, S, H, BLOCK_TOTAL=BLOCK)

        # 2) Element-wise gating: Bx = B * x_proj
        Bx = torch.empty((B, S, H), device=x.device, dtype=torch.float32)
        BLOCK = 1024
        grid = (triton.cdiv(B * S * H, BLOCK),)
        TritonGateKernel[grid](B_out, x_proj_out, Bx, B, S, H, BLOCK_TOTAL=BLOCK)

        # 3) Left-pad along sequence by PAD=3
        S_out = S + 3
        Bx_pad = torch.empty((B, H, S_out), device=x.device, dtype=torch.float32)
        BLOCK = 1024
        grid = (triton.cdiv(B * H * S_out, BLOCK),)
        TritonPadLeftKernel[grid](Bx, Bx_pad, B, H, S, 3, BLOCK_TOTAL=BLOCK)

        # 4) Grouped causal 1D convolution: conv_out[b, h, s] = sum_{k=0..3} Bx_pad[b, h, s+k] * conv_weight[h,k] + conv_bias[h]
        conv_weight = conv_weight.contiguous().to(torch.float32)  # shape (H,1,4)
        conv_bias = conv_bias.contiguous().to(torch.float32)      # shape (H)
        conv_out = torch.empty((B, H, S), device=x.device, dtype=torch.float32)
        BLOCK_S = 128
        grid = (B, H, triton.cdiv(S, BLOCK_S))
        TritonCausalConvKernel[grid](Bx_pad, conv_weight.view(-1), conv_bias, conv_out, B, H, S, 3, BLOCK_S=BLOCK_S)

        # 5) Output gating: y = C * conv_out (broadcast conv_out over H)
        y = torch.empty((B, S, H), device=x.device, dtype=torch.float32)
        BLOCK = 1024
        grid = (triton.cdiv(B * S * H, BLOCK),)
        TritonGateFinalKernel[grid](C_out, conv_out, y, B, S, H, BLOCK_TOTAL=BLOCK)

        # 6) Final linear projection
        out = torch.empty((B, S, H), device=x.device, dtype=torch.float32)
        out_proj_weight = out_proj_weight.contiguous().to(torch.float32)  # (H, H)
        out_proj_bias = out_proj_bias.contiguous().to(torch.float32)      # (H)
        BLOCK = 1024
        grid = (triton.cdiv(B * S * H, BLOCK),)
        TritonLinearFinalKernel[grid](y, out_proj_weight, out_proj_bias, out, B, S, H, BLOCK_TOTAL=BLOCK)

        return out


def run(*args):
    return ModelNew()(*args)
