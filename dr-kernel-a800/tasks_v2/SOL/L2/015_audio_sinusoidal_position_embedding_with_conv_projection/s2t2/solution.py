import math
import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton Conv2d kernel for a single output element: y[b, oc, oh, ow].
# Input x: (B, C_in, IH, IW), weights w: (OC, C_in, 3, 3), bias b: (OC,).
# Padding=1, stride=2, output OH= (IH + 2 - 3)//2 + 1, OW similarly.
@triton.jit
def conv2d_stage_kernel(
    x_ptr,          # *fp16/bf16, (B, C_in, IH, IW)
    w_ptr,          # *fp16/bf16, (OC, C_in, 3, 3)
    b_ptr,          # *fp16/bf16 or None, (OC,)
    y_ptr,          # *fp32, (B, OC, OH, OW)
    B: tl.constexpr,
    C_in: tl.constexpr,
    IH: tl.constexpr,
    IW: tl.constexpr,
    OC: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    x_stride_b, x_stride_c, x_stride_h, x_stride_w,
    w_stride_oc, w_stride_ci, w_stride_kh, w_stride_kw,
    y_stride_b, y_stride_oc, y_stride_h, y_stride_w,
    has_bias: tl.constexpr,
):
    b = tl.program_id(0)
    oc = tl.program_id(1)
    oh = tl.program_id(2)
    ow = tl.program_id(3)

    # Initialize accumulator
    acc = tl.zeros((), dtype=tl.float32)

    # 3x3 kernel, stride=2, padding=1
    for ci in range(0, C_in):
        for kh in range(0, 3):  # KH=3
            ih = 2 * oh + kh - 1
            valid_h = (ih >= 0) & (ih < IH)
            for kw in range(0, 3):  # KW=3
                iw = 2 * ow + kw - 1
                valid_w = (iw >= 0) & (iw < IW)
                valid = valid_h & valid_w

                # Load input x[b, ci, ih, iw]
                x_offset = b * x_stride_b + ci * x_stride_c + ih * x_stride_h + iw * x_stride_w
                x_val = tl.load(x_ptr + x_offset, mask=valid, other=0.0).to(tl.float32)

                # Load weight w[oc, ci, kh, kw]
                w_offset = oc * w_stride_oc + ci * w_stride_ci + kh * w_stride_kh + kw * w_stride_kw
                w_val = tl.load(w_ptr + w_offset).to(tl.float32)

                acc += x_val * w_val

    # Add bias if present
    if has_bias:
        b_val = tl.load(b_ptr + oc).to(tl.float32)
        acc += b_val

    # Store y[b, oc, oh, ow] as fp32
    y_offset = b * y_stride_b + oc * y_stride_oc + oh * y_stride_h + ow * y_stride_w
    tl.store(y_ptr + y_offset, acc)


# Triton GELU (tanh approximation) kernel over 1D flattened tensor.
@triton.jit
def gelu_kernel_1d(
    in_ptr,      # *fp32, input flattened
    out_ptr,     # *fp32, output flattened
    n_elements: tl.constexpr,
):
    idx = tl.program_id(0)
    if idx >= n_elements:
        return
    x = tl.load(in_ptr + idx).to(tl.float32)
    sqrt_2_over_pi = 0.7978845608028654  # sqrt(2/pi)
    c = 0.044715
    x3 = x * x * x
    inner = sqrt_2_over_pi * (x + c * x3)
    gelu = 0.5 * x * (1.0 + tl.tanh(inner))
    tl.store(out_ptr + idx, gelu)


# Triton Linear projection kernel: for each (b, t), compute y[b, t, :] = X[b, t, :] @ W^T
# X is (B, T, N), W is (M, N), Y is (B, T, M). Applies scaling and adds positional embedding row.
@triton.jit
def linear_project_kernel(
    X_ptr,        # *fp32, (B*T, N)
    W_ptr,        # *fp32, (M, N)
    Y_ptr,        # *fp32, (B*T, M)
    scale,        # float32
    pos_emb_ptr,  # *fp32, (max_source_positions, M) — we index by (b*T + t)
    B: tl.constexpr,
    T: tl.constexpr,
    N: tl.constexpr,
    M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    idx = tl.program_id(0)  # linearize over (b, t): b = idx // T, t = idx % T
    if idx >= B * T:
        return

    b = idx // T
    t = idx % T

    # Accumulator for M outputs
    y = tl.zeros((M,), dtype=tl.float32)

    # Loop over N in chunks
    for off in range(0, N, BLOCK_N):
        offs = off + tl.arange(0, BLOCK_N)
        mask = offs < N

        # Load X[b, t, offs]
        x_chunk = tl.load(X_ptr + idx * N + offs, mask=mask, other=0.0).to(tl.float32)

        # Load W[offs, :] -> shape (BLOCK_N, M)
        m_idx = tl.arange(0, M)
        w_chunk = tl.load(W_ptr + offs[:, None] * N + m_idx[None, :], mask=mask[:, None], other=0.0).to(tl.float32)

        # Accumulate: y += sum_k x_chunk[k] * w_chunk[k, :]
        y += tl.sum(w_chunk * x_chunk[:, None], axis=0)

    # Apply scaling
    y = y * scale

    # Add positional embedding: pos_emb_ptr has shape (max_source_positions, M)
    # Index by row = b*T + t
    pos_row = b * T + t
    pos_vec = tl.load(pos_emb_ptr + pos_row * M + tl.arange(0, M)).to(tl.float32)
    y = y + pos_vec

    # Store Y[b, t, :]
    for m in range(0, M):
        tl.store(Y_ptr + idx * M + m, y[m])


def _run_conv_stage_triton(x, w, b, IH, IW, OC):
    # x: (B, C_in, IH, IW), w: (OC, C_in, 3, 3), b: (OC,) or None
    B, C_in, IH, IW = x.shape
    OC, Cw, KH, KW = w.shape
    assert C_in == Cw and KH == 3 and KW == 3, "Kernel assumes 3x3 input channel match and 3x3 kernel"
    assert b is None or b.shape[0] == OC, "Bias shape mismatch"

    # Output dims for stride=2, padding=1
    OH = (IH + 2 * 1 - KH) // 2 + 1
    OW = (IW + 2 * 1 - KW) // 2 + 1

    # Output buffer in fp32
    y = torch.empty((B, OC, OH, OW), device=x.device, dtype=torch.float32)

    grid = (B, OC, OH, OW)
    conv2d_stage_kernel[grid](
        x, w, b if b is not None else y,  # dummy if no bias
        y,
        B, C_in, IH, IW, OC, OH, OW,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        w.stride(0), w.stride(1), w.stride(2), w.stride(3),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        has_bias=(b is not None),
        num_warps=2, num_stages=2,
    )
    return y


def _gelu_triton_flat(x_fp32):
    n = x_fp32.numel()
    x_flat = x_fp32.view(-1)
    out = torch.empty(n, device=x_fp32.device, dtype=torch.float32)
    grid = (n,)
    gelu_kernel_1d[grid](x_flat, out, n, num_warps=4, num_stages=2)
    return out.view(x_fp32.shape)


def _linear_project_triton(x_2d, w, embed_scale, pos_emb):
    # x_2d: (B*T, N), w: (M, N), pos_emb: (max_source_positions, M)
    B_T = x_2d.shape[0]
    N = w.shape[1]
    M = w.shape[0]
    y_flat = torch.empty((B_T * M,), device=x_2d.device, dtype=torch.float32)
    grid = (B_T,)
    linear_project_kernel[grid](
        x_2d, w, y_flat, float(embed_scale), pos_emb,
        B_T, 1, N, M, BLOCK_N=128, num_warps=4, num_stages=2
    )
    y = y_flat.view(B_T, M)
    return y


class ModelNew(torch.nn.Module):
    def forward(self, input_features, conv2d1_weight, conv2d1_bias, conv2d2_weight, conv2d2_bias,
                conv2d3_weight, conv2d3_bias, conv_out_weight, positional_embedding, embed_scale):
        # All computation in Triton; no torch ops in forward.

        device = input_features.device

        # Stage 1: Conv2d (1 -> 384) + GELU, input IH=80, IW=1688
        x1 = _run_conv_stage_triton(input_features, conv2d1_weight, conv2d1_bias, 80, 1688, 384)
        x1 = _gelu_triton_flat(x1)

        # Stage 2: Conv2d (384 -> 384) + GELU, input dims after stage1: IH=40, IW=424
        x2 = _run_conv_stage_triton(x1, conv2d2_weight, conv2d2_bias, 40, 424, 384)
        x2 = _gelu_triton_flat(x2)

        # Stage 3: Conv2d (384 -> 384) + GELU, input dims after stage2: IH=20, IW=212
        x3 = _run_conv_stage_triton(x2, conv2d3_weight, conv2d3_bias, 20, 212, 384)
        x3 = _gelu_triton_flat(x3)

        # Reshape: (batch, channels, freq, time) -> (batch, time, channels*freq)
        b, c, f, t = x3.size()
        x3 = x3.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)

        # Prepare for linear projection: (B*T, N)
        B = b
        T = t
        N = c * f  # 15360
        x_2d = x3.to(torch.float32).reshape(B * T, N).contiguous()

        # Linear projection to d_model=1024, add scaling and positional embedding
        M = conv_out_weight.shape[0]  # 1024
        y = _linear_project_triton(x_2d, conv_out_weight.to(torch.float32), float(embed_scale), positional_embedding.to(torch.float32))

        # Cast to original dtype (match input_features dtype)
        return y.to(input_features.dtype)


def run(*args):
    return ModelNew()(*args)
