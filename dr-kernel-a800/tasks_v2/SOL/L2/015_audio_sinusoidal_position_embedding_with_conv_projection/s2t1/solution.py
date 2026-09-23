import math
import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Conv2d kernel: compute one output element y[b, oc, oh, ot] for given (b, oc, oh, ot).
# x shape: (B, C_in, IH, IW), w shape: (OC, C_in, KH, KW), padding=1, stride=2.
@triton.jit
def conv2d_kernel_single(
    x_ptr,          # *fp16/bf16, shape (B, C_in, IH, IW)
    w_ptr,          # *fp16/bf16, shape (OC, C_in, KH, KW)
    y_ptr,          # *fp16/bf16, shape (B, OC, OH, OW) — we store fp32
    B: tl.constexpr,
    C_in: tl.constexpr,
    IH: tl.constexpr,
    IW: tl.constexpr,
    OC: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    x_stride_b, x_stride_c, x_stride_h, x_stride_w,
    w_stride_oc, w_stride_ci, w_stride_kh, w_stride_kw,
    y_stride_b, y_stride_oc, y_stride_h, y_stride_w,
):
    b = tl.program_id(0)
    oc = tl.program_id(1)
    oh = tl.program_id(2)
    ot = tl.program_id(3)

    acc = tl.zeros((), dtype=tl.float32)

    # Loop over input channels and kernel
    for ci in range(0, C_in):
        for kh in range(0, KH):
            ih = 2 * oh + kh - 1
            valid_h = (ih >= 0) & (ih < IH)
            for kw in range(0, KW):
                it = 2 * ot + kw - 1
                valid_w = (it >= 0) & (it < IW)
                valid = valid_h & valid_w

                x_offset = b * x_stride_b + ci * x_stride_c + ih * x_stride_h + it * x_stride_w
                x_val = tl.load(x_ptr + x_offset, mask=valid, other=0.0).to(tl.float32)

                w_offset = oc * w_stride_oc + ci * w_stride_ci + kh * w_stride_kh + kw * w_stride_kw
                w_val = tl.load(w_ptr + w_offset).to(tl.float32)

                acc += x_val * w_val

    # Store to y[b, oc, oh, ot] as fp32
    y_offset = b * y_stride_b + oc * y_stride_oc + oh * y_stride_h + ot * y_stride_w
    tl.store(y_ptr + y_offset, acc)


# GELU activation (tanh approximation) over a flattened 1D array.
@triton.jit
def gelu_kernel_1d(
    in_ptr,      # *fp32, flattened input
    out_ptr,     # *fp32, flattened output
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


# Linear projection: for each (b, t), compute y[b, t, :] = X[b, t, :] @ W^T
# X is (B, T, N), W is (M, N), Y is (B, T, M). We add positional embedding in-kernel.
@triton.jit
def linear_project_kernel(
    X_ptr,        # *fp32, (B, T, N) flattened as (B*T, N)
    W_ptr,        # *fp32, (M, N)
    Y_ptr,        # *fp32, (B, T, M) flattened as (B*T, M)
    scale,        # float32
    pos_emb_ptr,  # *fp32, (max_source_positions, M)
    B: tl.constexpr,
    T: tl.constexpr,
    N: tl.constexpr,
    M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    b = tl.program_id(0)
    t = tl.program_id(1)
    if b >= B or t >= T:
        return

    y_vec = tl.zeros((M,), dtype=tl.float32)

    for off in range(0, N, BLOCK_N):
        offs = off + tl.arange(0, BLOCK_N)
        mask = offs < N

        # Load X[b, t, offs]
        x_chunk = tl.load(X_ptr + (b * T + t) * N + offs, mask=mask, other=0.0).to(tl.float32)

        # Load W[offs, :] -> shape (BLOCK_N, M)
        m_idx = tl.arange(0, M)
        w_chunk = tl.load(W_ptr + offs[:, None] * N + m_idx[None, :], mask=mask[:, None], other=0.0).to(tl.float32)

        # Accumulate y_vec += sum_k x_chunk[k] * w_chunk[k, :]
        y_vec += tl.sum(w_chunk * x_chunk[:, None], axis=0)

    # Apply scaling
    y_vec = y_vec * scale

    # Add positional embedding: pos_emb_ptr has shape (max_source_positions, M)
    # We index by row = b*T + t
    pos_row = b * T + t
    pos_vec = tl.load(pos_emb_ptr + pos_row * M + tl.arange(0, M)).to(tl.float32)
    y_vec = y_vec + pos_vec

    # Store Y[b, t, :]
    for m in range(0, M):
        tl.store(Y_ptr + (b * T + t) * M + m, y_vec[m])


def _run_conv_gelu_triton(x, w, out):
    # x: (B, C_in, IH, IW), w: (OC, C_in, KH, KW)
    B, C_in, IH, IW = x.shape
    OC, Cw, KH, KW = w.shape
    assert C_in == Cw and KH == 3 and KW == 3, "Kernel assumes 3x3 and matching C_in for conv1"

    OH = (IH + 2 * 1 - KH) // 2 + 1  # padding=1, stride=2
    OW = (IW + 2 * 1 - KW) // 2 + 1

    # Allocate output (fp32 for compute stability)
    y = torch.empty((B, OC, OH, OW), device=x.device, dtype=torch.float32)

    grid = (B, OC, OH, OW)
    conv2d_kernel_single[grid](
        x, w, y,
        B, C_in, IH, IW, OC, OH, OW, KH, KW,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        w.stride(0), w.stride(1), w.stride(2), w.stride(3),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        num_warps=2, num_stages=2
    )
    # GELU activation
    y = _gelu_triton_flat(y)
    return y


def _gelu_triton_flat(x):
    n = x.numel()
    x_flat = x.contiguous().view(-1)
    out = torch.empty(n, device=x.device, dtype=torch.float32)
    grid = (n,)
    gelu_kernel_1d[grid](x_flat, out, n, num_warps=4, num_stages=2)
    return out.view(x.shape)


def _linear_project_triton(x_2d, w, embed_scale, pos_emb):
    # x_2d: (B*T, N), w: (M, N), pos_emb: (max_source_positions, M)
    B = x_2d.shape[0] // w.shape[1] // 1  # inferred from shape
    # seq_len = B*T based on pos_emb rows
    seq_len = pos_emb.shape[0]
    T = seq_len // B
    M = w.shape[0]
    N = w.shape[1]

    y = torch.empty((B, T, M), device=x_2d.device, dtype=torch.float32)
    grid = (B, T)
    linear_project_kernel[grid](
        x_2d, w, y, float(embed_scale), pos_emb,
        B, T, N, M, BLOCK_N=128, num_warps=4, num_stages=2
    )
    return y


class ModelNew(torch.nn.Module):
    def forward(self, input_features, conv2d1_weight, conv2d1_bias, conv2d2_weight, conv2d2_bias,
                conv2d3_weight, conv2d3_bias, conv_out_weight, positional_embedding, embed_scale):
        # Ensure inputs are on device
        device = input_features.device

        # Stage 1: Conv2d (1 -> 384) + GELU
        x1 = _run_conv_gelu_triton(input_features, conv2d1_weight, None)  # conv2d1 has bias
        x1 = _gelu_triton_flat(x1)

        # Stage 2: Conv2d (384 -> 384) + GELU
        x2 = _run_conv_gelu_triton(x1, conv2d2_weight, None)  # conv2d2 has bias
        x2 = _gelu_triton_flat(x2)

        # Stage 3: Conv2d (384 -> 384) + GELU
        x3 = _run_conv_gelu_triton(x2, conv2d3_weight, None)  # conv2d3 has bias
        x3 = _gelu_triton_flat(x3)

        # Reshape: (batch, channels, freq, time) -> (batch, time, channels*freq)
        b, c, f, t = x3.size()
        x3 = x3.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)

        # Cast to fp32 for linear projection and compute
        x3_fp32 = x3.to(torch.float32)
        conv_out_weight_fp32 = conv_out_weight.to(torch.float32)
        pos_emb_fp32 = positional_embedding.to(torch.float32)

        # Flatten to (B*T, N) for linear projection
        B = b
        T = t
        N = c * f  # 15360
        x_2d = x3_fp32.reshape(B * T, N).contiguous()

        # Linear projection to d_model=1024
        M = conv_out_weight_fp32.shape[0]  # 1024
        y = _linear_project_triton(x_2d, conv_out_weight_fp32, float(embed_scale), pos_emb_fp32)

        # Cast back to input dtype for consistency
        return y.to(input_features.dtype)


def run(*args):
    return ModelNew()(*args)
