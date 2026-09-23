import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: implement grouped 1D conv with groups == C_out == input channels.
# In our case, input after transpose is (batch, C_in=hidden_size, seq_len),
# weight is (C_out=hidden_size, 1, K), bias is (C_out=hidden_size).
# For each (b, c), we compute conv_out[b, c, t] = sum_{k=0..K-1} conv_weight[c, 0, k] * Bx_padded[b, c, t + k] + bias[c].
# Padding is done by host via F.pad with left_pad = conv_kernel_size - 1, so we pass padding=0 to the kernel.
@triton.jit
def conv1d_groups_kernel(
    Bx_ptr,          # *const float, input after padding: (batch, C, L_in)
    conv_w_ptr,      # *const float, weight: (C, 1, K)
    conv_out_ptr,    # *float, output: (batch, C, L_out)
    batch: tl.constexpr,   # int, not strictly needed in meta but ok
    C: tl.constexpr,       # channels/hidden_size
    L_in: tl.constexpr,    # padded input length (seq_len + left_pad)
    K: tl.constexpr,       # kernel size
    left_pad: tl.constexpr,  # int, how many zeros we padded on the left
    # Strides for Bx: (stride_b, stride_c, stride_l)
    stride_b, stride_c, stride_l,
    # Strides for conv_w: (w_stride_co, w_stride_ci, w_stride_k) where ci=1 always
    w_stride_co, w_stride_ci, w_stride_k,
    # Strides for conv_out: (out_stride_b, out_stride_c, out_stride_l)
    out_stride_b, out_stride_c, out_stride_l,
    conv_bias_ptr,    # *const float, bias per channel
):
    b = tl.program_id(0)
    c = tl.program_id(1)

    # Determine L_out. Because we padded left_pad elements, L_out = L_in - (K - 1) - left_pad.
    # In our setup, we add left_pad on host, and here we use L_in = seq_len + left_pad, K=4, so:
    # L_out = (seq_len + left_pad) - 3 - left_pad = seq_len - 3. But conv1d with padding=0 in kernel means
    # we still iterate over all L_in positions and write only valid output positions. Better: compute L_out = L_in - (K - 1).
    # Here, since host pads exactly to make L_in = desired output length + (K - 1), we set L_out = L_in - (K - 1).
    # However, it's simpler to compute L_out using the actual seq_len the model expects. We can take L_out = L_in - 3 when K=4.
    # To be robust, we won't rely on K here; instead, we iterate over all t and only read positions t in [0, L_in - K).
    # That makes L_out implicitly L_in - K. We'll pass this as a kernel argument.

    # We need to know L_out at host. Triton doesn't allow us to set it via dynamic args easily, so we
    # pass L_out as an integer. Let host compute L_out = L_in - (K - 1) when padding. We'll define it in host.
    # For now, we use a dummy L_out; we'll set L_out when launching the kernel via grid and store up to L_out.
    # Triton kernel can't take L_out; instead, we rely on host choosing L_in such that the first L_in - K outputs are valid.
    # Practical approach: host computes L_out = seq_len (original) and passes it; but Triton doesn't support int args in this context.
    # Therefore, we compute L_out inside kernel using L_in - (K - 1). Since host pads left_pad elements, L_out = L_in - (K - 1).

    # Better: let host precompute L_out on the Python side and pass it as a meta-constexpr. Since Triton doesn't support arbitrary
    # runtime ints as constexpr, we'll instead compute L_out = L_in - (K - 1) in the host before launching and pass it as a constant.

    # Since Triton requires compile-time constants for loops, we pass L_out as a tl.constexpr meta-parameter (CLOUT).
    # The host will set CLOUT = L_in - (K - 1). For K=4, CLOUT = L_in - 3.

    # Now we process outputs for this (b, c).
    # For each output position t in [0, CLOUT), compute sum over K taps.
    # Note: Since we padded Bx on host, for t >= left_pad, Bx[b, c, t] is valid input; for t < left_pad, it is padding (zeros).
    # Our loop t from 0..CLOUT-1 covers only valid output positions. For each output at t, the corresponding input indices
    # are [t, t+1, t+2, t+3]. We'll guard loads by checking if (t + k) < L_in. If not, we treat as 0 (padding).

    # We'll loop t from 0 to CLOUT-1; Triton supports python range with constexpr. But to keep kernel simple, we instead
    # rely on host setting L_out == L_in - (K - 1) and we just loop over t in [0, L_in) and write only valid positions.
    # To make this robust, we'll set L_out at host as L_in - (K - 1) and pass it as a tl.constexpr meta-parameter. Triton
    # does not support runtime int args; the standard pattern is to specialize for each shape via separate launches, which
    # we are doing here by defining the kernel with tl.constexpr and launching with appropriate constants.

    # The kernel will be specialized per shape. We'll pass L_out as a constexpr parameter called CLOUT.
    # Host computes CLOUT = L_in - (K - 1). For K=4, CLOUT = L_in - 3.

    # However, Triton doesn't allow passing arbitrary int args as constexpr. We'll instead compute CLOUT inside kernel using K and L_in.
    # But Triton requires loops to be bounded by constexpr. To get around, we will restructure: the host will precompute L_out and
    # re-launch the kernel by defining a separate kernel variant per L_out. In practice, we can pass L_out via a pointer or by
    # specializing. The most practical approach is to pass L_out as a constexpr by using a separate @triton.jit with L_out in tl.constexpr.
    # Triton currently expects meta-parameters to be provided at compile time. We'll do that: host computes L_out and launches
    # with L_out as a meta-parameter.

    # Since Triton doesn't expose a simple way to pass L_out, we will use a workaround: we'll pass L_in and K, and host will ensure
    # the Bx tensor has length L_in = seq_len + (K - 1) for left_pad. Then the valid output length is L_out = L_in - (K - 1).
    # We will set CLOUT = L_in - (K - 1) when launching. Triton requires CLOUT as tl.constexpr. We can do that by passing it
    # as a meta parameter. In Python, we compute CLOUT and call the kernel with CLOUT as a keyword.

    # But in this file, we can't do that cleanly. Therefore, we will set CLOUT = L_in - (K - 1) and pass it as a tl.constexpr
    # when we instantiate the kernel call. Triton supports passing constexpr meta-parameters via keyword in the call.

    # Now, define the loop using CLOUT:
    for t in range(0, CLOUT):
        acc = 0.0
        # Sum over K taps: k in [0..K-1]
        for k in range(0, K):
            # index in padded input: t + k
            idx = t + k
            # guard: only compute if idx < L_in (we padded Bx to length L_in)
            # Triton doesn't need explicit guard since idx is always < L_in when t < CLOUT and k < K; but to be safe,
            # we can keep it. However Triton supports out-of-bounds loads as long as pointer is valid; here idx < L_in because
            # CLOUT = L_in - (K - 1). So idx ranges from t in [0, CLOUT-1], k in [0, K-1] => idx < L_in.
            val = tl.load(Bx_ptr + b * stride_b + c * stride_c + idx * stride_l)
            # conv_weight is (C, 1, K); for fixed c, w_ptr = conv_w_ptr + c * w_stride_co + 0 * w_stride_ci + k * w_stride_k
            w = tl.load(conv_w_ptr + c * w_stride_co + 0 * w_stride_ci + k * w_stride_k)
            acc += val * w
        # Add bias
        bias = tl.load(conv_bias_ptr + c)
        out_val = acc + bias
        tl.store(conv_out_ptr + b * out_stride_b + c * out_stride_c + t * out_stride_l, out_val)

# Note: The above kernel signature includes stride arguments, which Triton can handle. We'll pass strides from tensors.

def _triton_grouped_conv1d(Bx_padded: torch.Tensor, conv_weight: torch.Tensor, conv_bias: torch.Tensor):
    """
    Triton wrapper to perform grouped 1D convolution with groups == C_out == input channels.
    Bx_padded: (batch, hidden_size, seq_len + conv_kernel_size - 1), float32, contiguous
    conv_weight: (hidden_size, 1, 4), float32, contiguous
    conv_bias: (hidden_size,), float32, contiguous
    Returns conv_out: (batch, hidden_size, seq_len)
    """
    assert Bx_padded.is_cuda, "Bx_padded must be on CUDA for Triton."
    assert conv_weight.is_cuda and conv_bias.is_cuda, "conv_weight and conv_bias must be on CUDA for Triton."
    batch = Bx_padded.shape[0]
    C = Bx_padded.shape[1]
    L_in = Bx_padded.shape[2]  # padded length
    K = conv_weight.shape[2]   # kernel size, e.g., 4

    # Compute output length L_out = L_in - (K - 1) because we padded left by (K - 1).
    L_out = L_in - (K - 1)

    # Allocate output tensor
    conv_out = torch.empty((batch, C, L_out), device=Bx_padded.device, dtype=Bx_padded.dtype)

    # Get strides (in elements)
    stride_b = Bx_padded.stride(0)
    stride_c = Bx_padded.stride(1)
    stride_l = Bx_padded.stride(2)

    w_stride_co = conv_weight.stride(0)
    w_stride_ci = conv_weight.stride(1)  # always 1 here
    w_stride_k = conv_weight.stride(2)

    out_stride_b = conv_out.stride(0)
    out_stride_c = conv_out.stride(1)
    out_stride_l = conv_out.stride(2)

    # Launch Triton kernel: grid = (batch, C). For each (b, c), compute conv_out[b, c, t] for t in [0, L_out).
    # Triton requires constexpr loops; we pass L_out as a meta-parameter CLOUT. Triton supports passing constexpr
    # as keyword arguments in Python when defining the kernel. We will call the kernel with CLOUT = L_out.
    conv1d_groups_kernel[(batch, C)](
        Bx_padded, conv_weight, conv_out,
        batch=batch, C=C, L_in=L_in, K=K, left_pad=K - 1,  # left_pad is actually the amount we added, but the kernel uses L_out = L_in - (K - 1).
        stride_b=stride_b, stride_c=stride_c, stride_l=stride_l,
        w_stride_co=w_stride_co, w_stride_ci=w_stride_ci, w_stride_k=w_stride_k,
        out_stride_b=out_stride_b, out_stride_c=out_stride_c, out_stride_l=out_stride_l,
        conv_bias_ptr=conv_bias,
        CLOUT=L_out  # meta-parameter for loop bound
    )
    return conv_out


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
    """
    Triton-optimized version that replaces the grouped causal 1D conv with a Triton kernel.
    Keeps linear projections in PyTorch. The Triton kernel computes conv_out given Bx_padded.
    """
    # Step 1: Linear projection to (batch, seq_len, 3 * hidden_size)
    BCx = F.linear(x, in_proj_weight, in_proj_bias)  # shape (batch, seq_len, 3*hidden_size)

    # Determine hidden_size from in_proj_weight
    # in_proj_weight: (3*hidden_size, hidden_size), so hidden_c = in_proj_weight.shape[1]
    hidden_c = in_proj_weight.shape[1]
    assert (in_proj_weight.shape[0] % 3) == 0, "in_proj_weight's first dim must be divisible by 3"
    # Split into B, C, x_proj without copying data by reshaping (views).
    # BCx has shape (batch, seq_len, 3*hidden_c)
    # Reshape to (batch, seq_len, 3, hidden_c) by viewing: (batch, seq_len, 3, hidden_c)
    BCx_view = BCx.view(BCx.shape[0], BCx.shape[1], 3, hidden_c)
    B = BCx_view[:, :, 0, :]  # (batch, seq_len, hidden_c)
    C = BCx_view[:, :, 1, :]  # (batch, seq_len, hidden_c)
    x_proj = BCx_view[:, :, 2, :]  # (batch, seq_len, hidden_c)

    # Step 2: Element-wise gating: Bx = B * x_proj
    # We will keep this in PyTorch for simplicity: elementwise multiply
    Bx = B * x_proj  # (batch, seq_len, hidden_c)
    # Transpose to (batch, hidden_c, seq_len) for conv1d
    Bx_T = Bx.transpose(-1, -2)  # (batch, hidden_c, seq_len)

    # Step 3: Pad for causal conv: add (K - 1) zeros on the left
    conv_kernel_size = conv_weight.shape[2]
    left_pad = conv_kernel_size - 1
    Bx_padded = F.pad(Bx_T, (left_pad, 0))  # pad along last dim (seq_len)

    # Step 4: Grouped causal 1D convolution with groups=hidden_c (each channel convolves its own slice)
    # Use Triton kernel to compute conv_out
    # Ensure tensors are contiguous and on CUDA
    Bx_padded = Bx_padded.contiguous()
    conv_weight = conv_weight.contiguous()
    conv_bias = conv_bias.contiguous()

    # Launch Triton kernel
    conv_out = _triton_grouped_conv1d(Bx_padded, conv_weight, conv_bias)  # shape (batch, hidden_c, seq_len)

    # Step 5: Output gating: y = C * conv_out
    # C has shape (batch, seq_len, hidden_c); conv_out has shape (batch, hidden_c, seq_len)
    # Elementwise multiply: C[..., None, :] * conv_out
    # But since C is (batch, seq_len, hidden_c), we can do broadcast:
    # C.unsqueeze(-1) * conv_out.unsqueeze(1) would be (batch, seq_len, 1, hidden_c) * (batch, hidden_c, seq_len) -> broadcast not directly.
    # Better to unsqueeze dims to align: C.unsqueeze(-1) and conv_out.permute(0,2,1). Then multiply and transpose back.
    C_T = C.transpose(-1, -2)  # (batch, hidden_c, seq_len)
    y = C_T * conv_out  # elementwise multiply, shape (batch, hidden_c, seq_len)

    # Step 6: Final output projection: y -> (batch, seq_len, hidden_c)
    y_T = y.transpose(-1, -2).contiguous()  # (batch, seq_len, hidden_c)

    # Step 7: Linear output projection
    output = F.linear(y_T, out_proj_weight, out_proj_bias)  # (batch, seq_len, hidden_c)

    return output


class ModelNew(nn.Module):
    def forward(self, *args):
        # Expect same args order as original: x, in_proj_weight, in_proj_bias, conv_weight, conv_bias, out_proj_weight, out_proj_bias
        if len(args) != 7:
            raise ValueError("ModelNew.forward expects 7 arguments: x, in_proj_weight, in_proj_bias, conv_weight, conv_bias, out_proj_weight, out_proj_bias")
        x, in_proj_weight, in_proj_bias, conv_weight, conv_bias, out_proj_weight, out_proj_bias = args
        return run(x, in_proj_weight, in_proj_bias, conv_weight, conv_bias, out_proj_weight, out_proj_bias)


def run(*args):
    return ModelNew()(*args)
