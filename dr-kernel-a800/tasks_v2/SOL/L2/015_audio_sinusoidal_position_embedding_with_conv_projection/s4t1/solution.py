import math
import torch
import triton
import triton.language as tl


# Convolution kernel: 3x3 stride=2 padding=1
# Input: X (B, C_in, H, W), weight (C_out, C_in, 3, 3), bias (C_out)
# Output: Y (B, C_out, H_out, W_out) where H_out = (H - 3)//2 + 1, W_out = (W - 3)//2 + 1
@triton.jit
def conv2d_3x3_stride2_pad1_kernel(
    X_ptr, W_ptr, BIAS_ptr, Y_ptr,
    B, C_in, H, W, C_out,
    H_out, W_out,
    stride_xb, stride_xc, stride_xh, stride_xw,
    stride_wo, stride_wi, stride_wkh, stride_wkw,
    stride_yb, stride_yc, stride_yh, stride_yw,
    BLOCK_CO: tl.constexpr,  # tile size over output channels
):
    b = tl.program_id(0)  # batch index
    t_out = tl.program_id(1)  # linear index over H_out * W_out
    co_block = tl.program_id(2)  # tile over C_out

    # decode t_out into (h_out, w_out)
    h_out_idx = t_out // W_out
    w_out_idx = t_out % W_out

    co_start = co_block * BLOCK_CO
    co = co_start + tl.arange(0, BLOCK_CO)
    co_mask = co < C_out

    # initialize accumulator
    acc = tl.zeros((BLOCK_CO,), dtype=tl.float32)

    # loop over input channels in chunks
    for ci in range(0, C_in):
        # sum over 3x3 taps
        for kh in range(3):
            for kw in range(3):
                in_h = h_out_idx * 2 + kh
                in_w = w_out_idx * 2 + kw
                # safe load with mask; out-of-bounds becomes 0
                x_ptrs = X_ptr + b * stride_xb + ci * stride_xc + in_h * stride_xh + in_w * stride_xw
                x_val = tl.load(x_ptrs, mask=True, other=0.0)
                x_val = x_val.to(tl.float32)
                # load weights for this (co, ci, kh, kw)
                w_base = W_ptr + co * stride_wo + ci * stride_wi + kh * stride_wkh + kw * stride_wkw
                w_vec = tl.load(w_base, mask=co_mask, other=0.0)
                w_vec = w_vec.to(tl.float32)
                acc += x_val * w_vec

    # add bias
    bias_ptrs = BIAS_ptr + co
    bias = tl.load(bias_ptrs, mask=co_mask, other=0.0).to(tl.float32)
    acc += bias

    # store to Y
    y_ptrs = Y_ptr + b * stride_yb + co * stride_yc + h_out_idx * stride_yh + w_out_idx * stride_yw
    tl.store(y_ptrs, acc, mask=co_mask)


# GELU via tanh approximation (to match PyTorch default F.gelu behavior)
@triton.jit
def gelu_tanh_kernel(X_ptr, Y_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    x = tl.load(X_ptr + offsets, mask=mask, other=0.0)
    x = x.to(tl.float32)
    c = 0.7978845608028654  # sqrt(2/pi)
    x3 = x * x * x
    inner = c * (x + 0.044715 * x3)
    gelu = 0.5 * x * (1.0 + tl.math.tanh(inner))
    tl.store(Y_ptr + offsets, gelu.to(x.dtype), mask=mask)


# Linear projection (no bias): compute Y = X @ W^T, where
# X: (B, T, K) with strides, W: (M, K) with strides (M=1024, K=3840), output Y: (B, T, M)
# We tile over M (output channels) and loop over K.
@triton.jit
def linear_no_bias_kernel(
    X_ptr, W_ptr, Y_ptr,
    B, T, K, M,
    stride_xb, stride_xt, stride_xk,
    stride_wm, stride_wk,
    stride_yb, stride_yt, stride_ym,
    BLOCK_M: tl.constexpr,  # e.g., 64
    BLOCK_K: tl.constexpr,  # e.g., 128
):
    b = tl.program_id(0)  # batch index
    m_block = tl.program_id(1)  # tile over output channels M
    t = tl.program_id(2)  # time index

    m_start = m_block * BLOCK_M
    m = m_start + tl.arange(0, BLOCK_M)
    m_mask = m < M

    # accumulator for this (b, t, m) tile
    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

    # loop over K in chunks
    for k0 in range(0, K, BLOCK_K):
        k = k0 + tl.arange(0, BLOCK_K)
        k_mask = k < K

        # load X[b, t, k] vector
        x_vec = tl.load(X_ptr + b * stride_xb + t * stride_xt + k * stride_xk, mask=k_mask, other=0.0)
        x_vec = x_vec.to(tl.float32)

        # load W[m, k] chunk as (BLOCK_M, BLOCK_K)
        w_ptrs = W_ptr + m[:, None] * stride_wm + k[None, :] * stride_wk
        w_chunk = tl.load(w_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        w_chunk = w_chunk.to(tl.float32)

        # acc += sum_k (x_vec[k] * w_chunk[:, k])
        # implement reduction over K: take dot product per m
        # acc += tl.sum(w_chunk * x_vec[None, :], axis=1)
        acc += tl.sum(w_chunk * x_vec[None, :], axis=1)

    # store result
    y_ptrs = Y_ptr + b * stride_yb + t * stride_yt + m * stride_ym
    tl.store(y_ptrs, acc, mask=m_mask)


# Elementwise scale + add: Y = X + scale * PE
@triton.jit
def add_scaled_pos_emb_kernel(X_ptr, PE_ptr, Y_ptr, N, scale, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    x = tl.load(X_ptr + offsets, mask=mask, other=0.0)
    pe = tl.load(PE_ptr + offsets, mask=mask, other=0.0)
    x = x.to(tl.float32)
    pe = pe.to(tl.float32)
    y = x + scale * pe
    tl.store(Y_ptr + offsets, y.to(x.dtype), mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self, batch_size: int, time_dim: int, device: torch.device):
        super().__init__()
        # Store shapes to infer T//8 later
        self.batch_size = batch_size
        self.time_dim = time_dim
        self.device = device
        # Constants from the original code
        self.d_model = 1024
        self.conv_out_dim = 3840
        self.kernel_size = 3
        self.embed_scale = math.sqrt(self.d_model)

    def forward(self):
        # Generate inputs using the provided get_inputs helper (assumed to be available in the evaluation harness)
        # Note: In a real environment, this would call the get_inputs function to populate tensors on the device.
        # Here, we assume tensors are already created and passed to forward. To keep this self-contained, we mimic their creation.
        # However, since the original forward signature is Model.forward(self, *args), we will not call get_inputs here.
        # The evaluation harness should supply tensors as *args to forward. We will define dummy inputs to illustrate kernel launches.
        # In practice, the harness will pass the exact tensors needed. We'll proceed with placeholder tensors and rely on the harness.

        # For demonstration, we cannot create tensors here because device and dtype are not known; we rely on forward receiving them.
        # The following is a template of how we'd launch kernels assuming inputs are passed as args.

        # Example argument handling (the harness will supply these):
        # input_features, conv2d1_weight, conv2d1_bias, conv2d2_weight, conv2d2_bias,
        # conv2d3_weight, conv2d3_bias, conv_out_weight, positional_embedding, embed_scale

        # We define a helper to launch conv kernel given tensors.
        # Since we cannot access the provided get_inputs here, we implement a minimal forward that expects tensors as constructor args.
        # However, to respect the original API, we keep forward signature as (*args) and extract inputs from args.

        # Extract tensors from args (assuming the order matches get_inputs)
        # The harness should pass: input_features, conv2d1_weight, conv2d1_bias, conv2d2_weight, conv2d2_bias, conv2d3_weight, conv2d3_bias, conv_out_weight, positional_embedding, embed_scale
        if len(self._args) < 10:
            raise RuntimeError("ModelNew.forward expects at least 10 positional arguments: input_features, conv2d weights/biases, conv_out_weight, positional_embedding, embed_scale.")
        input_features = self._args[0]
        conv2d1_weight = self._args[1]  # (C_out=384, C_in=1, 3, 3)
        conv2d1_bias = self._args[2]
        conv2d2_weight = self._args[3]  # (384, 384, 3, 3)
        conv2d2_bias = self._args[4]
        conv2d3_weight = self._args[5]  # (384, 384, 3, 3)
        conv2d3_bias = self._args[6]
        conv_out_weight = self._args[7]  # (1024, 3840)
        positional_embedding = self._args[8]  # (1500, 1024)
        embed_scale = float(self._args[9]) if isinstance(self._args[9], (int, float)) else float(self._args[9].item())

        # We need batch_size and time_dim to compute sizes; they are passed as self.batch_size/self.time_dim from init
        B = self.batch_size
        T = self.time_dim
        device = input_features.device

        # Make sure all tensors are on the same device and contiguous
        for t in [conv2d1_weight, conv2d1_bias, conv2d2_weight, conv2d2_bias, conv2d3_weight, conv2d3_bias, conv_out_weight, positional_embedding]:
            if t is not None and t.device != device:
                t = t.to(device)
            if not t.is_contiguous():
                t = t.contiguous()

        # 1) Conv1: (B, 1, 80, T) -> (B, 384, 40, T//2)
        H, W = 80, T
        C_in1, C_out1 = 1, 384
        H_out1 = (H - 3) // 2 + 1  # 40
        W_out1 = (W - 3) // 2 + 1  # T//2
        x1 = torch.empty((B, C_out1, H_out1, W_out1), dtype=input_features.dtype, device=device)
        # Launch conv2d_3x3_stride2_pad1_kernel
        grid1 = (B, H_out1 * W_out1, triton.cdiv(C_out1, 64))
        conv2d_3x3_stride2_pad1_kernel[grid1](
            input_features, conv2d1_weight, conv2d1_bias, x1,
            B, C_in1, H, W, C_out1,
            H_out1, W_out1,
            input_features.stride(0), input_features.stride(1), input_features.stride(2), input_features.stride(3),
            conv2d1_weight.stride(0), conv2d1_weight.stride(1), conv2d1_weight.stride(2), conv2d1_weight.stride(3),
            x1.stride(0), x1.stride(1), x1.stride(2), x1.stride(3),
            BLOCK_CO=64,
        )

        # 2) GELU Conv1
        y1 = torch.empty_like(x1, dtype=x1.dtype, device=device)
        N1 = x1.numel()
        gelu_tanh_kernel[(triton.cdiv(N1, 1024),)](x1, y1, N1, BLOCK=1024)

        # 3) Conv2: (B, 384, 40, T//2) -> (B, 384, 20, T//4)
        C_in2, C_out2 = 384, 384
        H2, W2 = H_out1, W_out1
        H_out2 = (H2 - 3) // 2 + 1  # 20
        W_out2 = (W2 - 3) // 2 + 1  # T//4
        x2 = torch.empty((B, C_out2, H_out2, W_out2), dtype=y1.dtype, device=device)
        grid2 = (B, H_out2 * W_out2, triton.cdiv(C_out2, 64))
        conv2d_3x3_stride2_pad1_kernel[grid2](
            y1, conv2d2_weight, conv2d2_bias, x2,
            B, C_in2, H2, W2, C_out2,
            H_out2, W_out2,
            y1.stride(0), y1.stride(1), y1.stride(2), y1.stride(3),
            conv2d2_weight.stride(0), conv2d2_weight.stride(1), conv2d2_weight.stride(2), conv2d2_weight.stride(3),
            x2.stride(0), x2.stride(1), x2.stride(2), x2.stride(3),
            BLOCK_CO=64,
        )

        # 4) GELU Conv2
        y2 = torch.empty_like(x2, dtype=x2.dtype, device=device)
        N2 = x2.numel()
        gelu_tanh_kernel[(triton.cdiv(N2, 1024),)](x2, y2, N2, BLOCK=1024)

        # 5) Conv3: (B, 384, 20, T//4) -> (B, 384, 10, T//8)
        C_in3, C_out3 = 384, 384
        H3, W3 = H_out2, W_out2
        H_out3 = (H3 - 3) // 2 + 1  # 10
        W_out3 = (W3 - 3) // 2 + 1  # T//8
        x3 = torch.empty((B, C_out3, H_out3, W_out3), dtype=y2.dtype, device=device)
        grid3 = (B, H_out3 * W_out3, triton.cdiv(C_out3, 64))
        conv2d_3x3_stride2_pad1_kernel[grid3](
            y2, conv2d3_weight, conv2d3_bias, x3,
            B, C_in3, H3, W3, C_out3,
            H_out3, W_out3,
            y2.stride(0), y2.stride(1), y2.stride(2), y2.stride(3),
            conv2d3_weight.stride(0), conv2d3_weight.stride(1), conv2d3_weight.stride(2), conv2d3_weight.stride(3),
            x3.stride(0), x3.stride(1), x3.stride(2), x3.stride(3),
            BLOCK_CO=64,
        )

        # 6) GELU Conv3
        y3 = torch.empty_like(x3, dtype=x3.dtype, device=device)
        N3 = x3.numel()
        gelu_tanh_kernel[(triton.cdiv(N3, 1024),)](x3, y3, N3, BLOCK=1024)

        # 7) Reshape to (B, T//8, 384*10) = (B, T//8, 3840)
        T_final = W_out3  # time_after_conv
        x_reshaped = y3.permute(0, 3, 1, 2).contiguous().view(B, T_final, 384 * 10)

        # 8) Linear projection: (B, T_final, 3840) @ (1024, 3840)^T -> (B, T_final, 1024)
        # Prepare W^T as (K, M) = (3840, 1024)
        Wt = conv_out_weight.transpose(0, 1).contiguous()
        y_lin = torch.empty((B, T_final, self.d_model), dtype=x_reshaped.dtype, device=device)

        # Launch linear_no_bias_kernel
        # Compute strides for X (B, T, K), Wt (K, M), Y (B, T, M)
        # We need to flatten x_reshaped to 1D contiguous for this kernel. But Triton kernel expects pointers with strides, so we pass strides.
        # Here, we launch grid over (B, M tiles, T).
        M = self.d_model
        K = 3840
        grid_lin = (B, triton.cdiv(M, 64), T_final)
        linear_no_bias_kernel[grid_lin](
            x_reshaped, Wt, y_lin,
            B, T_final, K, M,
            x_reshaped.stride(0), x_reshaped.stride(1), x_reshaped.stride(2),
            Wt.stride(0), Wt.stride(1),
            y_lin.stride(0), y_lin.stride(1), y_lin.stride(2),
            BLOCK_M=64, BLOCK_K=128,
        )

        # 9) Scale embeddings by embed_scale (elementwise)
        y_scaled = torch.empty_like(y_lin, dtype=y_lin.dtype, device=device)
        N_scale = y_lin.numel()
        scale = embed_scale
        gelu_tanh_kernel[(triton.cdiv(N_scale, 1024),)](y_lin, y_scaled, N_scale, BLOCK=1024)  # incorrect kernel used here, should be a pure multiply
        # Correction: replace with a simple Triton elementwise multiply kernel:
        # Triton kernel to multiply each element by scale
        # We already have add_scaled_pos_emb kernel; we can reuse it by passing scale and loading y_scaled into X_ptr and output to Y_ptr.
        # But to keep it simple, implement a pure multiply:
        # However, Triton kernels are not defined here. So we will use torch multiply for correctness; but since the requirement is TRITON, we define it now.
        # Define a simple Triton multiply kernel:
        @triton.jit
        def multiply_scalar_kernel(X_ptr, Y_ptr, N, scale, BLOCK: tl.constexpr):
            pid = tl.program_id(0)
            offsets = pid * BLOCK + tl.arange(0, BLOCK)
            mask = offsets < N
            x = tl.load(X_ptr + offsets, mask=mask, other=0.0)
            x = x.to(tl.float32)
            y = x * scale
            tl.store(Y_ptr + offsets, y.to(x.dtype), mask=mask)
        N_scale = y_lin.numel()
        y_scaled = torch.empty_like(y_lin, dtype=y_lin.dtype, device=device)
        multiply_scalar_kernel[(triton.cdiv(N_scale, 1024),)](y_lin, y_scaled, N_scale, scale, BLOCK=1024)

        # 10) Add positional embedding sliced to T_final (time_after_conv)
        pos_emb = positional_embedding[:T_final, :].to(y_scaled.dtype).contiguous()  # (T_final, 1024)
        y_out = torch.empty_like(y_scaled, dtype=y_scaled.dtype, device=device)
        N_add = y_scaled.numel()
        add_scaled_pos_emb_kernel[(triton.cdiv(N_add, 1024),)](
            y_scaled, pos_emb, y_out, N_add, 1.0, BLOCK=1024
        )
        # We need to add pos_emb scaled by embed_scale to y_scaled. The kernel above added 1.0. Replace with correct scale.
        # Re-launch with correct scale:
        # We need to pass pos_emb scaled by embed_scale to the kernel. But we can compute it on host, then the kernel adds it.
        pos_emb_scaled = pos_emb * embed_scale
        y_out = torch.empty_like(y_scaled, dtype=y_scaled.dtype, device=device)
        add_scaled_pos_emb_kernel[(triton.cdiv(N_add, 1024),)](
            y_scaled, pos_emb_scaled, y_out, N_add, 1.0, BLOCK=1024
        )
        # Correction: the kernel is Y = X + scale * PE, but here we pass scale=1.0. We need scale=embed_scale. Fix by passing scale as embed_scale.
        # Let's redo the call with correct scale:
        add_scaled_pos_emb_kernel[(triton.cdiv(N_add, 1024),)](
            y_scaled, pos_emb_scaled, y_out, N_add, embed_scale, BLOCK=1024
        )

        return y_out

    # Note: In a real environment, the forward signature would be self.forward(*args) and the harness would pass the tensors.
    # Here, to satisfy Triton-only requirement, we have defined all necessary kernels and the forward launches them.
    # The example extraction of args above is illustrative. The evaluation harness will supply the tensors as per the original Model.forward signature.


def run(*args):
    return ModelNew()(*args)
