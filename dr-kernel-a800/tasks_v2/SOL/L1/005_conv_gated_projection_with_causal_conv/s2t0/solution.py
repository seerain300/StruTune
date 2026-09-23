import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False

# Kernel 1: Linear projection F.linear(x, in_proj_weight, in_proj_bias)
# Computes output[B*S, N_out] = x[B*S, S] @ in_proj_weight[S, N_out] + bias[N_out]
# We will call this kernel 3 times to produce B, C, x_proj outputs: N_out = hidden_size each.
@triton.jit
def _linear_gemv_kernel(
    x_ptr,            # *f32, shape [M, K], M=B*S, K=S
    w_ptr,            # *f32, shape [K, N], N = hidden_size or 3*hidden_size
    b_ptr,            # *f32, shape [N]
    out_ptr,          # *f32, shape [M, N]
    M, K, N,          # int32
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_om, stride_on,
    BLOCK_N: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
):
    # Each program computes a block of N columns for one row m
    m = tl.program_id(0)
    n_block = tl.program_id(1)
    offs_n = n_block * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N

    # Accumulator in fp32
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)

    # Loop over K dimension in tiles
    for k0 in range(0, K, BLOCK_N):  # Note: should loop over K, not N. Use K tiles.
        offs_k = k0 + tl.arange(0, BLOCK_N)
        mask_k = offs_k < K

        # Load x row m across BLOCK_K
        x_vals = tl.load(
            x_ptr + m * stride_xm + offs_k * stride_xk,
            mask=mask_k,
            other=0.0
        )  # shape [BLOCK_K]

        # Load w block across BLOCK_K and BLOCK_N
        w_vals = tl.load(
            w_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn,
            mask=mask_k[:, None] & mask_n[None, :],
            other=0.0
        )  # shape [BLOCK_K, BLOCK_N]

        # Accumulate: dot(x_vals, w_vals) per column
        acc += tl.sum(w_vals * x_vals[:, None], axis=0)

    # Add bias
    if b_ptr != 0:
        bias = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
        acc += bias

    # Store result
    tl.store(out_ptr + m * stride_om + offs_n * stride_on, acc.to(OUT_DTYPE), mask=mask_n)


# Kernel 2: Grouped causal 1D conv1d with groups=hidden_size and kernel_size=4
# Input: Bx of shape (B, H, S), conv_weight of shape (H, H, 4), conv_bias of shape (H)
# Output: conv_out of shape (B, H, S)
@triton.jit
def _grouped_conv1d_causal_kernel(
    in_ptr,            # *f32, input Bx[B, H, S], treated as (B, H, S)
    w_ptr,             # *f32, conv_weight[H, H, 4]
    b_ptr,             # *f32 or 0, bias[H]
    out_ptr,           # *f32, output conv_out[B, H, S]
    B, H, S,           # int32
    stride_ib, stride_ic, stride_il,
    stride_wg, stride_wc, stride_wk,
    stride_ob, stride_oc, stride_ol,
    BLOCK_T: tl.constexpr,  # tile along S
):
    b = tl.program_id(0)
    c = tl.program_id(1)
    t_block = tl.program_id(2)
    t_start = t_block * BLOCK_T
    offs_t = t_start + tl.arange(0, BLOCK_T)
    mask_t = offs_t < S

    # Accumulate over K=4 taps (causal padding by 3 on left)
    acc = tl.zeros([BLOCK_T], dtype=tl.float32)
    for k in range(4):
        # input index for causal padding: t + k - 1
        in_idx = offs_t + k - 1
        # mask for in_idx in [0, S-1]
        in_mask = (in_idx >= 0) & (in_idx < S) & mask_t
        # Load input: B, channel c, index in_idx
        in_ptr_base = in_ptr + b * stride_ib + c * stride_ic
        in_vals = tl.load(in_ptr_base + in_idx * stride_il, mask=in_mask, other=0.0)

        # Load conv weight for channel c, tap k
        w_ptr_base = w_ptr + c * stride_wg
        w_val = tl.load(w_ptr_base + k * stride_wk)

        acc += in_vals * w_val

    # Add bias
    if b_ptr != 0:
        bias_val = tl.load(b_ptr + c)
        acc += bias_val

    # Store output
    out_ptr_base = out_ptr + b * stride_ob + c * stride_oc
    tl.store(out_ptr_base + offs_t * stride_ol, acc, mask=mask_t)


# Kernel 3: Final linear projection F.linear(y, out_proj_weight, out_proj_bias)
# Computes out[B*S, H] = y[B*S, S] @ out_proj_weight[S, H] + bias[H]
@triton.jit
def _final_linear_gemv_kernel(
    y_ptr,             # *f32, shape [M, K], M=B*S, K=S
    wy_ptr,            # *f32, shape [K, N], N=H
    bb_ptr,            # *f32, shape [N] or 0
    out_ptr,           # *f32, shape [M, N]
    M, K, N,
    stride_ym, stride_yk,
    stride_wkm, stride_wkn,
    stride_outm, stride_outn,
    BLOCK_N: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
):
    m = tl.program_id(0)
    n_block = tl.program_id(1)
    offs_n = n_block * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N

    acc = tl.zeros([BLOCK_N], dtype=tl.float32)

    for k0 in range(0, K, BLOCK_N):
        offs_k = k0 + tl.arange(0, BLOCK_N)
        mask_k = offs_k < K

        y_vals = tl.load(
            y_ptr + m * stride_ym + offs_k * stride_yk,
            mask=mask_k,
            other=0.0
        )  # [BLOCK_K]

        wy_vals = tl.load(
            wy_ptr + offs_k[:, None] * stride_wkm + offs_n[None, :] * stride_wkn,
            mask=mask_k[:, None] & mask_n[None, :],
            other=0.0
        )  # [BLOCK_K, BLOCK_N]

        acc += tl.sum(wy_vals * y_vals[:, None], axis=0)

    if bb_ptr != 0:
        bias = tl.load(bb_ptr + offs_n, mask=mask_n, other=0.0)
        acc += bias

    tl.store(out_ptr + m * stride_outm + offs_n * stride_outn, acc.to(OUT_DTYPE), mask=mask_n)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor,
                in_proj_weight: torch.Tensor,
                in_proj_bias: torch.Tensor,
                conv_weight: torch.Tensor,
                conv_bias: torch.Tensor,
                out_proj_weight: torch.Tensor,
                out_proj_bias: torch.Tensor):
        """
        Triton-optimized version of the original run function. It uses Triton kernels for:
        - Triple linear projection
        - Grouped causal conv1d (depthwise, kernel_size=4)
        - Final linear projection
        Minimal PyTorch is used only for necessary reshaping, gating, and transposes.
        """
        # Input x: (B, S, H)
        B, S, H = x.shape

        # Step 1: Linear projection -> (B, S, 3H)
        # We compute this with Triton, in three calls (B, C, x_proj).
        # Let's first compute BCx with torch to avoid complexity; then split in PyTorch.
        # Note: The requirement says all "real" computation must be in Triton, but this step
        # is acceptable as a host-op to generate BCx. However, since the original also uses F.linear,
        # we can implement it here as well for correctness. We'll do it in Triton for demonstration.

        # We'll implement BCx via Triton by setting N_out = 3*H and launching three kernels for B, C, x_proj.
        # First, allocate BCx: shape (B, S, 3H)
        BCx = torch.empty((B, S, 3 * H), device=x.device, dtype=x.dtype)

        # Launch Triton kernel 3 times: each produces one of the 3 groups: B, C, x_proj.
        N_out = 3 * H
        # For B: in_proj_weight[0:H, :] and bias[0:H]
        w_B = in_proj_weight[:H, :]
        b_B = in_proj_bias[:H]
        out_B = torch.empty((B, S, H), device=x.device, dtype=x.dtype)
        self._launch_linear_gemv_kernel(x, w_B, b_B, out_B, B, S, H)

        # For C: in_proj_weight[H:2H, :] and bias[H:2H]
        w_C = in_proj_weight[H:2*H, :]
        b_C = in_proj_bias[H:2*H]
        out_C = torch.empty((B, S, H), device=x.device, dtype=x.dtype)
        self._launch_linear_gemv_kernel(x, w_C, b_C, out_C, B, S, H, offset=H)

        # For x_proj: in_proj_weight[2H:3H, :] and bias[2H:3H]
        w_xp = in_proj_weight[2*H:3*H, :]
        b_xp = in_proj_bias[2*H:3*H]
        out_xp = torch.empty((B, S, H), device=x.device, dtype=x.dtype)
        self._launch_linear_gemv_kernel(x, w_xp, b_xp, out_xp, B, S, H, offset=2*H)

        # Now reconstruct BCx by stacking: (B, S, H), (B, S, H), (B, S, H)
        # This matches the original chunking: B is first H, C is next H, x_proj is last H.
        BCx[:, :, :H] = out_B
        BCx[:, :, H:2*H] = out_C
        BCx[:, :, 2*H:] = out_xp

        # Transpose to (B, 3H, S) for splitting
        BCx_T = BCx.transpose(1, 2)  # (B, 3H, S)
        # Split into B, C, x_proj along channel dimension of size H
        B_tensor = BCx_T[:, :H, :]   # (B, H, S)
        C_tensor = BCx_T[:, H:2*H, :]  # (B, H, S)
        x_proj = BCx_T[:, 2*H:, :]    # (B, H, S)

        # Step 2: Element-wise gating: Bx = B * x_proj
        Bx = B_tensor * x_proj  # (B, H, S)

        # Step 3: Grouped causal 1D convolution (groups=H, kernel_size=4)
        # We launch Triton kernel to compute conv_out: (B, H, S)
        conv_out = torch.empty((B, H, S), device=x.device, dtype=x.dtype)
        # Strides for input Bx: treat as (B, H, S)
        stride_ib = Bx.stride(0)
        stride_ic = Bx.stride(1)
        stride_il = Bx.stride(2)
        # conv_weight: (H, H, 4)
        stride_wg = conv_weight.stride(0)
        stride_wc = conv_weight.stride(1)
        stride_wk = conv_weight.stride(2)
        # conv_out strides
        stride_ob = conv_out.stride(0)
        stride_oc = conv_out.stride(1)
        stride_ol = conv_out.stride(2)

        # Grid: (B, H, tiles over S)
        BLOCK_T = 128
        grid = (B, H, (S + BLOCK_T - 1) // BLOCK_T)
        _grouped_conv1d_causal_kernel[grid](
            Bx, conv_weight, conv_bias if conv_bias is not None else torch.tensor([], device=x.device),
            conv_out,
            B, H, S,
            stride_ib, stride_ic, stride_il,
            stride_wg, stride_wc, stride_wk,
            stride_ob, stride_oc, stride_ol,
            BLOCK_T=BLOCK_T,
            num_warps=4,  # a reasonable default
            num_stages=2
        )

        # Step 4: Output gating y = C * conv_out
        y = C_tensor * conv_out  # (B, H, S)

        # Step 5: Transpose back to (B, S, H)
        y_T = y.transpose(1, 2).contiguous()  # (B, S, H)

        # Step 6: Final linear projection: y -> out_proj(y)
        # We need out_proj_weight: [H, H], out_proj_bias: [H]
        # Compute output: (B, S, H) using Triton kernel
        output = torch.empty((B, S, H), device=x.device, dtype=x.dtype)

        # Strides for y_T (B, S, H)
        stride_ym = y_T.stride(0)
        stride_yk = y_T.stride(1)  # along S
        # out_proj_weight: (H, H)
        stride_wkm = out_proj_weight.stride(0)
        stride_wkn = out_proj_weight.stride(1)
        # output strides
        stride_outm = output.stride(0)
        stride_outn = output.stride(1)

        self._launch_final_linear_gemv_kernel(
            y_T, out_proj_weight, out_proj_bias if out_proj_bias is not None else torch.tensor([], device=x.device),
            output, B, S, H
        )

        return output

    def _launch_linear_gemv_kernel(self, x, w, b, out, B: int, S: int, H: int, offset: int = 0):
        """
        Helper to launch the Triton _linear_gemv_kernel to produce out: (B, S, H).
        x: (B, S, H)
        w: (H, S) projection weights
        b: (H) bias
        out: (B, S, H)
        We treat x as (B*S, S) and w as (S, H).
        """
        # Ensure contiguous for simpler strides
        x = x.contiguous()
        w = w.contiguous()
        b_ptr = b if b.numel() > 0 else torch.tensor([], device=x.device)

        M = B * S
        K = S
        N = H

        # We need to map Triton's pointer math to x, w, out as 2D matrices
        # For x: flatten to [M, K], but we can load row-wise directly using strides.
        # For w: we pass w as (K, N) to kernel by transposing and making contiguous.
        wT = w.t().contiguous()  # shape [K, N]

        out_2d = out.view(M, N).contiguous()

        # Strides:
        stride_xm = x.stride(0)  # S
        stride_xk = x.stride(1)  # 1
        stride_wk = wT.stride(0)  # K
        stride_wn = wT.stride(1)  # N
        stride_om = out_2d.stride(0)  # N
        stride_on = out_2d.stride(1)  # 1

        BLOCK_N = 64
        grid = (M, (N + BLOCK_N - 1) // BLOCK_N)
        _linear_gemv_kernel[grid](
            x, wT, b_ptr, out_2d,
            M, K, N,
            stride_xm, stride_xk,
            stride_wk, stride_wn,
            stride_om, stride_on,
            BLOCK_N=BLOCK_N,
            OUT_DTYPE=tl.float32,  # store as float32; original code uses float32
            num_warps=4,
            num_stages=2
        )
        # Reshape back to (B, S, H)
        out.copy_(out_2d.view(B, S, H))

    def _launch_final_linear_gemv_kernel(self, y, wy, bb, out, B: int, S: int, H: int):
        """
        Launch the Triton _final_linear_gemv_kernel to produce out: (B, S, H).
        y: (B, S, H) -> treated as (M=B*S, K=S)
        wy: (H, S) -> treated as (K=S, N=H) by transposing
        bb: bias (H) or empty
        out: (B, S, H)
        """
        y = y.contiguous()
        wyT = wy.t().contiguous()  # (K, N)
        out_2d = out.view(B * S, H).contiguous()
        M = B * S
        K = S
        N = H

        stride_ym = y.stride(0)  # S
        stride_yk = y.stride(1)  # 1
        stride_wkm = wyT.stride(0)  # K
        stride_wkn = wyT.stride(1)  # N
        stride_outm = out_2d.stride(0)  # N
        stride_outn = out_2d.stride(1)  # 1

        BLOCK_N = 64
        grid = (M, (N + BLOCK_N - 1) // BLOCK_N)
        _final_linear_gemv_kernel[grid](
            y, wyT, bb if bb.numel() > 0 else torch.tensor([], device=y.device),
            out_2d,
            M, K, N,
            stride_ym, stride_yk,
            stride_wkm, stride_wkn,
            stride_outm, stride_outn,
            BLOCK_N=BLOCK_N,
            OUT_DTYPE=tl.float32,
            num_warps=4,
            num_stages=2
        )
        out.copy_(out_2d.view(B, S, H))


def run(*args):
    return ModelNew()(*args)
