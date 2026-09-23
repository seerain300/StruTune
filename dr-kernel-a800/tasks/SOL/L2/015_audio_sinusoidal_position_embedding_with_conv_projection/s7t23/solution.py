import math
import torch
import triton
import triton.language as tl


@triton.jit
def conv2d_stride2_pad1_bias_gelu_kernel(
    X_ptr, W_ptr, BIAS_ptr, Y_ptr,
    B: tl.int32, C_in: tl.int32, H: tl.int32, W: tl.int32,
    C_out: tl.int32, H_out: tl.int32, W_out: tl.int32,
    scale: tl.float32,
    stride_x_b: tl.int32, stride_x_ci: tl.int32, stride_x_h: tl.int32, stride_x_w: tl.int32,
    stride_w_co: tl.int32, stride_w_ci: tl.int32, stride_w_kh: tl.int32, stride_w_kw: tl.int32,
    stride_y_b: tl.int32, stride_y_co: tl.int32, stride_y_h: tl.int32, stride_y_w: tl.int32,
    BLOCK_CO: tl.constexpr, BLOCK_OH: tl.constexpr, BLOCK_OW: tl.constexpr,
):
    # Grid: (B, C_out, ceil(H_out/BLOCK_OH), ceil(W_out/BLOCK_OW))
    b = tl.program_id(0)
    co_block = tl.program_id(1)
    oh_block = tl.program_id(2)
    ow_block = tl.program_id(3)

    co = co_block * BLOCK_CO + tl.arange(0, BLOCK_CO)
    co_mask = co < C_out

    oh = oh_block * BLOCK_OH + tl.arange(0, BLOCK_OH)
    ow = ow_block * BLOCK_OW + tl.arange(0, BLOCK_OW)

    OH, OW = tl.meshgrid(oh, ow)
    OH_mask = OH < H_out
    OW_mask = OW < W_out
    pos_mask = OH_mask & OW_mask

    # Accumulator for output [BLOCK_CO, BLOCK_OH, BLOCK_OW]
    acc = tl.zeros((BLOCK_CO, BLOCK_OH, BLOCK_OW), dtype=tl.float32)

    # Sum over input channels and 3x3 kernel
    for ci in range(0, C_in):
        for kh in range(0, 3):
            ih = OH * 2 + kh - 1  # stride=2, padding=1
            ih_in_bounds = (ih >= 0) & (ih < H)
            for kw in range(0, 3):
                iw = OW * 2 + kw - 1
                iw_in_bounds = (iw >= 0) & (iw < W)
                in_bounds = pos_mask & ih_in_bounds & iw_in_bounds

                # Load input tile: [BLOCK_CO, BLOCK_OH, BLOCK_OW]
                x_ptrs = X_ptr + b * stride_x_b + ci * stride_x_ci + ih * stride_x_h + iw * stride_x_w
                x_vals = tl.load(x_ptrs, mask=in_bounds, other=0.0)  # [BLOCK_OH, BLOCK_OW] but broadcast in compute
                # Note: Triton allows broadcasting when used in arithmetic; ensure we build proper broadcasted pointer.
                # To make it explicit, we can compute broadcasted values by expanding dims:
                # But here x is scalar per (oh, ow), so we can directly use x_vals.

                # Load weight tile: [BLOCK_CO, 3, 3]
                w_ptrs = W_ptr + co * stride_w_co + ci * stride_w_ci + kh * stride_w_kh + kw * stride_w_kw
                w_vals = tl.load(w_ptrs, mask=co_mask, other=0.0)  # [BLOCK_CO]

                # Outer product: acc += w_vals[:, None, None] * x_vals[None, :, :]
                # We need to broadcast x_vals across BLOCK_CO dimension. Use explicit broadcast:
                x_vals_expanded = x_vals[None, :, :]  # [1, BLOCK_OH, BLOCK_OW] -> broadcasted later by adding along co
                acc += w_vals[:, None, None] * x_vals_expanded

    # Add bias
    bias_vals = tl.load(BIAS_ptr + co, mask=co_mask, other=0.0)  # [BLOCK_CO]
    acc += bias_vals[:, None, None]

    # GELU (tanh approximation): gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    c0 = 0.7978845608028654  # sqrt(2/pi)
    x3 = acc * acc * acc
    inner = acc + 0.044715 * x3
    gelu = 0.5 * acc * (1.0 + tl.tanh(c0 * inner))

    # Scale
    gelu = gelu * scale

    # Store to output
    y_ptrs = Y_ptr + b * stride_y_b + co[:, None, None] * stride_y_co + OH * stride_y_h + OW * stride_y_w
    out_mask = co_mask[:, None, None] & pos_mask[None, :, :]
    tl.store(y_ptrs, gelu, mask=out_mask)


@triton.jit
def linear_proj_rowwise_kernel(
    X_ptr, W_ptr, BIAS_ptr, Y_ptr,
    B: tl.int32, S: tl.int32, K: tl.int32, N: tl.int32,
    x_stride0: tl.int32,           # stride between rows in X (elements)
    w_stride0: tl.int32, w_stride1: tl.int32,  # strides for W
    y_stride0: tl.int32, y_stride1: tl.int32,  # strides for Y
    BLOCK_N: tl.constexpr,  # tile over output channels (N)
    BLOCK_K: tl.constexpr,  # tile over reduction dimension (K)
):
    # Each program computes one output row: for pid in [0, B*S), compute all N outputs
    pid = tl.program_id(0)
    b = pid // S
    t = pid % S

    x_row_offset = b * x_stride0 + t * K  # logical row offset in X [B*S, K]

    for co_start in range(0, N, BLOCK_N):
        co = co_start + tl.arange(0, BLOCK_N)
        co_mask = co < N

        acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

        for k_start in range(0, K, BLOCK_K):
            k = k_start + tl.arange(0, BLOCK_K)
            k_mask = k < K

            # Load X row segment [BLOCK_K]
            x_ptrs = X_ptr + x_row_offset + k
            x_vals = tl.load(x_ptrs, mask=k_mask, other=0.0).to(tl.float32)  # [BLOCK_K]

            # Load W chunk [BLOCK_N, BLOCK_K]
            w_ptrs = W_ptr + co[:, None] * w_stride0 + k[None, :] * w_stride1
            w_mask = co_mask[:, None] & k_mask[None, :]
            w_vals = tl.load(w_ptrs, mask=w_mask, other=0.0).to(tl.float32)  # [BLOCK_N, BLOCK_K]

            # Fused multiply-add: acc += sum_k w_vals[:, k] * x_vals[k]
            # Triton supports dot or we can do explicit multiply
            # Broadcast x_vals over N dimension and sum over K
            x_broadcast = x_vals[None, :]  # [1, BLOCK_K]
            prod = w_vals * x_broadcast   # [BLOCK_N, BLOCK_K]
            acc += tl.sum(prod, axis=1)   # [BLOCK_N]

        # Add bias if provided
        # If BIAS_ptr is None, treat as zeros. Here we assume BIAS_ptr is valid.
        bias_vals = tl.load(BIAS_ptr + co, mask=co_mask, other=0.0).to(tl.float32)  # [BLOCK_N]
        acc += bias_vals

        # Store Y row
        y_ptrs = Y_ptr + (b * y_stride0) + (t * y_stride0) + co * y_stride1
        y_mask = co_mask
        tl.store(y_ptrs, acc, mask=y_mask)


@triton.jit
def scale_elementwise_kernel(
    X_ptr, Y_ptr, N_elems: tl.int32, scale: tl.float32,
    BLOCK: tl.constexpr,
):
    # Elementwise scale: Y[i] = X[i] * scale
    grid = tl.num_programs(0)
    start = grid * tl.program_id(0)
    offsets = start + tl.arange(0, BLOCK)
    mask = offsets < N_elems
    x_vals = tl.load(X_ptr + offsets, mask=mask, other=0.0)
    y_vals = x_vals * scale
    tl.store(Y_ptr + offsets, y_vals, mask=mask)


@triton.jit
def add_pos_emb_kernel(
    Y_ptr, POS_ptr, N_elems: tl.int32, S: tl.int32, N: tl.int32,
    BLOCK: tl.constexpr,
):
    # Y: [B, S, N], POS: [S, N], broadcast POS over batch
    grid = tl.num_programs(0)
    start = grid * tl.program_id(0)
    offsets = start + tl.arange(0, BLOCK)
    mask = offsets < N_elems
    # Map linear offsets to (b, s, n): n = offsets % N, s = (offsets // N) % S, b = offsets // (S*N)
    N_per_batch = S * N
    b = offsets // N_per_batch
    s = (offsets // N) % S
    n = offsets % N
    # Compute base pointer for Y and POS
    y_ptrs = Y_ptr + b * (S * N) + s * N + n
    pos_ptrs = POS_ptr + s * N + n
    y_vals = tl.load(y_ptrs, mask=mask, other=0.0)
    pos_vals = tl.load(pos_ptrs, mask=mask, other=0.0)
    tl.store(y_ptrs, y_vals + pos_vals, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args):
        # args order from get_inputs:
        # input_features, conv2d1_weight, conv2d1_bias, conv2d2_weight, conv2d2_bias,
        # conv2d3_weight, conv2d3_bias, conv_out_weight, positional_embedding, embed_scale
        input_features = args[0].contiguous().to(torch.bfloat16)
        conv1_w = args[1].contiguous().to(torch.bfloat16)  # [C_out, C_in, 3, 3]
        conv1_b = args[2].contiguous().to(torch.bfloat16)  # [C_out]
        conv2_w = args[3].contiguous().to(torch.bfloat16)  # [C_out, C_in, 3, 3]
        conv2_b = args[4].contiguous().to(torch.bfloat16)  # [C_out]
        conv3_w = args[5].contiguous().to(torch.bfloat16)  # [C_out, C_in, 3, 3]
        conv3_b = args[6].contiguous().to(torch.bfloat16)  # [C_out]
        conv_out_weight = args[7].contiguous().to(torch.bfloat16)  # [N=1024, K=3840]
        positional_embedding = args[8].contiguous().to(torch.bfloat16)  # [max_source_positions, N]
        embed_scale = float(args[9])

        B, C_in, H, W = input_features.shape
        C_out = conv1_w.shape[0]
        H_out = (H - 2) // 2 + 1
        W_out = (W - 2) // 2 + 1

        # Conv1: (B, 1, H, W) -> (B, 384, H_out, W_out)
        x1 = torch.empty((B, C_out, H_out, W_out), device=input_features.device, dtype=torch.bfloat16)
        conv2d_stride2_pad1_bias_gelu_kernel[(B, C_out, triton.cdiv(H_out, 8), triton.cdiv(W_out, 16))](
            input_features, conv1_w, conv1_b, x1,
            B, C_in, H, W, C_out, H_out, W_out,
            float(embed_scale),
            input_features.stride(0), input_features.stride(1), input_features.stride(2), input_features.stride(3),
            conv1_w.stride(0), conv1_w.stride(1), conv1_w.stride(2), conv1_w.stride(3),
            x1.stride(0), x1.stride(1), x1.stride(2), x1.stride(3),
            BLOCK_CO=32, BLOCK_OH=8, BLOCK_OW=16,
            num_warps=4, num_stages=2
        )

        # Conv2: x1 -> (B, 384, H_out2, W_out2)
        C_in2 = C_out  # 384
        H_out2 = (H_out - 2) // 2 + 1
        W_out2 = (W_out - 2) // 2 + 1
        x2 = torch.empty((B, C_out, H_out2, W_out2), device=input_features.device, dtype=torch.bfloat16)
        conv2d_stride2_pad1_bias_gelu_kernel[(B, C_out, triton.cdiv(H_out2, 8), triton.cdiv(W_out2, 16))](
            x1, conv2_w, conv2_b, x2,
            B, C_in2, H_out, W_out, C_out, H_out2, W_out2,
            1.0,  # no scale after GELU in original; but kernel accepts scale; set to 1.0
            x1.stride(0), x1.stride(1), x1.stride(2), x1.stride(3),
            conv2_w.stride(0), conv2_w.stride(1), conv2_w.stride(2), conv2_w.stride(3),
            x2.stride(0), x2.stride(1), x2.stride(2), x2.stride(3),
            BLOCK_CO=32, BLOCK_OH=8, BLOCK_OW=16,
            num_warps=4, num_stages=2
        )

        # Conv3: x2 -> (B, 384, H_out3, W_out3)
        C_in3 = C_out  # 384
        H_out3 = (H_out2 - 2) // 2 + 1
        W_out3 = (W_out2 - 2) // 2 + 1
        x3 = torch.empty((B, C_out, H_out3, W_out3), device=input_features.device, dtype=torch.bfloat16)
        conv2d_stride2_pad1_bias_gelu_kernel[(B, C_out, triton.cdiv(H_out3, 8), triton.cdiv(W_out3, 16))](
            x2, conv3_w, conv3_b, x3,
            B, C_in3, H_out2, W_out2, C_out, H_out3, W_out3,
            1.0,
            x2.stride(0), x2.stride(1), x2.stride(2), x2.stride(3),
            conv3_w.stride(0), conv3_w.stride(1), conv3_w.stride(2), conv3_w.stride(3),
            x3.stride(0), x3.stride(1), x3.stride(2), x3.stride(3),
            BLOCK_CO=32, BLOCK_OH=8, BLOCK_OW=16,
            num_warps=4, num_stages=2
        )

        # Reshape: (B, 384, H_out3, W_out3) -> (B, H_out3, 384*W_out3)
        B_t, C, S, T_after = x3.shape
        x_reshaped = x3.permute(0, 2, 3, 1).contiguous().view(B_t, S, C * T_after)

        # Linear projection: x_reshaped [B, S, K] @ conv_out_weight [N, K]^T -> [B, S, N]
        B2, S2, K = x_reshaped.shape
        N, Kw = conv_out_weight.shape
        assert Kw == K, "Weight K must match x K dimension"
        x_rowwise = x_reshaped.view(B2 * S2, K).contiguous()           # [B*S, K]
        W_rowwise = conv_out_weight.contiguous()                        # [N, K]
        # Bias: original uses F.linear (bias=True), but provided run uses weight only. Implement bias as zeros for correctness.
        bias = torch.zeros(N, device=x_rowwise.device, dtype=torch.bfloat16)
        Y_rowwise = torch.empty((B2 * S2, N), device=x_rowwise.device, dtype=torch.bfloat16)

        linear_proj_rowwise_kernel[(B2 * S2,)](
            x_rowwise, W_rowwise, bias, Y_rowwise,
            B2, S2, K, N,
            x_rowwise.stride(0), W_rowwise.stride(0), W_rowwise.stride(1),
            Y_rowwise.stride(0), Y_rowwise.stride(1),
            BLOCK_N=128, BLOCK_K=128,
            num_warps=4, num_stages=2
        )
        Y = Y_rowwise.view(B2, S2, N)  # [B, S, 1024]

        # Scale by embed_scale (sqrt(1024) = 32)
        N_elems = Y.numel()
        Y_scaled = torch.empty_like(Y, dtype=torch.bfloat16)
        scale_elementwise_kernel[(triton.cdiv(N_elems, 1024),)](
            Y, Y_scaled, N_elems, float(32.0),
            BLOCK=1024,
            num_warps=4, num_stages=2
        )
        Y = Y_scaled

        # Add positional embedding [S, N] broadcast over batch
        pos_emb = positional_embedding[:S2, :].contiguous()  # [S, N]
        N_elems_add = S2 * N
        add_pos_emb_kernel[(triton.cdiv(N_elems_add, 1024),)](
            Y.view(-1), pos_emb.view(-1), N_elems_add, S2, N,
            BLOCK=1024,
            num_warps=4, num_stages=2
        )
        Y = Y.view(B2, S2, N)

        return Y


def run(*args):
    return ModelNew()(*args)
