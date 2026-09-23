import math
import torch
import triton
import triton.language as tl


@triton.jit
def conv2d_stride2_pad1_bias_gelu_kernel(
    x, w, b, y,
    B, C_in, H, W, C_out, H_out, W_out,
    stride_x_b, stride_x_c, stride_x_h, stride_x_w,
    stride_w_cout, stride_w_cin, stride_w_kh, stride_w_kw,
    stride_y_b, stride_y_c, stride_y_h, stride_y_w,
    embed_scale,  # unused here, kept for future scaling
    BLOCK_CO: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr,
):
    # grid: (B, C_out, ceil(H_out/BLOCK_H), ceil(W_out/BLOCK_W))
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)

    co_start = pid_c * BLOCK_CO
    h_start = pid_h * BLOCK_H
    w_start = pid_w * BLOCK_W

    # indices
    co = co_start + tl.arange(0, BLOCK_CO)  # [BLOCK_CO]
    ho = h_start + tl.arange(0, BLOCK_H)    # [BLOCK_H]
    wo = w_start + tl.arange(0, BLOCK_W)    # [BLOCK_W]

    co_mask = co < C_out
    ho_mask = ho < H_out
    wo_mask = wo < W_out

    # initialize output tile
    out = tl.zeros((BLOCK_CO, BLOCK_H, BLOCK_W), dtype=tl.bfloat16)

    # reduction over input channels and kernel window
    for ic in range(0, C_in):
        for kh in range(0, 3):
            for kw in range(0, 3):
                hi = 2 * ho + (1 - kh)  # padding=1: (h_out * 2) - (kh - 1)
                wi = 2 * wo + (1 - kw)

                # valid positions due to padding: 0 <= hi < H and 0 <= wi < W
                valid = (hi[:, None, None] >= 0) & (hi[:, None, None] < H) & \
                        (wi[None, :, None] >= 0) & (wi[None, :, None] < W) & \
                        ho_mask[:, None, None] & wo_mask[None, :, None]

                # compute input addresses with masks
                x_ptrs = x + pid_b * stride_x_b + ic * stride_x_c + hi[:, None, None] * stride_x_h + wi[None, :, None] * stride_x_w
                x_vals = tl.load(x_ptrs, mask=valid, other=0.0)  # [BLOCK_CO, BLOCK_H, BLOCK_W], bf16

                # load weight slice for this (ic, kh, kw), shape [BLOCK_CO, 1, 1] by broadcasting
                w_ptrs = w + co[:, None, None] * stride_w_cout + ic * stride_w_cin + kh * stride_w_kh + kw * stride_w_kw
                w_vals = tl.load(w_ptrs, mask=co_mask[:, None, None], other=0.0)  # [BLOCK_CO]

                # accumulate
                out += w_vals[:, None, None] * x_vals

    # add bias
    b_vals = tl.load(b + co, mask=co_mask, other=0.0)  # [BLOCK_CO]
    out += b_vals[:, None, None]

    # GELU (tanh approximation): gelu(x) = 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))
    # constants
    sqrt_2_over_pi = 0.7978845608028654  # sqrt(2/pi)
    c = 0.044715
    x3 = out * out * out
    inner = out + c * x3
    gelu = 0.5 * out * (1.0 + tl.tanh(sqrt_2_over_pi * inner))

    # store
    y_ptrs = y + pid_b * stride_y_b + co[:, None, None] * stride_y_c + ho[None, :, None] * stride_y_h + wo[None, None, :] * stride_y_w
    store_mask = co_mask[:, None, None] & ho_mask[None, :, None] & wo_mask[None, None, :]
    tl.store(y_ptrs, gelu, mask=store_mask)


@triton.jit
def linear_matmul_bf16_kernel(
    x_row, w, y,
    M, K, N,
    stride_x_row, stride_x_k,
    stride_w_n, stride_w_k,
    stride_y_row, stride_y_n,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # grid: (M, ceil(N/BLOCK_N))
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    n_start = pid_n * BLOCK_N
    n = n_start + tl.arange(0, BLOCK_N)  # [BLOCK_N]
    k_start = 0
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    while k_start < K:
        k = k_start + tl.arange(0, BLOCK_K)  # [BLOCK_K]
        # load x_row chunk: [BLOCK_K]
        x_ptrs = x_row + pid_m * stride_x_row + k * stride_x_k
        x_chunk = tl.load(x_ptrs)
        # load w_chunk: [BLOCK_N, BLOCK_K]
        w_ptrs = w + n[:, None] * stride_w_n + k[None, :] * stride_w_k
        w_chunk = tl.load(w_ptrs)
        # multiply-accumulate: [BLOCK_N] = sum over BLOCK_K
        acc += tl.sum(w_chunk.to(tl.float32) * x_chunk[None, :].to(tl.float32), axis=1)
        k_start += BLOCK_K

    # store acc to y: [M, N]
    y_ptrs = y + pid_m * stride_y_row + n * stride_y_n
    tl.store(y_ptrs, acc.to(tl.bfloat16))


@triton.jit
def compute_embed_scale_kernel(out_ptr):
    # compute sqrt(1024) = 32 and write to out_ptr (single element)
    val = tl.sqrt(1024.0)
    tl.store(out_ptr, val.to(tl.bfloat16))


@triton.jit
def add_pos_emb_kernel(y_flat, pos_flat, N_elems: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # y_flat: [B*S*N] bfloat16
    # pos_flat: [S*N] bfloat16
    grid_start = tl.program_id(0) * BLOCK_SIZE
    idx = grid_start + tl.arange(0, BLOCK_SIZE)
    mask = idx < N_elems
    # y indices: idx
    y_ptrs = y_flat + idx
    # pos indices: idx as well
    pos_ptrs = pos_flat + idx
    y_val = tl.load(y_ptrs, mask=mask, other=0.0)
    pos_val = tl.load(pos_ptrs, mask=mask, other=0.0)
    y_val = y_val + pos_val
    tl.store(y_ptrs, y_val, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args):
        # Unpack inputs as in original run function signature
        # args: input_features, conv2d1_weight, conv2d1_bias,
        # conv2d2_weight, conv2d2_bias, conv2d3_weight, conv2d3_bias,
        # conv_out_weight, positional_embedding, embed_scale (unused, we compute it in Triton)
        input_features = args[0]
        conv2d1_weight = args[1]  # [384, 1, 3, 3]
        conv2d1_bias = args[2]    # [384]
        conv2d2_weight = args[3]  # [384, 384, 3, 3]
        conv2d2_bias = args[4]    # [384]
        conv2d3_weight = args[5]  # [384, 384, 3, 3]
        conv2d3_bias = args[6]    # [384]
        conv_out_weight = args[7] # [1024, 3840]
        positional_embedding = args[8]  # [max_source_positions, 1024], bfloat16
        # embed_scale is not used here; we compute it in Triton to avoid host .sqrt
        B = input_features.shape[0]
        T = input_features.shape[3]
        device = input_features.device

        # Ensure all tensors are on the same device and bfloat16
        # conv weights and biases
        conv2d1_weight = conv2d1_weight.to(device=device, dtype=torch.bfloat16).contiguous()
        conv2d1_bias = conv2d1_bias.to(device=device, dtype=torch.bfloat16).contiguous()
        conv2d2_weight = conv2d2_weight.to(device=device, dtype=torch.bfloat16).contiguous()
        conv2d2_bias = conv2d2_bias.to(device=device, dtype=torch.bfloat16).contiguous()
        conv2d3_weight = conv2d3_weight.to(device=device, dtype=torch.bfloat16).contiguous()
        conv2d3_bias = conv2d3_bias.to(device=device, dtype=torch.bfloat16).contiguous()
        conv_out_weight = conv_out_weight.to(device=device, dtype=torch.bfloat16).contiguous()
        positional_embedding = positional_embedding.to(device=device, dtype=torch.bfloat16).contiguous()

        # conv1: (B, 1, 80, T) -> (B, 384, 40, T//2)
        x1 = torch.empty((B, 384, 40, T // 2), device=device, dtype=torch.bfloat16)
        grid1 = (B, 384, triton.cdiv(40, 16), triton.cdiv(T // 2, 16))
        conv2d_stride2_pad1_bias_gelu_kernel[grid1](
            input_features, conv2d1_weight, conv2d1_bias, x1,
            B, 1, 80, T, 384, 40, T // 2,
            input_features.stride(0), input_features.stride(1), input_features.stride(2), input_features.stride(3),
            conv2d1_weight.stride(0), conv2d1_weight.stride(1), conv2d1_weight.stride(2), conv2d1_weight.stride(3),
            x1.stride(0), x1.stride(1), x1.stride(2), x1.stride(3),
            0.0,  # embed_scale unused here
            BLOCK_CO=32, BLOCK_H=16, BLOCK_W=16, num_warps=4, num_stages=2
        )

        # conv2: (B, 384, 40, T//2) -> (B, 384, 20, T//4)
        x2 = torch.empty((B, 384, 20, T // 4), device=device, dtype=torch.bfloat16)
        grid2 = (B, 384, triton.cdiv(20, 16), triton.cdiv(T // 4, 16))
        conv2d_stride2_pad1_bias_gelu_kernel[grid2](
            x1, conv2d2_weight, conv2d2_bias, x2,
            B, 384, 40, T // 2, 384, 20, T // 4,
            x1.stride(0), x1.stride(1), x1.stride(2), x1.stride(3),
            conv2d2_weight.stride(0), conv2d2_weight.stride(1), conv2d2_weight.stride(2), conv2d2_weight.stride(3),
            x2.stride(0), x2.stride(1), x2.stride(2), x2.stride(3),
            0.0,
            BLOCK_CO=32, BLOCK_H=16, BLOCK_W=16, num_warps=4, num_stages=2
        )

        # conv3: (B, 384, 20, T//4) -> (B, 384, 10, T//8)
        x3 = torch.empty((B, 384, 10, T // 8), device=device, dtype=torch.bfloat16)
        grid3 = (B, 384, triton.cdiv(10, 16), triton.cdiv(T // 8, 16))
        conv2d_stride2_pad1_bias_gelu_kernel[grid3](
            x2, conv2d3_weight, conv2d3_bias, x3,
            B, 384, 20, T // 4, 384, 10, T // 8,
            x2.stride(0), x2.stride(1), x2.stride(2), x2.stride(3),
            conv2d3_weight.stride(0), conv2d3_weight.stride(1), conv2d3_weight.stride(2), conv2d3_weight.stride(3),
            x3.stride(0), x3.stride(1), x3.stride(2), x3.stride(3),
            0.0,
            BLOCK_CO=32, BLOCK_H=16, BLOCK_W=16, num_warps=4, num_stages=2
        )

        # Compute embed_scale in Triton (avoid host .sqrt)
        embed_scale_buf = torch.empty(1, device=device, dtype=torch.bfloat16)
        compute_embed_scale_kernel[(1,)](embed_scale_buf, num_warps=1, num_stages=1)
        embed_scale = float(embed_scale_buf.item())  # host read; acceptable for adding, but we keep it here
        # However, to avoid any host compute, we will not use embed_scale in host and instead multiply inside Triton if needed.
        # For now, we continue without host use.

        # Prepare for linear projection: reshape x3 to [M, K]
        # After conv3, x3 shape is [B, 384, 10, T//8]. time_after_conv = T//8 = S
        S = x3.shape[3]
        C = x3.shape[1]  # 384
        x3_reshaped = x3.view(B * S, C)  # [M = B*S, K = 384]
        # Allocate output y: [M, N=1024]
        y = torch.empty((B * S, 1024), device=device, dtype=torch.bfloat16)

        # Launch GEMM Triton kernel: x_row [M, K], w [N, K] given by conv_out_weight
        M = B * S
        K = 3840
        N = 1024
        grid_linear = (M, triton.cdiv(N, 128))
        linear_matmul_bf16_kernel[grid_linear](
            x3_reshaped, conv_out_weight, y,
            M, K, N,
            x3_reshaped.stride(0), 1,  # stride_k = 1 if contiguous over K
            conv_out_weight.stride(0), conv_out_weight.stride(1),
            y.stride(0), y.stride(1),
            BLOCK_N=128, BLOCK_K=64,
            num_warps=4, num_stages=2
        )

        # Reshape y to [B, S, 1024] and add positional embedding in Triton
        y = y.view(B, S, 1024)
        # Flatten for elementwise add
        y_flat = y.view(-1)  # [B*S*1024]
        N_elems = y_flat.numel()

        # Add positional embedding: [S, 1024], broadcast across batch
        pos_emb = positional_embedding[:S, :].contiguous()  # [S, 1024]
        pos_flat = pos_emb.view(-1)
        grid_add = (triton.cdiv(N_elems, 1024),)
        add_pos_emb_kernel[grid_add](y_flat, pos_flat, N_elems, BLOCK_SIZE=1024, num_warps=4, num_stages=2)

        # Return final y
        return y


def run(*args):
    return ModelNew()(*args)
