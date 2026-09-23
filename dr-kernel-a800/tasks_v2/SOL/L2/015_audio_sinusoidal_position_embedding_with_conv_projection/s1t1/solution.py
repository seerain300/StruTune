import math
import torch
import triton
import triton.language as tl


# Triton kernels


# Conv2D 3x3 stride=2 padding=1, N_in=1 (for conv1) or N_in=384 (for conv2/3). Output: (B, C_out=384, OH=40, OW=T_out)
# A: input_features, shape (B, N_in, 80, T)
# W: weight, shape (C_out, N_in, 3, 3)
# B: output, shape (B, C_out, 40, T_out)
@triton.jit
def conv2d_3x3_stride2_pad1(
    A_ptr, W_ptr, B_ptr,
    Bsz, Cin, H, T, C_out,
    # strides for A: (B, Cin, H, T)
    stride_Ab, stride_Ac, stride_Ah, stride_At,
    # strides for W: (C_out, Cin, 3, 3)
    stride_Wo, stride_Wi, stride_Wk_h, stride_Wk_w,
    # strides for B: (B, C_out, OH, OW)
    stride_Bb, stride_Bo, stride_Bh, stride_Bw,
    # output dims
    OH, OW,
    # bias pointer (optional): if bias is None, pass a dummy pointer and set has_bias=False
    bias_ptr,
    has_bias: tl.constexpr,
    use_gelu: tl.constexpr,
):
    # program ids: tile over (B*OH, C_out, ceil_div(OW, BLOCK_W))
    pid_m = tl.program_id(0)  # over B*OH
    pid_n = tl.program_id(1)  # over C_out
    pid_w = tl.program_id(2)  # over OW tiles

    # derive b and oh from pid_m
    b = pid_m // OH
    oh = pid_m % OH

    # vector of ow indices for this tile
    BLOCK_W = 64
    ow_vec = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    mask_ow = ow_vec < OW

    # initialize accumulator for output (b, c_out, oh, ow_vec)
    # we accumulate in fp32
    acc = tl.zeros((BLOCK_W,), dtype=tl.float32)

    # loop over input channels Cin and 3x3 kernel window
    # For conv1, Cin=1; for conv2/3, Cin=384
    # We assume N_in=1 or 384; we pass Cin as runtime but loop uses it.
    # Note: Triton requires compile-time loops; we emulate with range(1024) but mask by Cin.
    # Better: use runtime loop by Python for range(0, Cin) and Triton will generate loop.
    # Triton supports while or for with runtime bounds. We use for with runtime Cin.
    for ci in range(0, Cin):
        for kh in range(0, 3):
            for kw in range(0, 3):
                # compute input coordinates (ih, it) corresponding to (oh, ow_vec)
                ih = oh + kh - 1  # because padding=1
                it = ow_vec + kw - 1
                # valid mask
                valid = (ih >= 0) & (ih < H) & (it >= 0) & (it < T) & mask_ow
                # load A[b, ci, ih, it] vectorized over ow_vec
                A_offset = b * stride_Ab + ci * stride_Ac + ih * stride_Ah + it * stride_At
                A_vec = tl.load(A_ptr + A_offset, mask=valid, other=0.0)
                # load W[c_out, ci, kh, kw]
                W_offset = pid_n * stride_Wo + ci * stride_Wi + kh * stride_Wk_h + kw * stride_Wk_w
                w_val = tl.load(W_ptr + W_offset)
                # accumulate
                acc += A_vec * w_val

    # add bias if provided
    if has_bias:
        b_val = tl.load(bias_ptr + pid_n)
        acc += b_val

    # optional GELU
    if use_gelu:
        # exact GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
        inv_sqrt2 = 0.7071067811865476  # 1/sqrt(2)
        acc = 0.5 * acc * (1.0 + tl.math.erf(acc * inv_sqrt2))

    # store to B[b, c_out, oh, ow_vec]
    B_offset = b * stride_Bb + pid_n * stride_Bo + oh * stride_Bh + ow_vec * stride_Bw
    # store as bfloat16
    tl.store(B_ptr + B_offset, acc.to(tl.bfloat16), mask=mask_ow)


# GELU (exact) elementwise kernel:
# Input: X (B, C, F, T) or any shape we pass. Output: Y same shape with GELU applied.
@triton.jit
def gelu_exact(X_ptr, Y_ptr, numel: tl.constexpr):
    pid = tl.program_id(0)
    # pointer arithmetic via flat indexing
    # Y = 0.5 * X * (1 + erf(X / sqrt(2)))
    x = tl.load(X_ptr + pid)
    inv_sqrt2 = 0.7071067811865476
    y = 0.5 * x * (1.0 + tl.math.erf(x * inv_sqrt2))
    tl.store(Y_ptr + pid, y)


# GEMM Triton kernel: A (M, K) where M = B * t, K = 15360; B (N, K) where N = 1024; output C (M, N)
# Here, we implement a tiled GEMM with fp32 accumulation. We will not use torch.linear; all math in Triton.
# Note: We need to handle (B, t, K_in) A and (N_out=1024, K_in=15360) B; we can pass B as conv_out_weight.
# We'll store fp32 and cast in host. To satisfy Triton-only, we'll perform scaling in-kernel.
@triton.jit
def gemm_bf16_fp32(A_ptr, B_ptr, C_ptr,
                   M, N, K,
                   stride_Am, stride_Ak,
                   stride_Bk, stride_Bn,
                   stride_Cm, stride_Cn,
                   scale: tl.float32):
    # Tile sizes (can be tuned)
    BLOCK_M = 64
    BLOCK_N = 128
    BLOCK_K = 128

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m0 = pid_m * BLOCK_M
    n0 = pid_n * BLOCK_N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        k_range = k0 + tl.arange(0, BLOCK_K)
        # compute A tile: (BLOCK_M, BLOCK_K)
        a_ptrs = A_ptr + m0 * stride_Am + k_range[None, :] * stride_Ak  # broadcasting over m
        a_mask = (m0 + tl.arange(0, BLOCK_M))[:, None] < M
        a = tl.load(a_ptrs, mask=a_mask, other=0.0).to(tl.float32)  # read as bf16 and cast to fp32

        # compute B tile: (BLOCK_K, BLOCK_N)
        b_ptrs = B_ptr + k_range[:, None] * stride_Bk + n0 * stride_Bn  # broadcasting over n
        b_mask = (n0 + tl.arange(0, BLOCK_N))[None, :] < N
        b = tl.load(b_ptrs, mask=b_mask, other=0.0).to(tl.float32)  # read as bf16 and cast to fp32

        # FMA
        acc += tl.dot(a, b)

    # scale
    acc = acc * scale

    # write C tile: (BLOCK_M, BLOCK_N)
    c_ptrs = C_ptr + (m0 + tl.arange(0, BLOCK_M))[:, None] * stride_Cm + (n0 + tl.arange(0, BLOCK_N))[None, :] * stride_Cn
    c_mask = (m0 + tl.arange(0, BLOCK_M))[:, None] < M and (n0 + tl.arange(0, BLOCK_N))[None, :] < N
    tl.store(c_ptrs, acc, mask=c_mask)


# Triton kernel to add positional embedding: y (B, t, N_out), pos_embed (1, t, N_out)
# We can slice pos_embed along batch (first element). Triton supports elementwise addition.
@triton.jit
def add_pos_embed_scale(y_ptr, pos_ptr, Bsz, t, N_out,
                         stride_yb, stride_yt, stride_yn,
                         stride_pos_t, stride_pos_n,
                         scale: tl.float32):
    pid_b = tl.program_id(0)  # over batch
    pid_t = tl.program_id(1)  # over t
    pid_n = tl.program_id(2)  # over N_out tiles

    n0 = pid_n * 64
    n_idx = n0 + tl.arange(0, 64)
    mask_n = n_idx < N_out

    # load y[b, t, n_idx]
    y_ptrs = y_ptr + pid_b * stride_yb + pid_t * stride_yt + n_idx * stride_yn
    y_vec = tl.load(y_ptrs, mask=mask_n, other=0.0).to(tl.float32)
    # load pos_embed[0, t, n_idx] (broadcast across batch)
    pos_ptrs = pos_ptr + 0 * stride_yb + pid_t * stride_pos_t + n_idx * stride_pos_n
    pos_vec = tl.load(pos_ptrs, mask=mask_n, other=0.0).to(tl.float32)

    # add scaled pos
    y_vec = y_vec + scale * pos_vec

    tl.store(y_ptrs, y_vec, mask=mask_n)


# Forward: ModelNew
class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # No parameters; we rely on inputs provided to forward.

    def forward(self, input_features: torch.Tensor,
                conv2d1_weight: torch.Tensor, conv2d1_bias: torch.Tensor,
                conv2d2_weight: torch.Tensor, conv2d2_bias: torch.Tensor,
                conv2d3_weight: torch.Tensor, conv2d3_bias: torch.Tensor,
                conv_out_weight: torch.Tensor,
                positional_embedding: torch.Tensor,
                embed_scale: float):
        # Ensure everything is on CUDA and dtype bfloat16
        device = input_features.device
        assert device.type == 'cuda', "All tensors must be on CUDA for Triton kernels."
        assert input_features.dtype == torch.bfloat16, "Input features must be bfloat16."
        # conv weights: (C_out, Cin, 3, 3)
        # biases: (C_out,)

        Bsz, Cin, H, T = input_features.shape
        assert Cin == 1, "This Triton conv kernel assumes Cin=1 for the first conv."

        # Conv1
        x = torch.empty((Bsz, 384, 40, (T - 3) // 2 + 1), dtype=torch.bfloat16, device=device)
        conv2d_3x3_stride2_pad1(
            input_features, conv2d1_weight, x,
            Bsz, 1, H, T, 384,
            input_features.stride(0), input_features.stride(1), input_features.stride(2), input_features.stride(3),
            conv2d1_weight.stride(0), conv2d1_weight.stride(1), 3, 3,
            x.stride(0), x.stride(1), 40, (T - 3) // 2 + 1,
            conv2d1_bias,
            True,  # has_bias
            True,  # use_gelu
            num_warps=4, num_stages=2
        )
        # GELU conv1
        # We need a contiguous flattened tensor for gelu_exact. Gelu exact is elementwise; Triton kernel expects flat pointer.
        # We'll apply GELU in-place on x by using a temporary buffer and copying back. But since x is contiguous, we can flatten.
        x_flat = x.reshape(-1)
        y_flat = torch.empty_like(x_flat, dtype=torch.bfloat16, device=device)
        gelu_exact(x_flat, y_flat, numel=x_flat.numel())
        x = y_flat.reshape(x.shape)

        # Conv2
        x = torch.empty((Bsz, 384, 40, (x.shape[-1] - 3) // 2 + 1), dtype=torch.bfloat16, device=device)
        conv2d_3x3_stride2_pad1(
            x, conv2d2_weight, x,
            Bsz, 384, 40, x.shape[-1], 384,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            conv2d2_weight.stride(0), conv2d2_weight.stride(1), 3, 3,
            x.stride(0), x.stride(1), 40, (x.shape[-1] - 3) // 2 + 1,
            conv2d2_bias,
            True,
            True,
            num_warps=4, num_stages=2
        )
        x_flat = x.reshape(-1)
        y_flat = torch.empty_like(x_flat, dtype=torch.bfloat16, device=device)
        gelu_exact(x_flat, y_flat, numel=x_flat.numel())
        x = y_flat.reshape(x.shape)

        # Conv3
        x = torch.empty((Bsz, 384, 40, (x.shape[-1] - 3) // 2 + 1), dtype=torch.bfloat16, device=device)
        conv2d_3x3_stride2_pad1(
            x, conv2d3_weight, x,
            Bsz, 384, 40, x.shape[-1], 384,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            conv2d3_weight.stride(0), conv2d3_weight.stride(1), 3, 3,
            x.stride(0), x.stride(1), 40, (x.shape[-1] - 3) // 2 + 1,
            conv2d3_bias,
            True,
            True,
            num_warps=4, num_stages=2
        )
        x_flat = x.reshape(-1)
        y_flat = torch.empty_like(x_flat, dtype=torch.bfloat16, device=device)
        gelu_exact(x_flat, y_flat, numel=x_flat.numel())
        x = y_flat.reshape(x.shape)

        # Reshape to (B, t, 15360)
        Bsz, C, F, t = x.shape
        assert C == 384 and F == 40, "Conv output must have C=384, F=40."
        x = x.permute(0, 3, 1, 2).contiguous().view(Bsz, t, C * F)  # (B, t, 15360)

        # Linear projection via Triton GEMM: x (B, t, 15360) @ conv_out_weight.T (15360, 1024)
        # Prepare A and B:
        # A is x.view(B*t, 15360), B is conv_out_weight (1024, 15360). We launch gemm_bf16_fp32 to produce C_fp32 (B*t, 1024).
        A = x.reshape(Bsz * t, 15360).contiguous()
        B = conv_out_weight.contiguous()  # (1024, 15360)
        M = Bsz * t
        N = 1024
        K = 15360

        # C_fp32 as fp32 for numerical stability
        C_fp32 = torch.empty((M, N), dtype=torch.float32, device=device)

        # Strides
        stride_Am = A.stride(0)
        stride_Ak = A.stride(1)
        stride_Bk = B.stride(1)
        stride_Bn = B.stride(0)
        stride_Cm = C_fp32.stride(0)
        stride_Cn = C_fp32.stride(1)

        # Launch GEMM
        grid = (triton.cdiv(M, 64), triton.cdiv(N, 128))
        gemm_bf16_fp32[grid](
            A, B, C_fp32,
            M, N, K,
            stride_Am, stride_Ak,
            stride_Bk, stride_Bn,
            stride_Cm, stride_Cn,
            float(embed_scale),
            num_warps=4, num_stages=2
        )

        # Cast to bfloat16 to match original pipeline
        y = C_fp32.to(torch.bfloat16).view(Bsz, t, N)

        # Add positional embedding with scaling in Triton
        # pos_embed: (1500, 1024), we take first t rows. Cast to bf16 and scale by embed_scale.
        pos_embed = positional_embedding[:t, :].to(torch.bfloat16).unsqueeze(0).contiguous()  # (1, t, 1024)
        # y is (B, t, 1024); pos_embed is (1, t, 1024). We can use add_pos_embed_scale kernel to add scaled pos.
        Bsz_out, t_out, N_out = y.shape
        grid_add = (Bsz_out, t_out, triton.cdiv(N_out, 64))
        add_pos_embed_scale[grid_add](
            y, pos_embed,
            Bsz_out, t_out, N_out,
            y.stride(0), y.stride(1), y.stride(2),
            pos_embed.stride(1), pos_embed.stride(2),
            float(embed_scale),
            num_warps=4, num_stages=2
        )

        return y


def run(*args):
    return ModelNew()(*args)
