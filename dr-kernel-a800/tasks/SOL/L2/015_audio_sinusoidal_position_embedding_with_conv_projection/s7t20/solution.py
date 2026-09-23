import math
import triton
import triton.language as tl


@triton.jit
def linear_proj_rowwise_kernel(
    X_ptr,         # *ptr to [B*S, K], bfloat16
    W_ptr,         # *ptr to [N, K], bfloat16
    Y_ptr,         # *ptr to [B*S, N], bfloat16
    B: tl.int32, S: tl.int32, K: tl.int32, N: tl.int32,
    x_stride0: tl.int32,          # stride for X row (elements)
    w_stride0: tl.int32, w_stride1: tl.int32,  # strides for W
    y_stride0: tl.int32, y_stride1: tl.int32,  # strides for Y
    BLOCK_N: tl.constexpr,  # tile over output channels (N)
    BLOCK_K: tl.constexpr,  # tile over reduction dimension (K)
):
    # Each program computes one output row: for given pid in [0, B*S), compute all N outputs
    pid = tl.program_id(0)
    # Guard against overlaunch (shouldn't happen if grid=(B*S,), but keep for safety)
    # Compute (b, t) from pid
    b = pid // S
    t = pid % S

    # Precompute base offset for X row: X is logically [B*S, K] with row stride x_stride0
    # Note: We passed x_stride0 as element stride (PyTorch contiguous => x_stride0 == K)
    x_row_offset = b * x_stride0 + t * K

    # Iterate over output channels in chunks of BLOCK_N
    for co_start in range(0, N, BLOCK_N):
        co = co_start + tl.arange(0, BLOCK_N)
        co_mask = co < N

        # Accumulator for this output row chunk
        acc = tl.zeros((BLOCK_N,), dtype=tl.bfloat16)

        # Reduction over K in chunks of BLOCK_K
        for k_start in range(0, K, BLOCK_K):
            k = k_start + tl.arange(0, BLOCK_K)
            k_mask = k < K

            # Load X row segment [BLOCK_K]
            x = tl.load(X_ptr + x_row_offset + k, mask=k_mask, other=0.0)

            # Load W segment [BLOCK_N, BLOCK_K]
            w_ptrs = W_ptr + co[:, None] * w_stride0 + k[None, :] * w_stride1
            w_mask = (co_mask[:, None] & k_mask[None, :])
            w = tl.load(w_ptrs, mask=w_mask, other=0.0)

            # Accumulate: acc += sum_k w[:, k] * x[k]
            acc += tl.sum(w * x[None, :], axis=1)

        # Store results for this chunk
        y_row_ptr = Y_ptr + pid * y_stride0 + co * y_stride1
        tl.store(y_row_ptr, acc, mask=co_mask)


@triton.jit
def scale_elementwise_kernel(Y_ptr, scale: tl.float32, N_elems: tl.int32):
    # Elementwise: Y[i] = Y[i] * scale
    pid = tl.program_id(0)
    offsets = pid * 1024 + tl.arange(0, 1024)
    mask = offsets < N_elems
    y = tl.load(Y_ptr + offsets, mask=mask, other=0.0)
    y = y * scale
    tl.store(Y_ptr + offsets, y, mask=mask)


@triton.jit
def add_pos_emb_kernel(Y_ptr, Pos_ptr, N_elems: tl.int32, S: tl.int32, N: tl.int32):
    # Y_ptr is [B*S, N] flattened
    # Pos_ptr is [S, N] flattened (row-major)
    pid = tl.program_id(0)
    offsets = pid * 1024 + tl.arange(0, 1024)
    mask = offsets < N_elems
    co = offsets % N
    bt = offsets // N
    pos_offset = bt * N + co
    y = tl.load(Y_ptr + offsets, mask=mask, other=0.0)
    p = tl.load(Pos_ptr + pos_offset, mask=mask, other=0.0)
    y = y + p
    tl.store(Y_ptr + offsets, y, mask=mask)


def triton_linear_proj(x_rowwise: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """
    x_rowwise: [B, S, K] or [B*S, K], bfloat16, CUDA
    w: [N, K], bfloat16, CUDA
    Returns y: [B, S, N], bfloat16
    """
    assert x_rowwise.is_cuda and w.is_cuda, "Inputs must be CUDA tensors"
    B, S, K = x_rowwise.shape
    N, Kw = w.shape
    assert Kw == K, "Weight K must match x K dimension"
    x = x_rowwise.contiguous()    # [B*S, K]
    w_c = w.contiguous()          # [N, K]
    y_flat = torch.empty((B * S, N), device=x.device, dtype=torch.bfloat16)

    # Launch grid over B*S rows
    grid = (B * S,)
    linear_proj_rowwise_kernel[grid](
        x, w_c, y_flat,
        B, S, K, N,
        x.stride(0),  # x_stride0 (elements between rows)
        w_c.stride(0), w_c.stride(1),  # W strides
        y_flat.stride(0), y_flat.stride(1),
        BLOCK_N=128, BLOCK_K=128,
        num_warps=4, num_stages=2
    )
    return y_flat.view(B, S, N)


def triton_scale_elementwise(y: torch.Tensor, scale: float) -> torch.Tensor:
    """
    Scale y elementwise by scale. Returns tensor of same dtype as y.
    """
    y_c = y.contiguous()
    y_flat = y_c.view(-1)
    N_elems = y_flat.numel()
    grid = (triton.cdiv(N_elems, 1024),)
    scale_elementwise_kernel[grid](y_flat, scale, N_elems, num_warps=4, num_stages=2)
    return y_flat.view_as(y_c)


def triton_add_pos_emb(y: torch.Tensor, pos_emb: torch.Tensor) -> torch.Tensor:
    """
    y: [B, S, N], bfloat16, CUDA
    pos_emb: [S, N], bfloat16, CUDA
    Returns y + pos_emb (broadcast over batch).
    """
    B, S, N = y.shape
    y_c = y.contiguous()
    y_flat = y_c.view(-1)              # [B*S, N] flattened
    pos_flat = pos_emb.contiguous().view(-1)  # [S*N]
    N_elems = y_flat.numel()
    grid = (triton.cdiv(N_elems, 1024),)
    add_pos_emb_kernel[grid](y_flat, pos_flat, N_elems, S, N, num_warps=4, num_stages=2)
    return y_flat.view(B, S, N)


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Expect inputs identical to get_inputs(): positional arguments
        # We keep convs and GELU in PyTorch for correctness, and use Triton for linear, scale, and pos add.
        input_features = args[0]            # [B, 1, 80, T], bfloat16
        conv2d1_weight = args[1]            # [384, 1, 3, 3], bfloat16
        conv2d1_bias = args[2]              # [384], bfloat16
        conv2d2_weight = args[3]            # [384, 384, 3, 3], bfloat16
        conv2d2_bias = args[4]              # [384], bfloat16
        conv2d3_weight = args[5]            # [384, 384, 3, 3], bfloat16
        conv2d3_bias = args[6]              # [384], bfloat16
        conv_out_weight = args[7]           # [1024, 3840], bfloat16
        positional_embedding = args[8]      # [max_source_positions, 1024], bfloat16
        embed_scale = args[9]               # float

        # Stage 1: Conv2d (1 -> 384 channels) + GELU
        x1 = torch.nn.functional.conv2d(input_features, conv2d1_weight, conv2d1_bias, stride=2, padding=1)
        x1 = torch.nn.functional.gelu(x1)

        # Stage 2: Conv2d (384 -> 384 channels) + GELU
        x2 = torch.nn.functional.conv2d(x1, conv2d2_weight, conv2d2_bias, stride=2, padding=1)
        x2 = torch.nn.functional.gelu(x2)

        # Stage 3: Conv2d (384 -> 384 channels) + GELU
        x3 = torch.nn.functional.conv2d(x2, conv2d3_weight, conv2d3_bias, stride=2, padding=1)
        x3 = torch.nn.functional.gelu(x3)

        # Reshape: (batch, channels, freq, time) -> (batch, time, channels*freq)
        b, c, f, t = x3.size()
        x = x3.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)  # [B, T/8, 384*10] = [B, S, 3840]

        # Linear projection to d_model (1024). Triton kernel: X [B*S, K], W [N, K]
        B = x.shape[0]
        S = x.shape[1]
        K = x.shape[2]
        N = conv_out_weight.shape[0]  # 1024

        # Flatten x to [B*S, K]
        x_rowwise = x.view(B * S, K)

        # Triton GEMM: output is [B*S, N]
        y_flat = triton_linear_proj(x_rowwise, conv_out_weight)  # bfloat16

        # Reshape back to [B, S, N]
        y = y_flat.view(B, S, N)

        # Scale by embed_scale
        y = triton_scale_elementwise(y, float(embed_scale))

        # Add positional embedding [S, N], broadcast over batch
        pos_emb = positional_embedding[:S, :].contiguous()  # [S, N]
        y = triton_add_pos_emb(y, pos_emb)

        return y


def run(*args):
    return ModelNew()(*args)
