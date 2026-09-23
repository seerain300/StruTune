import math
import torch
import torch.nn.functional as F

# Triton kernels
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# GEMM kernel: compute Y = X @ W, where
# X is [M, K], M = B*t, K = c*f (e.g., 3840)
# W is [N, K], N = d_model = 1024
# Output Y is [M, N] which we reshape back to [B, t, N] in host code.
@triton.jit
def linear_matmul_kernel(
    x_ptr,        # *const bfloat16
    w_ptr,        # *const bfloat16
    y_ptr,        # *bfloat16
    M, K, N,
    stride_x_row, stride_x_k,   # X strides: row (M), k (K)
    stride_w_n, stride_w_k,     # W strides: n (N), k (K)
    stride_y_row, stride_y_n,   # Y strides: row (M), n (N)
    BLOCK_N: tl.constexpr,      # e.g., 128
    BLOCK_K: tl.constexpr,      # e.g., 64
    num_warps: tl.constexpr,
    num_stages: tl.constexpr,
):
    pid_row = tl.program_id(0)  # over M
    pid_nblk = tl.program_id(1) # over N blocks
    n_start = pid_nblk * BLOCK_N
    n_offsets = n_start + tl.arange(0, BLOCK_N)
    n_mask = n_offsets < N

    # accumulator in fp32
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)

    # loop over K dimension
    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < K

        # load X row block: x[pid_row, k_offsets]
        x_row_ptrs = x_ptr + pid_row * stride_x_row + k_offsets * stride_x_k
        x_block = tl.load(x_row_ptrs, mask=k_mask, other=0.0)  # bfloat16, will cast to fp32

        # load W[n_offsets, k_offsets] as [BLOCK_N, BLOCK_K]
        w_block_ptrs = w_ptr + n_offsets[:, None] * stride_w_n + k_offsets[None, :] * stride_w_k
        w_block_mask = (n_mask[:, None]) & (k_mask[None, :])
        w_block = tl.load(w_block_ptrs, mask=w_block_mask, other=0.0)  # bfloat16, will cast to fp32

        # accumulate: acc[n] += sum_k w_block[n, k] * x_block[k]
        acc += tl.sum(w_block.to(tl.float32) * x_block[None, :].to(tl.float32), axis=1)

    # store to Y
    y_row_ptrs = y_ptr + pid_row * stride_y_row + n_offsets * stride_y_n
    tl.store(y_row_ptrs, acc.to(tl.float16), mask=n_mask)


# Elementwise scale: y[i] *= scale (fp32 scale)
@triton.jit
def scale_elementwise_kernel(y_ptr, scale, N_elems: tl.constexpr, num_warps: tl.constexpr, num_stages: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * 1024 + tl.arange(0, 1024)
    mask = offs < N_elems
    y = tl.load(y_ptr + offs, mask=mask, other=0.0)
    y = y * scale  # scale is fp32, y is bfloat16; Triton handles casting
    tl.store(y_ptr + offs, y, mask=mask)


# Elementwise add positional embedding: y[i] += pos_emb[i, n]
# We assume y_ptr points to a flat [M, N] buffer and pos_ptr is a flat [M*N] buffer.
@triton.jit
def add_pos_emb_kernel(y_ptr, pos_ptr, N_elems: tl.constexpr, num_warps: tl.constexpr, num_stages: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * 1024 + tl.arange(0, 1024)
    mask = offs < N_elems
    y = tl.load(y_ptr + offs, mask=mask, other=0.0)
    pos = tl.load(pos_ptr + offs, mask=mask, other=0.0)
    y = y + pos
    tl.store(y_ptr + offs, y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(
        self,
        input_features: torch.Tensor,        # [B, 1, 80, T]
        conv2d1_weight: torch.Tensor,        # [384, 1, 3, 3]
        conv2d1_bias: torch.Tensor,          # [384]
        conv2d2_weight: torch.Tensor,        # [384, 384, 3, 3]
        conv2d2_bias: torch.Tensor,          # [384]
        conv2d3_weight: torch.Tensor,        # [384, 384, 3, 3]
        conv2d3_bias: torch.Tensor,          # [384]
        conv_out_weight: torch.Tensor,       # [N=1024, K=c*f=3840]
        positional_embedding: torch.Tensor,  # [max_source_positions, N=1024]
        embed_scale: float,                  # sqrt(1024) = 32.0
    ):
        # Ensure device/dtype: use bfloat16
        device = input_features.device
        dtype = torch.bfloat16

        # Stage 1 conv
        x = F.conv2d(input_features, conv2d1_weight, conv2d1_bias, stride=2, padding=1)
        x = F.gelu(x)

        # Stage 2 conv
        x = F.conv2d(x, conv2d2_weight, conv2d2_bias, stride=2, padding=1)
        x = F.gelu(x)

        # Stage 3 conv
        x = F.conv2d(x, conv2d3_weight, conv2d3_bias, stride=2, padding=1)
        x = F.gelu(x)  # x shape: [B, 384, 10, t] where t = time_after_conv = T // 8

        # Reshape exactly as original: permute to [B, t, 384, 10] then flatten last two dims
        B, C, f, t = x.shape  # C=384, f=10, t=time_after_conv
        x_perm = x.permute(0, 3, 1, 2).contiguous()  # [B, t, 384, 10]
        K = C * f  # 3840
        x_view = x_perm.view(B, t, K)  # [B, t, 3840]

        # Linear projection: y = x_view @ conv_out_weight -> [B, t, 1024]
        B2, t2, K2 = x_view.shape
        N = conv_out_weight.shape[0]  # 1024
        # Flatten to [M, K] for kernel, where M = B * t
        M = B2 * t2
        # Allocate output y as [M, N] then reshape to [B, t, N]
        y_flat = torch.empty((M, N), device=device, dtype=torch.bfloat16)

        # Prepare inputs: x_view is [M, K], conv_out_weight is [N, K]
        x_row = x_view.reshape(M, K2).contiguous()        # [M, K]
        w = conv_out_weight.contiguous()                  # [N, K]

        # Launch Triton GEMM kernel
        BLOCK_N = 128
        BLOCK_K = 64
        grid = (M, triton.cdiv(N, BLOCK_N))
        linear_matmul_kernel[grid](
            x_row, w, y_flat,
            M, K2, N,
            x_row.stride(0), x_row.stride(1),
            w.stride(0), w.stride(1),
            y_flat.stride(0), y_flat.stride(1),
            BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        # Reshape back to [B, t, N]
        y = y_flat.view(B2, t2, N)

        # Scale by embed_scale
        y_flat = y.view(-1)  # [B*t*N]
        N_elems = y_flat.numel()
        grid_scale = (triton.cdiv(N_elems, 1024),)
        scale_elementwise_kernel[grid_scale](y_flat, float(embed_scale), N_elems=N_elems, num_warps=4, num_stages=2)

        # Add positional embedding [t, N], broadcast over batch
        pos_emb = positional_embedding[:t2, :].to(torch.bfloat16).contiguous()  # [t, N]
        # Flatten for elementwise add
        grid_add = (triton.cdiv(N_elems, 1024),)
        add_pos_emb_kernel[grid_add](y_flat, pos_emb.view(-1), N_elems=N_elems, num_warps=4, num_stages=2)

        # Reshape back to [B, t, N]
        y = y_flat.view(B2, t2, N)

        return y


def run(*args):
    return ModelNew()(*args)
