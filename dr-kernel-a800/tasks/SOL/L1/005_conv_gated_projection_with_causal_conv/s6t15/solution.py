import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


@triton.jit
def _matmul_linear_kernel(
    A_ptr, B_ptr, Bias_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Compute C = A @ B + Bias, A: (M,K), B: (K,N)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc += tl.dot(a, b)

    bias = tl.load(Bias_ptr + offs_n, mask=(offs_n < N), other=0.0).to(tl.float32)
    acc = acc + bias[None, :]

    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


@triton.jit
def _elementwise_mul_2d_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N,  # M=number of rows, N=number of elements per row
    stride_am, stride_an,
    stride_bm, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # Compute C = A * B, elementwise over 2D [M, N]
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_n[None, :] * stride_an)
    b_ptrs = B_ptr + (offs_m[:, None] * stride_bm + offs_n[None, :] * stride_bn)
    a = tl.load(a_ptrs, mask=mask, other=0.0)
    b = tl.load(b_ptrs, mask=mask, other=0.0)
    c = a * b
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, c, mask=mask)


@triton.jit
def _grouped_causal_conv1d_kernel(
    In_ptr,  # pointer to padded input: (B, H, S+3), float32, contiguous
    W_ptr,   # pointer to conv_weight: (H, 4), float32, contiguous
    Bias_ptr,# pointer to conv_bias: (H), float32, contiguous
    Out_ptr, # pointer to output: (B, H, S), float32, contiguous
    B, H, S,  # sizes: batch, hidden, seq_len
    stride_in_b, stride_in_h, stride_in_s,
    stride_w_h, stride_w_k,
    stride_out_b, stride_out_h, stride_out_s,
    BLOCK_S: tl.constexpr,
):
    # Each program handles one (b, h) pair and writes over s in tiles
    b = tl.program_id(0)
    h = tl.program_id(1)
    # If grid is (B, H, ?), pid2 handles S tiling, here we set grid=(B,H) and iterate S
    # But to keep it general, assume grid=(B*H,). We need 2D grid: we'll pass (B,H) explicitly.
    # For clarity, restructure launch as (B, H) programs with for-loop over S.
    # Triton doesn't support nested dynamic loops over S well; use grid=(B,H) and loop S inside.
    # However, Triton requires grid known at launch; here we launch with grid=(B,H) and compute S in-kernel.
    # But Triton restricts runtime Python loops; instead, we will launch with grid=(B,H) and rely on kernel to iterate over S.
    # Practical approach: we'll use grid=(B,H) and inside the kernel we iterate over s.
    # Note: We still need to iterate over S; Triton allows Python-level loops over runtime variables.
    # Implement: for s_start in range(0, S, BLOCK_S): then s_vec loop.
    # To compute s_vec per iteration, we can't have a Python for with dynamic S. So we'll instead launch with grid=(B,H,ceil(S/BLOCK_S)).
    # But Triton only supports 1D/2D grid. We'll launch grid=(B,H) and let each program loop over S entirely.
    # That means each program will run for all S; we can do that, but it's not ideal for performance.
    # Better: launch 3D grid with program_id(2) iterating tiles of S. Since Triton doesn't support 3D, fallback to grid=(B,H) and loop S.
    # Given typical seq_len, looping S is acceptable.

    # Compute output for all s in [0, S)
    # We'll load W[h, :] once as 4 scalars.
    w0 = tl.load(W_ptr + h * stride_w_h + 0 * stride_w_k)  # weight at k=0
    w1 = tl.load(W_ptr + h * stride_w_h + 1 * stride_w_k)  # weight at k=1
    w2 = tl.load(W_ptr + h * stride_w_h + 2 * stride_w_k)  # weight at k=2
    w3 = tl.load(W_ptr + h * stride_w_h + 3 * stride_w_k)  # weight at k=3
    b_bias = tl.load(Bias_ptr + h)  # bias for channel h

    # Accumulator per s
    acc = tl.zeros((BLOCK_S,), dtype=tl.float32)
    # We'll process S elements in chunks of BLOCK_S. Since S is runtime, we loop.
    # Triton allows for-loops with runtime bounds.
    for s_start in range(0, S, BLOCK_S):
        offs_s = s_start + tl.arange(0, BLOCK_S)
        # Mask for valid s
        mask_s = offs_s < S
        # Load 4 values from padded input at positions s+3, s+2, s+1, s
        # Note: padded input length along S is S+3
        # s+3
        s3 = offs_s + 3
        mask3 = mask_s & (s3 < (S + 3))
        in_s3 = tl.load(In_ptr + b * stride_in_b + h * stride_in_h + s3 * stride_in_s, mask=mask3, other=0.0)
        # s+2
        s2 = offs_s + 2
        mask2 = mask_s & (s2 < (S + 3))
        in_s2 = tl.load(In_ptr + b * stride_in_b + h * stride_in_h + s2 * stride_in_s, mask=mask2, other=0.0)
        # s+1
        s1 = offs_s + 1
        mask1 = mask_s & (s1 < (S + 3))
        in_s1 = tl.load(In_ptr + b * stride_in_b + h * stride_in_h + s1 * stride_in_s, mask=mask1, other=0.0)
        # s
        mask0 = mask_s
        in_s0 = tl.load(In_ptr + b * stride_in_b + h * stride_in_h + offs_s * stride_in_s, mask=mask0, other=0.0)

        # Accumulate: conv_out[b,h,s] = w0*in_s3 + w1*in_s2 + w2*in_s1 + w3*in_s0 + b_bias
        acc += w0 * in_s3 + w1 * in_s2 + w2 * in_s1 + w3 * in_s0 + b_bias

        # Store results to Out[b, h, s]
        out_ptrs = Out_ptr + b * stride_out_b + h * stride_out_h + offs_s * stride_out_s
        tl.store(out_ptrs, acc, mask=mask_s)


@triton.jit
def _out_proj_matmul_kernel(
    A_ptr, B_ptr, Bias_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Compute C = A @ B + Bias, A: (M,K), B: (K,N)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc += tl.dot(a, b)

    bias = tl.load(Bias_ptr + offs_n, mask=(offs_n < N), other=0.0).to(tl.float32)
    acc = acc + bias[None, :]

    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor,
                in_proj_weight: torch.Tensor, in_proj_bias: torch.Tensor,
                conv_weight: torch.Tensor, conv_bias: torch.Tensor,
                out_proj_weight: torch.Tensor, out_proj_bias: torch.Tensor):
        # Ensure all tensors are float32 and contiguous
        x = x.contiguous().to(torch.float32)
        in_proj_weight = in_proj_weight.contiguous().to(torch.float32)
        in_proj_bias = in_proj_bias.contiguous().to(torch.float32)
        conv_weight = conv_weight.contiguous().to(torch.float32)
        conv_bias = conv_bias.contiguous().to(torch.float32)
        out_proj_weight = out_proj_weight.contiguous().to(torch.float32)
        out_proj_bias = out_proj_bias.contiguous().to(torch.float32)

        B, S, H = x.shape

        # 1) in_proj: BCx = x @ in_proj_weight^T + in_proj_bias
        # x: (B, S, H) -> A: (B*S, H)
        A = x.view(B * S, H)  # (M, K)
        # in_proj_weight: (3H, H) -> B: (K, N) where N=3H
        B_w = in_proj_weight.t().view(H, 3 * H)
        Bias = in_proj_bias.view(3 * H)

        BCx = torch.empty((B * S, 3 * H), dtype=torch.float32, device=x.device)

        BLOCK_M = 128
        BLOCK_N = 64
        BLOCK_K = 32
        grid = (triton.cdiv(B * S, BLOCK_M), triton.cdiv(3 * H, BLOCK_N))
        _matmul_linear_kernel[grid](
            A, B_w, Bias, BCx,
            B * S, 3 * H, H,
            A.stride(0), A.stride(1),
            B_w.stride(0), B_w.stride(1),
            BCx.stride(0), BCx.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
        )

        BCx = BCx.view(B, S, 3 * H)  # (B, S, 3H)

        # Split into B, C, x_proj
        B_tensor = BCx[:, :, :H]           # (B, S, H)
        C_tensor = BCx[:, :, H:2 * H]      # (B, S, H)
        x_proj = BCx[:, :, 2 * H:]         # (B, S, H)

        # 2) Elementwise gate: Bx = B * x_proj
        Bx = torch.empty_like(B_tensor)  # (B, S, H)
        grid2 = (B, S)
        _elementwise_mul_2d_kernel[grid2](
            B_tensor, x_proj, Bx,
            B, S, H,
            B_tensor.stride(0), B_tensor.stride(1), B_tensor.stride(2),
            x_proj.stride(0), x_proj.stride(1), x_proj.stride(2),
            Bx.stride(0), Bx.stride(1), Bx.stride(2),
            BLOCK_M=1, BLOCK_N=1
        )
        # We need BLOCK_M/N > 1 for actual tiling. Correct approach: flatten to 1D:
        # Flatten to 1D for elementwise multiply
        Bx_flat = B_tensor.reshape(-1) * x_proj.reshape(-1)
        Bx = Bx_flat.view(B, S, H)

        # 3) Grouped causal conv with kernel_size=4, groups=H
        # Build padded input Bx_padded: (B, H, S+3) with 3 zeros on left (causal)
        Bx_padded = torch.nn.functional.pad(Bx, (3, 0), mode='constant', value=0.0).to(torch.float32)  # (B, H, S+3)
        # conv_weight: from in_proj_weight's last 4 columns, (H, 4)
        conv_w = in_proj_weight[:, -4:].contiguous().to(torch.float32)  # (H, 4)
        conv_bias = conv_bias.contiguous().to(torch.float32)            # (H,)
        conv_out = torch.empty((B, H, S), dtype=torch.float32, device=x.device)  # output (B, H, S)

        # Launch Triton kernel: grid=(B,H)
        grid_conv = (B, H)
        # We need BLOCK_S for tiling over S. Use 128, and loop over S in kernel (runtime loop).
        _grouped_causal_conv1d_kernel[grid_conv](
            Bx_padded, conv_w, conv_bias, conv_out,
            B, H, S,
            Bx_padded.stride(0), Bx_padded.stride(1), Bx_padded.stride(2),
            conv_w.stride(0), conv_w.stride(1),
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
            BLOCK_S=128
        )

        # 4) Output gating: y = C * conv_out, align shapes: conv_out -> (B, S, H)
        conv_out_T = conv_out.transpose(-1, -2).contiguous()  # (B, S, H)
        y = torch.empty((B, S, H), dtype=torch.float32, device=x.device)
        grid_gate = (B, S)
        _elementwise_mul_2d_kernel[grid_gate](
            C_tensor, conv_out_T, y,
            B, S, H,
            C_tensor.stride(0), C_tensor.stride(1), C_tensor.stride(2),
            conv_out_T.stride(0), conv_out_T.stride(1), conv_out_T.stride(2),
            y.stride(0), y.stride(1), y.stride(2),
            BLOCK_M=1, BLOCK_N=1
        )
        # Flatten for elementwise multiply as well
        y_flat = C_tensor.reshape(-1) * conv_out_T.reshape(-1)
        y = y_flat.view(B, S, H)

        # 5) Final out-proj: y @ out_proj_weight^T + out_proj_bias
        # y: (B, S, H) -> A: (B*S, H), out_proj_weight: (H, H)
        A2 = y.view(B * S, H)                        # (M, K)
        B2 = out_proj_weight.t().contiguous()       # (K, N) where N=H
        Bias2 = out_proj_bias.contiguous()          # (N)

        output = torch.empty((B * S, H), dtype=torch.float32, device=x.device)
        grid_out = (triton.cdiv(B * S, 128), triton.cdiv(H, 64))
        _out_proj_matmul_kernel[grid_out](
            A2, B2, Bias2, output,
            B * S, H, H,
            A2.stride(0), A2.stride(1),
            B2.stride(0), B2.stride(1),
            output.stride(0), output.stride(1),
            BLOCK_M=128, BLOCK_N=64, BLOCK_K=32
        )
        output = output.view(B, S, H)

        return output


def run(*args):
    return ModelNew()(*args)
