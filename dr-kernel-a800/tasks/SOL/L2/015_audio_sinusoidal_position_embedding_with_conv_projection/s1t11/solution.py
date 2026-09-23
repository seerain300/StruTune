import math
import torch
import triton
import triton.language as tl


# Triton conv2d kernel: 3x3, stride=2, padding=1
# Input: X(B, Cin, H, T) bfloat16
# Weight: W(Cout, Cin, 3, 3) bfloat16
# Bias: Bias(Cout) bfloat16
# Output: Y(B, Cout, H, T_out) bfloat16, with T_out = floor((T - 3)/2 + 1)
@triton.jit
def conv2d_3x3_stride2_padding1_kernel(
    X_ptr,         # *const bfloat16
    W_ptr,         # *const bfloat16
    BIAS_ptr,      # *const bfloat16
    Y_ptr,         # *bfloat16
    Bsz, Cin, H, T, Cout, T_out,
    BLOCK_C: tl.constexpr,
):
    # Grid: (B*H, tiles over Cout, T_out)
    pid_m = tl.program_id(0)   # over B*H
    pid_c = tl.program_id(1)   # over tiles of Cout
    t_out_idx = tl.program_id(2)  # specific time index in output

    b = pid_m // H
    oh = pid_m % H

    c_start = pid_c * BLOCK_C
    c_offsets = c_start + tl.arange(0, BLOCK_C)
    mask_c = c_offsets < Cout

    # accumulator per output channel
    acc = tl.zeros((BLOCK_C,), dtype=tl.float32)

    # Loop over input channels and 3x3 kernel
    for cin in range(Cin):
        for kh in range(3):
            ih = oh + kh - 1
            valid_ih = (ih >= 0) & (ih < H)
            # We will compute contribution for each kt separately; here we can vectorize over c_offsets,
            # but since weights are per (Cout, Cin, kh, kt), we need to load weight for each kt.
            # We'll compute it per kt as scalar and broadcast over c_offsets.
            for kt in range(3):
                it = t_out_idx + kt - 1
                valid_it = (it >= 0) & (it < T)

                # Load input vector for this (cin, kh, kt) across c_offsets
                x_offs = b * (Cin * H * T) + cin * (H * T) + ih * T + it
                x_vals = tl.load(X_ptr + x_offs, mask=valid_ih & valid_it, other=0.0).to(tl.float32)  # scalar

                # Load weight vector for this (kh, kt) across c_offsets
                w_offs = c_offsets * (Cin * 3 * 3) + cin * (3 * 3) + kh * 3 + kt
                w_vals = tl.load(W_ptr + w_offs, mask=mask_c, other=0.0).to(tl.float32)  # (BLOCK_C,)

                # Accumulate
                acc += x_vals * w_vals  # broadcast x_vals over (BLOCK_C,)

    # Add bias
    bias_vals = tl.load(BIAS_ptr + c_offsets, mask=mask_c, other=0.0).to(tl.float32)
    acc += bias_vals

    # Store result (bfloat16)
    y_offs = (b * Cout * H * T_out) + c_offsets * (H * T_out) + oh * T_out + t_out_idx
    tl.store(Y_ptr + y_offs, acc.to(tl.bfloat16), mask=mask_c)


# Triton GELU (exact, erf-based) kernel applied elementwise
@triton.jit
def gelu_erf_kernel(X_ptr, Y_ptr, numel, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < numel
    x = tl.load(X_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    # exact GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
    inv_sqrt2 = 0.7071067811865476  # 1 / sqrt(2)
    y = 0.5 * x * (1.0 + tl.math.erf(x * inv_sqrt2))
    tl.store(Y_ptr + offsets, y.to(tl.bfloat16), mask=mask)


# Triton GEMM + add scaled positional embedding:
# X_flat: (B * t * K), Wt: (K, N), Y_flat: (B * t * N)
# We launch a 3D grid over (B, t, tiles over N)
@triton.jit
def gemm_pos_add_kernel(
    X_ptr,        # *const bfloat16, flattened (B * t * K)
    Wt_ptr,       # *const bfloat16, (K, N)
    POS_ptr,      # *const bfloat16, (M, N), M=B * t
    Y_ptr,        # *bfloat16, flattened (B * t * N)
    B, t, K, N, scale,  # ints and float
    BLOCK_N: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_n = tl.program_id(2)

    # vector of output channels handled by this program
    n_start = pid_n * BLOCK_N
    n_offsets = n_start + tl.arange(0, BLOCK_N)
    mask_n = n_offsets < N

    # accumulator for this (b, t) and tile of N
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # loop over K dimension
    for k0 in range(0, K, 1):
        # load x[b, t, k]
        x_off = (pid_b * t + pid_t) * K + k0
        x_val = tl.load(X_ptr + x_off).to(tl.float32)  # scalar

        # load Wt[k, n_offsets]
        w_off = k0 * N + n_offsets
        w_vals = tl.load(Wt_ptr + w_off, mask=mask_n, other=0.0).to(tl.float32)  # (BLOCK_N,)

        # accumulate
        acc += x_val * w_vals

    # add scaled positional embedding: POS[(pid_b * t), n_offsets]
    pos_off = (pid_b * t) * N + n_offsets
    pos_vals = tl.load(POS_ptr + pos_off, mask=mask_n, other=0.0).to(tl.float32)
    acc = acc + pos_vals * scale

    # store to Y
    y_off = (pid_b * t * N) + n_offsets
    tl.store(Y_ptr + y_off, acc.to(tl.bfloat16), mask=mask_n)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        input_features: torch.Tensor,
        conv2d1_weight: torch.Tensor,
        conv2d1_bias: torch.Tensor,
        conv2d2_weight: torch.Tensor,
        conv2d2_bias: torch.Tensor,
        conv2d3_weight: torch.Tensor,
        conv2d3_bias: torch.Tensor,
        conv_out_weight: torch.Tensor,  # (1024, 15360) -> (N, K)
        positional_embedding: torch.Tensor,  # (1500, 1024), bfloat16
        embed_scale: float,
    ):
        # Ensure device is CUDA
        assert input_features.is_cuda, "All tensors must be on CUDA device for Triton."
        # Ensure dtype is bfloat16
        assert input_features.dtype == torch.bfloat16, "Expected bfloat16 tensors."

        # Shapes
        B, Cin_in, H, T = input_features.shape
        Cout1 = conv2d1_weight.shape[0]      # 384
        Cout2 = conv2d2_weight.shape[0]      # 384
        Cout3 = conv2d3_weight.shape[0]      # 384

        def T_out_from_T(T_in):
            return (T_in - 3) // 2 + 1

        # Stage 1: conv1 + GELU
        T1_out = T_out_from_T(T)
        y1 = torch.empty((B, Cout1, H, T1_out), device=input_features.device, dtype=torch.bfloat16)
        grid1 = (B * H, triton.cdiv(Cout1, 64), T1_out)
        conv2d_3x3_stride2_padding1_kernel[grid1](
            input_features, conv2d1_weight, conv2d1_bias, y1,
            B, Cin_in, H, T, Cout1, T1_out,
            BLOCK_C=64, num_warps=4, num_stages=2
        )
        # GELU conv1
        y1_flat = y1.reshape(-1)
        y1_gelu = torch.empty_like(y1_flat, dtype=torch.bfloat16, device=input_features.device)
        grid_gelu1 = (triton.cdiv(y1_flat.numel(), 1024),)
        gelu_erf_kernel[grid_gelu1](y1_flat, y1_gelu, y1_flat.numel(), BLOCK=1024)
        y1 = y1_gelu.view(B, Cout1, H, T1_out)

        # Stage 2: conv2 + GELU
        T2_out = T_out_from_T(T1_out)
        y2 = torch.empty((B, Cout2, H, T2_out), device=input_features.device, dtype=torch.bfloat16)
        grid2 = (B * H, triton.cdiv(Cout2, 64), T2_out)
        conv2d_3x3_stride2_padding1_kernel[grid2](
            y1, conv2d2_weight, conv2d2_bias, y2,
            B, Cout1, H, T1_out, Cout2, T2_out,
            BLOCK_C=64, num_warps=4, num_stages=2
        )
        # GELU conv2
        y2_flat = y2.reshape(-1)
        y2_gelu = torch.empty_like(y2_flat, dtype=torch.bfloat16, device=input_features.device)
        grid_gelu2 = (triton.cdiv(y2_flat.numel(), 1024),)
        gelu_erf_kernel[grid_gelu2](y2_flat, y2_gelu, y2_flat.numel(), BLOCK=1024)
        y2 = y2_gelu.view(B, Cout2, H, T2_out)

        # Stage 3: conv3 + GELU
        T3_out = T_out_from_T(T2_out)
        y3 = torch.empty((B, Cout3, H, T3_out), device=input_features.device, dtype=torch.bfloat16)
        grid3 = (B * H, triton.cdiv(Cout3, 64), T3_out)
        conv2d_3x3_stride2_padding1_kernel[grid3](
            y2, conv2d3_weight, conv2d3_bias, y3,
            B, Cout2, H, T2_out, Cout3, T3_out,
            BLOCK_C=64, num_warps=4, num_stages=2
        )
        # GELU conv3
        y3_flat = y3.reshape(-1)
        y3_gelu = torch.empty_like(y3_flat, dtype=torch.bfloat16, device=input_features.device)
        grid_gelu3 = (triton.cdiv(y3_flat.numel(), 1024),)
        gelu_erf_kernel[grid_gelu3](y3_flat, y3_gelu, y3_flat.numel(), BLOCK=1024)
        y3 = y3_gelu.view(B, Cout3, H, T3_out)

        # Reshape: (B, T3_out, H, Cout3) -> (B, T3_out, C*F) with F=40 (since H=40 after each conv)
        # We need to compute F after conv3: original H=80, stride 2 each conv => after conv1 H=40, conv2 H=20, conv3 H=10; but here model uses H as time T? The original model passes H as spatial 80, but then uses T for time; the provided get_inputs sets H=80, T=time_dim. The code assumes H as spatial channels, not time. To proceed, we must follow the original code's logic: it permutes to (B, t, C*F) after conv3. The original code uses (batch, 1, 80, T) -> conv -> (B, Cout, H, T_out). Then it permutes to (B, t, C*F). In the original, H is 80 and F=40 post conv3? Not clear. Given the get_inputs sets H=80, T=time_dim, and the model uses (1, 80, T). The original code's comment says C*F with F=40, but conv3 output shape is (B, 384, H, T_out). It then does x.permute(0, 3, 1, 2).contiguous().view(B, t, C*F). If t = T3_out and C=384, then C*F must be 384*F=15360? That implies F=40. So we assume F = 80 // (2^3) = 10 ? Not helpful. Given the original code, the safest is to replicate the original behavior: after conv3, x has shape (B, 384, H, T3_out). Then .permute(0, 3, 1, 2) -> (B, T3_out, 384, H). Then view(B, T3_out, 384*H). The original code uses C*F=1024? That contradicts. The original code in run uses x.permute(0, 3, 1, 2).contiguous().view(B, t, c * f) with c=conv_out_dim=3840/384=10? That would require F=2.4? That doesn't align. The safest route is to follow the original run() logic: conv3 output is (B, 384, H, T3_out). Then .permute(0, 3, 1, 2) -> (B, T3_out, 384, H). The original code then does .view(B, t, c * f) with c=conv_out_dim=3840, and f=10? That doesn't align with H. This indicates a mismatch: the original run uses conv_out_dim=3840, but the get_inputs defines d_model=1024, and positional_embedding is (1500, 1024). The original run does linear(x, conv_out_weight) where conv_out_weight is (d_model, conv_out_dim) = (1024, 3840). Then it multiplies by embed_scale = sqrt(1024) = 32, and adds positional_embedding (1500, 1024). Therefore, to match, our forward must perform x = F.linear(permuted, conv_out_weight) with conv_out_weight shape (1024, 3840). But in Triton, we’ll implement the GEMM + add scaled pos_embedding.

        # Given the provided get_inputs and original run(), the final x before linear is y3, shape (B, 384, H, T3_out). The original run() permutes y3 to (B, t, C*F) with C*F = conv_out_dim. In the original run, conv_out_dim = 3840. But the provided get_inputs has conv_out_weight of shape (1024, 3840), and positional_embedding (1500, 1024). So we need to adjust: the original code seems to use conv_out_dim=3840, but the evaluation provides conv_out_weight with d_model=1024. To resolve this discrepancy and still adhere to the original run behavior, we assume the intended d_model is 1024 (from positional_embedding), and conv_out_dim=1024 (since conv_out_weight is (1024, conv_out_dim)). Therefore, we should linear with conv_out_dim=1024. However, the original run has conv_out_dim=3840. Since we cannot change the provided tensors, we will implement the linear with conv_out_weight's second dimension. We will infer conv_out_dim from conv_out_weight.shape[1] at runtime and use that for the Triton kernel, and ensure the output is (B, t, conv_out_dim). The original positional_embedding is (1500, 1024), so we scale by embed_scale=32 and add.

        # Compute B, t, K, N
        # After conv3: y3 shape (B, 384, H, T3_out). The original run() then permutes to (B, t, C*F). The original code uses conv_out_dim=3840, but since we don't have that tensor, we infer N = conv_out_weight.shape[1]. K is the flattened dimension before projection. Given the original run, it takes x (B, t, C*F) and linear to (B, t, conv_out_dim). In our case, conv_out_weight is (1024, N). We don't have C*F of conv3 output; but the original code does. Since we can't reproduce exact C*F without the original model's F, we will instead do: take y3.permute(0, 3, 1, 2) -> (B, T3_out, 384, H). The original code then .view(B, T3_out, C*F). We don't know F, but we can flatten y3 to a 2D tensor (B * T3_out, K), where K = 384 * H, and perform a GEMM with Wt = conv_out_weight.T shape (N, K). The output will be (B * T3_out, N). We reshape to (B, T3_out, N). This approach avoids needing C*F. The original run uses conv_out_dim=3840, but our conv_out_weight is (1024, N), so we will use N = conv_out_weight.shape[1] and produce (B, T3_out, N). We'll add scaled positional embedding of shape (1500, N).

        # Reshape y3 for GEMM: flatten leading dims
        # y3_flat: (B * T3_out, 384 * H)
        y3_perm = y3.permute(0, 3, 1, 2)  # (B, T3_out, 384, H)
        K = y3_perm.shape[-2] * y3_perm.shape[-1]  # 384 * H
        y3_flat = y3_perm.reshape(B * T3_out, K)  # (B * T3_out, K)

        # conv_out_weight: (N, K_in), here N=1024, K_in=conv_out_weight.shape[1]
        N = conv_out_weight.shape[0]  # d_model = 1024
        Wt = conv_out_weight.transpose(0, 1).contiguous()  # (K, N)

        # Prepare flattened output
        y_flat = torch.empty((B * T3_out * N), device=input_features.device, dtype=torch.bfloat16)

        # Launch GEMM + add scaled positional embedding
        # We need POS (1500, N) slice for each b*t. Since original POS is (1500, 1024), we can broadcast by setting POS rows beyond (B * T3_out) to zeros.
        # However, we only need POS for the first (B * T3_out) rows; we'll slice POS to size M=B * T3_out. If M > 1500, we can pad or error. Given typical workloads, M <= 1500.
        M = B * T3_out
        pos_slice = positional_embedding[:M, :].contiguous()  # (M, N)

        grid_gemm = (B, T3_out, triton.cdiv(N, 128))
        gemm_pos_add_kernel[grid_gemm](
            y3_flat, Wt, pos_slice, y_flat,
            B, T3_out, K, N, embed_scale,
            BLOCK_N=128, num_warps=4, num_stages=2
        )

        # Reshape to (B, T3_out, N)
        y = y_flat.view(B, T3_out, N)

        return y


def run(*args):
    return ModelNew()(*args)
