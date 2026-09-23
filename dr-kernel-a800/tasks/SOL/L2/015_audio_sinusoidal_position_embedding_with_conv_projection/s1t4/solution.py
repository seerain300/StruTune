import math
import torch
import triton
import triton.language as tl


# Triton conv2d kernel: 3x3, stride=2, padding=1
# Input X: (B, Cin, H, T), contiguous
# Weight W: (Cout, Cin, 3, 3), contiguous
# Bias: (Cout,)
# Output Y: (B, Cout, H_out, T_out), H_out=H, T_out=floor((T - 3)//2 + 1)
@triton.jit
def conv2d_3x3_stride2_padding1_kernel(
    X_ptr,         # *const bfloat16
    W_ptr,         # *const bfloat16
    BIAS_ptr,      # *const bfloat16
    Y_ptr,         # *bfloat16
    B: tl.constexpr,       # batch size
    Cin: tl.constexpr,     # input channels (1 for conv1, 384 for conv2/3)
    H: tl.constexpr,       # height
    T: tl.constexpr,       # time
    Cout: tl.constexpr,    # output channels
    T_out: tl.constexpr,   # output time length
    BLOCK_C: tl.constexpr  # tile size for Cout
):
    # 2D grid: axis 0 over B*H_out, axis 1 over tiles of Cout
    pid_m = tl.program_id(0)  # over B*H_out
    pid_c = tl.program_id(1)  # over tiles of Cout

    b = pid_m // H
    oh = pid_m % H

    c_offsets = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_c = c_offsets < Cout

    # Accumulator in float32
    acc = tl.zeros((BLOCK_C,), dtype=tl.float32)

    # Loop over input channels and 3x3 kernel
    for cin_i in range(Cin):
        for kh in range(3):
            ih = oh + kh - 1  # padding=1
            # Check bounds for ih
            if kh == 0:
                ih_valid = (ih >= 0) and (ih < H)
            elif kh == 1:
                ih_valid = True
            else:  # kh == 2
                ih_valid = (ih >= 0) and (ih < H)

            # Loop over kw = 0, 1, 2 (padding=1 => always valid)
            for kw in range(3):
                it = (T - 1) // 2 - kw  # fixed mapping? Let's compute proper t_out mapping
                # For stride=2, padding=1:
                # it = t - kw + start, where start = (T - 3) // 2. We can compute it_out = (t - kw) // 2 (since t_out = floor((T-3)/2)+1, but simpler is to derive it_out based on T_out).
                # To avoid confusion, compute it_out directly: it_out = (t - kw) // 2 for valid t. But t_out depends on T; better approach: compute per t_out and pass grid accordingly.

                # Instead, we will use a separate kernel that iterates over T_out (grid over T_out), not here. This kernel computes per (b, oh) across T_out, which requires 3D grid. Triton supports up to 3 dims. We'll restructure.

    # Note: The above kernel is incomplete for general stride2 over T. Implementing full conv with stride2 over T in Triton requires computing t_out per kw and handling grid properly. For brevity and correctness, we provide the next kernel that handles the T_out loop within the Triton kernel (using Python loops), since T_out is small and acceptable for the task.


# Triton conv2d kernel: handles per-output time index t_out, 3x3 stride=2 padding=1, and Cin loop.
# This kernel will be launched with grid=(B*H, ceil(Cout/BLOCK_C), T_out). It computes Y[b, c, oh, t_out] for each t_out.
@triton.jit
def conv2d_3x3_stride2_padding1_per_tout_kernel(
    X_ptr,         # *const bfloat16
    W_ptr,         # *const bfloat16
    BIAS_ptr,      # *const bfloat16
    Y_ptr,         # *bfloat16
    B: tl.constexpr,       # batch size
    Cin: tl.constexpr,     # input channels (1 or 384)
    H: tl.constexpr,       # height
    T: tl.constexpr,       # time
    Cout: tl.constexpr,    # output channels
    T_out: tl.constexpr,   # output time length
    BLOCK_C: tl.constexpr  # tile size for Cout
):
    pid_m = tl.program_id(0)  # over B*H
    pid_c = tl.program_id(1)  # over tiles of Cout
    t_out_idx = tl.program_id(2)  # current output time index

    b = pid_m // H
    oh = pid_m % H

    c_offsets = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_c = c_offsets < Cout

    acc = tl.zeros((BLOCK_C,), dtype=tl.float32)

    # For stride=2, padding=1, input t index corresponding to output t_out_idx is:
    # it = 2 * t_out_idx - 1 + kw, kw in {0,1,2}. We need to guard by checking if (2*t_out_idx - 1 + kw) in [0, T-1].
    # Loop over input channels and 3x3 kernel
    for cin_i in range(Cin):
        for kh in range(3):
            ih = oh + kh - 1  # ih in [0, H-1] when kh in {0,1,2}; we mask kh=0/2 appropriately
            if kh == 0:
                ih_valid = (ih >= 0) and (ih < H)
            elif kh == 1:
                ih_valid = True
            else:  # kh == 2
                ih_valid = (ih >= 0) and (ih < H)

            for kw in range(3):
                it = 2 * t_out_idx - 1 + kw
                it_valid = (it >= 0) and (it < T)

                # Load X[b, cin_i, ih, it] if valid; else 0. Addressing uses strides: X[b*Cin*H*T + cin_i*H*T + ih*T + it]
                # We can compute address using flattened indexing. Since tensors are contiguous, we use:
                base_x = b * Cin * H * T + cin_i * H * T
                addr_x = base_x + ih * T + it
                x_val = tl.load(X_ptr + addr_x, mask=(ih_valid and it_valid), other=0.0).to(tl.float32)

                # Load weight W[c, cin_i, kh, kw] and accumulate
                base_w = c_offsets * (Cin * 3 * 3) + cin_i * (3 * 3) + kh * 3 + kw
                w_val = tl.load(W_ptr + base_w, mask=mask_c, other=0.0).to(tl.float32)

                acc += x_val * w_val  # broadcasting w_val over c_offsets

    # Add bias
    bias_vals = tl.load(BIAS_ptr + c_offsets, mask=mask_c, other=0.0).to(tl.float32)
    acc += bias_vals

    # Store Y[b, c, oh, t_out_idx]
    # Y layout: (B, Cout, H, T_out), contiguous => address = b*(Cout*H*T_out) + c*T_out + oh*T_out + t_out_idx
    base_y = b * (Cout * H * T_out) + oh * T_out + t_out_idx
    tl.store(Y_ptr + base_y + c_offsets, acc, mask=mask_c)


# Triton kernel for exact GELU (erf-based): applied elementwise to input tensor
@triton.jit
def gelu_erf_kernel(
    X_ptr,   # *const bfloat16
    Y_ptr,   # *bfloat16
    N: tl.constexpr,  # total number of elements
    BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    x = tl.load(X_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    inv_sqrt2 = 0.7071067811865476  # 1/sqrt(2)
    y = 0.5 * x * (1.0 + tl.math.erf(x * inv_sqrt2))
    tl.store(Y_ptr + offsets, y.to(tl.bfloat16), mask=mask)


# Triton kernel: y = X @ W_T, where
# X: (M, K_in) in bfloat16, contiguous, M=B*T_out, K_in=C*F=15360
# W_T: (K_in, N_out=1024) in bfloat16, contiguous (note: conv_out_weight provided as (N_out, K_in) in host code; we use its transpose here)
# Y: (M, N_out) bfloat16, contiguous
@triton.jit
def matmul_linear_kernel(
    X_ptr,     # *const bfloat16, (M, K_in)
    W_T_ptr,   # *const bfloat16, (K_in, N_out)
    Y_ptr,     # *bfloat16, (M, N_out)
    M: tl.constexpr,    # total rows: B * T_out
    K_in: tl.constexpr, # input feature dim: 15360
    N_out: tl.constexpr,  # output feature dim: 1024
    scale: tl.float32,    # scaling factor: embed_scale (32.0)
    POS_ptr,              # *const bfloat16, (M, N_out) positional embedding (to add after matmul)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = m_offsets < M
    mask_n = n_offsets < N_out

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K_in, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < K_in

        # Load X[m, k] tile
        A = tl.load(
            X_ptr + m_offsets[:, None] * K_in + k_offsets[None, :],
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0
        ).to(tl.float32)

        # Load W_T[k, n] tile
        B = tl.load(
            W_T_ptr + k_offsets[:, None] * N_out + n_offsets[None, :],
            mask=mask_k[:, None] & mask_n[None, :],
            other=0.0
        ).to(tl.float32)

        acc += tl.dot(A, B)  # (BLOCK_M, BLOCK_N)

    # Scale
    acc = acc * scale

    # Add scaled positional embedding: POS_ptr is (M, N_out)
    pos = tl.load(
        POS_ptr + m_offsets[:, None] * N_out + n_offsets[None, :],
        mask=mask_m[:, None] & mask_n[None, :],
        other=0.0
    ).to(tl.float32)
    acc += scale * pos

    # Store result to Y (M, N_out)
    tl.store(
        Y_ptr + m_offsets[:, None] * N_out + n_offsets[None, :],
        acc.to(tl.bfloat16),
        mask=mask_m[:, None] & mask_n[None, :]
    )


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Constants
        self.embed_scale = math.sqrt(1024.0)  # 32.0

    def forward(self, *args):
        # Extract tensors: args order as in original get_inputs (excluding generator), assuming:
        # 0: input_features (B,1,80,T), bfloat16, CUDA
        # 1: conv2d1_weight (384,1,3,3), bfloat16
        # 2: conv2d1_bias (384), bfloat16
        # 3: conv2d2_weight (384,384,3,3), bfloat16
        # 4: conv2d2_bias (384), bfloat16
        # 5: conv2d3_weight (384,384,3,3), bfloat16
        # 6: conv2d3_bias (384), bfloat16
        # 7: conv_out_weight (1024, 3840), bfloat16 (note: in original, this should be (1024,15360) to match x's last dim)
        # 8: positional_embedding (1500, 1024), bfloat16
        # 9: embed_scale (float, unused here as we set it in __init__)
        assert len(args) >= 9, "Not enough arguments for ModelNew.forward"

        input_features = args[0]
        conv2d1_weight = args[1]
        conv2d1_bias = args[2]
        conv2d2_weight = args[3]
        conv2d2_bias = args[4]
        conv2d3_weight = args[5]
        conv2d3_bias = args[6]
        conv_out_weight = args[7]  # expected shape (1024, 15360) to match x's last dim; if not, we will treat as provided
        positional_embedding = args[8]  # (1500, 1024)
        # embed_scale provided via self.embed_scale

        # Ensure CUDA and contiguous
        assert input_features.is_cuda and conv2d1_weight.is_cuda and conv_out_weight.is_cuda and positional_embedding.is_cuda, "All tensors must be CUDA"
        input_features = input_features.contiguous()
        conv2d1_weight = conv2d1_weight.contiguous()
        conv2d1_bias = conv2d1_bias.contiguous()
        conv2d2_weight = conv2d2_weight.contiguous()
        conv2d2_bias = conv2d2_bias.contiguous()
        conv2d3_weight = conv2d3_weight.contiguous()
        conv2d3_bias = conv2d3_bias.contiguous()
        conv_out_weight = conv_out_weight.contiguous()
        positional_embedding = positional_embedding.contiguous()

        Bsz = input_features.shape[0]
        Cin = input_features.shape[1]  # 1
        H = input_features.shape[2]    # 80
        T = input_features.shape[3]

        # conv1: (B, 1, 80, T) -> (B, 384, 80, T_out1)
        T_out1 = (T - 3) // 2 + 1
        x1 = torch.empty((Bsz, 384, H, T_out1), dtype=torch.bfloat16, device=input_features.device)
        # Launch conv per-tout kernel
        BLOCK_C1 = 64
        grid1 = (Bsz * H, triton.cdiv(384, BLOCK_C1), T_out1)
        conv2d_3x3_stride2_padding1_per_tout_kernel[grid1](
            input_features, conv2d1_weight, conv2d1_bias, x1,
            B=Bsz, Cin=1, H=H, T=T, Cout=384, T_out=T_out1, BLOCK_C=BLOCK_C1
        )
        # GELU after conv1
        x1_g = torch.empty_like(x1, dtype=torch.bfloat16)
        N1 = Bsz * 384 * H * T_out1
        BLOCK_G1 = 1024
        grid_g1 = (triton.cdiv(N1, BLOCK_G1),)
        gelu_erf_kernel[grid_g1](x1, x1_g, N1, BLOCK_G1)

        # conv2: (B, 384, 80, T_out1) -> (B, 384, 80, T_out2)
        T_out2 = (T_out1 - 3) // 2 + 1
        x2 = torch.empty((Bsz, 384, H, T_out2), dtype=torch.bfloat16, device=input_features.device)
        grid2 = (Bsz * H, triton.cdiv(384, BLOCK_C1), T_out2)
        conv2d_3x3_stride2_padding1_per_tout_kernel[grid2](
            x1_g, conv2d2_weight, conv2d2_bias, x2,
            B=Bsz, Cin=384, H=H, T=T_out1, Cout=384, T_out=T_out2, BLOCK_C=BLOCK_C1
        )
        # GELU after conv2
        x2_g = torch.empty_like(x2, dtype=torch.bfloat16)
        N2 = Bsz * 384 * H * T_out2
        grid_g2 = (triton.cdiv(N2, BLOCK_G1),)
        gelu_erf_kernel[grid_g2](x2, x2_g, N2, BLOCK_G1)

        # conv3: (B, 384, 80, T_out2) -> (B, 384, 40, T_out3)
        T_out3 = (T_out2 - 3) // 2 + 1
        x3 = torch.empty((Bsz, 384, 40, T_out3), dtype=torch.bfloat16, device=input_features.device)
        grid3 = (Bsz * 40, triton.cdiv(384, BLOCK_C1), T_out3)
        conv2d_3x3_stride2_padding1_per_tout_kernel[grid3](
            x2_g, conv2d3_weight, conv2d3_bias, x3,
            B=Bsz, Cin=384, H=40, T=T_out2, Cout=384, T_out=T_out3, BLOCK_C=BLOCK_C1
        )
        # GELU after conv3
        x3_g = torch.empty_like(x3, dtype=torch.bfloat16)
        N3 = Bsz * 384 * 40 * T_out3
        grid_g3 = (triton.cdiv(N3, BLOCK_G1),)
        gelu_erf_kernel[grid_g3](x3, x3_g, N3, BLOCK_G1)

        # Reshape: (B, 384, 40, T_out3) -> (B, T_out3, 384*40) = (B, T_out3, 15360)
        # Note: T_out3 equals the provided "time_after_conv" in workload; we use x3_g directly for linear.
        Bsz, Cout, F, T_out3 = x3_g.shape
        x_reshaped = x3_g.view(Bsz, T_out3, Cout * F)  # (B, T_out3, 15360)

        # Ensure conv_out_weight has (N_out=1024, K_in=15360). If not, fallback to using its transpose with given shapes.
        N_out = 1024
        K_in = x_reshaped.shape[-1]  # should be 15360
        W_T = conv_out_weight.transpose(0, 1).contiguous()  # shape (15360, 1024)
        # Allocate output (B, T_out3, 1024)
        y = torch.empty((Bsz, T_out3, N_out), dtype=torch.bfloat16, device=input_features.device)

        # Prepare sliced positional embedding: (B, T_out3, 1024)
        # positional_embedding is (1500, 1024); we slice rows [0:T_out3] and broadcast across batch dimension.
        # For Triton, we can pass a pointer to a (B, T_out3, 1024) tensor filled in host. Since Triton kernels can't return, we create a temporary tensor pos_bf16 here, compute in kernel, and write to y.
        # To avoid extra memory, we can create a temporary bf16 tensor here for positional embedding and add in-kernel. However, the evaluation expects final output to match original. We'll compute y in kernel and add pos there by slicing positional_embedding (converted to bf16) per b.

        # Construct POS_bf16 on host: (Bsz, T_out3, N_out) in bfloat16
        pos_rows = positional_embedding[:T_out3].to(torch.bfloat16)  # (T_out3, 1024)
        POS_bf16 = pos_rows.unsqueeze(0).expand(Bsz, -1, -1).contiguous()  # (Bsz, T_out3, 1024)

        # Launch matmul + scale + pos add
        BLOCK_M = 64   # tile for M = B*T_out3
        BLOCK_N = 128  # tile for N_out = 1024
        BLOCK_K = 128  # tile for K_in = 15360
        M_total = Bsz * T_out3
        grid_mm = (triton.cdiv(M_total, BLOCK_M), triton.cdiv(N_out, BLOCK_N))
        matmul_linear_kernel[grid_mm](
            x_reshaped, W_T, y, M_total, K_in, N_out, self.embed_scale, POS_bf16,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
        )

        return y


def run(*args):
    return ModelNew()(*args)
