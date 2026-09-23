import math
import torch
import torch.nn as nn

# Triton is required
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: stride-2, padding-1 3x3 Conv2d with bias and fused GELU
# Input: x[N, Ci, Fi, Ti], weights[Cout, Ci, 3, 3], bias[Cout]
# Output: out[N, Cout, F_out, T_out]
@triton.jit
def conv2d_stride2_bias_gelu_kernel(
    x_ptr,                # *fp32
    weight_ptr,           # *fp32, shape [Cout, Ci, 3, 3]
    bias_ptr,             # *fp32, shape [Cout]
    out_ptr,              # *fp32, shape [N, Cout, F_out, T_out]
    N, Ci, Cout, Fi, Ti, F_out, T_out,
    x_sN, x_sCi, x_sFi, x_sTi,
    w_sC, w_sCi, w_sK, w_sL,     # weight strides
    out_sN, out_sC, out_sFo, out_sTo,
    embed_scale,           # scalar fp32
):
    pid_n = tl.program_id(0)  # batch
    pid_c = tl.program_id(1)  # output channel
    pid_f = tl.program_id(2)  # output f index
    pid_t = tl.program_id(3)  # output t index

    # bounds
    if (pid_n >= N) or (pid_c >= Cout) or (pid_f >= F_out) or (pid_t >= T_out):
        return

    # accumulate output
    acc = 0.0
    # For stride=2, padding=1, output dimensions: F_out = (Fi - 3)//2 + 1, T_out = (Ti - 3)//2 + 1
    # We sum over input channels Ci and 3x3 kernel
    for ic in range(0, Ci):
        for kh in range(0, 3):
            for kw in range(0, 3):
                fi = 2 * pid_f + 1 - kh  # from conv formula
                ti = 2 * pid_t + 1 - kw
                # check if (fi, ti) in bounds
                if (fi >= 0 and fi < Fi) and (ti >= 0 and ti < Ti):
                    x_off = pid_n * x_sN + ic * x_sCi + fi * x_sFi + ti * x_sTi
                    x_val = tl.load(x_ptr + x_off)  # fp32
                    acc += x_val

    # add bias
    b = tl.load(bias_ptr + pid_c)
    acc += b

    # GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))
    c0 = 0.7978845608028654  # sqrt(2/pi)
    x3 = acc * acc * acc
    gelu = 0.5 * acc * (1.0 + tl.tanh(c0 * (acc + 0.044715 * x3)))

    # scale
    gelu = gelu * embed_scale

    # store
    out_off = pid_n * out_sN + pid_c * out_sC + pid_f * out_sFo + pid_t * out_sTo
    tl.store(out_ptr + out_off, gelu)


# Triton kernel: batched GEMV for linear projection
# Input: X[N, T_out3, M], W[M, K] (conv_out_weight transposed)
# Output: Y[N, T_out3, K]
@triton.jit
def linear_gemv_kernel(
    X_ptr, W_ptr, Y_ptr,
    N, T_out3, M, K,
    X_sN, X_sT, X_sM,   # strides for X: [N, T_out3, M]
    W_sM, W_sK,         # strides for W: [M, K]
    Y_sN, Y_sT, Y_sK,   # strides for Y: [N, T_out3, K]
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)  # batch
    pid_t = tl.program_id(1)  # time index
    pid_k = tl.program_id(2)  # output channel index

    if (pid_n >= N) or (pid_t >= T_out3) or (pid_k >= K):
        return

    # Accumulate dot product over M in blocks
    acc = 0.0
    for m_start in range(0, M, BLOCK_K):
        k_ids = m_start + tl.arange(0, BLOCK_K)
        mask = k_ids < M
        # X[n, t, k] vector for these k_ids
        x_vals = tl.load(X_ptr + pid_n * X_sN + pid_t * X_sT + k_ids * X_sM, mask=mask, other=0.0)  # [BLOCK_K]
        # W[k, K] = conv_out_weight[j, k] where j corresponds to k_ids; but W is [M, K], so we access W[k_ids, pid_k]
        w_vals = tl.load(W_ptr + k_ids * W_sM + pid_k * W_sK, mask=mask, other=0.0)  # [BLOCK_K]
        acc += tl.sum(x_vals * w_vals, axis=0)

    # store
    y_off = pid_n * Y_sN + pid_t * Y_sT + pid_k * Y_sK
    tl.store(Y_ptr + y_off, acc)


# Triton kernel: elementwise scale and add positional embedding
# Input: X[N, T_out3, K], Pos[T_out3, K], Y[N, T_out3, K]
# Output: Y = X * scale + Pos
@triton.jit
def scale_add_pos_kernel(
    X_ptr, Pos_ptr, Y_ptr,
    N, T_out3, K,
    X_sN, X_sT, X_sK,
    Pos_sT, Pos_sK,
    Y_sN, Y_sT, Y_sK,
    scale,  # fp32 scalar
):
    pid_n = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_k = tl.program_id(2)

    if (pid_n >= N) or (pid_t >= T_out3) or (pid_k >= K):
        return

    x_off = pid_n * X_sN + pid_t * X_sT + pid_k * X_sK
    pos_off = pid_t * Pos_sT + pid_k * Pos_sK
    x_val = tl.load(X_ptr + x_off)
    pos_val = tl.load(Pos_ptr + pos_off)
    y_val = x_val * scale + pos_val
    y_off = pid_n * Y_sN + pid_t * Y_sT + pid_k * Y_sK
    tl.store(Y_ptr + y_off, y_val)


# Entry point: ModelNew.forward
class ModelNew(nn.Module):
    def forward(self, *args):
        # Unpack arguments exactly as original run() function
        # order: input_features, conv2d1_weight, conv2d1_bias, conv2d2_weight, conv2d2_bias, conv2d3_weight, conv2d3_bias, conv_out_weight, positional_embedding, embed_scale
        # The original run() uses defaults: input_features [N,1,80,time_dim], conv weights provided.
        # Here we assume the environment passes these 9+ tensors as in the original.
        # To keep it general, we construct them (as in the original get_inputs). However, the evaluator will supply inputs; we should not depend on get_inputs here.

        # We will perform all heavy computation via Triton kernels. We need to:
        # 1) Conv3 with stride=2, padding=1, bias, fused GELU
        # 2) Reshape to [N, T_out3, M] where M = C3 * F_out3 = 384 * ((F_out2 - 3)//2 + 1) = 384 * 8 = 3072
        # 3) Linear projection to d_model=1024 using GEMV
        # 4) Scale by embed_scale and add positional embedding

        # Extract tensors from args; args are already the tensors from original call site.
        # For clarity, let’s name them:
        input_features = args[0]  # [N, 1, 80, time_dim]
        conv2d1_weight = args[1]  # [384, 1, 3, 3]
        conv2d1_bias = args[2]    # [384]
        conv2d2_weight = args[3]  # [384, 384, 3, 3]
        conv2d2_bias = args[4]    # [384]
        conv2d3_weight = args[5]  # [384, 384, 3, 3]
        conv2d3_bias = args[6]    # [384]
        conv_out_weight = args[7] # [1024, 3072] (correct M=3072)
        positional_embedding = args[8]  # [max_time_after_conv, 1024]
        embed_scale = args[9]  # float

        assert TRITON_AVAILABLE, "Triton is not available"

        device = input_features.device
        # We will use float32 for computation in kernels for numerical stability
        # Convert inputs to float32; original pipeline uses bfloat16, but Triton kernels operate in fp32
        input_f32 = input_features.to(torch.float32)         # [N, 1, 80, time_dim]
        w1 = conv2d1_weight.to(torch.float32)               # [384, 1, 3, 3]
        b1 = conv2d1_bias.to(torch.float32)                 # [384]
        w2 = conv2d2_weight.to(torch.float32)               # [384, 384, 3, 3]
        b2 = conv2d2_bias.to(torch.float32)                 # [384]
        w3 = conv2d3_weight.to(torch.float32)               # [384, 384, 3, 3]
        b3 = conv2d3_bias.to(torch.float32)                 # [384]
        cout_w = conv_out_weight.to(torch.float32)          # [1024, 3072]
        pos_emb = positional_embedding.to(torch.float32)    # [T_out3, 1024]

        N = input_f32.shape[0]
        time_dim = input_f32.shape[3]

        # Dimensions for convs (stride=2, padding=1)
        # conv1: in (1), out (384), F_out = (80-3)//2 + 1 = 38, T_out = (time_dim-3)//2 + 1
        F1_out = (80 - 3) // 2 + 1  # 38
        T1_out = (time_dim - 3) // 2 + 1
        # conv2: in (384), out (384), F_out2 = (F1_out - 3)//2 + 1 = 17, T_out2 = (T1_out - 3)//2 + 1
        F2_out = (F1_out - 3) // 2 + 1  # 17
        T2_out = (T1_out - 3) // 2 + 1
        # conv3: in (384), out (384), F_out3 = (F2_out - 3)//2 + 1 = 8, T_out3 = (T2_out - 3)//2 + 1
        F3_out = (F2_out - 3) // 2 + 1  # 8
        T3_out = (T2_out - 3) // 2 + 1  # depends on input time_dim

        # Allocate conv3 output as float32 (we will compute conv3 via Triton kernel)
        C3 = 384
        M = C3 * F3_out  # 384 * 8 = 3072
        x3 = torch.empty((N, C3, F3_out, T3_out), device=device, dtype=torch.float32)

        # Launch Triton conv3 kernel: out = conv2d(x2, w3, b3, stride=2, pad=1) + GELU
        # First we need x2; conv2 needs conv1 output. We compute conv1 and conv2 via Triton similarly to avoid torch conv calls.
        # conv1 output: [N, 384, 38, T1_out


def run(*args):
    return ModelNew()(*args)
