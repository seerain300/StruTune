import math
import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: Conv2d 3x3, stride=2, padding=1, generic IC -> OC
# X: [B, IC, F_in, T_in], W: [OC, IC, 3, 3], bias: [OC], Y: [B, OC, F_out, T_out]
@triton.jit
def conv3x3_s2_p1_gelu(
    X_ptr, W_ptr, BIAS_ptr, Y_ptr,
    B, IC, F_in, T_in, OC, F_out, T_out,
    x_sN, x_sC, x_sF, x_sT,
    w_sOC, w_sIC, w_sKH, w_sKW,
    y_sN, y_sOC, y_sF, y_sT,
):
    # 1D launch per (b, oc, f_out, t_out)
    pid = tl.program_id(0)
    b = pid // (OC * F_out * T_out)
    rem = pid % (OC * F_out * T_out)
    oc = rem // (F_out * T_out)
    f_out = rem // T_out
    t_out = rem % T_out

    # Accumulator
    acc = tl.zeros((), dtype=tl.float32)  # scalar accumulation per output

    # Loop over input channels and 3x3 kernel
    for ic in range(0, IC):
        for kh in range(0, 3):
            for kw in range(0, 3):
                f_in = f_out * 2 + 1 - kh
                t_in = t_out * 2 + 1 - kw
                in_bounds = (f_in >= 0) & (f_in < F_in) & (t_in >= 0) & (t_in < T_in)
                x_val = tl.load(X_ptr + b * x_sN + ic * x_sC + f_in * x_sF + t_in * x_sT, mask=in_bounds, other=0.0)
                # load weight scalar
                w_val = tl.load(W_ptr + oc * w_sOC + ic * w_sIC + kh * w_sKH + kw * w_sKW)
                acc += x_val * w_val

    # Add bias
    b_val = tl.load(BIAS_ptr + oc)
    acc += b_val

    # GELU (tanh approximation)
    # gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    c = 0.7978845608028654  # sqrt(2/pi)
    x3 = acc * acc * acc
    inner = c * (acc + 0.044715 * x3)
    gelu = 0.5 * acc * (1.0 + tl.tanh(inner))

    # Store
    tl.store(Y_ptr + b * y_sN + oc * y_sOC + f_out * y_sF + t_out * y_sT, gelu)


# Triton matmul kernel: X [M, K] @ W^T [K, N] -> Y [M, N], here M=B*T3, K=3840, N=1024
@triton.jit
def matmul1024x3840x3840(X_ptr, W_ptr, Y_ptr,
                         M, K, N,
                         x_sM, x_sK,
                         w_sK, w_sN,
                         y_sM, y_sN):
    # one program per output element (m, n)
    pid = tl.program_id(0)
    m = pid // N
    n = pid % N
    acc = tl.zeros((), dtype=tl.float32)
    # reduce over K
    for k in range(0, K):
        x_val = tl.load(X_ptr + m * x_sM + k * x_sK)
        w_val = tl.load(W_ptr + k * w_sK + n * w_sN)
        acc += x_val * w_val
    tl.store(Y_ptr + m * y_sM + n * y_sN, acc)


# Triton elementwise kernel: scale and add positional embedding
# X: [B, T3, D], POS: [T3, D]
@triton.jit
def add_scale_and_pos_embed(
    X_ptr, POS_ptr, Y_ptr,
    B, T3, D,
    x_sB, x_sT, x_sD,
    pos_sT, pos_sD,
    y_sB, y_sT, y_sD,
    SCALE: tl.constexpr,
):
    pid = tl.program_id(0)
    total = B * T3 * D
    if pid >= total:
        return
    b = pid // (T3 * D)
    rem = pid % (T3 * D)
    t = rem // D
    d = rem % D
    x_val = tl.load(X_ptr + b * x_sB + t * x_sT + d * x_sD)
    pos_val = tl.load(POS_ptr + t * pos_sT + d * pos_sD)
    y_val = x_val * SCALE + pos_val
    tl.store(Y_ptr + b * y_sB + t * y_sT + d * y_sD, y_val)


# Helper to run conv stage via Triton
def conv3x3_s2_p1_gelu_launch(input_tensor, weight, bias, out_tensor):
    # input_tensor: [B, IC, F_in, T_in], float32
    B, IC, F_in, T_in = input_tensor.shape
    OC, IC_w, K_h, K_w = weight.shape
    assert IC_w == IC, "Weight IC must match input IC"
    assert K_h == 3 and K_w == 3, "Kernel must be 3x3"
    F_out = (F_in + 2 * 1 - 3) // 2 + 1
    T_out = (T_in + 2 * 1 - 3) // 2 + 1
    # Launch grid: one program per (b, oc, f_out, t_out)
    grid = (B * OC * F_out * T_out,)
    conv3x3_s2_p1_gelu[grid](
        input_tensor, weight, bias, out_tensor,
        B, IC, F_in, T_in, OC, F_out, T_out,
        input_tensor.stride(0), input_tensor.stride(1), input_tensor.stride(2), input_tensor.stride(3),
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3),
        out_tensor.stride(0), out_tensor.stride(1), out_tensor.stride(2), out_tensor.stride(3),
    )


# Model entry point: Triton-only implementation
class ModelNew(nn.Module):
    def forward(self, *args):
        # Extract arguments as in the original Model.forward signature
        # Note: args[0] is input_features: [B, 1, 80, time_dim]
        # Args 1-6: conv weights and biases; 7: conv_out_weight; 8: positional_embedding; 9: embed_scale
        input_features = args[0].contiguous().float()  # [B, 1, 80, T]
        conv2d1_weight = args[1].contiguous().float()  # [384, 1, 3, 3]
        conv2d1_bias = args[2].contiguous().float()   # [384]
        conv2d2_weight = args[3].contiguous().float() # [384, 384, 3, 3]
        conv2d2_bias = args[4].contiguous().float()   # [384]
        conv2d3_weight = args[5].contiguous().float() # [384, 384, 3, 3]
        conv2d3_bias = args[6].contiguous().float()   # [384]
        conv_out_weight = args[7].contiguous().float()  # [1024, 3840]
        positional_embedding = args[8]                 # [max_source_positions, 1024]
        embed_scale = float(args[9])                   # python float

        # Stage 1: Conv2d (1 -> 384 channels) + GELU
        B, _, F_in, T_in = input_features.shape
        y1 = torch.empty((B, 384, (F_in+2-3)//2+1, (T_in+2-3)//2+1), device=input_features.device, dtype=torch.float32)
        conv3x3_s2_p1_gelu_launch(input_features, conv2d1_weight, conv2d1_bias, y1)

        # Stage 2: Conv2d (384 -> 384 channels) + GELU
        y2 = torch.empty_like(y1)
        conv3x3_s2_p1_gelu_launch(y1, conv2d2_weight, conv2d2_bias, y2)

        # Stage 3: Conv2d (384 -> 384 channels) + GELU
        y3 = torch.empty_like(y2)
        conv3x3_s2_p1_gelu_launch(y2, conv2d3_weight, conv2d3_bias, y3)

        # Reshape: [B, channels, F, T] -> [B, T, channels*F]
        B, C, F, T = y3.size()
        x = y3.permute(0, 3, 1, 2).contiguous().view(B, T, C * F)  # [B, T, 1024]

        # Linear projection to d_model (no bias): matmul [B*T, 3840] @ [3840, 1024]
        B_T = B * T
        x_mat = x.view(B_T, 1024).contiguous().float()  # [B_T, 1024]
        w_mat = conv_out_weight.t().contiguous().float()  # [3840, 1024]
        out_mat = torch.empty((B_T, 1024), device=x.device, dtype=torch.float32)
        grid_mm = (B_T * 1024,)
        matmul1024x3840x3840[grid_mm](x_mat, w_mat, out_mat, B_T, 3840, 1024, x_mat.stride(0), x_mat.stride(1), w_mat.stride(0), w_mat.stride(1), out_mat.stride(0), out_mat.stride(1))

        # Reshape back to [B, T, 1024]
        out = out_mat.view(B, T, 1024)

        # Scale embeddings
        out_scaled = out * embed_scale

        # Add positional embeddings
        pos = positional_embedding[:T, :].contiguous().float()
        out_final = torch.empty_like(out_scaled)
        grid_pos = (B * T * 1024,)
        add_scale_and_pos_embed[grid_pos](
            out_scaled, pos, out_final,
            B, T, 1024,
            out_scaled.stride(0), out_scaled.stride(1), out_scaled.stride(2),
            pos.stride(0), pos.stride(1),
            out_final.stride(0), out_final.stride(1), out_final.stride(2),
            SCALE=embed_scale
        )

        return out_final


def run(*args):
    return ModelNew()(*args)
