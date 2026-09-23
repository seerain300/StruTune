import torch
import triton
import triton.language as tl


# Triton kernel: Triple linear projection
# x: (B, S, H) float32
# in_proj_weight: (Nproj, H) float32, Nproj = 3*H
# in_proj_bias: (Nproj,) float32
# BCx: (B, S, Nproj) float32
@triton.jit
def triple_linear_kernel(
    x_ptr: tl.pointer[tl.float32],                # *f32, shape (B, S, H)
    in_proj_weight_ptr: tl.pointer[tl.float32],   # *f32, shape (Nproj, H)
    in_proj_bias_ptr: tl.pointer[tl.float32],     # *f32, shape (Nproj,)
    BCx_ptr: tl.pointer[tl.float32],              # *f32, shape (B, S, Nproj)
    B: tl.constexpr, S: tl.constexpr, H: tl.constexpr, Nproj: tl.constexpr,
    stride_x_b, stride_x_s, stride_x_h,
    stride_w_co, stride_w_ci,
    stride_bc_b, stride_bc_s, stride_bc_co,
):
    b = tl.program_id(0)
    s = tl.program_id(1)
    co = tl.program_id(2)  # output channel index in [0, Nproj)

    if b >= B or s >= S or co >= Nproj:
        return

    # Compute dot product over H: BCx[b, s, co] = sum_h x[b, s, h] * in_proj_weight[co, h] + in_proj_bias[co]
    acc = 0.0
    for h in range(0, H):
        x_val = tl.load(x_ptr + b * stride_x_b + s * stride_x_s + h * stride_x_h)
        w_val = tl.load(in_proj_weight_ptr + co * stride_w_co + h * stride_w_ci)
        acc += x_val * w_val
    bias_val = tl.load(in_proj_bias_ptr + co)
    acc += bias_val

    # Store the result
    tl.store(BCx_ptr + b * stride_bc_b + s * stride_bc_s + co * stride_bc_co, acc)


# Triton kernel: Elementwise multiply, computes Bx = B * x_proj
# B: (B, S, H) float32, X: (B, S, H) float32, Out: (B, S, H) float32
@triton.jit
def elementwise_mul_kernel(
    B_ptr: tl.pointer[tl.float32],
    X_ptr: tl.pointer[tl.float32],
    Out_ptr: tl.pointer[tl.float32],
    B: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
    stride_b_b, stride_b_s, stride_b_h,
    stride_x_b, stride_x_s, stride_x_h,
    stride_out_b, stride_out_s, stride_out_h,
):
    b = tl.program_id(0)
    s = tl.program_id(1)
    h = tl.program_id(2)
    if b >= B or s >= S or h >= H:
        return
    B_val = tl.load(B_ptr + b * stride_b_b + s * stride_b_s + h * stride_b_h)
    X_val = tl.load(X_ptr + b * stride_x_b + s * stride_x_s + h * stride_x_h)
    out_val = B_val * X_val
    tl.store(Out_ptr + b * stride_out_b + s * stride_out_s + h * stride_out_h, out_val)


# Host-side grouped causal conv1d on Bx: kernel_size=4, groups=H, padding=3 (causal)
# Bx is (B, S, H) conceptual indexing (b, ci, t). We return (B, H, S).
def grouped_causal_conv1d(Bx, conv_weight, conv_bias):
    B, S, H = Bx.shape
    conv_out = torch.empty((B, H, S), dtype=torch.float32, device=Bx.device)
    for b in range(B):
        for ci in range(H):
            acc = torch.zeros((S,), dtype=torch.float32, device=Bx.device)
            # Causal conv with kernel size 4, padding implicitly handled by masked loads
            for k in range(4):
                for t in range(S):
                    x_pos = t + k
                    x_val = Bx[b, ci, x_pos] if x_pos < S else 0.0
                    w_val = conv_weight[ci, ci, k]
                    acc[t] += x_val * w_val
            acc += conv_bias[ci]
            conv_out[b, ci, :] = acc
    return conv_out


class ModelNew(torch.nn.Module):
    def forward(self, x, in_proj_weight, in_proj_bias, conv_weight, conv_bias, out_proj_weight, out_proj_bias):
        # Ensure float32 for Triton kernels
        device = x.device
        x_f32 = x.to(torch.float32).contiguous()
        in_proj_w_f32 = in_proj_weight.to(torch.float32).contiguous()
        in_proj_b_f32 = in_proj_bias.to(torch.float32).contiguous()
        out_proj_w_f32 = out_proj_weight.to(torch.float32).contiguous()
        out_proj_b_f32 = out_proj_bias.to(torch.float32).contiguous()
        conv_w_f32 = conv_weight.to(torch.float32).contiguous()
        conv_b_f32 = conv_bias.to(torch.float32).contiguous()

        B, S, H = x_f32.shape
        Nproj = in_proj_w_f32.shape[0]
        assert Nproj == 3 * H, "in_proj_weight must have Nproj=3*H"

        # 1) Triple linear projection: BCx = F.linear(x, in_proj_weight, in_proj_bias)
        BCx = torch.empty((B, S, Nproj), dtype=torch.float32, device=device)
        grid_tril = (B, S, Nproj)
        triple_linear_kernel[grid_tril](
            x_f32, in_proj_w_f32, in_proj_b_f32, BCx,
            B, S, H, Nproj,
            x_f32.stride(0), x_f32.stride(1), x_f32.stride(2),
            in_proj_w_f32.stride(0), in_proj_w_f32.stride(1),
            BCx.stride(0), BCx.stride(1), BCx.stride(2),
            num_warps=1, num_stages=1,
        )

        # 2) Split BCx into B and x_proj along channels (dim=1) and compute gating Bx = B * x_proj
        # BCx shape: (B, S, 3H), split by last dimension width H
        B = BCx[:, :, 0:H]                 # (B, S, H)
        C = BCx[:, :, H:2*H]               # (B, S, H)
        x_proj = BCx[:, :, 2*H:3*H]       # (B, S, H)
        Bx = torch.empty((B, S, H), dtype=torch.float32, device=device)
        grid_gm = (B, S, H)
        elementwise_mul_kernel[grid_gm](
            B, x_proj, Bx,
            B, S, H,
            B.stride(0), B.stride(1), B.stride(2),
            x_proj.stride(0), x_proj.stride(1), x_proj.stride(2),
            Bx.stride(0), Bx.stride(1), Bx.stride(2),
            num_warps=1, num_stages=1,
        )

        # 3) Grouped causal conv1d on Bx (depthwise, groups=H), kernel_size=4, padding=3
        conv_out = grouped_causal_conv1d(Bx, conv_w_f32, conv_b_f32)  # (B, H, S)

        # 4) Gating with C: y = C * conv_out; conv_out is (B, H, S), C is (B, S, H)
        conv_out_t = conv_out.transpose(1, 2)  # (B, S, H)
        y = C * conv_out_t  # (B, S, H)

        # 5) Final linear projection: y -> (B, S, H)
        out = torch.nn.functional.linear(y, out_proj_w_f32, out_proj_b_f32)

        return out


def run(*args):
    return ModelNew()(*args)
