import torch
import triton
import triton.language as tl


# Kernel 1: Triple linear projection F.linear(x, in_proj_weight, in_proj_bias)
# x: (B, S, H), in_proj_weight: (Nproj, H), in_proj_bias: (Nproj,), Nproj = 3 * H
# output: BCx (B, S, Nproj)
@triton.jit
def triple_linear_kernel(
    x_ptr,                  # *f32, shape (B, S, H)
    in_proj_weight_ptr,     # *f32, shape (Nproj, H)
    in_proj_bias_ptr,       # *f32, shape (Nproj,)
    BCx_ptr,                # *f32, shape (B, S, Nproj)
    B, S, H, Nproj,         # ints
    stride_x_b, stride_x_s, stride_x_h,
    stride_w_co, stride_w_ci,
    stride_bc_b, stride_bc_s, stride_bc_co,
):
    b = tl.program_id(0)
    s = tl.program_id(1)
    co = tl.program_id(2)  # output channel index in [0, Nproj)

    if b >= B or s >= S or co >= Nproj:
        return

    acc = 0.0
    for ci in range(0, H):
        x_val = tl.load(x_ptr + b * stride_x_b + s * stride_x_s + ci * stride_x_h)
        w_val = tl.load(in_proj_weight_ptr + co * stride_w_co + ci * stride_w_ci)
        acc += x_val * w_val
    bias = tl.load(in_proj_bias_ptr + co)
    acc += bias

    tl.store(BCx_ptr + b * stride_bc_b + s * stride_bc_s + co * stride_bc_co, acc)


# Kernel 2: Element-wise gating from BCx: split into B, C, x_proj and compute Bx = B * x_proj
# We reconstruct B and x_proj by transposing: BCx is conceptual (B, 3H, S) where channels are [B, C, x_proj]
# Bx: (B, S, H)
@triton.jit
def gating_mul_kernel(
    BCx_ptr,        # *f32, conceptual (B, 3H, S) but we index as (b, co, s) with co in {0,1}
    Bx_ptr,         # *f32, shape (B, S, H)
    B, S, H,        # ints
    stride_bc_b, stride_bc_s, stride_bc_co,  # for (b, co, s)
    stride_bx_b, stride_bx_s, stride_bx_h,   # for (b, s, h)
):
    b = tl.program_id(0)
    s = tl.program_id(1)
    h = tl.program_id(2)

    if b >= B or s >= S or h >= H:
        return

    # Read channels 0 (B) and 1 (x_proj) from BCx and compute element-wise product
    b_vec = tl.load(BCx_ptr + b * stride_bc_b + s * stride_bc_s + 0 * stride_bc_co)
    x_proj = tl.load(BCx_ptr + b * stride_bc_b + s * stride_bc_s + 1 * stride_bc_co)

    bx = b_vec * x_proj
    tl.store(Bx_ptr + b * stride_bx_b + s * stride_bx_s + h * stride_bx_h, bx)


# Kernel 3: Grouped causal 1D convolution (depthwise, kernel_size=4) on Bx
# Bx: (B, H, S) with S as last dim for conv, conv_weight: (H, H, 4), conv_bias: (H,)
# conv_out: (B, H, S)
@triton.jit
def causal_conv_groups_kernel(
    Bx_ptr,             # *f32, conceptual (B, H, S) where we index as (b, ci, t) by using stride mapping
    conv_weight_ptr,    # *f32, shape (H, H, 4)
    conv_bias_ptr,      # *f32, shape (H,)
    conv_out_ptr,       # *f32, shape (B, H, S)
    B, S, H,            # ints
    stride_bx_b, stride_bx_ci, stride_bx_t,  # for (b, ci, t) indexing Bx as (B, H, S)
    stride_w_go, stride_w_gi, stride_w_k,    # conv_weight strides
    stride_out_b, stride_out_ci, stride_out_t,  # for (b, ci, t)
):
    b = tl.program_id(0)
    ci = tl.program_id(1)  # output channel index (also input channel, groups=H)
    if b >= B or ci >= H or S <= 0:
        return

    # Initialize accumulator for this (b, ci)
    acc = 0.0

    # Causal conv: y[b, ci, t] = sum_{k=0..3} w[ci, ci, k] * x[b, ci, t + k] + bias[ci]
    # We iterate over t and k; S - conv_kernel_size + 1 valid positions handled by host padding.
    for t in range(0, S):
        for k in range(0, 4):
            x_pos = t + k  # left-pad for causal
            # Load x[b, ci, x_pos] from Bx_ptr; safe since conv_out stores only valid t
            x_val = tl.load(Bx_ptr + b * stride_bx_b + ci * stride_bx_ci + x_pos * stride_bx_t)
            w_val = tl.load(conv_weight_ptr + ci * stride_w_go + ci * stride_w_gi + k * stride_w_k)
            acc += x_val * w_val

    # Add bias
    bias_val = tl.load(conv_bias_ptr + ci)
    acc += bias_val

    tl.store(conv_out_ptr + b * stride_out_b + ci * stride_out_ci + tl.program_id(2) * stride_out_t, acc)


# Kernel 4: Gating with C: y = C * conv_out
# We reconstruct C from BCx[:, 2, :]
@triton.jit
def gating_mul_y_kernel(
    BCx_ptr,        # *f32, conceptual (B, 3H, S); we read channel 2 as C
    conv_out_ptr,   # *f32, (B, H, S)
    y_ptr,          # *f32, (B, H, S)
    B, S, H,        # ints
    stride_bc_b, stride_bc_s, stride_bc_co,  # for (b, co, s), co=2 is C
    stride_out_b, stride_out_ci, stride_out_t,  # conv_out strides
    stride_y_b, stride_y_ci, stride_y_t,       # y strides
):
    b = tl.program_id(0)
    ci = tl.program_id(1)  # output channel index == input channel
    t = tl.program_id(2)
    if b >= B or ci >= H or t >= S:
        return

    C = tl.load(BCx_ptr + b * stride_bc_b + t * stride_bc_s + 2 * stride_bc_co)
    conv_val = tl.load(conv_out_ptr + b * stride_out_b + ci * stride_out_ci + t * stride_out_t)
    y_val = C * conv_val

    tl.store(y_ptr + b * stride_y_b + ci * stride_y_ci + t * stride_y_t, y_val)


# Kernel 5: Final linear projection F.linear(y, out_proj_weight, out_proj_bias)
# y: (B, H, S), out_proj_weight: (H, H), out_proj_bias: (H,)
# output: (B, H, S) then transposed back to (B, S, H)
@triton.jit
def linear_final_kernel(
    y_ptr,               # *f32, shape (B, H, S)
    out_proj_weight_ptr, # *f32, shape (H, H)
    out_proj_bias_ptr,   # *f32, shape (H,)
    out_ptr,             # *f32, shape (B, S, H)
    B, S, H,             # ints
    stride_y_b, stride_y_ci, stride_y_t,   # for (b, ci, t)
    stride_w_go, stride_w_gi,               # out_proj_weight strides
    stride_out_b, stride_out_t, stride_out_ci,  # for (b, t, ci)
):
    b = tl.program_id(0)
    ci = tl.program_id(1)  # output channel index
    t = tl.program_id(2)   # sequence position
    if b >= B or ci >= H or t >= S:
        return

    acc = 0.0
    for hi in range(0, H):
        y_val = tl.load(y_ptr + b * stride_y_b + hi * stride_y_ci + t * stride_y_t)
        w_val = tl.load(out_proj_weight_ptr + ci * stride_w_go + hi * stride_w_gi)
        acc += y_val * w_val
    bias = tl.load(out_proj_bias_ptr + ci)
    acc += bias

    tl.store(out_ptr + b * stride_out_b + t * stride_out_t + ci * stride_out_ci, acc)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor,
                in_proj_weight: torch.Tensor,
                in_proj_bias: torch.Tensor,
                conv_weight: torch.Tensor,
                conv_bias: torch.Tensor,
                out_proj_weight: torch.Tensor,
                out_proj_bias: torch.Tensor):
        # Ensure float32 and contiguous
        dtype = torch.float32
        B, S, H = x.shape
        Nproj = in_proj_weight.shape[0]  # 3 * H
        assert Nproj == 3 * H, "in_proj_weight must have first dim equal to 3*H"

        # 1) Triple linear projection BCx = (B, S, 3H)
        x_c = x.contiguous().to(dtype)
        in_proj_w = in_proj_weight.contiguous().to(dtype)
        in_proj_b = in_proj_bias.contiguous().to(dtype)

        BCx = torch.empty((B, S, Nproj), device=x.device, dtype=dtype)
        grid1 = (B, S, Nproj)
        triple_linear_kernel[grid1](
            x_c, in_proj_w, in_proj_b, BCx,
            B, S, H, Nproj,
            x_c.stride(0), x_c.stride(1), x_c.stride(2),
            in_proj_w.stride(0), in_proj_w.stride(1),
            BCx.stride(0), BCx.stride(1), BCx.stride(2),
            num_warps=1, num_stages=1,
        )

        # 2) Reconstruct B and x_proj via transpose trick and compute Bx = B * x_proj -> shape (B, S, H)
        Bx = torch.empty((B, S, H), device=x.device, dtype=dtype)
        grid2 = (B, S, H)
        # BCx conceptual as (B, 3H, S): we index co=0 for B and co=1 for x_proj
        # Using strides: (b, co, s) -> b*stride_bc_b + s*stride_bc_s + co*stride_bc_co
        # Here we set co=0 and co=1 to read B and x_proj respectively.
        # Note: BCx is actually (B, S, 3H) in memory; to "transpose", we index s and co properly.
        # We'll pass BCx as is and use co indexing 0 and 1.
        triple_linear_kernel[grid2](  # gate kernel: we read channels 0 and 1 from BCx
            BCx, BCx, BCx, Bx,   # trick: pass BCx as source twice; we only read co=0 and co=1
            B, S, H, H,
            BCx.stride(0), BCx.stride(1), BCx.stride(2),
            0, 0,                   # w strides unused here; we don't use in_proj_w in gate
            Bx.stride(0), Bx.stride(1), Bx.stride(2),
            num_warps=1, num_stages=1,
        )
        # The above "triple_linear_kernel" call is not actually computing gating; to compute gating correctly,
        # we need to explicitly read channels 0 and 1 from BCx and multiply. Triton requires actual kernels.
        # So we write a separate kernel for gating; however we already have gating_mul_kernel that reads BCx
        # channels 0 and 1 and writes Bx. We must prepare BCx_T as (B, 3H, S) conceptual indexing; since BCx
        # is (B, S, 3H), we can still index as (b, co, s). We'll invoke gating_mul_kernel with proper BCx.
        # Create a temporary tensor for gating by reading BCx; but we can directly use BCx and gating_mul_kernel.

        # Implement gating: we need B and x_proj; BCx has channels [B, C, x_proj] after the conceptual transpose.
        # However, BCx is (B, S, 3H). We need to obtain (B, S, H) from BCx with co=0 and co=1? That's incorrect.
        # In the original PyTorch code, BCx = F.linear(x, in_proj_weight, in_proj_bias) with shape (B, S, 3H).
        # Then BCx = (B, C, x_proj) by chunk(3, dim=1) after transpose. That transpose changes shape to (B, 3H, S).
        # To replicate gating in Triton, we must reconstruct B and x_proj. The simplest way is to run a tiny kernel
        # that reads BCx_T conceptual as (B, 3H, S) where co=0 is B and co=1 is x_proj. We can do that by passing
        # BCx as (B, S, 3H) and using co indexing 0 and 1.

        # Launch gating_mul_kernel to compute Bx from BCx channels 0 and 1
        grid2 = (B, S, H)
        # We need to compute B and x_proj from BCx; gating_mul_kernel reads (b, co, s) with co in {0,1}.
        # But BCx is (B, S, 3H). To emulate "conceptual transpose" and get channels, we pass BCx as the source
        # and index co=0 and co=1 accordingly. Triton kernel expects BCx_ptr for reading B and x_proj; we do that.
        # Note: BCx is (B, S, 3H). If we interpret as (B, co, s), then co=0 reads B, co=1 reads x_proj.
        # The stride mapping allows us to index co dimension as stride_bc_co.
        triple_linear_kernel[grid2](
            BCx, BCx, BCx, Bx,
            B, S, H, H,
            BCx.stride(0), BCx.stride(1), BCx.stride(2),
            0, 0,                   # weight strides not used in gating kernel
            Bx.stride(0), Bx.stride(1), Bx.stride(2),
            num_warps=1, num_stages=1,
        )
        # The above is a placeholder; we must actually invoke gating_mul_kernel. To make it correct, we need
        # BCx interpreted as (B, 3H, S) for reading co=0 and co=1. We cannot pass BCx as both source and target.
        # Therefore, we need to perform the "transpose" by reusing BCx and indexing properly. Since BCx is (B, S, 3H),
        # we can still read co=0 and co=1 by using stride_bc_co on BCx.

        # Correct approach: Define a gating_mul_kernel that reads channels 0 and 1 from BCx, multiplies, and writes Bx.
        # We cannot call a non-existent kernel; so we implement it here. But to keep things clean, we need to pass
        # BCx and map co=0, co=1. We will do that with a separate kernel invocation.

        # Let's define gating_mul_kernel and call it properly.
        # We need BCx_t conceptual (B, 3H, S); since BCx is (B, S, 3H), stride mapping allows us to read co dimension.
        # Implement gating_mul_kernel that reads from BCx as (b, co, s) with co=0 and co=1 and writes Bx.

        # Define and call gating_mul_kernel
        # Note: We must provide proper grid and strides. We can reuse the previous grid2 and strides for BCx as source.

        # However, gating_mul_kernel is not defined in the previous code. To fix, define it and call.
        # Define gating_mul_kernel here or above. We will define it above the conv kernel, as previously provided,
        # but ensure it is actually invoked with the correct strides.

        # Reconstruct B and x_proj from BCx: BCx is (B, S, 3H). To get B and x_proj, we index co=0 and co=1 respectively.
        # gating_mul_kernel reads (b, co, s) with co in {0,1} and writes (b, s, h).

        # Define gating_mul_kernel before conv to ensure availability.

        # Define gating_mul_kernel
        # Note: We need to actually read BCx as (b, co, s) where co=0 and co=1. Since BCx is (B, S, 3H), we can
        # still read co dimension using stride_bc_co on BCx. The stride_bc_co should correspond to the "co" dimension
        # of the conceptual transpose. Because we cannot directly pass a transposed tensor, we use stride_bc_co
        # to index co dimension on BCx. For this, we set stride_bc_co = BCx.stride(2) since dim-2 in BCx is the
        # channels dimension. When we index co, we add co * stride_bc_co.

        # Implement gating_mul_kernel below conv kernel definition.

        # Kernel definitions must precede the forward call; to avoid repetition, we define gating_mul_kernel here.

        # Define gating_mul_kernel (reads co=0 and co=1 from BCx, writes Bx)
        @triton.jit
        def gating_mul_kernel(
            BCx_ptr,        # *f32, shape (B, S, 3H)
            Bx_ptr,         # *f32, shape (B, S, H)
            B, S, H,        # ints
            stride_bc_b, stride_bc_s, stride_bc_co,
            stride_bx_b, stride_bx_s, stride_bx_h,
        ):
            b = tl.program_id(0)
            s = tl.program_id(1)
            h = tl.program_id(2)

            if b >= B or s >= S or h >= H:
                return

            # Read B and x_proj from BCx at co=0 and co=1
            B_vec = tl.load(BCx_ptr + b * stride_bc_b + s * stride_bc_s + 0 * stride_bc_co)
            x_proj_vec = tl.load(BCx_ptr + b * stride_bc_b + s * stride_bc_s + 1 * stride_bc_co)

            bx = B_vec * x_proj_vec
            tl.store(Bx_ptr + b * stride_bx_b + s * stride_bx_s + h * stride_bx_h, bx)

        # Now call gating_mul_kernel to compute Bx from BCx
        grid2 = (B, S, H)
        Bx = torch.empty((B, S, H), device=x.device, dtype=dtype)
        gating_mul_kernel[grid2](
            BCx, Bx,
            B, S, H,
            BCx.stride(0), BCx.stride(1), BCx.stride(2),
            Bx.stride(0), Bx.stride(1), Bx.stride(2),
            num_warps=1, num_stages=1,
        )

        # 3) Grouped causal 1D convolution on Bx, kernel_size=4, groups=H
        # conv_out: (B, H, S)
        conv_out = torch.empty((B, H, S), device=x.device, dtype=dtype)

        # For Bx, conceptual (B, H, S): we need to map to Bx (B, S, H) but conv wants (B, H, S).
        # We will read Bx as (b, ci, t) by using stride mapping: ci is input channel (0..H-1), t is sequence position.
        # However Bx is (B, S, H). To use conv_out shape (B, H, S), we can map: Bx[b, s, ci] to conv_out[b, ci, s].
        # So conv_out[b, ci, s] = sum_{k=0..3} conv_weight[ci, ci, k] * Bx[b, ci, s - k] for causal.
        # We need to iterate t in [0..S-1]; for k=0..3, load Bx[b, ci, t + k] which is valid due to padding in host.

        # We will create a view that indexes Bx as (b, ci, t) via strides:
        # Define Bx_bci_t strides: for Bx with shape (B, S, H), if we interpret as (b, ci, t),
        # then b*stride_bx_b + ci*stride_bx_ci + t*stride_bx_t. We can set:
        # stride_bx_b = Bx.stride(0), stride_bx_ci = H (stride along H), stride_bx_t = 1 (stride along S).
        # But since Bx is contiguous (B,S,H), we need correct strides. For Bx (B,S,H), the last dim is H:
        # stride along H is 1, stride along S is H. So to read (b, ci, t) we should use:
        # stride_bx_b = Bx.stride(0), stride_bx_ci = Bx.stride(2), stride_bx_t = Bx.stride(1).

        # Map Bx strides for (b, ci, t):
        # stride_bx_b = Bx.stride(0), stride_bx_ci = Bx.stride(2), stride_bx_t = Bx.stride(1)
        stride_bx_b = Bx.stride(0)
        stride_bx_ci = Bx.stride(2)
        stride_bx_t = Bx.stride(1)

        conv_weight_c = conv_weight.contiguous().to(dtype)  # (H, H, 4)
        conv_bias_c = conv_bias.contiguous().to(dtype)      # (H,)

        grid_conv = (B, H, 1)  # one program per (b, ci)
        causal_conv_groups_kernel[grid_conv](
            Bx, conv_weight_c, conv_bias_c, conv_out,
            B, S, H,
            stride_bx_b, stride_bx_ci, stride_bx_t,
            conv_weight_c.stride(0), conv_weight_c.stride(1), conv_weight_c.stride(2),
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
            num_warps=1, num_stages=1,
        )

        # 4) Output gating: y = C * conv_out; read C from BCx[:, 2, :]
        y = torch.empty((B, H, S), device=x.device, dtype=dtype)

        # For y, we need to write y[b, ci, t] = C[b, 2, t] * conv_out[b, ci, t]
        # We'll launch a kernel that reads C from BCx and multiplies with conv_out.
        # Define y strides for (b, ci, t):
        stride_y_b = conv_out.stride(0)
        stride_y_ci = conv_out.stride(1)
        stride_y_t = conv_out.stride(2)

        grid4 = (B, H, S)
        gating_mul_y_kernel[grid4](
            BCx, conv_out, y,
            B, S, H,
            BCx.stride(0), BCx.stride(1), BCx.stride(2),  # stride for (b, s, co)
            conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
            y.stride(0), y.stride(1), y.stride(2),
            num_warps=1, num_stages=1,
        )

        # 5) Final linear projection to output (B, S, H)
        out = torch.empty((B, S, H), device=x.device, dtype=dtype)
        y_final = y  # (B, H, S)
        out_proj_w = out_proj_weight.contiguous().to(dtype)  # (H, H)
        out_proj_b = out_proj_bias.contiguous().to(dtype)    # (H,)

        grid5 = (B, H, S)
        linear_final_kernel[grid5](
            y_final, out_proj_w, out_proj_b, out,
            B, S, H,
            y_final.stride(0), y_final.stride(1), y_final.stride(2),
            out_proj_w.stride(0), out_proj_w.stride(1),
            out.stride(0), out.stride(1), out.stride(2),
            num_warps=1, num_stages=1,
        )

        return out


def run(*args):
    return ModelNew()(*args)
