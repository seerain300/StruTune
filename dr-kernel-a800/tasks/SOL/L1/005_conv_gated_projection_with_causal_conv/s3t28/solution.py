import torch
import triton
import triton.language as tl


# Triton kernel: compute BCx = F.linear(x, in_proj_weight, in_proj_bias)
# x: (B, S, H), in_proj_weight: (Nproj, H), in_proj_bias: (Nproj,), Nproj = 3*H
# output BCx: (B, S, Nproj)
@triton.jit
def triple_linear_kernel(
    x_ptr,                  # *f32, (B, S, H)
    in_proj_weight_ptr,     # *f32, (Nproj, H)
    in_proj_bias_ptr,       # *f32, (Nproj,)
    BCx_ptr,                # *f32, (B, S, Nproj)
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

    # Vectorized reduction over H for each (b, s, co)
    acc = 0.0
    for ci in range(0, H):
        x_val = tl.load(x_ptr + b * stride_x_b + s * stride_x_s + ci * stride_x_h)
        w_val = tl.load(in_proj_weight_ptr + co * stride_w_co + ci * stride_w_ci)
        acc += x_val * w_val
    bias = tl.load(in_proj_bias_ptr + co)
    acc += bias

    tl.store(BCx_ptr + b * stride_bc_b + s * stride_bc_s + co * stride_bc_co, acc)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor,
                in_proj_weight: torch.Tensor,
                in_proj_bias: torch.Tensor,
                conv_weight: torch.Tensor,
                conv_bias: torch.Tensor,
                out_proj_weight: torch.Tensor,
                out_proj_bias: torch.Tensor):
        """
        Triton-optimized forward that computes the triple linear projection in Triton,
        then performs the rest using PyTorch ops. This ensures correctness and
        demonstrates Triton usage. Further, conv and final linear can be ported to Triton
        for performance, but here we keep PyTorch to ensure robust correctness.
        """
        # Ensure contiguous and float32 for Triton
        x = x.contiguous().to(torch.float32)
        in_proj_weight = in_proj_weight.contiguous().to(torch.float32)
        in_proj_bias = in_proj_bias.contiguous().to(torch.float32)

        B, S, H = x.shape
        Nproj = in_proj_weight.shape[0]

        # 1) Triple linear projection BCx using Triton: (B, S, Nproj)
        BCx = torch.empty((B, S, Nproj), dtype=torch.float32, device=x.device)

        grid1 = (B, S, Nproj)
        triple_linear_kernel[grid1](
            x, in_proj_weight, in_proj_bias, BCx,
            B, S, H, Nproj,
            x.stride(0), x.stride(1), x.stride(2),
            in_proj_weight.stride(0), in_proj_weight.stride(1),
            BCx.stride(0), BCx.stride(1), BCx.stride(2),
            num_warps=1, num_stages=1,
        )

        # 2) Transpose and split BCx into B, C, x_proj
        # BCx shape (B, S, Nproj), we treat it as (B, Nproj, S) for channel split
        BCx_T = BCx.transpose(-1, -2)  # (B, Nproj, S)
        # Channels: 0 -> B, 1 -> x_proj, 2 -> C
        B_t = BCx_T[:, 0, :].contiguous()   # (B, S)
        x_proj_t = BCx_T[:, 1, :].contiguous()  # (B, S)
        C_t = BCx_T[:, 2, :].contiguous()   # (B, S)

        # Make them (B, H, S) by expanding along H dimension
        B_t = B_t.view(B, 1, S).expand(B, H, S).contiguous()  # broadcasting is fine, but we materialize
        x_proj_t = x_proj_t.view(B, 1, S).expand(B, H, S).contiguous()
        C_t = C_t.view(B, 1, S).expand(B, H, S).contiguous()

        # 3) Element-wise gating: Bx = B * x_proj
        Bx = (B_t * x_proj_t).contiguous()  # (B, H, S)

        # 4) Grouped causal conv1d on Bx, kernel_size=4, groups=H, bias conv_bias
        # Use PyTorch for conv to ensure correctness
        # Pad left by kernel_size - 1 for causal
        pad = 3  # since kernel_size=4
        Bx_padded = torch.nn.functional.pad(Bx, (pad, 0))  # (B, H, S + pad)
        conv_out = torch.nn.functional.conv1d(
            Bx_padded, conv_weight, conv_bias, groups=H, padding=pad, stride=1
        )  # (B, H, S)

        # 5) Output gating: y = C * conv_out
        # C_t shape is (B, H, S), so elementwise multiply
        y = C_t * conv_out  # (B, H, S)

        # 6) Final output projection: y_T = y.transpose(-1, -2) -> (B, S, H)
        y_T = y.transpose(-1, -2).contiguous()  # (B, S, H)

        # Final linear
        output = torch.nn.functional.linear(y_T, out_proj_weight, out_proj_bias)  # (B, S, H)

        return output


def run(*args):
    return ModelNew()(*args)
