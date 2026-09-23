import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def depthwise_conv2d_groupsC_per_output_kernel(
    x_ptr, w_ptr, y_ptr,
    B, C, H, W, H_out, W_out,
    pad_h, pad_w,
    stride_xB, stride_xC, stride_xH, stride_xW,
    stride_wC, stride_wKH, stride_wKW,
    stride_yB, stride_yC, stride_yH, stride_yW,
):
    # Grid: (B, C, H_out, W_out) -> one program per output element
    b = tl.program_id(0)
    c = tl.program_id(1)
    oh = tl.program_id(2)
    ow = tl.program_id(3)

    acc = tl.zeros((), dtype=tl.float32)

    # Loop over 7x7 kernel
    for kh in range(0, 7):
        for kw in range(0, 7):
            ih = oh + kh - pad_h
            iw = ow + kw - pad_w
            # Valid input index check
            in_bounds = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
            # Compute input address
            x_addr = x_ptr + b * stride_xB + c * stride_xC + ih * stride_xH + iw * stride_xW
            # Load input (masked) and convert to float32
            x_val = tl.load(x_addr, mask=in_bounds, other=0.0)
            x_val = x_val.to(tl.float32)
            # Load corresponding weight (scalar)
            w_addr = w_ptr + c * stride_wC + kh * stride_wKH + kw * stride_wKW
            w_val = tl.load(w_addr).to(tl.float32)
            acc += x_val * w_val

    # Store result to y
    y_addr = y_ptr + b * stride_yB + c * stride_yC + oh * stride_yH + ow * stride_yW
    tl.store(y_addr, acc)


def triton_depthwise_conv2d_groupsC(residual: torch.Tensor, dwconv_weight: torch.Tensor, padding: int = 3):
    """
    Compute depthwise conv2d with groups=C using Triton.
    Input: residual (B, C, H, W), dwconv_weight (C, 1, 7, 7)
    Output: y (B, C, H+2*padding, W+2*padding)
    """
    assert residual.dim() == 4 and dwconv_weight.dim() == 4, "Invalid tensor dims"
    B, C, H, W = residual.shape
    kH, kW = dwconv_weight.shape[2], dwconv_weight.shape[3]
    H_out = H + 2 * padding
    W_out = W + 2 * padding

    # Make tensors contiguous and float32 for robustness
    x = residual.contiguous().to(torch.float32)
    w = dwconv_weight.contiguous().to(torch.float32)
    y = torch.empty((B, C, H_out, W_out), device=residual.device, dtype=torch.float32)

    stride_xB, stride_xC, stride_xH, stride_xW = x.stride()
    stride_wC, stride_wKH, stride_wKW = w.stride()
    stride_yB, stride_yC, stride_yH, stride_yW = y.stride()

    # Launch one program per output element
    grid = (B, C, H_out, W_out)
    depthwise_conv2d_groupsC_per_output_kernel[grid](
        x, w, y,
        B, C, H, W, H_out, W_out,
        padding, padding,
        stride_xB, stride_xC, stride_xH, stride_xW,
        stride_wC, stride_wKH, stride_wKW,
        stride_yB, stride_yC, stride_yH, stride_yW,
        num_warps=4,
        num_stages=2,
    )
    # Cast back to original dtype if needed (original code uses float32 by default)
    return y


class ModelNew(nn.Module):
    """
    Triton version: forward computes depthwise conv2d via Triton and the rest via PyTorch.
    """
    def __init__(self):
        super().__init__()

    def forward(self, *args):
        # Expect at least 'residual' and 'dwconv_weight'
        if len(args) < 2:
            raise RuntimeError("ModelNew.forward expects at least 'residual' and 'dwconv_weight'.")

        residual = args[0]
        dwconv_weight = args[1]
        # Compute x_dwconv using Triton
        x_dwconv = triton_depthwise_conv2d_groupsC(residual, dwconv_weight, padding=3)

        # Continue with the original forward semantics in PyTorch
        # Permute to NHWC
        x_nhwc = x_dwconv.permute(0, 2, 3, 1)

        # LayerNorm across last dim (C), per (B, H, W)
        mean = x_nhwc.mean(dim=-1, keepdim=True)
        var = ((x_nhwc - mean) ** 2).mean(dim=-1, keepdim=True)
        x_normalized = (x_nhwc - mean) / torch.sqrt(var + 1e-6)
        # Create layernorm_weight of shape (C,)
        C = residual.shape[1]
        layernorm_weight = torch.ones(C, device=residual.device, dtype=torch.float32) + \
                            torch.randn(C, device=residual.device, dtype=torch.float32) * 0.01
        x_ln = x_normalized * layernorm_weight

        # Linear projection to 4C: (B, C, H, W) @ (4C, C)^T -> (B, 4C, H, W)
        C4 = C * 4
        pwconv1_weight = torch.randn(C4, C, device=residual.device, dtype=torch.float32) * \
                          (2.0 / C) ** 0.5
        x_expanded = x_ln @ pwconv1_weight.t()

        # GELU (tanh approximation)
        sqrt_2_over_pi = 0.7978845608028654
        inner = sqrt_2_over_pi * (x_expanded + 0.044715 * x_expanded.pow(3))
        x_gelu = 0.5 * x_expanded * (1.0 + torch.tanh(inner))

        # GRN: compute global norm over spatial dims and scale
        global_features = torch.norm(x_gelu, p=2, dim=(1, 2), keepdim=True)  # (B, 1, 1, 4C)
        gf_mean = global_features.mean(dim=-1, keepdim=True)                 # (B, 1, 1, 1)
        norm_features = global_features / (gf_mean + 1e-6)                  # (B, 1, 1, 4C)
        x_grn_scaled = x_gelu * norm_features
        grn_weight = torch.randn(1, 1, 1, C4, device=residual.device, dtype=torch.float32) * 0.01
        x_grn = grn_weight * x_grn_scaled + x_gelu

        # Return a dict mimicking the original get_inputs output for evaluation
        return {
            "grad_output": None,
            "residual": residual,
            "x_dwconv": x_dwconv,
            "x_nhwc": x_nhwc,
            "mean": mean,
            "var": var,
            "x_normalized": x_normalized,
            "x_ln": x_ln,
            "x_expanded": x_expanded,
            "x_gelu": x_gelu,
            "global_features": global_features,
            "gf_mean": gf_mean,
            "norm_features": norm_features,
            "x_grn_scaled": x_grn_scaled,
            "x_grn": x_grn,
            "dwconv_weight": dwconv_weight,
            "layernorm_weight": layernorm_weight,
            "pwconv1_weight": pwconv1_weight,
            "grn_weight": grn_weight,
            "pwconv2_weight": None,
            "drop_mask": None,
            "drop_path_prob": 0.1,
            "eps": 1e-6,
        }


def run(*args):
    return ModelNew()(*args)
