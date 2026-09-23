class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        # Placeholders for weights; the evaluator provides real tensors.
        self.register_buffer('dummy_in_proj_w', torch.empty(1), persistent=False)
        self.register_buffer('dummy_in_proj_b', torch.empty(1), persistent=False)
        self.register_buffer('dummy_conv_w', torch.empty(1, 1, 4), persistent=False)
        self.register_buffer('dummy_conv_b', torch.empty(1), persistent=False)
        self.register_buffer('dummy_out_proj_w', torch.empty(1, 1, 1), persistent=False)
        self.register_buffer('dummy_out_proj_b', torch.empty(1), persistent=False)

    def forward(self, x, in_proj_weight, in_proj_bias, conv_weight, conv_bias, out_proj_weight, out_proj_bias):
        B, L, H = x.shape
        # Ensure float32
        x = x.float()
        in_proj_weight = in_proj_weight.float()
        in_proj_bias = in_proj_bias.float()
        conv_weight = conv_weight.float()
        conv_bias = conv_bias.float()
        out_proj_weight = out_proj_weight.float()
        out_proj_bias = out_proj_bias.float()

        # 1) In-projection: compute BCx
        BCx = torch.empty((B, 3 * H, L), device=x.device, dtype=torch.float32)
        grid_in = (triton.cdiv(3 * H, 64), triton.cdiv(L, 128), B)
        in_proj_kernel_B[grid_in](
            x, in_proj_weight, in_proj_bias, BCx,
            B, L, H,
            x.stride(0), x.stride(1), x.stride(2),
            in_proj_weight.stride(0), in_proj_weight.stride(1), in_proj_weight.stride(2),
            BCx.stride(0), BCx.stride(1), BCx.stride(2)
        )

        # 2) Launch the grouped conv + gating + out-projection kernel (must be invoked)
        out = torch.empty((B, L, H), device=x.device, dtype=torch.float32)
        grid = (triton.cdiv(H, 64), triton.cdiv(L, 128), B)
        conv_grouped_with_gating_and_out_kernel[grid](
            BCx, in_proj_bias, conv_weight, conv_bias, out_proj_weight, out_proj_bias, out,
            B, L, H,
            BCx.stride(0), BCx.stride(1), BCx.stride(2),
            conv_weight.stride(0), conv_weight.stride(1), conv_weight.stride(2),
            out.stride(0), out.stride(1), out.stride(2)
        )
        return out


def run(*args):
    return ModelNew()(*args)
