class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor,
                conv1_weight: torch.Tensor, norm1_weight: torch.Tensor, norm1_bias: torch.Tensor,
                conv2_weight: torch.Tensor, norm2_weight: torch.Tensor, norm2_bias: torch.Tensor,
                eps: float):
        # Ensure we are on CUDA and use float32 for computation
        assert x.is_cuda, "Input tensor must be on CUDA for Triton kernels."
        # Cast to float32 for stable accumulation in Triton
        x32 = x.float()
        conv1_w32 = conv1_weight.float()
        conv2_w32 = conv2_weight.float()
        norm1_weight32 = norm1_weight.float()
        norm1_bias32 = norm1_bias.float()
        norm2_weight32 = norm2_weight.float()
        norm2_bias32 = norm2_bias.float()

        N, C, H, W = x32.shape
        C1, Ci1, _, _ = conv1_w32.shape  # C1 = C
        C2, Ci2, _, _ = conv2_w32.shape  # C2 = C

        # 1) Conv1: Triton
        out1 = torch.empty((N, C, H, W), device=x32.device, dtype=torch.float32)
        conv3x3_stride1_pad1_block_oc_kernel[(N, triton.cdiv(C, 1))](  # BLOCK_OC=1 for generality; we can set 32 for perf if desired
            x32, conv1_w32, out1,
            N, C, H, W, C,
            BLOCK_OC=1,
            num_warps=1, num_stages=1
        )

        # 2) GroupNorm1 (32 groups) + SiLU1
        out1_norm = torch.empty_like(out1)
        group_norm_32groups_kernel[(N, 32)](
            out1, norm1_weight32, norm1_bias32, out1_norm, N, C, H, W, eps,
            GROUPS=32, num_warps=4, num_stages=2
        )
        y1_silu = torch.empty_like(out1_norm)
        total = N * C * H * W
        silu_kernel[(total,)](out1_norm, y1_silu, N, C, H, W, num_warps=4, num_stages=2)

        # 3) Conv2: Triton
        out2_conv = torch.empty((N, C, H, W), device=x32.device, dtype=torch.float32)
        conv3x3_stride1_pad1_block_oc_kernel[(N, triton.cdiv(C, 1))](
            y1_silu, conv2_w32, out2_conv,
            N, C, H, W, C,
            BLOCK_OC=1,
            num_warps=1, num_stages=1
        )

        # 4) GroupNorm2 (32 groups) + SiLU2
        out2_norm = torch.empty_like(out2_conv)
        group_norm_32groups_kernel[(N, 32)](
            out2_conv, norm2_weight32, norm2_bias32, out2_norm, N, C, H, W, eps,
            GROUPS=32, num_warps=4, num_stages=2
        )
        y2_silu = torch.empty_like(out2_norm)
        silu_kernel[(total,)](out2_norm, y2_silu, N, C, H, W, num_warps=4, num_stages=2)

        # 5) Residual add: Triton
        y_out = torch.empty_like(y2_silu)
        add_residual_kernel[(total,)](
            y2_silu, x32, y_out, N, C, H, W, num_warps=4, num_stages=2
        )

        return y_out


def run(*args):
    return ModelNew()(*args)
