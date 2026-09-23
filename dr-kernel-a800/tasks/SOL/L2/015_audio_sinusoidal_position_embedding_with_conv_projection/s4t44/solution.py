class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # No parameters; we rely on the provided inputs

    def forward(self, input_features, conv2d1_weight, conv2d1_bias,
                conv2d2_weight, conv2d2_bias,
                conv2d3_weight, conv2d3_bias,
                conv_out_weight,
                positional_embedding,
                embed_scale):
        # Stage 1: Conv2d (1 -> 384) + GELU
        x = torch.nn.functional.conv2d(input_features, conv2d1_weight, conv2d1_bias, stride=2, padding=1)
        N1 = x.numel()
        x_gelu = torch.empty_like(x)
        gelu_tanh_kernel[(triton.cdiv(N1, 1024),)](x, x_gelu, N1, BLOCK=1024)

        # Stage 2: Conv2d (384 -> 384) + GELU
        x = torch.nn.functional.conv2d(x_gelu, conv2d2_weight, conv2d2_bias, stride=2, padding=1)
        N2 = x.numel()
        x_gelu = torch.empty_like(x)
        gelu_tanh_kernel[(triton.cdiv(N2, 1024),)](x, x_gelu, N2, BLOCK=1024)

        # Stage 3: Conv2d (384 -> 384) + GELU
        x = torch.nn.functional.conv2d(x_gelu, conv2d3_weight, conv2d3_bias, stride=2, padding=1)
        N3 = x.numel()
        x_gelu = torch.empty_like(x)
        gelu_tanh_kernel[(triton.cdiv(N3, 1024),)](x, x_gelu, N3, BLOCK=1024)

        # Reshape: (batch, channels, freq, time) -> (batch, time, channels*freq)
        b, c, f, t = x_gelu.size()
        x_gelu = x_gelu.permute(0, 3, 1, 2).contiguous()
        x_gelu = x_gelu.view(b, t, c * f)

        # Linear projection to d_model (1024) — no bias
        B = x_gelu.shape[0]
        T_after = x_gelu.shape[1]
        K = x_gelu.shape[2]  # 3840
        M = conv_out_weight.shape[0]  # 1024

        x_gelu_contig = x_gelu.contiguous()
        conv_out_weight_contig = conv_out_weight.contiguous()
        y = torch.empty((B, T_after, M), dtype=x_gelu_contig.dtype, device=x_gelu_contig.device)

        x_strides = x_gelu_contig.stride()
        w_strides = conv_out_weight_contig.stride()
        y_strides = y.stride()

        grid = (B, T_after, triton.cdiv(M, 128))
        linear_no_bias_kernel[grid](
            x_gelu_contig, conv_out_weight_contig, y,
            B, T_after, K, M,
            x_strides[0], x_strides[1], x_strides[2],
            w_strides[0], w_strides[1],
            y_strides[0], y_strides[1], y_strides[2],
            BLOCK_M=128, BLOCK_K=128
        )

        # Scale by embed_scale and add positional embedding
        scale = float(embed_scale)
        y_contig = y.contiguous()  # elementwise kernel expects contiguous
        pos_slice = positional_embedding[:T_after, :].contiguous()
        out = torch.empty_like(y_contig)

        grid_elem = (B, T_after, triton.cdiv(M, 128))
        # Strides for elementwise kernel (assumes standard contiguous)
        stride_xb, stride_xt, stride_xm = x_gelu_contig.stride()
        stride_pt, stride_pm = pos_slice.stride()
        stride_yb, stride_yt, stride_ym = y_contig.stride()

        # Pass 128 as BLOCK_M to cover typical sizes; M is 1024, so 128 tiles = 8
        scale_add_pos_emb_kernel[grid_elem](
            y_contig, pos_slice, out,
            B, T_after, M, scale,
            BLOCK_M=128
        )

        return out


def run(*args):
    return ModelNew()(*args)
