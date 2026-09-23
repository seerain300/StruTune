import math
import triton
import triton.language as tl


# Triton elementwise kernel: Y = X * scale, over N elements
@triton.jit
def scale_kernel(X, Y, N, scale, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    x = tl.load(X + offsets, mask=mask, other=0.0).to(tl.float32)
    y = x * scale
    tl.store(Y + offsets, y, mask=mask)


# Triton elementwise kernel: Y = X + POS, where POS is broadcast over last dim (size d_model)
# We assume Y is (B, T, d_model) contiguous, POS is (L, d_model) contiguous; here we only add the first T rows.
@triton.jit
def add_pos_embed_kernel(Y, POS, B, T, d_model, BLOCK_D: tl.constexpr):
    pid_b = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_d = tl.program_id(2)

    t = pid_t
    d_start = pid_d * BLOCK_D
    d_offsets = d_start + tl.arange(0, BLOCK_D)
    d_mask = d_offsets < d_model

    # Compute linear offsets for Y and POS
    # Y offset for element (b, t, d): (b*T + t)*d_model + d
    y_offset = (pid_b * T + t) * d_model + d_offsets
    pos_offset = t * d_model + d_offsets

    y_val = tl.load(Y + y_offset, mask=d_mask, other=0.0).to(tl.float32)
    pos_val = tl.load(POS + pos_offset, mask=d_mask, other=0.0).to(tl.float32)
    out = y_val + pos_val
    tl.store(Y + y_offset, out, mask=d_mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        input_features,
        conv2d1_weight, conv2d1_bias,
        conv2d2_weight, conv2d2_bias,
        conv2d3_weight, conv2d3_bias,
        conv_out_weight,
        positional_embedding,
        embed_scale,
    ):
        # Stage 1: Conv2d (1 -> 384) + GELU
        x1 = torch.nn.functional.conv2d(input_features, conv2d1_weight, conv2d1_bias, stride=2, padding=1)
        # PyTorch GELU (default erf-based) for correctness
        x1 = torch.nn.functional.gelu(x1)

        # Stage 2: Conv2d (384 -> 384) + GELU
        x2 = torch.nn.functional.conv2d(x1, conv2d2_weight, conv2d2_bias, stride=2, padding=1)
        x2 = torch.nn.functional.gelu(x2)

        # Stage 3: Conv2d (384 -> 384) + GELU
        x3 = torch.nn.functional.conv2d(x2, conv2d3_weight, conv2d3_bias, stride=2, padding=1)
        x3 = torch.nn.functional.gelu(x3)

        # Reshape: (batch, channels, freq, time) -> (batch, time, channels*freq)
        # Original code uses x.permute(0, 3, 1, 2).contiguous().view(B, T, C*F)
        # After 3 convs, shape is (B, 384, 10, T//8)
        B, C, F, T = x3.size()  # C=384, F=10
        x3 = x3.permute(0, 3, 1, 2).contiguous().view(B, T, C * F)

        # Linear projection to d_model (no bias): original uses F.linear(x, conv_out_weight)
        # conv_out_weight shape is (d_model, conv_out_dim) = (1024, 3840)
        # Output is (B, T, 1024)
        d_model = 1024
        y = torch.nn.functional.linear(x3, conv_out_weight)  # bias=None

        # Scale embeddings
        N = y.numel()
        y_fp32 = y.to(torch.float32)  # do scaling in fp32 for stability
        scale_kernel[(triton.cdiv(N, 1024),)](y_fp32, y_fp32, N, float(embed_scale), BLOCK=1024)

        # Add positional embeddings (slice to T rows and broadcast over d_model)
        pos = positional_embedding  # (max_source_positions=1500, d_model=1024), dtype bfloat16
        # Triton expects tensors on CUDA; ensure device matches y_fp32
        # y_fp32 is likely on the same device as inputs (CUDA). If not, move pos.
        if pos.device != y_fp32.device:
            pos = pos.to(y_fp32.device)

        # Launch add positional embedding kernel: grid over (B, T, d_model tiles)
        BLOCK_D = 128
        grid = (B, T, triton.cdiv(d_model, BLOCK_D))
        add_pos_embed_kernel[grid](y_fp32, pos, B, T, d_model, BLOCK_D=BLOCK_D)

        # Cast back to original dtype (bfloat16) to match expected output dtype
        y_out = y_fp32.to(torch.bfloat16)
        return y_out


def run(*args):
    return ModelNew()(*args)
