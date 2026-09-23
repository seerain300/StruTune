import math
import torch
import triton
import triton.language as tl


@triton.jit
def linear_gemv_nobias_kernel(
    x_ptr,          # *bfloat16, shape [B*T, K]
    w_ptr,          # *bfloat16, shape [d_model, K]
    y_ptr,          # *bfloat16, shape [B*T, d_model]
    B: tl.constexpr,
    T: tl.constexpr,
    K: tl.constexpr,
    d_model: tl.constexpr,
    BLOCK_D: tl.constexpr,  # tile size along d_model
):
    # Grid: (outer=B*T, d_chunks=d_model // BLOCK_D)
    outer = tl.program_id(0)
    d_chunk = tl.program_id(1)

    b = outer // T
    t = outer % T

    base_x = b * T + t  # row index in x (flattened B*T)
    # Accumulator for this (b, t)
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)  # accumulate in fp32 for better precision

    # Each program handles a tile of d_model features
    d_offsets = d_chunk * BLOCK_D + tl.arange(0, BLOCK_D)
    valid_d = d_offsets < d_model

    # For each k in K, load x[base_x, k] and w[d_offsets, k], accumulate acc += x * w
    # Unrolled loop over K (K is typically small here; Triton can handle this)
    for k in range(0, K):
        x_val = tl.load(x_ptr + base_x * K + k, mask=True, other=0.0).to(tl.float32)
        w_vec = tl.load(w_ptr + d_offsets * K + k, mask=valid_d, other=0.0).to(tl.float32)
        acc += x_val * w_vec

    # Store results to y[b*T + t, d_offsets]
    y_base = b * T + t
    y_ptrs = y_ptr + y_base * d_model + d_offsets
    tl.store(y_ptrs, acc, mask=valid_d)


@triton.jit
def scale_mul_kernel(y_ptr, scale, B, T, d_model):
    # Scale y[b, t, d] by scale (fp32)
    outer = tl.program_id(0)  # 0..B*T-1
    d = tl.program_id(1)      # 0..d_model-1
    # y index: outer * d_model + d
    y_index = outer * d_model + d
    # Load y, scale, store
    y_val = tl.load(y_ptr + y_index).to(tl.float32)
    y_val = y_val * scale
    tl.store(y_ptr + y_index, y_val)


@triton.jit
def add_pos_emb_kernel(y_ptr, pos_ptr, B, T, d_model):
    # y_ptr: [B*T*d_model] linear
    # pos_ptr: [T*d_model] linear
    outer = tl.program_id(0)  # 0..B*T-1
    d = tl.program_id(1)      # 0..d_model-1
    y_index = outer * d_model + d
    pos_index = (outer % T) * d_model + d
    y_val = tl.load(y_ptr + y_index).to(tl.float32)
    pos_val = tl.load(pos_ptr + pos_index).to(tl.float32)
    y_val = y_val + pos_val
    tl.store(y_ptr + y_index, y_val)


class ModelNew(torch.nn.Module):
    def forward(self, input_features, conv2d1_weight, conv2d1_bias,
                conv2d2_weight, conv2d2_bias, conv2d3_weight, conv2d3_bias,
                conv_out_weight, positional_embedding, embed_scale):
        """
        input_features: [B, 1, 80, time_dim], bfloat16
        conv* weights: [C_out, C_in, 3, 3], bfloat16
        conv* biases: [C_out], bfloat16
        conv_out_weight: [d_model, K], bfloat16, K=C_out3*H_out3*W_out3 (dynamic)
        positional_embedding: [max_source_positions, d_model], bfloat16
        embed_scale: float
        """

        # Perform convolutions using PyTorch to match original behavior (heavy, but ensures correctness)
        x1 = torch.nn.functional.conv2d(input_features, conv2d1_weight, conv2d1_bias, stride=2, padding=1)
        x2 = torch.nn.functional.conv2d(x1, conv2d2_weight, conv2d2_bias, stride=2, padding=1)
        x3 = torch.nn.functional.conv2d(x2, conv2d3_weight, conv2d3_bias, stride=2, padding=1)

        # Reshape x3 from [B, C, F, T] -> [B, T, K] where T = x3.size(-1), K = C*x3.size(-2)
        # x3: [B, 384, H_out3, W_out3]
        B, C, F, T = x3.shape  # C=384, F=H_out3, T=W_out3
        K = C * F  # dynamic per workload

        # We need x3 in shape [B, T, K]. Use .contiguous() then view; this does not change data layout.
        x3_reshaped = x3.permute(0, 3, 1, 2).contiguous()  # [B, T, 384, F]
        x2d = x3_reshaped.view(B, T, K).contiguous()      # [B, T, K]

        # Prepare conv_out_weight: [d_model, K], ensure on correct device/dtype
        d_model = conv_out_weight.shape[0]
        w = conv_out_weight  # [d_model, K]

        # Allocate output y: [B, T, d_model]
        y = torch.empty((B, T, d_model), dtype=x2d.dtype, device=x2d.device)

        # Launch Triton kernels: linear GEMV (no bias), scale, add positional embedding
        # For safety, pass B, T, K, d_model as constexpr-like ints. Triton treats them as runtime scalars, but loops
        # over K and d_model must be known for compilation. To keep robust, we set BLOCK_D=64 (fits d_model=1024).
        BLOCK_D = 64

        # Compute grid for linear: (B*T, ceil(d_model/BLOCK_D))
        grid_linear = (B * T, triton.cdiv(d_model, BLOCK_D))
        # y and w are bfloat16; Triton loads as float and we'll accumulate in float32, then store as float32.
        # To keep output dtype consistent, we cast y to bfloat16 after linear. The linear kernel will write float32
        # but we need bfloat16. So we'll allocate y as bfloat16, cast acc to bfloat16 on store.

        # Adjust: We'll cast acc to bfloat16 before store.
        # Prepare y as bfloat16
        y = torch.empty((B, T, d_model), dtype=torch.bfloat16, device=x2d.device)

        # Run linear kernel; pass pointers and grid
        # Note: Triton expects * pointers; x2d and w are bfloat16 tensors. We'll load and convert to float32 for acc,
        # then store as bfloat16 by casting acc.
        # Linear kernel signature expects y_ptr as bfloat16; we store acc (float32) cast to bfloat16.
        # We'll do this by creating a wrapper in Triton: compute in float32, cast to bfloat16 before store.
        # To do that cleanly, we'll change linear kernel to store acc.to(tl.bfloat16). However, Triton's store expects
        # the same pointer dtype; safer approach: have the kernel write float32 into a float32 y, then cast on host.
        # Given constraints, we'll allocate a float32 y_tmp for the kernel, and cast back to bfloat16 after.

        # Allocate float32 temporary for linear output
        y_tmp = torch.empty((B, T, d_model), dtype=torch.float32, device=x2d.device)

        # Cast x2d and w to float32 for kernel computation (load as bfloat16, convert to float32)
        # Triton loads from memory; we ensure x2d and w are contiguous and on device.
        # Pass as is; Triton loads bytes and we use .to(tl.float32) on loads. We don't have dtype in kernel; rely on
        # load returning tensor, then cast.
        # Run kernel
        linear_gemv_nobias_kernel[grid_linear](
            x2d, w, y_tmp, B, T, K, d_model, BLOCK_D,
            num_warps=4, num_stages=2,
        )

        # Scale y_tmp by embed_scale (float32 scalar)
        # We need y as float32 then cast to bfloat16 for final output. However, original output should be bfloat16.
        # Since get_inputs uses bfloat16, we cast to bfloat16 at the end.
        # Scale in Triton
        # Prepare a scaled float32 buffer; then cast to bfloat16 before adding pos.
        y_scaled = y_tmp

        # Scale kernel: grid (B*T, d_model)
        grid_scale = (B * T, d_model)
        # y_scaled is float32; we scale in-place
        scale_mul_kernel[grid_scale](
            y_scaled, float(embed_scale), B * T, d_model, num_warps=4, num_stages=2,
        )

        # Add positional embedding in Triton: pos slice is [T, d_model]
        pos = positional_embedding[:T, :].contiguous()  # [T, d_model], bfloat16
        # Linearize pos to [T*d_model]
        pos_flat = pos.view(-1)  # length = T * d_model

        # Add in Triton
        grid_add = (B * T, d_model)
        add_pos_emb_kernel[grid_add](
            y_scaled, pos_flat, B, T, d_model, num_warps=4, num_stages=2,
        )

        # Final result should be bfloat16. Cast y_scaled to bfloat16.
        y = y_scaled.to(torch.bfloat16)

        return y


def run(*args):
    return ModelNew()(*args)
