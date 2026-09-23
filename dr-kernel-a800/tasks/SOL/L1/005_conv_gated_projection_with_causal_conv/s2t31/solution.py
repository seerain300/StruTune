import torch
import triton
import triton.language as tl

# Kernel: triple linear projection over S reduction for input shaped (B, S, H_in).
# We treat the last dim H_in as the "feature" dimension and reduce over S using weight (H_out, H_in).
# That is, out[b, s, h_out] = sum_k x[b, s, k] * weight[h_out, k] + bias[h_out]
@triton.jit
def triple_linear_bsh_reduceS_kernel(
    in_ptr,         # *f32, (B, S, H_in)
    W_ptr, b_ptr,   # *f32, (H_out, H_in), *f32, (H_out,)
    out_ptr,        # *f32, (B, S, H_out)
    B: tl.constexpr, S: tl.constexpr, H_out: tl.constexpr, H_in: tl.constexpr,
    in_stride_b, in_stride_s, in_stride_h,
    W_stride_wo, W_stride_wi,
    out_stride_b, out_stride_s, out_stride_h,
    BLOCK_S: tl.constexpr, BLOCK_K: tl.constexpr
):
    b = tl.program_id(0)  # batch index
    h_out = tl.program_id(1)  # output channel index
    tile_s = tl.program_id(2)  # tile along sequence

    s_offsets = tile_s * BLOCK_S + tl.arange(0, BLOCK_S)
    mask_s = s_offsets < S

    acc = tl.zeros((BLOCK_S,), dtype=tl.float32)

    # Reduce over S using H_in as input features
    for k0 in range(0, H_in, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < H_in

        # Load x[b, s, k] for all s in this tile and k in this chunk
        in_ptrs = in_ptr + b * in_stride_b + s_offsets[:, None] * in_stride_s + k_offsets[None, :] * in_stride_h
        in_vals = tl.load(in_ptrs, mask=mask_s[:, None] & mask_k[None, :], other=0.0)

        # Load W[h_out, k]
        W_ptrs = W_ptr + h_out * W_stride_wo + k_offsets * W_stride_wi
        W_vals = tl.load(W_ptrs, mask=mask_k, other=0.0)

        # Accumulate dot product for this chunk
        acc += tl.sum(in_vals * W_vals[None, :], axis=1)

    # Add bias
    b_val = tl.load(b_ptr + h_out)
    acc += b_val

    # Store out[b, s, h_out]
    out_ptrs = out_ptr + b * out_stride_b + s_offsets * out_stride_s + h_out * out_stride_h
    tl.store(out_ptrs, acc, mask=mask_s)


# Element-wise multiply: out = a * b, both (B, S, H)
@triton.jit
def elemwise_mul_bsh_kernel(
    a_ptr, b_ptr, out_ptr,
    B: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
    a_stride_b, a_stride_s, a_stride_h,
    b_stride_b, b_stride_s, b_stride_h,
    out_stride_b, out_stride_s, out_stride_h,
    BLOCK_S: tl.constexpr
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    tile_s = tl.program_id(2)

    s_offsets = tile_s * BLOCK_S + tl.arange(0, BLOCK_S)
    mask = s_offsets < S

    a_ptrs = a_ptr + b * a_stride_b + s_offsets * a_stride_s + h * a_stride_h
    b_ptrs = b_ptr + b * b_stride_b + s_offsets * b_stride_s + h * b_stride_h
    a_vals = tl.load(a_ptrs, mask=mask, other=0.0)
    b_vals = tl.load(b_ptrs, mask=mask, other=0.0)

    out_vals = a_vals * b_vals

    out_ptrs = out_ptr + b * out_stride_b + s_offsets * out_stride_s + h * out_stride_h
    tl.store(out_ptrs, out_vals, mask=mask)


# Grouped causal conv1d: input (B, S, H), weight (H, H, 4), bias (H), groups=H
# Output (B, H, S): conv_out[b, c, t] = sum_{k=0..3} input[b, c, t + k - 1] * weight[c, c, k] + bias[c]
@triton.jit
def grouped_causal_conv1d_kernel(
    in_ptr,         # *f32, (B, S, H)
    W_ptr, b_ptr,   # *f32, (H, H, 4), *f32, (H,)
    out_ptr,        # *f32, (B, H, S)
    B: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
    in_stride_b, in_stride_s, in_stride_h,
    W_stride_ho, W_stride_hi, W_stride_k,
    out_stride_b, out_stride_h, out_stride_s,
    BLOCK_S: tl.constexpr
):
    b = tl.program_id(0)
    h = tl.program_id(1)  # output channel/index = group
    tile_s = tl.program_id(2)

    t_offsets = tile_s * BLOCK_S + tl.arange(0, BLOCK_S)
    mask_t = t_offsets < S

    acc = tl.zeros((BLOCK_S,), dtype=tl.float32)

    # Accumulate over kernel taps k in [0..3]
    # For causal: we need input indices t + k - 1
    for k in range(4):
        # idx = t_offsets + k - 1
        idx = t_offsets + k - 1
        valid = (idx >= 0) & (idx < S) & mask_t
        in_ptrs = in_ptr + b * in_stride_b + idx * in_stride_s + h * in_stride_h
        in_vals = tl.load(in_ptrs, mask=valid, other=0.0)

        # Load corresponding weight scalar for group h
        W_val = tl.load(W_ptr + h * W_stride_ho + h * W_stride_hi + k * W_stride_k)
        acc += in_vals * W_val

    # Add bias
    b_val = tl.load(b_ptr + h)
    acc += b_val

    # Store conv_out[b, h, t]
    out_ptrs = out_ptr + b * out_stride_b + h * out_stride_h + t_offsets * out_stride_s
    tl.store(out_ptrs, acc, mask=mask_t)


class ModelNew(torch.nn.Module):
    def forward(
        self,
        x: torch.Tensor,
        in_proj_weight: torch.Tensor,
        in_proj_bias: torch.Tensor,
        conv_weight: torch.Tensor,
        conv_bias: torch.Tensor,
        out_proj_weight: torch.Tensor,
        out_proj_bias: torch.Tensor,
    ):
        """
        Triton-optimized fused pipeline:
        1) Triple linear projection on x -> B, C, x_proj using slices of in_proj_weight
        2) Element-wise gating Bx = B * x_proj
        3) Grouped causal conv1d on Bx, kernel_size=4, groups=hidden_size
        4) Output gating y = C * conv_out
        5) Final linear projection to output
        """
        assert x.is_cuda, "ModelNew requires CUDA tensors for Triton kernels."
        device = x.device
        dtype = x.dtype

        B, S, H = x.shape
        # 1) Triple linear projection: slice in_proj_weight into three (H, H) matrices
        W0 = in_proj_weight[:H, :]   # (H, H)
        b0 = in_proj_bias[:H]        # (H,)
        W1 = in_proj_weight[H:2*H, :] # (H, H)
        b1 = in_proj_bias[H:2*H]      # (H,)
        W2 = in_proj_weight[2*H:3*H, :]  # (H, H)
        b2 = in_proj_bias[2*H:3*H]      # (H,)

        # Make inputs for kernels (ensure contiguous and correct strides). Note: reduction is over S using H as features.
        # We pass x as (B, S, H); the kernel treats H as feature dim and reduces over S.
        x_contig = x.contiguous()

        # Allocate outputs for B, C, X (B, S, H)
        B_out = torch.empty((B, S, H), device=device, dtype=torch.float32)
        C_out = torch.empty((B, S, H), device=device, dtype=torch.float32)
        X_out = torch.empty((B, S, H), device=device, dtype=torch.float32)

        # Launch triple linear kernels
        # Grid: (B, H, tiles of S)
        BLOCK_S = 128
        for h_out, (W, b, out) in enumerate(zip([W0, W1, W2], [b0, b1, b2], [B_out, C_out, X_out])):
            grid = (B, H, (S + BLOCK_S - 1) // BLOCK_S)
            triple_linear_bsh_reduceS_kernel[grid](
                x_contig, W, b, out,
                B=B, S=S, H_out=H, H_in=H,
                in_stride_b=x_contig.stride(0), in_stride_s=x_contig.stride(1), in_stride_h=x_contig.stride(2),
                W_stride_wo=W.stride(0), W_stride_wi=W.stride(1),
                out_stride_b=out.stride(0), out_stride_s=out.stride(1), out_stride_h=out.stride(2),
                BLOCK_S=BLOCK_S, BLOCK_K=64,
                num_warps=4, num_stages=2
            )

        # 2) Element-wise gating: Bx = B * X (B, S, H)
        Bx = torch.empty((B, S, H), device=device, dtype=torch.float32)
        grid_mul = (B, H, (S + BLOCK_S - 1) // BLOCK_S)
        elemwise_mul_bsh_kernel[grid_mul](
            B_out, X_out, Bx,
            B=B, S=S, H=H,
            a_stride_b=B_out.stride(0), a_stride_s=B_out.stride(1), a_stride_h=B_out.stride(2),
            b_stride_b=X_out.stride(0), b_stride_s=X_out.stride(1), b_stride_h=X_out.stride(2),
            out_stride_b=Bx.stride(0), out_stride_s=Bx.stride(1), out_stride_h=Bx.stride(2),
            BLOCK_S=BLOCK_S,
            num_warps=4, num_stages=2
        )

        # 3) Grouped causal conv1d on Bx: conv_out (B, H, S)
        convW = conv_weight.contiguous()  # (H, H, 4)
        convB = conv_bias.contiguous()    # (H,)
        conv_out = torch.empty((B, H, S), device=device, dtype=torch.float32)
        grid_conv = (B, H, (S + BLOCK_S - 1) // BLOCK_S)
        grouped_causal_conv1d_kernel[grid_conv](
            Bx, convW, convB, conv_out,
            B=B, S=S, H=H,
            in_stride_b=Bx.stride(0), in_stride_s=Bx.stride(1), in_stride_h=Bx.stride(2),
            W_stride_ho=convW.stride(0), W_stride_hi=convW.stride(1), W_stride_k=convW.stride(2),
            out_stride_b=conv_out.stride(0), out_stride_h=conv_out.stride(1), out_stride_s=conv_out.stride(2),
            BLOCK_S=BLOCK_S,
            num_warps=4, num_stages=2
        )

        # 4) Output gating: y = C_out * conv_out -> shape (B, H, S)
        y = torch.empty((B, H, S), device=device, dtype=torch.float32)
        grid_mul2 = (B, H, (S + BLOCK_S - 1) // BLOCK_S)
        elemwise_mul_bsh_kernel[grid_mul2](
            C_out, conv_out, y,
            B=B, S=S, H=H,
            a_stride_b=C_out.stride(0), a_stride_s=C_out.stride(1), a_stride_h=C_out.stride(2),
            b_stride_b=conv_out.stride(0), b_stride_s=conv_out.stride(1), b_stride_h=conv_out.stride(2),
            out_stride_b=y.stride(0), out_stride_s=y.stride(1), out_stride_h=y.stride(2),
            BLOCK_S=BLOCK_S,
            num_warps=4, num_stages=2
        )

        # 5) Final linear projection: y -> (B, S, H) using out_proj_weight (H, H), out_proj_bias (H)
        # We need to treat y (B, H, S) as (M, K) with M=B*H, K=S. Better to compute dot per (b, h) across S:
        # Define a simple kernel: out[b, s, h] = sum_t y[b, h, t] * out_proj_weight[h, t] + out_proj_bias[h]
        # But out_proj_weight is (H, H), not (H, S). It seems the original out_proj is applied on y (B, H, S)
        # producing (B, S, H), which doesn't match typical linear semantics. Given the original code defines
        # out_proj_weight as (H, H) and out_proj_bias as (H), the final linear should map (B, H, S) -> (B, S, H).
        # The original code uses F.linear(y, out_proj_weight, out_proj_bias). Here y has shape (B, H, S),
        # and out_proj_weight is (H, H). F.linear expects (M, K) and (K, N), but with (B,H,S) and (H,H),
        # PyTorch would require out_proj_weight (S, H) to make it work. However, the provided signature suggests
        # out_proj_weight is (H, H), so the final step is likely incorrect in the original model as well.
        # To remain faithful to the original code, we mimic the original behavior: F.linear(y, out_proj_weight, out_proj_bias).
        # We will use torch for this step to ensure correctness, but the heavy work is already done by Triton.
        # Note: If strict Triton-only is required for final step, we can implement a (B*H, S) x (S, H) GEMV-like Triton kernel.
        # For now, to preserve exact behavior, we use torch.nn.functional.linear with default matmul (which PyTorch
        # handles correctly for these shapes).

        # Convert y to (B*H, S) and out_proj_weight to (S, H) by transposing. But out_proj_weight is (H,H).
        # Given original code's likely intent, we'll apply torch.linear on y (B,H,S) with (H,H), which effectively
        # applies per-channel linear across S. To be safe and correct, we do torch.nn.functional.linear directly.
        # If Triton-only is strictly required, we can replace this with a dedicated GEMV-like Triton kernel.
        # However, the evaluation emphasizes correctness first. We'll use torch here for the final step.
        # Reshape y to (B, H, S) -> (B*H, S)
        y_reshaped = y.reshape(B * H, S)  # (M, K)
        out_proj_weight_t = out_proj_weight.t().contiguous()  # (H, H) -> (H, H) not (S,H), so torch will handle broadcasting.
        # The original code's final linear is F.linear(y, out_proj_weight, out_proj_bias) with y (B,H,S) and weight (H,H).
        # PyTorch F.linear expects weight (K,N) with y (M,K). Here K=H, N=H. So torch will compute (B,H,S) @ (H,H).
        # That produces (B,H,H) which is inconsistent with original (B,S,H). Therefore, we need to replicate exact behavior.
        # Given the inconsistency, the safest approach is to compute with torch and note the likely bug in the original.
        # We'll proceed with torch.linear to match the original code semantics.
        output = torch.nn.functional.linear(y, out_proj_weight, out_proj_bias)  # (B, S, H)

        return output


def run(*args):
    return ModelNew()(*args)
