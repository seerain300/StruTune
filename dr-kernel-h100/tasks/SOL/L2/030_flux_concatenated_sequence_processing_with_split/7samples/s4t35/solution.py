import torch
import triton
import triton.language as tl

@triton.jit
def compute_matvec_per_token(
    encoder_ptr,           # *f32 [B, T, H]
    image_ptr,             # *f32 [B, I, H]
    weight_ptr,            # *f32 [H, H]
    out_ptr,               # *f32 [B, T+I, H]
    B: tl.int32,
    T: tl.int32,
    I: tl.int32,
    H: tl.int32,
    # strides (in elements)
    encoder_stride_b, encoder_stride_t, encoder_stride_h,
    image_stride_b, image_stride_i, image_stride_h,
    weight_stride_h, weight_stride_k,   # weight is [H, H], so weight_stride_k == weight_stride_h
    out_stride_b, out_stride_t, out_stride_h,
    BLOCK_H: tl.constexpr,   # tile size along hidden dimension
    BLOCK_K: tl.constexpr,   # tile size along K (used in accumulation, but we iterate over K with step BLOCK_K inside H-loop; here K==H so we use BLOCK_H for K as well)
):
    # Each program handles one (b, t)
    b = tl.program_id(0)   # batch index
    t = tl.program_id(1)   # output token index in [0, T+I)

    # Compute output vector for this (b, t) across H
    # We loop over H in tiles
    for h_off in range(0, H, BLOCK_H):
        offs_h = h_off + tl.arange(0, BLOCK_H)
        mask_h = offs_h < H

        # Accumulator for this H-tile
        acc = tl.zeros((BLOCK_H,), dtype=tl.float32)

        # Now, for each hidden feature in this tile, accumulate dot products with input vector and weight
        # Since we want y[h] = sum_k W[h, k] * x[k], we loop h locally and sum over k (but Triton requires static tiling).
        # Instead, we load weight tiles [BLOCK_H, BLOCK_H] and input chunks [BLOCK_H] and do outer product-like accumulation.
        # Note: Here, K ranges over the same H (no bias), so we iterate across H tiles with inner loop stepping by BLOCK_H:
        # However, Triton prefers compile-time loops; to keep it simple and safe, we use nested loops over h and k with masks.
        # This approach avoids complex broadcasting and reduces risk of illegal memory access.

        # For each hidden index h in this tile, accumulate over K=H by stepping BLOCK_H at a time.
        # This is done to comply with Triton's requirement for compile-time ranges. In practice, we iterate over h explicitly.
        # But Triton requires tl.arange-based vectors for loads; thus we implement as:
        for k_off in range(0, H, BLOCK_H):  # iterate K across H
            offs_k = k_off + tl.arange(0, BLOCK_H)
            mask_k = offs_k < H

            # Select source tensor based on whether t comes from encoder (t < T) or image (t >= T)
            # If t >= T, we read from image at position t - T
            # Compute the pointer for x_vec: either encoder[b, t, :] or image[b, t - T, :]
            if t < T:
                # Load input vector from encoder
                # pointer = encoder_ptr + b*encoder_stride_b + t*encoder_stride_t + k*encoder_stride_h
                x_vec = tl.load(encoder_ptr + b * encoder_stride_b + t * encoder_stride_t + offs_k * encoder_stride_h, mask=mask_k, other=0.0).to(tl.float32)  # [BLOCK_H]
            else:
                t_img = t - T  # position in image stream
                x_vec = tl.load(image_ptr + b * image_stride_b + t_img * image_stride_i + offs_k * image_stride_h, mask=mask_k, other=0.0).to(tl.float32)  # [BLOCK_H]

            # Load weight block W[h, k] for h in this tile and k in this tile
            # pointer = weight_ptr + h_off*weight_stride_h + k_off*weight_stride_k
            W_block = tl.load(weight_ptr + offs_h[:, None] * weight_stride_h + offs_k[None, :] * weight_stride_k, mask=mask_h[:, None] & mask_k[None, :], other=0.0).to(tl.float32)  # [BLOCK_H, BLOCK_H]

            # Accumulate: acc[h] += sum_k (W_block[h, k] * x_vec[k])
            # Since x_vec is [BLOCK_H] corresponding to k indices, we align by k_off:
            # We need to index x_vec at each k; Triton allows multiplying W_block with x_vec broadcast along k.
            # But here x_vec is a vector; to accumulate correctly, we do per-k contribution by summing over k axis.
            # The outer-product-like form is W_block * x_vec[None, :], then sum over axis=1:
            contribution = tl.sum(W_block * x_vec[None, :], axis=1)  # [BLOCK_H]
            acc += contribution

        # Store the accumulated acc to output for this (b, t, h_off..)
        # Pointer to out[b, t, h] is out_ptr + b*out_stride_b + t*out_stride_t + h_off*out_stride_h
        tl.store(out_ptr + b * out_stride_b + t * out_stride_t + offs_h * out_stride_h, acc, mask=mask_h)

class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version of the original run function.
        Returns (processed_encoder_hidden_states, processed_image_hidden_states).
        """
        # Ensure inputs are CUDA tensors and contiguous
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Inputs must be on CUDA for Triton execution."
        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        H = encoder_hidden_states.shape[2]
        assert process_weight.shape[0] == H and process_weight.shape[1] == H, "process_weight must be [H, H]."

        # Make inputs contiguous
        encoder = encoder_hidden_states.contiguous()
        image = hidden_states.contiguous()
        weight = process_weight.contiguous()

        # Allocate output [B, T+I, H]
        out = torch.empty((B, T + I, H), device=encoder.device, dtype=torch.float32)

        # Launch Triton kernel: one program per (batch, output token)
        grid = (B, T + I)
        # Choose conservative tile sizes; tune if needed
        BLOCK_H = 64
        BLOCK_K = 64  # Here K==H looped via python for-loops; BLOCK_K is used for load tile size

        compute_matvec_per_token[grid](
            encoder, image, weight, out,
            B, T, I, H,
            encoder.stride(0), encoder.stride(1), encoder.stride(2),
            image.stride(0), image.stride(1), image.stride(2),
            weight.stride(0), weight.stride(1),   # weight is [H, H], so stride(1) is stride along H
            out.stride(0), out.stride(1), out.stride(2),
            BLOCK_H=BLOCK_H, BLOCK_K=BLOCK_K,
            num_warps=2,
        )

        # Split outputs
        processed_encoder = out[:, :T, :]
        processed_hidden = out[:, T:, :]

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
