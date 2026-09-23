import torch
import triton
import triton.language as tl

# Robust Triton kernel: compute processed = cat([encoder, hidden], dim=1) @ process_weight.T
# One program per (batch b, output token t). For each t, decide source from encoder or hidden.
@triton.jit
def concat_linear_per_token_kernel(
    image_ptr,           # *f16/f32 [B, I, H]
    encoder_ptr,         # *f16/f32 [B, T, H]
    weight_ptr,          # *f16/f32 [H, H]
    out_ptr,             # *f32 [B, T+I, H]  (we compute in float32 for stability)
    B: tl.int32,
    I: tl.int32,
    T: tl.int32,
    H: tl.int32,
    # strides (in elements)
    image_stride_b, image_stride_i, image_stride_h,
    encoder_stride_b, encoder_stride_t, encoder_stride_h,
    weight_stride_w, weight_stride_k,   # weight is [H, H]
    out_stride_b, out_stride_t, out_stride_h,
    # meta-parameters
    BLOCK_H: tl.constexpr,
):
    pid_b = tl.program_id(0)   # batch index
    pid_t = tl.program_id(1)   # output token index in [0, T+I)
    # Decide source: if pid_t < T, use encoder; else use hidden
    use_encoder = pid_t < T
    # Base pointers for this row
    base_image = image_ptr + pid_b * image_stride_b
    base_encoder = encoder_ptr + pid_b * encoder_stride_b

    # Output vector for this token, float32
    output_vec = tl.zeros((H,), dtype=tl.float32)

    # Loop over hidden dimension in tiles
    for h_off in range(0, H, BLOCK_H):
        offs_h = h_off + tl.arange(0, BLOCK_H)
        mask_h = offs_h < H
        # Initialize accumulator for this tile
        acc = tl.zeros((BLOCK_H,), dtype=tl.float32)

        # Loop over input dimension (K/H) in tiles to compute dot-product
        # For each hidden tile h_off, compute dot with input vector at position t_src
        # Determine source row index
        t_src = pid_t if use_encoder else (pid_t - T)
        # Compute input pointer for this row
        if use_encoder:
            x_ptr = base_encoder + t_src * encoder_stride_t
        else:
            x_ptr = base_image + (t_src - I) * image_stride_i  # t_src is t - T in [0, I)

        # Loop over K in tiles to compute the dot-product for this hidden tile
        for k_off in range(0, H, BLOCK_H):  # Note: weight is [H, H], so K=H
            offs_k = k_off + tl.arange(0, BLOCK_H)
            mask_k = offs_k < H

            # Load weight tile [BLOCK_H, BLOCK_H]
            w_ptrs = weight_ptr + (offs_h[:, None] * weight_stride_w + offs_k[None, :] * weight_stride_k)
            w_tile = tl.load(w_ptrs, mask=mask_h[:, None] & mask_k[None, :], other=0.0)
            w_tile = w_tile.to(tl.float32)

            # Load input vector chunk [BLOCK_H]
            x_ptrs = x_ptr + offs_k * (1 if use_encoder else image_stride_h)  # stride along hidden dim
            x_vec = tl.load(x_ptrs, mask=mask_k, other=0.0).to(tl.float32)  # [BLOCK_H]

            # Accumulate: acc[h] += sum_k w_tile[h, k] * x_vec[k]
            acc += tl.sum(w_tile * x_vec[None, :], axis=1)

        # Update output vector for this hidden tile
        output_vec = output_vec + acc

    # Store the output vector for this (b, t)
    out_ptrs = out_ptr + pid_b * out_stride_b + pid_t * out_stride_t + tl.arange(0, H) * out_stride_h
    mask_out = tl.arange(0, H) < H  # always true, but keep for safety
    tl.store(out_ptrs, output_vec, mask=mask_out)

# Optional: original PyTorch reference computation (for validation only, not used by default)
@torch.no_grad()
def reference_run(encoder_hidden_states: torch.Tensor, hidden_states: torch.Tensor, process_weight: torch.Tensor) -> torch.Tensor:
    text_seq_len = encoder_hidden_states.shape[1]
    img_seq_len = hidden_states.shape[1]
    concatenated = torch.cat([encoder_hidden_states, hidden_states], dim=1)  # [B, T+I, H]
    processed = torch.matmul(concatenated, process_weight.t())              # [B, T+I, H]
    return processed

class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> torch.Tensor:
        """
        Triton-optimized version: computes processed = (cat([encoder, hidden], dim=1)) @ process_weight.T
        and returns it split into (processed_encoder, processed_hidden).
        """
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Inputs must be on CUDA for Triton execution."

        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        H = hidden_states.shape[2]

        # We compute output in float32 for stability. We'll cast to original dtype after if needed.
        out = torch.empty((B, T + I, H), device=hidden_states.device, dtype=torch.float32)

        # Launch Triton kernel: one program per (b, t)
        grid = (B, T + I)
        concat_linear_per_token_kernel[grid](
            encoder_hidden_states, hidden_states, process_weight, out,
            B, I, T, H,
            encoder_hidden_states.stride(0), encoder_hidden_states.stride(1), encoder_hidden_states.stride(2),
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            process_weight.stride(0), process_weight.stride(1),
            out.stride(0), out.stride(1), out.stride(2),
            BLOCK_H=64,  # tune as needed
            num_warps=4, num_stages=2
        )

        # Split into two streams and cast to the input dtype if desired. The original returns float tensors by default.
        processed_encoder = out[:, :T, :]
        processed_hidden = out[:, T:, :]
        return processed_encoder, processed_hidden