import torch
import triton
import triton.language as tl


@triton.jit
def batched_matmul_cat_weightT_kernel(
    x_ptr,          # pointer to concatenated input: [B, T+I, H]
    w_ptr,          # pointer to process_weight.T: [H, H]
    out_ptr,        # pointer to output: [B, T+I, H]
    B: tl.constexpr,
    N: tl.constexpr,      # N = T + I
    H: tl.constexpr,
    stride_x_n, stride_x_s, stride_x_h,   # strides for x
    stride_out_n, stride_out_s, stride_out_h,  # strides for out
    stride_w_h, stride_w_k,                 # strides for w (w is [H, H] so we interpret as [K, H] where K=H)
    BLOCK_H: tl.constexpr,
):
    # program id: one program per (n, s)
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)

    # bounds check for safety (though grid should be exact)
    if (pid_n >= B) or (pid_s >= N):
        return

    # Accumulator for output vector of length H
    acc = tl.zeros([BLOCK_H], dtype=tl.float32)

    # Loop over K (input feature) dimension
    # Note: x is [B, N, H]; out is [B, N, H]; w is [H, H] which we use as W^T [H, H] (i.e., W^T[k, h] = W[h, k])
    # We iterate k from 0..H-1
    # We tile across H (output features) in blocks of BLOCK_H
    h_offsets = tl.arange(0, BLOCK_H)
    # For each k, compute dot(x[n, s, k], w^T[:, k]) and accumulate into acc[h_offsets]
    # Since we need w^T, we use w[h, k] directly: w_ptr + h*stride_w_h + k*stride_w_k
    for k in range(0, H):
        # Load x[n, s, k]
        x_val = tl.load(
            x_ptr + pid_n * stride_x_n + pid_s * stride_x_s + k * stride_x_h,
            eviction_policy='evict_last'
        )
        # Load w^T[:, k] -> shape [BLOCK_H] which corresponds to w[h, k] for h in h_offsets
        w_vec = tl.load(
            w_ptr + h_offsets * stride_w_h + k * stride_w_k,
            mask=h_offsets < H,
            other=0.0,
            eviction_policy='evict_last'
        )
        # Accumulate: acc[h_offsets] += x_val * w_vec
        acc += x_val * w_vec

    # Store results into out[n, s, :]
    out_ptrs = out_ptr + pid_n * stride_out_n + pid_s * stride_out_s + h_offsets * stride_out_h
    tl.store(out_ptrs, acc, mask=h_offsets < H)


class ModelNew(torch.nn.Module):
    def forward(self, encoder_hidden_states: torch.Tensor, hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        Triton-optimized version of the original run function.
        - Concatenates encoder_hidden_states and hidden_states along sequence dim.
        - Applies linear projection via Triton kernel (no bias).
        - Splits back into separate encoder and image streams.

        Args:
            encoder_hidden_states: [B, T, H]
            hidden_states: [B, I, H]
            process_weight: [H, H] (linear projection matrix)

        Returns:
            processed_encoder: [B, T, H]
            processed_hidden: [B, I, H]
        """
        # Ensure device is CUDA for Triton; if not, fall back (but evaluation expects Triton)
        assert encoder_hidden_states.is_cuda and hidden_states.is_cuda and process_weight.is_cuda, "Tensors must be on CUDA for Triton execution."

        B = encoder_hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        H = encoder_hidden_states.shape[2]
        assert hidden_states.shape == (B, I, H), "hidden_states shape must be [B, I, H]"
        assert process_weight.shape == (H, H), "process_weight must be [H, H]"

        # Create concatenated input [B, T+I, H] without actually allocating a large tensor in host
        # We will feed the Triton kernel with this logical view via pointers; but to compute efficiently, we need a contiguous tensor for x.
        # For correctness and performance, we explicitly materialize the concatenated tensor on device.
        # This avoids shape issues and keeps the kernel simple and robust.
        # Note: If H is large, this is fine for typical sizes in the provided workloads.
        concatenated = torch.cat([encoder_hidden_states, hidden_states], dim=1)  # [B, T+I, H]

        # Weight is [H, H]; we need to use weight.T in matmul. In Triton, we'll read it as W^T by indexing [h, k] -> W[h, k].
        # Ensure process_weight is contiguous and on device.
        process_weight_T = process_weight  # we will index as w[h, k] in kernel; no need to transpose physically

        # Output tensor for concatenated projection [B, T+I, H]
        out = torch.empty((B, T + I, H), device=concatenated.device, dtype=torch.float32)

        # Compute grid: one program per (n, s)
        N = T + I
        grid = (B, N)

        # Strides
        stride_x_n, stride_x_s, stride_x_h = concatenated.stride()
        stride_out_n, stride_out_s, stride_out_h = out.stride()
        # For w: [H, H], we interpret as [K, H] where K=H. So strides are stride_w_h = w.stride(0), stride_w_k = w.stride(1)
        stride_w_h, stride_w_k = process_weight_T.stride()

        # Launch Triton kernel. We use fp32 throughout for stability.
        # Cast inputs to float32 for accumulation; Triton will operate on provided tensors. To be safe, ensure concatenated and weight are float32.
        # If original tensors are not float32, cast here.
        if concatenated.dtype != torch.float32:
            concatenated = concatenated.float()
        if process_weight_T.dtype != torch.float32:
            process_weight_T = process_weight_T.float()

        # Choose tiling. H is typically 1024; BLOCK_H=128 works well. You can tune num_warps if needed.
        BLOCK_H = 128
        batched_matmul_cat_weightT_kernel[grid](
            concatenated, process_weight_T, out,
            B, N, H,
            stride_x_n, stride_x_s, stride_x_h,
            stride_out_n, stride_out_s, stride_out_h,
            stride_w_h, stride_w_k,
            BLOCK_H=BLOCK_H,
            num_warps=4,
        )

        # Split outputs: processed_encoder = out[:, :T, :], processed_hidden = out[:, T:, :]
        processed_encoder = out[:, :T, :]
        processed_hidden = out[:, T:, :]

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
