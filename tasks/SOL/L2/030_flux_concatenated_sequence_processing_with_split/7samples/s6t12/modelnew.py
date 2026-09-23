import torch
import triton
import triton.language as tl


@triton.jit
def batched_matmul_row_kernel(
    x_ptr,      # [B, L, H], float32, contiguous, where L = T + I
    w_ptr,      # [H, H], float32, contiguous
    out_ptr,    # [B, L, H], float32, contiguous
    B: tl.int32, L: tl.int32, H: tl.int32,
    stride_x_n: tl.int32, stride_x_s: tl.int32, stride_x_h: tl.int32,
    stride_w_h: tl.int32, stride_w_k: tl.int32,
    stride_out_n: tl.int32, stride_out_s: tl.int32, stride_out_h: tl.int32,
):
    # Grid: (B, L). Each program computes one output row (n, s).
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)

    # Prepare base pointers for this row
    x_row_ptr = x_ptr + pid_n * stride_x_n + pid_s * stride_x_s
    out_row_ptr = out_ptr + pid_n * stride_out_n + pid_s * stride_out_s

    # Accumulator vector for output (float32)
    acc = tl.zeros((H,), dtype=tl.float32)

    # Loop over input features k = 0..H-1
    # For each k, load x[n, s, k] and multiply by weight[k, :]
    k = 0
    while k < H:
        # Load input scalar x[n, s, k]
        x_val = tl.load(x_row_ptr + k * stride_x_h)
        # Load weight row weight[k, :]
        w_row_ptr = w_ptr + k * stride_w_h
        w_vec = tl.load(w_row_ptr + tl.arange(0, H) * stride_w_k)
        # Accumulate: out[n, s, :] += x_val * w_vec
        acc += x_val * w_vec
        k += 1

    # Store the accumulated output row
    tl.store(out_row_ptr + tl.arange(0, H) * stride_out_h, acc)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                encoder_hidden_states: torch.Tensor,
                process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version of the original run function.
        - Concatenate encoder_hidden_states and hidden_states along the sequence dimension.
        - Compute processed = concatenated @ process_weight.T using a Triton kernel.
        - Split into processed_encoder and processed_hidden.
        """
        # Ensure inputs are on the same device and dtype, and contiguous
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "All tensors must be on CUDA."
        assert hidden_states.dtype == torch.float32 and encoder_hidden_states.dtype == torch.float32 and process_weight.dtype == torch.float32, "Use float32 tensors."

        # Concatenate along sequence dimension: [B, T+I, H]
        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        H = hidden_states.shape[2]
        total_seq = T + I

        x = torch.cat([encoder_hidden_states, hidden_states], dim=1).contiguous()

        # Allocate output [B, T+I, H], float32
        out = torch.empty((B, total_seq, H), device=hidden_states.device, dtype=hidden_states.dtype)

        # Get strides (for contiguous tensors, strides are simple)
        # x: [B, T+I, H]
        stride_x_n, stride_x_s, stride_x_h = x.stride()
        # process_weight: [H, H]
        w = process_weight.contiguous()
        stride_w_h, stride_w_k = w.stride()
        # out: [B, T+I, H]
        stride_out_n, stride_out_s, stride_out_h = out.stride()

        # Launch Triton kernel over (B, T+I)
        grid = (B, total_seq)
        batched_matmul_row_kernel[grid](
            x, w, out,
            B, total_seq, H,
            stride_x_n, stride_x_s, stride_x_h,
            stride_w_h, stride_w_k,
            stride_out_n, stride_out_s, stride_out_h,
            num_warps=4,
        )

        # Split outputs into encoder and hidden streams
        processed_encoder = out[:, :T, :]
        processed_hidden = out[:, T:, :]

        return processed_encoder, processed_hidden