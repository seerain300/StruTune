import torch
import triton
import triton.language as tl


@triton.jit
def batched_matmul_cat_with_weightT_kernel(
    encoder_ptr,   # [B, T, H], float32, contiguous
    hidden_ptr,    # [B, I, H], float32, contiguous
    weight_ptr,    # [H, H], float32, contiguous
    out_ptr,       # [B, T+I, H], float32, contiguous
    B: tl.int32, T: tl.int32, I: tl.int32, H: tl.int32,
    stride_e_n: tl.int32, stride_e_s: tl.int32, stride_e_h: tl.int32,
    stride_h_n: tl.int32, stride_h_s: tl.int32, stride_h_h: tl.int32,
    stride_w_h: tl.int32, stride_w_k: tl.int32,
    stride_out_n: tl.int32, stride_out_s: tl.int32, stride_out_h: tl.int32,
    BLOCK_H: tl.constexpr,
):
    # Grid: (B, T+I). Each program computes one output row (n, s).
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)
    total_seq = T + I

    # Determine which source tensor to use for this sequence position
    use_encoder = pid_s < T

    # Base pointer for the input row
    if use_encoder:
        src_ptr = encoder_ptr + pid_n * stride_e_n + pid_s * stride_e_s
        # Input vector: encoder_hidden_states[n, s, :]
        input_vec = tl.load(src_ptr + tl.arange(0, H) * stride_e_h)
    else:
        src_ptr = hidden_ptr + pid_n * stride_h_n + (pid_s - T) * stride_h_s
        # Input vector: hidden_states[n, s - T, :]
        input_vec = tl.load(src_ptr + tl.arange(0, H) * stride_h_h)

    # Prepare output accumulator vector (float32)
    acc = tl.zeros((BLOCK_H,), dtype=tl.float32)

    # Multiply by weight.T: out[n, s, :] = input_vec @ weight.T
    # weight is [H, H], so weight.T is [H, H] with strides swapped in access: weight[k, h]
    for k in range(0, H):
        # Load weight row for k: [h] vector of size H
        w_row = tl.load(weight_ptr + k * stride_w_h + tl.arange(0, H) * stride_w_k)
        # Accumulate: sum_j input_vec[j] * weight[j, h]
        acc += input_vec[k] * w_row

    # Store the result to out[n, s, :]
    out_row_ptr = out_ptr + pid_n * stride_out_n + pid_s * stride_out_s
    tl.store(out_row_ptr + tl.arange(0, BLOCK_H) * stride_out_h, acc)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                encoder_hidden_states: torch.Tensor,
                process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton implementation of the original run function:
        - Concatenate encoder_hidden_states and hidden_states along the sequence dimension
        - Apply linear projection with process_weight.T
        - Split back into encoder and hidden outputs
        """
        # Ensure dtypes and contiguity
        assert hidden_states.dtype == torch.float32 and encoder_hidden_states.dtype == torch.float32 and process_weight.dtype == torch.float32
        e = encoder_hidden_states.contiguous()
        h = hidden_states.contiguous()
        w = process_weight.contiguous()

        B, T, H = e.shape
        I, _, Hh = h.shape
        assert H == Hh, "hidden_dim mismatch"
        # Output: [B, T+I, H]
        out = torch.empty((B, T + I, H), device=e.device, dtype=torch.float32)

        # Strides (in elements) for contiguous tensors
        stride_e_n, stride_e_s, stride_e_h = e.stride()
        stride_h_n, stride_h_s, stride_h_h = h.stride()
        # weight is [H, H]
        stride_w_h, stride_w_k = w.stride()  # for [H, H], usually (H, 1)
        stride_out_n, stride_out_s, stride_out_h = out.stride()

        # Launch Triton kernel: grid over (batch, sequence positions)
        grid = (B, T + I)
        # BLOCK_H covers the hidden_dim; for generality, use H directly. Since H can be dynamic, we set BLOCK_H=128 and rely on mask-like loop but here we iterate exact H, so BLOCK_H must be >= H.
        # To be safe across all workloads, choose BLOCK_H = 128 and rely on that H <= 128 for typical dims (and masking still works because we iterate exact H; Triton will compile for specific H at runtime).
        # If H > 128, we can fallback or increase BLOCK_H. For given workloads (H up to 1024), we choose 256 for robustness.
        BLOCK_H = 256

        batched_matmul_cat_with_weightT_kernel[grid](
            e, h, w, out,
            B, T, I, H,
            stride_e_n, stride_e_s, stride_e_h,
            stride_h_n, stride_h_s, stride_h_h,
            stride_w_h, stride_w_k,
            stride_out_n, stride_out_s, stride_out_h,
            BLOCK_H=BLOCK_H,
            num_warps=4,
        )

        # Split outputs into encoder and hidden streams
        processed_encoder = out[:, :T, :]
        processed_hidden = out[:, T:, :]

        return processed_encoder, processed_hidden