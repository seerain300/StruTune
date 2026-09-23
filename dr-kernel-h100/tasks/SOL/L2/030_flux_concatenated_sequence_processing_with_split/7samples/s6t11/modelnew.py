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
):
    # Grid: (B, T+I). Each program computes one output row (n, s).
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)
    total_seq = T + I

    # Determine whether this sequence position belongs to encoder or hidden stream
    is_encoder = pid_s < T

    # Output row pointer for (n, s)
    out_row_ptr = out_ptr + pid_n * stride_out_n + pid_s * stride_out_s

    # Accumulator for the output vector of length H
    acc = tl.zeros((H,), dtype=tl.float32)

    # Iterate over K dimension (input features = H)
    # Load input vector element by element and accumulate acc += input[k] * weight[k, h]
    for k in range(0, H):
        # Compute input value for this k: from encoder if is_encoder else hidden at s - T
        if is_encoder:
            # encoder_hidden_states[n, s, k]
            input_val = tl.load(encoder_ptr + pid_n * stride_e_n + pid_s * stride_e_s + k * stride_e_h)
        else:
            # hidden_states[n, s - T, k]
            k_idx = s_pos - T
            input_val = tl.load(hidden_ptr + pid_n * stride_h_n + k_idx * stride_h_s + k * stride_h_h)

        # Load corresponding weight row slice for output dim k
        # weight[k, h] for all h in 0..H-1
        # We'll accumulate over h in chunks for performance, but here we use direct element-wise accumulation.
        # Note: weight_ptr[k * stride_w_h + h * stride_w_k]
        # Since we iterate h in outer loop below, we can compute per-h multiplication here.
        pass  # placeholder to satisfy Triton JIT; actual accumulation happens in the next loop

    # Now accumulate over H using the weight matrix
    # We need to compute out[n, s, h] = sum_{k=0}^{H-1} input_vec[k] * weight[k, h]
    # Since we don't have input_vec anymore (we used scalar loads), we recompute per-h by using
    # the fact that input_val is scalar per k iteration above. Instead, we compute input_vec
    # by recomputing from the source tensor per h in the next loops. To avoid recomputation,
    # we can store input vector as a list, but Triton does not support dynamic Python lists.
    # Therefore, we directly compute per-h below:
    for h in range(0, H):
        # Re-load input value for each h (inefficient, but robust). A better approach would be
        # to vectorize across h and k, but to keep correctness and simplicity, we do element-wise.
        # For each h, compute input_vec[h] by checking is_encoder; however, input_vec is not stored.
        # Given the previous approach, we instead compute input_vec on the fly for each h by loading
        # from encoder or hidden based on s. This is correct but slow; for performance, we move to
        # vectorized loads in the next version.
        if is_encoder:
            input_val_h = tl.load(encoder_ptr + pid_n * stride_e_n + pid_s * stride_e_s + h * stride_e_h)
        else:
            k_idx = pid_s - T
            input_val_h = tl.load(hidden_ptr + pid_n * stride_h_n + k_idx * stride_h_s + h * stride_h_h)
        # Load weight row for output dim h
        # We need weight[:, h], i.e., all k entries for this h
        # Load as a vector over k tile; but since we don't have k tile here, we just load per k in a loop.
        # To keep code correct and simple, we emulate the accumulation using scalar loop over k:
        # However, to avoid double work, we instead load weight slice per h and use a small k-tile loop.
        # Triton supports loops with runtime bounds; we use a small tile to cover H.
        # Note: H may be large; to keep code simple, we load per-k and accumulate. This is correct.
        acc_h = 0.0
        for kk in range(0, H):
            if is_encoder:
                input_k = tl.load(encoder_ptr + pid_n * stride_e_n + pid_s * stride_e_s + kk * stride_e_h)
            else:
                k_idx = pid_s - T
                input_k = tl.load(hidden_ptr + pid_n * stride_h_n + k_idx * stride_h_s + kk * stride_h_h)
            # weight[kk, h] vector load
            # Using pointer arithmetic: weight_ptr + kk * stride_w_h + h * stride_w_k
            # Triton allows scalar loads; we'll load as scalar
            weight_val = tl.load(weight_ptr + kk * stride_w_h + h * stride_w_k)
            acc_h += input_k * weight_val
        acc[h] = acc_h

    # Store accumulated output vector
    # out_ptr + n*stride_out_n + s*stride_out_s + h*stride_out_h
    for h in range(0, H):
        tl.store(out_row_ptr + h * stride_out_h, acc[h])


class ModelNew(torch.nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        process_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version of:
        concatenated = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        processed = concatenated @ process_weight.t()
        processed_encoder = processed[:, :text_seq_len, :]
        processed_hidden = processed[:, text_seq_len:, :]
        Returns (processed_encoder, processed_hidden).
        """
        # Enforce device and dtype; use float32 for stability
        device = hidden_states.device
        dtype = torch.float32

        # Ensure inputs are contiguous and on the same device/dtype
        B, T, H_e = encoder_hidden_states.shape
        _, I, H_h = hidden_states.shape
        assert H_e == H_h, "hidden_dim must match between encoder_hidden_states and hidden_states"
        assert process_weight.shape[0] == H_e and process_weight.shape[1] == H_e, "process_weight must be [H, H]"

        encoder = encoder_hidden_states.contiguous().to(dtype)
        hidden = hidden_states.contiguous().to(dtype)
        weight = process_weight.contiguous().to(dtype)

        total_seq = T + I

        # Allocate output [B, T+I, H]
        out = torch.empty((B, total_seq, H_e), device=device, dtype=dtype)

        # Compute strides (in elements) for contiguous tensors
        stride_e_n, stride_e_s, stride_e_h = encoder.stride(0), encoder.stride(1), encoder.stride(2)
        stride_h_n, stride_h_s, stride_h_h = hidden.stride(0), hidden.stride(1), hidden.stride(2)
        # weight is [H, H]
        stride_w_h, stride_w_k = weight.stride(0), weight.stride(1)
        stride_out_n, stride_out_s, stride_out_h = out.stride(0), out.stride(1), out.stride(2)

        # Launch Triton kernel: grid over (batch, sequence positions)
        grid = (B, total_seq)
        # Use a modest number of warps; the kernel is simple but iterates over H
        batched_matmul_cat_with_weightT_kernel[grid](
            encoder, hidden, weight, out,
            B, T, I, H_e,
            stride_e_n, stride_e_s, stride_e_h,
            stride_h_n, stride_h_s, stride_h_h,
            stride_w_h, stride_w_k,
            stride_out_n, stride_out_s, stride_out_h,
            num_warps=4,
        )

        # Split outputs into encoder and hidden streams
        processed_encoder = out[:, :T, :]
        processed_hidden = out[:, T:, :]

        return processed_encoder, processed_hidden