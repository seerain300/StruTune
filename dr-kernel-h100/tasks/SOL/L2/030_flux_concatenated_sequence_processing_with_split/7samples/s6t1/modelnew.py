import torch
import triton
import triton.language as tl

# Triton kernel: for each (batch n, seq pos s), compute out[n, s, :] = input_vec @ weight.T
# We do not build the concatenated tensor; we pick the input vector based on s < T or s >= T.
@triton.jit
def vector_matmul_weightT(
    input_ptr,         # *ptr to input vectors [B, T+I, H] (but we index by (n, s))
    weight_ptr,        # *ptr to process_weight [H, H]
    out_ptr,           # *ptr to output [B, T+I, H]
    B: tl.constexpr,   # batch size
    T: tl.constexpr,   # text_seq_len
    I: tl.constexpr,   # img_seq_len
    H: tl.constexpr,   # hidden_dim
    stride_input_n,    # stride for batch in input
    stride_input_s,    # stride for seq in input
    stride_input_h,    # stride for hidden in input (normally 1)
    stride_weight_h,   # stride for row in weight (normally 1)
    stride_weight_k,   # stride for col in weight (normally H)
    stride_out_n,      # stride for batch in output
    stride_out_s,      # stride for seq in output
    stride_out_h,      # stride for hidden in output
    BLOCK_H: tl.constexpr,
):
    # Each program handles one (n, s)
    n = tl.program_id(0)
    s = tl.program_id(1)

    # Determine which input vector to use: s < T uses encoder; s >= T uses hidden
    use_encoder = s < T
    input_offset = n * stride_input_n + s * stride_input_s
    out_offset = n * stride_out_n + s * stride_out_s

    # Tile across output H dimension
    for h0 in range(0, H, BLOCK_H):
        h_idx = h0 + tl.arange(0, BLOCK_H)
        mask_h = h_idx < H

        # Accumulator for this tile
        acc = tl.zeros([BLOCK_H], dtype=tl.float32)

        # Dot product over K=H: acc += input_vec[k] * weight[k, h_idx]
        # Note: we do not have loop unrolling in Triton like Python; we use a runtime loop.
        # Triton will handle element-wise multiply and reduction via vectorized ops.
        for k in range(0, H):
            # Load input scalar
            input_val = tl.load(input_ptr + input_offset + k * stride_input_h)
            # Load weight column slice (vector of length BLOCK_H)
            w_col = tl.load(
                weight_ptr + k * stride_weight_h + h_idx * stride_weight_k,
                mask=mask_h,
                other=0.0,
            )
            # Accumulate
            acc += input_val * w_col

        # Store result to output
        tl.store(out_ptr + out_offset + h_idx * stride_out_h, acc, mask=mask_h)


class ModelNew(torch.nn.Module):
    def forward(self, encoder_hidden_states: torch.Tensor, hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        Triton-optimized version of the original run function.
        Returns: (processed_encoder_hidden_states, processed_hidden_states)
        Shapes:
          - encoder_hidden_states: [B, T, H]
          - hidden_states: [B, I, H]
          - process_weight: [H, H]
        """
        # Ensure CUDA and contiguity
        assert encoder_hidden_states.is_cuda and hidden_states.is_cuda and process_weight.is_cuda, "Tensors must be on CUDA for Triton kernel."
        B = encoder_hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        H = encoder_hidden_states.shape[2]
        assert hidden_states.shape == (B, I, H), "hidden_states must have shape [batch, img_seq_len, hidden_dim]"
        assert process_weight.shape == (H, H), "process_weight must have shape [hidden_dim, hidden_dim]"

        # Make inputs contiguous for predictable strides
        e = encoder_hidden_states.contiguous()
        h = hidden_states.contiguous()
        w = process_weight.contiguous()

        # Prepare output tensor: [B, T+I, H]
        out = torch.empty((B, T + I, H), device=e.device, dtype=e.dtype)

        # Strides
        stride_input_n, stride_input_s, stride_input_h = e.stride()
        stride_weight_h, stride_weight_k = w.stride()  # weight is [H, H], row stride, col stride
        stride_out_n, stride_out_s, stride_out_h = out.stride()

        # Launch Triton kernel: grid over (batch, sequence positions)
        grid = (B, T + I)

        # We pick a reasonable tile size for H. Use 128 or 256 depending on H; 128 is fine for typical dims.
        BLOCK_H = 128
        # num_warps can be tuned; 4 or 8 is common. For simple vector matmul, 4 is fine.
        vector_matmul_weightT[grid](
            e, w, out,
            B, T, I, H,
            stride_input_n, stride_input_s, stride_input_h,
            w.stride(0), w.stride(1),  # stride_weight_h, stride_weight_k
            stride_out_n, stride_out_s, stride_out_h,
            BLOCK_H=BLOCK_H,
            num_warps=4,
        )

        # Split outputs: first T rows come from encoder, remaining I rows from hidden
        processed_encoder = out[:, :T, :]
        processed_hidden = out[:, T:, :]

        return processed_encoder, processed_hidden