import torch
import triton
import triton.language as tl


@triton.jit
def concat_linear_split_kernel(
    encoder_hidden_states_ptr,  # [B, T, H]
    hidden_states_ptr,          # [B, I, H]
    process_weight_ptr,         # [H, H]
    processed_concat_ptr,       # [B, T+I, H]
    B: tl.constexpr,            # batch size
    T: tl.constexpr,            # text_seq_len
    I: tl.constexpr,            # img_seq_len
    H: tl.constexpr,            # hidden_dim
    stride_e_n, stride_e_s, stride_e_h,  # encoder strides
    stride_h_n, stride_h_s, stride_h_h,  # hidden strides
    stride_w_h, stride_w_k,               # process_weight strides (shape [H, H])
    stride_out_n, stride_out_s, stride_out_h,  # output strides
    total_seq: tl.constexpr,  # T + I
    tiles_h: tl.constexpr,    # number of H tiles
    BLOCK_K: tl.constexpr,    # tile over input features (K = H)
    BLOCK_H: tl.constexpr,    # tile over output features (H)
):
    # program ids: (n over batch, s over total sequence, h_tile over H tiles)
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_ht = tl.program_id(2)

    # Compute the offsets for H-tiles
    h_offsets = pid_ht * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = h_offsets < H

    # Select the input row based on position s
    # s < T uses encoder_hidden_states, else uses hidden_states at index (s - T)
    # Ensure s is within [0, total_seq)
    if pid_s < total_seq:
        if pid_s < T:
            # input from encoder: [n, s, :]
            e_ptr = encoder_hidden_states_ptr + pid_n * stride_e_n + pid_s * stride_e_s
        else:
            # input from hidden: [n, s - T, :]
            h_ptr = hidden_states_ptr + pid_n * stride_h_n + (pid_s - T) * stride_h_s

        # Accumulator for output vector
        out_vec = tl.zeros([BLOCK_H], dtype=tl.float32)

        # Loop over K (input features) in tiles
        for k0 in range(0, H, BLOCK_K):
            k_offsets = k0 + tl.arange(0, BLOCK_K)
            k_mask = k_offsets < H

            # Load input vector slice of length BLOCK_K
            if pid_s < T:
                e_ptrs = e_ptr + k_offsets * stride_e_h
                x = tl.load(e_ptrs, mask=k_mask, other=0.0)
            else:
                h_ptrs = h_ptr + k_offsets * stride_h_h
                x = tl.load(h_ptrs, mask=k_mask, other=0.0)

            # Load process_weight.T slice: [BLOCK_K, BLOCK_H]
            # process_weight is [H, H] with strides (stride_w_h, stride_w_k)
            w_ptrs = process_weight_ptr + k_offsets[:, None] * stride_w_k + h_offsets[None, :] * stride_w_h
            w_mask = (k_offsets[:, None] < H) & (h_offsets[None, :] < H)
            w_block = tl.load(w_ptrs, mask=w_mask, other=0.0)

            # Accumulate: out_vec += x @ w_block
            # x: [BLOCK_K], w_block: [BLOCK_K, BLOCK_H] -> result: [BLOCK_H]
            # Multiply each element of x with corresponding row of w_block and reduce
            # out_vec += sum_k x[k] * w_block[k, :]
            # Implemented as a loop over k for simplicity and correctness
            for kk in range(0, BLOCK_K):
                # valid_k = k0 + kk < H
                valid_k = k0 + kk < H
                row = w_block[kk, :]
                out_vec += x[kk] * row * valid_k

        # Store the accumulated output vector into processed_concat[n, s, :]
        out_ptrs = processed_concat_ptr + pid_n * stride_out_n + pid_s * stride_out_s + h_offsets * stride_out_h
        tl.store(out_ptrs, out_vec, mask=h_mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version of the original run function:
        - Concatenates encoder_hidden_states and hidden_states along the sequence dimension.
        - Applies linear projection with process_weight.T.
        - Splits back into encoder and image outputs.
        """
        # Shapes
        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        H = hidden_states.shape[2]
        assert encoder_hidden_states.shape[2] == H, "encoder_hidden_states hidden_dim must match hidden_states"
        assert process_weight.shape[0] == H and process_weight.shape[1] == H, "process_weight must be [H, H]"

        # Ensure contiguous tensors and float32 for predictable numerics
        e = encoder_hidden_states.contiguous().to(torch.float32)
        h = hidden_states.contiguous().to(torch.float32)
        w = process_weight.contiguous().to(torch.float32)

        total_seq = T + I
        # Allocate output buffer [B, total_seq, H]
        processed_concat = torch.empty((B, total_seq, H), dtype=torch.float32, device=e.device)

        # Compute tile sizes (use 128 for robustness across common H sizes)
        BLOCK_H = 128
        BLOCK_K = 128
        tiles_h = (H + BLOCK_H - 1) // BLOCK_H

        # Launch Triton kernel: grid over (batch, total_seq, H-tiles)
        grid = (B, total_seq, tiles_h)
        concat_linear_split_kernel[grid](
            e, h, w, processed_concat,
            B, T, I, H,
            e.stride(0), e.stride(1), e.stride(2),
            h.stride(0), h.stride(1), h.stride(2),
            w.stride(0), w.stride(1),
            processed_concat.stride(0), processed_concat.stride(1), processed_concat.stride(2),
            total_seq,
            tiles_h,
            BLOCK_K=BLOCK_K, BLOCK_H=BLOCK_H,
            num_warps=8,
            num_stages=2,
        )

        # Split outputs
        processed_encoder = processed_concat[:, :T, :]
        processed_hidden = processed_concat[:, T:, :]
        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
