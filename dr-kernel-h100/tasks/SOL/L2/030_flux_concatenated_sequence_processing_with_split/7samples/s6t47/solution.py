import torch
import triton
import triton.language as tl


@triton.jit
def concat_linear_split_blocked_kernel(
    # Inputs
    e_ptr,        # [B, T, H]
    h_ptr,        # [B, I, H]
    w_ptr,        # [H, H] (process_weight)
    out_ptr,      # [B, T+I, H] (processed_concat)
    # Meta-parameters
    B: tl.constexpr, T: tl.constexpr, I: tl.constexpr, H: tl.constexpr,
    BLOCK_S: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_K: tl.constexpr,
    # Strides (in elements)
    stride_e_n, stride_e_s, stride_e_h,
    stride_h_n, stride_h_s, stride_h_h,
    stride_w_h, stride_w_k,
    stride_out_n, stride_out_s, stride_out_h,
):
    # Grid: (B, tiles_seq, tiles_h)
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_h = tl.program_id(2)

    total_seq = T + I

    # Compute tile offsets
    s_offsets = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    h_offsets = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)

    # Masks for valid H and sequence positions
    mask_h = h_offsets < H
    mask_s = s_offsets < total_seq

    # Accumulator: [BLOCK_H, BLOCK_S] in float32
    acc = tl.zeros((BLOCK_H, BLOCK_S), dtype=tl.float32)

    # Iterate over K in tiles
    for k_start in range(0, H, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < H

        # Build input vectors for each s in the tile
        # Determine which source (encoder or hidden) based on s
        # s_offsets are < total_seq, but we need to decide per element:
        # if s < T -> from encoder, else from hidden at (s - T)
        # Create boolean per s
        from_encoder = s_offsets < T

        # For each k in the tile, accumulate contributions
        # We'll compute input chunks: [BLOCK_S] and weight chunks: [BLOCK_H]
        for kk in range(BLOCK_K):
            k_idx = k_start + kk
            valid_k = k_idx < H
            # For each kk, get input_vec for each s and weight_vec for h_offsets
            # Initialize input_vec as zeros, then set according to from_encoder
            # Note: k_idx is scalar; broadcast with masks
            input_vec = tl.zeros((BLOCK_S,), dtype=tl.float32)

            # Load from encoder or hidden depending on from_encoder
            # We build pointers for each s element
            # ptr shape: pointer expression depends on from_encoder[s], which is vector
            # Triton allows vector masks and pointer arithmetic with vector indices
            # First, compute base pointers for each s
            base_e = e_ptr + pid_n * stride_e_n
            base_h = h_ptr + pid_n * stride_h_n

            # For each s in vector, set pointer accordingly
            # ptr_vec is a vector of pointers of length BLOCK_S
            # Triton supports pointer arithmetic with boolean masks
            # We construct ptr_vec with where to choose encoder or hidden per s
            ptr_vec = tl.where(from_encoder, base_e + s_offsets[:, None] * stride_e_s, base_h + (s_offsets[:, None] - T) * stride_h_s)
            # Masked load: valid when both s is valid and from_encoder is True/False as needed
            # Since we built ptr_vec with where, we can load with mask for s valid
            input_vec = tl.load(ptr_vec, mask=mask_s[:, None], other=0.0)

            # Load weight chunk for h_offsets: W[k_idx, h_offsets]
            w_ptrs = w_ptr + k_idx * stride_w_h + h_offsets[None, :] * stride_w_k
            weight_vec = tl.load(w_ptrs, mask=mask_h[None, :], other=0.0)
            # Ensure scalar broadcasting: weight_vec is [BLOCK_H], input_vec is [BLOCK_S]
            # Accumulate: acc[h, s] += input_vec[s] * weight_vec[h]
            acc += input_vec[:, None] * weight_vec[None, :]

    # Store the computed tile into out
    # out[n, s, h] where s in s_offsets, h in h_offsets
    out_ptrs = out_ptr + pid_n * stride_out_n + s_offsets[None, :] * stride_out_s + h_offsets[:, None] * stride_out_h
    # Combine masks for h and s
    store_mask = mask_h[:, None] & mask_s[None, :]
    tl.store(out_ptrs, acc, mask=store_mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version of the original run:
        - Concatenates encoder_hidden_states and hidden_states along sequence dimension.
        - Applies linear projection with process_weight.T.
        - Splits outputs back into encoder and hidden streams.
        """
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "All tensors must be on CUDA for Triton."
        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        H = hidden_states.shape[2]
        assert process_weight.shape[0] == H and process_weight.shape[1] == H, "process_weight must have shape [H, H]."

        # Ensure dtype and contiguity
        # Keep computation in float32 for consistent numerics
        e = encoder_hidden_states.contiguous().to(torch.float32)
        h = hidden_states.contiguous().to(torch.float32)
        w = process_weight.contiguous().to(torch.float32)

        total_seq = T + I
        # Allocate output
        processed_concat = torch.empty((B, total_seq, H), dtype=torch.float32, device=e.device)

        # Strides (elements)
        stride_e_n, stride_e_s, stride_e_h = e.stride(0), e.stride(1), e.stride(2)
        stride_h_n, stride_h_s, stride_h_h = h.stride(0), h.stride(1), h.stride(2)
        stride_w_h, stride_w_k = w.stride(0), w.stride(1)  # w is [H, H]
        stride_out_n, stride_out_s, stride_out_h = processed_concat.stride(0), processed_concat.stride(1), processed_concat.stride(2)

        # Tile sizes: tune for typical H sizes
        # For H around 128/256, BLOCK_H=128 or 256 works well; BLOCK_K=64 balances register usage
        BLOCK_H = 128 if H >= 128 else 64
        BLOCK_S = 64  # tile across sequence dimension
        BLOCK_K = 64

        tiles_s = (total_seq + BLOCK_S - 1) // BLOCK_S
        tiles_h = (H + BLOCK_H - 1) // BLOCK_H

        # Launch Triton kernel
        grid = (B, tiles_s, tiles_h)
        concat_linear_split_blocked_kernel[grid](
            e, h, w, processed_concat,
            B, T, I, H,
            BLOCK_S, BLOCK_H, BLOCK_K,
            stride_e_n, stride_e_s, stride_e_h,
            stride_h_n, stride_h_s, stride_h_h,
            stride_w_h, stride_w_k,
            stride_out_n, stride_out_s, stride_out_h,
            num_warps=4,  # modest warps; 4 or 8 often fine for these tile sizes
            num_stages=2,
        )

        # Split outputs
        processed_encoder = processed_concat[:, :T, :]
        processed_hidden = processed_concat[:, T:, :]
        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
