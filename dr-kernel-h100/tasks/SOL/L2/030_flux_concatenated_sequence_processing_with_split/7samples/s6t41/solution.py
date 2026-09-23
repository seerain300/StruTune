import torch
import triton
import triton.language as tl


@triton.jit
def concat_linear_singlerow_kernel(
    encoder_hidden_states, hidden_states, process_weight_T, processed_concat,
    B, T, I, H, total_seq,
    stride_e_n, stride_e_s, stride_e_h,
    stride_h_n, stride_h_s, stride_h_h,
    stride_w_k, stride_w_h,  # process_weight_T is [H, H], original weight was [H, H]
    stride_out_n, stride_out_s, stride_out_h,
    BLOCK_K: tl.constexpr,
):
    # Grid: (B, total_seq, 1). Each program handles one (n, s).
    n = tl.program_id(0)
    s = tl.program_id(1)

    # Determine input source: if s < T, use encoder, else use hidden at index s - T
    use_encoder = s < T
    if use_encoder:
        # Load input row from encoder_hidden_states[n, s, :]
        # Ensure safe loads: hidden_states indices are only used when s >= T.
        e_ptr = encoder_hidden_states
        e_row = e_ptr + n * stride_e_n + s * stride_e_s
    else:
        # s >= T; load from hidden_states[n, s - T, :]
        h_ptr = hidden_states
        hs = s - T
        h_row = h_ptr + n * stride_h_n + hs * stride_h_s

    # Prepare output vector
    out = tl.zeros((H,), dtype=tl.float32)
    h_offsets = tl.arange(0, H)
    mask_h = h_offsets < H  # always true since BLOCK_H=H, but keep for generality

    # Loop over K dimension in chunks of BLOCK_K
    # Note: process_weight_T has shape [H, H], so we iterate K from 0 to H-1.
    for k0 in range(0, H, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < H

        # Load input vector chunk: for encoder we use e_row, for hidden we use h_row
        # Both are [H], we only need the k_offsets slice for multiplication, but here
        # we assume either encoder or hidden path is active. We'll implement by
        # loading from the chosen row pointer, but Triton requires explicit address.
        # Instead, we can load scalar by scalar via tl.load with address arithmetic.
        # For performance and simplicity, we load input scalars with address e_row + k_offsets * stride_e_h
        # when use_encoder is true; otherwise from h_row + k_offsets * stride_h_h.
        # However, Triton prefers vectorized loads; we'll do it via scalar loop to keep correctness.
        # To vectorize, we load a vector and mask. Triton supports vectorized loads: we can do it.
        # Build addresses for vector load: for encoder, address = e_row + k_offsets * stride_e_h
        # for hidden, address = h_row + k_offsets * stride_h_h.

        if use_encoder:
            input_vec = tl.load(e_row + k_offsets * stride_e_h, mask=mask_k, other=0.0)
        else:
            input_vec = tl.load(h_row + k_offsets * stride_h_h, mask=mask_k, other=0.0)

        # Load weight row chunk: process_weight_T[h_offsets, k_offsets]
        # Address: process_weight_T + h_offsets[:, None] * stride_w_h + k_offsets[None, :] * stride_w_k
        weight_mat = tl.load(
            process_weight_T + h_offsets[:, None] * stride_w_h + k_offsets[None, :] * stride_w_k,
            mask=(mask_h[:, None] & mask_k[None, :]),
            other=0.0,
        )
        # Accumulate: out[h] += sum_k weight_mat[h, k] * input_vec[k]
        # Using a small inner loop over BLOCK_K to avoid unsupported broadcasting.
        for kk in range(0, BLOCK_K):
            # Broadcast scalar input_vec[kk] and row slice of weight_mat for kk
            w_row_kk = weight_mat[:, kk]
            out += w_row_kk * input_vec[kk]

    # Store result into processed_concat[n, s, :]
    out_ptr = processed_concat + n * stride_out_n + s * stride_out_s
    tl.store(out_ptr + h_offsets * stride_out_h, out, mask=mask_h)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        Triton implementation of:
            concatenated = torch.cat([encoder_hidden_states, hidden_states], dim=1)  # [B, T+I, H]
            processed = torch.matmul(concatenated, process_weight.t())             # [B, T+I, H]
            processed_encoder = processed[:, :T, :]
            processed_hidden = processed[:, T:, :]
        """
        # Ensure dtype and contiguity
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Tensors must be on CUDA for Triton."
        # Make contiguous and float32 for predictable behavior
        hidden_states = hidden_states.contiguous().to(torch.float32)
        encoder_hidden_states = encoder_hidden_states.contiguous().to(torch.float32)
        # Transpose weight on host (metadata only); Triton will read its transposed layout directly
        process_weight_T = process_weight.t().contiguous().to(torch.float32)

        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        H = hidden_states.shape[2]
        total_seq = T + I

        # Allocate output
        processed_concat = torch.empty((B, total_seq, H), device=hidden_states.device, dtype=torch.float32)

        # Get strides
        stride_e_n, stride_e_s, stride_e_h = encoder_hidden_states.stride(0), encoder_hidden_states.stride(1), encoder_hidden_states.stride(2)
        stride_h_n, stride_h_s, stride_h_h = hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2)
        stride_w_k, stride_w_h = process_weight_T.stride(0), process_weight_T.stride(1)  # [H, H]
        stride_out_n, stride_out_s, stride_out_h = processed_concat.stride(0), processed_concat.stride(1), processed_concat.stride(2)

        # Launch Triton kernel: one program per (n, s), compute full H-length output vector
        grid = (B, total_seq, 1)
        # Choose BLOCK_K; H is typically 128/256 here, so 64 or 128 is fine
        BLOCK_K = 128 if H >= 128 else 64

        concat_linear_singlerow_kernel[grid](
            encoder_hidden_states, hidden_states, process_weight_T, processed_concat,
            B, T, I, H, total_seq,
            stride_e_n, stride_e_s, stride_e_h,
            stride_h_n, stride_h_s, stride_h_h,
            stride_w_k, stride_w_h,
            stride_out_n, stride_out_s, stride_out_h,
            BLOCK_K=BLOCK_K,
            num_warps=4,
            num_stages=2,
        )

        # Split outputs
        processed_encoder = processed_concat[:, :T, :]
        processed_hidden = processed_concat[:, T:, :]
        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
