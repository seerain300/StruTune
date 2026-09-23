import torch
import triton
import triton.language as tl

# Triton kernel: concatenate [B, T, H] and [B, I, H] into [B, T+I, H]
@triton.jit
def _concatenate_sequences_kernel(
    enc_ptr,            # *ptr to encoder_hidden_states [B, T, H]
    hid_ptr,            # *ptr to hidden_states [B, I, H]
    out_ptr,            # *ptr to output [B, T+I, H]
    B, T, I, H,
    enc_stride_b, enc_stride_t, enc_stride_h,
    hid_stride_b, hid_stride_i, hid_stride_h,
    out_stride_b, out_stride_l, out_stride_h,
    BLOCK_L: tl.constexpr,
):
    b = tl.program_id(0)  # one program per batch
    # Loop over the concatenated sequence length
    for l in range(0, T + I, BLOCK_L):
        offs = l + tl.arange(0, BLOCK_L)
        mask = offs < (T + I)

        # Determine source tensor for this index
        src_encoder = offs < T
        # Load from encoder if index < T, else from hidden
        enc_ptrs = enc_ptr + b * enc_stride_b + offs * enc_stride_t
        hid_ptrs = hid_ptr + b * hid_stride_b + (offs - T) * hid_stride_i  # offs - T indexes hidden part
        # Note: offs - T is negative when offs >= T; however, mask ensures we only load valid indices.
        # Use tl.load with mask for safety.
        vals_encoder = tl.load(enc_ptrs, mask=mask & src_encoder, other=0.0)
        vals_hidden = tl.load(hid_ptrs, mask=mask & (~src_encoder), other=0.0)
        # Select based on mask
        vals = tl.where(src_encoder & mask, vals_encoder, vals_hidden)

        # Store to out_cat[b, offs, :]
        out_ptrs = out_ptr + b * out_stride_b + offs * out_stride_l
        tl.store(out_ptrs, vals, mask=mask)


# Triton kernel: split [B, T+I, H] into [B, T, H] and [B, I, H]
# Each program handles one batch and iterates over H
@triton.jit
def _split_streams_kernel(
    in_ptr,            # *ptr to input processed concatenated [B*(T+I), H]
    out_enc_ptr,       # *ptr to output encoder [B, T, H]
    out_hid_ptr,       # *ptr to output hidden [B, I, H]
    B, T, I, H,
    in_stride_m, in_stride_k,
    enc_stride_b, enc_stride_t, enc_stride_h,
    hid_stride_b, hid_stride_i, hid_stride_h,
    BLOCK_H: tl.constexpr,
):
    b = tl.program_id(0)  # one program per batch
    # Iterate over sequence parts
    for l in range(0, T, BLOCK_H):
        offs_t = l + tl.arange(0, BLOCK_H)
        mask_t = offs_t < T
        # Map m index for encoder part: m = b*(T+I) + l
        m_t = b * (T + I) + l
        in_ptrs_t = in_ptr + m_t * in_stride_m + offs_t * in_stride_k
        vals_t = tl.load(in_ptrs_t, mask=mask_t, other=0.0)
        # Store to encoder output: out[b, l, :]
        enc_ptrs_t = out_enc_ptr + b * enc_stride_b + offs_t * enc_stride_t
        tl.store(enc_ptrs_t, vals_t, mask=mask_t)

    for l in range(0, I, BLOCK_H):
        offs_i = l + tl.arange(0, BLOCK_H)
        mask_i = offs_i < I
        # Map m index for hidden part: m = b*(T+I) + (T + l)
        m_i = b * (T + I) + (T + l)
        in_ptrs_i = in_ptr + m_i * in_stride_m + offs_i * in_stride_k
        vals_i = tl.load(in_ptrs_i, mask=mask_i, other=0.0)
        # Store to hidden output: out[b, l, :]
        hid_ptrs_i = out_hid_ptr + b * hid_stride_b + offs_i * hid_stride_i
        tl.store(hid_ptrs_i, vals_i, mask=mask_i)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        Triton-only implementation:
        - Concatenate encoder_hidden_states and hidden_states along sequence dimension using Triton.
        - Perform matrix multiplication using torch.matmul (for robustness).
        - Split result back into two streams using Triton.
        """
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "All tensors must be on CUDA device"
        B, I, H = hidden_states.shape
        T, H2, H3 = encoder_hidden_states.shape
        assert H2 == H, "hidden_states and encoder_hidden_states hidden_dim must match"
        assert H3 == H, "hidden_states and encoder_hidden_states hidden_dim must match"
        assert process_weight.shape[0] == H and process_weight.shape[1] == H, "process_weight must be [hidden_dim, hidden_dim]"

        # 1) Concatenate along sequence dimension
        out_cat = torch.empty((B, T + I, H), dtype=hidden_states.dtype, device=hidden_states.device)
        _concatenate_sequences_kernel[(B,)](
            encoder_hidden_states, hidden_states, out_cat,
            B, T, I, H,
            *encoder_hidden_states.stride(), *hidden_states.stride(),
            *out_cat.stride(),
            BLOCK_L=256,
            num_warps=1,
            num_stages=1,
        )

        # 2) Matrix multiplication: processed = out_cat @ process_weight.T
        # out_cat: [B, T+I, H], process_weight: [H, H] -> right-multiply
        processed = torch.matmul(out_cat, process_weight.t())

        # 3) Split into encoder and hidden streams
        processed_encoder = torch.empty((B, T, H), dtype=processed.dtype, device=processed.device)
        processed_hidden = torch.empty((B, I, H), dtype=processed.dtype, device=processed.device)

        # Flatten processed for input to split kernel: shape [B*(T+I), H]
        C = processed.reshape(B * (T + I), H)
        C = C.contiguous()  # ensure contiguous for Triton loads

        _split_streams_kernel[(B,)](
            C,
            processed_encoder, processed_hidden,
            B, T, I, H,
            *C.stride(),
            *processed_encoder.stride(), *processed_hidden.stride(),
            BLOCK_H=256,
            num_warps=1,
            num_stages=1,
        )

        return processed_encoder, processed_hidden