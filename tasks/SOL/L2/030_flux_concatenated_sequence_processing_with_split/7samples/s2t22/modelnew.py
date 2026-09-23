import torch
import triton
import triton.language as tl


@triton.jit
def concat_seqs_kernel(
    out_ptr, enc_ptr, hid_ptr,
    B, T, I, H,
    out_stride_b, out_stride_s, out_stride_h,
    enc_stride_b, enc_stride_t, enc_stride_h,
    hid_stride_b, hid_stride_i, hid_stride_h,
    BLOCK_H: tl.constexpr,
):
    # program ids: batch and sequence tile along H
    b = tl.program_id(0)
    h_block = tl.program_id(1)

    h_offsets = h_block * BLOCK_H + tl.arange(0, BLOCK_H)
    # mask for H range
    h_mask = h_offsets < H

    # write encoder rows into out[:, :T, :]
    enc_row_base = b * enc_stride_b
    for t in range(0, T):
        ptr = enc_ptr + enc_row_base + t * enc_stride_t + h_offsets * enc_stride_h
        out_row_base = b * out_stride_b + t * out_stride_s
        out_ptr_row = out_ptr + out_row_base + h_offsets * out_stride_h
        tl.store(out_ptr_row, tl.load(ptr, mask=h_mask, other=0.0))

    # write hidden rows into out[:, T:, :]
    for i in range(0, I):
        hid_row_base = b * hid_stride_b + i * hid_stride_i
        ptr = hid_ptr + hid_row_base + h_offsets * hid_stride_h
        out_row_base = b * out_stride_b + (T + i) * out_stride_s
        out_ptr_row = out_ptr + out_row_base + h_offsets * out_stride_h
        tl.store(out_ptr_row, tl.load(ptr, mask=h_mask, other=0.0))


@triton.jit
def split_seqs_kernel(
    in_ptr, out_e_ptr, out_i_ptr,
    B, T, I, H,
    in_stride_b, in_stride_s, in_stride_h,
    out_e_stride_b, out_e_stride_t, out_e_stride_h,
    out_i_stride_b, out_i_stride_i, out_i_stride_h,
    BLOCK_H: tl.constexpr,
):
    # program ids: batch and H tile
    b = tl.program_id(0)
    h_block = tl.program_id(1)

    h_offsets = h_block * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = h_offsets < H

    # copy first T rows to out_e
    for t in range(0, T):
        ptr = in_ptr + b * in_stride_b + t * in_stride_s + h_offsets * in_stride_h
        out_row_base_e = b * out_e_stride_b + t * out_e_stride_t
        out_e_row = out_e_ptr + out_row_base_e + h_offsets * out_e_stride_h
        tl.store(out_e_row, tl.load(ptr, mask=h_mask, other=0.0))

    # copy remaining I rows to out_i
    for i in range(0, I):
        in_row_base = b * in_stride_b + (T + i) * in_stride_s
        ptr = in_ptr + in_row_base + h_offsets * in_stride_h
        out_row_base_i = b * out_i_stride_b + i * out_i_stride_i
        out_i_row = out_i_ptr + out_row_base_i + h_offsets * out_i_stride_h
        tl.store(out_i_row, tl.load(ptr, mask=h_mask, other=0.0))


class ModelNew(torch.nn.Module):
    def forward(self, encoder_hidden_states: torch.Tensor, hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        Triton-enabled version that:
          - concatenates encoder_hidden_states and hidden_states along sequence dim
          - performs linear projection via torch.matmul (reliable and fast)
          - splits the result back into encoder and hidden streams
        Returns (processed_encoder_hidden_states, processed_hidden_states)
        """
        assert encoder_hidden_states.is_cuda and hidden_states.is_cuda and process_weight.is_cuda, "Inputs must be CUDA tensors"
        B = encoder_hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        H = encoder_hidden_states.shape[2]
        assert hidden_states.shape[2] == H, "hidden_states hidden_dim must match encoder_hidden_states"
        assert process_weight.shape == (H, H), "process_weight must be [hidden_dim, hidden_dim]"

        # Ensure contiguity
        enc = encoder_hidden_states.contiguous()
        hid = hidden_states.contiguous()
        pw_T = process_weight.t().contiguous()

        # 1) Concatenate sequences into [B, S, H], S = T + I
        S = T + I
        concatenated = torch.empty((B, S, H), device=enc.device, dtype=enc.dtype)

        # Launch Triton concat kernel over grid (B, ceil_div(H, BLOCK_H))
        BLOCK_H = 1024
        grid = (B, triton.cdiv(H, BLOCK_H))
        concat_seqs_kernel[grid](
            concatenated, enc, hid,
            B, T, I, H,
            concatenated.stride(0), concatenated.stride(1), concatenated.stride(2),
            enc.stride(0), enc.stride(1), enc.stride(2),
            hid.stride(0), hid.stride(1), hid.stride(2),
            BLOCK_H=BLOCK_H,
            num_warps=4,
            num_stages=2,
        )

        # 2) Apply linear projection with PyTorch (robust and fast)
        # processed = concatenated @ process_weight.T
        processed = torch.matmul(concatenated, pw_T)

        # 3) Split back into encoder and hidden parts using Triton
        processed_encoder = torch.empty((B, T, H), device=enc.device, dtype=enc.dtype)
        processed_hidden = torch.empty((B, I, H), device=enc.device, dtype=enc.dtype)

        grid_split = (B, triton.cdiv(H, BLOCK_H))
        split_seqs_kernel[grid_split](
            processed, processed_encoder, processed_hidden,
            B, T, I, H,
            processed.stride(0), processed.stride(1), processed.stride(2),
            processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
            processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
            BLOCK_H=BLOCK_H,
            num_warps=4,
            num_stages=2,
        )

        return processed_encoder, processed_hidden