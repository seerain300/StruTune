import torch
import triton
import triton.language as tl


@triton.jit
def concat_kernel(
    encoder_ptr,      # *f32, encoder_hidden_states [B, Stext, H]
    hidden_ptr,       # *f32, hidden_states [B, Simg, H]
    out_ptr,          # *f32, concatenated [B, T, H], T = Stext + Simg
    B, Stext, Simg, H,
    stride_e_b, stride_e_t, stride_e_h,
    stride_h_b, stride_h_s, stride_h_h,  # stride_h_s is along sequence s = t - Stext
    stride_o_b, stride_o_t, stride_o_h,
    BLOCK_H: tl.constexpr,
):
    # 2D grid over (B, T)
    b = tl.program_id(0)
    t = tl.program_id(1)
    if (b >= B) or (t >= T):
        return

    # Vector of offsets across hidden dimension
    h_offsets = tl.arange(0, BLOCK_H)
    mask = h_offsets < H

    # Determine source: if t < Stext, take from encoder; else take from hidden at s = t - Stext
    if t < Stext:
        src = encoder_ptr + b * stride_e_b + t * stride_e_t + h_offsets * stride_e_h
    else:
        s = t - Stext
        src = hidden_ptr + b * stride_h_b + s * stride_h_s + h_offsets * stride_h_h

    dst = out_ptr + b * stride_o_b + t * stride_o_t + h_offsets * stride_o_h

    vals = tl.load(src, mask=mask, other=0.0)
    tl.store(dst, vals)


@triton.jit
def matvec_kernel(
    in_ptr,           # *f32, concatenated [B, T, H], T = Stext + Simg
    weightT_ptr,      # *f32, process_weight.T [H, H]
    out_ptr,          # *f32, processed [B, T, H]
    B, T, H,
    stride_i_b, stride_i_t, stride_i_h,
    stride_w_k, stride_w_n,
    stride_o_b, stride_o_t, stride_o_h,
    BLOCK_K: tl.constexpr,
):
    # 2D grid over (B, T): each program computes one output vector for (b, t)
    b = tl.program_id(0)
    t = tl.program_id(1)
    if (b >= B) or (t >= T):
        return

    # Accumulator for outputs (float32)
    acc = tl.zeros([H], dtype=tl.float32)

    # Loop over input dimension H in chunks of BLOCK_K
    k0 = 0
    while k0 < H:
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < H

        # Load in[b, t, k_offsets] as a vector [BLOCK_K]
        in_ptrs = in_ptr + b * stride_i_b + t * stride_i_t + k_offsets * stride_i_h
        in_chunk = tl.load(in_ptrs, mask=mask_k, other=0.0)  # [BLOCK_K]

        # Accumulate acc += in_chunk * weightT[k_offsets, :]
        # Iterate n from 0 to H-1 and multiply each column
        n = 0
        while n < H:
            w_ptrs = weightT_ptr + k_offsets * stride_w_k + n * stride_w_n  # [BLOCK_K]
            w_chunk = tl.load(w_ptrs, mask=mask_k, other=0.0)               # [BLOCK_K]
            acc[n] += tl.sum(in_chunk * w_chunk, axis=0)                    # reduce over K
            n += 1

        k0 += BLOCK_K

    # Store result acc to out[b, t, :]
    out_ptrs = out_ptr + b * stride_o_b + t * stride_o_t + tl.arange(0, H) * stride_o_h
    tl.store(out_ptrs, acc)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        hidden_states: [batch, img_seq_len, hidden_dim]
        encoder_hidden_states: [batch, text_seq_len, hidden_dim]
        process_weight: [hidden_dim, hidden_dim] (no bias)
        Returns: (processed_encoder_hidden_states, processed_hidden_states)
        """
        assert hidden_states.device.type == 'cuda' and encoder_hidden_states.device.type == 'cuda' and process_weight.device.type == 'cuda', \
            "All inputs must be CUDA tensors for Triton kernels."

        # Ensure contiguity and dtype
        hidden = hidden_states.contiguous().to(torch.float32)
        encoder = encoder_hidden_states.contiguous().to(torch.float32)
        weight = process_weight.contiguous().to(torch.float32)

        B = hidden.shape[0]
        Stext = encoder.shape[1]
        Simg = hidden.shape[1]
        H = hidden.shape[2]
        T = Stext + Simg

        # Output concatenated tensor [B, T, H]
        concatenated = torch.empty((B, T, H), device=hidden.device, dtype=hidden.dtype)

        # Launch Triton concat kernel
        grid_concat = (B, T)
        concat_kernel[grid_concat](
            encoder, hidden, concatenated,
            B, Stext, Simg, H,
            encoder.stride(0), encoder.stride(1), encoder.stride(2),
            hidden.stride(0), hidden.stride(1), hidden.stride(2),
            concatenated.stride(0), concatenated.stride(1), concatenated.stride(2),
            BLOCK_H=H,  # vectorize across full H; Triton will handle masks if needed
            num_warps=4, num_stages=2,
        )

        # Prepare weight.T [H, H]
        weight_T = weight.transpose(0, 1).contiguous()  # [H, H]

        # Output processed tensor [B, T, H]
        processed = torch.empty((B, T, H), device=hidden.device, dtype=hidden.dtype)

        # Launch Triton matvec kernel: grid over (B, T)
        grid_matvec = (B, T)
        matvec_kernel[grid_matvec](
            concatenated, weight_T, processed,
            B, T, H,
            concatenated.stride(0), concatenated.stride(1), concatenated.stride(2),
            weight_T.stride(0), weight_T.stride(1),
            processed.stride(0), processed.stride(1), processed.stride(2),
            BLOCK_K=128,
            num_warps=4, num_stages=2,
        )

        # Split back into separate streams using torch slicing (metadata ops, no compute)
        processed_encoder = processed[:, :Stext, :]
        processed_hidden = processed[:, Stext:, :]

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
