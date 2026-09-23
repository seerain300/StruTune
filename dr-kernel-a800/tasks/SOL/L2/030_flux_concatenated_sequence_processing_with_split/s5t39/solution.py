import torch
import triton
import triton.language as tl


@triton.jit
def concat_kernel(
    encoder_ptr,      # *f32, encoder_hidden_states [B, Stext, H]
    hidden_ptr,       # *f32, hidden_states [B, Simg, H]
    out_ptr,          # *f32, concatenated [B, T, H], T = Stext + Simg
    B: tl.constexpr,
    Stext: tl.constexpr,
    Simg: tl.constexpr,
    H: tl.constexpr,
    stride_e_b: tl.constexpr,
    stride_e_t: tl.constexpr,
    stride_e_h: tl.constexpr,
    stride_h_b: tl.constexpr,
    stride_h_s: tl.constexpr,   # s = t - Stext for hidden
    stride_h_h: tl.constexpr,
    stride_o_b: tl.constexpr,
    stride_o_t: tl.constexpr,
    stride_o_h: tl.constexpr,
    BLOCK_COPY: tl.constexpr,   # number of hidden elements copied per iteration
):
    # Grid over (B, T)
    b = tl.program_id(0)
    t = tl.program_id(1)
    if (b >= B) or (t >= T):
        return

    # Determine source tensor based on t
    if t < Stext:
        src = encoder_ptr + b * stride_e_b + t * stride_e_t
    else:
        s = t - Stext
        src = hidden_ptr + b * stride_h_b + s * stride_h_s

    dst = out_ptr + b * stride_o_b + t * stride_o_t

    # Copy H elements from src to dst in chunks
    for off in range(0, H, BLOCK_COPY):
        idx = off + tl.arange(0, BLOCK_COPY)
        mask = idx < H
        vals = tl.load(src + idx * stride_e_h, mask=mask, other=0.0)
        tl.store(dst + idx * stride_o_h, vals, mask=mask)


@triton.jit
def matvec_kernel(
    in_ptr,           # *f32, concatenated [B, T, H]
    weightT_ptr,      # *f32, process_weight.T [H, H] (no bias)
    out_ptr,          # *f32, processed [B, T, H]
    B: tl.constexpr,
    T: tl.constexpr,
    H: tl.constexpr,
    stride_i_b: tl.constexpr,
    stride_i_t: tl.constexpr,
    stride_i_h: tl.constexpr,
    stride_w_k: tl.constexpr,  # weight_T dim-0 (k)
    stride_w_n: tl.constexpr,  # weight_T dim-1 (n)
    stride_o_b: tl.constexpr,
    stride_o_t: tl.constexpr,
    stride_o_h: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Grid over (B, T). Each program computes out[b, t, :]
    b = tl.program_id(0)
    t = tl.program_id(1)
    if (b >= B) or (t >= T):
        return

    # Base pointers for this (b, t)
    in_row = in_ptr + b * stride_i_b + t * stride_i_t
    out_vec = out_ptr + b * stride_o_b + t * stride_o_t

    # Accumulator
    acc = tl.zeros([H], dtype=tl.float32)

    # Loop over input dimension H in chunks
    for k0 in range(0, H, BLOCK_K):
        k_idx = k0 + tl.arange(0, BLOCK_K)
        mask_k = k_idx < H

        # Load input vector chunk: in[b, t, k_idx]
        in_chunk = tl.load(in_row + k_idx * stride_i_h, mask=mask_k, other=0.0)  # shape [BLOCK_K]

        # Load weight tile: weightT[k_idx, :] -> shape [BLOCK_K, H]
        n_idx = tl.arange(0, H)
        w_ptrs = weightT_ptr + k_idx[:, None] * stride_w_k + n_idx[None, :] * stride_w_n
        mask_w = mask_k[:, None]
        w_tile = tl.load(w_ptrs, mask=mask_w, other=0.0)  # shape [BLOCK_K, H]

        # Accumulate: sum over k of in_chunk[k] * w_tile[k, :]
        acc += tl.sum(w_tile * in_chunk[:, None], axis=0)

    # Store accumulated result
    tl.store(out_vec + tl.arange(0, H) * stride_o_h, acc)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                encoder_hidden_states: torch.Tensor,
                process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version of the original run:
        1) Concatenate encoder_hidden_states and hidden_states along sequence dim in Triton.
        2) Apply linear projection (no bias) via Triton matvec kernel: processed = concatenated @ process_weight.T
        3) Split back into separate encoder and image streams.
        """
        # Ensure inputs are on CUDA and float32
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, \
            "All inputs must be on CUDA device for Triton kernels."
        assert hidden_states.dtype == torch.float32 and encoder_hidden_states.dtype == torch.float32 and process_weight.dtype == torch.float32, \
            "This Triton implementation currently supports float32 tensors only."

        # Ensure contiguous
        encoder = encoder_hidden_states.contiguous()
        hidden = hidden_states.contiguous()
        weight_T = process_weight.t().contiguous()  # [H, H]

        B = encoder.shape[0]
        Stext = encoder.shape[1]
        Simg = hidden.shape[1]
        H = encoder.shape[2]
        T = Stext + Simg

        # Allocate concatenated tensor
        concatenated = torch.empty((B, T, H), device=encoder.device, dtype=encoder.dtype)

        # Launch Triton concat kernel: grid over (B, T)
        grid_concat = (B, T)
        concat_kernel[grid_concat](
            encoder, hidden, concatenated,
            B, Stext, Simg, H,
            encoder.stride(0), encoder.stride(1), encoder.stride(2),
            hidden.stride(0), hidden.stride(1), hidden.stride(2),
            concatenated.stride(0), concatenated.stride(1), concatenated.stride(2),
            BLOCK_COPY=128,
            num_warps=4,
        )

        # Allocate output processed tensor
        processed = torch.empty((B, T, H), device=encoder.device, dtype=encoder.dtype)

        # Launch Triton matvec kernel: grid over (B, T)
        grid_matvec = (B, T)
        matvec_kernel[grid_matvec](
            concatenated, weight_T, processed,
            B, T, H,
            concatenated.stride(0), concatenated.stride(1), concatenated.stride(2),
            weight_T.stride(0), weight_T.stride(1),
            processed.stride(0), processed.stride(1), processed.stride(2),
            BLOCK_K=128,
            num_warps=4,
        )

        # Split back into encoder and image streams (metadata ops, no compute)
        processed_encoder = processed[:, :Stext, :]
        processed_hidden = processed[:, Stext:, :]

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
