import torch
import triton
import triton.language as tl


@triton.jit
def concat_by_rows(e_ptr, h_ptr, out_ptr,
                    B: tl.constexpr, Stext: tl.constexpr, Simg: tl.constexpr, H: tl.constexpr,
                    stride_e_b: tl.constexpr, stride_e_t: tl.constexpr, stride_e_h: tl.constexpr,
                    stride_h_b: tl.constexpr, stride_h_t: tl.constexpr, stride_h_h: tl.constexpr,
                    stride_o_b: tl.constexpr, stride_o_t: tl.constexpr, stride_o_h: tl.constexpr,
                    BLOCK_H: tl.constexpr):
    # Grid: (B, T)
    b = tl.program_id(0)
    t = tl.program_id(1)

    # Compute source pointers based on t < Stext
    src_e = e_ptr + b * stride_e_b + t * stride_e_t
    src_h = h_ptr + b * stride_h_b + (t - Stext) * stride_h_t

    # Destination pointer for concatenated row
    dest = out_ptr + b * stride_o_b + t * stride_o_t

    # Vector of hidden indices
    offs_h = tl.arange(0, BLOCK_H)
    mask = offs_h < H

    # Load source row into registers
    vals = tl.load(src_e + offs_h * stride_e_h, mask=mask, other=0.0)
    # If t >= Stext, override with hidden row
    use_hidden = t >= Stext
    # Broadcast use_hidden to vector
    vals = tl.where(use_hidden, tl.load(src_h + offs_h * stride_h_h, mask=mask, other=0.0), vals)

    # Store concatenated row
    tl.store(dest + offs_h * stride_o_h, vals, mask=mask)


@triton.jit
def compute_elem_linear(in_ptr, wT_ptr, out_ptr,
                         B: tl.constexpr, T: tl.constexpr, H: tl.constexpr,
                         stride_i_b: tl.constexpr, stride_i_t: tl.constexpr, stride_i_h: tl.constexpr,
                         stride_w_b: tl.constexpr, stride_w_t: tl.constexpr, stride_w_h: tl.constexpr,
                         stride_o_b: tl.constexpr, stride_o_t: tl.constexpr, stride_o_h: tl.constexpr,
                         BLOCK_K: tl.constexpr):
    # Grid: (B, T, H)
    b = tl.program_id(0)
    t = tl.program_id(1)
    h = tl.program_id(2)

    # Accumulator for this (b, t, h)
    acc = 0.0

    # Loop over hidden dimension K in chunks
    k = 0
    while k < H:
        offs_k = k + tl.arange(0, BLOCK_K)
        mask_k = offs_k < H

        # Load in[b, t, offs_k]
        in_row_ptrs = in_ptr + b * stride_i_b + t * stride_i_t + offs_k * stride_i_h
        in_vals = tl.load(in_row_ptrs, mask=mask_k, other=0.0)  # shape [BLOCK_K]

        # Load wT[offs_k, h] (vector of BLOCK_K elements)
        w_ptrs = wT_ptr + offs_k * stride_w_b + h * stride_w_h
        w_vals = tl.load(w_ptrs, mask=mask_k, other=0.0)  # shape [BLOCK_K]

        # Accumulate dot product for this chunk
        acc += tl.sum(in_vals * w_vals, axis=0)

        k += BLOCK_K

    # Store result to out[b, t, h]
    out_ptr_elt = out_ptr + b * stride_o_b + t * stride_o_t + h * stride_o_h
    tl.store(out_ptr_elt, acc)


class ModelNew(torch.nn.Module):
    def forward(self,
                hidden_states: torch.Tensor,
                encoder_hidden_states: torch.Tensor,
                process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version of the given run function:
        1) Concatenate encoder_hidden_states and hidden_states along sequence dim
        2) Apply linear projection with process_weight.T
        3) Split back into separate streams

        All computation is performed by Triton kernels; no torch ops on data.
        """
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Tensors must be CUDA for Triton kernels."
        assert hidden_states.dtype == torch.float32 and encoder_hidden_states.dtype == torch.float32 and process_weight.dtype == torch.float32, "Use float32 tensors."

        # Ensure contiguous memory
        hidden_states = hidden_states.contiguous()
        encoder_hidden_states = encoder_hidden_states.contiguous()
        process_weight = process_weight.contiguous()

        B = hidden_states.shape[0]
        H = hidden_states.shape[2]  # hidden dim
        Stext = encoder_hidden_states.shape[1]
        Simg = hidden_states.shape[1]
        T = Stext + Simg

        # 1) Concatenate [B, T, H]
        out_cat = torch.empty((B, T, H), dtype=torch.float32, device=hidden_states.device)

        # Launch concat kernel: grid over (B, T)
        grid_concat = (B, T)
        BLOCK_H = 128  # vectorized load/store along H
        concat_by_rows[grid_concat](
            encoder_hidden_states, hidden_states, out_cat,
            B=B, Stext=Stext, Simg=Simg, H=H,
            stride_e_b=encoder_hidden_states.stride(0), stride_e_t=encoder_hidden_states.stride(1), stride_e_h=encoder_hidden_states.stride(2),
            stride_h_b=hidden_states.stride(0), stride_h_t=hidden_states.stride(1), stride_h_h=hidden_states.stride(2),
            stride_o_b=out_cat.stride(0), stride_o_t=out_cat.stride(1), stride_o_h=out_cat.stride(2),
            BLOCK_H=BLOCK_H,
            num_warps=4, num_stages=2
        )

        # 2) Compute linear projection: out[b, t, h] = sum_k out_cat[b, t, k] * process_weight.T[k, h]
        # Prepare weights^T as [H, H] contiguous
        W_T = process_weight.transpose(0, 1).contiguous()  # [H, H]

        out = torch.empty((B, T, H), dtype=torch.float32, device=hidden_states.device)

        grid_gemm = (B, T, H)  # one program per (b, t, h)
        BLOCK_K = 128  # loop over hidden dim in chunks
        compute_elem_linear[grid_gemm](
            out_cat, W_T, out,
            B=B, T=T, H=H,
            stride_i_b=out_cat.stride(0), stride_i_t=out_cat.stride(1), stride_i_h=out_cat.stride(2),
            stride_w_b=W_T.stride(0), stride_w_t=W_T.stride(1), stride_w_h=W_T.stride(2),
            stride_o_b=out.stride(0), stride_o_t=out.stride(1), stride_o_h=out.stride(2),
            BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2
        )

        # 3) Split along sequence dimension
        processed_encoder = out[:, :Stext, :]
        processed_hidden = out[:, Stext:, :]

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
