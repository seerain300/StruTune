import torch
import triton
import triton.language as tl


@triton.jit
def concat_by_row(e_ptr, h_ptr, out_ptr,
                   B: tl.constexpr, Stext: tl.constexpr, Simg: tl.constexpr, H: tl.constexpr,
                   stride_e_b: tl.constexpr, stride_e_t: tl.constexpr, stride_e_h: tl.constexpr,
                   stride_h_b: tl.constexpr, stride_h_t: tl.constexpr, stride_h_h: tl.constexpr,
                   stride_o_b: tl.constexpr, stride_o_t: tl.constexpr, stride_o_h: tl.constexpr):
    """
    Concatenate encoder_hidden_states [B, Stext, H] and hidden_states [B, Simg, H]
    along the sequence dimension to produce out [B, Stext + Simg, H].

    Launch grid: (B, Stext + Simg)
    Each program handles one batch b and one sequence position t and writes the full H vector.
    """
    b = tl.program_id(0)
    t = tl.program_id(1)

    # Determine source tensor and index in source
    is_encoder = t < Stext
    src_t = tl.where(is_encoder, t, t - Stext)

    # Compute base pointers
    e_base = e_ptr + b * stride_e_b + src_t * stride_e_t
    h_base = h_ptr + b * stride_h_b + src_t * stride_h_t
    o_base = out_ptr + b * stride_o_b + t * stride_o_t

    # Vector of hidden dimension offsets
    h_offsets = tl.arange(0, H)
    e_ptrs = e_base + h_offsets * stride_e_h
    h_ptrs = h_base + h_offsets * stride_h_h
    o_ptrs = o_base + h_offsets * stride_o_h

    # Load from appropriate source
    if is_encoder:
        vals = tl.load(e_ptrs)
    else:
        vals = tl.load(h_ptrs)

    # Store to output
    tl.store(o_ptrs, vals)


@triton.jit
def batched_linear_elem(b_ptr, wT_ptr, o_ptr,
                         B: tl.constexpr, T: tl.constexpr, H: tl.constexpr,
                         stride_b_b: tl.constexpr, stride_b_t: tl.constexpr, stride_b_h: tl.constexpr,
                         stride_w_b: tl.constexpr, stride_w_k: tl.constexpr, stride_w_h: tl.constexpr,
                         stride_o_b: tl.constexpr, stride_o_t: tl.constexpr, stride_o_h: tl.constexpr):
    """
    Compute out[b, t, h] = sum_k b[b, t, k] * wT[k, h], for all b, t, h.
    Launch grid: (B, T, H). Each program computes a single element (b, t, h).
    """
    b_id = tl.program_id(0)
    t_id = tl.program_id(1)
    h_id = tl.program_id(2)

    # Base pointers for the row and weight column
    b_base = b_ptr + b_id * stride_b_b + t_id * stride_b_t
    w_base = wT_ptr + h_id * stride_w_h  # column vector of length H

    # Accumulator
    acc = 0.0

    # Loop over k (hidden dimension) and accumulate
    for k in range(0, H):
        b_val = tl.load(b_base + k * stride_b_h)
        w_val = tl.load(w_base + k * stride_w_k)
        acc += b_val * w_val

    # Store result
    o_ptr_scalar = o_ptr + b_id * stride_o_b + t_id * stride_o_t + h_id * stride_o_h
    tl.store(o_ptr_scalar, acc)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                encoder_hidden_states: torch.Tensor,
                process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version of:
          concatenated = torch.cat([encoder_hidden_states, hidden_states], dim=1)
          processed = concatenated @ process_weight.T
          return processed[:, :text_seq_len, :], processed[:, text_seq_len:, :]

        Requirements:
        - No torch operations on data inside forward (no torch.cat, no transpose, no matmul, no slicing of data).
        - All computation must be in Triton kernels.
        """
        # Ensure CUDA
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Tensors must be on CUDA for Triton kernels."
        # Assume float32 tensors (as in the original). If not, you could cast, but we keep the original dtype.
        # Do not call .contiguous() or any torch ops on data.

        B = hidden_states.shape[0]
        H = hidden_states.shape[2]
        Stext = encoder_hidden_states.shape[1]
        Simg = hidden_states.shape[1]
        T = Stext + Simg

        # Allocate concatenated tensor [B, T, H]
        out_cat = torch.empty((B, T, H), dtype=hidden_states.dtype, device=hidden_states.device)

        # Launch concat kernel: grid over (B, T)
        grid_cat = (B, T)
        concat_by_row[grid_cat](
            encoder_hidden_states, hidden_states, out_cat,
            B=B, Stext=Stext, Simg=Simg, H=H,
            stride_e_b=encoder_hidden_states.stride(0), stride_e_t=encoder_hidden_states.stride(1), stride_e_h=encoder_hidden_states.stride(2),
            stride_h_b=hidden_states.stride(0), stride_h_t=hidden_states.stride(1), stride_h_h=hidden_states.stride(2),
            stride_o_b=out_cat.stride(0), stride_o_t=out_cat.stride(1), stride_o_h=out_cat.stride(2),
            num_warps=1, num_stages=1,
        )

        # Allocate output tensor [B, T, H]
        out = torch.empty((B, T, H), dtype=hidden_states.dtype, device=hidden_states.device)

        # Launch batched linear kernel: grid over (B, T, H), each program computes a single element
        grid_gemm = (B, T, H)
        batched_linear_elem[grid_gemm](
            out_cat, process_weight, out,
            B=B, T=T, H=H,
            stride_b_b=out_cat.stride(0), stride_b_t=out_cat.stride(1), stride_b_h=out_cat.stride(2),
            stride_w_b=process_weight.stride(0), stride_w_k=process_weight.stride(1), stride_w_h=process_weight.stride(2),
            stride_o_b=out.stride(0), stride_o_t=out.stride(1), stride_o_h=out.stride(2),
            num_warps=1, num_stages=1,
        )

        # Split along sequence dimension: return only the requested parts
        processed_encoder = out[:, :Stext, :]
        processed_hidden = out[:, Stext:, :]

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
