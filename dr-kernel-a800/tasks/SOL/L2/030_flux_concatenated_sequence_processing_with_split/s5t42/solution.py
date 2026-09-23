import torch
import triton
import triton.language as tl


@triton.jit
def concatenate_rows_kernel(
    encoder_ptr,      # *f32, [B, Stext, H]
    hidden_ptr,       # *f32, [B, Simg, H]
    out_ptr,          # *f32, [B, T, H]
    # constexpr sizes
    B: tl.constexpr,
    Stext: tl.constexpr,
    Simg: tl.constexpr,
    H: tl.constexpr,  # hidden_dim
    stride_e_b: tl.constexpr,
    stride_e_t: tl.constexpr,
    stride_e_h: tl.constexpr,
    stride_h_b: tl.constexpr,
    stride_h_s: tl.constexpr,  # Simg index corresponds to t - Stext
    stride_h_h: tl.constexpr,
    stride_o_b: tl.constexpr,
    stride_o_t: tl.constexpr,
    stride_o_h: tl.constexpr,
):
    # 2D grid over (B, T)
    b = tl.program_id(0)
    t = tl.program_id(1)
    if (b >= B) or (t >= T):
        return

    # Determine source: if t < Stext, take from encoder; else take from hidden at s = t - Stext
    # We will copy the row element by element.
    if t < Stext:
        src_ptr = encoder_ptr + b * stride_e_b + t * stride_e_t
        dst_ptr = out_ptr + b * stride_o_b + t * stride_o_t
    else:
        s = t - Stext
        src_ptr = hidden_ptr + b * stride_h_b + s * stride_h_s
        dst_ptr = out_ptr + b * stride_o_b + t * stride_o_t

    # Copy H elements
    for i in range(0, H):
        val = tl.load(src_ptr + i * stride_e_h)
        tl.store(dst_ptr + i * stride_o_h, val)


@triton.jit
def batched_gemm_row_kernel(
    in_ptr,           # *f32, concatenated [B, T, H]
    weight_ptr,       # *f32, process_weight [H, H]
    out_ptr,          # *f32, processed [B, T, H]
    # constexpr sizes
    B: tl.constexpr,
    T: tl.constexpr,
    H: tl.constexpr,  # input/output hidden_dim
    stride_i_b: tl.constexpr,
    stride_i_t: tl.constexpr,
    stride_i_h: tl.constexpr,
    stride_w_k: tl.constexpr,  # weight dim-0 (k)
    stride_w_h: tl.constexpr,  # weight dim-1 (h)
    stride_o_b: tl.constexpr,
    stride_o_t: tl.constexpr,
    stride_o_h: tl.constexpr,
):
    b = tl.program_id(0)
    t = tl.program_id(1)
    if (b >= B) or (t >= T):
        return

    # Load input row vector: in[b, t, :]
    in_row_ptrs = in_ptr + b * stride_i_b + t * stride_i_t
    in_vec = tl.zeros([H], dtype=tl.float32)
    for i in range(0, H):
        in_vec[i] = tl.load(in_row_ptrs + i * stride_i_h)

    # Compute out_row = in_vec @ weight (no bias), i.e., out[h] = sum_k in_vec[k] * weight[k, h]
    out_row_ptrs = out_ptr + b * stride_o_b + t * stride_o_t
    for h in range(0, H):
        acc = 0.0
        for k in range(0, H):
            w_val = tl.load(weight_ptr + k * stride_w_k + h * stride_w_h)
            acc += in_vec[k] * w_val
        tl.store(out_row_ptrs + h * stride_o_h, acc)


class ModelNew(torch.nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        process_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version:
        - Concatenates along the sequence dimension using a Triton kernel.
        - Applies linear projection using a Triton batched GEMV kernel.
        Returns (processed_encoder, processed_hidden) shaped [B, Stext/H, H] as per original run function.
        """
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, \
            "All inputs must be CUDA tensors for Triton execution."
        assert hidden_states.dtype == torch.float32 and encoder_hidden_states.dtype == torch.float32 and process_weight.dtype == torch.float32, \
            "Expected float32 tensors."

        B = hidden_states.shape[0]
        Simg = hidden_states.shape[1]
        Stext = encoder_hidden_states.shape[1]
        H = hidden_states.shape[2]
        T = Stext + Simg

        # Make tensors contiguous for simple stride handling
        e = encoder_hidden_states.contiguous()
        h = hidden_states.contiguous()
        w = process_weight.contiguous()

        # Allocate concatenated tensor
        concatenated = torch.empty((B, T, H), device=h.device, dtype=h.dtype)

        # Launch Triton concatenation kernel
        grid_concat = (B, T)
        concatenate_rows_kernel[grid_concat](
            e, h, concatenated,
            B=B, Stext=Stext, Simg=Simg, H=H,
            stride_e_b=e.stride(0), stride_e_t=e.stride(1), stride_e_h=e.stride(2),
            stride_h_b=h.stride(0), stride_h_s=h.stride(1), stride_h_h=h.stride(2),
            stride_o_b=concatenated.stride(0), stride_o_t=concatenated.stride(1), stride_o_h=concatenated.stride(2),
            num_warps=1,
        )

        # Allocate processed tensor
        processed = torch.empty((B, T, H), device=h.device, dtype=h.dtype)

        # Launch Triton batched GEMV kernel: processed[b, t, :] = concatenated[b, t, :] @ w
        grid_gemm = (B, T)
        batched_gemm_row_kernel[grid_gemm](
            concatenated, w, processed,
            B=B, T=T, H=H,
            stride_i_b=concatenated.stride(0), stride_i_t=concatenated.stride(1), stride_i_h=concatenated.stride(2),
            stride_w_k=w.stride(0), stride_w_h=w.stride(1),
            stride_o_b=processed.stride(0), stride_o_t=processed.stride(1), stride_o_h=processed.stride(2),
            num_warps=1,
        )

        # Split back into encoder and hidden streams
        processed_encoder = processed[:, :Stext, :]
        processed_hidden = processed[:, Stext:, :]
        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
