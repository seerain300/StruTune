import torch
import triton
import triton.language as tl


@triton.jit
def _concat_kernel(
    src1_ptr,  # *f32, encoder_hidden_states [B, Stext, H]
    src2_ptr,  # *f32, hidden_states [B, Simg, H]
    out_ptr,   # *f32, out_cat [B, T, H]
    B, Stext, Simg, H, T,
    stride_src1_b, stride_src1_t, stride_src1_h,
    stride_src2_b, stride_src2_t, stride_src2_h,
    stride_out_b, stride_out_t, stride_out_h,
):
    # 2D grid: (B, T)
    b = tl.program_id(0)
    t = tl.program_id(1)

    if (b >= B) or (t >= T):
        return

    if t < Stext:
        # Load from encoder_hidden_states[b, t, :]
        offs_h = tl.arange(0, H)
        src1_offset = b * stride_src1_b + t * stride_src1_t + offs_h * stride_src1_h
        vals = tl.load(src1_ptr + src1_offset)
        out_offset = b * stride_out_b + t * stride_out_t + offs_h * stride_out_h
        tl.store(out_ptr + out_offset, vals)
    else:
        # Load from hidden_states[b, t - Stext, :]
        t2 = t - Stext
        offs_h = tl.arange(0, H)
        src2_offset = b * stride_src2_b + t2 * stride_src2_t + offs_h * stride_src2_h
        vals = tl.load(src2_ptr + src2_offset)
        out_offset = b * stride_out_b + t * stride_out_t + offs_h * stride_out_h
        tl.store(out_ptr + out_offset, vals)


@triton.jit
def _batched_gemm_kernel(
    A_ptr,          # *f32, out_cat [B, T, H], contiguous
    B_ptr,          # *f32, process_weight_T [H, H], contiguous
    Out_ptr,        # *f32, output [B, T, H], contiguous
    B_dim, T_dim, H_dim,
    stride_A_b, stride_A_t, stride_A_h,
    stride_B_k, stride_B_h,  # B is [H, H]
    stride_Out_b, stride_Out_t, stride_Out_h,
    BLOCK_B: tl.constexpr, BLOCK_T: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Grid: (ceil_div(B, BLOCK_B), ceil_div(T, BLOCK_T), ceil_div(H, BLOCK_H))
    pid_b = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_h = tl.program_id(2)

    b_offsets = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    t_offsets = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    h_offsets = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)

    mask_b = b_offsets < B_dim
    mask_t = t_offsets < T_dim
    mask_h = h_offsets < H_dim

    # Accumulator for this tile: [BLOCK_T, BLOCK_H], per batch in BLOCK_B
    # We keep acc as [BLOCK_T, BLOCK_H] and loop over BLOCK_B to store per b
    # More efficient: create acc for each b in tile, so acc shape [BLOCK_B, BLOCK_T, BLOCK_H].
    # Triton can handle per-program accumulation into multiple outputs; simplest is to loop b
    # and store directly. This avoids a 3D accumulator.
    for b_i in range(0, BLOCK_B):
        b = b_offsets[b_i]
        if not mask_b[b_i]:
            continue

        acc = tl.zeros((BLOCK_T, BLOCK_H), dtype=tl.float32)

        # Loop over hidden dimension in chunks
        for k_start in range(0, H_dim, BLOCK_K):
            k_offsets = k_start + tl.arange(0, BLOCK_K)
            mask_k = k_offsets < H_dim

            # Load A tile: [BLOCK_T, BLOCK_K] for this b
            a_ptrs = A_ptr + b * stride_A_b + t_offsets[:, None] * stride_A_t + k_offsets[None, :] * stride_A_h
            a_mask = mask_t[:, None] & mask_k[None, :]
            a = tl.load(a_ptrs, mask=a_mask, other=0.0)

            # Load B tile: [BLOCK_K, BLOCK_H]
            b_ptrs = B_ptr + k_offsets[:, None] * stride_B_k + h_offsets[None, :] * stride_B_h
            b_mask = mask_k[:, None] & mask_h[None, :]
            b_tile = tl.load(b_ptrs, mask=b_mask, other=0.0)

            # Accumulate
            acc += tl.dot(a, b_tile)

        # Store result for this b: [BLOCK_T, BLOCK_H]
        out_ptrs = Out_ptr + b * stride_Out_b + t_offsets[:, None] * stride_Out_t + h_offsets[None, :] * stride_Out_h
        out_mask = mask_t[:, None] & mask_h[None, :]
        tl.store(out_ptrs, acc, mask=out_mask)


class ModelNew(torch.nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        process_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized forward:
        - Concatenates encoder_hidden_states and hidden_states along sequence dimension (Triton).
        - Applies linear projection process_weight.T to the concatenated sequence (Triton GEMM).
        - Splits the result back into processed_encoder and processed_hidden.
        """
        # Ensure CUDA tensors and contiguity
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "All tensors must be on CUDA for Triton."
        B = hidden_states.shape[0]
        Stext = encoder_hidden_states.shape[1]
        Simg = hidden_states.shape[1]
        H = hidden_states.shape[2]
        assert encoder_hidden_states.shape[0] == B and encoder_hidden_states.shape[2] == H
        assert process_weight.shape[0] == H and process_weight.shape[1] == H

        # Make inputs contiguous
        enc = encoder_hidden_states.contiguous()
        hid = hidden_states.contiguous()
        wt = process_weight.contiguous()
        T = Stext + Simg

        device = hidden_states.device
        dtype = hidden_states.dtype

        # Allocate concatenated output [B, T, H]
        out_cat = torch.empty((B, T, H), device=device, dtype=dtype)

        # Launch concat kernel: grid = (B, T)
        grid_concat = (B, T)
        _concat_kernel[grid_concat](
            enc, hid, out_cat,
            B, Stext, Simg, H, T,
            enc.stride(0), enc.stride(1), enc.stride(2),
            hid.stride(0), hid.stride(1), hid.stride(2),
            out_cat.stride(0), out_cat.stride(1), out_cat.stride(2),
            num_warps=2, num_stages=2,
        )

        # Prepare process_weight_T [H, H] contiguous
        wt_T = wt.transpose(0, 1).contiguous()

        # Allocate output [B, T, H]
        out = torch.empty((B, T, H), device=device, dtype=dtype)

        # Strides for contiguous tensors
        stride_A_b = T * H
        stride_A_t = H
        stride_A_h = 1

        stride_B_k = wt_T.stride(0)
        stride_B_h = wt_T.stride(1)

        stride_Out_b = T * H
        stride_Out_t = H
        stride_Out_h = 1

        # Tiling parameters
        BLOCK_B = 1  # we process one batch element per program in the GEMM kernel via loop
        BLOCK_T = 64
        BLOCK_H = 64
        BLOCK_K = 32

        # Grid: (ceil_div(B, BLOCK_B), ceil_div(T, BLOCK_T), ceil_div(H, BLOCK_H))
        grid_gemm = (triton.cdiv(B, BLOCK_B), triton.cdiv(T, BLOCK_T), triton.cdiv(H, BLOCK_H))

        _batched_gemm_kernel[grid_gemm](
            out_cat, wt_T, out,
            B, T, H,
            stride_A_b, stride_A_t, stride_A_h,
            stride_B_k, stride_B_h,
            stride_Out_b, stride_Out_t, stride_Out_h,
            BLOCK_B=BLOCK_B, BLOCK_T=BLOCK_T, BLOCK_H=BLOCK_H, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        # Split back into streams (views in Python; no torch compute)
        processed_encoder = out[:, :Stext, :]
        processed_hidden = out[:, Stext:, :]

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
