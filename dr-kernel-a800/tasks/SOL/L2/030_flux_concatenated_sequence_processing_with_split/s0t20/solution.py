import torch
import triton
import triton.language as tl


@triton.jit
def concat_seqs_kernel(
    ehs_ptr,           # *encoder_hidden_states: [B, L_txt, D]
    hs_ptr,            # *hidden_states: [B, L_img, D]
    dst_ptr,           # *output concatenated: [B, L_txt + L_img, D]
    B: tl.int32,
    L_txt: tl.int32,
    L_img: tl.int32,
    D: tl.int32,
    ehs_stride_b: tl.int32, ehs_stride_s: tl.int32, ehs_stride_d: tl.int32,
    hs_stride_b: tl.int32, hs_stride_s: tl.int32, hs_stride_d: tl.int32,
    dst_stride_b: tl.int32, dst_stride_s: tl.int32, dst_stride_d: tl.int32,
    BLOCK_S: tl.constexpr,
):
    # Grid: (B, ceil_div(L_txt + L_img, BLOCK_S))
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)

    # Offsets along the sequence dimension
    s_offsets = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)  # [BLOCK_S]
    total = L_txt + L_img
    mask = s_offsets < total

    # Base pointers for this batch
    ehs_b_ptr = ehs_ptr + pid_b * ehs_stride_b
    hs_b_ptr = hs_ptr + pid_b * hs_stride_b
    dst_b_ptr = dst_ptr + pid_b * dst_stride_b

    # Determine source: first L_txt from ehs, remaining from hs
    is_img = s_offsets >= L_txt
    s_img = s_offsets - L_txt  # valid when is_img

    # Vectorize along D
    d_offsets = tl.arange(0, D)

    # Compute element pointers (cast to int64 for safety)
    ehs_ptrs = ehs_b_ptr + (s_offsets.to(tl.int64) * ehs_stride_s) + (d_offsets[None, :] * ehs_stride_d)
    hs_ptrs = hs_b_ptr + (s_img.to(tl.int64) * hs_stride_s) + (d_offsets[None, :] * hs_stride_d)
    dst_ptrs = dst_b_ptr + (s_offsets.to(tl.int64) * dst_stride_s) + (d_offsets[None, :] * dst_stride_d)

    # Select source pointer based on is_img and mask to avoid out-of-bounds
    src_ptrs = tl.where(is_img[:, None], hs_ptrs, ehs_ptrs)
    mask2d = mask[:, None]

    # Load and store
    vals = tl.load(src_ptrs, mask=mask2d, other=0.0)
    tl.store(dst_ptrs, vals, mask=mask2d)


@triton.jit
def split_rows_kernel(
    src_ptr,           # *processed: [B, L_total, D], L_total = L_txt + L_img
    dst_ptr,           # *output: [B, L_out, D]
    B: tl.int32,
    L_out: tl.int32,
    D: tl.int32,
    src_stride_b: tl.int32, src_stride_s: tl.int32, src_stride_d: tl.int32,
    dst_stride_b: tl.int32, dst_stride_s: tl.int32, dst_stride_d: tl.int32,
    BLOCK_S: tl.constexpr,
):
    # Grid: (B, ceil_div(L_out, BLOCK_S))
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)

    s_offsets = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)  # [BLOCK_S]
    mask = s_offsets < L_out

    src_b_ptr = src_ptr + pid_b * src_stride_b
    dst_b_ptr = dst_ptr + pid_b * dst_stride_b

    src_ptrs = src_b_ptr + (s_offsets.to(tl.int64) * src_stride_s) + (tl.arange(0, D)[None, :] * src_stride_d)
    dst_ptrs = dst_b_ptr + (s_offsets.to(tl.int64) * dst_stride_s) + (tl.arange(0, D)[None, :] * dst_stride_d)

    vals = tl.load(src_ptrs, mask=mask[:, None], other=0.0)
    tl.store(dst_ptrs, vals, mask=mask[:, None])


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        Triton-optimized variant of the original model:
        - Concatenation and splitting are done via Triton kernels.
        - Matmul (linear projection) is done via torch.matmul for robustness.
        """
        # Ensure dtype is float32 and tensors are contiguous
        hidden_states = hidden_states.contiguous().float()
        encoder_hidden_states = encoder_hidden_states.contiguous().float()
        process_weight = process_weight.contiguous().float()

        B = hidden_states.shape[0]
        L_img = hidden_states.shape[1]
        L_txt = encoder_hidden_states.shape[1]
        D = hidden_states.shape[2]

        # 1) Concatenate sequences along sequence dimension using Triton
        L_total = L_txt + L_img
        concatenated = torch.empty((B, L_total, D), device=hidden_states.device, dtype=hidden_states.dtype)

        # Strides
        ehs_stride_b, ehs_stride_s, ehs_stride_d = encoder_hidden_states.stride()
        hs_stride_b, hs_stride_s, hs_stride_d = hidden_states.stride()
        dst_stride_b, dst_stride_s, dst_stride_d = concatenated.stride()

        # Launch concat kernel
        BLOCK_S = 128  # tile along sequence dimension
        grid_concat = (B, triton.cdiv(L_total, BLOCK_S))
        concat_seqs_kernel[grid_concat](
            encoder_hidden_states, hidden_states, concatenated,
            B, L_txt, L_img, D,
            ehs_stride_b, ehs_stride_s, ehs_stride_d,
            hs_stride_b, hs_stride_s, hs_stride_d,
            dst_stride_b, dst_stride_s, dst_stride_d,
            BLOCK_S=BLOCK_S,
            num_warps=4, num_stages=2
        )

        # 2) Linear projection using torch (robust and fast)
        processed = torch.matmul(concatenated, process_weight.t())

        # 3) Split back using Triton
        processed_encoder = torch.empty((B, L_txt, D), device=processed.device, dtype=processed.dtype)
        processed_hidden = torch.empty((B, L_img, D), device=processed.device, dtype=processed.dtype)

        src_stride_b, src_stride_s, src_stride_d = processed.stride()
        dst0_stride_b, dst0_stride_s, dst0_stride_d = processed_encoder.stride()
        dst1_stride_b, dst1_stride_s, dst1_stride_d = processed_hidden.stride()

        # Launch split kernels
        grid_split0 = (B, triton.cdiv(L_txt, BLOCK_S))
        split_rows_kernel[grid_split0](
            processed, processed_encoder,
            B, L_txt, D,
            src_stride_b, src_stride_s, src_stride_d,
            dst0_stride_b, dst0_stride_s, dst0_stride_d,
            BLOCK_S=BLOCK_S,
            num_warps=4, num_stages=2
        )

        grid_split1 = (B, triton.cdiv(L_img, BLOCK_S))
        split_rows_kernel[grid_split1](
            processed, processed_hidden,
            B, L_img, D,
            src_stride_b, src_stride_s, src_stride_d,
            dst1_stride_b, dst1_stride_s, dst1_stride_d,
            BLOCK_S=BLOCK_S,
            num_warps=4, num_stages=2
        )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
