import torch
import triton
import triton.language as tl


@triton.jit
def concat_seqs_kernel(
    ehs_ptr,    # *encoder_hidden_states [B, L_txt, D]
    hs_ptr,     # *hidden_states [B, L_img, D]
    dst_ptr,    # *output concatenated [B, L_txt + L_img, D]
    B: tl.int32,
    L_txt: tl.int32,
    L_img: tl.int32,
    D: tl.int32,
    ehs_stride_b: tl.int32, ehs_stride_s: tl.int32, ehs_stride_d: tl.int32,
    hs_stride_b: tl.int32, hs_stride_s: tl.int32, hs_stride_d: tl.int32,
    dst_stride_b: tl.int32, dst_stride_s: tl.int32, dst_stride_d: tl.int32,
    BLOCK_S: tl.constexpr,  # tile along sequence
    BLOCK_D: tl.constexpr,  # tile along feature
):
    # Program ids: batch, sequence-tiles, feature-tiles
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_d = tl.program_id(2)

    total_seq = L_txt + L_img

    s_offsets = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)  # sequence positions
    d_offsets = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)  # feature positions

    s_mask = s_offsets < total_seq
    d_mask = d_offsets < D
    mask = s_mask[:, None] & d_mask[None, :]

    # Destination pointers for the tile
    dst_ptrs = dst_ptr + pid_b * dst_stride_b + s_offsets[:, None] * dst_stride_s + d_offsets[None, :] * dst_stride_d

    # For each sequence position s, copy from encoder if s<L_txt, else from hidden at s-L_txt
    e_mask = (s_offsets[:, None] < L_txt) & mask
    h_mask = ((s_offsets[:, None] >= L_txt) & (s_offsets[:, None] < total_seq)) & mask
    e_ptrs = ehs_ptr + pid_b * ehs_stride_b + s_offsets[:, None] * ehs_stride_s + d_offsets[None, :] * ehs_stride_d
    h_ptrs = hs_ptr + pid_b * hs_stride_b + (s_offsets[:, None] - L_txt) * hs_stride_s + d_offsets[None, :] * hs_stride_d

    e_vals = tl.load(e_ptrs, mask=e_mask, other=0.0)
    h_vals = tl.load(h_ptrs, mask=h_mask, other=0.0)
    # If both masks are false, default to 0
    out_vals = tl.where(e_mask, e_vals, tl.where(h_mask, h_vals, 0.0))
    tl.store(dst_ptrs, out_vals, mask=mask)


@triton.jit
def copy_slice_kernel(
    src_ptr,      # *src [B, M, D]
    dst_ptr,      # *dst [B, M_slice, D]
    B: tl.int32,
    M: tl.int32,  # total rows in src
    M_slice: tl.int32,  # rows to copy
    D: tl.int32,
    src_stride_b: tl.int32, src_stride_m: tl.int32, src_stride_d: tl.int32,
    dst_stride_b: tl.int32, dst_stride_m: tl.int32, dst_stride_d: tl.int32,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_d = tl.program_id(2)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # rows in the slice
    d_offsets = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)  # feature positions

    m_mask = m_offsets < M_slice
    d_mask = d_offsets < D
    mask = m_mask[:, None] & d_mask[None, :]

    src_ptrs = src_ptr + pid_b * src_stride_b + m_offsets[:, None] * src_stride_m + d_offsets[None, :] * src_stride_d
    dst_ptrs = dst_ptr + pid_b * dst_stride_b + m_offsets[:, None] * dst_stride_m + d_offsets[None, :] * dst_stride_d

    vals = tl.load(src_ptrs, mask=mask, other=0.0)
    tl.store(dst_ptrs, vals, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        Triton-optimized forward:
        - Concatenate encoder_hidden_states and hidden_states along sequence (Triton).
        - Compute linear projection using torch (GPU matmul).
        - Split processed tensor into encoder and hidden outputs (Triton copy kernels).
        """
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "All inputs must be CUDA tensors."
        B = hidden_states.shape[0]
        L_txt = encoder_hidden_states.shape[1]
        L_img = hidden_states.shape[1]
        D = hidden_states.shape[2]

        # Ensure contiguous layout
        ehs = encoder_hidden_states.contiguous()
        hs = hidden_states.contiguous()
        # Concatenate along sequence dimension using Triton
        total_seq = L_txt + L_img
        dst_concat = torch.empty((B, total_seq, D), device=hs.device, dtype=hs.dtype)

        # Choose tiles; D is typically 64/128/256 etc. Use 64 for robustness
        BLOCK_S = 64
        BLOCK_D = 64
        grid_concat = (B, triton.cdiv(total_seq, BLOCK_S), triton.cdiv(D, BLOCK_D))
        concat_seqs_kernel[grid_concat](
            ehs, hs, dst_concat,
            B, L_txt, L_img, D,
            ehs.stride(0), ehs.stride(1), ehs.stride(2),
            hs.stride(0), hs.stride(1), hs.stride(2),
            dst_concat.stride(0), dst_concat.stride(1), dst_concat.stride(2),
            BLOCK_S=BLOCK_S, BLOCK_D=BLOCK_D,
            num_warps=4, num_stages=2,
        )

        # Linear projection using torch (GPU). process_weight is [D, D], we want W = process_weight.T
        W_T = process_weight.t().contiguous()  # [D, D]
        # processed [B, total_seq, D] = dst_concat @ W_T
        processed = torch.matmul(dst_concat, W_T)

        # Split into processed_encoder [B, L_txt, D] and processed_hidden [B, L_img, D] using Triton copy kernels
        # First, we allocate outputs and use Triton to copy slices.
        processed_encoder = torch.empty((B, L_txt, D), device=hs.device, dtype=hs.dtype)
        processed_hidden = torch.empty((B, L_img, D), device=hs.device, dtype=hs.dtype)

        # Grid over batch, rows, features
        BLOCK_M = 64
        BLOCK_D = 64
        grid_encoder = (B, triton.cdiv(L_txt, BLOCK_M), triton.cdiv(D, BLOCK_D))
        copy_slice_kernel[grid_encoder](
            processed, processed_encoder,
            B, processed.shape[1], L_txt, D,
            processed.stride(0), processed.stride(1), processed.stride(2),
            processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
            BLOCK_M=BLOCK_M, BLOCK_D=BLOCK_D,
            num_warps=4, num_stages=2,
        )

        grid_hidden = (B, triton.cdiv(L_img, BLOCK_M), triton.cdiv(D, BLOCK_D))
        # For hidden, slice from rows L_txt to L_txt + L_img - 1
        copy_slice_kernel[grid_hidden](
            processed, processed_hidden,
            B, processed.shape[1], L_img, D,
            processed.stride(0), processed.stride(1), processed.stride(2),
            processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
            BLOCK_M=BLOCK_M, BLOCK_D=BLOCK_D,
            num_warps=4, num_stages=2,
        )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
