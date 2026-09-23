import torch
import triton
import triton.language as tl


@triton.jit
def concat_seqs_kernel(
    ehs_ptr,  # *encoder_hidden_states [B, L_txt, D]
    hs_ptr,   # *hidden_states [B, L_img, D]
    dst_ptr,  # *output concatenated [B, L_txt + L_img, D]
    B: tl.int32,
    L_txt: tl.int32,
    L_img: tl.int32,
    D: tl.int32,
    ehs_stride_b: tl.int32, ehs_stride_s: tl.int32, ehs_stride_d: tl.int32,
    hs_stride_b: tl.int32, hs_stride_s: tl.int32, hs_stride_d: tl.int32,
    dst_stride_b: tl.int32, dst_stride_s: tl.int32, dst_stride_d: tl.int32,
    BLOCK_M: tl.constexpr,  # tile along sequence (rows)
    BLOCK_N: tl.constexpr,  # tile along feature (cols)
):
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    total_seq = L_txt + L_img

    # Tile offsets
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # sequence positions
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # feature positions

    # Bounds masks
    m_mask = m_offsets < total_seq
    n_mask = n_offsets < D
    mask = m_mask[:, None] & n_mask[None, :]

    # Destination pointers
    dst_ptrs = dst_ptr + pid_b * dst_stride_b + m_offsets[:, None] * dst_stride_s + n_offsets[None, :] * dst_stride_d

    # For each m, if m < L_txt -> load from encoder; else load from hidden at offset m - L_txt
    # Build pointers for encoder and hidden separately and select via masks.
    e_mask = (m_offsets[None, :] < L_txt) & mask
    h_mask = (~e_mask) & mask

    src_e_ptrs = ehs_ptr + pid_b * ehs_stride_b + m_offsets[:, None] * ehs_stride_s + n_offsets[None, :] * ehs_stride_d
    src_h_ptrs = hs_ptr + pid_b * hs_stride_b + (m_offsets[:, None] - L_txt) * hs_stride_s + n_offsets[None, :] * hs_stride_d

    # Load from encoder where applicable
    e_vals = tl.load(src_e_ptrs, mask=e_mask, other=0.0)
    # For rows where m >= L_txt, we need values from hidden at offset m - L_txt; but we need to avoid invalid loads for m < L_txt.
    # Here, h_mask selects rows with m >= L_txt. We'll compute h_vals only where h_mask is true and elsewhere 0 (will be overwritten).
    h_vals = tl.load(src_h_ptrs, mask=h_mask, other=0.0)

    # Compose final values: for m<L_txt, use e_vals; otherwise use h_vals (which corresponds to hidden rows). We use the mask logic:
    # out_vals = where(e_mask, e_vals, where(h_mask, h_vals, 0))
    out_vals = tl.where(e_mask, e_vals, tl.where(h_mask, h_vals, 0.0))

    # Store
    tl.store(dst_ptrs, out_vals, mask=mask)


@triton.jit
def copy_slice_kernel(
    src_ptr,   # processed [B, M_total, D], in fp32
    out_ptr,   # [B, L, D], destination dtype (same as original)
    B: tl.int32,
    start_seq: tl.int32,  # starting row in src to copy
    L: tl.int32,          # length to copy
    D: tl.int32,
    src_stride_b: tl.int32, src_stride_m: tl.int32, src_stride_d: tl.int32,
    out_stride_b: tl.int32, out_stride_m: tl.int32, out_stride_d: tl.int32,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # along L
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # along D

    mask_m = m_offsets < L
    mask_n = n_offsets < D
    mask = mask_m[:, None] & mask_n[None, :]

    # Source pointers: src[b, start_seq + m, d]
    src_ptrs = src_ptr + pid_b * src_stride_b + (start_seq + m_offsets[:, None]) * src_stride_m + n_offsets[None, :] * src_stride_d
    vals = tl.load(src_ptrs, mask=mask, other=0.0)  # fp32

    # Destination pointers: out[b, m, d]
    out_ptrs = out_ptr + pid_b * out_stride_b + m_offsets[:, None] * out_stride_m + n_offsets[None, :] * out_stride_d
    # Store fp32 to destination; destination dtype may be different. For correctness, we assume output tensors are allocated with desired dtype,
    # and Triton will cast appropriately on store. If needed, cast before store:
    tl.store(out_ptrs, vals, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        Triton-only forward with torch matmul:
        1) Concatenate encoder_hidden_states and hidden_states along sequence dimension (Triton).
        2) Compute processed = concatenated @ process_weight.T using torch.matmul (fp32) for correctness.
        3) Split outputs into processed_encoder and processed_hidden using Triton copy kernels.
        Returns: (processed_encoder [B, L_txt, D], processed_hidden [B, L_img, D])
        """
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "All inputs must be CUDA tensors for Triton."

        B = hidden_states.shape[0]
        L_txt = encoder_hidden_states.shape[1]
        L_img = hidden_states.shape[1]
        D = hidden_states.shape[2]

        # Ensure contiguous
        ehs = encoder_hidden_states.contiguous()
        hs = hidden_states.contiguous()
        pw_T = process_weight.t().contiguous()  # [D, D]

        # 1) Concatenate sequences in Triton: dst [B, L_txt + L_img, D]
        total_seq = L_txt + L_img
        dst_concat = torch.empty((B, total_seq, D), device=hs.device, dtype=hs.dtype)

        BLOCK_M = 128
        BLOCK_N = 128
        grid_concat = (B, triton.cdiv(total_seq, BLOCK_M), triton.cdiv(D, BLOCK_N))
        concat_seqs_kernel[grid_concat](
            ehs, hs, dst_concat,
            B, L_txt, L_img, D,
            ehs.stride(0), ehs.stride(1), ehs.stride(2),
            hs.stride(0), hs.stride(1), hs.stride(2),
            dst_concat.stride(0), dst_concat.stride(1), dst_concat.stride(2),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        )

        # 2) Matmul in PyTorch (fp32) for correctness: [B, total_seq, D] @ [D, D] -> [B, total_seq, D]
        processed = torch.matmul(dst_concat.float(), pw_T.float())  # fp32 matmul

        # 3) Split using Triton copy kernels
        processed_encoder = torch.empty((B, L_txt, D), device=hs.device, dtype=hs.dtype)
        processed_hidden = torch.empty((B, L_img, D), device=hs.device, dtype=hs.dtype)

        # Triton copy of first L_txt rows (encoder stream)
        grid_split1 = (B, triton.cdiv(L_txt, BLOCK_M), triton.cdiv(D, BLOCK_N))
        copy_slice_kernel[grid_split1](
            processed,  # fp32 source
            processed_encoder,  # destination in original dtype
            B, 0, L_txt, D,
            processed.stride(0), processed.stride(1), processed.stride(2),
            processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        )

        # Triton copy of remaining L_img rows (image stream)
        grid_split2 = (B, triton.cdiv(L_img, BLOCK_M), triton.cdiv(D, BLOCK_N))
        copy_slice_kernel[grid_split2](
            processed,
            processed_hidden,
            B, L_txt, L_img, D,
            processed.stride(0), processed.stride(1), processed.stride(2),
            processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
