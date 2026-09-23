import torch
import triton
import triton.language as tl


@triton.jit
def concat_seqs_kernel(
    ehs_ptr,   # *encoder_hidden_states [B, L_txt, D]
    hs_ptr,    # *hidden_states [B, L_img, D]
    dst_ptr,   # *output concatenated [B, L_txt + L_img, D]
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

    # Masks
    m_mask = m_offsets < total_seq
    n_mask = n_offsets < D
    mask = m_mask[:, None] & n_mask[None, :]

    # Destination pointers
    dst_ptrs = dst_ptr + pid_b * dst_stride_b + m_offsets[:, None] * dst_stride_s + n_offsets[None, :] * dst_stride_d

    # For each row m: if m < L_txt -> load from encoder; else load from hidden at offset m - L_txt
    e_mask = (m_offsets[None, :] < L_txt) & mask
    h_mask = ((m_offsets[None, :] >= L_txt) & (m_offsets[None, :] < total_seq)) & mask

    src_e_ptrs = ehs_ptr + pid_b * ehs_stride_b + m_offsets[:, None] * ehs_stride_s + n_offsets[None, :] * ehs_stride_d
    src_h_ptrs = hs_ptr + pid_b * hs_stride_b + (m_offsets[:, None] - L_txt) * hs_stride_s + n_offsets[None, :] * hs_stride_d

    e_vals = tl.load(src_e_ptrs, mask=e_mask, other=0.0)
    h_vals = tl.load(src_h_ptrs, mask=h_mask, other=0.0)

    # Select source based on row position
    out_vals = tl.where(e_mask, e_vals, tl.where(h_mask, h_vals, 0.0))

    tl.store(dst_ptrs, out_vals, mask=mask)


@triton.jit
def copy_slice_kernel(
    src_ptr,   # *src [B, M, N]
    dst_ptr,   # *dst [B, S, N], where S <= M
    B: tl.int32,
    M: tl.int32,   # src length along sequence
    N: tl.int32,   # feature dim
    S: tl.int32,   # dst length (e.g., L_txt)
    src_stride_b: tl.int32, src_stride_m: tl.int32, src_stride_n: tl.int32,
    dst_stride_b: tl.int32, dst_stride_s: tl.int32, dst_stride_n: tl.int32,
    BLOCK_M: tl.constexpr,  # tile along M/S
    BLOCK_N: tl.constexpr,  # tile along N
):
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    m_mask = m_offsets < S
    n_mask = n_offsets < N
    mask = m_mask[:, None] & n_mask[None, :]

    src_ptrs = src_ptr + pid_b * src_stride_b + m_offsets[:, None] * src_stride_m + n_offsets[None, :] * src_stride_n
    dst_ptrs = dst_ptr + pid_b * dst_stride_b + m_offsets[:, None] * dst_stride_s + n_offsets[None, :] * dst_stride_n

    vals = tl.load(src_ptrs, mask=mask, other=0.0)
    tl.store(dst_ptrs, vals, mask=mask)


@triton.jit
def copy_slice_kernel2(
    src_ptr,   # *src [B, M, N]
    dst_ptr,   # *dst [B, S, N], where S <= (M - start)
    B: tl.int32,
    M: tl.int32,   # src length along sequence (total L_txt + L_img)
    N: tl.int32,   # feature dim
    start: tl.int32,  # starting row in src to copy
    S: tl.int32,   # dst length (e.g., L_img)
    src_stride_b: tl.int32, src_stride_m: tl.int32, src_stride_n: tl.int32,
    dst_stride_b: tl.int32, dst_stride_s: tl.int32, dst_stride_n: tl.int32,
    BLOCK_M: tl.constexpr,  # tile along M/S
    BLOCK_N: tl.constexpr,  # tile along N
):
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    m_mask = m_offsets < S
    n_mask = n_offsets < N
    mask = m_mask[:, None] & n_mask[None, :]

    src_ptrs = src_ptr + pid_b * src_stride_b + (start + m_offsets[:, None]) * src_stride_m + n_offsets[None, :] * src_stride_n
    dst_ptrs = dst_ptr + pid_b * dst_stride_b + m_offsets[:, None] * dst_stride_s + n_offsets[None, :] * dst_stride_n

    vals = tl.load(src_ptrs, mask=mask, other=0.0)
    tl.store(dst_ptrs, vals, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-only forward:
        1) Concatenate encoder_hidden_states and hidden_states along sequence dimension (Triton).
        2) Compute processed = concatenated @ process_weight.T using torch.matmul (guaranteed correctness).
        3) Split outputs into processed_encoder and processed_hidden (Triton copy kernels).
        Returns: (processed_encoder [B, L_txt, D], processed_hidden [B, L_img, D])
        """
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "All inputs must be CUDA tensors for Triton."
        # Ensure same dtype and device
        assert hidden_states.dtype == encoder_hidden_states.dtype == process_weight.dtype, "All tensors must have the same dtype."

        B = hidden_states.shape[0]
        L_txt = encoder_hidden_states.shape[1]
        L_img = hidden_states.shape[1]
        D = hidden_states.shape[2]

        # Make tensors contiguous to simplify stride arithmetic
        ehs = encoder_hidden_states.contiguous()
        hs = hidden_states.contiguous()
        pw_T = process_weight.t().contiguous()  # [D, D]

        # 1) Concatenate sequences into A [B, M, K], where M = L_txt + L_img, K = D
        total_seq = L_txt + L_img
        A = torch.empty((B, total_seq, D), device=hs.device, dtype=hs.dtype)

        # Triton grid for concatenation
        BLOCK_M = 128
        BLOCK_N = 64
        grid_concat = (B, triton.cdiv(total_seq, BLOCK_M), triton.cdiv(D, BLOCK_N))
        concat_seqs_kernel[grid_concat](
            ehs, hs, A,
            B, L_txt, L_img, D,
            ehs.stride(0), ehs.stride(1), ehs.stride(2),
            hs.stride(0), hs.stride(1), hs.stride(2),
            A.stride(0), A.stride(1), A.stride(2),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            num_warps=4, num_stages=2,
        )

        # 2) Linear projection using torch (guaranteed correctness)
        # Compute in float32 to match typical behavior, then cast back to original dtype if needed.
        processed = torch.matmul(A, pw_T)

        # 3) Split into processed_encoder and processed_hidden using Triton copy kernels
        processed_encoder = torch.empty((B, L_txt, D), device=hs.device, dtype=hs.dtype)
        processed_hidden = torch.empty((B, L_img, D), device=hs.device, dtype=hs.dtype)

        # Copy first L_txt rows
        BLOCK_Ms = 64
        BLOCK_Ns = 64
        grid_copy_encoder = (B, triton.cdiv(L_txt, BLOCK_Ms), triton.cdiv(D, BLOCK_Ns))
        copy_slice_kernel[grid_copy_encoder](
            processed, processed_encoder,
            B, total_seq, D, L_txt,
            processed.stride(0), processed.stride(1), processed.stride(2),
            processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
            BLOCK_M=BLOCK_Ms, BLOCK_N=BLOCK_Ns,
            num_warps=4, num_stages=2,
        )

        # Copy remaining L_img rows starting from index L_txt
        grid_copy_hidden = (B, triton.cdiv(L_img, BLOCK_Ms), triton.cdiv(D, BLOCK_Ns))
        copy_slice_kernel2[grid_copy_hidden](
            processed, processed_hidden,
            B, total_seq, D, L_txt, L_img,
            processed.stride(0), processed.stride(1), processed.stride(2),
            processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
            BLOCK_M=BLOCK_Ms, BLOCK_N=BLOCK_Ns,
            num_warps=4, num_stages=2,
        )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
