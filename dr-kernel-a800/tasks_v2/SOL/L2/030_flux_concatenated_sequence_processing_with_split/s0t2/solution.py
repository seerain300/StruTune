import torch
import triton
import triton.language as tl


@triton.jit
def concat_seqs_kernel(
    src1_ptr,      # *ptr to encoder_hidden_states: [B, L_txt, D]
    src2_ptr,      # *ptr to hidden_states: [B, L_img, D]
    dst_ptr,       # *ptr to concatenated output: [B, L_txt + L_img, D]
    B: tl.int32,
    L_txt: tl.int32,
    L_img: tl.int32,
    D: tl.int32,
    stride_s1_b: tl.int32,
    stride_s1_m: tl.int32,
    stride_s1_k: tl.int32,
    stride_s2_b: tl.int32,
    stride_s2_m: tl.int32,
    stride_s2_k: tl.int32,
    stride_dst_b: tl.int32,
    stride_dst_m: tl.int32,
    stride_dst_k: tl.int32,
    BLOCK_M: tl.constexpr,   # tile along sequence (concat length)
    BLOCK_K: tl.constexpr,   # tile along hidden_dim (loop over k)
):
    # Grid: (B, ceil((L_txt + L_img) / BLOCK_M))
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)

    total_seq = L_txt + L_img
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = m_offsets < total_seq

    # For each hidden_dim element k
    for k in range(0, D):
        # We need to write to two regions in dst:
        # dst[:, :L_txt, k] from src1
        # dst[:, L_txt:, k] from src2 (with src2 index m = m_offsets - L_txt)
        m_src1 = m_offsets
        m_src2 = m_offsets - L_txt
        mask_src1 = mask_m
        mask_src2 = mask_m & (m_offsets >= L_txt)

        src1_ptrs = src1_ptr + pid_b * stride_s1_b + m_src1 * stride_s1_m + k * stride_s1_k
        src2_ptrs = src2_ptr + pid_b * stride_s2_b + m_src2 * stride_s2_m + k * stride_s2_k
        dst_ptrs = dst_ptr + pid_b * stride_dst_b + m_offsets * stride_dst_m + k * stride_dst_k

        # Load values (mask ensures we don't read out of bounds)
        val_src1 = tl.load(src1_ptrs, mask=mask_src1, other=0.0)
        val_src2 = tl.load(src2_ptrs, mask=mask_src2, other=0.0)
        # Select src1 for positions < L_txt, src2 otherwise
        val = tl.where(m_offsets < L_txt, val_src1, val_src2)

        # Store to destination
        tl.store(dst_ptrs, val, mask=mask_m)


@triton.jit
def split_encoder_kernel(
    src_ptr,       # *ptr to processed: [B, total_seq, D], total_seq = L_txt + L_img
    dst_ptr,       # *ptr to processed_encoder: [B, L_txt, D]
    B: tl.int32,
    L_txt: tl.int32,
    D: tl.int32,
    stride_src_b: tl.int32,
    stride_src_m: tl.int32,
    stride_src_k: tl.int32,
    stride_dst_b: tl.int32,
    stride_dst_m: tl.int32,
    stride_dst_k: tl.int32,
    BLOCK_M: tl.constexpr,   # tile along sequence for encoder part
    BLOCK_K: tl.constexpr,   # tile along hidden_dim
):
    # Grid: (B, ceil(L_txt / BLOCK_M))
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = m_offsets < L_txt

    for k in range(0, D):
        src_ptrs = src_ptr + pid_b * stride_src_b + m_offsets * stride_src_m + k * stride_src_k
        dst_ptrs = dst_ptr + pid_b * stride_dst_b + m_offsets * stride_dst_m + k * stride_dst_k
        vals = tl.load(src_ptrs, mask=mask_m, other=0.0)
        tl.store(dst_ptrs, vals, mask=mask_m)


@triton.jit
def split_hidden_kernel(
    src_ptr,       # *ptr to processed: [B, total_seq, D], total_seq = L_txt + L_img
    dst_ptr,       # *ptr to processed_hidden: [B, L_img, D]
    B: tl.int32,
    L_txt: tl.int32,
    L_img: tl.int32,
    D: tl.int32,
    stride_src_b: tl.int32,
    stride_src_m: tl.int32,
    stride_src_k: tl.int32,
    stride_dst_b: tl.int32,
    stride_dst_m: tl.int32,
    stride_dst_k: tl.int32,
    BLOCK_M: tl.constexpr,   # tile along sequence for hidden part
    BLOCK_K: tl.constexpr,   # tile along hidden_dim
):
    # Grid: (B, ceil(L_img / BLOCK_M))
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)

    total_seq = L_txt + L_img
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # relative indices within hidden part
    m_offsets_in_src = m_offsets + L_txt                # map to src: m = L_txt + m_offsets
    mask_m = m_offsets < L_img

    for k in range(0, D):
        src_ptrs = src_ptr + pid_b * stride_src_b + (L_txt + m_offsets) * stride_src_m + k * stride_src_k
        dst_ptrs = dst_ptr + pid_b * stride_dst_b + m_offsets * stride_dst_m + k * stride_dst_k
        vals = tl.load(src_ptrs, mask=mask_m, other=0.0)
        tl.store(dst_ptrs, vals, mask=mask_m)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        # Shapes
        B = hidden_states.shape[0]
        L_img = hidden_states.shape[1]
        L_txt = encoder_hidden_states.shape[1]
        D = hidden_states.shape[2]
        assert process_weight.shape == (D, D), f"process_weight must have shape [hidden_dim, hidden_dim], got {process_weight.shape}"

        # Ensure contiguity for performance
        src1 = encoder_hidden_states.contiguous()
        src2 = hidden_states.contiguous()
        W = process_weight.contiguous()

        # 1) Concatenate along sequence dimension using Triton kernel
        total_seq = L_txt + L_img
        concatenated = torch.empty((B, total_seq, D), device=src1.device, dtype=src1.dtype)

        BLOCK_M = 128  # tile over sequence
        BLOCK_K = 1    # iterate over hidden_dim element by element
        grid_concat = (B, triton.cdiv(total_seq, BLOCK_M))
        concat_seqs_kernel[grid_concat](
            src1, src2, concatenated,
            B, L_txt, L_img, D,
            src1.stride(0), src1.stride(1), src1.stride(2),
            src2.stride(0), src2.stride(1), src2.stride(2),
            concatenated.stride(0), concatenated.stride(1), concatenated.stride(2),
            BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=1,
        )

        # 2) Linear projection with torch.matmul (exact correctness)
        # processed = concatenated @ process_weight.T
        processed = torch.matmul(concatenated, W.t())

        # 3) Split into two streams using Triton kernels
        processed_encoder = torch.empty((B, L_txt, D), device=processed.device, dtype=processed.dtype)
        processed_hidden = torch.empty((B, L_img, D), device=processed.device, dtype=processed.dtype)

        BLOCK_M_SPLIT = 128
        BLOCK_K_SPLIT = 1
        # For encoder part: grid over (B, ceil(L_txt / BLOCK_M_SPLIT))
        grid_encoder = (B, triton.cdiv(L_txt, BLOCK_M_SPLIT))
        split_encoder_kernel[grid_encoder](
            processed, processed_encoder,
            B, L_txt, D,
            processed.stride(0), processed.stride(1), processed.stride(2),
            processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
            BLOCK_M=BLOCK_M_SPLIT, BLOCK_K=BLOCK_K_SPLIT,
            num_warps=4, num_stages=1,
        )

        # For hidden part: grid over (B, ceil(L_img / BLOCK_M_SPLIT))
        grid_hidden = (B, triton.cdiv(L_img, BLOCK_M_SPLIT))
        split_hidden_kernel[grid_hidden](
            processed, processed_hidden,
            B, L_txt, L_img, D,
            processed.stride(0), processed.stride(1), processed.stride(2),
            processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
            BLOCK_M=BLOCK_M_SPLIT, BLOCK_K=BLOCK_K_SPLIT,
            num_warps=4, num_stages=1,
        )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
