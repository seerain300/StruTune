import torch
import triton
import triton.language as tl

# Triton kernel: concatenate along sequence dimension (batch, seq, feature)
# dst: [B, M, D], src0: [B, L_txt, D], src1: [B, L_img, D]
@triton.jit
def concat_seqs_kernel(
    src0_ptr, src1_ptr, dst_ptr,
    B, L_txt, L_img, D,
    s0_stride_b, s0_stride_s, s0_stride_d,
    s1_stride_b, s1_stride_s, s1_stride_d,
    dst_stride_b, dst_stride_s, dst_stride_d,
):
    b = tl.program_id(0)
    t = tl.program_id(1)  # t in [0, M)
    mask_src0 = t < L_txt
    mask_src1 = (t >= L_txt) & (t < (L_txt + L_img))

    s0_off = b * s0_stride_b + t * s0_stride_s
    s1_off = b * s1_stride_b + (t - L_txt) * s1_stride_s
    dst_off = b * dst_stride_b + t * dst_stride_s

    v0 = tl.load(src0_ptr + s0_off, mask=mask_src0, other=0.0)
    v1 = tl.load(src1_ptr + s1_off, mask=mask_src1, other=0.0)

    # For each element, choose v0 or v1; tl.where works elementwise
    out = tl.where(mask_src0, v0, v1)
    tl.store(dst_ptr + dst_off, out)

# Triton kernel: split along sequence dimension into two outputs
# processed: [B, M, D], out1: [B, L_txt, D], out2: [B, L_img, D]
@triton.jit
def split_seqs_kernel(
    processed_ptr, out1_ptr, out2_ptr,
    B, L_txt, L_img, D,
    p_stride_b, p_stride_s, p_stride_d,
    o1_stride_b, o1_stride_s, o1_stride_d,
    o2_stride_b, o2_stride_s, o2_stride_d,
):
    b = tl.program_id(0)
    t = tl.program_id(1)  # t in [0, L_txt) for out1; t in [0, L_img) for out2

    # Copy processed[b, t, :] -> out1[b, t, :]
    p_off = b * p_stride_b + t * p_stride_s
    o1_off = b * o1_stride_b + t * o1_stride_s
    v1 = tl.load(processed_ptr + p_off)
    tl.store(out1_ptr + o1_off, v1)

    # Copy processed[b, t + L_txt, :] -> out2[b, t, :]
    p_off2 = b * p_stride_b + (t + L_txt) * p_stride_s
    o2_off = b * o2_stride_b + t * o2_stride_s
    v2 = tl.load(processed_ptr + p_off2)
    tl.store(out2_ptr + o2_off, v2)

# Triton kernel: matmul C = A @ W^T, where
# A: [B, M, D], W: [D, D], C: [B, M, D]
# Each program instance computes one output element C[b, m, n] by looping over K in chunks.
@triton.jit
def matmul_cat_kernel(
    A_ptr, W_ptr, C_ptr,
    B, M, D,  # A, W, C shapes: [B, M, D], [D, D]
    A_stride_b, A_stride_m, A_stride_d,
    W_stride_d0, W_stride_d1,  # W has shape (D, D): strides for d0 (row), d1 (col)
    C_stride_b, C_stride_m, C_stride_d,
    CHUNK_K: tl.constexpr,
):
    b = tl.program_id(0)
    m = tl.program_id(1)
    n = tl.program_id(2)

    # Accumulator in fp32
    acc = tl.zeros((), dtype=tl.float32)

    # Loop over K dimension in chunks
    for k0 in range(0, D, CHUNK_K):
        offs_k = k0 + tl.arange(0, CHUNK_K)
        mask_k = offs_k < D

        # Load A[b, m, offs_k] as vector
        A_offs = b * A_stride_b + m * A_stride_m + offs_k * A_stride_d
        a = tl.load(A_ptr + A_offs, mask=mask_k, other=0.0)

        # Load W[offs_k, n] as vector: row index offs_k, column index n
        W_offs = offs_k * W_stride_d0 + n * W_stride_d1
        w = tl.load(W_ptr + W_offs, mask=mask_k, other=0.0)

        # Accumulate dot product over chunk
        acc += tl.sum(a * w, axis=0)

    # Store result to C[b, m, n]
    C_off = b * C_stride_b + m * C_stride_m + n * C_stride_d
    tl.store(C_ptr + C_off, acc)

class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        Triton-optimized version of the original run():
        - Uses Triton for concatenation, matmul, and splitting.
        Returns (processed_encoder, processed_hidden) with shapes [B, L_txt, D] and [B, L_img, D].
        """
        assert hidden_states.dim() == 3 and encoder_hidden_states.dim() == 3, "Inputs must be 3D: [B, seq, D]"
        B, L_img, D = hidden_states.shape
        B_e, L_txt, D_e = encoder_hidden_states.shape
        assert B == B_e, "Batch size must match for both inputs"
        assert D == D_e, "Feature dimension must match for both inputs"
        assert process_weight.shape == (D, D), "process_weight must be [D, D]"
        device = hidden_states.device
        in_dtype = hidden_states.dtype

        # 1) Concatenate sequences along the sequence dimension using Triton
        M = L_txt + L_img
        dst = torch.empty((B, M, D), dtype=in_dtype, device=device)

        grid = (B, M)
        concat_seqs_kernel[grid](
            encoder_hidden_states, hidden_states, dst,
            B, L_txt, L_img, D,
            encoder_hidden_states.stride(0), encoder_hidden_states.stride(1), encoder_hidden_states.stride(2),
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            dst.stride(0), dst.stride(1), dst.stride(2),
            num_warps=1, num_stages=1
        )

        # 2) Apply linear projection using Triton matmul: processed = dst @ process_weight.T
        # Compute in fp32 for numerical stability, then cast back
        A = dst
        W = process_weight  # [D, D]
        A32 = A.to(torch.float32)
        W32 = W.to(torch.float32)
        processed32 = torch.empty((B, M, D), dtype=torch.float32, device=device)

        # Launch grid: (B, M, D)
        grid_gemm = (B, M, D)
        matmul_cat_kernel[grid_gemm](
            A32, W32, processed32,
            B, M, D,
            A32.stride(0), A32.stride(1), A32.stride(2),
            W32.stride(0), W32.stride(1),
            processed32.stride(0), processed32.stride(1), processed32.stride(2),
            CHUNK_K=64,  # tuneable; 64 or 128 are good starting points
            num_warps=1, num_stages=1
        )

        # 3) Split into separate streams using Triton
        processed_encoder = torch.empty((B, L_txt, D), dtype=in_dtype, device=device)
        processed_hidden = torch.empty((B, L_img, D), dtype=in_dtype, device=device)

        grid_split = (B, L_txt)
        split_seqs_kernel[grid_split](
            processed32, processed_encoder, processed_hidden,
            B, L_txt, L_img, D,
            processed32.stride(0), processed32.stride(1), processed32.stride(2),
            processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
            processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
            num_warps=1, num_stages=1
        )

        # Return outputs in original dtype
        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
