import torch
import triton
import triton.language as tl


@triton.jit
def concat_seqs_kernel(
    src1_ptr,   # *encoder_hidden_states [B, L_txt, D]
    src2_ptr,   # *hidden_states [B, L_img, D]
    dst_ptr,    # *output concatenated [B, L_txt + L_img, D]
    B: tl.int32,
    L_txt: tl.int32,
    L_img: tl.int32,
    D: tl.int32,
    src1_stride_b: tl.int32, src1_stride_s: tl.int32, src1_stride_d: tl.int32,
    src2_stride_b: tl.int32, src2_stride_s: tl.int32, src2_stride_d: tl.int32,
    dst_stride_b: tl.int32, dst_stride_s: tl.int32, dst_stride_d: tl.int32,
    BLOCK_D: tl.constexpr,
):
    # Grid: (B, L_txt + L_img, ceil(D / BLOCK_D))
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_d_block = tl.program_id(2)

    d_offsets = pid_d_block * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = d_offsets < D

    src1_b_ptr = src1_ptr + pid_b * src1_stride_b
    src2_b_ptr = src2_ptr + pid_b * src2_stride_b
    dst_b_ptr = dst_ptr + pid_b * dst_stride_b

    # Write encoder part: s in [0, L_txt)
    vals = tl.load(
        src1_b_ptr + pid_s * src1_stride_s + d_offsets * src1_stride_d,
        mask=mask_d,
        other=0.0
    )
    tl.store(
        dst_b_ptr + pid_s * dst_stride_s + d_offsets * dst_stride_d,
        vals,
        mask=mask_d
    )

    # Write hidden part: s in [L_txt, L_txt + L_img)
    vals2 = tl.load(
        src2_b_ptr + (pid_s - L_txt) * src2_stride_s + d_offsets * src2_stride_d,
        mask=mask_d,
        other=0.0
    )
    tl.store(
        dst_b_ptr + (pid_s + L_txt) * dst_stride_s + d_offsets * dst_stride_d,
        vals2,
        mask=mask_d
    )


@triton.jit
def batched_matmul_kernel(
    A_ptr,     # *concatenated [B, M, K]
    W_ptr,     # *process_weight.T [K, N]
    C_ptr,     # *output [B, M, N]
    B: tl.int32,
    M: tl.int32,
    N: tl.int32,
    K: tl.int32,
    A_stride_b: tl.int32, A_stride_m: tl.int32, A_stride_k: tl.int32,
    W_stride_k: tl.int32, W_stride_n: tl.int32,
    C_stride_b: tl.int32, C_stride_m: tl.int32, C_stride_n: tl.int32,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Grid: (B, ceil(M / BLOCK_M), ceil(N / BLOCK_N))
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)

        # Load A tile: [BLOCK_M, BLOCK_K]
        a_ptrs = A_ptr + pid_b * A_stride_b + m_offsets[:, None] * A_stride_m + k_offsets[None, :] * A_stride_k
        a_mask = (m_offsets[:, None] < M) & (k_offsets[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # Load W^T tile: [BLOCK_K, BLOCK_N]
        wt_ptrs = W_ptr + k_offsets[:, None] * W_stride_k + n_offsets[None, :] * W_stride_n
        wt_mask = (k_offsets[:, None] < K) & (n_offsets[None, :] < N)
        wt = tl.load(wt_ptrs, mask=wt_mask, other=0.0)

        # Accumulate in float32: acc += a @ wt
        acc += tl.dot(a.to(tl.float32), wt.to(tl.float32))

    # Store results
    c_ptrs = C_ptr + pid_b * C_stride_b + m_offsets[:, None] * C_stride_m + n_offsets[None, :] * C_stride_n
    c_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


@triton.jit
def split_seqs_kernel(
    processed_ptr,   # *C [B, M, D] with M=L_txt+L_img
    out1_ptr,        # *processed_encoder [B, L_txt, D]
    out2_ptr,        # *processed_hidden [B, L_img, D]
    B: tl.int32,
    L_txt: tl.int32,
    L_img: tl.int32,
    D: tl.int32,
    processed_stride_b: tl.int32, processed_stride_s: tl.int32, processed_stride_d: tl.int32,
    out1_stride_b: tl.int32, out1_stride_s: tl.int32, out1_stride_d: tl.int32,
    out2_stride_b: tl.int32, out2_stride_s: tl.int32, out2_stride_d: tl.int32,
    BLOCK_D: tl.constexpr,
):
    # Write processed_encoder: rows s in [0, L_txt)
    # Grid: (B, L_txt, ceil(D / BLOCK_D))
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_d_block = tl.program_id(2)

    d_offsets = pid_d_block * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = d_offsets < D

    src_b_ptr = processed_ptr + pid_b * processed_stride_b
    out1_b_ptr = out1_ptr + pid_b * out1_stride_b
    out2_b_ptr = out2_ptr + pid_b * out2_stride_b

    vals1 = tl.load(
        src_b_ptr + pid_s * processed_stride_s + d_offsets * processed_stride_d,
        mask=mask_d,
        other=0.0
    )
    tl.store(
        out1_b_ptr + pid_s * out1_stride_s + d_offsets * out1_stride_d,
        vals1,
        mask=mask_d
    )

    # Write processed_hidden: rows s in [L_txt, L_txt + L_img)
    vals2 = tl.load(
        src_b_ptr + (pid_s + L_txt) * processed_stride_s + d_offsets * processed_stride_d,
        mask=mask_d,
        other=0.0
    )
    tl.store(
        out2_b_ptr + (pid_s - L_txt) * out2_stride_s + d_offsets * out2_stride_d,
        vals2,
        mask=mask_d
    )


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized forward:
        - Concatenate along sequence dim using Triton copy kernels (no torch.cat).
        - Compute linear projection via Triton batched matmul (no torch.matmul).
        - Split outputs back using Triton copy kernel.
        All computation is performed by Triton kernels launched from forward.
        """
        assert hidden_states.ndim == 3 and encoder_hidden_states.ndim == 3, "Inputs must be 3D tensors [B, L, D]"
        B = hidden_states.shape[0]
        L_img = hidden_states.shape[1]
        L_txt = encoder_hidden_states.shape[1]
        D = hidden_states.shape[2]
        assert encoder_hidden_states.shape[0] == B and encoder_hidden_states.shape[2] == D
        assert process_weight.shape == (D, D)

        device = hidden_states.device
        dtype = hidden_states.dtype

        # 1) Concatenate sequences along the sequence dimension using Triton
        # Allocate output concatenated tensor [B, L_txt + L_img, D]
        M = L_txt + L_img
        concatenated = torch.empty((B, M, D), device=device, dtype=dtype)
        # Launch Triton kernel
        BLOCK_D = 128
        grid_concat = (B, M, triton.cdiv(D, BLOCK_D))
        concat_seqs_kernel[grid_concat](
            encoder_hidden_states, hidden_states, concatenated,
            B, L_txt, L_img, D,
            encoder_hidden_states.stride(0), encoder_hidden_states.stride(1), encoder_hidden_states.stride(2),
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            concatenated.stride(0), concatenated.stride(1), concatenated.stride(2),
            BLOCK_D=BLOCK_D,
            num_warps=4, num_stages=2,
        )

        # 2) Batched matmul in Triton: C = concatenated @ process_weight.T
        # Ensure contiguous for simpler strides
        A = concatenated.contiguous()  # [B, M, K]
        Wt = process_weight.t().contiguous()  # [K, N] with N=K=D

        C = torch.empty((B, M, D), device=device, dtype=dtype)
        # Choose block sizes; 64 works well generally. Masks handle partials.
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 64

        grid_matmul = (B, triton.cdiv(M, BLOCK_M), triton.cdiv(D, BLOCK_N))
        batched_matmul_kernel[grid_matmul](
            A, Wt, C,
            B, M, D, D,
            A.stride(0), A.stride(1), A.stride(2),
            Wt.stride(0), Wt.stride(1),
            C.stride(0), C.stride(1), C.stride(2),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        # 3) Split outputs using Triton
        processed_encoder = torch.empty((B, L_txt, D), device=device, dtype=dtype)
        processed_hidden = torch.empty((B, L_img, D), device=device, dtype=dtype)

        BLOCK_SPLIT = 128
        grid_split = (B, L_txt, triton.cdiv(D, BLOCK_SPLIT))
        split_seqs_kernel[grid_split](
            C, processed_encoder, processed_hidden,
            B, L_txt, L_img, D,
            C.stride(0), C.stride(1), C.stride(2),
            processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
            processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
            BLOCK_D=BLOCK_SPLIT,
            num_warps=4, num_stages=2,
        )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
