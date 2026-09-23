import torch
import triton
import triton.language as tl


@triton.jit
def concat_seqs_kernel(
    ehs_ptr,     # *encoder_hidden_states [B, L_txt, D]
    hs_ptr,      # *hidden_states [B, L_img, D]
    out_ptr,     # *output concatenated [B, L_txt + L_img, D]
    B: tl.int32,
    L_txt: tl.int32,
    L_img: tl.int32,
    D: tl.int32,
    ehs_stride_b: tl.int32, ehs_stride_s: tl.int32, ehs_stride_d: tl.int32,
    hs_stride_b: tl.int32, hs_stride_s: tl.int32, hs_stride_d: tl.int32,
    out_stride_b: tl.int32, out_stride_s: tl.int32, out_stride_d: tl.int32,
    BLOCK_D: tl.constexpr,
):
    # Grid: (B, L_txt + L_img, ceil(D / BLOCK_D))
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_d = tl.program_id(2)

    d_offsets = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = d_offsets < D

    ehs_b_ptr = ehs_ptr + pid_b * ehs_stride_b
    hs_b_ptr = hs_ptr + pid_b * hs_stride_b
    out_b_ptr = out_ptr + pid_b * out_stride_b

    # Copy encoder part: s in [0, L_txt)
    vals_e = tl.load(
        ehs_b_ptr + pid_s * ehs_stride_s + d_offsets * ehs_stride_d,
        mask=mask_d & (pid_s < L_txt),
        other=0.0
    )
    tl.store(
        out_b_ptr + pid_s * out_stride_s + d_offsets * out_stride_d,
        vals_e,
        mask=mask_d & (pid_s < L_txt)
    )

    # Copy hidden part: s in [L_txt, L_txt + L_img)
    vals_h = tl.load(
        hs_b_ptr + (pid_s - L_txt) * hs_stride_s + d_offsets * hs_stride_d,
        mask=mask_d & (pid_s >= L_txt) & (pid_s < (L_txt + L_img)),
        other=0.0
    )
    tl.store(
        out_b_ptr + pid_s * out_stride_s + d_offsets * out_stride_d,
        vals_h,
        mask=mask_d & (pid_s >= L_txt) & (pid_s < (L_txt + L_img))
    )


@triton.jit
def batched_matmul_kernel(
    a_ptr,      # *A [B, M, K], where A is concatenated input
    w_ptr,      # *W [K, N], where W = process_weight.T
    c_ptr,      # *C [B, M, N] output
    B: tl.int32, M: tl.int32, N: tl.int32, K: tl.int32,
    a_stride_b: tl.int32, a_stride_m: tl.int32, a_stride_k: tl.int32,
    w_stride_k: tl.int32, w_stride_n: tl.int32,
    c_stride_b: tl.int32, c_stride_m: tl.int32, c_stride_n: tl.int32,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Grid: (B, tiles along M, tiles along N)
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K in blocks
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)

        # A tile: [BLOCK_M, BLOCK_K]
        a_ptrs = a_ptr + pid_b * a_stride_b + m_offsets[:, None] * a_stride_m + k_offsets[None, :] * a_stride_k
        a_mask = (m_offsets[:, None] < M) & (k_offsets[None, :] < K)
        a_tile = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # W tile: [BLOCK_K, BLOCK_N]
        w_ptrs = w_ptr + k_offsets[:, None] * w_stride_k + n_offsets[None, :] * w_stride_n
        w_mask = (k_offsets[:, None] < K) & (n_offsets[None, :] < N)
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)

        # Accumulate
        acc += tl.dot(a_tile.to(tl.float32), w_tile.to(tl.float32))

    # Write back C tile
    c_ptrs = c_ptr + pid_b * c_stride_b + m_offsets[:, None] * c_stride_m + n_offsets[None, :] * c_stride_n
    c_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    # Cast back to output dtype (assumed same as input dtype)
    # Triton doesn't carry dtype info from pointer; store float32 as float32. If inputs are fp16/bf16,
    # we can cast acc to that dtype before store. We assume float32 here for robustness.
    tl.store(c_ptrs, acc, mask=c_mask)


@triton.jit
def split_seqs_kernel(
    c_ptr,        # *C [B, M, D] processed
    out_e_ptr,    # *processed_encoder [B, L_txt, D]
    out_h_ptr,    # *processed_hidden [B, L_img, D]
    B: tl.int32, M: tl.int32, N: tl.int32,  # here N==D, M==L_txt+L_img
    c_stride_b: tl.int32, c_stride_m: tl.int32, c_stride_n: tl.int32,
    out_e_stride_b: tl.int32, out_e_stride_s: tl.int32, out_e_stride_d: tl.int32,
    out_h_stride_b: tl.int32, out_h_stride_s: tl.int32, out_h_stride_d: tl.int32,
    L_txt: tl.int32, L_img: tl.int32,
    BLOCK_N: tl.constexpr,  # BLOCK_N along D
):
    # Grid for encoder: (B, L_txt, ceil(D / BLOCK_N))
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_d = tl.program_id(2)

    d_offsets = pid_d * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_d = d_offsets < N

    c_b_ptr = c_ptr + pid_b * c_stride_b
    out_e_b_ptr = out_e_ptr + pid_b * out_e_stride_b
    out_h_b_ptr = out_h_ptr + pid_b * out_h_stride_b

    # Copy encoder part: rows 0..L_txt-1
    vals_e = tl.load(
        c_b_ptr + pid_s * c_stride_m + d_offsets * c_stride_n,
        mask=mask_d & (pid_s < L_txt),
        other=0.0
    )
    tl.store(
        out_e_b_ptr + pid_s * out_e_stride_s + d_offsets * out_e_stride_d,
        vals_e,
        mask=mask_d & (pid_s < L_txt)
    )

    # Copy hidden part: rows L_txt..L_txt+L_img-1
    vals_h = tl.load(
        c_b_ptr + (pid_s + L_txt) * c_stride_m + d_offsets * c_stride_n,
        mask=mask_d & (pid_s < L_img),
        other=0.0
    )
    tl.store(
        out_h_b_ptr + pid_s * out_h_stride_s + d_offsets * out_h_stride_d,
        vals_h,
        mask=mask_d & (pid_s < L_img)
    )


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        Triton-only implementation:
        - Concatenate along sequence dim with Triton copy kernel.
        - Compute matmul with Triton batched GEMM kernel.
        - Split outputs with Triton copy kernel.
        """
        # Shapes
        B = hidden_states.shape[0]
        L_img = hidden_states.shape[1]
        L_txt = encoder_hidden_states.shape[1]
        D = hidden_states.shape[2]
        assert D == encoder_hidden_states.shape[2], "hidden_dim must match across inputs"
        assert process_weight.shape[0] == D and process_weight.shape[1] == D, "process_weight must be [D, D]"

        # Ensure contiguous for predictable strides
        ehs = encoder_hidden_states.contiguous()
        hs = hidden_states.contiguous()
        w = process_weight.contiguous()  # [D, D]

        # 1) Concatenate into A [B, M, K] where M = L_txt + L_img, K = D
        M = L_txt + L_img
        A = torch.empty((B, M, D), dtype=ehs.dtype, device=ehs.device)

        # Launch concat kernel
        BLOCK_D = 128  # tile along D dimension
        grid_concat = (B, M, triton.cdiv(D, BLOCK_D))
        concat_seqs_kernel[grid_concat](
            ehs, hs, A,
            B, L_txt, L_img, D,
            ehs.stride(0), ehs.stride(1), ehs.stride(2),
            hs.stride(0), hs.stride(1), hs.stride(2),
            A.stride(0), A.stride(1), A.stride(2),
            BLOCK_D=BLOCK_D,
            num_warps=4, num_stages=2,
        )

        # 2) Compute C = A @ W^T using Triton GEMM, W^T is [K, N] with N=D
        # We pass W directly; Triton kernel interprets it as [K, N].
        C = torch.empty((B, M, D), dtype=A.dtype, device=A.device)

        # Choose block sizes; 64x64x64 is a good baseline
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 64
        grid_gemm = (B, triton.cdiv(M, BLOCK_M), triton.cdiv(D, BLOCK_N))
        batched_matmul_kernel[grid_gemm](
            A, w, C,
            B, M, D, D,
            A.stride(0), A.stride(1), A.stride(2),
            w.stride(0), w.stride(1),
            C.stride(0), C.stride(1), C.stride(2),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        # 3) Split C back into encoder and hidden parts
        processed_encoder = torch.empty((B, L_txt, D), dtype=C.dtype, device=C.device)
        processed_hidden = torch.empty((B, L_img, D), dtype=C.dtype, device=C.device)

        # Launch split kernel
        BLOCK_SPLIT = 128  # tile along D for split
        grid_split = (B, L_txt, triton.cdiv(D, BLOCK_SPLIT)), (B, L_img, triton.cdiv(D, BLOCK_SPLIT))
        # We need two launches: one for encoder, one for hidden
        split_seqs_kernel[grid_split[0]](
            C, processed_encoder, processed_hidden,
            B, M, D,
            C.stride(0), C.stride(1), C.stride(2),
            processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
            processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
            L_txt, L_img,
            BLOCK_N=BLOCK_SPLIT,
            num_warps=4, num_stages=2,
        )
        split_seqs_kernel[grid_split[1]](
            C, processed_encoder, processed_hidden,
            B, M, D,
            C.stride(0), C.stride(1), C.stride(2),
            processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
            processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
            L_txt, L_img,
            BLOCK_N=BLOCK_SPLIT,
            num_warps=4, num_stages=2,
        )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
