import torch
import triton
import triton.language as tl


@triton.jit
def concat_seqs_kernel(
    ehs_ptr, hs_ptr, dst_ptr,
    B: tl.int32, L_txt: tl.int32, L_img: tl.int32, D: tl.int32,
    ehs_stride_b: tl.int32, ehs_stride_s: tl.int32, ehs_stride_d: tl.int32,
    hs_stride_b: tl.int32, hs_stride_s: tl.int32, hs_stride_d: tl.int32,
    dst_stride_b: tl.int32, dst_stride_s: tl.int32, dst_stride_d: tl.int32,
    BLOCK_S: tl.constexpr, BLOCK_D: tl.constexpr,
):
    # program ids: batch, tile over sequence, tile over dim
    b = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_d = tl.program_id(2)

    s_offsets = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    d_offsets = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)

    # Mask for valid s and d
    mask_s = s_offsets < L_txt
    mask_d = d_offsets < D
    mask_e = mask_s[:, None] & mask_d[None, :]  # [BLOCK_S, BLOCK_D]

    # Compute base offsets for source and destination
    # Destination offset for encoder part
    dst_base_e = (b * dst_stride_b) + (s_offsets[:, None] * dst_stride_s) + (d_offsets[None, :] * dst_stride_d)
    # Source offset for encoder part
    src_base_e = (b * ehs_stride_b) + (s_offsets[:, None] * ehs_stride_s) + (d_offsets[None, :] * ehs_stride_d)

    # Load and store encoder hidden states
    e_vals = tl.load(ehs_ptr + src_base_e, mask=mask_e, other=0.0)
    tl.store(dst_ptr + dst_base_e, e_vals, mask=mask_e)

    # Destination offset for hidden part (shift by L_txt)
    dst_base_h = (b * dst_stride_b) + ((s_offsets[:, None] + L_txt) * dst_stride_s) + (d_offsets[None, :] * dst_stride_d)
    # Source offset for hidden part
    src_base_h = (b * hs_stride_b) + (s_offsets[:, None] * hs_stride_s) + (d_offsets[None, :] * hs_stride_d)

    mask_h = mask_s[:, None] & mask_d[None, :]
    h_vals = tl.load(hs_ptr + src_base_h, mask=mask_h, other=0.0)
    tl.store(dst_ptr + dst_base_h, h_vals, mask=mask_h)


@triton.jit
def gemm_bmn_kernel(
    A_ptr, Wt_ptr, C_ptr,
    B: tl.int32, M: tl.int32, N: tl.int32, K: tl.int32,
    A_stride_b: tl.int32, A_stride_m: tl.int32, A_stride_k: tl.int32,
    Wt_stride_k: tl.int32, Wt_stride_n: tl.int32,
    C_stride_b: tl.int32, C_stride_m: tl.int32, C_stride_n: tl.int32,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Grid over (B, tiles along M, tiles along N)
    b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K in chunks
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)

        # Masks for valid ranges
        mask_m = m_offsets < M
        mask_n = n_offsets < N
        mask_k = k_offsets < K

        # Load A tile: [BLOCK_M, BLOCK_K]
        A_ptrs = (
            A_ptr
            + b * A_stride_b
            + m_offsets[:, None] * A_stride_m
            + k_offsets[None, :] * A_stride_k
        )
        A_mask = (mask_m[:, None] & mask_k[None, :])
        A_tile = tl.load(A_ptrs, mask=A_mask, other=0.0).to(tl.float32)

        # Load Wt tile: [BLOCK_K, BLOCK_N]
        Wt_ptrs = (
            Wt_ptr
            + k_offsets[:, None] * Wt_stride_k
            + n_offsets[None, :] * Wt_stride_n
        )
        Wt_mask = (mask_k[:, None] & mask_n[None, :])
        Wt_tile = tl.load(Wt_ptrs, mask=Wt_mask, other=0.0).to(tl.float32)

        # Accumulate: acc += A_tile @ Wt_tile
        # Triton supports tl.dot for matrix multiply on tiles
        acc += tl.dot(A_tile, Wt_tile)

    # Store result
    C_ptrs = (
        C_ptr
        + b * C_stride_b
        + m_offsets[:, None] * C_stride_m
        + n_offsets[None, :] * C_stride_n
    )
    C_mask = (mask_m[:, None] & mask_n[None, :])
    tl.store(C_ptrs, acc, mask=C_mask)


@triton.jit
def split_rows_kernel(
    src_ptr, out_ptr,
    B: tl.int32, S: tl.int32, D: tl.int32,
    src_stride_b: tl.int32, src_stride_s: tl.int32, src_stride_d: tl.int32,
    out_stride_b: tl.int32, out_stride_s: tl.int32, out_stride_d: tl.int32,
    BLOCK_S: tl.constexpr, BLOCK_D: tl.constexpr,
):
    b = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_d = tl.program_id(2)

    s_offsets = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    d_offsets = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)

    mask_s = s_offsets < S
    mask_d = d_offsets < D
    mask = mask_s[:, None] & mask_d[None, :]

    src_base = (b * src_stride_b) + (s_offsets[:, None] * src_stride_s) + (d_offsets[None, :] * src_stride_d)
    out_base = (b * out_stride_b) + (s_offsets[:, None] * out_stride_s) + (d_offsets[None, :] * out_stride_d)

    vals = tl.load(src_ptr + src_base, mask=mask, other=0.0)
    tl.store(out_ptr + out_base, vals, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Kernel config parameters
        self.BLOCK_S = 128  # for concat/split tiles along S
        self.BLOCK_D = 128  # for tiles along D
        self.BLOCK_M = 64   # for GEMM tiles along M
        self.BLOCK_N = 64   # for GEMM tiles along N
        self.BLOCK_K = 64   # for GEMM K-chunk
        self.num_warps = 4
        self.num_stages = 2

    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Shapes
        B, L_img, D = hidden_states.shape
        _, L_txt, D_ehs = encoder_hidden_states.shape
        assert D == D_ehs, "hidden_dim must match between encoder and image streams"
        assert process_weight.shape[0] == D and process_weight.shape[1] == D, "process_weight must be [D, D]"

        # Ensure contiguous for simple stride math
        ehs = encoder_hidden_states.contiguous()
        hs = hidden_states.contiguous()
        W = process_weight.contiguous()  # [D, D]

        # Allocate destination concatenated [B, L_txt + L_img, D]
        concatenated = torch.empty((B, L_txt + L_img, D), device=hs.device, dtype=hs.dtype)

        # Launch concat kernel: copy ehs into [:, :L_txt, :], and hs into[:, L_txt:, :]
        grid_concat = (
            B,
            triton.cdiv(L_txt, self.BLOCK_S),
            triton.cdiv(D, self.BLOCK_D),
        )
        concat_seqs_kernel[grid_concat](
            ehs, hs, concatenated,
            B, L_txt, L_img, D,
            ehs.stride(0), ehs.stride(1), ehs.stride(2),
            hs.stride(0), hs.stride(1), hs.stride(2),
            concatenated.stride(0), concatenated.stride(1), concatenated.stride(2),
            BLOCK_S=self.BLOCK_S, BLOCK_D=self.BLOCK_D,
            num_warps=self.num_warps, num_stages=self.num_stages
        )

        # Compute A @ W^T where A = concatenated [B, M, K], W^T = W^T [K, N]
        A = concatenated  # [B, M, K], M = L_txt + L_img, K = D
        Wt = W.t().contiguous()  # [K, N], N = D

        M = L_txt + L_img
        N = D
        K = D

        # Output processed [B, M, N]
        processed = torch.empty((B, M, N), device=A.device, dtype=A.dtype)

        # Launch GEMM kernel
        grid_gemm = (
            B,
            triton.cdiv(M, self.BLOCK_M),
            triton.cdiv(N, self.BLOCK_N),
        )
        gemm_bmn_kernel[grid_gemm](
            A, Wt, processed,
            B, M, N, K,
            A.stride(0), A.stride(1), A.stride(2),
            Wt.stride(0), Wt.stride(1),
            processed.stride(0), processed.stride(1), processed.stride(2),
            BLOCK_M=self.BLOCK_M, BLOCK_N=self.BLOCK_N, BLOCK_K=self.BLOCK_K,
            num_warps=self.num_warps, num_stages=self.num_stages
        )

        # Allocate outputs
        processed_encoder = torch.empty((B, L_txt, D), device=processed.device, dtype=processed.dtype)
        processed_hidden = torch.empty((B, L_img, D), device=processed.device, dtype=processed.dtype)

        # Launch split kernels
        grid_split0 = (B, triton.cdiv(L_txt, self.BLOCK_S), triton.cdiv(D, self.BLOCK_D))
        split_rows_kernel[grid_split0](
            processed, processed_encoder,
            B, L_txt, D,
            processed.stride(0), processed.stride(1), processed.stride(2),
            processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
            BLOCK_S=self.BLOCK_S, BLOCK_D=self.BLOCK_D,
            num_warps=self.num_warps, num_stages=self.num_stages
        )

        grid_split1 = (B, triton.cdiv(L_img, self.BLOCK_S), triton.cdiv(D, self.BLOCK_D))
        split_rows_kernel[grid_split1](
            processed, processed_hidden,
            B, L_img, D,
            processed.stride(0), processed.stride(1), processed.stride(2),
            processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
            BLOCK_S=self.BLOCK_S, BLOCK_D=self.BLOCK_D,
            num_warps=self.num_warps, num_stages=self.num_stages
        )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
