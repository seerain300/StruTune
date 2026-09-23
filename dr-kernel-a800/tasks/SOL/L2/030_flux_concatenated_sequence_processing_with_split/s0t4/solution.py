import torch
import triton
import triton.language as tl


@triton.jit
def concat_seqs_kernel(
    src1_ptr,  # *ptr to [B, L1, D]
    src2_ptr,  # *ptr to [B, L2, D]
    dst_ptr,   # *ptr to [B, L1+L2, D]
    B: tl.int32,
    L1: tl.int32,
    L2: tl.int32,
    D: tl.int32,
    stride_src1b: tl.int32, stride_src1l: tl.int32, stride_src1d: tl.int32,
    stride_src2b: tl.int32, stride_src2l: tl.int32, stride_src2d: tl.int32,
    stride_dstb: tl.int32, stride_dstl: tl.int32, stride_dstd: tl.int32,
    BLOCK_L: tl.constexpr, BLOCK_D: tl.constexpr,
):
    # Grid: (B, ceil((L1+L2)/BLOCK_L), ceil(D/BLOCK_D))
    pid_b = tl.program_id(0)
    pid_l = tl.program_id(1)
    pid_d = tl.program_id(2)

    l_offsets = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
    d_offsets = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)

    total_seq = L1 + L2
    # Mask for dst bounds
    mask_dst = (l_offsets[:, None] < total_seq) & (d_offsets[None, :] < D)

    # For each l in l_offsets, decide source: src1 if l<L1 else src2
    # Build source pointers accordingly
    # Note: Triton supports elementwise if over tensor-like index
    l_vec = l_offsets[:, None]  # shape [BLOCK_L, 1]
    use_src1 = l_vec < L1
    # Compute src pointers for each element in the tile:
    # src1_ptr + pid_b*stride_src1b + l*stride_src1l + d*stride_src1d
    # src2_ptr + pid_b*stride_src2b + (l - L1)*stride_src2l + d*stride_src2d
    # We'll select elementwise using mask-like logic in Triton via tl.where.
    src_l1_ptrs = src1_ptr + pid_b * stride_src1b + l_vec * stride_src1l + d_offsets[None, :] * stride_src1d
    src_l2_ptrs = src2_ptr + pid_b * stride_src2b + (l_vec - L1) * stride_src2l + d_offsets[None, :] * stride_src2d

    # Select source data based on whether l < L1
    # Triton will broadcast pointers and masks; load from src1 where use_src1, else 0
    # We'll do two loads and combine; but simpler is to construct a pointer for each element and load with mask
    # Here, we'll use masked load: each element is valid if mask_dst and use_src1 (or use_src2 implicitly via mask_dst)
    # To avoid overlap issues, perform two masked loads and then select.
    # However, Triton does not support elementwise selection of pointer tensors; we need to load with a combined mask.
    # Since we don't know which element comes from src1/src2, we can't select. Instead, we compute per-element pointers
    # by using the fact that l_vec is a [BLOCK_L, 1] and do per-element masked loads by using masks and tl.load with other=0.

    # For masked loads, we need to create a per-element mask for src1 and src2
    mask_src1 = mask_dst & use_src1
    mask_src2 = mask_dst & (~use_src1)

    # Initialize output tile
    # We need to write to dst_ptr + pid_b*stride_dstb + l*stride_dstl + d*stride_dstd
    dst_ptrs = dst_ptr + pid_b * stride_dstb + l_vec * stride_dstl + d_offsets[None, :] * stride_dstd

    # Load from src1 for elements where use_src1, else 0
    val1 = tl.load(src_l1_ptrs, mask=mask_src1, other=0.0)
    # Load from src2 for elements where use_src2, else 0
    val2 = tl.load(src_l2_ptrs, mask=mask_src2, other=0.0)
    # Sum gives the correct value: elements not in src1 are not in src2 (due to mask_src2), and vice versa.
    # However, we must ensure no double-write. Since mask_src1 and mask_src2 are disjoint, val1 and val2 are zeros where the other source is invalid.
    # Sum them; where both are zero, dst should be zero. We still need to store with mask_dst.
    out_tile = val1 + val2

    # Store to destination
    tl.store(dst_ptrs, out_tile, mask=mask_dst)


@triton.jit
def copy_block_kernel(
    A_ptr, B_ptr,
    B_size, M, N, D,
    stride_Ab, stride_Am, stride_Ad,
    stride_Bb, stride_Bm, stride_Bd,
    pid_b, pid_m, pid_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    # A: [B_size, M, N], B: [B_size, M, N]
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    d_offsets = tl.arange(0, BLOCK_D)  # we can copy along feature dimension too; here N is last dim, D is feature dim

    # Masks
    mask_m = m_offsets < M
    mask_n = n_offsets < N
    # A is 3D (B, M, N), but we pass D as N for feature dimension? We need to clarify: A is [B, M, N]; D is feature dim if applicable.
    # In our use, we will pass A as [B, M, N] and set D=N. To be generic, we treat D as the 'N' dimension we copy.
    # So we copy from A[:, m, n] to B[:, m, n] for b in [pid_b].
    # We launch grid over (B, M tiles, N tiles), so pid_b selects batch, pid_m selects M tile, pid_n selects N tile.

    # Compute pointers for A and B for the tile
    A_ptrs = A_ptr + pid_b * stride_Ab + m_offsets[:, None] * stride_Am + n_offsets[None, :] * stride_Ad
    B_ptrs = B_ptr + pid_b * stride_Bb + m_offsets[:, None] * stride_Bm + n_offsets[None, :] * stride_Bd

    # Combined mask
    mask = mask_m[:, None] & mask_n[None, :]

    # Load and store
    tile = tl.load(A_ptrs, mask=mask, other=0.0)
    tl.store(B_ptrs, tile, mask=mask)


@triton.jit
def batched_matmul_kernel(
    A_ptr,  # *ptr to A [B, M, K]
    W_ptr,  # *ptr to W [K, N]
    C_ptr,  # *ptr to C [B, M, N]
    B_size: tl.int32,
    M: tl.int32,
    N: tl.int32,
    K: tl.int32,
    stride_Ab: tl.int32, stride_Am: tl.int32, stride_Ak: tl.int32,
    stride_Wk: tl.int32, stride_Wn: tl.int32,
    stride_Cb: tl.int32, stride_Cm: tl.int32, stride_Cn: tl.int32,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Grid: (B_size, ceil(M/BLOCK_M), ceil(N/BLOCK_N))
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)

        # Load A tile [BLOCK_M, BLOCK_K]
        A_ptrs = A_ptr + pid_b * stride_Ab + m_offsets[:, None] * stride_Am + k_offsets[None, :] * stride_Ak
        A_mask = (m_offsets[:, None] < M) & (k_offsets[None, :] < K)
        A_tile = tl.load(A_ptrs, mask=A_mask, other=0.0)

        # Load W^T tile [BLOCK_K, BLOCK_N] (W is [K, N], we index W[k, n])
        W_ptrs = W_ptr + k_offsets[:, None] * stride_Wk + n_offsets[None, :] * stride_Wn
        W_mask = (k_offsets[:, None] < K) & (n_offsets[None, :] < N)
        Wt_tile = tl.load(W_ptrs, mask=W_mask, other=0.0)

        # Accumulate
        acc += tl.dot(A_tile, Wt_tile)

    # Store result
    C_ptrs = C_ptr + pid_b * stride_Cb + m_offsets[:, None] * stride_Cm + n_offsets[None, :] * stride_Cn
    C_mask = (m_offsets[:, None] < M) & (n_offsets[None] < N)
    tl.store(C_ptrs, acc, mask=C_mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-only forward:
        - Concatenate sequences along sequence dimension in Triton
        - Compute batched matmul in Triton: processed = concatenated @ process_weight.T
        - Split into two outputs via Triton copy kernels
        Returns: (processed_encoder [B, L_txt, D], processed_hidden [B, L_img, D])
        """
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "All inputs must be CUDA tensors for Triton."
        B = hidden_states.shape[0]
        L_txt = encoder_hidden_states.shape[1]
        L_img = hidden_states.shape[1]
        D = hidden_states.shape[2]
        # Ensure contiguous for predictable strides
        ehs = encoder_hidden_states.contiguous()
        hs = hidden_states.contiguous()
        pw_T = process_weight.t().contiguous()  # [D, D], process_weight.T

        # 1) Concatenate in Triton: dst [B, L_txt + L_img, D]
        total_seq = L_txt + L_img
        dst_concat = torch.empty((B, total_seq, D), device=hs.device, dtype=hs.dtype)

        # Launch concat kernel
        BLOCK_L = 64
        BLOCK_D = 64
        grid = (B, triton.cdiv(total_seq, BLOCK_L), triton.cdiv(D, BLOCK_D))
        concat_seqs_kernel[grid](
            ehs, hs, dst_concat,
            B, L_txt, L_img, D,
            ehs.stride(0), ehs.stride(1), ehs.stride(2),
            hs.stride(0), hs.stride(1), hs.stride(2),
            dst_concat.stride(0), dst_concat.stride(1), dst_concat.stride(2),
            BLOCK_L=BLOCK_L, BLOCK_D=BLOCK_D,
        )

        # 2) Batched matmul in Triton: C[b, M, N] = A[b, M, K] @ W[K, N], A = dst_concat, W = pw_T
        M = total_seq
        K = D
        N = D
        C = torch.empty((B, M, N), device=hs.device, dtype=hs.dtype)

        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 32
        grid_matmul = (B, triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        batched_matmul_kernel[grid_matmul](
            dst_concat, pw_T, C,
            B, M, N, K,
            dst_concat.stride(0), dst_concat.stride(1), dst_concat.stride(2),
            pw_T.stride(0), pw_T.stride(1),
            C.stride(0), C.stride(1), C.stride(2),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        )

        # 3) Triton copies for split: processed_encoder = C[:, :L_txt, :], processed_hidden = C[:, L_txt:, :]
        # We launch two copy_block_kernel instances:
        # Note: copy_block_kernel is designed to copy a [M, N] tile per batch. Our C is [B, M, N]. We'll treat N as feature dim and set M = seq length; but our kernel expects 3D and we pass M=N=feature_dim? To keep simple, we'll implement per-batch copying via two kernels that write the outputs directly.

        # Allocate outputs
        processed_encoder = torch.empty((B, L_txt, D), device=hs.device, dtype=hs.dtype)
        processed_hidden = torch.empty((B, L_img, D), device=hs.device, dtype=hs.dtype)

        # Copy first L_txt sequences
        grid_part1 = (B, triton.cdiv(L_txt, BLOCK_M), triton.cdiv(D, BLOCK_D))
        copy_block_kernel[grid_part1](
            C, processed_encoder,
            B, M, D, D,  # we use M=L_txt and N=D for this copy
            C.stride(0), C.stride(1), C.stride(2),
            processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
            0, 0, 0,  # pid_b, pid_m, pid_n – grid handles batching
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_D, BLOCK_D=BLOCK_D,
        )

        # Copy remaining L_img sequences (offset L_txt)
        grid_part2 = (B, triton.cdiv(L_img, BLOCK_M), triton.cdiv(D, BLOCK_D))
        copy_block_kernel[grid_part2](
            C, processed_hidden,
            B, L_img, D, D,  # M = L_img, N = D (feature dim)
            C.stride(0), C.stride(1), C.stride(2),
            processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
            0, 0, 0,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_D, BLOCK_D=BLOCK_D,
        )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
