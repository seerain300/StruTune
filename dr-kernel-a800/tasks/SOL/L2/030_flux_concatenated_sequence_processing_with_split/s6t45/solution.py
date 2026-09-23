import torch
import triton
import triton.language as tl


@triton.jit
def _concat_seq_kernel(
    A_ptr,         # *f32, [B, M, H]
    B_ptr,         # *f32, [B, N, H]
    Out_ptr,       # *f32, [B, C, H], C = M + N
    B: tl.constexpr,   # batch size
    M: tl.constexpr,   # text_seq_len
    N: tl.constexpr,   # img_seq_len
    H: tl.constexpr,   # hidden_dim
    stride_ab,     # int: stride along batch for A
    stride_am,     # int: stride along seq for A
    stride_ah,     # int: stride along hidden for A
    stride_bb,     # int: stride along batch for B
    stride_bn,     # int: stride along seq for B
    stride_bh,     # int: stride along hidden for B
    stride_ob,     # int: stride along batch for Out
    stride_oc,     # int: stride along seq for Out
    stride_oh,     # int: stride along hidden for Out
    BLOCK_M: tl.constexpr,  # tile over M (rows)
    BLOCK_N: tl.constexpr,  # tile over N (rows)
):
    # 2D grid: (B, blocks over C = M + N)
    b = tl.program_id(0)
    block = tl.program_id(1)

    # Each program handles a tile of the sequence dimension
    m_block = block * BLOCK_M
    n_block = block * BLOCK_N

    m_offsets = m_block + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    n_offsets = n_block + tl.arange(0, BLOCK_N)  # [BLOCK_N]

    mask_m = m_offsets < M
    mask_n = n_offsets < N

    # For each hidden dimension h, copy BLOCK_M rows from A and BLOCK_N rows from B into Out.
    for h in range(0, H):
        # Copy A rows to Out[b, m, h] for m in [0..M)
        out_ptrs_a = Out_ptr + b * stride_ob + m_offsets * stride_oc + h * stride_oh
        a_ptrs = A_ptr + b * stride_ab + m_offsets * stride_am + h * stride_ah
        tl.store(out_ptrs_a, tl.load(a_ptrs, mask=mask_m, other=0.0))

        # Copy B rows to Out[b, M + n, h] for n in [0..N)
        out_ptrs_b = Out_ptr + b * stride_ob + (M + n_offsets) * stride_oc + h * stride_oh
        b_ptrs = B_ptr + b * stride_bb + n_offsets * stride_bn + h * stride_bh
        tl.store(out_ptrs_b, tl.load(b_ptrs, mask=mask_n, other=0.0))


@triton.jit
def _batched_matmul_kernel_explicit(
    X_ptr,  # *f32, [B, C, K]
    WT_ptr, # *f32, [K, K] (process_weight.T)
    P_ptr,  # *f32, [B, C, K]
    B: tl.constexpr,      # batch size
    C: tl.constexpr,      # sequence length (M + N)
    K: tl.constexpr,      # hidden dim (constexpr)
    stride_xb,  # int: stride along batch for X
    stride_xc,  # int: stride along seq for X
    stride_xk,  # int: stride along hidden for X
    stride_wk,  # int: stride along rows for WT (dim 0, i.e., K in W)
    stride_wh,  # int: stride along cols for WT (dim 1, i.e., K in W)
    stride_pb,  # int: stride along batch for P
    stride_pc,  # int: stride along seq for P
    stride_pk,  # int: stride along hidden for P
    BLOCK_M: tl.constexpr,  # tile over C
    BLOCK_N: tl.constexpr,  # tile over K (output hidden)
    BLOCK_K: tl.constexpr,  # tile over reduction K
):
    # 3D grid: (B, tiles over C, tiles over K)
    b = tl.program_id(0)
    tile_m = tl.program_id(1)
    tile_n = tl.program_id(2)

    m_offsets = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    n_offsets = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]

    mask_m = m_offsets < C
    mask_n = n_offsets < K

    # Initialize accumulator for this tile
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over reduction dimension K in chunks of BLOCK_K
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)  # [BLOCK_K]
        mask_k = k_offsets < K

        # Load A tile: X[b, m, k_offsets] -> shape [BLOCK_M, BLOCK_K]
        a_ptrs = X_ptr + b * stride_xb + m_offsets[:, None] * stride_xc + k_offsets[None, :] * stride_xk
        a_mask = mask_m[:, None] & mask_k[None, :]
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)  # [BLOCK_M, BLOCK_K]

        # Load B tile: WT[k_offsets, n_offsets] -> shape [BLOCK_K, BLOCK_N]
        # WT is [K, K], WT[i, j] = W[j, i], so row i comes from original W's column i.
        b_ptrs = WT_ptr + k_offsets[:, None] * stride_wk + n_offsets[None, :] * stride_wh
        b_mask = mask_k[:, None] & mask_n[None, :]
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)  # [BLOCK_K, BLOCK_N]

        # Accumulate
        acc += tl.dot(a, b)  # [BLOCK_M, BLOCK_N]

    # Store results: P[b, m, n_offsets]
    p_ptrs = P_ptr + b * stride_pb + m_offsets[:, None] * stride_pc + n_offsets[None, :] * stride_pk
    p_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(p_ptrs, acc, mask=p_mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                encoder_hidden_states: torch.Tensor,
                process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version of the original run:
        - Concatenate encoder_hidden_states and hidden_states along sequence dimension in Triton.
        - Apply linear projection (X @ process_weight.T) in Triton GEMM.
        - Split outputs back into encoder and image streams.
        """
        # Ensure CUDA tensors
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Inputs must be CUDA tensors"
        B = hidden_states.shape[0]
        M = encoder_hidden_states.shape[1]
        N = hidden_states.shape[1]
        H = hidden_states.shape[2]
        assert encoder_hidden_states.shape == (B, M, H), "encoder_hidden_states must be [batch, text_seq_len, hidden_dim]"
        assert hidden_states.shape == (B, N, H), "hidden_states must be [batch, img_seq_len, hidden_dim]"
        assert process_weight.shape == (H, H), "process_weight must be [hidden_dim, hidden_dim]"

        # Ensure dtype is float32 and contiguous
        A = encoder_hidden_states.contiguous().to(torch.float32)
        Bt = hidden_states.contiguous().to(torch.float32)
        W = process_weight.contiguous().to(torch.float32)  # [H, H]
        W_T = W.transpose(0, 1).contiguous()  # [H, H], WT[k, n] = W[n, k]

        # 1) Concatenate along sequence dimension: Out [B, C, H], C = M + N
        C = M + N
        Out = torch.empty((B, C, H), dtype=torch.float32, device=hidden_states.device)

        # Launch concatenation kernel
        BLOCK_M = 64
        BLOCK_N = 64
        grid_concat = (B, triton.cdiv(C, BLOCK_M))
        _concat_seq_kernel[grid_concat](
            A, Bt, Out,
            B=B, M=M, N=N, H=H,
            stride_ab=A.stride(0), stride_am=A.stride(1), stride_ah=A.stride(2),
            stride_bb=Bt.stride(0), stride_bn=Bt.stride(1), stride_bh=Bt.stride(2),
            stride_ob=Out.stride(0), stride_oc=Out.stride(1), stride_oh=Out.stride(2),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            num_warps=4, num_stages=2,
        )

        # 2) Batched GEMM: X @ W_T -> P [B, C, H]
        P = torch.empty((B, C, H), dtype=torch.float32, device=hidden_states.device)

        # Strides for X (Out), W_T, and P
        stride_xb, stride_xc, stride_xk = Out.stride(0), Out.stride(1), Out.stride(2)
        stride_wk, stride_wh = W_T.stride(0), W_T.stride(1)
        stride_pb, stride_pc, stride_pk = P.stride(0), P.stride(1), P.stride(2)

        # Tiling parameters (robust defaults; can be tuned for performance)
        BLOCK_M_gemm = 64
        BLOCK_N_gemm = 64
        BLOCK_K_gemm = 64

        grid_gemm = (B, triton.cdiv(C, BLOCK_M_gemm), triton.cdiv(H, BLOCK_N_gemm))
        _batched_matmul_kernel_explicit[grid_gemm](
            Out, W_T, P,
            B=B, C=C, K=H,  # specialize for current hidden size
            stride_xb=stride_xb, stride_xc=stride_xc, stride_xk=stride_xk,
            stride_wk=stride_wk, stride_wh=stride_wh,
            stride_pb=stride_pb, stride_pc=stride_pc, stride_pk=stride_pk,
            BLOCK_M=BLOCK_M_gemm, BLOCK_N=BLOCK_N_gemm, BLOCK_K=BLOCK_K_gemm,
            num_warps=4, num_stages=3,
        )

        # 3) Split outputs along sequence dimension
        processed_encoder = P[:, :M, :]
        processed_hidden = P[:, M:, :]

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
