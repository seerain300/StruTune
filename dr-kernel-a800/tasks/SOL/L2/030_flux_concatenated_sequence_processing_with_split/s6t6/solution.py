import torch
import triton
import triton.language as tl


@triton.jit
def _concat_seq_kernel(
    A_ptr,        # *f32, [B, M, H]
    B_ptr,        # *f32, [B, N, H]
    Out_ptr,      # *f32, [B, C, H], C = M + N
    B: tl.constexpr,    # batch size
    M: tl.constexpr,    # text_seq_len
    N: tl.constexpr,    # img_seq_len
    H: tl.constexpr,    # hidden_dim
    stride_ab,    # int: stride along batch for A
    stride_am,    # int: stride along seq for A
    stride_ah,    # int: stride along hidden for A
    stride_bb,    # int: stride along batch for B
    stride_bn,    # int: stride along seq for B
    stride_bh,    # int: stride along hidden for B
    stride_ob,    # int: stride along batch for Out
    stride_oc,    # int: stride along seq for Out
    stride_oh,    # int: stride along hidden for Out
    BLOCK_M: tl.constexpr,  # tile along seq (M+N)
    BLOCK_H: tl.constexpr,  # tile along hidden
):
    # Grid: (B, ceil_div(C, BLOCK_M), ceil_div(H, BLOCK_H))
    b = tl.program_id(0)
    m_block = tl.program_id(1)
    h_block = tl.program_id(2)

    # Sequence offsets for this tile
    m_offsets = m_block * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M], where C = M + N
    h_offsets = h_block * BLOCK_H + tl.arange(0, BLOCK_H)  # [BLOCK_H]

    # Masks
    mask_m = m_offsets < (M + N)
    mask_h = h_offsets < H

    # Determine source (A or B) for each sequence position
    from_A = m_offsets < M  # boolean per m

    # Compute input pointers and masks
    # A[b, m, h]
    a_ptrs = A_ptr + b * stride_ab + m_offsets[:, None] * stride_am + h_offsets[None, :] * stride_ah
    a_mask = mask_m[:, None] & mask_h[None, :]
    a_vals = tl.load(a_ptrs, mask=a_mask, other=0.0)

    # B[b, m-M, h]
    b_ptrs = B_ptr + b * stride_bb + (m_offsets - M)[:, None] * stride_bn + h_offsets[None, :] * stride_bh
    b_mask = (~from_A)[:, None] & mask_m[:, None] & mask_h[None, :]
    b_vals = tl.load(b_ptrs, mask=b_mask, other=0.0)

    # Select based on from_A
    vals = tl.where(from_A[:, None], a_vals, b_vals)  # [BLOCK_M, BLOCK_H]

    # Store to Out
    out_ptrs = Out_ptr + b * stride_ob + m_offsets[:, None] * stride_oc + h_offsets[None, :] * stride_oh
    tl.store(out_ptrs, vals, mask=(mask_m[:, None] & mask_h[None, :]))


@triton.jit
def _batched_gemm_kernel_blocked(
    X_ptr,  # *f32, [B, C, K] where C = M + N
    W_ptr,  # *f32, [K, K] (process_weight, we will use W^T in loads)
    P_ptr,  # *f32, [B, C, K]
    B: tl.constexpr,           # batch size (specialization)
    C: tl.constexpr,           # sequence length (M + N)
    K: tl.constexpr,           # hidden dim (specialization)
    stride_xb,  # int: stride along batch in X
    stride_xc,  # int: stride along seq in X
    stride_xk,  # int: stride along hidden in X
    stride_w0,  # int: stride along dim 0 of W (rows for W^T) -> W[n, k], so row stride = W.stride(1), col stride = W.stride(0)
    stride_w1,  # int: stride along dim 1 of W (cols for W^T) -> W[n, k], so col stride = W.stride(0), row stride = W.stride(1)
    stride_pb,  # int: stride along batch in P
    stride_pc,  # int: stride along seq in P
    stride_pk,  # int: stride along hidden in P
    BLOCK_M: tl.constexpr,  # tile over seq (C)
    BLOCK_N: tl.constexpr,  # tile over hidden (K) in output
    BLOCK_K: tl.constexpr,  # tile over reduction dim (K)
):
    # 3D grid: (B, ceil_div(C, BLOCK_M), ceil_div(K, BLOCK_N))
    b = tl.program_id(0)
    m_block = tl.program_id(1)
    n_block = tl.program_id(2)

    # Tile offsets
    m_offsets = m_block * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    n_offsets = n_block * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]

    # Masks for boundaries
    mask_m = m_offsets < C
    mask_n = n_offsets < K

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over reduction dimension K in chunks of BLOCK_K
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)  # [BLOCK_K]
        mask_k = k_offsets < K

        # Load X tile: [BLOCK_M, BLOCK_K]
        x_ptrs = X_ptr + b * stride_xb + m_offsets[:, None] * stride_xc + k_offsets[None, :] * stride_xk
        x_mask = mask_m[:, None] & mask_k[None, :]
        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)

        # Load W^T tile: [BLOCK_K, BLOCK_N]
        # W is [K, K], W^T[k, n] = W[n, k]. Pointer: W_ptr + n * stride_w0 + k * stride_w1.
        w_ptrs = W_ptr + n_offsets[None, :] * stride_w0 + k_offsets[:, None] * stride_w1
        w_mask = mask_n[None, :] & mask_k[:, None]
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)

        # Accumulate
        acc += tl.dot(x_tile, w_tile)

    # Store result to P
    p_ptrs = P_ptr + b * stride_pb + m_offsets[:, None] * stride_pc + n_offsets[None, :] * stride_pk
    tl.store(p_ptrs, acc, mask=(mask_m[:, None] & mask_n[None, :]))


class ModelNew(torch.nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,   # [B, img_seq_len, hidden_dim]
        encoder_hidden_states: torch.Tensor,  # [B, text_seq_len, hidden_dim]
        process_weight: torch.Tensor,         # [hidden_dim, hidden_dim]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Ensure CUDA and dtype float32
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Inputs must be on CUDA"
        assert hidden_states.dtype == torch.float32 and encoder_hidden_states.dtype == torch.float32 and process_weight.dtype == torch.float32, "Use float32"

        B = hidden_states.shape[0]
        M = encoder_hidden_states.shape[1]
        N = hidden_states.shape[1]
        H = hidden_states.shape[2]
        C = M + N

        # Make inputs contiguous
        A = encoder_hidden_states.contiguous()   # [B, M, H]
        X = hidden_states.contiguous()           # [B, N, H]
        W = process_weight.contiguous()          # [H, H]

        # Allocate concatenated tensor Out [B, C, H]
        Out = torch.empty((B, C, H), dtype=torch.float32, device=hidden_states.device)

        # Launch concatenation kernel
        BLOCK_M = 64
        BLOCK_H = 64
        grid_concat = (B, triton.cdiv(C, BLOCK_M), triton.cdiv(H, BLOCK_H))
        _concat_seq_kernel[grid_concat](
            A, X, Out,
            B=B, M=M, N=N, H=H,
            stride_ab=A.stride(0), stride_am=A.stride(1), stride_ah=A.stride(2),
            stride_bb=X.stride(0), stride_bn=X.stride(1), stride_bh=X.stride(2),
            stride_ob=Out.stride(0), stride_oc=Out.stride(1), stride_oh=Out.stride(2),
            BLOCK_M=BLOCK_M, BLOCK_H=BLOCK_H,
            num_warps=4, num_stages=2,
        )

        # Allocate output P [B, C, H]
        P = torch.empty((B, C, H), dtype=torch.float32, device=hidden_states.device)

        # Launch GEMM kernel: compute P = Out @ W^T
        BLOCK_M_gemm = 64
        BLOCK_N_gemm = 64
        BLOCK_K_gemm = 64
        grid_gemm = (B, triton.cdiv(C, BLOCK_M_gemm), triton.cdiv(H, BLOCK_N_gemm))
        _batched_gemm_kernel_blocked[grid_gemm](
            Out, W, P,
            B=B, C=C, K=H,
            stride_xb=Out.stride(0), stride_xc=Out.stride(1), stride_xk=Out.stride(2),
            stride_w0=W.stride(1), stride_w1=W.stride(0),  # W^T indexing: rows=k (W.stride(0)), cols=n (W.stride(1))
            stride_pb=P.stride(0), stride_pc=P.stride(1), stride_pk=P.stride(2),
            BLOCK_M=BLOCK_M_gemm, BLOCK_N=BLOCK_N_gemm, BLOCK_K=BLOCK_K_gemm,
            num_warps=4, num_stages=2,
        )

        # Split outputs
        processed_encoder = P[:, :M, :]
        processed_hidden = P[:, M:, :]

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
