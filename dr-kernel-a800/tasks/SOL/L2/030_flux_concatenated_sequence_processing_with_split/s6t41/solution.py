import torch
import triton
import triton.language as tl


@triton.jit
def _concat_seq_kernel(
    A_ptr,        # *f32, [B, M, H]
    B_ptr,        # *f32, [B, N, H]
    Out_ptr,      # *f32, [B, C, H], C = M + N
    B: tl.constexpr,    # batch size (constexpr for specialization)
    M: tl.constexpr,    # text_seq_len (constexpr)
    N: tl.constexpr,    # img_seq_len (constexpr)
    H: tl.constexpr,    # hidden_dim (constexpr)
    stride_ab,    # int: stride along batch for A
    stride_am,    # int: stride along seq for A
    stride_ah,    # int: stride along hidden for A
    stride_bb,    # int: stride along batch for B
    stride_bn,    # int: stride along seq for B
    stride_bh,    # int: stride along hidden for B
    stride_ob,    # int: stride along batch for Out
    stride_oc,    # int: stride along seq for Out
    stride_oh,    # int: stride along hidden for Out
    BLOCK_M: tl.constexpr,  # tile along seq for A/B
    BLOCK_N: tl.constexpr,  # tile along hidden dim
):
    # Grid: (B, ceil_div(C, BLOCK_M))
    b = tl.program_id(0)
    seq_block = tl.program_id(1)

    # Offsets along the sequence (C) for this tile
    c_offsets = seq_block * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M], correspond to positions in [0, M) and [M, M+N)
    c_mask = c_offsets < (M + N)

    # Split which rows come from A (0..M-1) and B (M..M+N-1)
    from_A = c_offsets < M
    n_offsets = c_offsets - M  # valid only where from_A is False; but we won't use B for those

    # Hidden feature offsets
    h_offsets = tl.arange(0, BLOCK_N)  # [BLOCK_N], cover hidden dim H
    h_mask = h_offsets < H

    # Build A and B pointers and masks
    # A: [B, M, H]
    a_ptrs = A_ptr + b * stride_ab + c_offsets[:, None] * stride_am + h_offsets[None, :] * stride_ah
    a_mask = c_mask[:, None] & h_mask[None, :]

    # B: [B, N, H], but we write to Out at position c_offsets >= M
    b_ptrs = B_ptr + b * stride_bb + n_offsets[:, None] * stride_bn + h_offsets[None, :] * stride_bh
    b_mask = (~from_A)[:, None] & (c_offsets < M) & h_mask[None, :]  # Note: c < M is equivalent to ~from_A; adjust to c >= M

    # We need correct b_mask: for rows where from_A is False (i.e., c >= M), use c_offsets - M
    # Better: use mask based on c_mask
    b_mask = (~from_A)[:, None] & h_mask[None, :] & (c_offsets[:, None] >= M)

    # Load A and B
    a_vals = tl.load(a_ptrs, mask=a_mask, other=0.0)
    b_vals = tl.load(b_ptrs, mask=b_mask, other=0.0)

    # Select values: if from_A, take a_vals; else take b_vals (broadcast across hidden dim)
    vals = tl.where(from_A[:, None], a_vals, b_vals)

    # Store to Out[b, c_offsets, h_offsets]
    out_ptrs = Out_ptr + b * stride_ob + c_offsets[:, None] * stride_oc + h_offsets[None, :] * stride_oh
    store_mask = c_mask[:, None] & h_mask[None, :]
    tl.store(out_ptrs, vals, mask=store_mask)


@triton.jit
def _batched_gemm_2d_kernel(
    X_ptr,  # *f32, [B, C, K] where C = M + N, K = H
    W_ptr,  # *f32, [K, K] (process_weight)
    P_ptr,  # *f32, [B, C, K]
    B: tl.constexpr,     # batch size
    C: tl.constexpr,     # sequence length
    K: tl.constexpr,     # hidden dim
    # Strides for X: we treat X as (B, C, K)
    stride_xb,  # int
    stride_xc,  # int
    stride_xk,  # int
    # Strides for W: [K, K]
    stride_w0,  # int
    stride_w1,  # int
    # Strides for P: [B, C, K]
    stride_pb,  # int
    stride_pc,  # int
    stride_pk,  # int
    BLOCK_M: tl.constexpr,  # tile over seq dimension C (output rows)
    BLOCK_N: tl.constexpr,  # tile over hidden dim K (output cols)
    BLOCK_K: tl.constexpr,  # tile over reduction dim K (input hidden)
):
    # 2D grid: (batch, tiles along C)
    b = tl.program_id(0)
    c_block = tl.program_id(1)

    # tile offsets over output features
    c_offsets = c_block * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M], output rows (sequence positions)
    c_mask = c_offsets < C

    k_offsets = tl.arange(0, BLOCK_N)                     # [BLOCK_N], output columns (hidden features)
    k_mask = k_offsets < K

    # Initialize accumulator [BLOCK_M, BLOCK_N]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over reduction dimension K in chunks
    for k0 in range(0, K, BLOCK_K):
        kk = k0 + tl.arange(0, BLOCK_K)  # [BLOCK_K]
        k_mask_tile = kk < K

        # Load X tile [BLOCK_M, BLOCK_K]: X[b, c_offsets, kk]
        x_ptrs = X_ptr + b * stride_xb + c_offsets[:, None] * stride_xc + kk[None, :] * stride_xk
        x_mask = c_mask[:, None] & k_mask_tile[None, :]
        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)  # [BLOCK_M, BLOCK_K]

        # Load W tile [BLOCK_K, BLOCK_N]: W[kk, k_offsets] (since X @ W^T, we read W directly as [K, K])
        w_ptrs = W_ptr + kk[:, None] * stride_w0 + k_offsets[None, :] * stride_w1
        w_mask = k_mask_tile[:, None] & k_mask[None, :]
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)  # [BLOCK_K, BLOCK_N]

        # Accumulate
        acc += tl.dot(x_tile, w_tile)

    # Store accumulator to P[b, c_offsets, k_offsets]
    p_ptrs = P_ptr + b * stride_pb + c_offsets[:, None] * stride_pc + k_offsets[None, :] * stride_pk
    p_mask = c_mask[:, None] & k_mask[None, :]
    tl.store(p_ptrs, acc, mask=p_mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                encoder_hidden_states: torch.Tensor,
                process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version:
        - Concatenate encoder_hidden_states and hidden_states along sequence dimension in Triton.
        - Apply linear projection (X @ process_weight.T) in Triton GEMM.
        - Split outputs back into two tensors.
        """
        # Ensure CUDA tensors; Triton kernels require CUDA
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, \
            "Triton kernels require CUDA tensors."

        # Shapes
        B = hidden_states.shape[0]
        M = encoder_hidden_states.shape[1]  # text_seq_len
        N = hidden_states.shape[1]          # img_seq_len
        H = hidden_states.shape[2]          # hidden_dim (also size of process_weight)

        # Make inputs contiguous
        A = encoder_hidden_states.contiguous()         # [B, M, H]
        B_img = hidden_states.contiguous()             # [B, N, H]
        W = process_weight.contiguous()                # [H, H]

        # Allocate concatenated Out [B, C, H], C = M + N
        C = M + N
        Out = torch.empty((B, C, H), dtype=torch.float32, device=hidden_states.device)

        # Launch concatenation Triton kernel
        BLOCK_M = 64
        BLOCK_N = 64
        grid_concat = (B, triton.cdiv(C, BLOCK_M))
        _concat_seq_kernel[grid_concat](
            A, B_img, Out,
            B=B, M=M, N=N, H=H,
            stride_ab=A.stride(0), stride_am=A.stride(1), stride_ah=A.stride(2),
            stride_bb=B_img.stride(0), stride_bn=B_img.stride(1), stride_bh=B_img.stride(2),
            stride_ob=Out.stride(0), stride_oc=Out.stride(1), stride_oh=Out.stride(2),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            num_warps=4, num_stages=2
        )

        # Allocate output P [B, C, H]
        P = torch.empty((B, C, H), dtype=torch.float32, device=hidden_states.device)

        # Launch Triton GEMM kernel: P = Out @ W^T
        BLOCK_M_GEMM = 128
        BLOCK_N_GEMM = 128
        BLOCK_K_GEMM = 64
        grid_gemm = (B, triton.cdiv(C, BLOCK_M_GEMM))
        _batched_gemm_2d_kernel[grid_gemm](
            Out, W, P,
            B=B, C=C, K=H,
            stride_xb=Out.stride(0), stride_xc=Out.stride(1), stride_xk=Out.stride(2),
            stride_w0=W.stride(0), stride_w1=W.stride(1),
            stride_pb=P.stride(0), stride_pc=P.stride(1), stride_pk=P.stride(2),
            BLOCK_M=BLOCK_M_GEMM, BLOCK_N=BLOCK_N_GEMM, BLOCK_K=BLOCK_K_GEMM,
            num_warps=8, num_stages=3
        )

        # Split back
        processed_encoder = P[:, :M, :]
        processed_hidden = P[:, M:, :]
        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
