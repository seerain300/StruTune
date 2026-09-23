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

    # Store to Out[b, m, h]
    out_ptrs = Out_ptr + b * stride_ob + m_offsets[:, None] * stride_oc + h_offsets[None, :] * stride_oh
    tl.store(out_ptrs, vals, mask=mask_m[:, None] & mask_h[None, :])


@triton.jit
def _batched_matmul_kernel_explicit(
    X_ptr,  # *f32, [B, C, K] (Out tensor)
    W_ptr,  # *f32, [K, K] (process_weight)
    P_ptr,  # *f32, [B, C, K] (output)
    B: tl.constexpr,      # batch size
    C: tl.constexpr,      # sequence length (M + N)
    K: tl.constexpr,      # hidden dim (H)
    # Strides for X: [B, C, K]
    stride_xb,  # int
    stride_xc,  # int
    stride_xk,  # int
    # Strides for W: [K, K]
    stride_w0,  # int: stride along dim 0 (rows)
    stride_w1,  # int: stride along dim 1 (cols)
    # Strides for P: [B, C, K]
    stride_pb,  # int
    stride_pc,  # int
    stride_pk,  # int
    BLOCK_M: tl.constexpr,  # tile over C (sequences)
    BLOCK_N: tl.constexpr,  # tile over K (hidden dim)
    BLOCK_K: tl.constexpr,  # tile over reduction dim K
):
    # Grid: (B, ceil_div(C, BLOCK_M))
    b = tl.program_id(0)
    m_block = tl.program_id(1)

    # Offsets
    m_offsets = m_block * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    n_offsets = tl.arange(0, BLOCK_N)                     # [BLOCK_N]
    k_offsets = tl.arange(0, BLOCK_K)                     # [BLOCK_K]

    # Masks
    mask_m = m_offsets < C
    mask_n = n_offsets < K

    # Initialize accumulator [BLOCK_M, BLOCK_N]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over reduction dimension K in chunks
    for k0 in range(0, K, BLOCK_K):
        # Load X tile: X[b, m, k0:k0+BLOCK_K] -> [BLOCK_M, BLOCK_K]
        x_ptrs = X_ptr + b * stride_xb + m_offsets[:, None] * stride_xc + (k0 + k_offsets[None, :]) * stride_xk
        x_mask = mask_m[:, None] & ((k0 + k_offsets[None, :]) < K)
        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)  # [BLOCK_M, BLOCK_K]

        # For each hidden column n in the tile, accumulate with the corresponding W[:, n]
        # acc[m, n] += sum_{k in block} x_tile[m, k] * W[k, n]
        for n_idx in range(0, BLOCK_N):
            # Current hidden column index in this tile
            n = n_offsets[n_idx]
            valid_n = n < K
            # Gather W[:, n] for current hidden column: indices [0..K-1]
            w_ptrs = W_ptr + n * stride_w0 + (k0 + k_offsets) * stride_w1
            w_mask = ((k0 + k_offsets) < K) & valid_n
            w_col = tl.load(w_ptrs, mask=w_mask, other=0.0)  # [BLOCK_K]
            # Accumulate for all m rows
            acc[:, n_idx] += tl.sum(x_tile * w_col[None, :], axis=1)

    # Store P[b, m, n] = acc
    p_ptrs = P_ptr + b * stride_pb + m_offsets[:, None] * stride_pc + n_offsets[None, :] * stride_pk
    tl.store(p_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                encoder_hidden_states: torch.Tensor,
                process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version of the original run:
        1) Concatenate [B, M, H] and [B, N, H] along sequence to [B, C, H] with C = M + N.
        2) Compute P = concatenated @ process_weight.T using a Triton GEMM (explicit K-loop).
        3) Split P back into encoder and hidden streams along sequence.
        All heavy computation is performed by Triton kernels; forward orchestrates launches.
        """
        # Ensure CUDA tensors and consistent dtype
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, \
            "All inputs must be CUDA tensors for Triton kernels."

        # Enforce float32 for numerical consistency
        A = encoder_hidden_states.contiguous().to(torch.float32)
        Bt = hidden_states.contiguous().to(torch.float32)
        W = process_weight.contiguous().to(torch.float32)

        B = A.shape[0]
        M = A.shape[1]
        N = Bt.shape[1]
        H = A.shape[2]
        assert Bt.shape[2] == H and W.shape[0] == H and W.shape[1] == H, \
            "Hidden sizes must match: [B, M, H], [B, N, H], and process_weight [H, H]."

        C = M + N

        # 1) Concatenate along sequence into Out [B, C, H]
        Out = torch.empty((B, C, H), dtype=torch.float32, device=A.device)

        # Choose tile sizes for concatenation
        BLOCK_M = 64  # tile along sequence dimension
        BLOCK_H = 64  # tile along hidden dimension
        grid_concat = (B, triton.cdiv(C, BLOCK_M), triton.cdiv(H, BLOCK_H))
        _concat_seq_kernel[grid_concat](
            A, Bt, Out,
            B=B, M=M, N=N, H=H,
            stride_ab=A.stride(0), stride_am=A.stride(1), stride_ah=A.stride(2),
            stride_bb=Bt.stride(0), stride_bn=Bt.stride(1), stride_bh=Bt.stride(2),
            stride_ob=Out.stride(0), stride_oc=Out.stride(1), stride_oh=Out.stride(2),
            BLOCK_M=BLOCK_M, BLOCK_H=BLOCK_H,
            num_warps=4, num_stages=2,
        )

        # 2) Batched GEMM: P = Out @ W^T, output [B, C, K] where K=H
        P = torch.empty((B, C, H), dtype=torch.float32, device=A.device)

        # Strides for Out/X, W, and P
        stride_xb, stride_xc, stride_xk = Out.stride(0), Out.stride(1), Out.stride(2)
        stride_pb, stride_pc, stride_pk = P.stride(0), P.stride(1), P.stride(2)
        stride_w0, stride_w1 = W.stride(0), W.stride(1)

        BLOCK_Mg = 64
        BLOCK_Ng = 64
        BLOCK_Kg = 64
        grid_g = (B, triton.cdiv(C, BLOCK_Mg))
        _batched_matmul_kernel_explicit[grid_g](
            Out, W, P,
            B=B, C=C, K=H,
            stride_xb=stride_xb, stride_xc=stride_xc, stride_xk=stride_xk,
            stride_w0=stride_w0, stride_w1=stride_w1,
            stride_pb=stride_pb, stride_pc=stride_pc, stride_pk=stride_pk,
            BLOCK_M=BLOCK_Mg, BLOCK_N=BLOCK_Ng, BLOCK_K=BLOCK_Kg,
            num_warps=4, num_stages=3,
        )

        # 3) Split back into separate streams
        processed_encoder = P[:, :M, :]
        processed_hidden = P[:, M:, :]

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
