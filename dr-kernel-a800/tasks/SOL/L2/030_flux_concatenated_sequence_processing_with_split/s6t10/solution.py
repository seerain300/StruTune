import torch
import triton
import triton.language as tl


@triton.jit
def _concat_seq_kernel(
    A_ptr,        # *f32, [B, M, H]
    B_ptr,        # *f32, [B, N, H]
    Out_ptr,      # *f32, [B, C, H], C = M + N
    B: tl.constexpr,    # batch size (constexpr)
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
    BLOCK_M: tl.constexpr,  # tile over sequences (C dimension)
    BLOCK_N: tl.constexpr,  # tile over hidden dim (H)
):
    # Grid is (B, ceil_div(C, BLOCK_M))
    b = tl.program_id(0)
    c_block = tl.program_id(1)
    C = M + N

    # sequence offsets handled by this program
    seq_offsets = c_block * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    valid = seq_offsets < C
    m_mask = seq_offsets < M
    n_mask = seq_offsets >= M

    # iterate over hidden dimension in chunks
    for h0 in range(0, H, BLOCK_N):
        h_offsets = h0 + tl.arange(0, BLOCK_N)  # [BLOCK_N]
        h_mask = h_offsets < H

        # Load from A where seq_offsets < M
        a_ptrs = A_ptr + b * stride_ab + seq_offsets[:, None] * stride_am + h_offsets[None, :] * stride_ah
        a_mask = m_mask[:, None] & h_mask[None, :]
        a_vals = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # Load from B where seq_offsets >= M
        b_ptrs = B_ptr + b * stride_bb + (seq_offsets - M)[:, None] * stride_bn + h_offsets[None, :] * stride_bh
        b_mask = n_mask[:, None] & h_mask[None, :]
        b_vals = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Select based on validity; for invalid seq_offsets, we won't store
        vals = tl.where(valid[:, None], a_vals, b_vals)  # [BLOCK_M, BLOCK_N]

        # Store to Out
        out_ptrs = Out_ptr + b * stride_ob + seq_offsets[:, None] * stride_oc + h_offsets[None, :] * stride_oh
        tl.store(out_ptrs, vals, mask=(valid[:, None] & h_mask[None, :]))


@triton.jit
def _batched_matmul_kernel(
    X_ptr,        # *f32, [B, C, K] where C = M + N, K = H
    W_ptr,        # *f32, [K, K] (process_weight)
    P_ptr,        # *f32, [B, C, K] output
    B: tl.constexpr,      # batch size
    C: tl.constexpr,      # sequence length (M + N)
    K: tl.constexpr,      # hidden dim
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
    BLOCK_M: tl.constexpr,  # tile over sequences (C)
    BLOCK_N: tl.constexpr,  # tile over hidden dim (K)
    BLOCK_K: tl.constexpr,  # tile over reduction dim (K)
):
    # Grid is (B, ceil_div(C, BLOCK_M))
    b = tl.program_id(0)
    c_block = tl.program_id(1)

    # tile offsets
    m_offsets = c_block * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    n_offsets = tl.arange(0, BLOCK_N)                     # [BLOCK_N]
    m_mask = m_offsets < C
    n_mask = n_offsets < K

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over reduction dimension K in chunks of BLOCK_K
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)  # [BLOCK_K]
        k_mask = k_offsets < K

        # Load X tile: [BLOCK_M, BLOCK_K]
        x_ptrs = X_ptr + b * stride_xb + m_offsets[:, None] * stride_xc + k_offsets[None, :] * stride_xk
        x_mask = m_mask[:, None] & k_mask[None, :]
        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)

        # Load W tile as [BLOCK_K, BLOCK_N]: W[k_offsets, n_offsets]
        w_ptrs = W_ptr + k_offsets[:, None] * stride_w0 + n_offsets[None, :] * stride_w1
        w_mask = k_mask[:, None] & n_mask[None, :]
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)

        # Accumulate
        acc += tl.dot(x_tile, w_tile)  # [BLOCK_M, BLOCK_N]

    # Store result to P
    p_ptrs = P_ptr + b * stride_pb + m_offsets[:, None] * stride_pc + n_offsets[None, :] * stride_pk
    store_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(p_ptrs, acc, mask=store_mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton implementation:
          - Concatenates encoder_hidden_states and hidden_states along the sequence dimension using Triton.
          - Computes the linear projection P = X @ process_weight.T using a Triton batched matmul kernel.
          - Splits P back into (processed_encoder, processed_hidden) streams.
        """
        # Ensure CUDA and float32
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "All inputs must be CUDA tensors."
        assert hidden_states.dtype == torch.float32 and encoder_hidden_states.dtype == torch.float32 and process_weight.dtype == torch.float32, "All tensors must be float32."

        B, M, H = encoder_hidden_states.shape
        B2, N, H2 = hidden_states.shape
        assert B == B2 and H == H2, "Incompatible shapes for inputs."
        assert process_weight.shape[0] == H and process_weight.shape[1] == H, "process_weight must be [H, H]."

        # Make contiguous
        A = encoder_hidden_states.contiguous()  # [B, M, H]
        Bt = hidden_states.contiguous()         # [B, N, H]
        W = process_weight.contiguous()         # [H, H]

        # Allocate output for concatenation
        C = M + N
        Out = torch.empty((B, C, H), dtype=torch.float32, device=A.device)

        # Launch Triton concatenation kernel
        BLOCK_M = 128  # tile over sequence dimension
        BLOCK_N = 64   # tile over hidden dimension
        grid_concat = (B, triton.cdiv(C, BLOCK_M))
        _concat_seq_kernel[grid_concat](
            A, Bt, Out,
            B=B, M=M, N=N, H=H,
            stride_ab=A.stride(0), stride_am=A.stride(1), stride_ah=A.stride(2),
            stride_bb=Bt.stride(0), stride_bn=Bt.stride(1), stride_bh=Bt.stride(2),
            stride_ob=Out.stride(0), stride_oc=Out.stride(1), stride_oh=Out.stride(2),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            num_warps=4, num_stages=2
        )

        # Allocate output for GEMM
        P = torch.empty((B, C, H), dtype=torch.float32, device=A.device)

        # Launch Triton GEMM kernel: P = Out @ W^T
        BLOCK_M_GEMM = 64   # tile over sequences (C)
        BLOCK_N_GEMM = 64   # tile over hidden dim (K)
        BLOCK_K_GEMM = 64   # reduction tile (K)
        grid_gemm = (B, triton.cdiv(C, BLOCK_M_GEMM))
        _batched_matmul_kernel[grid_gemm](
            Out, W, P,
            B=B, C=C, K=H,
            stride_xb=Out.stride(0), stride_xc=Out.stride(1), stride_xk=Out.stride(2),
            stride_w0=W.stride(0), stride_w1=W.stride(1),
            stride_pb=P.stride(0), stride_pc=P.stride(1), stride_pk=P.stride(2),
            BLOCK_M=BLOCK_M_GEMM, BLOCK_N=BLOCK_N_GEMM, BLOCK_K=BLOCK_K_GEMM,
            num_warps=8, num_stages=3
        )

        # Split along sequence dimension
        processed_encoder = P[:, :M, :]
        processed_hidden = P[:, M:, :]

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
