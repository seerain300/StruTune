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
    BLOCK_M: tl.constexpr,  # tile along seq (M+N)
    BLOCK_H: tl.constexpr,  # tile along hidden
):
    # Grid: (B, ceil_div(C, BLOCK_M), ceil_div(H, BLOCK_H))
    b = tl.program_id(0)
    m_block = tl.program_id(1)
    h_block = tl.program_id(2)

    # Sequence and hidden offsets for this tile
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

    # Store to Out: Out[b, m, h]
    out_ptrs = Out_ptr + b * stride_ob + m_offsets[:, None] * stride_oc + h_offsets[None, :] * stride_oh
    tl.store(out_ptrs, vals, mask=(mask_m[:, None] & mask_h[None, :]))


@triton.jit
def _batched_gemm_kernel(
    X_ptr,  # *f32, [B, C, K] where C = M + N
    W_T_ptr,  # *f32, [K, H] (transpose of process_weight: W_T[k, h] = W[h, k])
    P_ptr,  # *f32, [B, C, K] (result)
    B: tl.constexpr,      # batch size
    C: tl.constexpr,      # sequence length
    K: tl.constexpr,      # hidden dim (constexpr for Triton)
    # Strides for X: treated as (B, C, K)
    stride_xb,  # int
    stride_xc,  # int
    stride_xk,  # int
    # Strides for W_T: [K, H]
    stride_wtk,  # int: stride along dim 0 (rows)
    stride_wth,  # int: stride along dim 1 (cols)
    # Strides for P: [B, C, K]
    stride_pb,  # int
    stride_pc,  # int
    stride_pk,  # int
    BLOCK_M: tl.constexpr,  # tile over C (sequences)
    BLOCK_N: tl.constexpr,  # tile over K (hidden dim)
    BLOCK_K: tl.constexpr,  # tile over reduction dim K
):
    # 3D grid: (B, ceil_div(C, BLOCK_M), ceil_div(K, BLOCK_N))
    b = tl.program_id(0)
    c_block = tl.program_id(1)
    k_block = tl.program_id(2)

    # tile offsets
    c_offsets = c_block * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    k_offsets = k_block * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]

    # masks for boundaries
    mask_c = c_offsets < C
    mask_k = k_offsets < K

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over reduction dimension K in chunks of BLOCK_K
    for k0 in range(0, K, BLOCK_K):
        # X tile: [BLOCK_M x BLOCK_K]
        x_ptrs = X_ptr + b * stride_xb + c_offsets[:, None] * stride_xc + (k0 + tl.arange(0, BLOCK_K))[None, :] * stride_xk
        x_mask = mask_c[:, None] & ((k0 + tl.arange(0, BLOCK_K))[None, :] < K)
        x_vals = tl.load(x_ptrs, mask=x_mask, other=0.0)

        # W^T tile: [BLOCK_K x BLOCK_N]
        w_ptrs = W_T_ptr + (k0 + tl.arange(0, BLOCK_K))[:, None] * stride_wtk + k_offsets[None, :] * stride_wth
        w_mask = ((k0 + tl.arange(0, BLOCK_K))[:, None] < K) & (k_offsets[None, :] < H)  # W_T is [K, H], second dim is H
        w_vals = tl.load(w_ptrs, mask=w_mask, other=0.0)

        # Accumulate
        acc += tl.dot(x_vals, w_vals)

    # Store results to P
    p_ptrs = P_ptr + b * stride_pb + c_offsets[:, None] * stride_pc + k_offsets[None, :] * stride_pk
    tl.store(p_ptrs, acc, mask=(mask_c[:, None] & mask_k[None, :]))


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                encoder_hidden_states: torch.Tensor,
                process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-only implementation:
        - Concatenate encoder_hidden_states and hidden_states along sequence dim using a Triton kernel.
        - Compute processed = concatenated @ process_weight.T using a Triton blocked GEMM kernel.
        - Split processed back into (processed_encoder, processed_hidden).
        """
        # Ensure CUDA and contiguous
        device = hidden_states.device
        B = hidden_states.shape[0]
        M = encoder_hidden_states.shape[1]
        N = hidden_states.shape[1]
        H = hidden_states.shape[2]
        assert encoder_hidden_states.shape == (B, M, H), "encoder_hidden_states must be [batch, text_seq_len, hidden_dim]"
        assert hidden_states.shape == (B, N, H), "hidden_states must be [batch, img_seq_len, hidden_dim]"
        assert process_weight.shape == (H, H), "process_weight must be [hidden_dim, hidden_dim]"

        # Make sure tensors are on CUDA and contiguous
        A = encoder_hidden_states.contiguous().to(device=device, dtype=torch.float32)
        Bt = hidden_states.contiguous().to(device=device, dtype=torch.float32)
        W = process_weight.contiguous().to(device=device, dtype=torch.float32)

        # 1) Concatenate along sequence dimension: Out [B, C, H], C = M + N
        C = M + N
        Out = torch.empty((B, C, H), dtype=torch.float32, device=device)

        # Launch concatenation kernel
        BLOCK_M = 64  # tile along sequence
        BLOCK_H = 64  # tile along hidden
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

        # 2) Compute processed = Out @ W^T, where W^T is [K, H]
        # Prepare W^T (transpose of process_weight)
        # Note: W is [H, H]; W^T is [H, H] but we treat it as [K, H] where K=H
        W_T = W.transpose(0, 1).contiguous()  # [H, H]

        P = torch.empty((B, C, H), dtype=torch.float32, device=device)

        # GEMM kernel parameters
        BLOCK_M_GEMM = 64
        BLOCK_N_GEMM = 64
        BLOCK_K_GEMM = 64
        grid_gemm = (B, triton.cdiv(C, BLOCK_M_GEMM), triton.cdiv(H, BLOCK_N_GEMM))
        _batched_gemm_kernel[grid_gemm](
            Out, W_T, P,
            B=B, C=C, K=H,  # specialize on H
            stride_xb=Out.stride(0), stride_xc=Out.stride(1), stride_xk=Out.stride(2),
            stride_wtk=W_T.stride(0), stride_wth=W_T.stride(1),
            stride_pb=P.stride(0), stride_pc=P.stride(1), stride_pk=P.stride(2),
            BLOCK_M=BLOCK_M_GEMM, BLOCK_N=BLOCK_N_GEMM, BLOCK_K=BLOCK_K_GEMM,
            num_warps=4, num_stages=3,
        )

        # 3) Split outputs along sequence dimension
        processed_encoder = P[:, :M, :]
        processed_hidden = P[:, M:, :]

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
