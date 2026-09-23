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
    BLOCK_M: tl.constexpr,  # tile along sequence (C dimension)
):
    # Grid: (B, ceil_div(C, BLOCK_M))
    b = tl.program_id(0)
    c_block = tl.program_id(1)

    # Compute offsets for sequence dimension
    m_offsets = c_block * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    C = M + N
    mask_m = m_offsets < C

    # For each sequence position m, determine source: A if m < M else B
    for mi in range(BLOCK_M):
        m = m_offsets[mi]
        valid = mask_m[mi]
        # Compute pointers for Out
        out_ptr = Out_ptr + b * stride_ob + m * stride_oc + tl.arange(0, H) * stride_oh  # [H]
        # Decide source and copy
        is_from_A = m < M
        if is_from_A:
            a_ptr = A_ptr + b * stride_ab + m * stride_am + tl.arange(0, H) * stride_ah  # [H]
            tl.store(out_ptr, tl.load(a_ptr, mask=valid, other=0.0))
        else:
            n = m - M
            b_ptr = B_ptr + b * stride_bb + n * stride_bn + tl.arange(0, H) * stride_bh  # [H]
            tl.store(out_ptr, tl.load(b_ptr, mask=valid, other=0.0))


@triton.jit
def _batched_gemm_per_m_kernel(
    X_ptr,   # *f32, [B, C, K] (concatenated input)
    W_ptr,   # *f32, [K, K] (process_weight)
    P_ptr,   # *f32, [B, C, K] (output)
    B: tl.constexpr,        # batch size (constexpr)
    C: tl.constexpr,        # sequence length (M + N) (constexpr)
    K: tl.constexpr,        # hidden_dim (constexpr)
    stride_xb,  # int: stride for batch in X
    stride_xc,  # int: stride for seq in X
    stride_xk,  # int: stride for hidden in X
    stride_w0,  # int: stride for dim 0 (rows) in W
    stride_w1,  # int: stride for dim 1 (cols) in W
    stride_pb,  # int: stride for batch in P
    stride_pc,  # int: stride for seq in P
    stride_pk,  # int: stride for hidden in P
    BLOCK_K: tl.constexpr,  # tile size along K
):
    # Grid: (B, C) -> each program computes one sequence position m for one batch b
    b = tl.program_id(0)
    m = tl.program_id(1)  # m in [0, C)

    # Accumulator for this m
    acc = tl.zeros((K,), dtype=tl.float32)

    # Loop over K in tiles
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)  # [BLOCK_K]
        k_mask = k_offsets < K

        # Load X[b, m, k_offsets] as a vector
        x_ptrs = X_ptr + b * stride_xb + m * stride_xc + k_offsets * stride_xk
        x_vec = tl.load(x_ptrs, mask=k_mask, other=0.0)  # [BLOCK_K]

        # Load W[k_offsets, k_offsets] as a BLOCK_K x BLOCK_K tile
        w_ptrs = W_ptr + k_offsets[:, None] * stride_w0 + k_offsets[None, :] * stride_w1  # [BLOCK_K, BLOCK_K]
        w_mask = (k_offsets[:, None] < K) & (k_offsets[None, :] < K)
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)  # [BLOCK_K, BLOCK_K]

        # Accumulate: acc += sum_j x_vec[j] * w_tile[j, :]
        # Compute outer product accumulate per j
        # acc[i] += x_vec[j] * w_tile[j, i]
        for j in range(BLOCK_K):
            j_mask = (k0 + j) < K
            # w_col_j is w_tile[j, :] -> [BLOCK_K]
            w_col_j = w_tile[j, :]
            # acc += x_vec[j] * w_col_j
            # x_vec[j] is scalar; w_col_j is [BLOCK_K]; Triton will broadcast multiply
            acc += x_vec[j] * w_col_j

    # Store acc to P[b, m, :]
    p_ptrs = P_ptr + b * stride_pb + m * stride_pc + tl.arange(0, K) * stride_pk
    tl.store(p_ptrs, acc, mask=(tl.arange(0, K) < K))


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        Triton-only implementation of:
          concatenated = cat([encoder_hidden_states, hidden_states], dim=1)
          processed = concatenated @ process_weight.T
          return processed_encoder[:, :M, :], processed_hidden[:, :N, :]
        """
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Inputs must be CUDA tensors for Triton."
        B = hidden_states.shape[0]
        M = encoder_hidden_states.shape[1]
        N = hidden_states.shape[1]
        H = hidden_states.shape[2]
        assert encoder_hidden_states.shape == (B, M, H), "encoder_hidden_states shape must be [batch, text_seq_len, hidden_dim]"
        assert hidden_states.shape == (B, N, H), "hidden_states shape must be [batch, img_seq_len, hidden_dim]"
        assert process_weight.shape == (H, H), "process_weight shape must be [hidden_dim, hidden_dim]"

        # Ensure contiguous
        A = encoder_hidden_states.contiguous()
        Bt = hidden_states.contiguous()
        W = process_weight.contiguous()

        # 1) Concatenate along sequence dimension: Out [B, C, H], C = M + N
        C = M + N
        Out = torch.empty((B, C, H), device=A.device, dtype=torch.float32)

        # Launch concat kernel
        BLOCK_M = 256  # tile along C
        grid = (B, triton.cdiv(C, BLOCK_M))
        _concat_seq_kernel[grid](
            A, Bt, Out,
            B=B, M=M, N=N, H=H,
            stride_ab=A.stride(0), stride_am=A.stride(1), stride_ah=A.stride(2),
            stride_bb=Bt.stride(0), stride_bn=Bt.stride(1), stride_bh=Bt.stride(2),
            stride_ob=Out.stride(0), stride_oc=Out.stride(1), stride_oh=Out.stride(2),
            BLOCK_M=BLOCK_M,
            num_warps=4, num_stages=2
        )

        # 2) GEMM: P = Out @ W^T, result [B, C, H]
        P = torch.empty((B, C, H), device=A.device, dtype=torch.float32)
        # Use simplified per-m kernel for robustness
        grid2 = (B, C)
        _batched_gemm_per_m_kernel[grid2](
            Out, W, P,
            B=B, C=C, K=H,
            stride_xb=Out.stride(0), stride_xc=Out.stride(1), stride_xk=Out.stride(2),
            stride_w0=W.stride(0), stride_w1=W.stride(1),
            stride_pb=P.stride(0), stride_pc=P.stride(1), stride_pk=P.stride(2),
            BLOCK_K=64,  # tile along K, H is tl.constexpr, so this works for H up to 4096
            num_warps=4, num_stages=2
        )

        # 3) Split back into encoder and hidden parts
        processed_encoder = P[:, :M, :]
        processed_hidden = P[:, M:, :]

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
