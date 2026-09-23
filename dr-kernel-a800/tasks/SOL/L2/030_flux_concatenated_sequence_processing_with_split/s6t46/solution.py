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
    BLOCK_M: tl.constexpr,  # tile size along seq
    BLOCK_H: tl.constexpr,  # tile size along hidden
):
    # Grid: (B, ceil_div(C, BLOCK_M))
    b = tl.program_id(0)
    tile = tl.program_id(1)

    # sequence offsets handled by this program
    seq_offsets = tile * BLOCK_M + tl.arange(0, BLOCK_M)  # length BLOCK_M
    C_total = M + N

    # For each sequence position, determine source: A if < M, else B at offset - M
    from_A = seq_offsets < M
    from_B_offsets = seq_offsets - M  # valid where not from_A

    # Loop over hidden dimension in chunks
    for h0 in range(0, H, BLOCK_H):
        h_offsets = h0 + tl.arange(0, BLOCK_H)
        mask_h = h_offsets < H

        # Load from A where applicable
        a_ptrs = A_ptr + b * stride_ab + seq_offsets[:, None] * stride_am + h_offsets[None, :] * stride_ah
        load_mask_a = from_A[:, None] & mask_h[None, :]
        vals_a = tl.load(a_ptrs, mask=load_mask_a, other=0.0)

        # Store to Out at positions < M
        out_ptrs_a = Out_ptr + b * stride_ob + seq_offsets[:, None] * stride_oc + h_offsets[None, :] * stride_oh
        store_mask_a = load_mask_a
        tl.store(out_ptrs_a, vals_a, mask=store_mask_a)

        # Load from B where applicable
        b_ptrs = B_ptr + b * stride_bb + from_B_offsets[:, None] * stride_bn + h_offsets[None, :] * stride_bh
        load_mask_b = (~from_A)[:, None] & mask_h[None, :]  # False where from_A, otherwise True
        vals_b = tl.load(b_ptrs, mask=load_mask_b, other=0.0)

        # Store to Out at positions M + (seq_offsets - M)
        out_ptrs_b = Out_ptr + b * stride_ob + (M + seq_offsets[:, None]) * stride_oc + h_offsets[None, :] * stride_oh
        store_mask_b = load_mask_b
        tl.store(out_ptrs_b, vals_b, mask=store_mask_b)


@triton.jit
def _batched_matmul_block_kernel(
    X_ptr,  # *f32, [B, C, K] where C = M + N, K = H
    W_ptr,  # *f32, [K, K] (process_weight)
    P_ptr,  # *f32, [B, C, K]
    B: tl.constexpr,      # batch size
    C: tl.constexpr,      # sequence length
    K: tl.constexpr,      # hidden dim
    stride_xb,  # int: stride for batch in X
    stride_xc,  # int: stride for seq in X
    stride_xk,  # int: stride for hidden in X
    stride_w0,  # int: stride for rows in W (dim 0)
    stride_w1,  # int: stride for cols in W (dim 1)
    stride_pb,  # int: stride for batch in P
    stride_pc,  # int: stride for seq in P
    stride_pk,  # int: stride for hidden in P
    BLOCK_M: tl.constexpr,  # tile over C
    BLOCK_N: tl.constexpr,  # tile over K
    BLOCK_K: tl.constexpr,  # tile over reduction dim K
):
    # 3D grid: (B, ceil_div(C, BLOCK_M), ceil_div(K, BLOCK_N))
    b = tl.program_id(0)
    m_block = tl.program_id(1)
    n_block = tl.program_id(2)

    m_offsets = m_block * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    n_offsets = n_block * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]
    mask_m = m_offsets < C
    mask_n = n_offsets < K

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)  # [BLOCK_K]
        mask_k = k_offsets < K

        # Load X tile: [BLOCK_M, BLOCK_K], X[b, m, k]
        x_ptrs = X_ptr + b * stride_xb + m_offsets[:, None] * stride_xc + k_offsets[None, :] * stride_xk
        x_mask = mask_m[:, None] & mask_k[None, :]
        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)

        # Load W tile as [BLOCK_K, BLOCK_N], W[k, n]
        w_ptrs = W_ptr + k_offsets[:, None] * stride_w0 + n_offsets[None, :] * stride_w1
        w_mask = mask_k[:, None] & mask_n[None, :]
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)

        # Accumulate
        acc += tl.dot(x_tile, w_tile)

    # Store result to P[b, m, n]
    p_ptrs = P_ptr + b * stride_pb + m_offsets[:, None] * stride_pc + n_offsets[None, :] * stride_pk
    p_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(p_ptrs, acc, mask=p_mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        Triton-optimized version:
        - Concatenate encoder_hidden_states and hidden_states along sequence in Triton
        - Compute processed = concatenated @ process_weight.T in Triton (batched GEMM)
        - Split outputs into separate streams
        """
        # Shapes
        B = hidden_states.shape[0]
        M = encoder_hidden_states.shape[1]  # text_seq_len
        N = hidden_states.shape[1]          # img_seq_len
        H = hidden_states.shape[2]          # hidden_dim
        C = M + N

        # We assume inputs are on CUDA (as in the original). No .contiguous() calls, no torch GPU ops in forward.
        # Allocate output buffer for full processed [B, C, H]. We do NOT perform any torch GPU ops here beyond allocation.
        ProcessedAll = torch.empty((B, C, H), device=hidden_states.device, dtype=torch.float32)

        # Launch concatenation kernel
        # Strides for A
        stride_ab, stride_am, stride_ah = encoder_hidden_states.stride(0), encoder_hidden_states.stride(1), encoder_hidden_states.stride(2)
        # Strides for B (hidden_states)
        stride_bb, stride_bn, stride_bh = hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2)
        # Strides for Out
        stride_ob, stride_oc, stride_oh = ProcessedAll.stride(0), ProcessedAll.stride(1), ProcessedAll.stride(2)

        # Tile sizes (robust defaults)
        BLOCK_M = 128
        BLOCK_H = 64
        grid_concat = (B, triton.cdiv(C, BLOCK_M))
        _concat_seq_kernel[grid_concat](
            encoder_hidden_states, hidden_states, ProcessedAll,
            B=B, M=M, N=N, H=H,
            stride_ab=stride_ab, stride_am=stride_am, stride_ah=stride_ah,
            stride_bb=stride_bb, stride_bn=stride_bn, stride_bh=stride_bh,
            stride_ob=stride_ob, stride_oc=stride_oc, stride_oh=stride_oh,
            BLOCK_M=BLOCK_M, BLOCK_H=BLOCK_H,
            num_warps=4, num_stages=2,
        )

        # Compute P = X @ W^T where X = ProcessedAll [B, C, H], W [H, H], P [B, C, H]
        # Allocate output P
        P = torch.empty((B, C, H), device=hidden_states.device, dtype=torch.float32)

        # Strides for X (ProcessedAll), W, P
        stride_xb, stride_xc, stride_xk = ProcessedAll.stride(0), ProcessedAll.stride(1), ProcessedAll.stride(2)
        stride_w0, stride_w1 = process_weight.stride(0), process_weight.stride(1)
        stride_pb, stride_pc, stride_pk = P.stride(0), P.stride(1), P.stride(2)

        # GEMM tiling parameters
        BLOCK_M_gemm = 64
        BLOCK_N_gemm = 64
        BLOCK_K_gemm = 64

        grid_gemm = (B, triton.cdiv(C, BLOCK_M_gemm), triton.cdiv(H, BLOCK_N_gemm))
        _batched_matmul_block_kernel[grid_gemm](
            ProcessedAll, process_weight, P,
            B=B, C=C, K=H,
            stride_xb=stride_xb, stride_xc=stride_xc, stride_xk=stride_xk,
            stride_w0=stride_w0, stride_w1=stride_w1,
            stride_pb=stride_pb, stride_pc=stride_pc, stride_pk=stride_pk,
            BLOCK_M=BLOCK_M_gemm, BLOCK_N=BLOCK_N_gemm, BLOCK_K=BLOCK_K_gemm,
            num_warps=4, num_stages=3,
        )

        # Split along sequence dimension: return [B, M, H] and [B, N, H]
        processed_encoder = P[:, :M, :]
        processed_hidden = P[:, M:, :]

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
