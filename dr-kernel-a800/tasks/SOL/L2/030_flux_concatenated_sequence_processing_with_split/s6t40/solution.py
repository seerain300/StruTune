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
    BLOCK_M: tl.constexpr,  # tile size along seq from A (first M)
    BLOCK_N: tl.constexpr,  # tile size along seq from B (second N)
):
    # 2D grid: (batch, ceil_div(C, BLOCK_M + BLOCK_N))
    b = tl.program_id(0)
    tile_start = tl.program_id(1) * (BLOCK_M + BLOCK_N)

    # Offsets within C
    m_offsets = tile_start + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    n_offsets = tile_start + tl.arange(0, BLOCK_N)  # [BLOCK_N]
    m_mask = m_offsets < M
    n_mask = n_offsets < N

    # For each hidden h in H
    for h in range(0, H):
        # Load from A where applicable: Out[b, m, h]
        a_ptrs = A_ptr + b * stride_ab + m_offsets * stride_am + h * stride_ah
        a_mask = m_mask
        a_vals = tl.load(a_ptrs, mask=a_mask, other=0.0)  # [BLOCK_M]

        # Load from B where applicable: Out[b, m + M, h]
        b_ptrs = B_ptr + b * stride_bb + (m_offsets + M) * stride_bn + h * stride_bh
        b_mask = n_mask
        b_vals = tl.load(b_ptrs, mask=b_mask, other=0.0)  # [BLOCK_N]

        # Store to Out[b, m_offsets, h] and Out[b, m_offsets + M, h]
        out_m_ptrs = Out_ptr + b * stride_ob + m_offsets * (stride_oc) + h * (stride_oh)
        out_n_ptrs = Out_ptr + b * stride_ob + (m_offsets + M) * (stride_oc) + h * (stride_oh)

        # Combine: write a_vals at first half and b_vals at second half
        # We need to store into two separate locations. Triton allows elementwise stores.
        # Here we store a_vals into Out[:, :M, h] and b_vals into Out[:, M:, h].
        # We implement by storing a_vals with mask for m_mask and zeros elsewhere, and b_vals similarly for n_mask.
        # But simpler: since m_offsets covers [tile_start : tile_start + BLOCK_M) and n_offsets covers [tile_start : tile_start + BLOCK_N),
        # we can compute out column index c_idx = tile_start + offs, and mask based on offs < M or offs >= M.
        offs = tl.arange(0, BLOCK_M + BLOCK_N)  # full block along C
        c_idx = tile_start + offs
        store_mask = c_idx < (M + N)
        is_m = c_idx < M
        is_n = ~is_m

        # Prepare values to store: zeros with a_vals at m positions and b_vals at n positions
        # Create 1D store values of length (BLOCK_M + BLOCK_N)
        # Initialize zeros
        store_vals = tl.zeros((BLOCK_M + BLOCK_N,), dtype=tl.float32)

        # For m part: index within m_offsets
        m_part = store_vals
        m_part = tl.where(m_mask, a_vals, m_part)  # but a_vals is [BLOCK_M], need to place at positions < M

        # More directly, we compute a_vals at m positions and zeros elsewhere using offs and is_m
        # Since a_vals is [BLOCK_M], and we want to store into Out at c_idx < M, we map a_vals to those positions:
        # We can compute per-lane: if is_m: a_vals[offs - tile_start], else 0.
        # However Triton doesn't support indexing a tensor with a tensor. So we do per-block:
        # We can't interleave m and n here; instead, we'll perform two stores: one for m, one for n.
        # For clarity, we'll store m part and n part separately via masked stores at the respective pointers.

        # Store m part: columns < M
        m_store_ptrs = Out_ptr + b * stride_ob + (tile_start + tl.arange(0, BLOCK_M)) * stride_oc + h * stride_oh
        m_store_mask = (tile_start + tl.arange(0, BLOCK_M)) < M
        # We need to store a_vals into Out[b, m, h] for m in m_offsets
        tl.store(m_store_ptrs, a_vals, mask=m_mask)

        # Store n part: columns >= M
        n_store_ptrs = Out_ptr + b * stride_ob + (tile_start + M + tl.arange(0, BLOCK_N)) * stride_oc + h * stride_oh
        n_store_mask = (tile_start + tl.arange(0, BLOCK_N)) < N
        tl.store(n_store_ptrs, b_vals, mask=n_store_mask)

    # Note: The above separate stores for m and n halves is correct and avoids tricky interleave construction.
    # The initial approach to vectorize across full C was incorrect due to tensor indexing limitations.
    # This revised kernel uses a 2D grid where each program handles a disjoint chunk of C, so we don't need to interleave m and n.


@triton.jit
def _batched_gemm_kernel(
    X_ptr,   # *f32, [B, C, K]
    W_ptr,   # *f32, [K, K]
    P_ptr,   # *f32, [B, C, K]
    B: tl.constexpr,   # batch size (constexpr)
    C: tl.constexpr,   # sequence length (M + N) (constexpr)
    K: tl.constexpr,   # hidden dim (constexpr)
    stride_xb,   # int: stride for batch in X
    stride_xc,   # int: stride for seq in X
    stride_xk,   # int: stride for hidden in X
    stride_w0,   # int: stride for dim 0 in W (rows)
    stride_w1,   # int: stride for dim 1 in W (cols)
    stride_pb,   # int: stride for batch in P
    stride_pc,   # int: stride for seq in P
    stride_pk,   # int: stride for hidden in P
    BLOCK_M: tl.constexpr,  # tile along C
    BLOCK_N: tl.constexpr,  # tile along K (output hidden)
    BLOCK_K: tl.constexpr,  # reduction tile along K
):
    # 2D grid over (batch, tiles along C). No tiling over K in the grid: loop over K.
    b = tl.program_id(0)
    m_block = tl.program_id(1)

    m_offsets = m_block * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M], sequence positions
    n_offsets = tl.arange(0, BLOCK_N)                     # [BLOCK_N], hidden positions
    k_offsets = tl.arange(0, BLOCK_K)                     # [BLOCK_K], reduction positions

    # Masks for boundaries
    mask_m = m_offsets < C
    mask_n = n_offsets < K

    # Initialize accumulator [BLOCK_M, BLOCK_N]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension in chunks of BLOCK_K
    for k0 in range(0, K, BLOCK_K):
        k_ids = k0 + k_offsets  # [BLOCK_K]
        k_mask = k_ids < K

        # Load X tile: [BLOCK_M, BLOCK_K] -> X[b, m, k]
        x_ptrs = X_ptr + b * stride_xb + m_offsets[:, None] * stride_xc + k_ids[None, :] * stride_xk
        x_mask = mask_m[:, None] & k_mask[None, :]
        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)  # [BLOCK_M, BLOCK_K]

        # Load W tile: [BLOCK_K, BLOCK_N] -> W[k, n]
        w_ptrs = W_ptr + k_ids[:, None] * stride_w0 + n_offsets[None, :] * stride_w1
        w_mask = k_mask[:, None] & mask_n[None, :]
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)  # [BLOCK_K, BLOCK_N]

        # Accumulate
        acc += tl.dot(x_tile, w_tile)  # [BLOCK_M, BLOCK_N]

    # Store result to P[b, m, n]
    p_ptrs = P_ptr + b * stride_pb + m_offsets[:, None] * stride_pc + n_offsets[None, :] * stride_pk
    store_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(p_ptrs, acc, mask=store_mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        Triton-optimized version:
        - Concatenates encoder_hidden_states and hidden_states along sequence dim via Triton.
        - Computes processed = X @ process_weight.T via Triton GEMM.
        - Splits outputs back into two streams.
        """
        # Ensure CUDA and contiguous, use float32 for consistency with typical default
        device = hidden_states.device
        assert device.type == 'cuda', "ModelNew requires CUDA tensors."

        B = hidden_states.shape[0]
        M = encoder_hidden_states.shape[1]
        N = hidden_states.shape[1]
        H = hidden_states.shape[2]
        # Ensure dtype is float32 for predictable behavior
        A = encoder_hidden_states.contiguous().to(torch.float32)
        Bn = hidden_states.contiguous().to(torch.float32)
        W = process_weight.contiguous().to(torch.float32)

        # Allocate output of concatenation [B, C, H]
        C = M + N
        Out = torch.empty((B, C, H), device=device, dtype=torch.float32)

        # Launch concatenation kernel
        BLOCK_M = 64
        BLOCK_N = 64
        grid_concat = (B, triton.cdiv(C, BLOCK_M + BLOCK_N))
        _concat_seq_kernel[grid_concat](
            A, Bn, Out,
            B=B, M=M, N=N, H=H,
            stride_ab=A.stride(0), stride_am=A.stride(1), stride_ah=A.stride(2),
            stride_bb=Bn.stride(0), stride_bn=Bn.stride(1), stride_bh=Bn.stride(2),
            stride_ob=Out.stride(0), stride_oc=Out.stride(1), stride_oh=Out.stride(2),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            num_warps=4, num_stages=3
        )

        # Now compute processed = Out @ W^T using Triton GEMM
        # P will be [B, C, H]
        P = torch.empty((B, C, H), device=device, dtype=torch.float32)

        # Strides
        stride_xb, stride_xc, stride_xk = Out.stride(0), Out.stride(1), Out.stride(2)
        stride_w0, stride_w1 = W.stride(0), W.stride(1)  # W is [K, K]
        stride_pb, stride_pc, stride_pk = P.stride(0), P.stride(1), P.stride(2)

        # GEMM launch config (tile sizes). Use moderate tiles for robustness.
        BLOCK_M_gemm = 64
        BLOCK_N_gemm = 64
        BLOCK_K_gemm = 64
        grid_gemm = (B, triton.cdiv(C, BLOCK_M_gemm))
        _batched_gemm_kernel[grid_gemm](
            Out, W, P,
            B=B, C=C, K=H,
            stride_xb=stride_xb, stride_xc=stride_xc, stride_xk=stride_xk,
            stride_w0=stride_w0, stride_w1=stride_w1,
            stride_pb=stride_pb, stride_pc=stride_pc, stride_pk=stride_pk,
            BLOCK_M=BLOCK_M_gemm, BLOCK_N=BLOCK_N_gemm, BLOCK_K=BLOCK_K_gemm,
            num_warps=4, num_stages=3
        )

        # Split along sequence dimension
        processed_encoder = P[:, :M, :]
        processed_hidden = P[:, M:, :]
        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
