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
    C = M + N

    # sequence offsets handled by this program
    seq_offsets = tile * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    mask_m = seq_offsets < C  # for stores

    # Determine if each seq index comes from A or B
    from_A = seq_offsets < M
    from_B = ~from_A

    # Hidden dimension offsets
    h_offsets = tl.arange(0, BLOCK_H)  # [BLOCK_H], we'll mask by h < H
    mask_h = h_offsets < H

    # Compute pointers and loads
    a_ptrs = A_ptr + b * stride_ab + (seq_offsets[:, None] - 0) * stride_am + h_offsets[None, :] * stride_ah
    b_ptrs = B_ptr + b * stride_bb + (seq_offsets[:, None] - M) * stride_bn + h_offsets[None, :] * stride_bh

    a_mask = mask_m[:, None] & mask_h[None, :]
    b_mask = mask_m[:, None] & mask_h[None, :]
    a_vals = tl.load(a_ptrs, mask=a_mask, other=0.0)
    b_vals = tl.load(b_ptrs, mask=b_mask, other=0.0)

    vals = tl.where(from_A[:, None], a_vals, b_vals)  # [BLOCK_M, BLOCK_H]

    # Store to Out
    out_ptrs = Out_ptr + b * stride_ob + seq_offsets[:, None] * stride_oc + h_offsets[None, :] * stride_oh
    out_mask = mask_m[:, None] & mask_h[None, :]
    tl.store(out_ptrs, vals, mask=out_mask)


@triton.jit
def _batched_matmul_kernel_tl_dot(
    X_ptr,        # *f32, [B, C, K], where C = M + N
    W_ptr,        # *f32, [K, K] (process_weight), note: we pass W.T (rows=K, cols=K)
    P_ptr,        # *f32, [B, C, K] output
    B: tl.constexpr,      # batch size (constexpr for specialization)
    C: tl.constexpr,      # sequence length
    K: tl.constexpr,      # hidden dim (constexpr)
    stride_xb,    # int: stride along batch for X
    stride_xc,    # int: stride along seq for X
    stride_xk,    # int: stride along hidden for X
    stride_w0,    # int: stride along dim 0 (rows) for W (which should be K)
    stride_w1,    # int: stride along dim 1 (cols) for W (which should be K)
    stride_pb,    # int: stride along batch for P
    stride_pc,    # int: stride along seq for P
    stride_pk,    # int: stride along hidden for P
    BLOCK_M: tl.constexpr,  # tile over C (sequences)
    BLOCK_N: tl.constexpr,  # tile over K (hidden dim)
    BLOCK_K: tl.constexpr,  # tile over reduction dim K
):
    # 3D grid: (B, ceil_div(C, BLOCK_M), ceil_div(K, BLOCK_N))
    b = tl.program_id(0)
    m_block = tl.program_id(1)
    n_block = tl.program_id(2)

    # tile offsets
    m_offsets = m_block * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    n_offsets = n_block * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]

    # masks for boundaries
    mask_m = m_offsets < C
    mask_n = n_offsets < K

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over reduction dimension K in chunks of BLOCK_K
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)  # [BLOCK_K]
        k_mask = k_offsets < K

        # Load X tile: [BLOCK_M, BLOCK_K]
        # X[b, m, k] -> pointers: X_ptr + b*stride_xb + m*stride_xc + k*stride_xk
        x_ptrs = X_ptr + b * stride_xb + m_offsets[:, None] * stride_xc + k_offsets[None, :] * stride_xk
        x_mask = mask_m[:, None] & k_mask[None, :]
        x_vals = tl.load(x_ptrs, mask=x_mask, other=0.0)  # [BLOCK_M, BLOCK_K]

        # Load W^T tile (W is [K, K], W^T has rows=k_offsets (K), cols=k_offsets (K)):
        # W^T[k, k'] -> W[k', k] -> W_ptr + k'*stride_w0 + k*stride_w1
        w_ptrs = W_ptr + k_offsets[:, None] * stride_w0 + k_offsets[None, :] * stride_w1
        w_mask = k_mask[:, None] & k_mask[None, :]
        w_vals = tl.load(w_ptrs, mask=w_mask, other=0.0)  # [BLOCK_K, BLOCK_K]

        # Accumulate
        # Note: x_vals: [BM, BK], w_vals: [BK, BK] (but we only use BK columns corresponding to current BK chunk)
        # We need w_vals as [BK, BN] for this tile. Here BK == BN (BLOCK_K == BLOCK_N). So:
        w_tile = w_vals[:, :BLOCK_N]  # keep leading BN columns
        acc += tl.dot(x_vals, w_tile)  # [BM, BN]

    # Store accumulated result to P: P[b, m, n]
    p_ptrs = P_ptr + b * stride_pb + m_offsets[:, None] * stride_pc + n_offsets[None, :] * stride_pk
    p_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(p_ptrs, acc, mask=p_mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton version of the original run function:
        - Concatenates encoder_hidden_states and hidden_states along sequence dimension in Triton.
        - Applies linear projection via Triton GEMM (X @ process_weight.T).
        - Splits the result back into encoder and image streams.
        Returns (processed_encoder, processed_hidden).
        """
        # Extract shapes
        B, M, H = encoder_hidden_states.shape
        B2, N, H2 = hidden_states.shape
        assert B == B2, "Batch size mismatch between encoder_hidden_states and hidden_states"
        assert H == H2, "Hidden dim mismatch between encoder_hidden_states and hidden_states"
        C = M + N

        # Allocate output tensor P [B, C, H]
        # Note: We need to return two outputs; here we compute the full processed [B, C, H]
        # and then slice. Allocation is necessary for return; we ensure it's contiguous.
        # We avoid any torch GPU ops in forward besides allocation, since tensors are already on CUDA.
        P = torch.empty((B, C, H), dtype=torch.float32, device=hidden_states.device)

        # Launch Triton concat kernel: write A into P[:, :M, :], B into P[:, M:, :]
        # Make sure inputs are contiguous (they should be from typical usage)
        stride_ab, stride_am, stride_ah = encoder_hidden_states.stride()
        stride_bb, stride_bn, stride_bh = hidden_states.stride()
        stride_pb, stride_pc, stride_ph = P.stride()

        # Choose tiling parameters for concat (conservative)
        BLOCK_M = 64
        BLOCK_H = 64
        grid_concat = (B, triton.cdiv(C, BLOCK_M))
        _concat_seq_kernel[grid_concat](
            encoder_hidden_states, hidden_states, P,
            B=B, M=M, N=N, H=H,
            stride_ab=stride_ab, stride_am=stride_am, stride_ah=stride_ah,
            stride_bb=stride_bb, stride_bn=stride_bn, stride_bh=stride_bh,
            stride_ob=stride_pb, stride_oc=stride_pc, stride_oh=stride_ph,
            BLOCK_M=BLOCK_M, BLOCK_H=BLOCK_H,
            num_warps=4, num_stages=3,
        )

        # GEMM: P = P @ process_weight.T using Triton tl.dot kernel
        # process_weight is [H, H]; we pass W.T as [H, H] without actual transpose (PyTorch view is fine),
        # but since forward must not perform torch GPU ops, we will feed process_weight directly and rely on Triton kernel.
        # Note: Triton kernel takes W as [K, K] and uses strides accordingly. We can pass process_weight directly as W_ptr.
        stride_xb, stride_xc, stride_xk = P.stride(0), P.stride(1), P.stride(2)
        # process_weight is [H, H]; we pass as W_ptr. We need W^T within kernel via strides, but kernel loads as W[k, k'] via strides.
        # Here, process_weight.T with strides: W_ptr = process_weight, stride_w0 = 1, stride_w1 = H for contiguous [H, H].
        # However, to be robust, we get strides from process_weight.t() view (no actual data move).
        Wt = process_weight.t()  # view, no copy; we provide strides
        stride_w0 = Wt.stride(0)  # typically 1 for contiguous along rows
        stride_w1 = Wt.stride(1)  # typically H for contiguous along cols

        stride_pb_out, stride_pc_out, stride_pk_out = P.stride(0), P.stride(1), P.stride(2)

        # Tile sizes for GEMM (modest, to avoid OOB and ensure correctness)
        BLOCK_M_gemm = 64
        BLOCK_N_gemm = 64
        BLOCK_K_gemm = 64

        grid_gemm = (B, triton.cdiv(C, BLOCK_M_gemm), triton.cdiv(H, BLOCK_N_gemm))
        _batched_matmul_kernel_tl_dot[grid_gemm](
            P, process_weight, P,  # output P is reused as result
            B=B, C=C, K=H,
            stride_xb=stride_xb, stride_xc=stride_xc, stride_xk=stride_xk,
            stride_w0=stride_w0, stride_w1=stride_w1,
            stride_pb=stride_pb_out, stride_pc=stride_pc_out, stride_pk=stride_pk_out,
            BLOCK_M=BLOCK_M_gemm, BLOCK_N=BLOCK_N_gemm, BLOCK_K=BLOCK_K_gemm,
            num_warps=4, num_stages=3,
        )

        # Split P into encoder and hidden parts
        processed_encoder = P[:, :M, :]
        processed_hidden = P[:, M:, :]

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
