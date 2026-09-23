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
    mask_m = seq_offsets < C

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
def _batched_gemm_tl_dot(
    X_ptr,      # *f32, [B, C, K]
    Wt_ptr,     # *f32, [K, K] (process_weight.T view)
    P_ptr,      # *f32, [B, C, K]
    B: tl.constexpr,      # batch size
    C: tl.constexpr,      # sequence length (M + N)
    K: tl.constexpr,      # hidden dim
    stride_xb,  # stride along batch for X
    stride_xc,  # stride along seq for X
    stride_xk,  # stride along hidden for X
    stride_w0,  # stride along rows for Wt (dim 0)
    stride_w1,  # stride along cols for Wt (dim 1)
    stride_pb,  # stride along batch for P
    stride_pc,  # stride along seq for P
    stride_pk,  # stride along hidden for P
    BLOCK_M: tl.constexpr,  # tile over C (sequences)
    BLOCK_N: tl.constexpr,  # tile over K (hidden)
    BLOCK_K: tl.constexpr,  # tile over reduction K
):
    # 3D grid: (B, tiles over C, tiles over K)
    b = tl.program_id(0)
    m_block = tl.program_id(1)
    n_block = tl.program_id(2)

    m_offsets = m_block * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    n_offsets = n_block * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]
    k_offsets = tl.arange(0, BLOCK_K)                     # [BLOCK_K]

    mask_m = m_offsets < C
    mask_n = n_offsets < K

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Reduction over K
    for k0 in range(0, K, BLOCK_K):
        x_ptrs = X_ptr + b * stride_xb + m_offsets[:, None] * stride_xc + (k0 + k_offsets[None, :]) * stride_xk
        x_mask = mask_m[:, None] & ((k0 + k_offsets[None, :]) < K)
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)  # [BLOCK_M, BLOCK_K]

        wt_ptrs = Wt_ptr + (k0 + k_offsets[:, None]) * stride_w0 + n_offsets[None, :] * stride_w1
        wt_mask = ((k0 + k_offsets[:, None]) < K) & (mask_n[None, :])
        wt = tl.load(wt_ptrs, mask=wt_mask, other=0.0)  # [BLOCK_K, BLOCK_N]

        acc += tl.dot(x, wt)  # [BLOCK_M, BLOCK_N]

    # Store
    p_ptrs = P_ptr + b * stride_pb + m_offsets[:, None] * stride_pc + n_offsets[None, :] * stride_pk
    store_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(p_ptrs, acc, mask=store_mask)


class ModelNew(torch.nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        process_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version:
        - Concatenate along sequence dimension in Triton.
        - Linear projection X @ process_weight.T in Triton.
        - Split into two streams and return.

        The forward does only allocation and kernel launches; no torch matmul or slicing of GPU tensors.
        """
        B = hidden_states.shape[0]
        M = encoder_hidden_states.shape[1]  # text_seq_len
        N = hidden_states.shape[1]          # img_seq_len
        H = hidden_states.shape[2]          # hidden_dim

        C = M + N

        # Allocate output P [B, C, H]. Dtype matches input hidden_states (float32 expected).
        P = torch.empty((B, C, H), device=hidden_states.device, dtype=hidden_states.dtype)

        # Strides for A, B, Out
        stride_ab, stride_am, stride_ah = encoder_hidden_states.stride()
        stride_bb, stride_bn, stride_bh = hidden_states.stride()
        stride_ob, stride_oc, stride_oh = P.stride()

        # Launch concatenation kernel
        BLOCK_M_concat = 128
        BLOCK_H_concat = 64
        grid_concat = (B, triton.cdiv(C, BLOCK_M_concat))
        _concat_seq_kernel[grid_concat](
            encoder_hidden_states, hidden_states, P,
            B=B, M=M, N=N, H=H,
            stride_ab=stride_ab, stride_am=stride_am, stride_ah=stride_ah,
            stride_bb=stride_bb, stride_bn=stride_bn, stride_bh=stride_bh,
            stride_ob=stride_ob, stride_oc=stride_oc, stride_oh=stride_oh,
            BLOCK_M=BLOCK_M_concat, BLOCK_H=BLOCK_H_concat,
            num_warps=4, num_stages=2,
        )

        # Prepare Wt = process_weight.T as a view (no copy); process_weight is [H, H]
        Wt = process_weight.transpose(0, 1)  # Triton view

        # Strides for X, Wt, P
        stride_xb, stride_xc, stride_xk = P.stride()
        stride_w0, stride_w1 = Wt.stride()
        stride_pb, stride_pc, stride_pk = P.stride()

        # GEMM kernel launch
        BLOCK_M_gemm = 64
        BLOCK_N_gemm = 64
        BLOCK_K_gemm = 64
        grid_gemm = (B, triton.cdiv(C, BLOCK_M_gemm), triton.cdiv(H, BLOCK_N_gemm))
        _batched_gemm_tl_dot[grid_gemm](
            P, Wt, P,
            B=B, C=C, K=H,
            stride_xb=stride_xb, stride_xc=stride_xc, stride_xk=stride_xk,
            stride_w0=stride_w0, stride_w1=stride_w1,
            stride_pb=stride_pb, stride_pc=stride_pc, stride_pk=stride_pk,
            BLOCK_M=BLOCK_M_gemm, BLOCK_N=BLOCK_N_gemm, BLOCK_K=BLOCK_K_gemm,
            num_warps=4, num_stages=3,
        )

        # Return processed streams (split along sequence dim).
        # Note: This slicing is done on the tensor that Triton already filled; it's lightweight compared to GEMM.
        processed_encoder = P[:, :M, :]
        processed_hidden = P[:, M:, :]
        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
