import torch
import triton
import triton.language as tl


@triton.jit
def _concat_seq_kernel_A(
    A_ptr,        # *f32, [B, M, H]
    Out_ptr,      # *f32, [B, C, H], we write into first M rows (i.e., rows < M)
    B: tl.constexpr,    # batch size (constexpr for specialization)
    M: tl.constexpr,    # text_seq_len (constexpr)
    H: tl.constexpr,    # hidden_dim (constexpr)
    stride_ab,    # int: stride along batch for A
    stride_am,    # int: stride along seq for A
    stride_ah,    # int: stride along hidden for A
    stride_ob,    # int: stride along batch for Out
    stride_oc,    # int: stride along seq for Out
    stride_oh,    # int: stride along hidden for Out
    BLOCK_M: tl.constexpr,  # tile along sequence
    BLOCK_N: tl.constexpr,  # tile along hidden (not used in this kernel, but kept for consistency)
):
    # grid: (B, ceil_div(M, BLOCK_M))
    b = tl.program_id(0)
    m_block = tl.program_id(1)

    m_offsets = m_block * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    mask_m = m_offsets < M

    # For each hidden dim element, copy from A into Out[:, m, h]
    for h in range(0, H):
        a_ptrs = A_ptr + b * stride_ab + m_offsets * stride_am + h * stride_ah
        a_vals = tl.load(a_ptrs, mask=mask_m, other=0.0)  # shape [BLOCK_M]
        out_ptrs = Out_ptr + b * stride_ob + m_offsets * stride_oc + h * stride_oh
        tl.store(out_ptrs, a_vals, mask=mask_m)


@triton.jit
def _concat_seq_kernel_B(
    B_ptr,        # *f32, [B, N, H]
    Out_ptr,      # *f32, [B, C, H], we write into rows M..M+N (i.e., rows >= M)
    B: tl.constexpr,    # batch size (constexpr)
    N: tl.constexpr,    # img_seq_len (constexpr)
    H: tl.constexpr,    # hidden_dim (constexpr)
    stride_bb,    # int: stride along batch for B
    stride_bn,    # int: stride along seq for B
    stride_bh,    # int: stride along hidden for B
    stride_ob,    # int: stride along batch for Out
    stride_oc,    # int: stride along seq for Out
    stride_oh,    # int: stride along hidden for Out
    BLOCK_M: tl.constexpr,  # tile along sequence
    BLOCK_N: tl.constexpr,  # tile along hidden (not used in this kernel)
):
    # grid: (B, ceil_div(N, BLOCK_M))
    b = tl.program_id(0)
    n_block = tl.program_id(1)

    n_offsets = n_block * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    mask_n = n_offsets < N

    # For each hidden dim element, copy from B into Out[:, M + n, h]
    for h in range(0, H):
        b_ptrs = B_ptr + b * stride_bb + n_offsets * stride_bn + h * stride_bh
        b_vals = tl.load(b_ptrs, mask=mask_n, other=0.0)  # shape [BLOCK_M]
        out_ptrs = Out_ptr + b * stride_ob + (n_offsets + M) * stride_oc + h * stride_oh
        tl.store(out_ptrs, b_vals, mask=mask_n)


@triton.jit
def _batched_matmul_kernel(
    X_ptr,   # *f32, [B, C, K] where C = M + N, K = H
    W_ptr,   # *f32, [K, K] (process_weight), row-major
    P_ptr,   # *f32, [B, C, K]
    B: tl.constexpr,   # batch size (constexpr)
    C: tl.constexpr,   # sequence length (M + N)
    K: tl.constexpr,   # hidden dim (constexpr)
    stride_xb, stride_xc, stride_xk,   # strides for X
    stride_w0, stride_w1,              # strides for W [K, K]
    stride_pb, stride_pc, stride_pk,   # strides for P
    BLOCK_M: tl.constexpr,  # tile over C (seq)
    BLOCK_N: tl.constexpr,  # tile over K (hidden)
    BLOCK_K: tl.constexpr,  # tile over reduction dim
):
    # Grid: (B, ceil_div(C, BLOCK_M), ceil_div(K, BLOCK_N))
    b = tl.program_id(0)
    c_block = tl.program_id(1)
    k_block = tl.program_id(2)

    # Offsets
    c_offsets = c_block * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    k_offsets = k_block * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]
    mask_c = c_offsets < C
    mask_k = k_offsets < K

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over reduction dimension K in chunks
    for k0 in range(0, K, BLOCK_K):
        k_ids = k0 + tl.arange(0, BLOCK_K)  # [BLOCK_K]
        k_mask = k_ids < K

        # Load X tile: [BLOCK_M, BLOCK_K] -> X[b, c, k]
        x_ptrs = X_ptr + b * stride_xb + c_offsets[:, None] * stride_xc + k_ids[None, :] * stride_xk
        x_mask = mask_c[:, None] & k_mask[None, :]
        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)  # [BLOCK_M, BLOCK_K]

        # Load W tile as W^T: we need W[k, k'] but in memory, W[k', k] -> [BLOCK_K, BLOCK_N]
        w_ptrs = W_ptr + k_ids[:, None] * stride_w0 + k_offsets[None, :] * stride_w1  # [BLOCK_K, BLOCK_N]
        w_mask = k_mask[:, None] & mask_k[None, :]
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)  # [BLOCK_K, BLOCK_N]

        # Accumulate
        acc += tl.dot(x_tile, w_tile)  # [BLOCK_M, BLOCK_N]

    # Store result
    p_ptrs = P_ptr + b * stride_pb + c_offsets[:, None] * stride_pc + k_offsets[None, :] * stride_pk
    p_mask = mask_c[:, None] & mask_k[None, :]
    tl.store(p_ptrs, acc, mask=p_mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-only implementation:
        - Concatenate encoder_hidden_states and hidden_states along sequence dim (Triton).
        - Apply linear projection (X @ process_weight.T) via Triton GEMM kernel.
        - Split outputs into (processed_encoder, processed_hidden).
        """
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Inputs must be CUDA tensors"
        assert hidden_states.dim() == 3 and encoder_hidden_states.dim() == 3, "Inputs must be 3D: [B, L, H]"
        B = hidden_states.shape[0]
        M = encoder_hidden_states.shape[1]
        N = hidden_states.shape[1]
        H = hidden_states.shape[2]
        assert encoder_hidden_states.shape[2] == H, "Hidden dims must match"
        assert process_weight.shape == (H, H), "process_weight must be [H, H]"

        # Ensure contiguity
        A = encoder_hidden_states.contiguous()
        B_t = hidden_states.contiguous()
        W = process_weight.contiguous()  # [H, H]

        # Allocate concatenated X [B, C, H], C = M + N
        C = M + N
        Out = torch.empty((B, C, H), device=hidden_states.device, dtype=torch.float32)

        # Launch Triton kernels for concatenation
        BLOCK = 128  # tile size along sequence for copies
        grid_A = (B, triton.cdiv(M, BLOCK))
        grid_B = (B, triton.cdiv(N, BLOCK))

        _concat_seq_kernel_A[grid_A](
            A, Out,
            B=B, M=M, H=H,
            stride_ab=A.stride(0), stride_am=A.stride(1), stride_ah=A.stride(2),
            stride_ob=Out.stride(0), stride_oc=Out.stride(1), stride_oh=Out.stride(2),
            BLOCK_M=BLOCK, BLOCK_N=64,
            num_warps=4, num_stages=2
        )

        _concat_seq_kernel_B[grid_B](
            B_t, Out,
            B=B, N=N, H=H,
            stride_bb=B_t.stride(0), stride_bn=B_t.stride(1), stride_bh=B_t.stride(2),
            stride_ob=Out.stride(0), stride_oc=Out.stride(1), stride_oh=Out.stride(2),
            BLOCK_M=BLOCK, BLOCK_N=64,
            num_warps=4, num_stages=2
        )

        # Allocate output for projection
        P = torch.empty((B, C, H), device=hidden_states.device, dtype=torch.float32)

        # Launch Triton GEMM: P = Out @ W^T
        BLOCK_M_GEMM = 64
        BLOCK_N_GEMM = 64
        BLOCK_K_GEMM = 64
        grid = (B, triton.cdiv(C, BLOCK_M_GEMM), triton.cdiv(H, BLOCK_N_GEMM))
        _batched_matmul_kernel[grid](
            Out, W, P,
            B=B, C=C, K=H,
            stride_xb=Out.stride(0), stride_xc=Out.stride(1), stride_xk=Out.stride(2),
            stride_w0=W.stride(0), stride_w1=W.stride(1),
            stride_pb=P.stride(0), stride_pc=P.stride(1), stride_pk=P.stride(2),
            BLOCK_M=BLOCK_M_GEMM, BLOCK_N=BLOCK_N_GEMM, BLOCK_K=BLOCK_K_GEMM,
            num_warps=4, num_stages=2
        )

        # Split outputs: processed_encoder [B, M, H], processed_hidden [B, N, H]
        processed_encoder = P[:, :M, :]
        processed_hidden = P[:, M:, :]
        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
