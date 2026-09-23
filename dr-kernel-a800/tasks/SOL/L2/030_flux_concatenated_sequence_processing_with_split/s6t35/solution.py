import torch
import triton
import triton.language as tl


@triton.jit
def _batched_matmul_kernel_2d(
    X_ptr,  # *f32, [B, C, K], where C = M + N, K = H
    W_ptr,  # *f32, [K, K] (process_weight)
    P_ptr,  # *f32, [B, C, K]
    B: tl.constexpr,      # batch size (constexpr for specialization)
    C: tl.constexpr,      # sequence length
    K: tl.constexpr,      # hidden dim (constexpr)
    stride_xb,  # int: stride along batch in X
    stride_xc,  # int: stride along seq in X
    stride_xk,  # int: stride along hidden in X
    stride_w0,  # int: stride along dim 0 (rows) in W
    stride_w1,  # int: stride along dim 1 (cols) in W
    stride_pb,  # int: stride along batch in P
    stride_pc,  # int: stride along seq in P
    stride_pk,  # int: stride along hidden in P
    BLOCK_M: tl.constexpr,  # tile over C
    BLOCK_N: tl.constexpr,  # tile over K
    BLOCK_K: tl.constexpr,  # tile over reduction dim
):
    # 2D grid: (B, ceil_div(C, BLOCK_M))
    b = tl.program_id(0)
    c_block = tl.program_id(1)

    # Tile offsets
    c_offsets = c_block * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    n_offsets = tl.arange(0, BLOCK_N)                     # [BLOCK_N]

    # Masks
    mask_c = c_offsets < C
    mask_n = n_offsets < K

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over reduction dimension K in chunks
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)  # [BLOCK_K]
        mask_k = k_offsets < K

        # Load X tile: X[b, c, k] -> shape [BLOCK_M, BLOCK_K]
        x_ptrs = X_ptr + b * stride_xb + c_offsets[:, None] * stride_xc + k_offsets[None, :] * stride_xk
        x_mask = mask_c[:, None] & mask_k[None, :]
        x_vals = tl.load(x_ptrs, mask=x_mask, other=0.0)  # [BLOCK_M, BLOCK_K], f32

        # Load W tile as W^T: we need W[k, n] -> shape [BLOCK_K, BLOCK_N]
        w_ptrs = W_ptr + k_offsets[:, None] * stride_w0 + n_offsets[None, :] * stride_w1
        w_mask = mask_k[:, None] & mask_n[None, :]
        w_vals = tl.load(w_ptrs, mask=w_mask, other=0.0)  # [BLOCK_K, BLOCK_N], f32

        # Accumulate in float32
        acc += tl.dot(x_vals, w_vals)

    # Store acc to P[b, c, n]
    p_ptrs = P_ptr + b * stride_pb + c_offsets[:, None] * stride_pc + n_offsets[None, :] * stride_pk
    tl.store(p_ptrs, acc, mask=(mask_c[:, None] & mask_n[None, :]))


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version of the original run function:
        - Avoids torch.cat and torch.matmul in forward.
        - Uses Triton kernel to compute concatenated sequences @ process_weight.T.
        - Splits the result back into encoder and hidden streams.
        Returns (processed_encoder_hidden_states, processed_hidden_states).
        """
        # Ensure CUDA and contiguous
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "All tensors must be on CUDA"

        B, M, H = encoder_hidden_states.shape
        B2, N, H2 = hidden_states.shape
        assert B == B2 and H == H2, "Batch and hidden_dim must match across inputs"
        C = M + N

        # Cast to float32 to ensure consistent accumulation and match PyTorch default behavior
        A = encoder_hidden_states.contiguous().to(torch.float32)  # [B, M, H]
        Bn = hidden_states.contiguous().to(torch.float32)         # [B, N, H]
        W = process_weight.contiguous().to(torch.float32)         # [H, H]

        # Allocate output for concatenation X [B, C, H]
        X = torch.empty((B, C, H), dtype=torch.float32, device=A.device)

        # Build X by concatenation: first M rows from A, next N rows from Bn
        # We'll write into X directly using masks to ensure correctness.
        # Define kernel launch for concatenation with conservative tiles
        BLOCK_M, BLOCK_N = 64, 64
        grid_concat = (B, triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        # NOTE: A and Bn are both float32; concatenation is a simple copy. We can implement it as two stores into X:
        # First part: Out[b, 0:M, h] = A[b, :, :]
        # Second part: Out[b, M:M+N, h] = Bn[b, :, :]
        # We'll implement this as a single kernel that writes both halves for robustness.

        # Triton kernel to fill X: writes both halves
        @triton.jit
        def _fill_X_two_sources(
            A_ptr, B_ptr, Out_ptr,
            B: tl.constexpr, M: tl.constexpr, N: tl.constexpr, H: tl.constexpr,
            stride_ab, stride_am, stride_ah,
            stride_bb, stride_bn, stride_bh,
            stride_ob, stride_oc, stride_oh,
            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
        ):
            b = tl.program_id(0)
            m_block = tl.program_id(1)
            n_block = tl.program_id(2)

            m_offsets = m_block * BLOCK_M + tl.arange(0, BLOCK_M)
            n_offsets = n_block * BLOCK_N + tl.arange(0, BLOCK_N)
            h_idx = tl.arange(0, H)

            mask_m = m_offsets < M
            mask_n = n_offsets < N
            mask_h = h_idx < H

            # Load A tile: [BLOCK_M, H]
            a_ptrs = A_ptr + b * stride_ab + m_offsets[:, None] * stride_am + h_idx[None, :] * stride_ah
            a_mask = mask_m[:, None] & mask_h[None, :]
            a_vals = tl.load(a_ptrs, mask=a_mask, other=0.0)

            # Load B tile: [BLOCK_N, H]
            b_ptrs = B_ptr + b * stride_bb + n_offsets[:, None] * stride_bn + h_idx[None, :] * stride_bh
            b_mask = mask_n[:, None] & mask_h[None, :]
            b_vals = tl.load(b_ptrs, mask=b_mask, other=0.0)

            # Store first half: Out[b, m, h]
            out_ptrs_m = Out_ptr + b * stride_ob + m_offsets[:, None] * stride_oc + h_idx[None, :] * stride_oh
            tl.store(out_ptrs_m, a_vals, mask=(mask_m[:, None] & mask_h[None, :]))

            # Store second half shifted by M: Out[b, m + N, h]
            out_ptrs_n = Out_ptr + b * stride_ob + (n_offsets + M)[:, None] * stride_oc + h_idx[None, :] * stride_oh
            tl.store(out_ptrs_n, b_vals, mask=(mask_n[:, None] & mask_h[None, :]))

        # Launch concatenation
        grid_concat = (B, triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        _fill_X_two_sources[grid_concat](
            A, Bn, X,
            B=B, M=M, N=N, H=H,
            stride_ab=A.stride(0), stride_am=A.stride(1), stride_ah=A.stride(2),
            stride_bb=Bn.stride(0), stride_bn=Bn.stride(1), stride_bh=Bn.stride(2),
            stride_ob=X.stride(0), stride_oc=X.stride(1), stride_oh=X.stride(2),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        )

        # Allocate output for GEMM P [B, C, H]
        P = torch.empty((B, C, H), dtype=torch.float32, device=A.device)

        # Launch GEMM Triton kernel: 2D grid over (B, tiles along C)
        BLOCK_M_B, BLOCK_N_B, BLOCK_K_B = 64, 64, 64
        grid_gemm = (B, triton.cdiv(C, BLOCK_M_B))
        _batched_matmul_kernel_2d[grid_gemm](
            X, W, P,
            B=B, C=C, K=H,
            stride_xb=X.stride(0), stride_xc=X.stride(1), stride_xk=X.stride(2),
            stride_w0=W.stride(0), stride_w1=W.stride(1),
            stride_pb=P.stride(0), stride_pc=P.stride(1), stride_pk=P.stride(2),
            BLOCK_M=BLOCK_M_B, BLOCK_N=BLOCK_N_B, BLOCK_K=BLOCK_K_B,
            num_warps=4, num_stages=2,
        )

        # Split outputs along sequence dimension
        processed_encoder = P[:, :M, :]
        processed_hidden = P[:, M:, :]
        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
