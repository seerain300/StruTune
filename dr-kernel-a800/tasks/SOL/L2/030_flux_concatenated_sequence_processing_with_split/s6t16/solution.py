import torch
import triton
import triton.language as tl


@triton.jit
def _concat_seq_kernel(
    A_ptr,        # *f32, [B, M, H]
    B_ptr,        # *f32, [B, N, H]
    Out_ptr,      # *f32, [B, C, H], C = M + N
    M: tl.constexpr,     # text_seq_len
    N: tl.constexpr,     # img_seq_len
    H: tl.constexpr,     # hidden_dim
    stride_ab,    # int: stride for batch in A
    stride_am,    # int: stride for seq in A
    stride_ah,    # int: stride for hidden in A
    stride_bb,    # int: stride for batch in B
    stride_bn,    # int: stride for seq in B
    stride_bh,    # int: stride for hidden in B
    stride_ob,    # int: stride for batch in Out
    stride_oc,    # int: stride for seq in Out
    stride_oh,    # int: stride for hidden in Out
    BLOCK_M: tl.constexpr,  # tile over M (unused directly, loop over sequences)
    BLOCK_N: tl.constexpr,  # tile over N
):
    # Grid: (B, ceil_div(M, BLOCK_M), ceil_div(N, BLOCK_N))
    b = tl.program_id(0)
    m_block = tl.program_id(1)
    n_block = tl.program_id(2)

    # Process sequences along M
    for mi in range(0, M):
        # Determine source: from A if mi < M else from B (mi >= M corresponds to n = mi - M)
        from_A = True  # always A for m < M

        # Compute pointers for A row
        a_ptrs = A_ptr + b * stride_ab + mi * stride_am + tl.arange(0, H) * stride_ah  # [H]
        vals = tl.load(a_ptrs)

        # Compute destination index in Out
        dest = mi
        out_ptrs = Out_ptr + b * stride_ob + dest * stride_oc + tl.arange(0, H) * stride_oh
        tl.store(out_ptrs, vals)

    # Process sequences along N (shifted part)
    for nj in range(0, N):
        dest = M + nj
        b_ptrs = B_ptr + b * stride_bb + nj * stride_bn + tl.arange(0, H) * stride_bh
        vals = tl.load(b_ptrs)
        out_ptrs = Out_ptr + b * stride_ob + dest * stride_oc + tl.arange(0, H) * stride_oh
        tl.store(out_ptrs, vals)


@triton.jit
def _batched_matmul_kernel(
    X_ptr,      # *f32, [B, C, K], C = M + N, reduction over K
    W_ptr,      # *f32, [K, K]
    P_ptr,      # *f32, [B, C, K]
    B: tl.constexpr,      # batch size (grid dim 0 handles it)
    C: tl.constexpr,      # sequence length (M + N)
    K: tl.constexpr,      # hidden_dim (must be constexpr)
    stride_xb,  # int: stride along batch for X
    stride_xc,  # int: stride along seq for X
    stride_xk,  # int: stride along hidden for X
    stride_w0,  # int: stride along dim 0 (rows, corresponds to K_in)
    stride_w1,  # int: stride along dim 1 (cols, corresponds to K_out)
    stride_pb,  # int: stride along batch for P
    stride_pc,  # int: stride along seq for P
    stride_pk,  # int: stride along hidden for P
    BLOCK_M: tl.constexpr,  # tile over C (sequences)
    BLOCK_N: tl.constexpr,  # tile over K (output features)
    BLOCK_K: tl.constexpr,  # tile over reduction (input features)
):
    # 2D grid: (batch, tiles along C)
    pid_b = tl.program_id(0)
    pid_cm = tl.program_id(1)

    # Tile coordinates
    m_offsets = pid_cm * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    n_offsets = tl.arange(0, BLOCK_N)                    # [BLOCK_N]

    # Masks for boundaries
    mask_m = m_offsets < C
    mask_n = n_offsets < K

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over reduction dimension K in chunks
    for k0 in range(0, K, BLOCK_K):
        k_ids = k0 + tl.arange(0, BLOCK_K)  # [BLOCK_K]
        mask_k = k_ids < K

        # Load X tile: [BLOCK_M, BLOCK_K]
        x_ptrs = X_ptr + pid_b * stride_xb + m_offsets[:, None] * stride_xc + k_ids[None, :] * stride_xk
        x_mask = mask_m[:, None] & mask_k[None, :]
        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)  # [BLOCK_M, BLOCK_K]

        # Load W tile: [BLOCK_K, BLOCK_N], W is [K, K] -> rows=k_ids, cols=n_offsets
        w_ptrs = W_ptr + k_ids[:, None] * stride_w0 + n_offsets[None, :] * stride_w1
        w_mask = mask_k[:, None] & mask_n[None, :]
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)  # [BLOCK_K, BLOCK_N]

        # Accumulate
        acc += tl.dot(x_tile, w_tile)  # [BLOCK_M, BLOCK_N]

    # Store result to P
    p_ptrs = P_ptr + pid_b * stride_pb + m_offsets[:, None] * stride_pc + n_offsets[None, :] * stride_pk
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
        - Concatenates encoder_hidden_states and hidden_states along sequence dimension using Triton.
        - Applies linear projection (X @ process_weight.T) using a Triton GEMM kernel.
        - Splits the result back into two streams and returns.
        """
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "All inputs must be CUDA tensors."
        assert hidden_states.dim() == 3 and encoder_hidden_states.dim() == 3 and process_weight.dim() == 2, "Invalid input shapes."
        B = hidden_states.shape[0]
        M = encoder_hidden_states.shape[1]
        N = hidden_states.shape[1]
        H = hidden_states.shape[2]
        assert encoder_hidden_states.shape[2] == H and process_weight.shape[0] == H and process_weight.shape[1] == H, "Mismatched hidden dimensions."

        # Ensure contiguous tensors
        A = encoder_hidden_states.contiguous()
        B_b = hidden_states.contiguous()
        W = process_weight.contiguous()

        # Allocate output of concatenation [B, C, H], C = M + N
        C = M + N
        Out = torch.empty((B, C, H), device=hidden_states.device, dtype=torch.float32)

        # Launch Triton concat kernel
        grid_concat = (B, 1, 1)  # simple linearization; we loop over M and N inside kernel
        _concat_seq_kernel[grid_concat](
            A, B_b, Out,
            M, N, H,
            A.stride(0), A.stride(1), A.stride(2),
            B_b.stride(0), B_b.stride(1), B_b.stride(2),
            Out.stride(0), Out.stride(1), Out.stride(2),
            BLOCK_M=64, BLOCK_N=64,
            num_warps=4, num_stages=2,
        )

        # Allocate output of GEMM [B, C, H]
        P = torch.empty((B, C, H), device=hidden_states.device, dtype=torch.float32)

        # Launch Triton GEMM kernel: compute P = Out @ W^T
        grid_gemm = (B, triton.cdiv(C, 128))
        _batched_matmul_kernel[grid_gemm](
            Out, W, P,
            B, C, H,
            Out.stride(0), Out.stride(1), Out.stride(2),
            W.stride(0), W.stride(1),
            P.stride(0), P.stride(1), P.stride(2),
            BLOCK_M=128, BLOCK_N=128, BLOCK_K=64,
            num_warps=8, num_stages=3,
        )

        # Split back along sequence dimension
        processed_encoder = P[:, :M, :]
        processed_hidden = P[:, M:, :]
        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
