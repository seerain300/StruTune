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
    BLOCK_M: tl.constexpr,  # tile size along sequence (M+N)
):
    # Grid is (B, tiles along (M+N))
    b = tl.program_id(0)
    m_block = tl.program_id(1)

    # Offsets for this tile
    m_offsets = m_block * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    C = M + N
    mask_m = m_offsets < C  # valid sequence positions

    # Determine which positions come from A and which from B
    from_A = m_offsets < M  # True for first M, False for last N

    # Compute hidden offsets 0..H-1
    h = tl.arange(0, H)  # [H]
    # For A: indices m_offsets, hidden dim h
    a_ptrs = A_ptr + b * stride_ab + m_offsets[:, None] * stride_am + h[None, :] * stride_ah
    a_mask = (mask_m[:, None]) & (h[None, :] < H)
    a_tile = tl.load(a_ptrs, mask=a_mask, other=0.0)  # [BLOCK_M, H]

    # For B: indices (m_offsets - M), hidden dim h
    b_ptrs = B_ptr + b * stride_bb + (m_offsets - M)[:, None] * stride_bn + h[None, :] * stride_bh
    b_mask = (~from_A[:, None]) & (mask_m[:, None]) & (h[None, :] < H)
    b_tile = tl.load(b_ptrs, mask=b_mask, other=0.0)  # [BLOCK_M, H]

    # Select based on from_A: broadcast to [BLOCK_M, H]
    vals = tl.where(from_A[:, None], a_tile, b_tile)  # [BLOCK_M, H]

    # Store to Out at position (b, m_offsets, h)
    out_ptrs = Out_ptr + b * stride_ob + m_offsets[:, None] * stride_oc + h[None, :] * stride_oh
    # mask across m and h
    tl.store(out_ptrs, vals, mask=(mask_m[:, None] & (h[None, :] < H)))


@triton.jit
def _batched_gemm_kernel(
    X_ptr,  # *f32, [B, C, K] where C = M + N, K = H
    W_ptr,  # *f32, [K, K] (process_weight)
    P_ptr,  # *f32, [B, C, K]
    B: tl.constexpr,      # batch size (grid dim 0)
    C: tl.constexpr,      # sequence length (M + N)
    K: tl.constexpr,      # hidden dim (constexpr)
    # Strides for X: we treat X as (B, C, K)
    stride_xb,  # int: stride along batch
    stride_xc,  # int: stride along seq
    stride_xk,  # int: stride along hidden
    # Strides for W: [K, K]
    stride_w0,  # int: stride along dim 0 (rows, corresponds to input K)
    stride_w1,  # int: stride along dim 1 (cols, corresponds to output K)
    # Strides for P: [B, C, K]
    stride_pb,  # int
    stride_pc,  # int
    stride_pk,  # int
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

    # Loop over reduction dimension K in chunks of BLOCK_K
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

    # Store result tile
    p_ptrs = P_ptr + pid_b * stride_pb + m_offsets[:, None] * stride_pc + n_offsets[None, :] * stride_pk
    store_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(p_ptrs, acc, mask=store_mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Shapes: hidden_states [B, N, H], encoder_hidden_states [B, M, H], process_weight [H, H]
        B = hidden_states.shape[0]
        M = encoder_hidden_states.shape[1]  # text_seq_len
        N = hidden_states.shape[1]          # img_seq_len
        H = hidden_states.shape[2]

        # Ensure CUDA and contiguous
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Tensors must be on CUDA device."
        hidden_states = hidden_states.contiguous()
        encoder_hidden_states = encoder_hidden_states.contiguous()
        process_weight = process_weight.contiguous()

        # 1) Concatenate along sequence dimension using Triton: Out [B, C, H], C = M + N
        C = M + N
        Out = torch.empty((B, C, H), device=hidden_states.device, dtype=torch.float32)

        # Launch concat kernel: grid over (B, tiles along C)
        grid_concat = (B, triton.cdiv(C, 128))
        _concat_seq_kernel[grid_concat](
            encoder_hidden_states, hidden_states, Out,
            B, M, N, H,
            encoder_hidden_states.stride(0), encoder_hidden_states.stride(1), encoder_hidden_states.stride(2),
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            Out.stride(0), Out.stride(1), Out.stride(2),
            BLOCK_M=128,
            num_warps=4,
        )

        # 2) Compute P = Out @ process_weight.T using Triton GEMM: P [B, C, H]
        P = torch.empty((B, C, H), device=hidden_states.device, dtype=torch.float32)

        # Grid over (batch, tiles along C)
        grid_gemm = (B, triton.cdiv(C, 128))
        _batched_gemm_kernel[grid_gemm](
            Out, process_weight, P,
            B, C, H,
            Out.stride(0), Out.stride(1), Out.stride(2),
            # Strides for W: [H, H]
            process_weight.stride(0), process_weight.stride(1),
            P.stride(0), P.stride(1), P.stride(2),
            BLOCK_M=128, BLOCK_N=128, BLOCK_K=64,
            num_warps=4,
        )

        # 3) Split P along sequence dimension
        processed_encoder = P[:, :M, :]
        processed_hidden = P[:, M:, :]

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
