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
    stride_ab,    # int: stride for A along batch
    stride_am,    # int: stride for A along seq (M)
    stride_ah,    # int: stride for A along hidden (H)
    stride_bb,    # int: stride for B along batch
    stride_bn,    # int: stride for B along seq (N)
    stride_bh,    # int: stride for B along hidden (H)
    stride_ob,    # int: stride for Out along batch
    stride_oc,    # int: stride for Out along seq (C)
    stride_oh,    # int: stride for Out along hidden (H)
    BLOCK_M: tl.constexpr,  # tile size for sequences
    BLOCK_N: tl.constexpr,  # tile size for hidden (in copy from B)
):
    # Grid: (B, tiles along M, tiles along N)
    b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]
    h_ids = tl.arange(0, H)                             # [H]

    mask_m = m_offsets < M
    mask_n = n_offsets < N
    mask_h = h_ids < H

    # For each m in this tile, write to Out[b, m, :] if m < M, else to Out[b, m - N, :]
    for i in range(0, BLOCK_M):
        m = m_offsets[i]
        valid_m = mask_m[i]
        if valid_m:
            a_ptrs = A_ptr + b * stride_ab + m * stride_am + h_ids * stride_ah
            a_mask = mask_h
            a_vals = tl.load(a_ptrs, mask=a_mask, other=0.0)
            out_ptrs = Out_ptr + b * stride_ob + m * stride_oc + h_ids * stride_oh
            tl.store(out_ptrs, a_vals)

    # For each n in this tile, write to Out[b, M + n, :]
    for j in range(0, BLOCK_N):
        n = n_offsets[j]
        valid_n = mask_n[j]
        if valid_n:
            b_ptrs = B_ptr + b * stride_bb + n * stride_bn + h_ids * stride_bh
            b_mask = mask_h
            b_vals = tl.load(b_ptrs, mask=b_mask, other=0.0)
            out_base = b * stride_ob + (M + n) * stride_oc
            out_ptrs = Out_ptr + out_base + h_ids * stride_oh
            tl.store(out_ptrs, b_vals)


@triton.jit
def _batched_gemm_outer_kernel(
    X_ptr,  # *f32, [B, C, K] where C = M + N
    W_ptr,  # *f32, [K, K]
    P_ptr,  # *f32, [B, C, K]
    B: tl.constexpr,      # batch size
    C: tl.constexpr,      # sequence length
    K: tl.constexpr,      # hidden dim
    stride_xb,  # int: stride for X along batch
    stride_xc,  # int: stride for X along seq
    stride_xk,  # int: stride for X along hidden
    stride_w0,  # int: stride for W along dim 0 (rows = K_in)
    stride_w1,  # int: stride for W along dim 1 (cols = K_out)
    stride_pb,  # int: stride for P along batch
    stride_pc,  # int: stride for P along seq
    stride_pk,  # int: stride for P along hidden
    BLOCK_M: tl.constexpr,  # tile along C
    BLOCK_N: tl.constexpr,  # tile along K (output features)
    BLOCK_K: tl.constexpr,  # tile along reduction K
):
    # 3D grid: (batch, tiles along C, tiles along K)
    pid_b = tl.program_id(0)
    pid_cm = tl.program_id(1)
    pid_ck = tl.program_id(2)

    # Tile coordinates
    m_offsets = pid_cm * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    n_offsets = pid_ck * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]

    # Masks for boundaries
    mask_m = m_offsets < C
    mask_n = n_offsets < K

    # Initialize accumulator [BLOCK_M, BLOCK_N]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over reduction dimension K in chunks of BLOCK_K
    for k0 in range(0, K, BLOCK_K):
        k_ids = k0 + tl.arange(0, BLOCK_K)  # [BLOCK_K]
        mask_k = k_ids < K

        # Load X tile: [BLOCK_M, BLOCK_K]
        x_ptrs = X_ptr + pid_b * stride_xb + m_offsets[:, None] * stride_xc + k_ids[None, :] * stride_xk
        x_mask = mask_m[:, None] & mask_k[None, :]
        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)  # [BLOCK_M, BLOCK_K]

        # Load W chunk: [BLOCK_K, BLOCK_N], W is [K, K] (rows=k_ids, cols=n_offsets)
        w_ptrs = W_ptr + k_ids[:, None] * stride_w0 + n_offsets[None, :] * stride_w1
        w_mask = mask_k[:, None] & mask_n[None, :]
        w_chunk = tl.load(w_ptrs, mask=w_mask, other=0.0)  # [BLOCK_K, BLOCK_N]

        # Outer-product accumulation for this chunk
        # For each kk in k_ids, add x_tile[:, kk][:, None] * w_chunk[kk, :][None, :]
        for kk in range(0, BLOCK_K):
            # Masked vector for this kk
            kk_valid = (k_ids[kk] < K)
            x_vec = x_tile[:, kk]  # [BLOCK_M]
            w_vec = w_chunk[kk, :]  # [BLOCK_N]
            # Broadcast to [BLOCK_M, BLOCK_N]
            acc += x_vec[:, None] * w_vec[None, :]

    # Store accumulated tile
    p_ptrs = P_ptr + pid_b * stride_pb + m_offsets[:, None] * stride_pc + n_offsets[None, :] * stride_pk
    p_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(p_ptrs, acc, mask=p_mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        Triton-optimized version:
        - Concatenates encoder_hidden_states and hidden_states along sequence dimension using Triton.
        - Applies linear projection using Triton GEMM (outer-product accumulation).
        - Splits the result back into two outputs.

        Args:
            hidden_states: [B, N, H]
            encoder_hidden_states: [B, M, H]
            process_weight: [H, H]
        Returns:
            (processed_encoder: [B, M, H], processed_hidden: [B, N, H])
        """
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "All tensors must be on CUDA device"
        assert hidden_states.dim() == 3 and encoder_hidden_states.dim() == 3, "Inputs must be 3D [B, L, H]"
        B = hidden_states.shape[0]
        M = encoder_hidden_states.shape[1]
        N = hidden_states.shape[1]
        H = hidden_states.shape[2]
        assert process_weight.shape[0] == H and process_weight.shape[1] == H, "process_weight must be [H, H]"

        # Ensure contiguous
        A = encoder_hidden_states.contiguous()
        B_b = hidden_states.contiguous()
        W = process_weight.contiguous()

        # 1) Concatenate along sequence dimension: Out [B, C, H], C = M + N
        C = M + N
        Out = torch.empty((B, C, H), device=A.device, dtype=torch.float32)

        # Launch Triton concat kernel: grid over (B, tiles along M and N)
        # Choose moderate tile sizes to handle varying M,N robustly
        BLOCK_M = 128
        BLOCK_N = 128
        grid_concat = (B, triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        _concat_seq_kernel[grid_concat](
            A, B_b, Out,
            B, M, N, H,
            A.stride(0), A.stride(1), A.stride(2),
            B_b.stride(0), B_b.stride(1), B_b.stride(2),
            Out.stride(0), Out.stride(1), Out.stride(2),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            num_warps=4,
        )

        # 2) GEMM using Triton: P = Out @ W^T -> [B, C, H]
        P = torch.empty((B, C, H), device=A.device, dtype=torch.float32)

        # Strides
        stride_xb, stride_xc, stride_xk = Out.stride(0), Out.stride(1), Out.stride(2)
        stride_w0, stride_w1 = W.stride(0), W.stride(1)  # W is [K, K]
        stride_pb, stride_pc, stride_pk = P.stride(0), P.stride(1), P.stride(2)

        # GEMM tiling: use moderate tiles, BLOCK_M along C, BLOCK_N along K (output features), BLOCK_K along reduction
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 64
        grid_gemm = (B, triton.cdiv(C, BLOCK_M), triton.cdiv(H, BLOCK_N))
        _batched_gemm_outer_kernel[grid_gemm](
            Out, W, P,
            B, C, H,
            stride_xb, stride_xc, stride_xk,
            stride_w0, stride_w1,
            stride_pb, stride_pc, stride_pk,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4,
        )

        # 3) Split along sequence dimension to match original API
        processed_encoder = P[:, :M, :]
        processed_hidden = P[:, M:, :]
        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
