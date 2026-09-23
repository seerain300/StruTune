import torch
import triton
import triton.language as tl


@triton.jit
def _concat_seq_kernel(
    enc_ptr,  # *f32, [B, T, H]
    hid_ptr,  # *f32, [B, I, H]
    out_ptr,  # *f32, [B, T+I, H]
    B, T, I, H,
    enc_stride_b, enc_stride_t, enc_stride_h,
    hid_stride_b, hid_stride_i, hid_stride_h,
    out_stride_b, out_stride_l, out_stride_h,
    BLOCK_L: tl.constexpr,
):
    b = tl.program_id(0)
    # compute output row offsets for this batch
    total_L = T + I
    for l in range(0, total_L, BLOCK_L):
        l_offsets = l + tl.arange(0, BLOCK_L)
        mask_l = l_offsets < total_L
        # load from encoder if l < T else from hidden
        is_encoder = (l_offsets[:, None] < T)
        # base pointers
        enc_base = enc_ptr + b * enc_stride_b
        hid_base = hid_ptr + b * hid_stride_b
        out_base = out_ptr + b * out_stride_b
        # element-wise pointers
        enc_ptrs = enc_base + l_offsets[:, None] * enc_stride_t + tl.arange(0, H)[None, :] * enc_stride_h
        hid_ptrs = hid_base + (l_offsets - T)[:, None] * hid_stride_i + tl.arange(0, H)[None, :] * hid_stride_h
        out_ptrs = out_base + l_offsets[:, None] * out_stride_l + tl.arange(0, H)[None, :] * out_stride_h
        # masked loads
        enc_vals = tl.load(enc_ptrs, mask=(mask_l[:, None] & is_encoder), other=0.0)
        hid_vals = tl.load(hid_ptrs, mask=(mask_l[:, None] & (~is_encoder)), other=0.0)
        vals = enc_vals + hid_vals  # select via mask
        tl.store(out_ptrs, vals, mask=mask_l[:, None])


@triton.jit
def _batched_gemm_right_kernel(
    A_ptr,      # *f32, [M, K], M = B*(T+I), K = H
    W_ptr,      # *f32, [K, K] (process_weight, right-multiply by W^T)
    C_ptr,      # *f32, [M, K] output
    M, K,       # int sizes
    A_stride_m, A_stride_k,
    W_stride_k, W_stride_col,
    C_stride_m, C_stride_k,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D grid over output tiles
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = m_offsets < M
    mask_n = n_offsets < K

    # accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # reduce over K
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < K

        # A[m, k] tile
        a_ptrs = A_ptr + m_offsets[:, None] * A_stride_m + k_offsets[None, :] * A_stride_k
        A_tile = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)

        # W^T: we load W[k, cols] where cols are n_offsets
        # W has shape [K, K], indexing W[k, n]
        W_ptrs = W_ptr + k_offsets[:, None] * W_stride_k + n_offsets[None, :] * W_stride_col
        W_tile = tl.load(W_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)

        # accumulate
        acc += tl.dot(A_tile, W_tile)

    # store C[m, n] = acc
    C_ptrs = C_ptr + m_offsets[:, None] * C_stride_m + n_offsets[None, :] * C_stride_k
    tl.store(C_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


@triton.jit
def _split_streams_kernel(
    C_ptr,              # *f32, [M, K], M=B*(T+I), K=H
    encoder_ptr,        # *f32, [B, T, K]
    hidden_ptr,         # *f32, [B, I, K]
    M, T, I, K,
    C_stride_m, C_stride_k,
    enc_stride_b, enc_stride_t, enc_stride_k,
    hid_stride_b, hid_stride_i, hid_stride_k,
    BLOCK_S: tl.constexpr,
):
    b = tl.program_id(0)
    # map row index to (batch, seq) for encoder and hidden
    total_L = T + I
    for s_idx in range(0, total_L, BLOCK_S):
        s_offsets = s_idx + tl.arange(0, BLOCK_S)
        mask_s = s_offsets < total_L
        # encoder rows
        enc_base = encoder_ptr + b * enc_stride_b
        hid_base = hidden_ptr + b * hid_stride_b
        C_row_base = C_ptr + (b * (total_L) + s_offsets) * C_stride_m
        # for each s, write to enc or hid
        for i in range(BLOCK_S):
            if s_offsets[i] < T:
                # copy C[b * (T+I) + s, :] to encoder[b, s, :]
                C_row_ptr = C_row_base + i * C_stride_m
                enc_row_ptr = enc_base + s_offsets[i] * enc_stride_t
                vals = tl.load(C_row_ptr + tl.arange(0, K) * C_stride_k, mask=mask_s[i], other=0.0)
                tl.store(enc_row_ptr + tl.arange(0, K) * enc_stride_k, vals, mask=mask_s[i])
            else:
                # copy to hidden[b, s - T, :]
                local_idx = s_offsets[i] - T
                C_row_ptr = C_row_base + i * C_stride_m
                hid_row_ptr = hid_base + local_idx * hid_stride_i
                vals = tl.load(C_row_ptr + tl.arange(0, K) * C_stride_k, mask=mask_s[i], other=0.0)
                tl.store(hid_row_ptr + tl.arange(0, K) * hid_stride_k, vals, mask=mask_s[i])


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        hidden_states: torch.Tensor,      # [B, I, H]
        encoder_hidden_states: torch.Tensor,  # [B, T, H]
        process_weight: torch.Tensor,     # [H, H]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-only implementation:
        1) Concatenate along sequence dimension into [B, T+I, H] (Triton).
        2) Compute processed = concatenated @ process_weight.T (Triton 2D-tiled GEMM).
        3) Split back into encoder and hidden streams (Triton).
        Returns (processed_encoder [B, T, H], processed_hidden [B, I, H]).
        """
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "All tensors must be on CUDA for Triton kernels."
        assert hidden_states.dim() == 3 and encoder_hidden_states.dim() == 3 and process_weight.dim() == 2
        B, I, H = hidden_states.shape
        T, H2 = encoder_hidden_states.shape[1], hidden_states.shape[2]
        assert H2 == H, "hidden_dim must match for both inputs"
        assert process_weight.shape[0] == H and process_weight.shape[1] == H, "process_weight must be [H, H]"

        # 1) Concatenate sequences: out_cat [B, T+I, H]
        out_cat = torch.empty((B, T + I, H), dtype=torch.float32, device=hidden_states.device)

        _concat_seq_kernel[(B,)](
            encoder_hidden_states, hidden_states, out_cat,
            B, T, I, H,
            *encoder_hidden_states.stride(), *hidden_states.stride(), *out_cat.stride(),
            BLOCK_L=256, num_warps=4, num_stages=2
        )
        # Ensure out_cat is contiguous for GEMM (A)
        A = out_cat.contiguous()  # [M, K], M = B*(T+I), K = H

        # 2) Batched GEMM: C = A @ process_weight.T, shape [M, H]
        M = B * (T + I)
        K = H
        W = process_weight.contiguous()          # [K, K]
        C = torch.empty((M, K), dtype=torch.float32, device=hidden_states.device)

        grid_m = triton.cdiv(M, 128)
        grid_n = triton.cdiv(K, 128)
        _batched_gemm_right_kernel[(grid_m, grid_n)](
            A, W, C,
            M, K,
            A.stride(0), A.stride(1),
            W.stride(0), W.stride(1),
            C.stride(0), C.stride(1),
            BLOCK_M=128, BLOCK_N=128, BLOCK_K=64,
            num_warps=4, num_stages=2
        )

        # 3) Split C back into [B, T, H] and [B, I, H]
        encoder_out = torch.empty((B, T, H), dtype=torch.float32, device=hidden_states.device)
        hidden_out = torch.empty((B, I, H), dtype=torch.float32, device=hidden_states.device)

        _split_streams_kernel[(B,)](
            C, encoder_out, hidden_out,
            M, T, I, K,
            *C.stride(), *encoder_out.stride(), *hidden_out.stride(),
            BLOCK_S=256, num_warps=4, num_stages=2
        )

        return encoder_out, hidden_out