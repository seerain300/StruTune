import torch
import triton
import triton.language as tl


@triton.jit
def _batched_proj_two_inputs_kernel(
    Xe_ptr,       # *f32, [B, M, K] = encoder_hidden_states
    Xi_ptr,       # *f32, [B, N, K] = hidden_states (image)
    We_ptr,       # *f32, [K, K] = process_weight (left-multiplied, i.e., W)
    Pe_ptr,       # *f32, [B, M, K] = output for encoder stream
    Pi_ptr,       # *f32, [B, N, K] = output for image stream
    B: tl.constexpr,      # batch size
    M: tl.constexpr,      # text_seq_len
    N: tl.constexpr,      # img_seq_len
    K: tl.constexpr,      # hidden_dim
    stride_xeb,   # stride for batch in Xe
    stride_xem,   # stride for seq in Xe (M)
    stride_xek,   # stride for hidden in Xe (K)
    stride_xib,   # stride for batch in Xi
    stride_xin,   # stride for seq in Xi (N)
    stride_xik,   # stride for hidden in Xi
    stride_w0,    # stride along dim 0 (rows) in We (K)
    stride_w1,    # stride along dim 1 (cols) in We (K)
    stride_peb,   # stride for batch in Pe
    stride_pem,   # stride for seq in Pe (M)
    stride_pek,   # stride for hidden in Pe (K)
    stride_pib,   # stride for batch in Pi
    stride_pin,   # stride for seq in Pi (N)
    stride_pik,   # stride for hidden in Pi (K)
    BLOCK_M: tl.constexpr,  # tile size over M
    BLOCK_N: tl.constexpr,  # tile size over N
    BLOCK_K: tl.constexpr,  # tile size over K
):
    # Grid: (B, ceil_div(M, BLOCK_M), ceil_div(N, BLOCK_N))
    b = tl.program_id(0)
    m_block = tl.program_id(1)
    n_block = tl.program_id(2)

    # Offsets within tiles
    m_offsets = m_block * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    n_offsets = n_block * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]
    k_offsets = tl.arange(0, BLOCK_K)                     # [BLOCK_K]

    # Masks for boundaries
    mask_m = m_offsets < M
    mask_n = n_offsets < N

    # Initialize accumulators for encoder and image streams
    acc_e = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)  # [M_tile, K_tile]
    acc_i = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)  # [N_tile, K_tile]

    # Loop over hidden dimension K in chunks of BLOCK_K
    for k0 in range(0, K, BLOCK_K):
        k = k0 + k_offsets  # [BLOCK_K]
        mask_k = k < K

        # Load Xe tile: [BLOCK_M, BLOCK_K], Xe[b, m, k]
        xe_ptrs = Xe_ptr + b * stride_xeb + m_offsets[:, None] * stride_xem + k[None, :] * stride_xek
        xe_mask = mask_m[:, None] & mask_k[None, :]
        xe_tile = tl.load(xe_ptrs, mask=xe_mask, other=0.0)

        # Load We tile as [BLOCK_K, BLOCK_K] for dot: We[k, k2] where k is reduction, k2 is output hidden
        we_ptrs = We_ptr + k[:, None] * stride_w0 + k[None, :] * stride_w1  # W[k, k2]
        we_mask = mask_k[:, None] & mask_k[None, :]
        we_tile = tl.load(we_ptrs, mask=we_mask, other=0.0)

        # Accumulate for encoder
        acc_e += tl.dot(xe_tile, we_tile)  # [BLOCK_M, BLOCK_K]

        # Load Xi tile: [BLOCK_N, BLOCK_K], Xi[b, n, k]
        xi_ptrs = Xi_ptr + b * stride_xib + n_offsets[:, None] * stride_xin + k[None, :] * stride_xik
        xi_mask = mask_n[:, None] & mask_k[None, :]
        xi_tile = tl.load(xi_ptrs, mask=xi_mask, other=0.0)

        # Accumulate for image
        acc_i += tl.dot(xi_tile, we_tile)  # [BLOCK_N, BLOCK_K]

    # Store results for encoder stream: Pe[b, m, k]
    pe_ptrs = Pe_ptr + b * stride_peb + m_offsets[:, None] * stride_pem + k_offsets[None, :] * stride_pek
    pe_mask = mask_m[:, None] & (k_offsets[None, :] < K)
    tl.store(pe_ptrs, acc_e, mask=pe_mask)

    # Store results for image stream: Pi[b, n, k]
    pi_ptrs = Pi_ptr + b * stride_pib + n_offsets[:, None] * stride_pin + k_offsets[None, :] * stride_pik
    pi_mask = mask_n[:, None] & (k_offsets[None, :] < K)
    tl.store(pi_ptrs, acc_i, mask=pi_mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        Triton-optimized version without torch concatenation or matmul in forward.
        Computes:
          - processed_encoder[b, t, h] = sum_k encoder_hidden_states[b, t, k] * process_weight[k, h]
          - processed_hidden[b, i, h] = sum_k hidden_states[b, i, k] * process_weight[k, h]
        """
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "All tensors must be on CUDA"
        B = hidden_states.shape[0]
        M = encoder_hidden_states.shape[1]
        N = hidden_states.shape[1]
        K = hidden_states.shape[2]
        device = hidden_states.device

        # Ensure contiguity
        A = encoder_hidden_states.contiguous()     # [B, M, K]
        X_img = hidden_states.contiguous()         # [B, N, K]
        W = process_weight.contiguous()            # [K, K]

        # Allocate outputs [B, M, K] and [B, N, K]
        processed_encoder = torch.empty((B, M, K), dtype=torch.float32, device=device)
        processed_hidden = torch.empty((B, N, K), dtype=torch.float32, device=device)

        # Launch Triton kernel
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 64
        grid = (B, triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        _batched_proj_two_inputs_kernel[grid](
            A, X_img, W, processed_encoder, processed_hidden,
            B=B, M=M, N=N, K=K,
            stride_xeb=A.stride(0), stride_xem=A.stride(1), stride_xek=A.stride(2),
            stride_xib=X_img.stride(0), stride_xin=X_img.stride(1), stride_xik=X_img.stride(2),
            stride_w0=W.stride(0), stride_w1=W.stride(1),
            stride_peb=processed_encoder.stride(0), stride_pem=processed_encoder.stride(1), stride_pek=processed_encoder.stride(2),
            stride_pib=processed_hidden.stride(0), stride_pin=processed_hidden.stride(1), stride_pik=processed_hidden.stride(2),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2
        )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
