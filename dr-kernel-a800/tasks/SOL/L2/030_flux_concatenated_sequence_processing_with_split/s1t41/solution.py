import torch
import triton
import triton.language as tl


# Triton kernel: concatenate encoder_hidden_states [B, T, K] and hidden_states [B, P, K]
# into Acat [B, T+P, K] without torch.cat.
@triton.jit
def _concatenate_kernel(
    encoder_ptr, hidden_ptr, Acat_ptr,
    B, T, P, K,
    encoder_stride_b, encoder_stride_t, encoder_stride_k,
    hidden_stride_b, hidden_stride_p, hidden_stride_k,
    Acat_stride_b, Acat_stride_l, Acat_stride_k,
    BLOCK_L: tl.constexpr, BLOCK_K: tl.constexpr,
):
    b = tl.program_id(0)
    l_tile = tl.program_id(1)
    k_tile = tl.program_id(2)

    l_offsets = l_tile * BLOCK_L + tl.arange(0, BLOCK_L)  # sequence positions [0, T+P)
    k_offsets = k_tile * BLOCK_K + tl.arange(0, BLOCK_K)  # hidden dimension
    L = T + P

    l_mask = l_offsets < L
    k_mask = k_offsets < K

    # If l < T: load from encoder[b, l, :], else: load from hidden[b, l - T, :]
    is_encoder = l_offsets[:, None] < T  # [BLOCK_L, 1], boolean
    mask_encoder = l_mask[:, None] & k_mask[None, :]
    mask_hidden = (~is_encoder) & l_mask[:, None] & k_mask[None, :]

    # Pointers for encoder loads
    enc_ptrs = encoder_ptr + b * encoder_stride_b + l_offsets[:, None] * encoder_stride_t + k_offsets[None, :] * encoder_stride_k
    enc_vals = tl.load(enc_ptrs, mask=mask_encoder, other=0.0)

    # Pointers for hidden loads (shift l by -T for hidden)
    hid_ptrs = hidden_ptr + b * hidden_stride_b + (l_offsets[:, None] - T) * hidden_stride_p + k_offsets[None, :] * hidden_stride_k
    hid_vals = tl.load(hid_ptrs, mask=mask_hidden, other=0.0)

    # Select based on is_encoder
    vals = tl.where(is_encoder, enc_vals, hid_vals)

    # Store into Acat[b, l, :]
    Acat_ptrs = Acat_ptr + b * Acat_stride_b + l_offsets[:, None] * Acat_stride_l + k_offsets[None, :] * Acat_stride_k
    tl.store(Acat_ptrs, vals, mask=(l_mask[:, None] & k_mask[None, :]))


# Triton kernel: matmul on A [M, K] and W [K, K] -> C [M, K], where
# A is Acat reshaped as [B*(T+P), K], W is process_weight.T [K, K].
@triton.jit
def _matmul_kernel(
    A_ptr, W_ptr, C_ptr,
    M, K, N,  # N here equals K, since W is [K, K]
    A_stride_m, A_stride_k,
    W_stride_k, W_stride_n,  # W is [K, N], with N=K
    C_stride_m, C_stride_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_b = tl.program_id(0)
    m_tile = tl.program_id(1)
    n_tile = tl.program_id(2)

    m_offsets = m_tile * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = n_tile * BLOCK_N + tl.arange(0, BLOCK_N)
    k_offsets = tl.arange(0, BLOCK_K)

    m_mask = m_offsets < M
    n_mask = n_offsets < N

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K in chunks of BLOCK_K
    for k_start in range(0, K, BLOCK_K):
        k_curr = k_start + k_offsets  # [BLOCK_K]
        k_mask = k_curr < K

        # Load A_tile: [BLOCK_M, BLOCK_K]
        A_ptrs = A_ptr + m_offsets[:, None] * A_stride_m + k_curr[None, :] * A_stride_k
        A_mask = m_mask[:, None] & k_mask[None, :]
        A_tile = tl.load(A_ptrs, mask=A_mask, other=0.0)

        # Load W_tile: [BLOCK_K, BLOCK_N]
        W_ptrs = W_ptr + k_curr[:, None] * W_stride_k + n_offsets[None, :] * W_stride_n
        W_mask = k_mask[:, None] & n_mask[None, :]
        W_tile = tl.load(W_ptrs, mask=W_mask, other=0.0)

        # Accumulate
        acc += tl.dot(A_tile, W_tile)

    # Store result to C
    C_ptrs = C_ptr + m_offsets[:, None] * C_stride_m + n_offsets[None, :] * C_stride_n
    C_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(C_ptrs, acc, mask=C_mask)


# Triton kernel: split C [B, T+P, K] into processed_encoder [B, T, K]
@triton.jit
def _split_encoder_kernel(
    C_ptr, out_ptr,
    B, T, K,
    C_stride_b, C_stride_t, C_stride_k,
    out_stride_b, out_stride_t, out_stride_k,
    BLOCK_T: tl.constexpr, BLOCK_K: tl.constexpr,
):
    b = tl.program_id(0)
    t_tile = tl.program_id(1)
    k_tile = tl.program_id(2)

    t_offsets = t_tile * BLOCK_T + tl.arange(0, BLOCK_T)
    k_offsets = k_tile * BLOCK_K + tl.arange(0, BLOCK_K)

    t_mask = t_offsets < T
    k_mask = k_offsets < K

    C_ptrs = C_ptr + b * C_stride_b + t_offsets[:, None] * C_stride_t + k_offsets[None, :] * C_stride_k
    vals = tl.load(C_ptrs, mask=(t_mask[:, None] & k_mask[None, :]), other=0.0)

    out_ptrs = out_ptr + b * out_stride_b + t_offsets[:, None] * out_stride_t + k_offsets[None, :] * out_stride_k
    tl.store(out_ptrs, vals, mask=(t_mask[:, None] & k_mask[None, :]))


# Triton kernel: split C [B, T+P, K] into processed_hidden [B, P, K]
@triton.jit
def _split_hidden_kernel(
    C_ptr, out_ptr,
    B, T, P, K,
    C_stride_b, C_stride_t, C_stride_k,
    out_stride_b, out_stride_p, out_stride_k,
    BLOCK_P: tl.constexpr, BLOCK_K: tl.constexpr,
):
    b = tl.program_id(0)
    p_tile = tl.program_id(1)
    k_tile = tl.program_id(2)

    p_offsets = p_tile * BLOCK_P + tl.arange(0, BLOCK_P)
    k_offsets = k_tile * BLOCK_K + tl.arange(0, BLOCK_K)

    p_mask = p_offsets < P
    k_mask = k_offsets < K

    C_ptrs = C_ptr + b * C_stride_b + (T + p_offsets)[:, None] * C_stride_t + k_offsets[None, :] * C_stride_k
    vals = tl.load(C_ptrs, mask=(p_mask[:, None] & k_mask[None, :]), other=0.0)

    out_ptrs = out_ptr + b * out_stride_b + p_offsets[:, None] * out_stride_p + k_offsets[None, :] * out_stride_k
    tl.store(out_ptrs, vals, mask=(p_mask[:, None] & k_mask[None, :]))


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version of the original run function:
        - Concatenate along sequence dimension in Triton.
        - Perform GEMM in Triton.
        - Split outputs in Triton.
        Returns (processed_encoder [B, T, K], processed_hidden [B, P, K]).
        """
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "All tensors must be CUDA for Triton"
        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        P = hidden_states.shape[1]
        K = hidden_states.shape[2]
        assert encoder_hidden_states.shape[0] == B and encoder_hidden_states.shape[2] == K
        assert process_weight.shape[0] == K and process_weight.shape[1] == K

        # 1) Concatenate encoder_hidden_states and hidden_states along sequence dimension using Triton
        # Acat: [B, T+P, K]
        L = T + P
        Acat = torch.empty((B, L, K), device=hidden_states.device, dtype=hidden_states.dtype)

        BLOCK_L = 128
        BLOCK_K = 128
        grid_concat = (B, triton.cdiv(L, BLOCK_L), triton.cdiv(K, BLOCK_K))
        _concatenate_kernel[grid_concat](
            encoder_hidden_states, hidden_states, Acat,
            B, T, P, K,
            encoder_hidden_states.stride(0), encoder_hidden_states.stride(1), encoder_hidden_states.stride(2),
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            Acat.stride(0), Acat.stride(1), Acat.stride(2),
            BLOCK_L=BLOCK_L, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2
        )

        # 2) GEMM: Acat @ process_weight.T -> C_flat [M, K], where M = B * (T+P)
        # Treat Acat as [M, K] and process_weight.T as [K, K]
        M = B * L
        # Ensure dtype float32 for GEMM
        A_flat = Acat.reshape(M, K).to(torch.float32)
        W_T = process_weight.transpose(0, 1).to(torch.float32)  # [K, K]
        C_flat = torch.empty((M, K), device=hidden_states.device, dtype=torch.float32)

        # Launch Triton matmul kernel
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 64
        grid_matmul = (B, triton.cdiv(M, BLOCK_M), triton.cdiv(K, BLOCK_N))
        _matmul_kernel[grid_matmul](
            A_flat, W_T, C_flat,
            M, K, K,
            A_flat.stride(0), A_flat.stride(1),
            W_T.stride(0), W_T.stride(1),
            C_flat.stride(0), C_flat.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2
        )

        # Reshape C_flat back to [B, T+P, K]
        C = C_flat.view(B, L, K)

        # 3) Split outputs in Triton
        processed_encoder = torch.empty((B, T, K), device=hidden_states.device, dtype=torch.float32)
        processed_hidden = torch.empty((B, P, K), device=hidden_states.device, dtype=torch.float32)

        # Split encoder: copy C[:, :T, :]
        BLOCK_T = 64
        grid_e = (B, triton.cdiv(T, BLOCK_T), triton.cdiv(K, BLOCK_K))
        _split_encoder_kernel[grid_e](
            C, processed_encoder,
            B, T, K,
            C.stride(0), C.stride(1), C.stride(2),
            processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
            BLOCK_T=BLOCK_T, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2
        )

        # Split hidden: copy C[:, T:, :]
        BLOCK_P = 64
        grid_i = (B, triton.cdiv(P, BLOCK_P), triton.cdiv(K, BLOCK_K))
        _split_hidden_kernel[grid_i](
            C, processed_hidden,
            B, T, P, K,
            C.stride(0), C.stride(1), C.stride(2),
            processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
            BLOCK_P=BLOCK_P, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2
        )

        # Return results (outputs are float32 as per original code using default float32 process_weight and inputs)
        return processed_encoder, processed_hidden


# Example usage (for local testing):
# model = ModelNew().cuda()
# B, T, P, K = 1, 77, 4096, 1024
# x = torch.randn(B, T, K, device='cuda', dtype=torch.float32)
# h = torch.randn(B, P, K, device='cuda', dtype=torch.float32)
# W = torch.randn(K, K, device='cuda', dtype=torch.float32)  # process_weight [K, K]
# y_e, y_h = model(h, x, W)
# print(y_e.shape, y_h.shape)  # should be (B, T, K) and (B, P, K)


def run(*args):
    return ModelNew()(*args)
