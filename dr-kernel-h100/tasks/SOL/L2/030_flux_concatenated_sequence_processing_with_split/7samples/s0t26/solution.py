import torch
import triton
import triton.language as tl


@triton.jit
def concat_seq_dim1_kernel(
    A_ptr, E_ptr, H_ptr,
    B, T, I, K,
    stride_A_b, stride_A_m, stride_A_k,
    stride_E_b, stride_E_t, stride_E_k,
    stride_H_b, stride_H_i, stride_H_k,
):
    # Grid: (B, tiles along M, tiles along K)
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_k = tl.program_id(2)

    BLOCK_M = 128
    BLOCK_K = 64

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    k_offsets = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)

    mask_m = m_offsets < (T + I)
    mask_k = k_offsets < K

    # Determine source tensor: encoder (E) if m < T, else image (H)
    is_encoder = m_offsets < T

    # We'll loop over K tile and store to A
    for k_idx in range(0, BLOCK_K):
        k = k_offsets[k_idx]
        k_valid = mask_k[k_idx]
        # For each m, select source pointer
        e_ptrs = E_ptr + pid_b * stride_E_b + m_offsets * stride_E_t + k * stride_E_k
        h_ptrs = H_ptr + pid_b * stride_H_b + (m_offsets - T) * stride_H_i + k * stride_H_k
        a_ptrs = A_ptr + pid_b * stride_A_b + m_offsets * stride_A_m + k * stride_A_k

        vals = tl.zeros([BLOCK_M], dtype=tl.float32)
        # For valid m and k, load from either E or H
        # Build masks
        mask_load = mask_m & k_valid
        # Select values: if encoder, use e_ptrs; else use h_ptrs
        # Use where to select per m
        vals = tl.where(is_encoder & mask_load, tl.load(e_ptrs, mask=mask_load, other=0.0),
                        tl.where((~is_encoder) & mask_load, tl.load(h_ptrs, mask=mask_load, other=0.0), vals))

        # Store to A for all valid m
        tl.store(a_ptrs, vals, mask=mask_m & k_valid)


@triton.jit
def batched_matmul_kernel(
    C_ptr, A_ptr, W_ptr,
    B, M, N, K,
    stride_A_b, stride_A_m, stride_A_k,
    stride_W_k, stride_W_n,
    stride_C_b, stride_C_m, stride_C_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    # Grid: (B, tiles over M, tiles over N)
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = m_offsets < M
    mask_n = n_offsets < N

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < K

        # Load A tile: [BLOCK_M, BLOCK_K]
        a_ptrs = A_ptr + pid_b * stride_A_b + m_offsets[:, None] * stride_A_m + k_offsets[None, :] * stride_A_k
        a_mask = mask_m[:, None] & mask_k[None, :]
        A_tile = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # Load W tile: [BLOCK_K, BLOCK_N]
        w_ptrs = W_ptr + k_offsets[:, None] * stride_W_k + n_offsets[None, :] * stride_W_n
        w_mask = mask_k[:, None] & mask_n[None, :]
        W_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)

        acc += tl.dot(A_tile, W_tile)

    # Store result
    c_ptrs = C_ptr + pid_b * stride_C_b + m_offsets[:, None] * stride_C_m + n_offsets[None, :] * stride_C_n
    c_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(c_ptrs, acc, mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                encoder_hidden_states: torch.Tensor,
                process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version of the original run function.
        - Concatenates encoder_hidden_states and hidden_states along sequence dim using Triton.
        - Computes processed = concatenated @ process_weight.T using a Triton batched matmul.
        - Splits result back into encoder and hidden outputs.
        """
        # Ensure CUDA tensors
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, \
            "All inputs must be CUDA tensors."

        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        K = hidden_states.shape[2]
        M = T + I

        # 1) Allocate A [B, M, K] and fill via Triton concat kernel
        A = torch.empty((B, M, K), dtype=torch.float32, device=hidden_states.device)

        # Strides
        stride_A_b, stride_A_m, stride_A_k = A.stride()
        stride_E_b, stride_E_t, stride_E_k = encoder_hidden_states.contiguous().stride()
        stride_H_b, stride_H_i, stride_H_k = hidden_states.contiguous().stride()

        grid_concat = (B, triton.cdiv(M, 128), triton.cdiv(K, 64))
        concat_seq_dim1_kernel[grid_concat](
            A, encoder_hidden_states.contiguous(), hidden_states.contiguous(),
            B, T, I, K,
            stride_A_b, stride_A_m, stride_A_k,
            stride_E_b, stride_E_t, stride_E_k,
            stride_H_b, stride_H_i, stride_H_k,
            num_warps=4, num_stages=2
        )

        # 2) Batched matmul: C = A @ W, W is process_weight [K, K]
        W = process_weight.to(torch.float32).contiguous()
        C = torch.empty((B, M, K), dtype=torch.float32, device=hidden_states.device)

        stride_A_b_calc, stride_A_m_calc, stride_A_k_calc = A.stride()
        stride_W_k, stride_W_n = W.stride()
        stride_C_b, stride_C_m, stride_C_n = C.stride()

        BLOCK_M = 128
        BLOCK_N = 128
        BLOCK_K = 64
        grid_gemm = (B, triton.cdiv(M, BLOCK_M), triton.cdiv(K, BLOCK_N))
        batched_matmul_kernel[grid_gemm](
            C, A, W,
            B, M, K, K,
            stride_A_b_calc, stride_A_m_calc, stride_A_k_calc,
            stride_W_k, stride_W_n,
            stride_C_b, stride_C_m, stride_C_n,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=8, num_stages=3
        )

        # 3) Split back and cast to original dtypes
        processed_encoder = C[:, :T, :]
        processed_hidden = C[:, T:, :]

        processed_encoder = processed_encoder.to(encoder_hidden_states.dtype)
        processed_hidden = processed_hidden.to(hidden_states.dtype)

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
