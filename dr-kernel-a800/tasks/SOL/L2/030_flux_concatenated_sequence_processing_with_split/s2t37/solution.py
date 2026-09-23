import triton
import triton.language as tl


# Triton kernel: concatenate encoder_hidden_states [B, T, H] and hidden_states [B, I, H]
# into out_cat [B, L, H] along sequence dim, where L = T + I.
@triton.jit
def concat_sequences_kernel(
    encoder_ptr, hidden_ptr, out_ptr,
    B, T, I, H,
    stride_b_e, stride_t_e, stride_h_e,   # encoder strides
    stride_b_h, stride_i_h, stride_h_h,   # hidden strides
    stride_b_o, stride_l_o, stride_h_o,   # out strides
    BLOCK_H: tl.constexpr,
):
    b = tl.program_id(0)
    m = tl.program_id(1)
    k_block = tl.program_id(2)

    k_offsets = k_block * BLOCK_H + tl.arange(0, BLOCK_H)
    k_mask = k_offsets < H

    # Compute source pointers
    if m < T:
        src_ptr = encoder_ptr + b * stride_b_e + m * stride_t_e + k_offsets * stride_h_e
    else:
        src_ptr = hidden_ptr + b * stride_b_h + (m - T) * stride_i_h + k_offsets * stride_h_h

    dest_ptr = out_ptr + b * stride_b_o + m * stride_l_o + k_offsets * stride_h_o

    values = tl.load(src_ptr, mask=k_mask, other=0.0)
    tl.store(dest_ptr, values, mask=k_mask)


# Triton batched GEMM:
# A: [B, M, K] = out_cat, where M = L = T + I, K = H
# B: [K, N] = process_weight.T, N = H (output dim)
# C: [B, M, N] = processed
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 64}, num_warps=2, num_stages=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def bmm_kernel(
    A_ptr, B_ptr, C_ptr,
    B, M, N, K,
    stride_a_b, stride_a_m, stride_a_k,   # A strides: [B, M, K]
    stride_b_k, stride_b_n,               # B strides: [K, N] (W_T)
    stride_c_b, stride_c_m, stride_c_n,   # C strides: [B, M, N]
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    m_mask = m_offsets < M
    n_mask = n_offsets < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < K

        # Load A tile: [BLOCK_M, BLOCK_K]
        A_ptrs = A_ptr + pid_b * stride_a_b + m_offsets[:, None] * stride_a_m + k_offsets[None, :] * stride_a_k
        A_mask = m_mask[:, None] & k_mask[None, :]
        a = tl.load(A_ptrs, mask=A_mask, other=0.0).to(tl.float32)

        # Load B tile: [BLOCK_K, BLOCK_N] from W_T [K, N], note strides are swapped for interpretation
        B_ptrs = B_ptr + k_offsets[:, None] * stride_b_k + n_offsets[None, :] * stride_b_n
        B_mask = k_mask[:, None] & n_mask[None, :]
        b = tl.load(B_ptrs, mask=B_mask, other=0.0).to(tl.float32)

        acc += tl.dot(a, b)

    C_ptrs = C_ptr + pid_b * stride_c_b + m_offsets[:, None] * stride_c_m + n_offsets[None, :] * stride_c_n
    C_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(C_ptrs, acc, mask=C_mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        hidden_states: [B, I, H]
        encoder_hidden_states: [B, T, H]
        process_weight: [H, H]
        Returns: (processed_encoder: [B, T, H], processed_hidden: [B, I, H])
        """
        # Basic validation
        assert hidden_states.dim() == 3 and encoder_hidden_states.dim() == 3, "Inputs must be 3D [B, dim, H]"
        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        H = hidden_states.shape[2]
        assert process_weight.shape[0] == H and process_weight.shape[1] == H, "process_weight must be [H, H]"
        assert hidden_states.dtype == encoder_hidden_states.dtype == process_weight.dtype, "All tensors must have same dtype"
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "All tensors must be on CUDA for Triton"

        L = T + I

        # 1) Concatenate sequences into out_cat [B, L, H] using Triton
        out_cat = torch.empty((B, L, H), dtype=hidden_states.dtype, device=hidden_states.device)

        BLOCK_H = 64
        grid_concat = (B, L, triton.cdiv(H, BLOCK_H))
        triton.run(
            concat_sequences_kernel[grid_concat](
                encoder_hidden_states, hidden_states, out_cat,
                B, T, I, H,
                *encoder_hidden_states.stride(), *hidden_states.stride(),
                *out_cat.stride(),
                BLOCK_H=BLOCK_H,
                num_warps=4,
                num_stages=2,
            )
        )

        # 2) Prepare W_T = process_weight.T [H, H] with correct strides for [K, N]
        W_T = process_weight.transpose(0, 1).contiguous()  # [H, H]
        stride_bk = W_T.stride(1)  # stride along K
        stride_bn = W_T.stride(0)  # stride along N (output dim)

        # 3) Allocate output processed [B, L, H] and perform GEMM with Triton
        processed = torch.empty((B, L, H), dtype=out_cat.dtype, device=out_cat.device)

        # 3D grid over (batch, tiles over M=L, tiles over N=H), K=H
        BLOCK_M = 64
        BLOCK_N = 64
        grid = (B, triton.cdiv(L, BLOCK_M), triton.cdiv(H, BLOCK_N))
        triton.run(
            bmm_kernel[grid](
                out_cat, W_T, processed,
                B, L, H, H,  # M=L, N=H, K=H
                *out_cat.stride(),   # A strides: [B, M, K]
                stride_bk, stride_bn,  # B strides: [K, N]
                *processed.stride(),  # C strides: [B, M, N]
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=64,
            )
        )

        # 4) Split back into separate streams
        processed_encoder = processed[:, :T, :]
        processed_hidden = processed[:, T:, :]

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
