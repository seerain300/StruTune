import torch
import triton
import triton.language as tl

@triton.jit
def batched_matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    A_stride_m, A_stride_k,
    B_stride_k, B_stride_n,
    C_stride_m, C_stride_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    # program ids: (batch, tile over M, tile over N)
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    # compute offsets for this tile
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # accumulate in fp32 for stability
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # loop over K dimension
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)

        # pointers
        # A is [M, K] with strides (A_stride_m, A_stride_k)
        A_tile_ptr = A_ptr + m_offsets[:, None] * A_stride_m + k_offsets[None, :] * A_stride_k
        # B is [K, N] with strides (B_stride_k, B_stride_n)
        B_tile_ptr = B_ptr + k_offsets[:, None] * B_stride_k + n_offsets[None, :] * B_stride_n

        # masks for bounds
        a_mask = (m_offsets[:, None] < M) & (k_offsets[None, :] < K)
        b_mask = (k_offsets[:, None] < K) & (n_offsets[None, :] < N)

        # loads and casts to fp32
        A_tile = tl.load(A_tile_ptr, mask=a_mask, other=0.0)
        B_tile = tl.load(B_tile_ptr, mask=b_mask, other=0.0)

        # accumulate
        acc += tl.dot(A_tile.to(tl.float32), B_tile.to(tl.float32))

    # store results back to C
    C_tile_ptr = C_ptr + m_offsets[:, None] * C_stride_m + n_offsets[None, :] * C_stride_n
    c_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    tl.store(C_tile_ptr, acc, mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,           # [B, I, H]
        encoder_hidden_states: torch.Tensor,   # [B, T, H]
        process_weight: torch.Tensor,          # [H, H]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Shapes
        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        H = hidden_states.shape[2]

        # 1) Concatenate along sequence dimension using torch (data movement, not heavy compute)
        concatenated = torch.cat([encoder_hidden_states, hidden_states], dim=1)  # [B, S, H], S = T + I

        # 2) Transpose process_weight for GEMM: B [K=H, N=H]
        B_weight = process_weight.t().contiguous()  # [H, H]

        # 3) Allocate output C [B, S, H]
        S = T + I
        C = torch.empty((B, S, H), device=concatenated.device, dtype=concatenated.dtype)

        # 4) Launch Triton batched matmul kernel: C[b, s, :] = concatenated[b, s, :] @ process_weight.T
        #    We flatten b into the first grid dim by combining (B, tiles_m, tiles_n)
        #    Grid dims:
        #      - pid_b = tl.program_id(0) iterates over batch
        #      - pid_m = tl.program_id(1) over tiles of M=S
        #      - pid_n = tl.program_id(2) over tiles of N=H
        BLOCK_M = 64
        BLOCK_N = 128
        BLOCK_K = 64

        grid = (B, triton.cdiv(S, BLOCK_M), triton.cdiv(H, BLOCK_N))
        batched_matmul_kernel[grid](
            concatenated, B_weight, C,
            S, H, H,            # M, N, K
            concatenated.stride(0), concatenated.stride(2),  # A strides: (m=S dim, k=H dim)
            B_weight.stride(0), B_weight.stride(1),         # B strides: (k, n)
            C.stride(0), C.stride(2),                      # C strides: (m=S, n=H)
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=3,
        )

        # 5) Split C back into encoder and hidden streams using torch (simple and correct)
        processed_encoder = C[:, :T, :]
        processed_hidden = C[:, T:, :]

        return processed_encoder, processed_hidden