import torch
import triton
import triton.language as tl


@triton.jit
def concat_matmul_kernel(
    A_ptr,          # *const float, shape [B, L, H] (concatenated)
    W_ptr,          # *const float, shape [H, H] (process_weight, not transposed)
    C_ptr,          # *float, shape [B, L, H] (output)
    B,              # int32: batch size
    L,              # int32: total sequence length = T + I
    H,              # int32: hidden_dim
    BLOCK_M: tl.constexpr,  # tile in M (rows) = B*L
    BLOCK_N: tl.constexpr,  # tile in N (columns) = H
    BLOCK_K: tl.constexpr,  # tile in K (reduction) = H
):
    # 2D launch grid: (grid_m, grid_n) cover (B*L) x H
    grid_m = tl.program_id(0)
    grid_n = tl.program_id(1)

    # Offsets for this program along M and N
    m_offsets = grid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # over rows: [0..B*L)
    n_offsets = grid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # over columns: [0..H)

    # We treat A as [M, K] = [(B*L), H], but it is logically [B, L, H]
    # Derive b and l for each m in m_offsets
    # b = m // L, l = m % L
    b = m_offsets // L
    l = m_offsets % L

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Reduction over K dimension (hidden_dim)
    # We iterate k in steps of BLOCK_K; W is [H, H], we need W^T but we can load as W[k, n] via strides
    # For this, since W is contiguous [H, H], reading W[k, n] is fine: pointer arithmetic uses strides.
    for k_start in range(0, H, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)  # [BLOCK_K]
        # Load A[m, k] which corresponds to A[b, l, k]
        # A is [B, L, H] contiguous: index = b*stride_b + l*stride_l + k*stride_k
        # With contiguous, stride_b = L*H, stride_l = H, stride_k = 1
        # But to keep it general, we'll assume A is contiguous [B, L, H] with strides (L*H, H, 1)
        A_row_ptrs = A_ptr + b[:, None] * (L * H) + l[:, None] * H + k_offsets[None, :] * 1
        # Load mask for A: only valid when m_offsets < B*L and k_offsets < H
        a_mask = (m_offsets[:, None] < (B * L)) & (k_offsets[None, :] < H)
        A_vals = tl.load(A_row_ptrs, mask=a_mask, other=0.0)  # shape [BLOCK_M, BLOCK_K]

        # Load W[k, n] where W is [H, H] contiguous
        # index = k_offsets[:, None]*H + n_offsets[None, :]
        W_ptrs = W_ptr + k_offsets[:, None] * H + n_offsets[None, :]
        w_mask = (k_offsets[:, None] < H) & (n_offsets[None, :] < H)
        W_vals = tl.load(W_ptrs, mask=w_mask, other=0.0)  # shape [BLOCK_K, BLOCK_N]

        # Accumulate: acc += A_vals @ W_vals
        acc += tl.dot(A_vals, W_vals)

    # Write back to C[b, l, n]
    # C is [B, L, H] contiguous: index = b*stride_b + l*stride_l + n*stride_n
    # Strides for C: (L*H, H, 1)
    C_row_ptrs = C_ptr + b[:, None] * (L * H) + l[:, None] * H + n_offsets[None, :]
    c_mask = (m_offsets[:, None] < (B * L)) & (n_offsets[None, :] < H)
    tl.store(C_row_ptrs, acc, mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version of the original run:
        - Concatenates encoder_hidden_states and hidden_states along sequence dimension.
        - Applies linear projection via a Triton matmul kernel.
        - Splits the result back into separate encoder and image streams.
        """
        # Shapes
        batch = hidden_states.shape[0]
        # hidden_states: [B, I, H]
        # encoder_hidden_states: [B, T, H]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        H = hidden_states.shape[2]
        assert encoder_hidden_states.shape[2] == H, "encoder_hidden_states hidden_dim must match hidden_states"
        assert process_weight.shape[0] == H and process_weight.shape[1] == H, "process_weight must be [H, H]"

        # Concatenate along sequence dimension: [B, T + I, H]
        # Note: this is a simple torch operation; it does not use Triton (it's not a numerical compute in our Triton-only requirement).
        concatenated = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        assert concatenated.shape == (batch, T + I, H), "Concatenation failed"

        # Ensure contiguous for predictable strides
        concatenated = concatenated.contiguous()
        process_weight = process_weight.contiguous()

        # Allocate output [B, T+I, H]
        L = T + I
        output = torch.empty((batch, L, H), device=concatenated.device, dtype=concatenated.dtype)

        # Launch Triton kernel: grid over (B*L, H)
        # Choose tiles; these are good defaults for varied H
        BLOCK_M = 32   # rows = B*L can be large, so this keeps per-program work modest
        BLOCK_N = 64   # columns = H; typical hidden dims are multiples of 64
        BLOCK_K = 32   # reduction over H; masks handle remainder

        grid_m = triton.cdiv(batch * L, BLOCK_M)
        grid_n = triton.cdiv(H, BLOCK_N)

        # Run kernel
        concat_matmul_kernel[(grid_m, grid_n)](
            concatenated,              # A_ptr
            process_weight,            # W_ptr (not transposed; we load W[k, n] directly)
            output,                    # C_ptr
            batch, L, H,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4,
            num_stages=2,
        )

        # Split back into separate streams
        processed_encoder = output[:, :T, :]
        processed_hidden = output[:, T:, :]

        return processed_encoder, processed_hidden