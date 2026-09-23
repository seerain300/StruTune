import torch

# Triton is required for the numerical computation
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: batched matmul X[B, C, K] @ W[K, K] -> P[B, C, K]
# We implement C as the sequence dimension of the concatenated tensor.
# K is the hidden_dim (feature dim).
@triton.jit
def _batched_matmul_kernel(
    X_ptr,          # pointer to X [B, C, K]
    W_ptr,          # pointer to W [K, K]
    P_ptr,          # pointer to P [B, C, K]
    B,              # int: batch size
    C,              # int: concatenated sequence length (M + N)
    K,              # int: hidden dim (H)
    stride_xb,      # int: stride for batch in X
    stride_xc,      # int: stride for seq in X
    stride_xk,      # int: stride for feature in X (should be 1 if contiguous)
    stride_w0,      # int: stride for row in W (K, K) (should be K for row-major contiguous)
    stride_w1,      # int: stride for col in W (should be 1)
    stride_pb,      # int: stride for batch in P
    stride_pc,      # int: stride for seq in P
    stride_pk,      # int: stride for feature in P (should be 1)
    BLOCK_M: tl.constexpr,  # tile size along sequence (C)
    BLOCK_N: tl.constexpr,  # tile size along feature (K)
    BLOCK_K: tl.constexpr,  # tile size along feature for accumulation
):
    # Program ids: each program handles one (batch, block_of_seq)
    b = tl.program_id(0)
    pid_m = tl.program_id(1)

    # Compute offsets for this tile
    m_start = pid_m * BLOCK_M
    m_offsets = m_start + tl.arange(0, BLOCK_M)
    n_offsets = tl.arange(0, BLOCK_N)  # along K output dimension
    # Loop over K dimension in chunks
    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)

        # Pointers for X tile: shape [BLOCK_M, BLOCK_K]
        x_ptrs = X_ptr + b * stride_xb + m_offsets[:, None] * stride_xc + k_offsets[None, :] * stride_xk
        # Mask for X: valid rows and cols
        x_mask = (m_offsets[:, None] < C) & (k_offsets[None, :] < K)

        # Load X tile
        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)

        # Pointers for W tile: shape [BLOCK_K, BLOCK_N]
        # W is [K, K], we want rows k_offsets and cols n_offsets
        w_ptrs = W_ptr + k_offsets[:, None] * stride_w0 + n_offsets[None, :] * stride_w1
        w_mask = (k_offsets[:, None] < K) & (n_offsets[None, :] < K)
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)

        # Accumulate: acc [BLOCK_M, BLOCK_N]
        acc = tl.dot(x_tile, w_tile)

    # After looping over K, store the result
    p_ptrs = P_ptr + b * stride_pb + m_offsets[:, None] * stride_pc + n_offsets[None, :] * stride_pk
    p_mask = (m_offsets[:, None] < C) & (n_offsets[None, :] < K)
    tl.store(p_ptrs, acc, mask=p_mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version of the original run function.
        - Concatenates encoder_hidden_states and hidden_states along sequence dimension.
        - Applies a batched linear projection using a Triton kernel (GEMM).
        - Splits the result back into encoder and image streams.
        """
        # Ensure inputs are on CUDA and contiguous. Triton requires CUDA tensors.
        assert hidden_states.dim() == 3 and encoder_hidden_states.dim() == 3 and process_weight.dim() == 2, \
            "hidden_states: [B, N, H], encoder_hidden_states: [B, M, H], process_weight: [H, H]"
        B = hidden_states.shape[0]
        M = encoder_hidden_states.shape[1]
        N = hidden_states.shape[1]
        K = hidden_states.shape[2]
        assert process_weight.shape[0] == K and process_weight.shape[1] == K, "process_weight must be [H, H] matching hidden_dim"

        device = hidden_states.device
        if device.type != "cuda":
            raise RuntimeError("ModelNew requires CUDA tensors. Move inputs to CUDA (e.g., .to('cuda')).")

        # Concatenate along sequence dimension: [B, M+N, K]
        # We ensure contiguity
        concatenated = torch.cat([encoder_hidden_states, hidden_states], dim=1).contiguous()

        # Ensure process_weight is contiguous [K, K]
        W = process_weight.contiguous()

        # Allocate output P: [B, M+N, K], float32 for compute stability
        # We'll compute in float32; if inputs are float16, cast to float32 for the matmul, then cast outputs back.
        # In most provided workloads, tensors are float32. We keep dtype consistent with concatenated (which is float32).
        # To be safe, we will allocate P with the same dtype as concatenated.
        P = torch.empty((B, M + N, K), device=device, dtype=concatenated.dtype)

        # Compute strides (assuming contiguous tensors)
        # For X: [B, C, K]
        C = M + N
        stride_xb = C * K
        stride_xc = K
        stride_xk = 1

        # For W: [K, K]
        stride_w0 = K
        stride_w1 = 1

        # For P: [B, C, K]
        stride_pb = C * K
        stride_pc = K
        stride_pk = 1

        # Choose tiling parameters
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 32

        # Grid: (B, ceil_div(C, BLOCK_M))
        grid = (B, triton.cdiv(C, BLOCK_M))

        # Launch Triton kernel
        _batched_matmul_kernel[grid](
            concatenated,       # X_ptr
            W,                  # W_ptr
            P,                  # P_ptr
            B, C, K,
            stride_xb, stride_xc, stride_xk,
            stride_w0, stride_w1,
            stride_pb, stride_pc, stride_pk,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        # Split outputs
        processed_encoder = P[:, :M, :]
        processed_hidden = P[:, M:, :]

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
