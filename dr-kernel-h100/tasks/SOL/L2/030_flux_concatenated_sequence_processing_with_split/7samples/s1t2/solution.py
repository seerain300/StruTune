import torch
import triton
import triton.language as tl

# Kernel: for each (batch, seq_row) tile, compute output of size [BLOCK_M, N] = [tile along sequence, D]
# Input A is a vector per row (per batch): A[b, i, :] with size (1, D) in this setting, but we generalize to (M, D).
# Weight W is [D, D], and we want output Y = A @ W (no bias).
# In our usage, A is either hidden_states[:, :, :] or encoder_hidden_states[:, :, :], and we launch one program per tile of rows.

@triton.jit
def _matmul_seq_to_dim_kernel(
    A_ptr,            # *ptr to input A: shape [M, D], where M is the tile of sequence length
    W_ptr,            # *ptr to weight: shape [D, D]
    Y_ptr,            # *ptr to output: shape [M, D]
    M: tl.constexpr,  # number of rows in A (and Y) per batch, here typically T or I
    N: tl.constexpr,  # hidden_dim (columns), equals D
    K: tl.constexpr,  # also D, but keep for clarity
    # strides for A
    sA_b, sA_m, sA_k,
    # strides for W
    sW0, sW1,
    # strides for Y
    sY_b, sY_m, sY_k,
    # tiling parameters
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_WARPS: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    # Program ids: tile along sequence rows and batch
    pid_m = tl.program_id(axis=0)
    pid_b = tl.program_id(axis=1)

    # Offsets for this tile
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = tl.arange(0, BLOCK_N)
    k_offsets = tl.arange(0, BLOCK_K)

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K (hidden_dim) in tiles
    # We use a while loop to cover all K even when M is not a multiple of BLOCK_M (not needed here since M is the sequence dimension and is passed correctly).
    k = 0
    while k < K:
        k_offsets_curr = k + k_offsets  # shape [BLOCK_K]
        # Build pointers for A block: [BLOCK_M, BLOCK_K]
        # A[b, m, k] with b fixed by pid_b
        A_ptrs = A_ptr + pid_b * sA_b + m_offsets[:, None] * sA_m + k_offsets_curr[None, :] * sA_k
        # Mask for valid m,k within M and K
        A_mask = (m_offsets[:, None] < M) & (k_offsets_curr[None, :] < K)
        A_block = tl.load(A_ptrs, mask=A_mask, other=0.0)  # [BLOCK_M, BLOCK_K]

        # Build pointers for W block: [BLOCK_K, BLOCK_N]
        # W[k, n] for k in [k_offsets_curr], n in [n_offsets]
        W_ptrs = W_ptr + k_offsets_curr[:, None] * sW0 + n_offsets[None, :] * sW1
        W_mask = (k_offsets_curr[:, None] < K) & (n_offsets[None, :] < N)
        W_block = tl.load(W_ptrs, mask=W_mask, other=0.0)  # [BLOCK_K, BLOCK_N]

        # Accumulate
        acc += tl.dot(A_block, W_block)  # [BLOCK_M, BLOCK_N]
        k += BLOCK_K

    # Store result
    Y_ptrs = Y_ptr + pid_b * sY_b + m_offsets[:, None] * sY_m + n_offsets[None, :] * sY_k
    Y_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    tl.store(Y_ptrs, acc, mask=Y_mask)


@triton.jit
def _matmul_img_to_dim_kernel(
    H_ptr,            # *ptr to input hidden_states: shape [B, I, D]
    W_ptr,            # *ptr to weight: shape [D, D]
    Y_ptr,            # *ptr to output: shape [B, I, D]
    B: tl.constexpr,  # batch size
    I: tl.constexpr,  # img_seq_len
    D: tl.constexpr,  # hidden_dim
    # strides for H
    sH_b, sH_i, sH_d,
    # strides for W
    sW0, sW1,
    # strides for Y
    sY_b, sY_i, sY_d,
    # tiling parameters
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_WARPS: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    # This is a specialized version of _matmul_seq_to_dim_kernel used for hidden_states.
    # We launch grid = (ceil_div(I, BLOCK_M), B), each program handles one tile along I for a given batch.
    pid_m = tl.program_id(axis=0)  # tile along I (sequence rows)
    pid_b = tl.program_id(axis=1)  # batch

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # along I
    n_offsets = tl.arange(0, BLOCK_N)  # along D
    k_offsets = tl.arange(0, BLOCK_K)  # along K (D)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    k = 0
    while k < D:
        k_offsets_curr = k + k_offsets
        # H[b, i, k] -> [BLOCK_M, BLOCK_K]
        H_ptrs = H_ptr + pid_b * sH_b + m_offsets[:, None] * sH_i + k_offsets_curr[None, :] * sH_d
        H_mask = (m_offsets[:, None] < I) & (k_offsets_curr[None, :] < D)
        H_block = tl.load(H_ptrs, mask=H_mask, other=0.0)

        # W[k, n] -> [BLOCK_K, BLOCK_N]
        W_ptrs = W_ptr + k_offsets_curr[:, None] * sW0 + n_offsets[None, :] * sW1
        W_mask = (k_offsets_curr[:, None] < D) & (n_offsets[None, :] < D)
        W_block = tl.load(W_ptrs, mask=W_mask, other=0.0)

        acc += tl.dot(H_block, W_block)
        k += BLOCK_K

    # Store Y[b, i, n]
    Y_ptrs = Y_ptr + pid_b * sY_b + m_offsets[:, None] * sY_i + n_offsets[None, :] * sY_d
    Y_mask = (m_offsets[:, None] < I) & (n_offsets[None, :] < D)
    tl.store(Y_ptrs, acc, mask=Y_mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        Triton-optimized version of the original run().

        - Avoids concatenation by computing two separate batched matmuls:
          - yA = encoder_hidden_states @ process_weight.T -> [B, T, D]
          - yB = hidden_states @ process_weight.T        -> [B, I, D]
        - This is mathematically equivalent to concatenating, applying matmul, and splitting.
        """
        # Ensure inputs are contiguous and on CUDA
        B = hidden_states.shape[0]
        I = hidden_states.shape[1]
        T = encoder_hidden_states.shape[1]
        D = hidden_states.shape[2]
        assert D == encoder_hidden_states.shape[2], "hidden_dim must match between inputs and weight"
        assert process_weight.shape[0] == D and process_weight.shape[1] == D, "process_weight must be [D, D]"

        # Make sure all tensors are on CUDA and contiguous
        device = hidden_states.device
        assert device.type == 'cuda', "Triton requires CUDA tensors. Move inputs to CUDA."
        # Cast to float32 for numerical stability in kernels (Triton default math). If inputs are fp16/bf16, cast to fp32.
        # Note: We will compute in fp32; outputs will be fp32. If you need original dtype, you can cast back.
        # However, for correctness comparison, fp32 is fine here.
        H = hidden_states.contiguous().to(torch.float32)
        E = encoder_hidden_states.contiguous().to(torch.float32)
        W = process_weight.contiguous().to(torch.float32)

        # Allocate outputs
        processed_encoder = torch.empty((B, T, D), device=device, dtype=torch.float32)
        processed_hidden = torch.empty((B, I, D), device=device, dtype=torch.float32)

        # Tiling parameters: choose BLOCK_K=32 to ensure it divides common D values (e.g., 32, 64, 128, 256, ..., up to 4096)
        # BLOCK_M and BLOCK_N can be 64 for good throughput. These are reasonable defaults.
        BLOCK_M = 128  # along sequence rows (T or I)
        BLOCK_N = 64   # along hidden_dim columns (D)
        BLOCK_K = 32   # along K (D), ensure divides D for correctness
        NUM_WARPS = 4
        NUM_STAGES = 2

        # Launch kernel for encoder hidden states: output [B, T, D]
        grid_encoder = (triton.cdiv(T, BLOCK_M), B)
        _matmul_seq_to_dim_kernel[grid_encoder](
            E, W, processed_encoder,
            T, D, D,
            E.stride(0), E.stride(1), E.stride(2),
            W.stride(0), W.stride(1),
            processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            NUM_WARPS=NUM_WARPS, NUM_STAGES=NUM_STAGES,
        )

        # Launch kernel for image hidden states: output [B, I, D]
        grid_hidden = (triton.cdiv(I, BLOCK_M), B)
        _matmul_img_to_dim_kernel[grid_hidden](
            H, W, processed_hidden,
            B, I, D,
            H.stride(0), H.stride(1), H.stride(2),
            W.stride(0), W.stride(1),
            processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            NUM_WARPS=NUM_WARPS, NUM_STAGES=NUM_STAGES,
        )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
