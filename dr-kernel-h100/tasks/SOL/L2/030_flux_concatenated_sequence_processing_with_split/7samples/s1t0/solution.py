import torch
import triton
import triton.language as tl


@triton.jit
def _matmul_seq_to_dim_kernel(
    E_ptr,            # pointer to encoder_hidden_states: [B, T, D]
    W_ptr,            # pointer to process_weight: [D, D]
    Y_ptr,            # pointer to output: [B, T, D]
    B, T, D,
    sE_b, sE_t, sE_d,  # strides for E
    sY_b, sY_t, sY_d,  # strides for Y
    sW0, sW1,          # strides for W (we expect [D, D])
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Each program handles one (batch, row) pair: output row t for batch b
    b = tl.program_id(0)
    t = tl.program_id(1)

    # Pointers to the b-th batch slice of E and Y
    E_row_ptr = E_ptr + b * sE_b + t * sE_t
    Y_row_ptr = Y_ptr + b * sY_b + t * sY_t

    # Initialize accumulator for output vector of length D
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # Loop over K (hidden_dim) dimension in tiles
    for k0 in range(0, D, BLOCK_K):
        k_range = k0 + tl.arange(0, BLOCK_K)  # [BLOCK_K]
        # Load weight submatrix W_sub: shape [BLOCK_K, BLOCK_N]
        # W is [D, D], we iterate over k0 and accumulate across k
        # We need a matrix of pointers: for each k in BLOCK_K, load corresponding row across N.
        # Construct pointers for W_sub:
        W_sub_ptrs = W_ptr + k_range[:, None] * sW0 + tl.arange(0, BLOCK_N)[None, :] * sW1
        # Mask for valid k indices
        mask_k = k_range < D
        # Load E row vector for these D positions (vector length BLOCK_N)
        # E_row_ptr points to the start of row t for batch b; we load across D
        E_vec_ptrs = E_row_ptr + tl.arange(0, BLOCK_N) * sE_d
        mask_n = tl.arange(0, BLOCK_N) < D
        E_vec = tl.load(E_vec_ptrs, mask=mask_n, other=0.0).to(tl.float32)  # [BLOCK_N]
        # Load W_sub
        W_sub = tl.load(W_sub_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0).to(tl.float32)  # [BLOCK_K, BLOCK_N]
        # Accumulate: acc += sum_k (E_vec[k] * W_sub[k, :])
        # We can use a simple loop over BLOCK_K since it's small; or we can do tl.dot if we reshape appropriately.
        # Compute per-k contribution and accumulate
        # E_vec is [BLOCK_N], W_sub[:, j] is per-k vector across N; we'll do per-k inner product.
        for kk in range(BLOCK_K):
            k_valid = kk + k0 < D
            # vector from W_sub[:, kk] across N
            w_vec = W_sub[kk, :]  # [BLOCK_N]
            e_k = E_vec[kk] if k_valid else 0.0
            acc += e_k * w_vec

    # Store accumulated result to output row
    Y_out_ptrs = Y_row_ptr + tl.arange(0, BLOCK_N) * sY_d
    mask_n = tl.arange(0, BLOCK_N) < D
    # Cast back to original dtype of output tensor; here we assume float32; the original code uses float32.
    tl.store(Y_out_ptrs, acc, mask=mask_n)


@triton.jit
def _matmul_img_to_dim_kernel(
    H_ptr,            # pointer to hidden_states: [B, I, D]
    W_ptr,            # pointer to process_weight: [D, D]
    Y_ptr,            # pointer to output: [B, I, D]
    B, I, D,
    sH_b, sH_i, sH_d,  # strides for H
    sY_b, sY_i, sY_d,  # strides for Y
    sW0, sW1,          # strides for W
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Each program handles one (batch, row) pair: output row i for batch b
    b = tl.program_id(0)
    i = tl.program_id(1)

    # Pointers to the b-th batch slice of H and Y
    H_row_ptr = H_ptr + b * sH_b + i * sH_i
    Y_row_ptr = Y_ptr + b * sY_b + i * sY_i

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for k0 in range(0, D, BLOCK_K):
        k_range = k0 + tl.arange(0, BLOCK_K)
        mask_k = k_range < D

        # Load E row across D
        H_vec_ptrs = H_row_ptr + tl.arange(0, BLOCK_N) * sH_d
        mask_n = tl.arange(0, BLOCK_N) < D
        H_vec = tl.load(H_vec_ptrs, mask=mask_n, other=0.0).to(tl.float32)  # [BLOCK_N]

        # Load W submatrix [BLOCK_K, BLOCK_N]
        W_sub_ptrs = W_ptr + k_range[:, None] * sW0 + tl.arange(0, BLOCK_N)[None, :] * sW1
        W_sub = tl.load(W_sub_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0).to(tl.float32)

        for kk in range(BLOCK_K):
            k_valid = kk + k0 < D
            w_vec = W_sub[kk, :]
            h_k = H_vec[kk] if k_valid else 0.0
            acc += h_k * w_vec

    Y_out_ptrs = Y_row_ptr + tl.arange(0, BLOCK_N) * sY_d
    mask_n = tl.arange(0, BLOCK_N) < D
    tl.store(Y_out_ptrs, acc, mask=mask_n)


def _choose_blocks(D, M, N=None):
    # Choose tile sizes heuristically
    # BLOCK_N: up to 128 or cap at D
    if D <= 64:
        BLOCK_N = 64
    elif D <= 128:
        BLOCK_N = 128
    else:
        BLOCK_N = 128  # cap to 128 for good occupancy
    # BLOCK_K: up to 64
    if D <= 64:
        BLOCK_K = 32
    else:
        BLOCK_K = 64
    # BLOCK_M: rows tile; use 128 if M large, else 64
    BLOCK_M = 128 if M >= 128 else 64
    # num_warps: 4 for smaller, 8 for larger N
    num_warps = 8 if BLOCK_N >= 128 else 4
    # num_stages: 2-4, simple heuristic
    num_stages = 3
    return BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        Triton-optimized version of the original run function:
        - Does not concatenate; computes two batched matmuls directly with Triton.
        - Returns (processed_encoder_hidden_states, processed_hidden_states) with shapes [B, T, D] and [B, I, D].
        """
        # Shapes
        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        D = hidden_states.shape[2]
        assert encoder_hidden_states.shape[2] == D, "Hidden dims must match"
        assert process_weight.shape[0] == D and process_weight.shape[1] == D, "process_weight must be [D, D]"

        # Ensure inputs are contiguous for performance
        E = encoder_hidden_states.contiguous()
        H = hidden_states.contiguous()
        W = process_weight.contiguous()

        # Allocate outputs (float32; original uses float32). We can assume float32 for this benchmark.
        processed_encoder = torch.empty((B, T, D), device=E.device, dtype=torch.float32)
        processed_hidden = torch.empty((B, I, D), device=H.device, dtype=torch.float32)

        # Strides
        sE_b, sE_t, sE_d = E.stride()
        sH_b, sH_i, sH_d = H.stride()
        sY_b, sY_t, sY_d = processed_encoder.stride()
        sY2_b, sY2_i, sY2_d = processed_hidden.stride()
        sW0, sW1 = W.stride()

        # Choose blocks
        BLOCK_M_E, BLOCK_N, BLOCK_K, num_warps, num_stages = _choose_blocks(D, T)
        BLOCK_M_H, _, _, _, _ = _choose_blocks(D, I)  # same N/K

        # Launch kernels: one program per (batch, row)
        grid_encoder = (B, T)
        grid_image = (B, I)

        _matmul_seq_to_dim_kernel[grid_encoder](
            E, W, processed_encoder,
            B, T, D,
            sE_b, sE_t, sE_d,
            sY_b, sY_t, sY_d,
            sW0, sW1,
            BLOCK_M=BLOCK_M_E,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            num_warps=num_warps,
            num_stages=num_stages,
        )

        _matmul_img_to_dim_kernel[grid_image](
            H, W, processed_hidden,
            B, I, D,
            sH_b, sH_i, sH_d,
            sY2_b, sY2_i, sY2_d,
            sW0, sW1,
            BLOCK_M=BLOCK_M_H,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            num_warps=num_warps,
            num_stages=num_stages,
        )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
