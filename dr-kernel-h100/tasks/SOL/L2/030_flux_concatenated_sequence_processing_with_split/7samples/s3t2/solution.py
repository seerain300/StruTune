import torch
import triton
import triton.language as tl

# Triton kernel: concatenate encoder_hidden_states and hidden_states along sequence dim into a single tensor.
# out[n, t, :] = encoder_hidden_states[n, t, :] if t < L_txt else hidden_states[n, t - L_txt, :]
@triton.jit
def _concat_sequences_kernel(
    out_ptr,               # *float32, output [N, L_total, K]
    ehs_ptr,               # *float32, encoder_hidden_states [N, L_txt, K]
    hs_ptr,                # *float32, hidden_states [N, L_img, K]
    N: tl.constexpr,       # int
    L_txt: tl.constexpr,   # int
    K: tl.constexpr,       # int
    BLOCK_K: tl.constexpr, # int
):
    # Grid: (N, L_total, ceil_div(K, BLOCK_K))
    n = tl.program_id(0)
    t = tl.program_id(1)
    k_tile = tl.program_id(2)

    # Compute K offsets for this tile
    k_offsets = k_tile * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_k = k_offsets < K

    # Determine source tensor based on position t
    # t in [0, L_total), if t < L_txt -> ehs; else -> hs at index (t - L_txt)
    is_ehs = t < L_txt

    # Compute source and destination pointers
    # Destination: out[n, t, k_offsets]
    out_row_base = n * L_txt * K + t * K
    out_ptrs = out_ptr + out_row_base + k_offsets

    # Source: either ehs[n, t, k_offsets] or hs[n, t - L_txt, k_offsets]
    if is_ehs:
        src_row_base = n * (L_txt * K) + t * K
        src_ptrs = ehs_ptr + src_row_base + k_offsets
    else:
        src_row_base = n * (L_img * K) + (t - L_txt) * K
        src_ptrs = hs_ptr + src_row_base + k_offsets

    # Load and store (no bias, just copy)
    vals = tl.load(src_ptrs, mask=mask_k, other=0.0)
    tl.store(out_ptrs, vals, mask=mask_k)

# Triton kernel: batched GEMM over rows of A_cat and B=process_weight.T
# A_rows: [N_rows, K], B: [K, K], C_rows: [N_rows, K]
@triton.jit
def _matmul_batched_rows_kernel(
    C_ptr,                # *float32, output rows [N_rows, K]
    A_ptr,                # *float32, input rows [N_rows, K]
    B_ptr,                # *float32, process_weight.T [K, K]
    N_rows: tl.constexpr, # int
    K: tl.constexpr,      # int
    BLOCK_M: tl.constexpr,# int (rows tile, here N_rows dim is 1D so we keep it for API symmetry)
    BLOCK_N: tl.constexpr,# int (output columns tile)
    BLOCK_K: tl.constexpr,# int (inner reduction tile)
):
    # Grid: (N_rows, tiles along K_out, tiles along K_in)
    m = tl.program_id(0)  # row index in A/B/C
    n_tile = tl.program_id(1)
    k_tile = tl.program_id(2)

    n_offsets = n_tile * BLOCK_N + tl.arange(0, BLOCK_N)  # output columns
    k_offsets = k_tile * BLOCK_K + tl.arange(0, BLOCK_K)  # reduction columns

    mask_n = n_offsets < K
    mask_k = k_offsets < K

    # Initialize accumulator
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # Reduction over K
    # A[m, k] for k in [0..K) as tiles; B[k, n] for n_offsets and k_offsets tiles
    for kk in range(0, K, BLOCK_K):
        # Note: for each iteration, we compute A[m, k_offsets] and B[k_offsets, n_offsets]
        # A is [N_rows, K] contiguous per row; B is [K, K] contiguous.
        # A[m, k_offsets]: m is a scalar row index, k_offsets vector
        a_ptrs = A_ptr + m * K + k_offsets
        a_vals = tl.load(a_ptrs, mask=mask_k, other=0.0)  # [BLOCK_K]

        # B[k_offsets, n_offsets]: load a BLOCK_K x BLOCK_N tile
        b_ptrs = B_ptr + k_offsets[:, None] * K + n_offsets[None, :]  # [BLOCK_K, BLOCK_N]
        b_mask = mask_k[:, None] & mask_n[None, :]
        b_vals = tl.load(b_ptrs, mask=b_mask, other=0.0)  # [BLOCK_K, BLOCK_N]

        # Outer product accumulate: acc[n] += sum_k a_vals[k] * b_vals[k, n]
        # Implement elementwise multiply and reduce over K axis
        # Triton doesn't have direct 3D indexing for this, so we do a loop over k in the tile
        for k_idx in range(0, BLOCK_K):
            # If k_idx exceeds K, masked load already zeroed it
            a_elem = a_vals[k_idx]
            b_col = b_vals[k_idx, :]  # [BLOCK_N]
            acc += a_elem * b_col

    # Store the accumulated result for this tile into C[m, n_offsets]
    c_ptrs = C_ptr + m * K + n_offsets
    tl.store(c_ptrs, acc, mask=mask_n)

class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        Triton-optimized forward:
        - Concatenate encoder_hidden_states and hidden_states into a single tensor along sequence dimension using Triton.
        - Apply linear projection with process_weight.T using a Triton batched GEMM kernel.
        - Split outputs back into encoder and hidden streams.

        All numeric work is done by Triton kernels; no torch.cat or torch.matmul in host code.
        """
        # Ensure inputs are on CUDA and contiguous
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Inputs must be CUDA tensors"
        hidden_states = hidden_states.contiguous()
        encoder_hidden_states = encoder_hidden_states.contiguous()
        process_weight = process_weight.contiguous()

        N = hidden_states.shape[0]
        L_txt = encoder_hidden_states.shape[1]
        L_img = hidden_states.shape[1]
        K = hidden_states.shape[2]
        L_total = L_txt + L_img

        # 1) Concatenate along sequence dimension using Triton
        # Allocate output concatenated [N, L_total, K]
        concatenated = torch.empty((N, L_total, K), device=hidden_states.device, dtype=hidden_states.dtype)

        # Launch _concat_sequences_kernel
        # Choose BLOCK_K for K tiling. 64 or 128 work well for typical K. Use 64 for generality.
        BLOCK_K = 64
        grid_concat = (N, L_total, triton.cdiv(K, BLOCK_K))
        _concat_sequences_kernel[grid_concat](
            concatenated, encoder_hidden_states, hidden_states,
            N, L_txt, K, BLOCK_K,
            num_warps=4,  # reasonable default
            num_stages=2,
        )

        # 2) Apply linear projection via Triton GEMM: processed = concatenated @ process_weight.T
        # process_weight.T is [K, K], contiguous
        B = process_weight.t().contiguous()  # [K, K]

        # Flatten concatenated to [N_rows, K], where N_rows = N * L_total
        N_rows = N * L_total
        # Allocate output rows [N_rows, K]
        C_rows = torch.empty((N_rows, K), device=hidden_states.device, dtype=hidden_states.dtype)

        # Launch _matmul_batched_rows_kernel
        # We treat each row of concatenated (i.e., each (n, t) position) as a separate "batch" row to be multiplied by B.
        BLOCK_M = 1  # one row per program instance
        BLOCK_N = 128  # output columns tile
        BLOCK_K = 64   # reduction tile
        grid_gemm = (N_rows, triton.cdiv(K, BLOCK_N), triton.cdiv(K, BLOCK_K))
        _matmul_batched_rows_kernel[grid_gemm](
            C_rows, concatenated.view(N_rows, K), B,
            N_rows, K,
            BLOCK_M, BLOCK_N, BLOCK_K,
            num_warps=4,
            num_stages=2,
        )

        # 3) Reshape back to [N, L_total, K] and split into encoder and hidden parts
        processed = C_rows.view(N, L_total, K)
        processed_encoder = processed[:, :L_txt, :]
        processed_hidden = processed[:, L_txt:, :]

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
