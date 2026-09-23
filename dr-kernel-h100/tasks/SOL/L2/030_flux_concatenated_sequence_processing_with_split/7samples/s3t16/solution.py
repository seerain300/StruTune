import torch
import triton
import triton.language as tl


@triton.jit
def _concat_sequences_kernel(out_ptr, encoder_ptr, hidden_ptr,
                             N, L_txt, L_img, K,
                             BLOCK_K: tl.constexpr):
    # Grid: (N, L_total, tiles_along_K)
    n = tl.program_id(0)
    t = tl.program_id(1)
    k_tile = tl.program_id(2)

    # Compute k offsets for this tile
    k_offsets = k_tile * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_k = k_offsets < K

    # Pointers to out[n, t, :]
    out_row_ptr = out_ptr + n * (L_txt + L_img) * K + t * K + k_offsets

    # Select source based on whether t < L_txt
    # If t < L_txt: read from encoder[n, t, :]
    # Else: read from hidden[n, t - L_txt, :]
    is_encoder = t < L_txt

    # Compute source pointers. For hidden, index is t - L_txt.
    # We will choose a source pointer based on is_encoder.
    # Since Triton supports control flow, we branch and compute only one path.
    if is_encoder:
        src_row_ptr = encoder_ptr + n * L_txt * K + t * K + k_offsets
    else:
        src_row_idx = t - L_txt
        src_row_ptr = hidden_ptr + n * L_img * K + src_row_idx * K + k_offsets

    # Load and store
    vals = tl.load(src_row_ptr, mask=mask_k, other=0.0)
    tl.store(out_row_ptr, vals, mask=mask_k)


@triton.jit
def _matmul_row_kernel(C_ptr, A_ptr, B_ptr,
                       N_rows, K: tl.constexpr, BLOCK_K: tl.constexpr):
    # One program per row
    row_id = tl.program_id(0)
    # Guard: row_id < N_rows (in case grid is larger than N_rows)
    # We'll assume grid == (N_rows,), but include guard for safety.
    if row_id >= N_rows:
        return

    # Initialize output row accumulator in fp32
    acc = tl.zeros((K,), dtype=tl.float32)

    # Reduction over K in chunks of BLOCK_K
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < K

        # Load A[row_id, k_offsets] -> shape [BLOCK_K]
        a = tl.load(A_ptr + row_id * K + k_offsets, mask=mask_k, other=0.0)

        # Load B[k_offsets, :] -> shape [BLOCK_K, K]
        # B is [K, K], so load rows k_offsets across columns [0..K-1]
        b = tl.load(B_ptr + k_offsets[:, None] * K + tl.arange(0, K)[None, :], mask=mask_k[:, None], other=0.0)

        # Compute partial dot: sum over BLOCK_K of a[i] * b[i, :]
        # Result is [K]
        acc += tl.sum(a[:, None] * b, axis=0)

    # Store result to C[row_id, :]
    tl.store(C_ptr + row_id * K + tl.arange(0, K), acc, mask=True)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                encoder_hidden_states: torch.Tensor,
                process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version of the original run function.
        - Concatenates encoder_hidden_states and hidden_states along the sequence dimension using a Triton kernel.
        - Applies linear projection (no bias) using a Triton per-row GEMM kernel.
        - Splits the result back into two streams.

        Returns:
        (processed_encoder_hidden_states, processed_hidden)
        """
        # Shapes
        N = hidden_states.shape[0]
        L_txt = encoder_hidden_states.shape[1]
        L_img = hidden_states.shape[1]
        K = hidden_states.shape[2]
        L_total = L_txt + L_img

        device = hidden_states.device
        dtype = hidden_states.dtype

        # Allocate output for concatenation: [N, L_total, K]
        out = torch.empty((N, L_total, K), device=device, dtype=dtype)

        # 1) Triton concatenation: out = concat(encoder_hidden_states, hidden_states)
        # Grid: (N, L_total, tiles_along_K). Use BLOCK_K=128 for typical K (512, 1024, 2048, 4096).
        BLOCK_K = 128
        grid_concat = (N, L_total, triton.cdiv(K, BLOCK_K))
        _concat_sequences_kernel[grid_concat](
            out, encoder_hidden_states, hidden_states,
            N, L_txt, L_img, K,
            BLOCK_K=BLOCK_K,
        )

        # 2) Triton GEMM per-row: compute processed = out @ process_weight.T
        # Prepare A_rows = out.view(N * L_total, K)
        A_rows = out.view(N * L_total, K).contiguous()  # [N_rows, K]
        B = process_weight.t().contiguous()  # [K, K]

        # Allocate C_rows as fp32 for accumulation
        C_rows = torch.empty((N * L_total, K), device=device, dtype=torch.float32)

        # Launch per-row GEMM. One program per row.
        grid_gemm = (N * L_total,)
        # Choose BLOCK_K for reduction. Use 256 for larger K, 128 for smaller K.
        BLOCK_K_GEMM = 256 if K >= 256 else 128
        _matmul_row_kernel[grid_gemm](
            C_rows, A_rows, B,
            N * L_total, K=K, BLOCK_K=BLOCK_K_GEMM,
            num_warps=4, num_stages=2,
        )

        # Cast back to original dtype if necessary
        if C_rows.dtype != dtype:
            C_rows = C_rows.to(dtype)

        # Reshape back to [N, L_total, K] and split
        processed = C_rows.view(N, L_total, K)
        processed_encoder = processed[:, :L_txt, :]
        processed_hidden = processed[:, L_txt:, :]

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
