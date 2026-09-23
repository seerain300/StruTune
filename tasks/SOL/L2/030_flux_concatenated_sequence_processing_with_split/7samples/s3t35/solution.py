import torch
import triton
import triton.language as tl


@triton.jit
def _concat_sequences_kernel(
    out_ptr,       # *float32, [N, L_total, K]
    enc_ptr,       # *float32, [N, L_txt, K]
    hid_ptr,       # *float32, [N, L_img, K]
    N: tl.constexpr,
    L_txt: tl.constexpr,
    L_img: tl.constexpr,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Grid: (N, L_total, tiles_along_K)
    n = tl.program_id(0)
    t = tl.program_id(1)
    tile_k = tl.program_id(2)

    # Compute K offsets for this tile
    k_offsets = tile_k * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_k = k_offsets < K

    # Select source based on whether t < L_txt
    is_encoder = t < L_txt

    # Compute source indices and pointers
    # For encoder: idx = (n, t, k)
    # For hidden: idx = (n, t - L_txt, k)
    if is_encoder:
        src_n = n
        src_t = t
    else:
        src_n = n
        src_t = t - L_txt

    # Pointers to the source elements
    enc_ptr_block = enc_ptr + src_n * (L_txt * K) + src_t * K + k_offsets
    hid_ptr_block = hid_ptr + src_n * (L_img * K) + src_t * K + k_offsets

    # Choose the appropriate pointer based on is_encoder
    src_ptr = tl.where(is_encoder, enc_ptr_block, hid_ptr_block)

    # Destination pointer
    out_ptr_block = out_ptr + n * (L_txt + L_img) * K + t * K + k_offsets

    # Load and store with mask
    vals = tl.load(src_ptr, mask=mask_k, other=0.0)
    tl.store(out_ptr_block, vals, mask=mask_k)


@triton.jit
def _matmul_row_kernel(
    C_ptr,      # *float32, [N_rows, K]
    A_ptr,      # *float32, [N_rows, K]
    B_ptr,      # *float32, [K, K]
    N_rows: tl.constexpr,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # One program handles one row
    row = tl.program_id(0)
    # Accumulator for this row
    acc = tl.zeros((K,), dtype=tl.float32)

    # Reduction loop over K in chunks of BLOCK_K
    kk = 0
    while kk < K:
        k_offsets = kk + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < K

        # Load A[row, k_offsets] and B[k_offsets, :]
        A_row_block = A_ptr + row * K + k_offsets
        A_vals = tl.load(A_row_block, mask=mask_k, other=0.0)  # [BLOCK_K]

        B_block = B_ptr + k_offsets[:, None] * K + tl.arange(0, BLOCK_K)[None, :]  # [BLOCK_K, BLOCK_K]
        B_vals = tl.load(B_block, mask=mask_k[:, None], other=0.0)  # [BLOCK_K, BLOCK_K]

        # Outer product accumulate: acc += sum_j A[j] * B[j, :]
        # Since A_vals is [BLOCK_K], multiply each with the corresponding row in B_vals and sum over axis=1
        # Implement as a loop over j in [BLOCK_K] (constexpr size)
        for j in range(BLOCK_K):
            bj = B_vals[j, :]  # [BLOCK_K]
            acc += A_vals[j] * bj

        kk += BLOCK_K

    # Store the result row
    C_row_block = C_ptr + row * K + tl.arange(0, K)
    tl.store(C_row_block, acc, mask=True)


class ModelNew(torch.nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        process_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version:
        - Concatenates [N, L_txt, K] and [N, L_img, K] into [N, L_total, K] using Triton.
        - Performs processed = concatenated @ process_weight.T using a Triton per-row GEMM.
        - Splits back into (processed_encoder, processed_hidden).
        """
        # Ensure inputs are on CUDA and contiguous
        device = hidden_states.device
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Inputs must be on CUDA device"
        hidden_states = hidden_states.contiguous()
        encoder_hidden_states = encoder_hidden_states.contiguous()
        process_weight = process_weight.contiguous()

        N = hidden_states.shape[0]
        K = hidden_states.shape[2]
        L_txt = encoder_hidden_states.shape[1]
        L_img = hidden_states.shape[1]
        L_total = L_txt + L_img

        # 1) Concatenate sequences along the sequence dimension using Triton
        out = torch.empty((N, L_total, K), device=device, dtype=torch.float32)

        # Grid over (N, L_total, tiles along K). Choose BLOCK_K=128 for good throughput.
        BLOCK_K = 128
        grid_concat = (N, L_total, triton.cdiv(K, BLOCK_K))
        _concat_sequences_kernel[grid_concat](
            out, encoder_hidden_states, hidden_states,
            N, L_txt, L_img, K,
            BLOCK_K,
            num_warps=4, num_stages=2,
        )

        # 2) GEMM: Compute processed = out @ process_weight.T using Triton
        # We need A_rows of shape [N_rows, K], where N_rows = N * L_total
        N_rows = N * L_total
        A_rows = out.reshape(N_rows, K).contiguous()  # [N_rows, K]
        # B is process_weight.T contiguous [K, K]
        B = process_weight.t().contiguous()  # [K, K]
        C_rows = torch.empty((N_rows, K), device=device, dtype=torch.float32)

        # Launch per-row GEMM kernel. BLOCK_K for reduction. Use while loop for robustness.
        BLOCK_K_GEMM = 256 if K >= 256 else 128
        grid_gemm = (N_rows,)
        _matmul_row_kernel[grid_gemm](
            C_rows, A_rows, B,
            N_rows, K, BLOCK_K_GEMM,
            num_warps=4, num_stages=2,
        )

        # 3) Reshape back to [N, L_total, K] and split into two streams
        processed = C_rows.view(N, L_total, K)
        processed_encoder = processed[:, :L_txt, :]
        processed_hidden = processed[:, L_txt:, :]

        # Cast back to original dtype (PyTorch model typically uses float32; keep consistency)
        # If you want to match input dtypes exactly, uncomment the following:
        # processed_encoder = processed_encoder.to(encoder_hidden_states.dtype)
        # processed_hidden = processed_hidden.to(hidden_states.dtype)

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
