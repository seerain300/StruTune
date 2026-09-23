import torch
import triton
import triton.language as tl


@triton.jit
def _concat_sequences_kernel(
    out_ptr,        # *fp32, [N, L_total, K]
    enc_ptr,        # *fp32, [N, L_txt, K]
    hid_ptr,        # *fp32, [N, L_img, K]
    N,              # int32
    L_txt,          # int32
    L_img,          # int32
    K,              # int32 (hidden_dim)
    BLOCK_K: tl.constexpr,
):
    # program ids
    pid_n = tl.program_id(0)  # batch
    pid_t = tl.program_id(1)  # total sequence position
    pid_k = tl.program_id(2)  # tile along K

    # Compute K offsets for this program
    k_offsets = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_k = k_offsets < K

    # Determine source tensor based on sequence position
    # If pid_t < L_txt: take from encoder, else take from hidden at offset pid_t - L_txt
    take_encoder = pid_t < L_txt

    # Base pointers for out, enc, hid for this batch row
    out_base = out_ptr + pid_n * L_total * K
    enc_base = enc_ptr + pid_n * L_txt * K
    hid_base = hid_ptr + pid_n * L_img * K

    # Source base pointer
    src_base = enc_base if take_encoder else (hid_base + (pid_t - L_txt) * K)

    # Load a K-tile from source
    src_ptrs = src_base + k_offsets
    vals = tl.load(src_ptrs, mask=mask_k, other=0.0)  # fp32 load

    # Store into out at position (n, t, :)
    out_row_ptrs = out_base + pid_t * K
    out_ptrs = out_row_ptrs + k_offsets
    tl.store(out_ptrs, vals, mask=mask_k)


@triton.jit
def _batched_matmul_tiled_kernel(
    C_ptr,      # *fp32, [N_rows, K] output rows
    A_ptr,      # *fp32, [N_rows, K] input rows (concatenated)
    B_ptr,      # *fp32, [K, K] weight transposed
    N_rows: tl.constexpr,  # number of rows (N * L_total)
    K: tl.constexpr,       # hidden_dim
    BLOCK_M: tl.constexpr, # rows per program (we'll set M=1)
    BLOCK_N: tl.constexpr, # output columns tile
    BLOCK_K: tl.constexpr, # reduction tile
):
    # Each program handles one row and a tile of output columns
    pid_row = tl.program_id(0)  # 0..N_rows-1
    pid_col = tl.program_id(1)  # tile along output columns

    # Output column indices this program handles
    n_offsets = pid_col * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = n_offsets < K  # since output is [N_rows, K], but here we use N_rows == N*L_total and K as dims, adjust:

    # Note: In our usage, we set N_rows = total_rows = N * L_total and K is hidden_dim.
    # The grid will be (total_rows, cdiv(K, BLOCK_N)). So mask_n = n_offsets < K is correct.

    # Accumulator for this row's output tile
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # Reduction over K in chunks of BLOCK_K
    # We use static_range so Triton can compile it; K and BLOCK_K are constexpr.
    for k0 in tl.static_range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < K

        # Load A_row_chunk: element (row=pid_row, k=k_offsets)
        a_ptrs = A_ptr + pid_row * K + k_offsets
        a = tl.load(a_ptrs, mask=mask_k, other=0.0)  # [BLOCK_K]

        # Load B_chunk: elements [k_offsets, n_offsets]
        b_ptrs = B_ptr + k_offsets[:, None] * K + n_offsets[None, :]
        mask_b = mask_k[:, None] & mask_n[None, :]
        b = tl.load(b_ptrs, mask=mask_b, other=0.0)  # [BLOCK_K, BLOCK_N]

        # Accumulate outer product: sum_k a[k] * b[k, :]
        acc += tl.sum(b * a[:, None], axis=0)

    # Store result to C_ptr row pid_row
    c_ptrs = C_ptr + pid_row * K + n_offsets
    tl.store(c_ptrs, acc, mask=mask_n)


def _triton_concat_and_gemm(
    encoder_hidden_states: torch.Tensor,
    hidden_states: torch.Tensor,
    process_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Triton implementation:
    - Concatenate encoder_hidden_states and hidden_states along sequence dim.
    - Compute processed = concatenated @ process_weight.T using Triton GEMM.
    - Split into encoder and hidden streams.
    Returns (processed_encoder, processed_hidden) with same dtypes as inputs.
    """
    assert encoder_hidden_states.is_cuda and hidden_states.is_cuda and process_weight.is_cuda, "All inputs must be CUDA tensors"
    device = encoder_hidden_states.device

    N = encoder_hidden_states.shape[0]
    L_txt = encoder_hidden_states.shape[1]
    K = encoder_hidden_states.shape[2]
    L_img = hidden_states.shape[1]
    L_total = L_txt + L_img

    # Ensure inputs are contiguous and in fp32 for kernel
    enc = encoder_hidden_states.contiguous().to(torch.float32)
    hid = hidden_states.contiguous().to(torch.float32)
    B = process_weight.t().contiguous().to(torch.float32)  # [K, K]

    # 1) Concatenate into [N, L_total, K] using Triton
    out = torch.empty((N, L_total, K), device=device, dtype=torch.float32)

    BLOCK_K = 128  # tile along K
    grid_concat = (N, L_total, triton.cdiv(K, BLOCK_K))
    _concat_sequences_kernel[grid_concat](
        out, enc, hid,
        N, L_txt, L_img, K,
        BLOCK_K=BLOCK_K,
        num_warps=4,
        num_stages=2,
    )

    # 2) GEMM: A_rows = out.reshape(N * L_total, K) @ B -> C_rows [N * L_total, K]
    total_rows = N * L_total
    A_rows = out.reshape(total_rows, K).contiguous()
    C_rows = torch.empty((total_rows, K), device=device, dtype=torch.float32)

    # Choose tile sizes. Use BLOCK_N along output columns, and BLOCK_K along reduction.
    # For typical K in [512, 4096], 128 works well. Masks handle tails.
    BLOCK_N = 128
    BLOCK_K_GEMM = 128 if K >= 128 else 64
    grid_gemm = (total_rows, triton.cdiv(K, BLOCK_N))
    _batched_matmul_tiled_kernel[grid_gemm](
        C_rows, A_rows, B,
        N_rows=total_rows, K=K,
        BLOCK_M=1, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K_GEMM,
        num_warps=4, num_stages=2,
    )

    # 3) Reshape back and split
    processed = C_rows.view(N, L_total, K)
    processed_encoder = processed[:, :L_txt, :]
    processed_hidden = processed[:, L_txt:, :]

    # Cast outputs back to original dtype if needed
    if processed_encoder.dtype != encoder_hidden_states.dtype:
        processed_encoder = processed_encoder.to(encoder_hidden_states.dtype)
    if processed_hidden.dtype != hidden_states.dtype:
        processed_hidden = processed_hidden.to(hidden_states.dtype)

    return processed_encoder, processed_hidden


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Ensure CUDA tensors
        if not hidden_states.is_cuda or not encoder_hidden_states.is_cuda or not process_weight.is_cuda:
            raise RuntimeError("ModelNew expects CUDA tensors for Triton kernels.")
        # Launch Triton kernels
        return _triton_concat_and_gemm(encoder_hidden_states, hidden_states, process_weight)


def run(*args):
    return ModelNew()(*args)
