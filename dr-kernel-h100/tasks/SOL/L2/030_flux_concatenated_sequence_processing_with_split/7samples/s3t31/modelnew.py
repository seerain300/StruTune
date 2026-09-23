import torch
import triton
import triton.language as tl


@triton.jit
def _concat_sequences_kernel(out_ptr, enc_ptr, hid_ptr, N, L_txt, L_img, K, BLOCK_K: tl.constexpr):
    # Grid: (N, L_total, tiles_along_K)
    n = tl.program_id(0)
    t = tl.program_id(1)
    k_tile = tl.program_id(2)

    # Compute K offsets for this tile
    k_offsets = k_tile * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_k = k_offsets < K

    # Output pointer for row (n, t)
    out_row_ptr = out_ptr + (n * (L_txt + L_img) + t) * K + k_offsets

    if t < L_txt:
        # Source from encoder_hidden_states[n, t, :]
        enc_row_ptr = enc_ptr + (n * L_txt + t) * K + k_offsets
        x = tl.load(enc_row_ptr, mask=mask_k, other=0.0)
        tl.store(out_row_ptr, x, mask=mask_k)
    else:
        # Source from hidden_states[n, t - L_txt, :]
        hid_row_ptr = hid_ptr + (n * L_img + (t - L_txt)) * K + k_offsets
        x = tl.load(hid_row_ptr, mask=mask_k, other=0.0)
        tl.store(out_row_ptr, x, mask=mask_k)


@triton.jit
def _matmul_row_kernel(C_row_ptr, A_row_ptr, B_ptr, K, BLOCK_K: tl.constexpr):
    # Each program handles one output row across K columns. Iterate reduction with while loop.
    # A_row_ptr: [K], B_ptr: [K, K], C_row_ptr: [K]
    kk = 0
    acc = tl.zeros((BLOCK_K,), dtype=tl.float32)
    while kk < K:
        # Load a chunk of A_row
        a_sub = tl.load(A_row_ptr + kk + tl.arange(0, BLOCK_K), mask=tl.arange(0, BLOCK_K) < K, other=0.0)
        # Load corresponding chunk of B row (process_weight.T: [K, K])
        b_sub = tl.load(B_ptr + kk + tl.arange(0, BLOCK_K), mask=tl.arange(0, BLOCK_K) < K, other=0.0)
        acc += a_sub * b_sub
        kk += BLOCK_K
    # Store the accumulated result row
    k_offsets = tl.arange(0, BLOCK_K)
    tl.store(C_row_ptr + k_offsets, acc, mask=k_offsets < K)


def _triton_concat_and_gemm(encoder_hidden_states: torch.Tensor, hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # Ensure CUDA tensors
    device = encoder_hidden_states.device
    N = encoder_hidden_states.shape[0]
    L_txt = encoder_hidden_states.shape[1]
    L_img = hidden_states.shape[1]
    K = encoder_hidden_states.shape[2]
    assert hidden_states.shape[0] == N and hidden_states.shape[2] == K, "Mismatched batch or hidden_dim."
    assert process_weight.shape[0] == K and process_weight.shape[1] == K, "process_weight must be [K, K]."

    # 1) Concatenate in Triton: [N, L_total, K]
    L_total = L_txt + L_img
    concatenated = torch.empty((N, L_total, K), device=device, dtype=torch.float32)

    # Triton kernel launch: grid over (N, L_total, K tiles)
    BLOCK_K = 128
    grid = (N, L_total, triton.cdiv(K, BLOCK_K))
    _concat_sequences_kernel[grid](
        concatenated, encoder_hidden_states.contiguous().to(torch.float32), hidden_states.contiguous().to(torch.float32),
        N, L_txt, L_img, K,
        BLOCK_K=BLOCK_K,
        num_warps=4,
        num_stages=2,
    )

    # 2) GEMM in Triton: A_rows = concatenated viewed as [N_rows, K], B = process_weight.T as [K, K]
    A_rows = concatenated.reshape(N * L_total, K).contiguous().to(torch.float32)  # [N_rows, K]
    B = process_weight.t().contiguous().to(torch.float32)  # [K, K]

    # Output rows buffer for GEMM
    C_rows = torch.empty((N * L_total, K), device=device, dtype=torch.float32)

    # Launch Triton GEMM per-row kernel
    grid_gemm = (N * L_total,)
    _matmul_row_kernel[grid_gemm](
        C_rows, A_rows, B,
        K,
        BLOCK_K=BLOCK_K,
        num_warps=4,
        num_stages=2,
    )

    # 3) Reshape back and split
    processed = C_rows.view(N, L_total, K)
    processed_encoder = processed[:, :L_txt, :]
    processed_hidden = processed[:, L_txt:, :]

    # Cast outputs back to original dtypes if needed
    if processed_encoder.dtype != encoder_hidden_states.dtype:
        processed_encoder = processed_encoder.to(encoder_hidden_states.dtype)
    if processed_hidden.dtype != hidden_states.dtype:
        processed_hidden = processed_hidden.to(hidden_states.dtype)

    return processed_encoder, processed_hidden


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Use Triton for all numeric work
        return _triton_concat_and_gemm(encoder_hidden_states, hidden_states, process_weight)