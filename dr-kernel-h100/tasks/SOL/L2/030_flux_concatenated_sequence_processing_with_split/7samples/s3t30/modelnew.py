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
    mask = k_offsets < K

    # Pointers for the current row in output and source
    out_row_ptr = out_ptr + n * (L_txt + L_img) * K + t * K
    if t < L_txt:
        enc_row_ptr = enc_ptr + n * L_txt * K + t * K
        # Load from encoder and store
        x = tl.load(enc_row_ptr + k_offsets, mask=mask, other=0.0)
        tl.store(out_row_ptr, x, mask=mask)
    else:
        hid_row_ptr = hid_ptr + n * L_img * K + (t - L_txt) * K
        # Load from hidden and store
        x = tl.load(hid_row_ptr + k_offsets, mask=mask, other=0.0)
        tl.store(out_row_ptr, x, mask=mask)


@triton.jit
def _matmul_row_kernel(C_row_ptr, A_row_ptr, B_ptr, K, BLOCK_K: tl.constexpr):
    # Each program handles one output row (across K columns), iterating reduction with while loop.
    # A_row_ptr: [K], B_ptr: [K, K], C_row_ptr: [K]
    # We accumulate in fp32
    acc = tl.zeros((BLOCK_K,), dtype=tl.float32)
    kk = 0
    while kk < K:
        b_row_ptr = B_ptr + kk + tl.arange(0, BLOCK_K)  # [BLOCK_K]
        a_sub = tl.load(A_row_ptr + kk + tl.arange(0, BLOCK_K), mask=tl.arange(0, BLOCK_K) < K, other=0.0)  # [BLOCK_K]
        b_sub = tl.load(b_row_ptr, mask=tl.arange(0, BLOCK_K) < K, other=0.0)  # [BLOCK_K]
        # Cast to fp32 for accumulation
        acc += (a_sub.to(tl.float32)) * (b_sub.to(tl.float32))
        kk += BLOCK_K
    # Store accumulated results back into C_row_ptr
    out_k = tl.arange(0, BLOCK_K)
    tl.store(C_row_ptr + out_k, acc, mask=out_k < K)


def _triton_concat_and_gemm(encoder_hidden_states: torch.Tensor, hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # Ensure CUDA tensors and consistent dtype; accumulate in fp32
    device = encoder_hidden_states.device
    N = encoder_hidden_states.shape[0]
    L_txt = encoder_hidden_states.shape[1]
    L_img = hidden_states.shape[1]
    K = encoder_hidden_states.shape[2]
    L_total = L_txt + L_img

    # Allocate concatenated tensor [N, L_total, K]
    concatenated = torch.empty((N, L_total, K), device=device, dtype=torch.float32)
    # Launch concatenation kernel
    BLOCK_K = 128
    grid = (N, L_total, triton.cdiv(K, BLOCK_K))
    _concat_sequences_kernel[grid](
        concatenated, encoder_hidden_states.contiguous().to(torch.float32), hidden_states.contiguous().to(torch.float32),
        N, L_txt, L_img, K,
        BLOCK_K=BLOCK_K,
        num_warps=4,
        num_stages=2,
    )

    # Prepare inputs for GEMM: A_rows = concatenated viewed as [N_rows, K], B = process_weight.T as [K, K]
    A_rows = concatenated.reshape(N * L_total, K).contiguous()
    B = process_weight.t().contiguous()  # [K, K]
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

    # Reshape back to [N, L_total, K] and split
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
        # This implementation uses Triton kernels for concatenation and GEMM.
        # Ensure inputs are on CUDA for Triton; if not, you can move them, but the evaluation provides CUDA tensors.
        return _triton_concat_and_gemm(encoder_hidden_states, hidden_states, process_weight)