import torch
import triton
import triton.language as tl


@triton.jit
def triton_concat_sequences_kernel(out_ptr, enc_ptr, hid_ptr,
                                   N, L_txt, L_img, K,
                                   out_stride_n, out_stride_t, out_stride_k,
                                   enc_stride_n, enc_stride_t, enc_stride_k,
                                   hid_stride_n, hid_stride_t, hid_stride_k,
                                   BLOCK_K: tl.constexpr):
    # Grid: (N, L_total, tiles along K)
    pid_n = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_k = tl.program_id(2)

    # Compute K indices for this tile
    k_offsets = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_k = k_offsets < K

    # Select source based on pid_t
    # If pid_t < L_txt: copy from enc; else: copy from hid
    # Compute pointers
    # Note: out index for (n, t, k) is n*out_stride_n + t*out_stride_t + k*out_stride_k
    # enc index for (n, t, k) is n*enc_stride_n + t*enc_stride_t + k*enc_stride_k
    # hid index for (n, t, k) is n*hid_stride_n + t*hid_stride_t + k*hid_stride_k

    is_text = pid_t < L_txt
    # Build masks
    mask = mask_k
    # Compute base offsets
    out_base = pid_n * out_stride_n + pid_t * out_stride_t
    # Load from encoder if is_text else hidden
    # We'll use scalar branching (ok in Triton) based on is_text
    if is_text:
        enc_base = pid_n * enc_stride_n + pid_t * enc_stride_t
        enc_ptrs = enc_ptr + enc_base + k_offsets * enc_stride_k
        out_ptrs = out_ptr + out_base + k_offsets * out_stride_k
        vals = tl.load(enc_ptrs, mask=mask, other=0.0)
        tl.store(out_ptrs, vals, mask=mask)
    else:
        hid_base = pid_n * hid_stride_n + (pid_t - L_txt) * hid_stride_t
        hid_ptrs = hid_ptr + hid_base + k_offsets * hid_stride_k
        out_ptrs = out_ptr + out_base + k_offsets * out_stride_k
        vals = tl.load(hid_ptrs, mask=mask, other=0.0)
        tl.store(out_ptrs, vals, mask=mask)


@triton.jit
def triton_matmul_rows_cols_kernel(C_ptr, A_ptr, B_ptr,
                                   N_ROWS, K,
                                   C_stride_m, C_stride_n,
                                   A_stride_m, A_stride_k,
                                   B_stride_k, B_stride_n,
                                   BLOCK_M: tl.constexpr,  # number of rows per program (we set 1 to compute per-row)
                                   BLOCK_N: tl.constexpr,  # output columns per program
                                   BLOCK_K: tl.constexpr):  # reduction tile
    # Compute which rows this program handles
    pid_m = tl.program_id(0)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = rows < N_ROWS

    # Output columns handled by this program
    pid_n = tl.program_id(1)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = cols < K  # since output is [N_ROWS, K], mask_n = cols < K

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension in chunks
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < K

        # Load A rows x K chunk
        A_ptrs = A_ptr + rows[:, None] * A_stride_m + k_offsets[None, :] * A_stride_k
        A_mask = mask_m[:, None] & mask_k[None, :]
        A_vals = tl.load(A_ptrs, mask=A_mask, other=0.0)  # [BLOCK_M, BLOCK_K]

        # Load B K x cols chunk
        B_ptrs = B_ptr + k_offsets[:, None] * B_stride_k + cols[None, :] * B_stride_n
        B_mask = mask_k[:, None] & mask_n[None, :]
        B_vals = tl.load(B_ptrs, mask=B_mask, other=0.0)  # [BLOCK_K, BLOCK_N]

        # Accumulate
        acc += tl.dot(A_vals, B_vals)

    # Store results
    C_ptrs = C_ptr + rows[:, None] * C_stride_m + cols[None, :] * C_stride_n
    C_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(C_ptrs, acc, mask=C_mask)


def _triton_concat_and_linear(hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> torch.Tensor:
    # Ensure tensors are contiguous and on CUDA
    assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Tensors must be on CUDA device"
    N = hidden_states.shape[0]
    L_txt = encoder_hidden_states.shape[1]
    L_img = hidden_states.shape[1]
    K = hidden_states.shape[2]
    device = hidden_states.device

    # Allocate concatenated output [N, L_total, K]
    L_total = L_txt + L_img
    out = torch.empty((N, L_total, K), device=device, dtype=torch.float32)

    # Launch concat kernel: grid over (N, L_total, K tiles)
    BLOCK_K = 128
    grid_concat = (N, L_total, triton.cdiv(K, BLOCK_K))
    triton_concat_sequences_kernel[grid_concat](
        out, encoder_hidden_states, hidden_states,
        N, L_txt, L_img, K,
        out.stride(0), out.stride(1), out.stride(2),
        encoder_hidden_states.stride(0), encoder_hidden_states.stride(1), encoder_hidden_states.stride(2),
        hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
        BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=2,
    )

    # Flatten rows and compute matmul: A_rows = out.view(N_rows, K), B = process_weight.T
    N_rows = N * L_total
    A_rows = out.reshape(N_rows, K).contiguous()
    # B is [K, K]
    B = process_weight.t().contiguous()

    # Allocate output rows [N_rows, K] in float32
    C_rows = torch.empty((N_rows, K), device=device, dtype=torch.float32)

    # Launch Triton matmul kernel. We set BLOCK_M=1 to compute per-row. Grid over rows and output cols tiles.
    BLOCK_N = 128
    grid_gemm = (triton.cdiv(N_rows, 1), triton.cdiv(K, BLOCK_N))
    triton_matmul_rows_cols_kernel[grid_gemm](
        C_rows, A_rows, B,
        N_rows, K,
        C_rows.stride(0), C_rows.stride(1),
        A_rows.stride(0), A_rows.stride(1),
        B.stride(0), B.stride(1),
        BLOCK_M=1, BLOCK_N=BLOCK_N, BLOCK_K=64,
        num_warps=4, num_stages=2,
    )

    # Reshape back to [N, L_total, K]
    processed = C_rows.view(N, L_total, K)

    # If original input dtype was not float32, cast back (the reference uses float32 by default)
    # Here we assume default float32; if needed, cast to original dtype
    # Determine original dtype from hidden_states (first arg)
    orig_dtype = hidden_states.dtype
    if processed.dtype != orig_dtype:
        processed = processed.to(orig_dtype)

    return processed


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Triton-only forward: numeric computation done by Triton kernels.
        processed = _triton_concat_and_linear(hidden_states, encoder_hidden_states, process_weight)
        # Split using torch (no torch.matmul on tensors in host code)
        processed_encoder = processed[:, :encoder_hidden_states.shape[1], :]
        processed_hidden = processed[:, encoder_hidden_states.shape[1]:, :]
        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
