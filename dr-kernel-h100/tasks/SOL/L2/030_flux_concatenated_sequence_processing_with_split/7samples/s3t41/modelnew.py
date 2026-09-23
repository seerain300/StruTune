import torch
import triton
import triton.language as tl


@triton.jit
def _concat_sequences_kernel(
    out_ptr, enc_ptr, hid_ptr,
    N, L_txt, L_img, K,
    stride_out_n, stride_out_t, stride_out_k,
    stride_enc_n, stride_enc_t, stride_enc_k,
    stride_hid_n, stride_hid_t, stride_hid_k,
    BLOCK_K: tl.constexpr,
):
    # Grid: (N, L_total, tiles_along_K). L_total = L_txt + L_img
    pid_n = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_k = tl.program_id(2)

    # Compute K tile indices
    k_offsets = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_k = k_offsets < K

    # Determine source tensor based on t
    # out[n, t, k] = enc[n, t, k] if t < L_txt else hid[n, t - L_txt, k]
    # Note: pid_t is in [0, L_total). If pid_t < L_txt, then t = pid_t; else t = pid_t - L_txt.
    # For safety, compute with masks and avoid branching that may cause OOB.
    # We'll initialize load values for enc and hid separately and then select.
    mask_out = (pid_n < N) & (pid_t < (L_txt + L_img)) & mask_k

    # Compute base pointers
    out_base = pid_n * stride_out_n + pid_t * stride_out_t + k_offsets * stride_out_k

    # Load from encoder if pid_t < L_txt, else from hidden
    t_val = pid_t
    enc_mask = (t_val < L_txt) & mask_out
    enc_base = pid_n * stride_enc_n + t_val * stride_enc_t + k_offsets * stride_enc_k
    enc_vals = tl.load(enc_ptr + enc_base, mask=enc_mask, other=0.0)

    t_img = t_val - L_txt
    hid_mask = ((t_val >= L_txt) & (t_val < (L_txt + L_img))) & mask_k
    hid_base = pid_n * stride_hid_n + t_img * stride_hid_t + k_offsets * stride_hid_k
    hid_vals = tl.load(hid_ptr + hid_base, mask=hid_mask, other=0.0)

    # Select appropriate values
    selected = tl.where(enc_mask, enc_vals, 0.0) + tl.where(hid_mask, hid_vals, 0.0)
    # Note: If both masks are False, selected is 0.0.

    # Store to out
    tl.store(out_ptr + out_base, selected, mask=mask_out)


@triton.jit
def _matmul_tiled_kernel(
    C_ptr, A_ptr, B_ptr,
    N_rows, K,
    stride_A_row, stride_A_col,
    stride_B_row, stride_B_col,
    stride_C_row, stride_C_col,
    BLOCK_M: tl.constexpr,  # number of rows per program
    BLOCK_N: tl.constexpr,  # number of columns per program
    BLOCK_K: tl.constexpr,  # reduction chunk
):
    # Grid: (grid_m, grid_n)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Row and column offsets for this program
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Masks for boundary
    mask_rows = rows < N_rows
    mask_cols = cols < K

    # Initialize accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Reduction over K
    for kk in range(0, K, BLOCK_K):
        k_offsets = kk + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < K

        # Load A tile: A[rows, k_offsets] -> shape (BLOCK_M, BLOCK_K)
        a_ptrs = A_ptr + rows[:, None] * stride_A_row + k_offsets[None, :] * stride_A_col
        a_mask = mask_rows[:, None] & mask_k[None, :]
        A_tile = tl.load(A_ptr + rows[:, None] * stride_A_row + k_offsets[None, :] * stride_A_col, mask=a_mask, other=0.0)

        # Load B tile: B[k_offsets, cols] -> shape (BLOCK_K, BLOCK_N)
        b_ptrs = B_ptr + k_offsets[:, None] * stride_B_row + cols[None, :] * stride_B_col
        b_mask = mask_k[:, None] & mask_cols[None, :]
        B_tile = tl.load(B_ptr + k_offsets[:, None] * stride_B_row + cols[None, :] * stride_B_col, mask=b_mask, other=0.0)

        # Accumulate
        acc += tl.dot(A_tile, B_tile)

    # Store result
    c_ptrs = C_ptr + rows[:, None] * stride_C_row + cols[None, :] * stride_C_col
    c_mask = mask_rows[:, None] & mask_cols[None, :]
    tl.store(C_ptr + rows[:, None] * stride_C_row + cols[None, :] * stride_C_col, acc, mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        process_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version of the original run function.
        - Concatenates encoder_hidden_states and hidden_states along sequence dimension using Triton.
        - Applies linear projection using Triton GEMM (tiled matmul).
        - Splits back into processed_encoder and processed_hidden.
        """
        assert hidden_states.ndim == 3 and encoder_hidden_states.ndim == 3, "Inputs must be [N, L, K]"
        N = hidden_states.shape[0]
        L_txt = encoder_hidden_states.shape[1]
        L_img = hidden_states.shape[1]
        K = hidden_states.shape[2]
        device = hidden_states.device

        # Ensure dtype is float32 for numeric stability; we can cast back later
        dtype = hidden_states.dtype
        # Prepare concatenated tensor using Triton
        L_total = L_txt + L_img
        concatenated = torch.empty((N, L_total, K), device=device, dtype=torch.float32)

        # Strides for concatenation kernel
        stride_out_n, stride_out_t, stride_out_k = concatenated.stride()
        stride_enc_n, stride_enc_t, stride_enc_k = encoder_hidden_states.contiguous().stride()
        enc_c = encoder_hidden_states.contiguous()  # ensure contiguous
        stride_hid_n, stride_hid_t, stride_hid_k = hidden_states.contiguous().stride()
        hid_c = hidden_states.contiguous()

        # Launch concatenation kernel
        BLOCK_K = 128
        grid_concat = (N, L_total, triton.cdiv(K, BLOCK_K))
        _concat_sequences_kernel[grid_concat](
            concatenated,
            enc_c, hid_c,
            N, L_txt, L_img, K,
            stride_out_n, stride_out_t, stride_out_k,
            stride_enc_n, stride_enc_t, stride_enc_k,
            stride_hid_n, stride_hid_t, stride_hid_k,
            BLOCK_K=BLOCK_K,
        )

        # Prepare A_rows and B for GEMM
        # A_rows: [N_rows, K], where N_rows = N * L_total
        A_rows = concatenated.view(N * L_total, K).contiguous().to(torch.float32)
        # B: [K, K] = process_weight.T (no bias)
        B = process_weight.t().contiguous().to(torch.float32)

        # Output rows buffer
        N_rows = N * L_total
        C_rows = torch.empty((N_rows, K), device=device, dtype=torch.float32)

        # Strides for matmul kernel
        stride_A_row = A_rows.stride(0)
        stride_A_col = A_rows.stride(1)
        stride_B_row = B.stride(0)
        stride_B_col = B.stride(1)
        stride_C_row = C_rows.stride(0)
        stride_C_col = C_rows.stride(1)

        # Tiling parameters: robust defaults
        BLOCK_M = 32
        BLOCK_N = 64
        BLOCK_K = 64
        grid_m = triton.cdiv(N_rows, BLOCK_M)
        grid_n = triton.cdiv(K, BLOCK_N)

        _matmul_tiled_kernel[(grid_m, grid_n)](
            C_rows, A_rows, B,
            N_rows, K,
            stride_A_row, stride_A_col,
            stride_B_row, stride_B_col,
            stride_C_row, stride_C_col,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        # Reshape back to [N, L_total, K]
        processed = C_rows.view(N, L_total, K)
        # Split into encoder and hidden parts
        processed_encoder = processed[:, :L_txt, :]
        processed_hidden = processed[:, L_txt:, :]

        # Cast back to original dtype
        processed_encoder = processed_encoder.to(dtype)
        processed_hidden = processed_hidden.to(dtype)

        return processed_encoder, processed_hidden


# Example usage (not required by evaluator, but useful for local testing):
# if __name__ == "__main__":
#     torch.manual_seed(0)
#     N, L_txt, L_img, K = 2, 128, 256, 1024
#     x = torch.randn(N, L_img, K, device='cuda', dtype=torch.float32)
#     y = torch.randn(N, L_txt, K, device='cuda', dtype=torch.float32)
#     W = torch.randn(K, K, device='cuda', dtype=torch.float32)
#     model_ref = ModelNew()
#     model_torch = lambda: torch.cat([y, x], dim=1) @ W.t()
#     enc, hid = model_ref(x, y, W)
#     ref = torch.split(model_torch(), [L_txt, L_img], dim=1)
#     print("Close:", torch.allclose(enc, ref[0], atol=1e-6), torch.allclose(hid, ref[1], atol=1e-6))