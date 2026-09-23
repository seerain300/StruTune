import torch
import triton
import triton.language as tl


@triton.jit
def _concat_sequences_kernel(
    out_ptr,  # *float, shape [N, L_total, K]
    enc_ptr,  # *float, shape [N, L_txt, K]
    hid_ptr,  # *float, shape [N, L_img, K]
    N, L_txt, L_img, K,
    BLOCK_K: tl.constexpr,
):
    # Grid: (N, L_total, tiles_along_K)
    pid_n = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_k = tl.program_id(2)

    # Compute offsets along K
    k_offsets = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    k_mask = k_offsets < K

    # Compute source pointers depending on whether pid_t < L_txt
    # If true: copy from enc_ptr; else: copy from hid_ptr at offset pid_t - L_txt
    # out_ptr indexing: out[n, t, k] = enc[n, t, k] or hid[n, t - L_txt, k]
    # We need to compute base addresses for enc/hid and out.

    # Base offsets for out, enc, hid
    out_base = (pid_n * L_total + pid_t) * K
    enc_base = pid_n * (L_txt * K)
    hid_base = pid_n * (L_img * K)

    # Select source based on pid_t
    # If pid_t < L_txt:
    #   src_ptr = enc_ptr + enc_base + pid_t*K + k_offsets
    # else:
    #   src_ptr = hid_ptr + hid_base + (pid_t - L_txt)*K + k_offsets
    # Create src_ptr for both branches and select with tl.where
    is_encoder = pid_t < L_txt
    src_ptr = tl.where(is_encoder,
                       enc_ptr + enc_base + pid_t * K + k_offsets,
                       hid_ptr + hid_base + (pid_t - L_txt) * K + k_offsets)

    # Out pointer
    out_index = out_ptr + out_base + k_offsets

    # Load and store
    vals = tl.load(src_ptr, mask=k_mask, other=0.0)
    tl.store(out_index, vals, mask=k_mask)


@triton.jit
def _matmul_row_kernel(
    C_ptr,   # *float, shape [N_rows, K], contiguous
    A_ptr,   # *float, shape [N_rows, K], contiguous
    B_ptr,   # *float, shape [K, K], contiguous
    N_rows, K,  # int
    BLOCK_K: tl.constexpr,
):
    # One program per row: compute C[row, :] = A[row, :] @ B
    row_id = tl.program_id(0)
    if row_id >= N_rows:
        return

    # Accumulator
    acc = tl.zeros((K,), dtype=tl.float32)

    # Iterate over K in chunks of BLOCK_K
    k0 = 0
    while k0 < K:
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < K

        # Load A[row, k_offsets]
        a_ptr = A_ptr + row_id * K + k_offsets
        a_vals = tl.load(a_ptr, mask=k_mask, other=0.0)  # [BLOCK_K]

        # Load B[k_offsets, :] as a [BLOCK_K, K] matrix
        # Address for each i in [BLOCK_K] and j in [K]: B[k_offsets[i], j]
        # We need a 2D pointer: (k_offsets[:, None], j_offsets[None, :])
        # Create j_offsets
        j_offsets = tl.arange(0, K)
        j_mask = j_offsets < K

        # B[k, j] with k = k_offsets
        b_ptrs = B_ptr + k_offsets[:, None] * K + j_offsets[None, :]
        b_vals = tl.load(b_ptrs, mask=(k_mask[:, None] & j_mask[None, :]), other=0.0)  # [BLOCK_K, K]

        # Accumulate: acc += sum_i a_vals[i] * b_vals[i, :]
        # Do a small loop to accumulate
        for i in range(BLOCK_K):
            k_valid = k_mask[i]
            a_i = a_vals[i]
            b_i = b_vals[i, :]  # [K]
            acc += tl.where(k_valid, a_i * b_i, 0.0)

        k0 += BLOCK_K

    # Store result to C
    c_ptrs = C_ptr + row_id * K + tl.arange(0, K)
    tl.store(c_ptrs, acc, mask=(tl.arange(0, K) < K))


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                encoder_hidden_states: torch.Tensor,
                process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version:
        - Concatenate encoder_hidden_states and hidden_states along sequence dimension in Triton.
        - Apply linear projection in Triton via per-row GEMM (no bias).
        - Split back into separate encoder and image streams.
        """
        assert hidden_states.ndim == 3 and encoder_hidden_states.ndim == 3, "Inputs must be 3D tensors"
        N = hidden_states.shape[0]
        L_img = hidden_states.shape[1]
        L_txt = encoder_hidden_states.shape[1]
        K = hidden_states.shape[2]
        device = hidden_states.device

        # Ensure contiguous inputs
        enc = encoder_hidden_states.contiguous()
        hid = hidden_states.contiguous()
        weight = process_weight.contiguous()

        # 1) Concatenate along sequence dimension into out_cat [N, L_total, K]
        L_total = L_txt + L_img
        out_cat = torch.empty((N, L_total, K), device=device, dtype=torch.float32)

        # Grid for concat: (N, L_total, tiles along K)
        BLOCK_K = 128
        grid = (N, L_total, triton.cdiv(K, BLOCK_K))
        _concat_sequences_kernel[grid](
            out_cat, enc, hid,
            N, L_txt, L_img, K,
            BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        # 2) GEMM: out_cat @ weight.T -> [N, L_total, K]
        # Flatten rows and allocate output
        A_rows = out_cat.reshape(N * L_total, K).contiguous()  # [N_rows, K]
        B = weight.t().contiguous()  # [K, K]
        N_rows = N * L_total
        C_rows = torch.empty((N_rows, K), device=device, dtype=torch.float32)  # accumulate in fp32

        # Launch per-row GEMM kernel
        BLOCK_K_GEMM = 128  # robust default; adjust as needed
        grid_gemm = (N_rows,)
        _matmul_row_kernel[grid_gemm](
            C_rows, A_rows, B,
            N_rows, K,
            BLOCK_K=BLOCK_K_GEMM,
            num_warps=4, num_stages=2,
        )

        # 3) Reshape back to [N, L_total, K] and split into encoder and hidden parts
        processed = C_rows.view(N, L_total, K)
        processed_encoder = processed[:, :L_txt, :]
        processed_hidden = processed[:, L_txt:, :]

        # Cast back to original dtypes if necessary
        if processed_encoder.dtype != encoder_hidden_states.dtype:
            processed_encoder = processed_encoder.to(encoder_hidden_states.dtype)
        if processed_hidden.dtype != hidden_states.dtype:
            processed_hidden = processed_hidden.to(hidden_states.dtype)

        return processed_encoder, processed_hidden