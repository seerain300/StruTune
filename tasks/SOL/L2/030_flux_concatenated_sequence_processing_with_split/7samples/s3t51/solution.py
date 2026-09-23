import torch
import triton
import triton.language as tl


@triton.jit
def _concat_sequences_kernel(out_ptr, enc_ptr, hid_ptr,
                              N, L_txt, L_img, K,
                              stride_out_n, stride_out_t, stride_out_k,
                              stride_enc_n, stride_enc_t, stride_enc_k,
                              stride_hid_n, stride_hid_t, stride_hid_k,
                              BLOCK_K: tl.constexpr):
    # Grid: (N, L_total, tiles along K)
    pid_n = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_k = tl.program_id(2)

    n = pid_n
    t = pid_t
    # Determine source based on t < L_txt
    is_encoder = t < L_txt

    # Compute K tile offsets
    k_start = pid_k * BLOCK_K
    k_offsets = k_start + tl.arange(0, BLOCK_K)
    k_mask = k_offsets < K

    # Compute row base offsets for out, enc, hid
    # out[n, t, k]
    out_row_base = n * stride_out_n + t * stride_out_t
    out_vec_ptrs = out_ptr + out_row_base + k_offsets * stride_out_k

    # Load from encoder if t < L_txt else from hidden
    # enc[n, t, k] or hid[n, t - L_txt, k]
    if is_encoder:
        enc_row_base = n * stride_enc_n + t * stride_enc_t
        enc_vec_ptrs = enc_ptr + enc_row_base + k_offsets * stride_enc_k
        vals = tl.load(enc_vec_ptrs, mask=k_mask, other=0.0)
    else:
        hid_row_base = n * stride_hid_n + (t - L_txt) * stride_hid_t
        hid_vec_ptrs = hid_ptr + hid_row_base + k_offsets * stride_hid_k
        vals = tl.load(hid_vec_ptrs, mask=k_mask, other=0.0)

    # Store to out[n, t, k]
    tl.store(out_vec_ptrs, vals, mask=k_mask)


@triton.jit
def _matmul_perrow_kernel(C_ptr, A_ptr, B_ptr,
                           N_rows, K,
                           stride_C_row, stride_C_k,
                           stride_A_row, stride_A_k,
                           stride_B_row, stride_B_k,
                           BLOCK_K: tl.constexpr):
    # One program computes a single row (flattened (n, t)) of C
    pid = tl.program_id(0)
    # Output vector base
    C_row_base = pid * stride_C_row
    # Initialize accumulator
    acc = tl.zeros([K], dtype=tl.float32)

    # Reduction over K in chunks of BLOCK_K
    k = 0
    while k < K:
        k_offsets = k + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < K

        # Load A_row[pid, k_offsets]
        A_row_ptrs = A_ptr + pid * stride_A_row + k_offsets * stride_A_k
        a_vals = tl.load(A_row_ptrs, mask=k_mask, other=0.0)  # [BLOCK_K] vector

        # Load B[k_offsets, :] which is process_weight.T chunk
        B_row_ptrs = B_ptr + k_offsets * stride_B_row + tl.arange(0, K) * stride_B_k
        # Note: tl.arange(0, K) must be valid; Triton will vectorize along K, we use masked load with k_mask
        b_chunk = tl.load(B_row_ptrs, mask=k_mask[:, None], other=0.0)  # [BLOCK_K, K]
        # Accumulate: dot(a_vals, b_chunk) -> [K]
        acc += tl.sum(b_chunk * a_vals[:, None], axis=0)

        k += BLOCK_K

    # Store the accumulated row
    C_vec_ptrs = C_ptr + C_row_base + tl.arange(0, K) * stride_C_k
    # We have acc of length K; store with mask for safety
    store_mask = tl.arange(0, K) < K
    tl.store(C_vec_ptrs, acc, mask=store_mask)


def _triton_cat(encoder_hidden_states: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
    """
    Concatenate along sequence dimension using Triton: [N, L_txt, K] + [N, L_img, K] -> [N, L_txt + L_img, K]
    """
    N, L_txt, K = encoder_hidden_states.shape
    N2, L_img, K2 = hidden_states.shape
    assert N == N2 and K == K2, "encoder_hidden_states and hidden_states must have same N and K"

    out = torch.empty((N, L_txt + L_img, K), device=encoder_hidden_states.device, dtype=encoder_hidden_states.dtype)
    # Ensure contiguous for predictable strides
    enc = encoder_hidden_states.contiguous()
    hid = hidden_states.contiguous()

    BLOCK_K = 128
    grid = (N, L_txt + L_img, triton.cdiv(K, BLOCK_K))
    _concat_sequences_kernel[grid](
        out, enc, hid,
        N, L_txt, L_img, K,
        out.stride(0), out.stride(1), out.stride(2),
        enc.stride(0), enc.stride(1), enc.stride(2),
        hid.stride(0), hid.stride(1), hid.stride(2),
        BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=2,
    )
    return out


def _triton_gemm_perrow(A_rows: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Compute C_rows = A_rows @ B where:
      - A_rows: [N_rows, K], contiguous
      - B: [K, K] (process_weight.T), contiguous
      - Output: C_rows [N_rows, K]
    """
    N_rows, K = A_rows.shape
    # Ensure contiguous float32 for safe accumulation
    A = A_rows.contiguous().to(torch.float32)
    B = B.contiguous().to(torch.float32)
    C = torch.empty((N_rows, K), device=A.device, dtype=torch.float32)

    BLOCK_K = 64
    grid = (N_rows,)
    _matmul_perrow_kernel[grid](
        C, A, B,
        N_rows, K,
        C.stride(0), C.stride(1),
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=2,
    )
    return C


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version of the original run function.
        - Concatenate encoder_hidden_states and hidden_states along sequence dimension.
        - Apply linear projection with process_weight.
        - Split back into processed_encoder and processed_hidden.
        """
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "All tensors must be on CUDA for Triton"
        # 1) Concatenate along sequence dimension using Triton
        concatenated = _triton_cat(encoder_hidden_states, hidden_states)

        # 2) Apply linear projection using Triton GEMM (per-row kernel)
        # process_weight has shape [K, K]; concatenated has shape [N, L_total, K]
        N, L_total, K = concatenated.shape
        A_rows = concatenated.reshape(N * L_total, K)  # [N_rows, K]
        B = process_weight.t().contiguous()  # [K, K]

        C_rows = _triton_gemm_perrow(A_rows, B)  # [N_rows, K]

        # 3) Reshape back and split
        processed = C_rows.view(N, L_total, K)
        processed_encoder = processed[:, :encoder_hidden_states.shape[1], :]
        processed_hidden = processed[:, encoder_hidden_states.shape[1]:, :]

        # Cast back to original dtypes to match PyTorch behavior
        processed_encoder = processed_encoder.to(encoder_hidden_states.dtype)
        processed_hidden = processed_hidden.to(hidden_states.dtype)

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
