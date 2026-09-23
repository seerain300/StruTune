import torch
import triton
import triton.language as tl


@triton.jit
def _concat_sequences_kernel(out_ptr, enc_ptr, hid_ptr,
                              N, L_txt, L_img, K,
                              stride_out_n, stride_out_t, stride_out_k,
                              stride_enc_n, stride_enc_t, stride_enc_k,
                              stride_hid_n, stride_hid_t, stride_hid_k):
    # Grid: (N, L_total, tiles along K)
    pid_n = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_k = tl.program_id(2)

    # Compute offsets for K tile
    k_offsets = pid_k * 128 + tl.arange(0, 128)
    mask_k = k_offsets < K

    # Determine source tensor based on t
    # t < L_txt -> encoder, else -> hidden (offset t - L_txt in the concatenated dimension)
    use_encoder = pid_t < L_txt

    # Base pointers
    out_base = out_ptr + pid_n * stride_out_n + pid_t * stride_out_t
    if use_encoder:
        src_base = enc_ptr + pid_n * stride_enc_n + pid_t * stride_enc_t
    else:
        src_base = hid_ptr + pid_n * stride_hid_n + (pid_t - L_txt) * stride_hid_t

    # Compute pointers for this K tile
    out_ptrs = out_base + k_offsets * stride_out_k
    src_ptrs = src_base + k_offsets * stride_enc_k  # enc_k/hidden_k stride is same, k is last dim

    # Load and store with mask
    x = tl.load(src_ptrs, mask=mask_k, other=0.0)
    tl.store(out_ptrs, x, mask=mask_k)


@triton.jit
def _matmul_row_kernel(C_ptr, A_ptr, B_ptr,
                       N_rows, K,
                       stride_c_row, stride_c_k,
                       stride_a_row, stride_a_k,
                       stride_b_k, stride_b_k2,
                       BLOCK_K: tl.constexpr):
    # One program per row
    row_id = tl.program_id(0)
    # Accumulator vector
    acc = tl.zeros((BLOCK_K,), dtype=tl.float32)

    # Chunked reduction over K
    k0 = 0
    while k0 < K:
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < K

        # Load A[row, k_offsets] as vector
        a_ptrs = A_ptr + row_id * stride_a_row + k_offsets * stride_a_k
        a = tl.load(a_ptrs, mask=mask_k, other=0.0)  # [BLOCK_K]

        # Load B[k_offsets, k_offsets] as matrix [BLOCK_K, BLOCK_K]
        b_ptrs = B_ptr + k_offsets[:, None] * stride_b_k + k_offsets[None, :] * stride_b_k2
        b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_k[None, :], other=0.0)  # [BLOCK_K, BLOCK_K]

        # Accumulate: acc += a @ b
        # Sum over the last dim (BLOCK_K)
        acc += tl.sum(a[:, None] * b, axis=1)

        k0 += BLOCK_K

    # Store the result back to C[row, :]
    c_ptrs = C_ptr + row_id * stride_c_row + tl.arange(0, BLOCK_K) * stride_c_k
    store_mask = tl.arange(0, BLOCK_K) < K
    tl.store(c_ptrs, acc, mask=store_mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                encoder_hidden_states: torch.Tensor,
                process_weight: torch.Tensor):
        """
        Triton implementation of:
        1) Concatenate encoder_hidden_states and hidden_states along sequence dimension.
        2) Apply linear projection: concatenated @ process_weight.T
        3) Split back into encoder and hidden streams.
        All numeric work is performed by Triton kernels.
        """
        # Shapes
        N = hidden_states.shape[0]
        L_txt = encoder_hidden_states.shape[1]
        L_img = hidden_states.shape[1]
        K = hidden_states.shape[2]
        L_total = L_txt + L_img

        device = hidden_states.device
        dtype = hidden_states.dtype

        # Ensure tensors are contiguous
        enc = encoder_hidden_states.contiguous()
        hid = hidden_states.contiguous()
        B = process_weight.t().contiguous()  # [K, K]

        # 1) Concatenate along sequence dimension into out [N, L_total, K]
        out = torch.empty((N, L_total, K), device=device, dtype=torch.float32)

        # Strides
        stride_out_n, stride_out_t, stride_out_k = out.stride()
        stride_enc_n, stride_enc_t, stride_enc_k = enc.stride()
        stride_hid_n, stride_hid_t, stride_hid_k = hid.stride()

        grid_concat = (N, L_total, (K + 127) // 128)
        _concat_sequences_kernel[grid_concat](
            out, enc, hid,
            N, L_txt, L_img, K,
            stride_out_n, stride_out_t, stride_out_k,
            stride_enc_n, stride_enc_t, stride_enc_k,
            stride_hid_n, stride_hid_t, stride_hid_k,
            num_warps=4, num_stages=2,
        )

        # 2) GEMM: per-row matmul out @ B -> C_rows [N * L_total, K] in float32
        A_rows = out.reshape(N * L_total, K).contiguous()  # [N_rows, K]
        C_rows = torch.empty((N * L_total, K), device=device, dtype=torch.float32)

        # Launch one program per row
        grid_gemm = (N * L_total,)
        _matmul_row_kernel[grid_gemm](
            C_rows, A_rows, B,
            N * L_total, K,
            C_rows.stride(0), C_rows.stride(1),
            A_rows.stride(0), A_rows.stride(1),
            B.stride(0), B.stride(1),
            BLOCK_K=128,
            num_warps=4, num_stages=2,
        )

        # 3) Reshape back to [N, L_total, K] and split
        processed = C_rows.view(N, L_total, K)
        processed_encoder = processed[:, :L_txt, :]
        processed_hidden = processed[:, L_txt:, :]

        # Return in original dtype
        processed_encoder = processed_encoder.to(dtype)
        processed_hidden = processed_hidden.to(dtype)

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
