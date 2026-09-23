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
    k_start = pid_k * BLOCK_K
    k_offsets = k_start + tl.arange(0, BLOCK_K)
    mask_k = k_offsets < K

    # Determine source: encoder (t < L_txt) or hidden (t >= L_txt)
    is_encoder = t < L_txt

    # Compute base pointers
    out_row_ptr = out_ptr + n * stride_out_n + t * stride_out_t
    if is_encoder:
        enc_row_ptr = enc_ptr + n * stride_enc_n + t * stride_enc_t
        vals = tl.load(enc_row_ptr + k_offsets * stride_enc_k, mask=mask_k, other=0.0)
    else:
        t_hidden = t - L_txt
        hid_row_ptr = hid_ptr + n * stride_hid_n + t_hidden * stride_hid_t
        vals = tl.load(hid_row_ptr + k_offsets * stride_hid_k, mask=mask_k, other=0.0)

    tl.store(out_row_ptr + k_offsets * stride_out_k, vals, mask=mask_k)


@triton.jit
def _matmul_tiled_rows_kernel(C, A, B,
                               N_rows, K,
                               stride_c_row, stride_c_col,
                               stride_a_row, stride_a_col,
                               stride_b_row, stride_b_col,
                               BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    # Each program computes a tile of size [BLOCK_N] for one row.
    pid_row = tl.program_id(0)
    pid_col = tl.program_id(1)

    # Row index in [0, N_rows)
    row = pid_row
    # Column tile index
    col_start = pid_col * BLOCK_N
    col_offsets = col_start + tl.arange(0, BLOCK_N)
    mask_out = col_offsets < K

    # Accumulator for this output tile (vector of length BLOCK_N)
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # Reduction over K in chunks
    for kk in range(0, K, BLOCK_K):
        k_offsets = kk + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < K

        # Load A_tile: shape [BLOCK_N, BLOCK_K]
        # A layout: [N_rows, K], row is fixed by pid_row, cols across k_offsets
        a_ptrs = A + row * stride_a_row + k_offsets[None, :] * stride_a_col  # broadcasting over [1, BLOCK_K]
        a_ptrs = a_ptrs + tl.arange(0, BLOCK_N)[:, None] * stride_a_row      # broadcasting over [BLOCK_N, 1]
        A_tile = tl.load(a_ptrs, mask=(mask_out[:, None] & mask_k[None, :]), other=0.0)

        # Load B_tile: shape [BLOCK_K, BLOCK_N], B is process_weight.T
        b_ptrs = B + k_offsets[:, None] * stride_b_row + col_offsets[None, :] * stride_b_col
        B_tile = tl.load(b_ptrs, mask=(mask_k[:, None] & mask_out[None, :]), other=0.0)

        # Accumulate: acc[BLOCK_N] += A_tile @ B_tile
        acc += tl.sum(A_tile @ B_tile, axis=0)  # sum over K-chunk dimension

    # Store results
    C_ptrs = C + row * stride_c_row + col_offsets * stride_c_col
    tl.store(C_ptrs, acc, mask=mask_out)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                encoder_hidden_states: torch.Tensor,
                process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version of:
          concatenated = cat([encoder_hidden_states, hidden_states], dim=1)
          processed = concatenated @ process_weight.T
          return processed_encoder, processed_hidden
        where processed_encoder = processed[:, :text_seq_len], processed_hidden = processed[:, text_seq_len:].
        """
        # Ensure on CUDA and contiguous; use float32 for accumulation
        device = hidden_states.device
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Tensors must be on CUDA device"

        N = hidden_states.shape[0]
        L_txt = encoder_hidden_states.shape[1]
        L_img = hidden_states.shape[1]
        K = hidden_states.shape[2]
        L_total = L_txt + L_img

        # 1) Concatenate along sequence dim in Triton: [N, L_total, K]
        out = torch.empty((N, L_total, K), device=device, dtype=torch.float32)

        enc = encoder_hidden_states.contiguous().to(torch.float32)
        hid = hidden_states.contiguous().to(torch.float32)

        BLOCK_K = 256  # vectorize along K; can tune (64/128/256)
        grid_concat = (N, L_total, triton.cdiv(K, BLOCK_K))
        _concat_sequences_kernel[grid_concat](
            out, enc, hid,
            N, L_txt, L_img, K,
            out.stride(0), out.stride(1), out.stride(2),
            enc.stride(0), enc.stride(1), enc.stride(2),
            hid.stride(0), hid.stride(1), hid.stride(2),
            BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        # 2) GEMM: A = out [N_rows, K], B = process_weight.T [K, K], C = [N_rows, K]
        N_rows = N * L_total
        A = out.contiguous()  # already float32
        B = process_weight.t().contiguous().to(torch.float32)

        C_rows = torch.empty((N_rows, K), device=device, dtype=torch.float32)

        BLOCK_N = 128
        grid_gemm = (N_rows, triton.cdiv(K, BLOCK_N))
        _matmul_tiled_rows_kernel[grid_gemm](
            C_rows, A, B,
            N_rows, K,
            C_rows.stride(0), C_rows.stride(1),
            A.stride(0), A.stride(1),
            B.stride(0), B.stride(1),
            BLOCK_N=BLOCK_N, BLOCK_K=64,  # BLOCK_K used in reduction loop
            num_warps=4, num_stages=2,
        )

        # 3) Reshape back to [N, L_total, K] and split into two streams
        processed = C_rows.view(N, L_total, K)
        processed_encoder = processed[:, :L_txt, :]
        processed_hidden = processed[:, L_txt:, :]

        # Cast back to original dtypes (match PyTorch behavior)
        processed_encoder = processed_encoder.to(encoder_hidden_states.dtype)
        processed_hidden = processed_hidden.to(hidden_states.dtype)

        return processed_encoder, processed_hidden