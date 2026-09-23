import torch
import triton
import triton.language as tl


@triton.jit
def _concat_sequences_kernel(out_ptr, enc_ptr, hid_ptr,
                              N, L_txt, L_total, K,
                              stride_out_n, stride_out_t, stride_out_k,
                              stride_enc_n, stride_enc_t, stride_enc_k,
                              stride_hid_n, stride_hid_t, stride_hid_k):
    # Grid: (N, L_total, tiles_along_K)
    pid_n = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_k = tl.program_id(2)

    n = pid_n
    t = pid_t

    # Determine source (encoder or hidden) and compute offsets
    is_encoder = t < L_txt
    # K tile size
    BLOCK_K = 128
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    mask = (n < N) & (t < L_total) & (offs_k < K) & is_encoder

    if is_encoder:
        # Load from encoder_hidden_states [N, L_txt, K]
        enc_offs = n * stride_enc_n + t * stride_enc_t + offs_k * stride_enc_k
        vals = tl.load(enc_ptr + enc_offs, mask=mask, other=0.0)
        # Store into out [N, L_total, K] at position (n, t, :)
        out_offs = n * stride_out_n + t * stride_out_t + offs_k * stride_out_k
        tl.store(out_ptr + out_offs, vals, mask=mask)
    else:
        # Load from hidden_states [N, L_img, K]
        img_pos = t - L_txt
        hid_offs = n * stride_hid_n + img_pos * stride_hid_t + offs_k * stride_hid_k
        vals = tl.load(hid_ptr + hid_offs, mask=mask, other=0.0)
        out_offs = n * stride_out_n + t * stride_out_t + offs_k * stride_out_k
        tl.store(out_ptr + out_offs, vals, mask=mask)


@triton.jit
def _matmul_tiled_rows_kernel(C, A, B,
                               N_rows, K,
                               stride_c_m, stride_c_n,
                               stride_a_m, stride_a_k,
                               stride_b_k, stride_b_n,
                               BLOCK_M: tl.constexpr,  # rows per program
                               BLOCK_N: tl.constexpr,  # output columns per program
                               BLOCK_K: tl.constexpr):  # reduction tile
    # One program computes a tile of size (BLOCK_M, BLOCK_N) for output rows
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # rows
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # output columns

    # Masks for bounds
    mask_m = m_offsets < N_rows
    mask_n = n_offsets < K

    # Accumulator for this tile
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K in chunks
    for kk in range(0, K, BLOCK_K):
        k_offsets = kk + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < K

        # Load A tile: A[m, k] -> shape (BLOCK_M, BLOCK_K)
        a_ptrs = A + m_offsets[:, None] * stride_a_m + k_offsets[None, :] * stride_a_k
        a_mask = mask_m[:, None] & mask_k[None, :]
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # Load B tile: B[k, n] -> shape (BLOCK_K, BLOCK_N)
        b_ptrs = B + k_offsets[:, None] * stride_b_k + n_offsets[None, :] * stride_b_n
        b_mask = mask_k[:, None] & mask_n[None, :]
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Accumulate: acc += a @ b
        acc += tl.dot(a, b)

    # Store results to C
    c_ptrs = C + m_offsets[:, None] * stride_c_m + n_offsets[None, :] * stride_c_n
    c_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(c_ptrs, acc, mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                encoder_hidden_states: torch.Tensor,
                process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version of:
            processed = torch.cat([encoder_hidden_states, hidden_states], dim=1) @ process_weight.t()
            processed_encoder = processed[:, :text_seq_len, :]
            processed_hidden = processed[:, text_seq_len:, :]
        We only use Triton for numeric work; no torch.cat/matmul in forward.
        """
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Inputs must be on CUDA for Triton"
        assert hidden_states.shape[0] == encoder_hidden_states.shape[0], "Batch sizes must match"
        assert hidden_states.shape[2] == encoder_hidden_states.shape[2] == process_weight.shape[1], "Hidden dim must match weight shape"

        N = hidden_states.shape[0]
        L_txt = encoder_hidden_states.shape[1]
        L_img = hidden_states.shape[1]
        K = hidden_states.shape[2]
        L_total = L_txt + L_img

        # 1) Concatenate sequences into a single [N, L_total, K] tensor using Triton
        out = torch.empty((N, L_total, K), device=hidden_states.device, dtype=torch.float32)
        grid_concat = (N, L_total, triton.cdiv(K, 128))
        _concat_sequences_kernel[grid_concat](
            out,
            encoder_hidden_states.contiguous(),
            hidden_states.contiguous(),
            N, L_txt, L_total, K,
            out.stride(0), out.stride(1), out.stride(2),
            encoder_hidden_states.stride(0), encoder_hidden_states.stride(1), encoder_hidden_states.stride(2),
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            num_warps=4, num_stages=2,
        )

        # 2) GEMM: C_rows = A_rows @ B, where
        #    A_rows = out.view(N * L_total, K)  -> [N_rows, K]
        #    B = process_weight.T                -> [K, K]
        # We compute in float32 for stability.
        A_rows = out.reshape(N * L_total, K).contiguous()
        B = process_weight.t().contiguous()  # [K, K]
        C_rows = torch.empty((N * L_total, K), device=hidden_states.device, dtype=torch.float32)

        N_rows = N * L_total
        # Choose conservative tiles; these worked in earlier successful runs
        BLOCK_M = 16
        BLOCK_N = 128
        BLOCK_K = 64

        grid_gemm = (triton.cdiv(N_rows, BLOCK_M), triton.cdiv(K, BLOCK_N))
        _matmul_tiled_rows_kernel[grid_gemm](
            C_rows, A_rows, B,
            N_rows, K,
            C_rows.stride(0), C_rows.stride(1),
            A_rows.stride(0), A_rows.stride(1),
            B.stride(0), B.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        # 3) Reshape back to [N, L_total, K] and split into two streams
        processed = C_rows.view(N, L_total, K)
        processed_encoder = processed[:, :L_txt, :]
        processed_hidden = processed[:, L_txt:, :]

        # Cast back to original dtype
        processed_encoder = processed_encoder.to(encoder_hidden_states.dtype)
        processed_hidden = processed_hidden.to(hidden_states.dtype)

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
