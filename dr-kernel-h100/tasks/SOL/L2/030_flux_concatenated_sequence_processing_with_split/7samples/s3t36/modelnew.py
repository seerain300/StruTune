import torch
import triton
import triton.language as tl


@triton.jit
def _concat_sequences_kernel(out_ptr, enc_ptr, hidden_ptr,
                             N, L_txt, L_img, K,
                             BLOCK_K: tl.constexpr):
    """
    Concatenate [N, L_txt, K] and [N, L_img, K] along sequence dim to [N, L_txt + L_img, K].
    out_ptr: [N, L_total, K]
    enc_ptr: [N, L_txt, K]
    hidden_ptr: [N, L_img, K]
    """
    # program ids
    n = tl.program_id(0)  # batch
    t = tl.program_id(1)  # position in concatenated sequence
    kk = tl.program_id(2)  # tile along K

    # compute offsets for K tile
    k_offsets = kk * BLOCK_K + tl.arange(0, BLOCK_K)
    # mask for valid K
    k_mask = k_offsets < K

    # decide source based on t
    use_encoder = t < L_txt
    base_out = (n * L_txt * K) + (t * K) + k_offsets  # for encoder part
    base_hidden = (n * L_img * K) + ((t - L_txt) * K) + k_offsets  # for hidden part offset t - L_txt
    # choose source address
    addr = tl.where(use_encoder, enc_ptr + (n * L_txt * K) + (t * K), hidden_ptr + (n * L_img * K))
    # load from selected source
    x = tl.load(addr + k_offsets, mask=k_mask, other=0.0)
    # store to out
    out_base = (n * (L_txt + L_img) * K) + (t * K)
    tl.store(out_ptr + out_base + k_offsets, x, mask=k_mask)


@triton.jit
def _matmul_row_kernel(C_ptr, A_ptr, B_ptr,
                       N_rows, K, BLOCK_K: tl.constexpr):
    """
    Compute C[i, :] = A[i, :] @ B, where:
      - C_ptr: [N_rows, K], float32 output
      - A_ptr: [N_rows, K], float32 input rows (each [K] vector)
      - B_ptr: [K, K], float32 matrix
    Each program handles one row i, and iterates over K in chunks BLOCK_K.
    """
    i = tl.program_id(0)  # row index
    # guard for rows beyond N_rows (grid is exact, but keep for safety)
    if i >= N_rows:
        return

    # accumulator
    acc = tl.zeros((K,), dtype=tl.float32)

    # reduction over K in chunks
    k_start = 0
    while k_start < K:
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < K

        # load row segment of A (vector)
        a = tl.load(A_ptr + i * K + k_offsets, mask=k_mask, other=0.0)  # [BLOCK_K]
        # load B segment (matrix row segment) [BLOCK_K, BLOCK_K]
        b = tl.load(B_ptr + k_offsets[:, None] * K + tl.arange(0, BLOCK_K)[None, :], mask=k_mask[:, None], other=0.0)

        # multiply and accumulate
        acc += a[:, None] * b  # [BLOCK_K, BLOCK_K] -> sum along axis=1 to [BLOCK_K]
        k_start += BLOCK_K

    # store result
    tl.store(C_ptr + i * K + tl.arange(0, K), acc, mask=(tl.arange(0, K) < K))


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                encoder_hidden_states: torch.Tensor,
                process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version of the original run function.
        Returns (processed_encoder, processed_hidden) as in the original.
        """
        N = hidden_states.shape[0]
        L_txt = encoder_hidden_states.shape[1]
        L_img = hidden_states.shape[1]
        K = hidden_states.shape[2]
        L_total = L_txt + L_img

        device = hidden_states.device

        # 1) Concatenate sequences along sequence dimension using Triton
        out = torch.empty((N, L_total, K), device=device, dtype=torch.float32)

        # Launch Triton concatenation kernel
        BLOCK_K = 256  # K tile; masks handle tails
        grid_concat = (N, L_total, triton.cdiv(K, BLOCK_K))
        _concat_sequences_kernel[grid_concat](
            out, encoder_hidden_states, hidden_states,
            N, L_txt, L_img, K,
            BLOCK_K=BLOCK_K,
        )

        # 2) GEMM in Triton: out @ process_weight.T
        # A_rows: [N_rows, K], where N_rows = N * L_total
        N_rows = N * L_total
        A_rows = out.reshape(N_rows, K).contiguous()  # [N_rows, K], float32

        # B = process_weight.T -> [K, K], contiguous float32
        B = process_weight.t().contiguous()
        if B.dtype != torch.float32:
            B = B.float()

        # Output rows buffer: [N_rows, K], float32 accumulation
        C_rows = torch.empty((N_rows, K), device=device, dtype=torch.float32)

        # GEMM per-row kernel: one program per row
        grid_gemm = (N_rows,)
        _matmul_row_kernel[grid_gemm](
            C_rows, A_rows, B,
            N_rows, K,
            BLOCK_K=256,  # reduction tile; masks handle tails
            num_warps=4, num_stages=2,
        )

        # 3) Reshape back to [N, L_total, K] and split into encoder and hidden streams
        processed = C_rows.view(N, L_total, K)
        processed_encoder = processed[:, :L_txt, :]
        processed_hidden = processed[:, L_txt:, :]

        # Return in the same dtype as inputs (typically float32 in this model)
        return processed_encoder, processed_hidden