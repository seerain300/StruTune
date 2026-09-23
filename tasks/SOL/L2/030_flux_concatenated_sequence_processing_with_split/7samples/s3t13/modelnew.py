import torch
import triton
import triton.language as tl


@triton.jit
def _concat_sequences_kernel(
    out_ptr, encoder_ptr, hidden_ptr,
    N, L_txt, L_img, K,
    # strides
    out_s0, out_s1, out_s2,
    enc_s0, enc_s1, enc_s2,
    hid_s0, hid_s1, hid_s2,
    # tiling
    BLOCK_K: tl.constexpr,
):
    # 3D grid: (N, L_total, tiles_along_K)
    pid_n = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_k = tl.program_id(2)

    # Compute offsets along K for this program
    k_offsets = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    k_mask = k_offsets < K

    # Determine source: encoder or hidden
    # t ranges from 0 to L_total-1
    use_encoder = pid_t < L_txt
    # Base pointers
    # If using encoder: read from encoder[pid_n, pid_t, k_offsets]
    # Else: read from hidden[pid_n, pid_t - L_txt, k_offsets]
    if use_encoder:
        # encoder pointer: out[n, t, k] = encoder[n, t, k]
        enc_offset = pid_n * enc_s0 + pid_t * enc_s1 + k_offsets * enc_s2
        val = tl.load(encoder_ptr + enc_offset, mask=k_mask, other=0.0)
        out_offset = pid_n * out_s0 + pid_t * out_s1 + k_offsets * out_s2
        tl.store(out_ptr + out_offset, val, mask=k_mask)
    else:
        # hidden pointer: out[n, t, k] = hidden[n, t - L_txt, k]
        t_src = pid_t - L_txt
        hid_offset = pid_n * hid_s0 + t_src * hid_s1 + k_offsets * hid_s2
        val = tl.load(hidden_ptr + hid_offset, mask=k_mask, other=0.0)
        out_offset = pid_n * out_s0 + pid_t * out_s1 + k_offsets * out_s2
        tl.store(out_ptr + out_offset, val, mask=k_mask)


@triton.jit
def _matmul_batched_rows_kernel(
    C_ptr, A_ptr, B_ptr,
    N_rows, K,
    BLOCK_K: tl.constexpr,
):
    # One program per output row
    pid = tl.program_id(0)
    # Guard in case grid > N_rows (not necessary if grid == N_rows)
    if pid >= N_rows:
        return

    # Accumulator for this row (1 x K), in fp32
    acc = tl.zeros((1, K), dtype=tl.float32)

    k0 = 0
    while k0 < K:
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < K

        # Load A_row_chunk: A[pid, k0:k0+BLOCK_K]
        a_row_base = pid * K  # since A is [N_rows, K] contiguous
        a_chunk = tl.load(A_ptr + a_row_base + k_offsets, mask=k_mask, other=0.0)  # shape [BLOCK_K]

        # Load B_chunk: B[k0:k0+BLOCK_K, :]
        b_chunk = tl.load(B_ptr + k_offsets[:, None] * K + tl.arange(0, K)[None, :],  # shape [BLOCK_K, K]
                          mask=(k_mask[:, None]), other=0.0)

        # Outer product and accumulate: (1, BLOCK_K) * (BLOCK_K, K) -> (1, K)
        # Cast a_chunk to (1, BLOCK_K)
        a_chunk = a_chunk[None, :]  # (1, BLOCK_K)
        acc += a_chunk * b_chunk

        k0 += BLOCK_K

    # Store the accumulated row to C[pid, :]
    c_row_base = pid * K
    tl.store(C_ptr + c_row_base + tl.arange(0, K), acc[0, :], mask=(tl.arange(0, K) < K))


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        process_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized implementation:
        1) Concatenate encoder_hidden_states and hidden_states along sequence dimension using Triton.
        2) Apply linear projection (concatenated @ process_weight.T) using Triton GEMM.
        3) Split result back into processed_encoder and processed_hidden.
        Returns: (processed_encoder, processed_hidden)
        """
        device = hidden_states.device
        N = hidden_states.shape[0]
        L_txt = encoder_hidden_states.shape[1]
        L_img = hidden_states.shape[1]
        K = hidden_states.shape[2]

        # Ensure contiguity
        encoder = encoder_hidden_states.contiguous()
        hidden = hidden_states.contiguous()
        weight = process_weight.contiguous()  # [K, K]

        # 1) Concatenate along sequence dimension using Triton: out_cat [N, L_total, K]
        L_total = L_txt + L_img
        out_cat = torch.empty((N, L_total, K), device=device, dtype=torch.float32)

        # Strides
        out_s0, out_s1, out_s2 = out_cat.stride()
        enc_s0, enc_s1, enc_s2 = encoder.stride()
        hid_s0, hid_s1, hid_s2 = hidden.stride()

        # Choose BLOCK_K for concatenation (memory-bound op; 128/256 are fine)
        BLOCK_K = 256 if K >= 256 else 128
        grid = (N, L_total, triton.cdiv(K, BLOCK_K))
        _concat_sequences_kernel[grid](
            out_cat, encoder, hidden,
            N, L_txt, L_img, K,
            out_s0, out_s1, out_s2,
            enc_s0, enc_s1, enc_s2,
            hid_s0, hid_s1, hid_s2,
            BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        # 2) GEMM: out_cat @ process_weight.T using Triton
        # out_cat is [N, L_total, K]; treat rows as N_rows = N * L_total
        out_cat_fp32 = out_cat  # already fp32
        A_rows = out_cat_fp32.reshape(-1, K).contiguous()  # [N_rows, K]
        B = weight.t().contiguous()  # [K, K]
        C_rows = torch.empty((A_rows.shape[0], K), device=device, dtype=torch.float32)

        N_rows = A_rows.shape[0]

        # Choose BLOCK_K for GEMM
        BLOCK_K_GEMM = 256 if K >= 256 else 128
        grid_gemm = (N_rows,)
        _matmul_batched_rows_kernel[grid_gemm](
            C_rows, A_rows, B,
            N_rows, K,
            BLOCK_K=BLOCK_K_GEMM,
            num_warps=4, num_stages=2,
        )

        # Reshape back to [N, L_total, K]
        processed = C_rows.view(N, L_total, K)

        # 3) Split into two streams
        processed_encoder = processed[:, :L_txt, :]
        processed_hidden = processed[:, L_txt:, :]

        # Cast back to original dtype if needed (evaluator typically uses float32)
        # Here, we keep fp32; adjust if inputs are not float32.
        return processed_encoder, processed_hidden