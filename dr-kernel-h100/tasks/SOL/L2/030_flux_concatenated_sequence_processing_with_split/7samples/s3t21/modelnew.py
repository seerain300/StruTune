import torch
import triton
import triton.language as tl

# 1) Concatenation Triton kernel: out[n, t, :] = encoder[n, t, :] if t < L_txt
#                                    out[n, t, :] = hidden[n, t - L_txt, :] if t >= L_txt
@triton.jit
def _concat_sequences_kernel(
    out_ptr,           # *float32, [N, L_total, K]
    encoder_ptr,       # *float32, [N, L_txt, K]
    hidden_ptr,        # *float32, [N, L_img, K]
    L_txt, L_img, K,   # int32
    BLOCK_K: tl.constexpr,
):
    n = tl.program_id(0)      # batch index
    t = tl.program_id(1)      # concatenated sequence position [0, L_total)
    k_block = tl.program_id(2)  # tile along K dimension

    K_offsets = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
    mask = K_offsets < K

    # Determine source: if t < L_txt -> encoder; else -> hidden at t - L_txt
    use_encoder = t < L_txt

    # Compute base offsets
    # out[n, t, k]
    out_base = (n * L_total + t) * K
    out_offs = out_base + K_offsets

    if use_encoder:
        # encoder[n, t, k]
        enc_base = n * L_txt * K + t * K
        enc_offs = enc_base + K_offsets
        vals = tl.load(encoder_ptr + enc_offs, mask=mask, other=0.0)
        tl.store(out_ptr + out_offs, vals, mask=mask)
    else:
        # hidden[n, t - L_txt, k]
        h_idx = t - L_txt
        h_base = n * L_img * K + h_idx * K
        h_offs = h_base + K_offsets
        vals = tl.load(hidden_ptr + h_offs, mask=mask, other=0.0)
        tl.store(out_ptr + out_offs, vals, mask=mask)


# 2) Batched GEMM Triton kernel: computes C_rows[i, :] = A_rows[i, :] @ B, one row per program
@triton.jit
def _matmul_row_kernel(
    C_ptr,             # *float32, [N_rows, K]
    A_ptr,             # *float32, [N_rows, K]
    B_ptr,             # *float32, [K, K]
    N_rows, K,         # int32
    BLOCK_K: tl.constexpr,
):
    i = tl.program_id(0)  # row index in [0, N_rows)
    # Accumulator for this row
    acc = tl.zeros((BLOCK_K,), dtype=tl.float32)

    # Iterate over K dimension in chunks
    k0 = 0
    while k0 < K:
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < K

        # Load A_row chunk: A[i, k_offsets]
        a_offs = i * K + k_offsets
        a_vals = tl.load(A_ptr + a_offs, mask=mask_k, other=0.0)  # [BLOCK_K]

        # Load B chunk: B[k_offsets, :] -> [BLOCK_K, K]
        b_offs = k_offsets[:, None] * K + tl.arange(0, K)[None, :]  # [BLOCK_K, K]
        b_vals = tl.load(B_ptr + b_offs, mask=mask_k[:, None], other=0.0)  # [BLOCK_K, K]

        # Accumulate: acc += a_vals[:, None] * b_vals[:, :]
        # Broadcast multiply and sum along K axis
        acc += tl.sum(a_vals[:, None] * b_vals, axis=1)

        k0 += BLOCK_K

    # Store result
    c_offs = i * K + tl.arange(0, K)
    tl.store(C_ptr + c_offs, acc, mask=True)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        Triton-optimized version:
        - Concatenate encoder_hidden_states and hidden_states along sequence dimension via Triton.
        - Apply linear projection via Triton GEMM: A_cat @ process_weight.T
        - Split outputs back into encoder and hidden streams.
        """
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Tensors must be on CUDA device for Triton kernels"

        device = hidden_states.device
        N = hidden_states.shape[0]
        L_txt = encoder_hidden_states.shape[1]
        L_img = hidden_states.shape[1]
        K = hidden_states.shape[2]
        L_total = L_txt + L_img

        # 1) Allocate output for concatenation: [N, L_total, K] in float32 for robust math
        out = torch.empty((N, L_total, K), device=device, dtype=torch.float32)

        # 2) Launch Triton concatenation kernel
        # Grid: (N, L_total, tiles along K). Choose BLOCK_K to balance occupancy and memory throughput.
        BLOCK_K_concat = 128
        grid_concat = (N, L_total, triton.cdiv(K, BLOCK_K_concat))
        _concat_sequences_kernel[grid_concat](
            out, encoder_hidden_states.contiguous(), hidden_states.contiguous(),
            L_txt, L_img, K,
            BLOCK_K=BLOCK_K_concat,
            num_warps=4, num_stages=2,
        )

        # 3) Prepare inputs for GEMM: A_rows = out.view(N*L_total, K) and B = process_weight.T
        A_rows = out.reshape(N * L_total, K).contiguous()  # [N_rows, K]
        B = process_weight.t().contiguous()               # [K, K]

        # 4) Allocate output rows buffer and run GEMM Triton kernel
        C_rows = torch.empty((N * L_total, K), device=device, dtype=torch.float32)
        N_rows = N * L_total
        BLOCK_K_gemm = 128 if K >= 128 else 64
        grid_gemm = (N_rows,)
        _matmul_row_kernel[grid_gemm](
            C_rows, A_rows, B,
            N_rows, K,
            BLOCK_K=BLOCK_K_gemm,
            num_warps=4, num_stages=2,
        )

        # 5) Reshape back to [N, L_total, K] and split into two streams
        processed = C_rows.view(N, L_total, K)
        processed_encoder = processed[:, :L_txt, :]
        processed_hidden = processed[:, L_txt:, :]

        # Cast back to original dtype if needed
        orig_dtype = hidden_states.dtype
        if processed.dtype != orig_dtype:
            processed_encoder = processed_encoder.to(orig_dtype)
            processed_hidden = processed_hidden.to(orig_dtype)

        return processed_encoder, processed_hidden