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

    L_total = L_txt + L_img
    # Guard pid_t against L_total if grid overruns (not strictly necessary if grid is exact)
    if pid_t >= L_total:
        return

    # Compute tile start along K
    k_start = pid_k * BLOCK_K
    k_offsets = k_start + tl.arange(0, BLOCK_K)
    mask_k = k_offsets < K

    # Select source tensor based on pid_t (which corresponds to the sequence position)
    # t in [0, L_txt) -> encoder, else -> hidden
    # Note: tl.where expects scalars; pid_t is the sequence index
    # Compute base pointers for both sources
    enc_base = enc_ptr + pid_n * stride_enc_n + pid_t * stride_enc_t
    hid_base = hid_ptr + pid_n * stride_hid_n + (pid_t - L_txt) * stride_hid_t  # since pid_t may exceed L_txt

    # Load from appropriate source
    vals = tl.zeros([BLOCK_K], dtype=tl.float32)
    if pid_t < L_txt:
        vals = tl.load(enc_base + k_offsets * stride_enc_k, mask=mask_k, other=0.0)
    else:
        vals = tl.load(hid_base + k_offsets * stride_hid_k, mask=mask_k, other=0.0)

    # Store into out[n, pid_t, k_offsets]
    out_base = out_ptr + pid_n * stride_out_n + pid_t * stride_out_t
    tl.store(out_base + k_offsets * stride_out_k, vals, mask=mask_k)


@triton.jit
def _matmul_row_kernel(C_ptr, A_ptr, B_ptr,
                       N_rows, K,
                       stride_C_row, stride_C_k,
                       stride_A_row, stride_A_k,
                       stride_B_k, stride_B_n,
                       BLOCK_K: tl.constexpr, BLOCK_OUT: tl.constexpr):
    # One program per output row (n, t) across output columns in chunks of BLOCK_OUT
    pid_row = tl.program_id(0)
    pid_out = tl.program_id(1)  # tile along output columns

    # Output column offsets for this program
    out_start = pid_out * BLOCK_OUT
    out_offsets = out_start + tl.arange(0, BLOCK_OUT)
    mask_out = out_offsets < K

    # Accumulator for this row slice
    acc = tl.zeros([BLOCK_OUT], dtype=tl.float32)

    # Loop over reduction dimension K in chunks of BLOCK_K
    # This is a while loop with compile-time chunk size; K is runtime but we iterate in chunks.
    k = 0
    while k < K:
        k_offsets = k + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < K

        # Load A_row chunk: A is [N_rows, K], row index = pid_row
        # Address for A[pid_row, k_offsets]
        A_row_addr = A_ptr + pid_row * stride_A_row + k_offsets * stride_A_k
        A_vals = tl.load(A_row_addr, mask=mask_k, other=0.0)  # shape [BLOCK_K]

        # Load B chunk: B is [K, K], we need B[k_offsets, out_offsets]
        B_tile_addr = B_ptr + k_offsets[:, None] * stride_B_k + out_offsets[None, :] * stride_B_n
        mask_tile = mask_k[:, None] & mask_out[None, :]
        B_tile = tl.load(B_tile_addr, mask=mask_tile, other=0.0)  # shape [BLOCK_K, BLOCK_OUT]

        # Fused multiply-add: acc += sum_i A[i] * B[i, :]
        # Reduce along the K chunk dimension
        # Convert B_tile to float32 and multiply
        B_tile_f = B_tile.to(tl.float32)
        acc += tl.sum(B_tile_f * A_vals[:, None], axis=0)

        k += BLOCK_K

    # Store accumulated results to C[pid_row, out_offsets]
    C_row_addr = C_ptr + pid_row * stride_C_row + out_offsets * stride_C_k
    tl.store(C_row_addr, acc, mask=mask_out)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                encoder_hidden_states: torch.Tensor,
                process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version of the original run function:
        - Concatenate sequences along sequence dimension in Triton.
        - Apply linear projection (A @ W.T) in Triton using per-row kernel without Python loops.
        - Split back into encoder and image streams.
        """
        device = hidden_states.device
        assert device.type == 'cuda', "This Triton implementation requires CUDA tensors."

        N = hidden_states.shape[0]
        L_txt = encoder_hidden_states.shape[1]
        L_img = hidden_states.shape[1]
        K = hidden_states.shape[2]
        L_total = L_txt + L_img

        # 1) Allocate concatenated tensor [N, L_total, K]
        concatenated = torch.empty((N, L_total, K), device=device, dtype=torch.float32)

        # 2) Launch concatenation Triton kernel to fill concatenated
        # Grid: (N, L_total, tiles along K)
        BLOCK_K = 128
        grid_concat = (N, L_total, triton.cdiv(K, BLOCK_K))
        _concat_sequences_kernel[grid_concat](
            concatenated, encoder_hidden_states, hidden_states,
            N, L_txt, L_img, K,
            concatenated.stride(0), concatenated.stride(1), concatenated.stride(2),
            encoder_hidden_states.stride(0), encoder_hidden_states.stride(1), encoder_hidden_states.stride(2),
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            BLOCK_K=BLOCK_K,
        )

        # 3) Apply linear projection: processed = concatenated @ process_weight.T
        # Shape: concatenated [N, L_total, K] -> A [N_rows, K], B [K, K]
        N_rows = N * L_total
        A_rows = concatenated.reshape(N_rows, K).contiguous()
        B = process_weight.t().contiguous()  # [K, K]

        # Prepare output rows [N_rows, K] in float32
        C_rows = torch.empty((N_rows, K), device=device, dtype=torch.float32)

        # Use per-row kernel without Python for-loops: iterate K with while in Triton
        BLOCK_OUT = 256  # output columns per program
        grid_gemm = (N_rows, triton.cdiv(K, BLOCK_OUT))
        _matmul_row_kernel[grid_gemm](
            C_rows, A_rows, B,
            N_rows, K,
            C_rows.stride(0), C_rows.stride(1),
            A_rows.stride(0), A_rows.stride(1),
            B.stride(0), B.stride(1),
            BLOCK_K=64, BLOCK_OUT=BLOCK_OUT,
            num_warps=4, num_stages=2,
        )

        # 4) Reshape back to [N, L_total, K] and split
        processed = C_rows.view(N, L_total, K)
        processed_encoder = processed[:, :L_txt, :]
        processed_hidden = processed[:, L_txt:, :]

        # Cast back to original dtype if needed (the original returned same dtype)
        processed_encoder = processed_encoder.to(encoder_hidden_states.dtype)
        processed_hidden = processed_hidden.to(hidden_states.dtype)

        return processed_encoder, processed_hidden