import torch
import triton
import triton.language as tl


@triton.jit
def _concat_sequences_kernel(
    out_ptr,        # *const T, [N, L_total, K]
    enc_ptr,        # *const T, [N, L_txt, K]
    hid_ptr,        # *const T, [N, L_img, K]
    N, L_txt, L_img, K,  # int32
    TILE_K: tl.constexpr,
):
    # program ids: (n, t, tile_k)
    n = tl.program_id(0)
    t = tl.program_id(1)
    tile_k = tl.program_id(2)

    k_offsets = tile_k * TILE_K + tl.arange(0, TILE_K)
    mask_k = k_offsets < K

    # Determine source tensor: if t < L_txt, use encoder; else use hidden at (t - L_txt)
    use_encoder = t < L_txt

    # Base offsets for row (n, t) in both sources
    enc_row_offset = n * (L_txt * K) + t * K
    hid_row_offset = n * (L_img * K) + (t - L_txt) * K

    # Compute pointers
    if use_encoder:
        # Load from encoder[n, t, :]
        # enc_ptr is contiguous: element at (n, t, k) has offset = n*(L_txt*K) + t*K + k
        enc_vals = tl.load(enc_ptr + enc_row_offset + k_offsets, mask=mask_k, other=0.0)
    else:
        # Load from hidden[n, t - L_txt, :]
        hid_vals = tl.load(hid_ptr + hid_row_offset + k_offsets, mask=mask_k, other=0.0)
        enc_vals = hid_vals  # just reuse to pass type; not used when use_encoder is False

    # Compute out row offset: out[n, t, k] has offset = n*(L_total*K) + t*K + k
    L_total = L_txt + L_img
    out_row_offset = n * (L_total * K) + t * K

    # Store to out
    tl.store(out_ptr + out_row_offset + k_offsets, enc_vals, mask=mask_k)


@triton.jit
def _matmul_row_kernel(
    C_ptr,      # *T, [N_rows, K]
    A_ptr,      # *const T, [N_rows, K]
    B_ptr,      # *const T, [K, K]
    N_rows, K,  # int32
    BLOCK_K: tl.constexpr,
):
    # One program per row
    row_id = tl.program_id(0)

    # Accumulator for the output row (1 x K), keep in fp32 for numeric stability
    acc = tl.zeros((1, K), dtype=tl.float32)

    k_start = 0
    while k_start < K:
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < K

        # Load A chunk: A[row_id, k_offsets]
        # A_ptr is contiguous: element at (row_id, k) has offset = row_id*K + k
        a = tl.load(A_ptr + row_id * K + k_offsets, mask=mask_k, other=0.0)

        # Load B chunk: B[k_offsets, :]
        # B_ptr is contiguous: element at (k, kk) has offset = k*K + kk
        b = tl.load(B_ptr + k_offsets[:, None] * K + tl.arange(0, K), mask=mask_k[:, None], other=0.0)

        # Compute outer product for this chunk: [1, K]
        acc += a[:, None] * b

        k_start += BLOCK_K

    # Store result row
    tl.store(C_ptr + row_id * K + tl.arange(0, K), acc[0, :], mask=(tl.arange(0, K) < K))


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        Triton-optimized version of:
            concatenated = torch.cat([encoder_hidden_states, hidden_states], dim=1)
            processed = torch.matmul(concatenated, process_weight.t())
            processed_encoder = processed[:, :text_seq_len, :]
            processed_hidden = processed[:, text_seq_len:, :]
        """
        assert hidden_states.ndim == 3 and encoder_hidden_states.ndim == 3, "Inputs must be [N, L, K]"
        N = hidden_states.shape[0]
        L_img = hidden_states.shape[1]
        L_txt = encoder_hidden_states.shape[1]
        K = hidden_states.shape[2]
        assert encoder_hidden_states.shape[0] == N and encoder_hidden_states.shape[2] == K, "Mismatched batch/hidden_dim"
        assert process_weight.shape[0] == K and process_weight.shape[1] == K, "process_weight must be [K, K]"

        device = hidden_states.device
        dtype = hidden_states.dtype

        # Ensure contiguity for Triton
        enc = encoder_hidden_states.contiguous()
        hid = hidden_states.contiguous()
        W = process_weight.contiguous()

        # 1) Concatenate along sequence dim: [N, L_total, K]
        L_total = L_txt + L_img
        out_cat = torch.empty((N, L_total, K), device=device, dtype=dtype)

        # Triton grid: (N, L_total, tiles along K)
        TILE_K = 128  # K tile for concatenation; masks handle tails
        grid_concat = (N, L_total, triton.cdiv(K, TILE_K))
        _concat_sequences_kernel[grid_concat](
            out_cat, enc, hid,
            N, L_txt, L_img, K,
            TILE_K=TILE_K,
        )

        # 2) GEMM: out_cat @ W.T -> [N, L_total, K]
        # Flatten rows: N_rows = N * L_total
        N_rows = N * L_total
        A_rows = out_cat.reshape(N_rows, K).contiguous()  # [N_rows, K]
        # B = W.T -> [K, K]
        B = W.t().contiguous()  # [K, K], fp32 for compute stability
        C_rows = torch.empty((N_rows, K), device=device, dtype=torch.float32)

        # Launch per-row GEMM kernel
        BLOCK_K = 256 if K >= 256 else 128
        grid_gemm = (N_rows,)
        _matmul_row_kernel[grid_gemm](
            C_rows, A_rows, B,
            N_rows, K,
            BLOCK_K=BLOCK_K,
        )

        # Reshape back to [N, L_total, K]
        processed = C_rows.view(N, L_total, K)

        # 3) Split into two streams
        processed_encoder = processed[:, :L_txt, :]
        processed_hidden = processed[:, L_txt:, :]

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
