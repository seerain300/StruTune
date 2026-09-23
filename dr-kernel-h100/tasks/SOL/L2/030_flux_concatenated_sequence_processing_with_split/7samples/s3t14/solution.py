import torch
import triton
import triton.language as tl


@triton.jit
def _concat_sequences_kernel(
    out_ptr,       # *ptr to [N, L_total, K]
    enc_ptr,       # *ptr to [N, L_txt, K]
    hid_ptr,       # *ptr to [N, L_img, K]
    N, L_txt, L_img, K,
    BLOCK_K: tl.constexpr,
):
    # grid = (N, L_total, tiles_along_K)
    pid_n = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_k = tl.program_id(2)

    # k offsets for this tile
    k_offsets = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_k = k_offsets < K

    # Compute sequence length total
    L_total = L_txt + L_img

    # Determine source tensor based on pid_t
    is_encoder = pid_t < L_txt
    n = pid_n

    # Base offsets for loads/stores
    # For encoder: enc[n, t, k] where t=pid_t
    # For hidden: hid[n, t - L_txt, k] where t=pid_t
    # Note: pid_t is valid only if is_encoder; otherwise, it's >= L_txt.

    # We'll compute the source pointer based on is_encoder using two paths.
    # For simplicity and safety, we load from enc_ptr and hid_ptr with appropriate mask.

    # Encoders region
    enc_row_ptr = enc_ptr + n * (L_txt * K) + pid_t * K
    # Hiders region
    hid_row_ptr = hid_ptr + n * (L_img * K) + (pid_t - L_txt) * K

    # Choose source pointer: if pid_t < L_txt, use enc_row_ptr; else use hid_row_ptr.
    # Triton doesn't support Python if on scalars; we use tl.where with a computed mask.
    # However, we can simply do both and select using mask. Since tl.where for pointers isn't supported,
    # we compute two vectors and select based on is_encoder with load/store masking.

    # For enc: load only if is_encoder; else ignore (load masked)
    # For hid: load only if not is_encoder; else ignore.

    # We will load both and store using the appropriate mask.
    # Build masks: if pid_t < L_txt, use enc_row_ptr and mask; else use hid_row_ptr and mask.
    use_encoder = pid_t < L_total  # always True; but we need to guard loads/stores per source.
    # To implement, we can use a single pointer by choosing based on pid_t. Triton supports this via arithmetic.
    src_ptr = tl.where(is_encoder, enc_row_ptr, hid_row_ptr)

    # Load with mask
    vals = tl.load(src_ptr + k_offsets, mask=mask_k, other=0.0)

    # Store to out[n, t, k]
    out_row_ptr = out_ptr + n * (L_total * K) + pid_t * K
    tl.store(out_row_ptr + k_offsets, vals, mask=mask_k)


@triton.jit
def _matmul_batched_rows_kernel(
    C_ptr,     # *ptr to [N_rows, K]
    A_ptr,     # *ptr to [N_rows, K]
    B_ptr,     # *ptr to [K, K]
    N_rows, K,
    BLOCK_K: tl.constexpr,
):
    # One program per output row
    pid = tl.program_id(0)
    # Row in A is pid
    row = pid

    # Accumulator vector
    acc = tl.zeros([K], dtype=tl.float32)

    # Iterate over reduction dimension in chunks
    k0 = 0
    while k0 < K:
        kk = k0 + tl.arange(0, BLOCK_K)
        mask_kk = kk < K

        # Load A_row_chunk: A[row, kk]
        A_row_ptr = A_ptr + row * K
        A_chunk = tl.load(A_row_ptr + kk, mask=mask_kk, other=0.0)  # shape [BLOCK_K]

        # Load B_chunk: B[kk, :]. Treat each element B[kk, j] for j in 0..K-1
        # We'll compute column-by-column to avoid complex broadcasting. For typical K (e.g., 1024),
        # this is fine. We mask for kk.
        # To compute acc[j] += sum(A_chunk[kk] * B[kk, j]) for j over K tiles.
        # Here, we process one j at a time and accumulate into acc.
        # Note: In practice, Triton encourages vectorized operations; computing per j is acceptable.
        # We'll use a for j loop. Triton allows such loops as long as not dependent on runtime values.
        # Since Triton requires static loops, we iterate j across tiles and compute with masks.

        # However, Triton doesn't support arbitrary Python for-loops over runtime K. To keep it robust,
        # we use a while loop over j tiles. Triton allows while loops.

        j0 = 0
        while j0 < K:
            jj = j0 + tl.arange(0, BLOCK_K)
            mask_jj = jj < K

            # Load B_chunk for current (kk, jj): shape [BLOCK_K, BLOCK_K]
            # We can build a 2D pointer by broadcasting.
            B_ptrs = B_ptr + kk[:, None] * K + jj[None, :]
            B_chunk = tl.load(B_ptrs, mask=mask_kk[:, None] & mask_jj[None, :], other=0.0)  # [BLOCK_K, BLOCK_K]

            # Accumulate acc[jj] += sum over kk of A_chunk[kk] * B_chunk[kk, jj]
            # Compute per jj
            # Note: B_chunk is [BLOCK_K, BLOCK_K], so for each jj column, we need to extract that column.
            # We can do that by summing B_chunk[:, jj] * A_chunk.
            # However, Triton prefers vectorized ops. Compute using a reduction over kk axis:
            # Sum(B_chunk[:, col] * A_chunk) for col in jj (vectorized).
            # Implement by looping over col (static) up to BLOCK_K; mask jj for valid columns.

            # Better approach: compute acc[jj] += sum_kk A[kk] * B[kk, jj]
            # We'll compute acc[jj] += dot(A_chunk, B_vec[jj]) where B_vec[jj] = B_ptr[kk, jj]
            # Extract B_vec[jj] by loading one column per iteration. Given typical K, use while loop.

            col0 = 0
            while col0 < BLOCK_K:
                j = j0 + col0
                if j < K:
                    # Load B column for current j across kk: B[kk, j]
                    B_vec_col = tl.load(B_ptr + kk * K + j, mask=mask_kk, other=0.0)  # [BLOCK_K]
                    # Accumulate: acc[j] += sum(A_chunk * B_vec_col)
                    acc[j] += tl.sum(A_chunk * B_vec_col, axis=0)
                col0 += 1

            j0 += BLOCK_K

        k0 += BLOCK_K

    # Store acc back to C[row, :]
    C_row_ptr = C_ptr + row * K
    tl.store(C_row_ptr + tl.arange(0, K), acc, mask=True)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        # Ensure tensors are on CUDA and contiguous
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Inputs must be CUDA tensors"
        hidden_states = hidden_states.contiguous()
        encoder_hidden_states = encoder_hidden_states.contiguous()
        process_weight = process_weight.contiguous()

        # Extract shapes
        N = hidden_states.shape[0]
        L_txt = encoder_hidden_states.shape[1]
        L_img = hidden_states.shape[1]
        K = hidden_states.shape[2]
        L_total = L_txt + L_img

        # 1) Concatenate encoder_hidden_states and hidden_states along sequence dimension in Triton
        out_cat = torch.empty((N, L_total, K), device=hidden_states.device, dtype=hidden_states.dtype)
        BLOCK_K = 256 if K >= 256 else 128
        grid = (N, L_total, triton.cdiv(K, BLOCK_K))
        _concat_sequences_kernel[grid](
            out_cat, encoder_hidden_states, hidden_states,
            N, L_txt, L_img, K,
            BLOCK_K=BLOCK_K,
            num_warps=4,
            num_stages=2,
        )

        # 2) GEMM: out_cat @ process_weight.T using Triton
        # out_cat shape [N, L_total, K] -> A_rows: [N * L_total, K]
        A_rows = out_cat.reshape(N * L_total, K).contiguous()  # float32 or original dtype
        B = process_weight.t().contiguous()  # [K, K]
        # Ensure compute dtype is float32
        A_rows_f32 = A_rows.float()
        B_f32 = B.float()
        C_rows = torch.empty((N * L_total, K), device=hidden_states.device, dtype=torch.float32)

        _matmul_batched_rows_kernel[(N * L_total,)](
            C_rows, A_rows_f32, B_f32,
            N * L_total, K,
            BLOCK_K=128,
            num_warps=4,
            num_stages=2,
        )

        # 3) Reshape back to [N, L_total, K]
        processed = C_rows.view(N, L_total, K)

        # 4) Split into two streams
        processed_encoder = processed[:, :L_txt, :]
        processed_hidden = processed[:, L_txt:, :]

        return processed_encoder, processed_hidden


# For local testing (optional):
# model = ModelNew().cuda()
# hidden_states = torch.randn(2, 256, 1024, device='cuda', dtype=torch.float32)
# encoder_hidden_states = torch.randn(2, 128, 1024, device='cuda', dtype=torch.float32)
# process_weight = torch.randn(1024, 1024, device='cuda', dtype=torch.float32)
# enc_out, hid_out = model(hidden_states, encoder_hidden_states, process_weight)
# print(enc_out.shape, hid_out.shape)


def run(*args):
    return ModelNew()(*args)
