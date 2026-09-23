import torch
import triton
import triton.language as tl


@triton.jit
def _concat_sequences_kernel(
    out_ptr,           # *fp32
    encoder_ptr,       # *fp32
    hidden_ptr,        # *fp32
    L_txt: tl.constexpr,   # int
    L_img: tl.constexpr,   # int
    K: tl.constexpr,       # int (hidden_dim)
    N: tl.constexpr,       # int (batch_size)
    BLOCK_K: tl.constexpr  # int (tile size along K)
):
    # Grid: (N, L_total, tiles along K)
    n = tl.program_id(0)  # batch index
    t = tl.program_id(1)  # concatenated sequence position
    k_tile = tl.program_id(2)

    k_offsets = k_tile * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_k = k_offsets < K

    # Determine source tensor and index
    is_encoder = t < L_txt
    src_idx = t if is_encoder else (t - L_txt)

    # Compute base offsets
    # out[n, t, k] -> out_ptr + ((n * (L_txt + L_img) + t) * K + k)
    out_row_base = (n * (L_txt + L_img) + t) * K
    out_vec = out_ptr + out_row_base + k_offsets

    if is_encoder:
        # encoder[n, t, k] -> encoder_ptr + ((n * L_txt + t) * K + k)
        enc_row_base = (n * L_txt + t) * K
        enc_vec = encoder_ptr + enc_row_base + k_offsets
        tl.store(out_vec, tl.load(enc_vec, mask=mask_k))
    else:
        # hidden[n, t - L_txt, k] -> hidden_ptr + ((n * L_img + src_idx) * K + k)
        hid_row_base = (n * L_img + src_idx) * K
        hid_vec = hidden_ptr + hid_row_base + k_offsets
        tl.store(out_vec, tl.load(hid_vec, mask=mask_k))


@triton.jit
def _row_matmul_kernel(
    C_row_ptr,         # *fp32, shape [K]
    A_row_ptr,         # *fp32, shape [K]
    B_ptr,             # *fp32, shape [K, K]
    K: tl.constexpr,   # int
    BLOCK_K: tl.constexpr
):
    # One program per output row (A_row). Accumulate into C_row (fp32).
    acc = tl.zeros((K,), dtype=tl.float32)

    # Iterate over reduction dimension K in chunks
    for k_start in range(0, K, BLOCK_K):
        kk = k_start + tl.arange(0, BLOCK_K)
        mask_k = kk < K

        # Load A_row chunk (scalar per kk): A_row_ptr[kk]
        a = tl.load(A_row_ptr + kk, mask=mask_k, other=0.0)  # [BLOCK_K]
        # Load B chunk as [BLOCK_K, K] tile: B[kk, :]. Here we build the 2D pointers.
        # Triton allows this pattern: each kk picks a row in B, and we accumulate across kk.
        # We'll perform the dot accumulation manually.
        for jj in range(0, BLOCK_K):
            k_j = k_start + jj
            # mask for jj within K
            # We need b_vec[k_j, :] = B[k_j, :]. Pointer: B_ptr + k_j * K + j_offsets
            j_offsets = tl.arange(0, K)
            b_vec = tl.load(B_ptr + k_j * K + j_offsets, mask=j_offsets < K, other=0.0)
            # Accumulate: acc[j_offsets] += a[jj] * b_vec
            acc += a[jj] * b_vec

    # Store accumulated result
    tl.store(C_row_ptr, acc)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        hidden_states: [N, L_img, K]
        encoder_hidden_states: [N, L_txt, K]
        process_weight: [K, K] (no bias)
        Returns: (processed_encoder: [N, L_txt, K], processed_hidden: [N, L_img, K])
        """
        # Ensure device and dtype consistency
        device = hidden_states.device
        dtype = hidden_states.dtype

        N = hidden_states.shape[0]
        L_img = hidden_states.shape[1]
        L_txt = encoder_hidden_states.shape[1]
        K = hidden_states.shape[2]
        L_total = L_txt + L_img

        # 1) Concatenate along sequence using Triton
        out = torch.empty((N, L_total, K), device=device, dtype=torch.float32)  # compute in fp32
        BLOCK_K = 256 if K >= 256 else 128
        grid_concat = (N, L_total, triton.cdiv(K, BLOCK_K))
        _concat_sequences_kernel[grid_concat](
            out, encoder_hidden_states, hidden_states,
            L_txt, L_img, K, N, BLOCK_K,
            num_warps=4, num_stages=2,
        )

        # 2) Linear projection: out @ process_weight.T using Triton
        # Prepare B = process_weight.T as contiguous fp32
        B = process_weight.t().contiguous().to(torch.float32)  # [K, K]

        # Flatten out to [N_rows, K], compute C_rows [N_rows, K], then reshape
        N_rows = N * L_total
        A_rows = out.reshape(N_rows, K).contiguous().to(torch.float32)  # [N_rows, K], fp32 for accumulation
        C_rows = torch.empty((N_rows, K), device=device, dtype=torch.float32)

        # Launch one program per row
        grid_gemm = (N_rows,)
        _row_matmul_kernel[grid_gemm](
            C_rows, A_rows, B,
            K, BLOCK_K,
            num_warps=4, num_stages=2,
        )

        # 3) Reshape back to [N, L_total, K] and split
        processed = C_rows.view(N, L_total, K)
        processed_encoder = processed[:, :L_txt, :]
        processed_hidden = processed[:, L_txt:, :]

        # Cast back to original dtype if needed
        if processed_encoder.dtype != dtype or processed_hidden.dtype != dtype:
            processed_encoder = processed_encoder.to(dtype)
            processed_hidden = processed_hidden.to(dtype)

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
