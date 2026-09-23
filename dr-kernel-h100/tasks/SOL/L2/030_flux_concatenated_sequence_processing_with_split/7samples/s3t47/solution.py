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

    # Compute t relative to L_txt
    t = pid_t

    # K tile size
    BLOCK_K = 128

    # Create K offsets and mask
    k_offsets = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    k_mask = k_offsets < K

    # Compute base pointers for current (n, t) and each K chunk
    enc_ptrs = enc_ptr + pid_n * stride_enc_n + t * stride_enc_t + k_offsets * stride_enc_k
    hid_ptrs = hid_ptr + pid_n * stride_hid_n + t * stride_hid_t + k_offsets * stride_hid_k
    out_ptrs = out_ptr + pid_n * stride_out_n + t * stride_out_t + k_offsets * stride_out_k

    # Select source based on t (t in [0, L_txt) -> encoder; else -> hidden)
    # Masks ensure we do not read beyond bounds. We compute both enc and hid loads and select via a pointer-based conditional.
    # For Triton, use tl.where to choose which pointer to load from.
    # Note: t is an int; we compare against L_txt.
    is_encoder = t < L_txt

    # Load values: if t < L_txt, load from enc_ptr; else from hid_ptr
    enc_vals = tl.load(enc_ptrs, mask=k_mask, other=0.0)
    hid_vals = tl.load(hid_ptrs, mask=k_mask, other=0.0)
    vals = tl.where(is_encoder, enc_vals, hid_vals)

    # Store into out at (n, t, k_offsets)
    tl.store(out_ptrs, vals, mask=k_mask)


@triton.jit
def _matmul_row_kernel(C_ptr, A_ptr, B_ptr,
                        N, L_total, K,
                        stride_C_n, stride_C_t, stride_C_k,
                        stride_A_n, stride_A_t, stride_A_k,
                        stride_B_k, stride_B_out,
                        BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    # One program handles one output row (n, t). We loop over K in chunks of BLOCK_K and vectorize across output columns in chunks of BLOCK_N.
    pid_row = tl.program_id(0)  # pid_row ranges over N * L_total
    # Map pid_row to (n, t)
    n = pid_row // L_total
    t = pid_row % L_total

    # Accumulator for the row output vector of length K
    acc = tl.zeros((K,), dtype=tl.float32)

    # Loop over K in chunks
    for kk in range(0, K, BLOCK_K):
        k_offsets = kk + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < K

        # Load A_row: concatenated[n, t, k_offsets] -> shape [BLOCK_K]
        A_row_ptrs = A_ptr + n * stride_A_n + t * stride_A_t + k_offsets * stride_A_k
        A_vals = tl.load(A_row_ptrs, mask=k_mask, other=0.0).to(tl.float32)  # [BLOCK_K]

        # Load B chunk: process_weight.T[k_offsets, :] -> shape [BLOCK_K, BLOCK_N]
        B_ptrs = B_ptr + k_offsets[:, None] * stride_B_k + tl.arange(0, BLOCK_N)[None, :] * stride_B_out
        n_mask = tl.arange(0, BLOCK_N) < K  # mask for output columns
        B_vals = tl.load(B_ptrs, mask=(k_mask[:, None] & n_mask[None, :]), other=0.0).to(tl.float32)  # [BLOCK_K, BLOCK_N]

        # Compute partial dot: sum over k of A_vals[k] * B_vals[k, :]
        # Broadcast A_vals to [1, BLOCK_K], multiply with B_vals, reduce along axis=0
        partial = tl.sum(A_vals[:, None] * B_vals, axis=0)  # [BLOCK_N]
        acc[kk:kk + BLOCK_N] += partial  # add into accumulator

    # Store accumulator back into C[n, t, :]
    C_row_ptrs = C_ptr + n * stride_C_n + t * stride_C_t + tl.arange(0, K) * stride_C_k
    tl.store(C_row_ptrs, acc, mask=True)  # mask=True always; acc already has K elements


def _run_concatenate(enc: torch.Tensor, hid: torch.Tensor) -> torch.Tensor:
    # Ensure contiguous tensors
    enc = enc.contiguous()
    hid = hid.contiguous()
    assert enc.shape[0] == hid.shape[0], "Batch sizes must match"
    N = enc.shape[0]
    L_txt = enc.shape[1]
    L_img = hid.shape[1]
    K = enc.shape[2]
    assert enc.shape[2] == hid.shape[2], "Hidden dims must match"
    assert enc.is_cuda and hid.is_cuda, "Inputs must be CUDA tensors for Triton"

    # Allocate output
    out = torch.empty((N, L_txt + L_img, K), device=enc.device, dtype=enc.dtype)

    # Compute grid: (N, L_total, tiles along K)
    L_total = L_txt + L_img
    grid = (N, L_total, triton.cdiv(K, 128))  # K tile size used by kernel

    _concat_sequences_kernel[grid](
        out, enc, hid,
        N, L_txt, L_img, K,
        out.stride(0), out.stride(1), out.stride(2),
        enc.stride(0), enc.stride(1), enc.stride(2),
        hid.stride(0), hid.stride(1), hid.stride(2),
        num_warps=4, num_stages=2,
    )
    return out


def _run_matmul_row(C_rows: torch.Tensor, A_rows: torch.Tensor, B: torch.Tensor, N: int, L_total: int, K: int):
    # Ensure tensors are float32 for stable accumulation
    if C_rows.dtype != torch.float32:
        C_rows = C_rows.float()
    if A_rows.dtype != torch.float32:
        A_rows = A_rows.float()
    if B.dtype != torch.float32:
        B = B.float()

    # Launch one program per output row
    grid = (N * L_total,)
    _matmul_row_kernel[grid](
        C_rows, A_rows, B,
        N, L_total, K,
        C_rows.stride(0), C_rows.stride(1), C_rows.stride(2),
        A_rows.stride(0), A_rows.stride(1), A_rows.stride(2),
        B.stride(0), B.stride(1),
        BLOCK_N=256, BLOCK_K=64,
        num_warps=4, num_stages=2,
    )
    return C_rows


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        Triton-optimized version:
        - Concatenate encoder_hidden_states and hidden_states along sequence dimension in Triton.
        - Apply linear projection via Triton row-wise GEMM.
        - Split into separate streams.
        """
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "All tensors must be CUDA for Triton"
        # 1) Concatenate sequences along sequence dimension
        concatenated = _run_concatenate(encoder_hidden_states, hidden_states)  # [N, L_total, K]

        # 2) Apply linear projection (no bias): processed = concatenated @ process_weight.T
        N = concatenated.shape[0]
        L_total = concatenated.shape[1]
        K = concatenated.shape[2]
        A_rows = concatenated.contiguous().view(N * L_total, K)  # [N_rows, K], N_rows = N * L_total
        B = process_weight.t().contiguous()  # [K, K]
        C_rows = torch.empty((N * L_total, K), device=concatenated.device, dtype=torch.float32)

        C_rows = _run_matmul_row(C_rows, A_rows, B, N, L_total, K)

        # 3) Reshape back and split
        processed = C_rows.view(N, L_total, K)
        processed_encoder = processed[:, :encoder_hidden_states.shape[1], :]
        processed_hidden = processed[:, encoder_hidden_states.shape[1]:, :]

        # Cast back to original dtype
        processed_encoder = processed_encoder.to(encoder_hidden_states.dtype)
        processed_hidden = processed_hidden.to(hidden_states.dtype)

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
