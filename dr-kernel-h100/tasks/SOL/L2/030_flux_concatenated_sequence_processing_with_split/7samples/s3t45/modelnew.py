import torch
import triton
import triton.language as tl


@triton.jit
def _concat_sequences_kernel(out_ptr, enc_ptr, hid_ptr,
                              N, L_txt, L_img, K,
                              out_stride_n, out_stride_t, out_stride_k,
                              enc_stride_n, enc_stride_t, enc_stride_k,
                              hid_stride_n, hid_stride_t, hid_stride_k):
    # Grid: (N, L_total, tiles along K)
    n = tl.program_id(0)
    t = tl.program_id(1)
    k_tile = tl.program_id(2)

    # K offsets for this tile
    BLOCK_K = 128
    k_offsets = k_tile * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_k = k_offsets < K

    # Determine source: first L_txt positions from encoder, rest from hidden
    is_enc = t < L_txt

    # Compute pointers
    out_ptr_row = out_ptr + n * out_stride_n + t * out_stride_t + k_offsets * out_stride_k

    if is_enc:
        enc_ptr_row = enc_ptr + n * enc_stride_n + t * enc_stride_t + k_offsets * enc_stride_k
        tl.load(enc_ptr_row, mask=mask_k, other=0.0)
        tl.store(out_ptr_row, tl.load(enc_ptr_row, mask=mask_k, other=0.0))
    else:
        hid_ptr_row = hid_ptr + n * hid_stride_n + (t - L_txt) * hid_stride_t + k_offsets * hid_stride_k
        tl.load(hid_ptr_row, mask=mask_k, other=0.0)
        tl.store(out_ptr_row, tl.load(hid_ptr_row, mask=mask_k, other=0.0))


@triton.jit
def _matmul_tiled_kernel(C, A, B,
                          M, N, K,
                          stride_c_m, stride_c_n, stride_c_k,
                          stride_a_m, stride_a_k, stride_a_n,  # Note: A is [M, K] treated as [M, N] with stride_a_n = stride_a_k, but we pass strides explicitly
                          stride_b_k, stride_b_n,
                          BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    # Grid: (ceil_div(M, BLOCK_M), ceil_div(N, BLOCK_N), ceil_div(K, BLOCK_K))
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    k_offsets = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)

    # Pointers for current tiles
    A_ptrs = A + m_offsets[:, None] * stride_a_m + k_offsets[None, :] * stride_a_k
    B_ptrs = B + k_offsets[:, None] * stride_b_k + n_offsets[None, :] * stride_b_n

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for kk in range(0, K, BLOCK_K):
        # Masks for tails
        mask_a = (m_offsets[:, None] < M) & (kk + k_offsets[None, :] < K)
        mask_b = (kk + k_offsets[:, None] < K) & (n_offsets[None, :] < N)

        a = tl.load(A_ptrs, mask=mask_a, other=0.0)  # [BLOCK_M, BLOCK_K]
        b = tl.load(B_ptrs, mask=mask_b, other=0.0)  # [BLOCK_K, BLOCK_N]

        # Accumulate
        acc += tl.dot(a, b)

        # Advance pointers along K
        A_ptrs += BLOCK_K * stride_a_k
        B_ptrs += BLOCK_K * stride_b_k

    # Store result
    C_ptrs = C + m_offsets[:, None] * stride_c_m + n_offsets[None, :] * stride_c_n
    mask_c = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    tl.store(C_ptrs, acc, mask=mask_c)


def _triton_concat_and_linear(hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Triton version: concatenate along sequence, then apply linear projection via Triton GEMM, and return split streams.
    """
    assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Triton requires CUDA tensors"
    N = hidden_states.shape[0]
    L_txt = encoder_hidden_states.shape[1]
    L_img = hidden_states.shape[1]
    K = hidden_states.shape[2]
    L_total = L_txt + L_img

    # Allocate concatenated tensor
    concatenated = torch.empty((N, L_total, K), device=hidden_states.device, dtype=torch.float32)

    # Launch Triton concat kernel
    BLOCK_K = 128
    grid_concat = (N, L_total, triton.cdiv(K, BLOCK_K))
    _concat_sequences_kernel[grid_concat](
        concatenated, encoder_hidden_states, hidden_states,
        N, L_txt, L_img, K,
        concatenated.stride(0), concatenated.stride(1), concatenated.stride(2),
        encoder_hidden_states.stride(0), encoder_hidden_states.stride(1), encoder_hidden_states.stride(2),
        hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
        num_warps=4, num_stages=2,
    )

    # Prepare operands for GEMM:
    # A = concatenated as [N_rows, K], B = process_weight.T as [K, K]
    N_rows = N * L_total
    A = concatenated.contiguous().view(N_rows, K)  # float32
    B = process_weight.t().contiguous()           # float32
    C_rows = torch.empty((N_rows, K), device=hidden_states.device, dtype=torch.float32)

    # GEMM: C_rows = A @ B
    BLOCK_M = 1  # We can vary; 1 is fine for generality, but 32 may be better. We keep 1 to avoid host-loop and let grid cover rows.
    BLOCK_N = 128
    BLOCK_K = 128
    grid_gemm = (triton.cdiv(N_rows, BLOCK_M), triton.cdiv(K, BLOCK_N), triton.cdiv(K, BLOCK_K))
    _matmul_tiled_kernel[grid_gemm](
        C_rows, A, B,
        N_rows, K, K,  # note: K as N and K for reduction
        C_rows.stride(0), C_rows.stride(1), 0,  # stride_c_k assumed by kernel is not needed; we pass strides for m and n
        A.stride(0), A.stride(1), A.stride(2),  # for A: (m=rows stride, k stride, n stride? pass a,b strides to match pointer math)
        B.stride(0), B.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=2,
    )

    # Reshape back and split using torch slicing (simple and robust)
    processed = C_rows.view(N, L_total, K)
    processed_encoder = processed[:, :L_txt, :]
    processed_hidden = processed[:, L_txt:, :]

    return processed_encoder, processed_hidden


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Triton forward: use Triton for concat and GEMM (numeric work)
        processed_encoder, processed_hidden = _triton_concat_and_linear(hidden_states, encoder_hidden_states, process_weight)
        return processed_encoder, processed_hidden