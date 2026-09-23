import torch
import triton
import triton.language as tl


@triton.jit
def _concat_sequences_kernel(out_ptr, encoder_ptr, hidden_ptr,
                             N: tl.int32, L_txt: tl.int32, L_img: tl.int32, K: tl.int32,
                             BLOCK_K: tl.constexpr):
    # Grid: (N, L_total, tiles_along_K)
    pid_n = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_k = tl.program_id(2)

    # Compute offsets
    k_offsets = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    k_mask = k_offsets < K

    # Determine source tensor and index
    is_encoder = pid_t < L_txt
    src_n = pid_n

    # Compute base addresses
    # out[n, t, k] with k vector
    out_row_offset = (pid_n * L_txt + pid_t) * K
    # encoder or hidden base offsets depend on is_encoder
    if is_encoder:
        src_row_offset = src_n * L_txt * K + pid_t * K
    else:
        src_row_offset = src_n * (L_txt + L_img) * K + (pid_t - L_txt) * K

    # Load and store
    vals = tl.load(encoder_ptr + src_row_offset + k_offsets, mask=k_mask, other=0.0) if is_encoder else \
           tl.load(hidden_ptr + src_row_offset + k_offsets, mask=k_mask, other=0.0)
    tl.store(out_ptr + out_row_offset + k_offsets, vals, mask=k_mask)


@triton.jit
def _gemb_row_gemm_fp32(C_rows_ptr, A_rows_ptr, B_ptr,
                        N_rows: tl.int32, K: tl.int32, BLOCK_K: tl.constexpr):
    # Each program handles one row (one [K] vector), iterates over K in chunks of BLOCK_K
    row_id = tl.program_id(0)  # 0..N_rows-1
    # Initialize accumulator
    acc = tl.zeros([K], dtype=tl.float32)

    # Iterate over reduction dimension K in chunks
    kk = 0
    while kk < K:
        k_offsets = kk + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < K

        # Load A row chunk: A_rows[row_id, k_offsets]
        a_ptrs = A_rows_ptr + row_id * K + k_offsets
        a_vals = tl.load(a_ptrs, mask=k_mask, other=0.0)  # float32

        # Load B chunk: B[k_offsets, :]
        b_ptrs = B_ptr + k_offsets[:, None] * K + tl.arange(0, K)[None, :]
        b_mask = (k_offsets[:, None] < K) & (tl.arange(0, K)[None, :] < K)
        b_vals = tl.load(b_ptrs, mask=b_mask, other=0.0)  # float32 [BLOCK_K, K]

        # Accumulate outer product: acc += a_vals[:, None] * b_vals
        # a_vals[:, None]: [BLOCK_K, 1]
        acc += tl.sum(a_vals[:, None] * b_vals, axis=0)

        kk += BLOCK_K

    # Store result
    c_ptrs = C_rows_ptr + row_id * K + tl.arange(0, K)
    tl.store(c_ptrs, acc)


def _triton_concat_and_gemm(encoder_hidden_states: torch.Tensor,
                            hidden_states: torch.Tensor,
                            process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # Ensure CUDA and dtype (compute in fp32)
    device = encoder_hidden_states.device
    assert encoder_hidden_states.is_cuda and hidden_states.is_cuda and process_weight.is_cuda, "Tensors must be CUDA for Triton."

    # Shapes
    N = encoder_hidden_states.shape[0]
    L_txt = encoder_hidden_states.shape[1]
    L_img = hidden_states.shape[1]
    K = encoder_hidden_states.shape[2]
    assert hidden_states.shape[2] == K and process_weight.shape[0] == K and process_weight.shape[1] == K

    # Make inputs contiguous and cast to float32 for numeric stability
    encoder = encoder_hidden_states.contiguous().to(torch.float32)
    hidden = hidden_states.contiguous().to(torch.float32)
    weight_T = process_weight.t().contiguous().to(torch.float32)  # [K, K]

    # 1) Concatenate encoder and hidden along sequence dim: out [N, L_total, K]
    L_total = L_txt + L_img
    out = torch.empty((N, L_total, K), device=device, dtype=torch.float32)

    # Launch concatenation kernel
    BLOCK_K = 128  # tile along K, any reasonable value; masked for tails
    grid = (N, L_total, triton.cdiv(K, BLOCK_K))
    _concat_sequences_kernel[grid](
        out, encoder, hidden,
        N, L_txt, L_img, K,
        BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=2,
    )

    # 2) GEMM: out_rows @ weight_T -> C_rows [N_rows, K], where out_rows = out.view(N_rows, K)
    N_rows = N * L_total
    A_rows = out.view(N_rows, K)  # [N_rows, K], float32
    C_rows = torch.empty((N_rows, K), device=device, dtype=torch.float32)

    # Launch GEMM kernel: one program per row
    BLOCK_K_GEMM = 256  # reduction chunk; loop handles tail
    grid_gemm = (N_rows,)
    _gemb_row_gemm_fp32[grid_gemm](
        C_rows, A_rows, weight_T,
        N_rows, K,
        BLOCK_K=BLOCK_K_GEMM,
        num_warps=4, num_stages=2,
    )

    # 3) Reshape back and split into encoder and hidden parts
    processed = C_rows.view(N, L_total, K)  # [N, L_total, K]
    processed_encoder = processed[:, :L_txt, :]
    processed_hidden = processed[:, L_txt:, :]

    # Cast back to original dtypes
    processed_encoder = processed_encoder.to(encoder_hidden_states.dtype)
    processed_hidden = processed_hidden.to(hidden_states.dtype)

    return processed_encoder, processed_hidden


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return _triton_concat_and_gemm(encoder_hidden_states, hidden_states, process_weight)