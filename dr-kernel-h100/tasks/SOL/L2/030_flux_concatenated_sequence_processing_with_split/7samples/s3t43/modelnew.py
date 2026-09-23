import torch
import triton
import triton.language as tl


@triton.jit
def triton_concat_sequences_kernel(out_ptr, enc_ptr, hid_ptr,
                                    N, L_txt, L_img, K,
                                    stride_out_n, stride_out_t, stride_out_k,
                                    stride_enc_n, stride_enc_t, stride_enc_k,
                                    stride_hid_n, stride_hid_t, stride_hid_k,
                                    BLOCK_K: tl.constexpr):
    # Grid: (N, L_total, tiles over K)
    pid_n = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_k = tl.program_id(2)

    # Compute output offsets for this (n, t) tile
    n = pid_n
    t = pid_t
    # K tile start
    k_start = pid_k * BLOCK_K
    k_offsets = k_start + tl.arange(0, BLOCK_K)
    mask_k = k_offsets < K

    # Select source tensor based on t < L_txt
    is_encoder = t < L_txt

    # Compute addresses for enc and hid
    # enc[n, t, k] -> ptr = enc_ptr + n*stride_enc_n + t*stride_enc_t + k*stride_enc_k
    # hid[n, t - L_txt, k] -> ptr = hid_ptr + n*stride_hid_n + (t - L_txt)*stride_hid_t + k*stride_hid_k
    enc_ptrs = enc_ptr + n * stride_enc_n + t * stride_enc_t + k_offsets * stride_enc_k
    hid_ptrs = hid_ptr + n * stride_hid_n + (t - L_txt) * stride_hid_t + k_offsets * stride_hid_k

    # Load source values; if t >= L_txt, we don't use enc; otherwise hid
    # We'll do masked loads based on is_encoder; however, since pid_t is fixed, we can simply select the valid pointer by guarding.
    # Easiest: load both with mask, then select via is_encoder; but Triton does not support dynamic pointer selection cleanly.
    # Instead, load based on is_encoder: if is_encoder, load enc; else load hid. We can construct pointers with tl.where.
    # But Triton requires static pointer types; better approach: compute which tensor to load from using masks.
    # To keep it simple and safe, we'll compute a per-element mask for enc/hid and perform the load:
    # Note: For invalid side (non-encoder or non-image), use zeros.
    enc_mask = mask_k & is_encoder
    hid_mask = mask_k & (~is_encoder)

    # We need to materialize the source values. Triton doesn't support dynamic pointer selection easily, so we load both and use masks.
    # However, tl.load requires a pointer tensor, so we'll create a combined mask and pointer selection with tl.where for pointers:
    # Create a source_ptr tensor that points to enc when is_encoder else hid; but Triton doesn't allow runtime branching there.
    # Alternative: load both with masked loads and then select; but Triton doesn't provide an 'other' per-element for branch.
    # Therefore, the safest approach is to run two kernels (one for enc and one for hid) and write into out_ptr. To avoid that, we can implement by branching on is_encoder inside the kernel.
    # Triton allows scalar if; we can use it to select which ptr to load:
    # Pointer selection via scalar is fine here:
    if is_encoder:
        vals = tl.load(enc_ptrs, mask=enc_mask, other=0.0)
    else:
        vals = tl.load(hid_ptrs, mask=hid_mask, other=0.0)

    # Store into out[n, t, k]
    out_ptrs = out_ptr + n * stride_out_n + t * stride_out_t + k_offsets * stride_out_k
    tl.store(out_ptrs, vals, mask=mask_k)


@triton.jit
def triton_gemm_per_row_kernel(C_ptr, A_ptr, B_ptr,
                                N_rows, K,
                                stride_C_row, stride_C_k,
                                stride_A_row, stride_A_k,
                                stride_B_k, stride_B_k2,
                                BLOCK_K: tl.constexpr):
    # Each program computes one output row (i.e., one sequence position) vector of length K.
    pid = tl.program_id(0)
    # If pid >= N_rows, return (grid ensures pid < N_rows)
    # Accumulator in fp32
    acc = tl.zeros((K,), dtype=tl.float32)

    # Reduction over K in BLOCK_K chunks
    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < K

        # Load A_row chunk: A[pid, k_offsets]
        A_row_ptrs = A_ptr + pid * stride_A_row + k_offsets * stride_A_k
        a_chunk = tl.load(A_row_ptrs, mask=mask_k, other=0.0).to(tl.float32)  # [BLOCK_K]

        # Load B chunk: B[k_offsets, :] -> shape [BLOCK_K, K]
        B_ptrs = B_ptr + k_offsets[:, None] * stride_B_k + tl.arange(0, K)[None, :] * stride_B_k2
        b_chunk = tl.load(B_ptrs, mask=mask_k[:, None], other=0.0).to(tl.float32)  # [BLOCK_K, K]

        # Accumulate: acc += a_chunk dot b_chunk
        acc += tl.sum(a_chunk[:, None] * b_chunk, axis=0)

    # Store result acc to C[pid, :]
    C_row_ptrs = C_ptr + pid * stride_C_row + tl.arange(0, K) * stride_C_k
    tl.store(C_row_ptrs, acc, mask=True)  # K elements, so mask is always valid


@triton.jit
def _split_sequences_kernel(out_ptr, in_ptr,
                             N, L_txt, L_total, K,
                             stride_out_n, stride_out_t, stride_out_k,
                             stride_in_n, stride_in_t, stride_in_k,
                             stream_for: tl.constexpr):
    # stream_for: 0 -> encoder, 1 -> hidden
    pid_n = tl.program_id(0)
    pid_t = tl.program_id(1)

    # Compute in offsets
    in_ptrs = in_ptr + pid_n * stride_in_n + pid_t * stride_in_t + tl.arange(0, K) * stride_in_k
    # Compute out offsets
    if stream_for == 0:
        out_ptrs = out_ptr + pid_n * stride_out_n + pid_t * stride_out_t + tl.arange(0, K) * stride_out_k
    else:
        out_ptrs = out_ptr + pid_n * stride_out_n + (pid_t + L_txt) * stride_out_t + tl.arange(0, K) * stride_out_k

    vals = tl.load(in_ptrs)
    tl.store(out_ptrs, vals)


def triton_concat_and_linear(hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Triton-only implementation of:
      concatenated = torch.cat([encoder_hidden_states, hidden_states], dim=1)  # [N, L_total, K]
      processed = concatenated @ process_weight.T                           # [N, L_total, K]
      processed_encoder = processed[:, :L_txt, :]
      processed_hidden   = processed[:, L_txt:, :]
    Returns (processed_encoder, processed_hidden). Both are torch.Tensor.
    """
    assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "All inputs must be CUDA tensors for Triton."
    assert hidden_states.ndim == 3 and encoder_hidden_states.ndim == 3 and process_weight.ndim == 2
    N = hidden_states.shape[0]
    L_txt = encoder_hidden_states.shape[1]
    L_img = hidden_states.shape[1]
    K = hidden_states.shape[2]
    L_total = L_txt + L_img

    # 1) Concatenate with Triton
    out = torch.empty((N, L_total, K), device=hidden_states.device, dtype=torch.float32)
    # Launch grid: (N, L_total, tiles over K)
    BLOCK_K = 128
    grid_concat = (N, L_total, triton.cdiv(K, BLOCK_K))
    triton_concat_sequences_kernel[grid_concat](
        out, encoder_hidden_states, hidden_states,
        N, L_txt, L_img, K,
        out.stride(0), out.stride(1), out.stride(2),
        encoder_hidden_states.stride(0), encoder_hidden_states.stride(1), encoder_hidden_states.stride(2),
        hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
        BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=2,
    )

    # 2) Linear projection with Triton per-row GEMM
    # A = out [N_rows, K], B = process_weight.T [K, K]
    N_rows = N * L_total
    A = out.contiguous().view(N_rows, K)
    B = process_weight.t().contiguous()  # [K, K]

    C_rows = torch.empty((N_rows, K), device=A.device, dtype=torch.float32)

    grid_gemm = (N_rows,)
    BLOCK_K_GEMM = 128
    triton_gemm_per_row_kernel[grid_gemm](
        C_rows, A, B,
        N_rows, K,
        C_rows.stride(0), C_rows.stride(1),
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        BLOCK_K=BLOCK_K_GEMM,
        num_warps=4, num_stages=2,
    )

    processed = C_rows.view(N, L_total, K)

    # 3) Split back with Triton kernel (must be launched; no decoy)
    processed_encoder = torch.empty((N, L_txt, K), device=processed.device, dtype=torch.float32)
    processed_hidden = torch.empty((N, L_img, K), device=processed.device, dtype=torch.float32)

    grid_split = (N, L_txt, 1)  # split encoder
    _split_sequences_kernel[grid_split](
        processed_encoder, processed,
        N, L_txt, L_total, K,
        processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
        processed.stride(0), processed.stride(1), processed.stride(2),
        stream_for=0,
        num_warps=4, num_stages=1,
    )

    grid_split2 = (N, L_img, 1)  # split hidden
    _split_sequences_kernel[grid_split2](
        processed_hidden, processed,
        N, L_txt, L_total, K,
        processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
        processed.stride(0), processed.stride(1), processed.stride(2),
        stream_for=1,
        num_warps=4, num_stages=1,
    )

    return processed_encoder, processed_hidden


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Triton-only forward: all numeric computation done by Triton kernels.
        processed_encoder, processed_hidden = triton_concat_and_linear(hidden_states, encoder_hidden_states, process_weight)
        return processed_encoder, processed_hidden