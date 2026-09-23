import torch
import triton
import triton.language as tl


@triton.jit
def concat_seqs_kernel(
    enc_ptr,          # *const T, [B, T, H]
    hid_ptr,          # *const T, [B, I, H]
    concat_ptr,       # *T,       [B, S, H], S = T + I
    B: tl.constexpr, T: tl.constexpr, I: tl.constexpr, H: tl.constexpr,
    enc_stride0, enc_stride1, enc_stride2,
    hid_stride0, hid_stride1, hid_stride2,
    concat_stride0, concat_stride1, concat_stride2,
    BLOCK_H: tl.constexpr,
):
    # Each program handles one batch b and one sequence index
    pid_m = tl.program_id(axis=0)
    b = pid_m // (T + I)
    m = pid_m % (T + I)
    # We'll iterate over H in blocks to handle general cases
    offs_n = tl.arange(0, BLOCK_H)
    # Determine if this m belongs to encoder or hidden stream
    is_encoder = m < T
    # Compute source pointers for A[m, :]
    if is_encoder:
        a_ptr = enc_ptr + b * enc_stride0 + m * enc_stride1
    else:
        m_i = m - T
        a_ptr = hid_ptr + b * hid_stride0 + m_i * hid_stride1
    # Destination pointer in concatenated
    c_ptr = concat_ptr + b * concat_stride0 + m * concat_stride1

    # Loop over H dimension
    for n0 in range(0, H, BLOCK_H):
        n_idx = n0 + offs_n
        mask = n_idx < H
        a = tl.load(a_ptr + n_idx * enc_stride2, mask=mask, other=0.0)
        tl.store(c_ptr + n_idx * concat_stride2, a, mask=mask)


@triton.jit
def matmul_seqs_kernel(
    A_ptr,            # *const T, [S, H], i.e., rows of concatenated
    B_ptr,            # *const T, [H, H], process_weight.T
    C_ptr,            # *T,       [S, H]
    S: tl.constexpr,  # S = B*T + B*I, but we also pass B, T, I via runtime args and compute in code if needed
    H: tl.constexpr,
    A_stride0, A_stride1,
    B_stride0, B_stride1,
    C_stride0, C_stride1,
    BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # Grid: (S, 1) -> each program handles one row m in A, computes C[m, :]
    m = tl.program_id(axis=0)
    # Accumulator for output row m
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    # Loop over K dimension (hidden_dim)
    for k0 in range(0, H, BLOCK_K):
        k = k0 + tl.arange(0, BLOCK_K)
        k_mask = k < H
        # Load A[m, k] vector
        a = tl.load(A_ptr + m * A_stride0 + k * A_stride1, mask=k_mask, other=0.0)
        # Load B[k, :] block, shape [BLOCK_K, BLOCK_N]
        b_block = tl.load(
            B_ptr + k[:, None] * B_stride0 + tl.arange(0, BLOCK_N)[None, :] * B_stride1,
            mask=k_mask[:, None], other=0.0
        )
        # Accumulate: sum over K of a[:, None] * b_block
        acc += tl.sum(a[:, None] * b_block, axis=0)
    # Store acc to C[m, :]
    tl.store(C_ptr + m * C_stride0 + tl.arange(0, BLOCK_N) * C_stride1, acc, mask=tl.arange(0, BLOCK_N) < H)


@triton.jit
def split_seqs_kernel(
    C_ptr,            # *const T, [S, H]
    out_enc_ptr,      # *T,       [B, T, H]
    out_hid_ptr,      # *T,       [B, I, H]
    B: tl.constexpr, T: tl.constexpr, I: tl.constexpr, H: tl.constexpr, S: tl.constexpr,
    C_stride0, C_stride1, C_stride2,
    out_enc_stride0, out_enc_stride1, out_enc_stride2,
    out_hid_stride0, out_hid_stride1, out_hid_stride2,
    BLOCK_H: tl.constexpr,
):
    # Each program handles one batch b
    b = tl.program_id(axis=0)
    # We will iterate over H in blocks to handle general cases
    offs_n = tl.arange(0, BLOCK_H)
    # Copy encoder part: rows 0..T-1
    for m in range(0, T):
        c_ptr = C_ptr + b * C_stride0 + m * C_stride1
        out_e_ptr = out_enc_ptr + b * out_enc_stride0 + m * out_enc_stride1
        for n0 in range(0, H, BLOCK_H):
            n_idx = n0 + offs_n
            mask = n_idx < H
            vals = tl.load(c_ptr + n_idx * C_stride2, mask=mask, other=0.0)
            tl.store(out_e_ptr + n_idx * out_enc_stride2, vals, mask=mask)
    # Copy hidden part: rows T..T+I-1
    for m in range(0, I):
        m_i = T + m
        c_ptr = C_ptr + b * C_stride0 + m_i * C_stride1
        out_h_ptr = out_hid_ptr + b * out_hid_stride0 + m * out_hid_stride1
        for n0 in range(0, H, BLOCK_H):
            n_idx = n0 + offs_n
            mask = n_idx < H
            vals = tl.load(c_ptr + n_idx * C_stride2, mask=mask, other=0.0)
            tl.store(out_h_ptr + n_idx * out_hid_stride2, vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, encoder_hidden_states: torch.Tensor, hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Ensure CUDA and contiguous
        assert encoder_hidden_states.is_cuda and hidden_states.is_cuda and process_weight.is_cuda, "Inputs must be CUDA tensors"
        B, T, H = encoder_hidden_states.shape
        B2, I, H2 = hidden_states.shape
        assert B == B2 and H == H2, "encoder_hidden_states and hidden_states must have same batch and hidden_dim"
        assert process_weight.shape == (H, H), "process_weight must be [hidden_dim, hidden_dim]"
        # Make contiguous
        enc = encoder_hidden_states.contiguous()
        hid = hidden_states.contiguous()
        Bw = process_weight.contiguous()

        # 1) Concatenate along sequence dimension: [B, S, H], S = T + I
        S = T + I
        concatenated = torch.empty((B, S, H), device=enc.device, dtype=enc.dtype)

        # Launch concat kernel: one program per element m in [B*S]
        grid_concat = (B * S,)
        concat_seqs_kernel[grid_concat](
            enc, hid, concatenated,
            B, T, I, H,
            enc.stride(0), enc.stride(1), enc.stride(2),
            hid.stride(0), hid.stride(1), hid.stride(2),
            concatenated.stride(0), concatenated.stride(1), concatenated.stride(2),
            BLOCK_H=64,
            num_warps=4, num_stages=2,
        )

        # 2) Matmul: processed = concatenated @ process_weight.T
        # A: [S, H] (rows of concatenated), B: [H, H] (process_weight.T)
        A = concatenated  # using concatenated as A directly
        Bw_T = Bw.transpose(0, 1)  # [H, H]
        C = torch.empty((S, H), device=enc.device, dtype=enc.dtype)

        grid_matmul = (S,)
        matmul_seqs_kernel[grid_matmul](
            A, Bw_T, C,
            S, H,
            A.stride(0), A.stride(1),
            Bw_T.stride(0), Bw_T.stride(1),
            C.stride(0), C.stride(1),
            BLOCK_K=64, BLOCK_N=H,
            num_warps=4, num_stages=2,
        )

        # 3) Split C into encoder and hidden parts
        processed_encoder = torch.empty((B, T, H), device=enc.device, dtype=enc.dtype)
        processed_hidden = torch.empty((B, I, H), device=enc.device, dtype=enc.dtype)

        grid_split = (B,)
        split_seqs_kernel[grid_split](
            C, processed_encoder, processed_hidden,
            B, T, I, H, S,
            C.stride(0), C.stride(1), C.stride(2),
            processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
            processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
            BLOCK_H=64,
            num_warps=4, num_stages=2,
        )

        return processed_encoder, processed_hidden