import torch
import triton
import triton.language as tl


@triton.jit
def concat_seqs_kernel(
    encoder_ptr, hidden_ptr, out_ptr,
    B, T, I, H,
    enc_stride0, enc_stride1, enc_stride2,
    hid_stride0, hid_stride1, hid_stride2,
    out_stride0, out_stride1, out_stride2,
    BLOCK_T: tl.constexpr, BLOCK_I: tl.constexpr
):
    # For each batch b, copy rows 0..T-1 from encoder into out[b, 0..T-1, :]
    # and rows 0..I-1 from hidden into out[b, T..T+I-1, :].
    b = 0  # We'll iterate b on host side; this kernel is 1D over batch
    # Loop over batch
    for b in range(0, B):
        # Copy encoder rows
        for t in range(0, T, BLOCK_T):
            t_offsets = t + tl.arange(0, BLOCK_T)
            mask_t = t_offsets < T
            for h_off in range(0, H, BLOCK_I):
                h_offsets = h_off + tl.arange(0, BLOCK_I)
                mask_h = h_offsets < H
                ptr = encoder_ptr + b * enc_stride0 + t_offsets[:, None] * enc_stride1 + h_offsets[None, :] * enc_stride2
                # Load with masks
                vals = tl.load(ptr, mask=mask_t[:, None] & mask_h[None, :], other=0.0)
                out_ptrs = out_ptr + b * out_stride0 + (t_offsets[:, None]) * out_stride1 + (h_offsets[None, :]) * out_stride2
                tl.store(out_ptrs, vals, mask=mask_t[:, None] & mask_h[None, :])

        # Copy hidden rows starting at T
        for i in range(0, I, BLOCK_I):
            i_offsets = i + tl.arange(0, BLOCK_I)
            mask_i = i_offsets < I
            for h_off in range(0, H, BLOCK_I):
                h_offsets = h_off + tl.arange(0, BLOCK_I)
                mask_h = h_offsets < H
                ptr = hidden_ptr + b * hid_stride0 + i_offsets[:, None] * hid_stride1 + h_offsets[None, :] * hid_stride2
                vals = tl.load(ptr, mask=mask_i[:, None] & mask_h[None, :], other=0.0)
                out_ptrs = out_ptr + b * out_stride0 + (T + i_offsets[:, None]) * out_stride1 + (h_offsets[None, :]) * out_stride2
                tl.store(out_ptrs, vals, mask=mask_i[:, None] & mask_h[None, :])


@triton.jit
def matmul_seqs_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    A_stride0, A_stride1, A_stride2,   # A is [M, K] interpreted as [B*S, H] via strides (S=sequence, H=hidden)
    B_stride0, B_stride1,              # B is [K, N] = [H, H] via process_weight.T
    C_stride0, C_stride1, C_stride2,   # C is [M, N] = [B*S, H]
    BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr
):
    # Each program computes one row of C (m in 0..M-1) and one tile over N (n in 0..N-1 with BLOCK_N)
    m = tl.program_id(0)
    n_block = tl.program_id(1)
    n_offsets = n_block * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = n_offsets < N

    acc = tl.zeros([BLOCK_N], dtype=tl.float32)

    # Loop over K in chunks
    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < K

        # Load A[m, k_offsets] as a vector (shape [BLOCK_K])
        A_row_ptr = A_ptr + m * A_stride0 + k_offsets * A_stride1  # A_stride2 corresponds to hidden dim stride, not used here
        a = tl.load(A_row_ptr, mask=mask_k, other=0.0)  # [BLOCK_K]

        # Load B[k_offsets, n_offsets] as a matrix (shape [BLOCK_K, BLOCK_N])
        B_ptr_mat = B_ptr + k_offsets[:, None] * B_stride0 + n_offsets[None, :] * B_stride1
        b = tl.load(B_ptr_mat, mask=mask_k[:, None] & mask_n[None, :], other=0.0)

        # Accumulate
        acc += tl.sum(a[:, None] * b, axis=0)

    # Store result to C[m, n_offsets]
    C_row_ptr = C_ptr + m * C_stride0 + n_offsets * C_stride1
    tl.store(C_row_ptr, acc, mask=mask_n)


@triton.jit
def split_encoder_kernel(
    C_ptr, out_ptr,
    B, T, H,
    C_stride0, C_stride1, C_stride2,
    out_stride0, out_stride1, out_stride2,
    BLOCK_T: tl.constexpr, BLOCK_H: tl.constexpr
):
    # Copy C[:, :T, :] into out [B, T, H]
    for b in range(0, B):
        for t in range(0, T, BLOCK_T):
            t_offsets = t + tl.arange(0, BLOCK_T)
            mask_t = t_offsets < T
            for h in range(0, H, BLOCK_H):
                h_offsets = h + tl.arange(0, BLOCK_H)
                mask_h = h_offsets < H
                ptr = C_ptr + b * C_stride0 + t_offsets[:, None] * C_stride1 + h_offsets[None, :] * C_stride2
                vals = tl.load(ptr, mask=mask_t[:, None] & mask_h[None, :], other=0.0)
                out_ptrs = out_ptr + b * out_stride0 + t_offsets[:, None] * out_stride1 + h_offsets[None, :] * out_stride2
                tl.store(out_ptrs, vals, mask=mask_t[:, None] & mask_h[None, :])


@triton.jit
def split_hidden_kernel(
    C_ptr, out_ptr,
    B, I, H,
    C_stride0, C_stride1, C_stride2,
    out_stride0, out_stride1, out_stride2,
    BLOCK_I: tl.constexpr, BLOCK_H: tl.constexpr
):
    # Copy C[:, T:, :] into out [B, I, H]
    # Note: S = T + I, start_row = S
    S = T + I
    for b in range(0, B):
        for i in range(0, I, BLOCK_I):
            i_offsets = i + tl.arange(0, BLOCK_I)
            mask_i = i_offsets < I
            for h in range(0, H, BLOCK_H):
                h_offsets = h + tl.arange(0, BLOCK_H)
                mask_h = h_offsets < H
                ptr = C_ptr + b * C_stride0 + (T + i_offsets[:, None]) * C_stride1 + h_offsets[None, :] * C_stride2
                vals = tl.load(ptr, mask=mask_i[:, None] & mask_h[None, :], other=0.0)
                out_ptrs = out_ptr + b * out_stride0 + i_offsets[:, None] * out_stride1 + h_offsets[None, :] * out_stride2
                tl.store(out_ptrs, vals, mask=mask_i[:, None] & mask_h[None, :])


class ModelNew(torch.nn.Module):
    def forward(self, encoder_hidden_states: torch.Tensor, hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Ensure inputs are CUDA and contiguous
        assert encoder_hidden_states.is_cuda and hidden_states.is_cuda and process_weight.is_cuda, "All inputs must be on CUDA"
        assert encoder_hidden_states.dtype == torch.float32 and hidden_states.dtype == torch.float32 and process_weight.dtype == torch.float32, "Use float32"
        enc = encoder_hidden_states.contiguous()
        hid = hidden_states.contiguous()
        Wt = process_weight.t().contiguous()  # process_weight.T [H, H]

        B, T, H = enc.shape
        I = hid.shape[1]
        S = T + I

        # 1) Concatenate into [B, S, H] using Triton
        concatenated = torch.empty((B, S, H), device=enc.device, dtype=enc.dtype)
        grid_concat = (B,)
        concat_seqs_kernel[grid_concat](
            enc, hid, concatenated,
            B, T, I, H,
            enc.stride(0), enc.stride(1), enc.stride(2),
            hid.stride(0), hid.stride(1), hid.stride(2),
            concatenated.stride(0), concatenated.stride(1), concatenated.stride(2),
            BLOCK_T=64, BLOCK_I=64
        )

        # 2) Compute C_full = concatenated @ Wt using Triton matmul
        # Treat concatenated as [M, K] where M = B*S, K = H
        M = B * S
        # Output C_full is [M, H]
        C_full = torch.empty((M, H), device=enc.device, dtype=enc.dtype)

        grid_matmul = (M, 1)  # single tile over N since BLOCK_N=H (1024)
        matmul_seqs_kernel[grid_matmul](
            concatenated, Wt, C_full,
            M, H, H,  # N=H
            concatenated.stride(0), concatenated.stride(1), concatenated.stride(2),  # A strides: row stride (S), col stride (1)
            Wt.stride(0), Wt.stride(1),                        # B strides: [H, H]
            C_full.stride(0), C_full.stride(1), C_full.stride(2),
            BLOCK_K=128, BLOCK_N=H  # BLOCK_N=1024 for given workloads
        )

        # 3) Split C_full into encoder and hidden parts using Triton
        processed_encoder = torch.empty((B, T, H), device=enc.device, dtype=enc.dtype)
        processed_hidden = torch.empty((B, I, H), device=enc.device, dtype=enc.dtype)

        # Each row of C_full corresponds to concatenated[b, s, :] for s in 0..S-1
        # We'll recompute indices using B to split. Use Triton split kernels.
        grid_split_e = (B,)
        split_encoder_kernel[grid_split_e](
            C_full, processed_encoder,
            B, T, H,
            C_full.stride(0), C_full.stride(1), C_full.stride(2),
            processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
            BLOCK_T=64, BLOCK_H=128
        )

        grid_split_h = (B,)
        split_hidden_kernel[grid_split_h](
            C_full, processed_hidden,
            B, I, H,
            C_full.stride(0), C_full.stride(1), C_full.stride(2),
            processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
            BLOCK_I=64, BLOCK_H=128
        )

        return processed_encoder, processed_hidden