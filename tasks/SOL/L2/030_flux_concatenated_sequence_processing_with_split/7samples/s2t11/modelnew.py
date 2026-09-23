import torch
import triton
import triton.language as tl

@triton.jit
def concat_seq_kernel(
    A_e, A_i, A_out,
    B, T, I, H,
    A_e_stride0, A_e_stride1, A_e_stride2,
    A_i_stride0, A_i_stride1, A_i_stride2,
    A_out_stride0, A_out_stride1, A_out_stride2,
    BLOCK_S: tl.constexpr, BLOCK_H: tl.constexpr,
):
    # program ids
    b = tl.program_id(0)         # batch index
    pid_s = tl.program_id(1)     # tile over sequence (S = T + I)
    pid_h = tl.program_id(2)     # tile over hidden dim (H)

    # offsets within tiles
    s_offsets = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)  # [BLOCK_S]
    h_offsets = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)  # [BLOCK_H]

    # masks for bounds
    mask_s = s_offsets < (T + I)
    mask_h = h_offsets < H
    mask = mask_s[:, None] & mask_h[None, :]

    # For each sequence position s, decide source: s < T -> encoder, else -> image.
    # Triton supports scalar-controlled loops and masked loads/stores.
    for si in range(BLOCK_S):
        s = s_offsets[si]
        valid_s = s < (T + I)
        if valid_s:
            # which source
            is_encoder = s < T
            src_h = h_offsets[None, :]  # [1, BLOCK_H]

            # base pointers for this batch
            A_out_ptr = A_out + b * A_out_stride0
            # compute input pointers
            if is_encoder:
                A_src_ptr = A_e + b * A_e_stride0 + s * A_e_stride1
            else:
                A_src_ptr = A_i + b * A_i_stride0 + (s - T) * A_i_stride1

            # load from source, masked along H dimension
            vals = tl.load(A_src_ptr + src_h * A_e_stride2, mask=mask_s[si], other=0.0)  # [BLOCK_H]
            # store to output at (b, s, h)
            tl.store(A_out_ptr + s * A_out_stride1 + h_offsets * A_out_stride2, vals, mask=mask_h)

@triton.jit
def batched_matmul_kernel(
    A, B, C,
    M, K, N,
    A_stride0, A_stride1,
    B_stride0, B_stride1,
    C_stride0, C_stride1, C_stride2,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # program ids
    b = tl.program_id(0)          # batch
    pid_m = tl.program_id(1)      # tiles over M (rows of A)
    pid_n = tl.program_id(2)      # tiles over N (cols of C)

    # tile offsets
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]

    # masks for bounds
    mask_m = m_offsets < M
    mask_n = n_offsets < N

    # accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # loop over K
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)  # [BLOCK_K]
        mask_k = k_offsets < K

        # A block: shape [BLOCK_M, BLOCK_K]
        a_ptrs = A + b * A_stride0 + m_offsets[:, None] * A_stride1 + k_offsets[None, :] * A_stride1
        a_mask = mask_m[:, None] & mask_k[None, :]
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # B block: shape [BLOCK_K, BLOCK_N]
        b_ptrs = B + k_offsets[:, None] * B_stride0 + n_offsets[None, :] * B_stride1
        b_mask = mask_k[:, None] & mask_n[None, :]
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # accumulate
        acc += tl.dot(a, b)

    # store results
    c_ptrs = C + b * C_stride0 + m_offsets[:, None] * C_stride1 + n_offsets[None, :] * C_stride2
    c_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(c_ptrs, acc, mask=c_mask)

@triton.jit
def split_kernel(
    C, out_e, out_i,
    B, T, I, H,
    C_stride0, C_stride1, C_stride2,
    out_e_stride0, out_e_stride1, out_e_stride2,
    out_i_stride0, out_i_stride1, out_i_stride2,
    BLOCK_T: tl.constexpr, BLOCK_I: tl.constexpr, BLOCK_H: tl.constexpr,
):
    b = tl.program_id(0)
    pid_t = tl.program_id(1)  # tiles over T
    pid_i = tl.program_id(2)  # tiles over I
    pid_h = tl.program_id(3)  # tiles over H

    t_offsets = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)  # [BLOCK_T]
    i_offsets = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)  # [BLOCK_I]
    h_offsets = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)  # [BLOCK_H]

    mask_t = t_offsets < T
    mask_i = i_offsets < I
    mask_h = h_offsets < H

    # For each (t, i, h), copy C[b, t, h] to out_e and C[b, T+i, h] to out_i
    for ti in range(BLOCK_T):
        t = t_offsets[ti]
        if t < T:
            # source row index for encoder part
            src_row = t
            # load C[b, src_row, h] across h tile
            c_src_ptr = C + b * C_stride0 + src_row * C_stride1
            vals_e = tl.load(c_src_ptr + h_offsets * C_stride2, mask=mask_h, other=0.0)
            # store to out_e[b, t, h]
            out_e_ptr = out_e + b * out_e_stride0 + t * out_e_stride1
            tl.store(out_e_ptr + h_offsets * out_e_stride2, vals_e, mask=mask_h)

    for ii in range(BLOCK_I):
        i = i_offsets[ii]
        if i < I:
            # source row index for image part
            src_row = T + i
            c_src_ptr = C + b * C_stride0 + src_row * C_stride1
            vals_i = tl.load(c_src_ptr + h_offsets * C_stride2, mask=mask_h, other=0.0)
            out_i_ptr = out_i + b * out_i_stride0 + i * out_i_stride2  # out_i has H dim as last, batch first
            tl.store(out_i_ptr + h_offsets * out_i_stride2, vals_i, mask=mask_h)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        # Ensure CUDA and dtype float32 for numerical consistency with reference
        device = hidden_states.device
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "All tensors must be on CUDA"
        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        H = hidden_states.shape[2]
        assert encoder_hidden_states.shape == (B, T, H), "encoder_hidden_states must be [batch, text_seq_len, hidden_dim]"
        assert hidden_states.shape == (B, I, H), "hidden_states must be [batch, img_seq_len, hidden_dim]"
        assert process_weight.shape == (H, H), "process_weight must be [hidden_dim, hidden_dim]"

        # Make inputs contiguous
        A_e = encoder_hidden_states.contiguous()
        A_i = hidden_states.contiguous()
        B_weight = process_weight.t().contiguous()  # [H, H]
        dtype = torch.float32
        B_ei = B  # batch size

        # 1) Concatenate along sequence dimension: A_cat [B, S=T+I, H]
        A_cat = torch.empty((B, T + I, H), device=device, dtype=dtype)
        BLOCK_S, BLOCK_H = 128, 64
        grid_concat = (B, triton.cdiv(T + I, BLOCK_S), triton.cdiv(H, BLOCK_H))
        concat_seq_kernel[grid_concat](
            A_e, A_i, A_cat,
            B_ei, T, I, H,
            A_e.stride(0), A_e.stride(1), A_e.stride(2),
            A_i.stride(0), A_i.stride(1), A_i.stride(2),
            A_cat.stride(0), A_cat.stride(1), A_cat.stride(2),
            BLOCK_S=BLOCK_S, BLOCK_H=BLOCK_H,
            num_warps=4, num_stages=2,
        )

        # 2) GEMM: C = A_cat @ B_weight, where A_cat is viewed as [M=B*(T+I), K=H], B is [K=H, N=H]
        M = B * (T + I)
        K = H
        N = H
        C = torch.empty((B, T + I, H), device=device, dtype=dtype)

        # Reshape A_cat to [M, K] logically by computing pointers per (b, s) row; Triton kernel uses A_cat as [B,S,H] but
        # the pointer arithmetic for A considers b via batch stride and s via S stride. We don't need to materialize reshape.
        # Instead, we pass A_cat as is and let the kernel read rows using strides. The kernel expects A to be [B,S,H].
        grid_gemm = (B, triton.cdiv(M, 64), triton.cdiv(N, 128))
        # Note: M and N here are per-batch sequence length and hidden dim; grid uses tiles over M (S) and N (H).
        batched_matmul_kernel[grid_gemm](
            A_cat, B_weight, C,
            M, K, N,
            A_cat.stride(0), A_cat.stride(1),
            B_weight.stride(0), B_weight.stride(1),
            C.stride(0), C.stride(1), C.stride(2),
            BLOCK_M=64, BLOCK_N=128, BLOCK_K=64,
            num_warps=4, num_stages=3,
        )

        # 3) Split C into encoder and hidden parts
        processed_encoder = torch.empty((B, T, H), device=device, dtype=dtype)
        processed_hidden = torch.empty((B, I, H), device=device, dtype=dtype)

        BLOCK_T, BLOCK_I, BLOCK_H_split = 128, 128, 64
        grid_split = (B, triton.cdiv(T, BLOCK_T), triton.cdiv(I, BLOCK_I), triton.cdiv(H, BLOCK_H_split))
        split_kernel[grid_split](
            C, processed_encoder, processed_hidden,
            B, T, I, H,
            C.stride(0), C.stride(1), C.stride(2),
            processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
            processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
            BLOCK_T=BLOCK_T, BLOCK_I=BLOCK_I, BLOCK_H=BLOCK_H_split,
            num_warps=4, num_stages=2,
        )

        return processed_encoder, processed_hidden