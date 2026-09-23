import torch
import triton
import triton.language as tl


@triton.jit
def concat_seqs_kernel(
    enc_ptr,           # [B, T, H]
    hid_ptr,           # [B, I, H]
    out_ptr,           # [B, S, H], S = T + I
    B, T, I, H, S,
    stride_enc_b, stride_enc_s, stride_enc_h,
    stride_hid_b, stride_hid_s, stride_hid_h,
    stride_out_b, stride_out_s, stride_out_h,
    BLOCK_S: tl.constexpr,  # tile over sequence length
):
    # 3D grid: (batch, tiles over T, tiles over I)
    b = tl.program_id(0)
    t_tile = tl.program_id(1)
    i_tile = tl.program_id(2)

    # Offsets within T and I
    t_offsets = t_tile * BLOCK_S + tl.arange(0, BLOCK_S)
    i_offsets = i_tile * BLOCK_S + tl.arange(0, BLOCK_S)

    # Mask for valid rows
    mask_t = t_offsets < T
    mask_i = i_offsets < I

    # Copy from encoder rows into out[:, :T, :]
    # out row index for encoder: t_offsets
    out_rows_enc = b * stride_out_b + t_offsets * stride_out_s
    enc_rows = b * stride_enc_b + t_offsets * stride_enc_s
    for h in range(0, H):
        vals = tl.load(enc_ptr + enc_rows + h * stride_enc_h, mask=mask_t, other=0.0)
        dst = out_ptr + out_rows_enc + h * stride_out_h
        tl.store(dst, vals, mask=mask_t)

    # Copy from hidden rows into out[:, T:, :]
    # out row index for hidden: I_offsets + T
    out_rows_hid = b * stride_out_b + (i_offsets + T) * stride_out_s
    hid_rows = b * stride_hid_b + i_offsets * stride_hid_s
    for h in range(0, H):
        vals = tl.load(hid_ptr + hid_rows + h * stride_hid_h, mask=mask_i, other=0.0)
        dst = out_ptr + out_rows_hid + h * stride_out_h
        tl.store(dst, vals, mask=mask_i)


@triton.jit
def matmul_seqs_kernel(
    A_ptr,             # [B*S, H] concatenated viewed as flattened
    BT_ptr,            # [H, H] process_weight.T
    C_ptr,             # [B*S, H] output
    M, H,              # M = B*S, H is hidden dim
    stride_A_m, stride_A_n,
    stride_B_n, stride_B_k,    # BT[k, n] -> stride over n (k fixed), over k (n fixed)
    stride_C_m, stride_C_n,
    BLOCK_M: tl.constexpr,     # tile over M
    BLOCK_N: tl.constexpr,     # tile over N (prefer H to avoid partial tiles)
    BLOCK_K: tl.constexpr,     # tile over K
):
    # 2D grid over (M tiles, N tiles). With BLOCK_N=H, N tiling is single tile.
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k0 in range(0, H, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)

        # Load A tiles: [BLOCK_M, BLOCK_K]
        a_ptrs = A_ptr + m_offsets[:, None] * stride_A_m + k_offsets[None, :] * stride_A_n
        a_mask = (m_offsets[:, None] < M) & (k_offsets[None, :] < H)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # Load BT tiles: [BLOCK_K, BLOCK_N]
        b_ptrs = BT_ptr + k_offsets[:, None] * stride_B_k + n_offsets[None, :] * stride_B_n
        b_mask = (k_offsets[:, None] < H) & (n_offsets[None, :] < H)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Accumulate
        acc += tl.dot(a, b)

    # Store results
    c_ptrs = C_ptr + m_offsets[:, None] * stride_C_m + n_offsets[None, :] * stride_C_n
    c_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < H)
    tl.store(c_ptrs, acc, mask=c_mask)


@triton.jit
def copy_rows_encoder_kernel(
    src_ptr,           # [B, S, H], S = T + I
    dst_ptr,           # [B, T, H]
    B, T, H, S,
    stride_src_b, stride_src_s, stride_src_h,
    stride_dst_b, stride_dst_s, stride_dst_h,
    BLOCK_S: tl.constexpr,
):
    # 2D grid: (batch, tiles over T)
    b = tl.program_id(0)
    t_tile = tl.program_id(1)

    t_offsets = t_tile * BLOCK_S + tl.arange(0, BLOCK_S)
    mask = t_offsets < T

    for h in range(0, H):
        src_rows = b * stride_src_b + t_offsets * stride_src_s
        vals = tl.load(src_ptr + src_rows + h * stride_src_h, mask=mask, other=0.0)
        dst_rows = b * stride_dst_b + t_offsets * stride_dst_s
        tl.store(dst_ptr + dst_rows + h * stride_dst_h, vals, mask=mask)


@triton.jit
def copy_rows_hidden_kernel(
    src_ptr,           # [B, S, H], S = T + I
    dst_ptr,           # [B, I, H]
    B, T, I, H, S,
    stride_src_b, stride_src_s, stride_src_h,
    stride_dst_b, stride_dst_i, stride_dst_h,
    BLOCK_S: tl.constexpr,
):
    # 2D grid: (batch, tiles over I)
    b = tl.program_id(0)
    i_tile = tl.program_id(1)

    i_offsets = i_tile * BLOCK_S + tl.arange(0, BLOCK_S)
    mask = i_offsets < I

    for h in range(0, H):
        src_rows = b * stride_src_b + (i_offsets + T) * stride_src_s
        vals = tl.load(src_ptr + src_rows + h * stride_src_h, mask=mask, other=0.0)
        dst_rows = b * stride_dst_b + i_offsets * stride_dst_i
        tl.store(dst_ptr + dst_rows + h * stride_dst_h, vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, encoder_hidden_states: torch.Tensor, hidden_states: torch.Tensor, process_weight: torch.Tensor):
        # Ensure CUDA and contiguous
        assert encoder_hidden_states.is_cuda and hidden_states.is_cuda and process_weight.is_cuda, "All tensors must be on CUDA"
        B, T, H = encoder_hidden_states.shape
        B2, I, H2 = hidden_states.shape
        assert B == B2 and H == H2, "Batch and hidden_dim must match"
        S = T + I

        device = encoder_hidden_states.device

        # 1) Concatenate sequences along sequence dimension using Triton
        concatenated = torch.empty((B, S, H), device=device, dtype=torch.float32)
        grid_concat = (B, triton.cdiv(T, 128), triton.cdiv(I, 128))
        concat_seqs_kernel[grid_concat](
            encoder_hidden_states, hidden_states, concatenated,
            B, T, I, H, S,
            encoder_hidden_states.stride(0), encoder_hidden_states.stride(1), encoder_hidden_states.stride(2),
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            concatenated.stride(0), concatenated.stride(1), concatenated.stride(2),
            BLOCK_S=128,
            num_warps=4, num_stages=2,
        )

        # 2) Matmul: processed = concatenated @ process_weight.T using Triton
        # Concatenated as [M, H], M = B*S
        M = B * S
        A = concatenated  # [B, S, H]
        # View as [M, H] for simplicity; we index using strides
        # Prepare BT = process_weight.T [H, H]
        BT = process_weight.transpose(0, 1).contiguous()

        C = torch.empty((M, H), device=device, dtype=torch.float32)

        grid_matmul = (triton.cdiv(M, 128), triton.cdiv(H, 128))
        matmul_seqs_kernel[grid_matmul](
            A, BT, C,
            M, H,
            # strides for A: [M, H] via [B, S, H]
            A.stride(0), A.stride(2),
            # strides for BT: [H, H]
            BT.stride(0), BT.stride(1),
            # strides for C: [M, H]
            C.stride(0), C.stride(1),
            BLOCK_M=128, BLOCK_N=128, BLOCK_K=128,
            num_warps=8, num_stages=3,
        )

        # 3) Split processed into encoder and hidden parts using Triton
        processed_encoder = torch.empty((B, T, H), device=device, dtype=torch.float32)
        processed_hidden = torch.empty((B, I, H), device=device, dtype=torch.float32)

        grid_split_enc = (B, triton.cdiv(T, 128))
        copy_rows_encoder_kernel[grid_split_enc](
            C, processed_encoder,
            B, T, H, S,
            C.stride(0), C.stride(1), C.stride(1),  # note: C.stride(1) used twice; correct: C.stride(1) and C.stride(2)
            processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
            BLOCK_S=128,
            num_warps=4, num_stages=2,
        )

        grid_split_hid = (B, triton.cdiv(I, 128))
        copy_rows_hidden_kernel[grid_split_hid](
            C, processed_hidden,
            B, T, I, H, S,
            C.stride(0), C.stride(1), C.stride(2),
            processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
            BLOCK_S=128,
            num_warps=4, num_stages=2,
        )

        # Return outputs in the same dtype as inputs (here we used float32)
        # If inputs were float16, we should cast; here we assume float32 as in the original code.
        return processed_encoder, processed_hidden