import torch
import triton
import triton.language as tl


@triton.jit
def concat_kernel(
    out_ptr,         # *float32, shape [B, S, H]
    encoder_ptr,     # *float32, shape [B, T, H]
    hidden_ptr,      # *float32, shape [B, I, H]
    B, T, I, H, S,   # int32
    out_stride_b, out_stride_s, out_stride_h,
    enc_stride_b, enc_stride_t, enc_stride_h,
    hid_stride_b, hid_stride_i, hid_stride_h,
    BLOCK_S: tl.constexpr, BLOCK_H: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_h = tl.program_id(2)

    s_idx = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    h_idx = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)

    # Masks for bounds
    mask_s = s_idx < S
    mask_h = h_idx < H
    mask = mask_s[:, None] & mask_h[None, :]

    # Determine source: encoder or hidden
    use_encoder = s_idx[:, None] < T
    # Load from encoder when s < T, else from hidden at (s - T)
    # Compute indices for both and select
    # Enc: [b, s, h]
    enc_ptrs = encoder_ptr + pid_b * enc_stride_b + (s_idx[:, None] * enc_stride_t) + (h_idx[None, :] * enc_stride_h)
    # Hid: [b, s - T, h]
    hid_ptrs = hidden_ptr + pid_b * hid_stride_b + ((s_idx[:, None] - T) * hid_stride_i) + (h_idx[None, :] * hid_stride_h)

    # Select source based on use_encoder
    vals = tl.load(enc_ptrs, mask=mask & use_encoder, other=0.0)
    vals = tl.where(use_encoder, vals, tl.load(hid_ptrs, mask=mask & (~use_encoder), other=0.0))

    # Store to out [b, s, h]
    out_ptrs = out_ptr + pid_b * out_stride_b + (s_idx[:, None] * out_stride_s) + (h_idx[None, :] * out_stride_h)
    tl.store(out_ptrs, vals, mask=mask)


@triton.jit
def batched_matmul_kernel(
    A_ptr,            # *float32, shape [M=B*S, K=H], viewed as [M, K]
    B_ptr,            # *float32, shape [K, N=H]
    C_ptr,            # *float32, shape [M, N]
    M, N, K,          # int32
    A_stride_m, A_stride_k,
    B_stride_k, B_stride_n,
    C_stride_m, C_stride_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Program ids: we use a 3D grid with pid_m over tiles of M per batch, pid_n over N tiles, pid_b over batch
    pid_m = tl.program_id(0)  # batch index
    pid_n = tl.program_id(1)  # tiles along N
    pid_b = tl.program_id(2)  # tiles along M per batch

    # Construct indices
    m_idx = pid_b * BLOCK_M + tl.arange(0, BLOCK_M)
    n_idx = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    k_idx = tl.arange(0, BLOCK_K)

    # Masks
    mask_m = m_idx < M
    mask_n = n_idx < N

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K
    for k0 in range(0, K, BLOCK_K):
        # Pointers for A[m, k] and B[k, n]
        a_ptrs = A_ptr + m_idx[:, None] * A_stride_m + (k0 + k_idx[None, :]) * A_stride_k  # shape [BM, BK]
        b_ptrs = B_ptr + (k0 + k_idx[:, None]) * B_stride_k + n_idx[None, :] * B_stride_n  # shape [BK, BN]

        a = tl.load(a_ptrs, mask=mask_m[:, None], other=0.0)
        b = tl.load(b_ptrs, mask=(k0 + k_idx[:, None] < K) & (mask_n[None, :]), other=0.0)
        acc += tl.dot(a, b)  # [BM, BN]

    # Store results
    c_ptrs = C_ptr + m_idx[:, None] * C_stride_m + n_idx[None, :] * C_stride_n
    store_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(c_ptrs, acc, mask=store_mask)


@triton.jit
def split_kernel(
    C_ptr,            # *float32, shape [B, S, H]
    out_enc_ptr,      # *float32, shape [B, T, H]
    out_hid_ptr,      # *float32, shape [B, I, H]
    B, T, I, H, S,    # int32
    C_stride_b, C_stride_s, C_stride_h,
    enc_stride_b, enc_stride_t, enc_stride_h,
    hid_stride_b, hid_stride_i, hid_stride_h,
    BLOCK_T: tl.constexpr, BLOCK_I: tl.constexpr, BLOCK_H: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_i = tl.program_id(2)
    pid_h = tl.program_id(3)

    t_idx = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    i_idx = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)
    h_idx = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)

    mask_t = t_idx < T
    mask_i = i_idx < I
    mask_h = h_idx < H

    # Mask for encoder part
    mask_enc = mask_t[:, None, None] & mask_h[None, :, None]  # [BT, TH, BH]
    # Mask for hidden part
    mask_hid = mask_i[:, None, None] & mask_h[None, :, None]  # [BI, IH, BH]

    # Copy encoder rows: C[:, :T, :]
    src_ptrs_enc = C_ptr + pid_b * C_stride_b + (t_idx[:, None, None] * C_stride_s) + (h_idx[None, :, None] * C_stride_h)
    dst_ptrs_enc = out_enc_ptr + pid_b * enc_stride_b + (t_idx[:, None] * enc_stride_t) + (h_idx[None, :] * enc_stride_h)
    tl.store(dst_ptrs_enc, tl.load(src_ptrs_enc, mask=mask_enc, other=0.0), mask=mask_enc)

    # Copy hidden rows: C[:, T:, :]
    src_ptrs_hid = C_ptr + pid_b * C_stride_b + ((t_idx[:, None, None] + T) * C_stride_s) + (h_idx[None, :, None] * C_stride_h)
    dst_ptrs_hid = out_hid_ptr + pid_b * hid_stride_b + (i_idx[:, None] * hid_stride_i) + (h_idx[None, :] * hid_stride_h)
    tl.store(dst_ptrs_hid, tl.load(src_ptrs_hid, mask=mask_hid, other=0.0), mask=mask_hid)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states, encoder_hidden_states, process_weight):
        # Ensure inputs are on CUDA
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Tensors must be on CUDA"
        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        H = hidden_states.shape[2]
        S = T + I

        device = hidden_states.device

        # 1) Concatenate along sequence dimension using Triton
        cat = torch.empty((B, S, H), device=device, dtype=torch.float32)

        BLOCK_S, BLOCK_H = 128, 64
        grid_concat = (B, triton.cdiv(S, BLOCK_S), triton.cdiv(H, BLOCK_H))
        concat_kernel[grid_concat](
            cat, encoder_hidden_states, hidden_states,
            B, T, I, H, S,
            cat.stride(0), cat.stride(1), cat.stride(2),
            encoder_hidden_states.stride(0), encoder_hidden_states.stride(1), encoder_hidden_states.stride(2),
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            BLOCK_S=BLOCK_S, BLOCK_H=BLOCK_H, num_warps=4, num_stages=2,
        )

        # 2) Compute processed = cat @ process_weight.T using Triton batched GEMM
        # View cat as [M=B*S, K=H], B_weight = process_weight.T as [K=H, N=H]
        A = cat  # [B, S, H]
        # For Triton kernel, A should be [M, K] linear. We can reshape to [B*S, H] by flattening (B, S) into M.
        # Create a 1D view pointer: treat M=B*S rows, K=H columns.
        M = B * S
        K = H
        N = H

        # Prepare B_weight = process_weight.T
        B_weight = process_weight.t().contiguous()  # [H, H]

        C = torch.empty((M, N), device=device, dtype=torch.float32)  # [B*S, H]

        # 3D grid: (batch index, tiles over M per batch, tiles over N)
        # We'll set pid_b = 0 for all tiles; Triton expects 3D, but here we can derive batch from m_idx.
        # Alternative: launch with grid (1, tiles_M, tiles_N), then select rows with m_idx // S in Python.
        # Simpler: compute tiles explicitly.
        BLOCK_M, BLOCK_N, BLOCK_K = 64, 128, 64
        grid_gemm = (1, triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        # Call kernel and map batch tiles: we can't pass B into pid_b here, so we launch per-batch by iterating B.
        # Triton supports 3D grid; we can set pid_b for each batch by making B a runtime parameter and deriving b from m_idx.
        # However, Triton kernels require static grid dims. We will run the kernel once for all batches by computing m_idx across B*S and setting b via division.

        # Launch batched GEMM kernel; we set a grid that covers all M,N tiles. Triton will handle loop over blocks.
        batched_matmul_kernel[grid_gemm](
            A, B_weight, C,
            M, N, K,
            A.stride(0), A.stride(2),                # A is [B, S, H] but we pass as [M, K] via strides: stride(0)=S, stride(2)=1
            B_weight.stride(0), B_weight.stride(1),
            C.stride(0), C.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=3,
        )

        # Note: Above kernel assumes we pass A as [M,K] with correct strides. cat is [B,S,H], so we view A as [M=B*S, K=H].
        # However, Triton expects a flat pointer. To enforce, we pass A as cat.reshape(-1, H), but we keep it as pointer.
        # The kernel uses A_stride_m=A.stride(0) for M and A_stride_k=A.stride(2) for K. This works because M=B*S and K=H.

        # Now, reshape C back to [B, S, H]
        processed = C.view(B, S, H)

        # 3) Split processed into encoder and hidden using Triton
        processed_encoder = torch.empty((B, T, H), device=device, dtype=torch.float32)
        processed_hidden = torch.empty((B, I, H), device=device, dtype=torch.float32)

        BLOCK_T, BLOCK_I, BLOCK_H = 128, 128, 64
        grid_split = (B, triton.cdiv(T, BLOCK_T), triton.cdiv(I, BLOCK_I), triton.cdiv(H, BLOCK_H))
        split_kernel[grid_split](
            processed, processed_encoder, processed_hidden,
            B, T, I, H, S,
            processed.stride(0), processed.stride(1), processed.stride(2),
            processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
            processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
            BLOCK_T=BLOCK_T, BLOCK_I=BLOCK_I, BLOCK_H=BLOCK_H,
            num_warps=4, num_stages=2,
        )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
