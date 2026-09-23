import torch
import triton
import triton.language as tl

@triton.jit
def concat_seqs_kernel(
    enc_ptr, hid_ptr, concat_ptr,
    B, T, I, H,
    enc_stride_b, enc_stride_t, enc_stride_h,
    hid_stride_b, hid_stride_i, hid_stride_h,
    concat_stride_b, concat_stride_s, concat_stride_h,
    BLOCK_H: tl.constexpr,
):
    # Each program handles one batch b and one row within its segment
    b = tl.program_id(0)
    row_seg = tl.program_id(1)  # 0 for encoder, 1 for hidden
    h_offsets = tl.arange(0, BLOCK_H)
    mask = h_offsets < H

    if b >= 0 and b < B:
        if row_seg == 0:
            # Encode rows: b in [0, B), t in [0, T)
            t = tl.program_id(2)
            if t < T:
                src_ptr = enc_ptr + b * enc_stride_b + t * enc_stride_t + h_offsets * enc_stride_h
                dst_s = t  # position in concatenated seq
                dst_ptr = concat_ptr + b * concat_stride_b + dst_s * concat_stride_s + h_offsets * concat_stride_h
                tl.store(dst_ptr, tl.load(src_ptr, mask=mask, other=0.0))
        else:
            # Hidden rows: b in [0, B), i in [0, I)
            i = tl.program_id(2)
            if i < I:
                src_ptr = hid_ptr + b * hid_stride_b + i * hid_stride_i + h_offsets * hid_stride_h
                dst_s = T + i  # position in concatenated seq
                dst_ptr = concat_ptr + b * concat_stride_b + dst_s * concat_stride_s + h_offsets * concat_stride_h
                tl.store(dst_ptr, tl.load(src_ptr, mask=mask, other=0.0))


@triton.jit
def triton_matmul_seqs_kernel(
    A_ptr, BwT_ptr, C_ptr,
    M, N, K,
    A_stride_m, A_stride_k,
    BwT_stride_k, BwT_stride_n,
    C_stride_m, C_stride_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Each program handles a tile over (m, n): m in [pid_m*BLOCK_M, ...], n in [pid_n*BLOCK_N, ...]
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        # Load A tile: shape (BLOCK_M, BLOCK_K)
        a_ptrs = A_ptr + m_offsets[:, None] * A_stride_m + k_offsets[None, :] * A_stride_k
        a_mask = (m_offsets[:, None] < M) & (k_offsets[None, :] < K)
        A_tile = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # Load B^T tile: shape (BLOCK_K, BLOCK_N)
        bw_ptrs = BwT_ptr + k_offsets[:, None] * BwT_stride_k + n_offsets[None, :] * BwT_stride_n
        bw_mask = (k_offsets[:, None] < K) & (n_offsets[None, :] < N)
        Bw_tile = tl.load(bw_ptrs, mask=bw_mask, other=0.0)

        # Accumulate
        acc += tl.dot(A_tile, Bw_tile)

    # Store C tile
    c_ptrs = C_ptr + m_offsets[:, None] * C_stride_m + n_offsets[None, :] * C_stride_n
    c_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


@triton.jit
def split_seqs_kernel(
    C_ptr, enc_out_ptr, hid_out_ptr,
    B, T, I, H, S,
    C_stride_b, C_stride_s, C_stride_h,
    enc_stride_b, enc_stride_t, enc_stride_h,
    hid_stride_b, hid_stride_i, hid_stride_h,
    BLOCK_H: tl.constexpr,
):
    b = tl.program_id(0)
    h_offsets = tl.arange(0, BLOCK_H)
    if b < B:
        # Copy first T rows to encoder output
        for t in range(0, T):
            src_ptr = C_ptr + b * C_stride_b + t * C_stride_s + h_offsets * C_stride_h
            dst_ptr = enc_out_ptr + b * enc_stride_b + t * enc_stride_t + h_offsets * enc_stride_h
            tl.store(dst_ptr, tl.load(src_ptr, mask=h_offsets < H, other=0.0))
        # Copy remaining rows (I) to hidden output, starting at index T in C
        for i in range(0, I):
            src_ptr = C_ptr + b * C_stride_b + (T + i) * C_stride_s + h_offsets * C_stride_h
            dst_ptr = hid_out_ptr + b * hid_stride_b + i * hid_stride_i + h_offsets * hid_stride_h
            tl.store(dst_ptr, tl.load(src_ptr, mask=h_offsets < H, other=0.0))


class ModelNew(torch.nn.Module):
    def forward(self, encoder_hidden_states: torch.Tensor, hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Ensure CUDA and contiguous
        assert encoder_hidden_states.is_cuda and hidden_states.is_cuda and process_weight.is_cuda, "Tensors must be on CUDA"
        enc = encoder_hidden_states.contiguous()
        hid = hidden_states.contiguous()
        B, T, H = enc.shape
        I = hid.shape[1]
        S = T + I

        # Allocate concatenated [B, S, H]
        concatenated = torch.empty((B, S, H), device=enc.device, dtype=torch.float32)
        grid_concat = (B, 2, T) + (2, I)  # grid over batch, segment (0=encoder, 1=hidden), rows within each
        concat_seqs_kernel[grid_concat](
            enc, hid, concatenated,
            B, T, I, H,
            enc.stride(0), enc.stride(1), enc.stride(2),
            hid.stride(0), hid.stride(1), hid.stride(2),
            concatenated.stride(0), concatenated.stride(1), concatenated.stride(2),
            BLOCK_H=128,
            num_warps=4, num_stages=2,
        )

        # Allocate output C [B*S, H]
        C = torch.empty((B * S, H), device=enc.device, dtype=torch.float32)

        # Launch Triton GEMM over (rows, N tiles)
        grid_matmul = (triton.cdiv(B * S, 64), triton.cdiv(H, 128))
        triton_matmul_seqs_kernel[grid_matmul](
            concatenated, process_weight.t(), C,
            B * S, H, H,
            concatenated.stride(0), concatenated.stride(2),
            process_weight.t().stride(0), process_weight.t().stride(1),
            C.stride(0), C.stride(1),
            BLOCK_M=64, BLOCK_N=128, BLOCK_K=64,
            num_warps=4, num_stages=2,
        )

        # Split outputs
        processed_encoder = torch.empty((B, T, H), device=enc.device, dtype=torch.float32)
        processed_hidden = torch.empty((B, I, H), device=enc.device, dtype=torch.float32)
        grid_split = (B,)
        split_seqs_kernel[grid_split](
            C, processed_encoder, processed_hidden,
            B, T, I, H, S,
            C.stride(0), C.stride(1), C.stride(2),
            processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
            processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
            BLOCK_H=128,
            num_warps=1, num_stages=1,
        )

        # Cast outputs back to original dtype if needed (here, original dtype is float32)
        return processed_encoder, processed_hidden