import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


@triton.jit
def _concatenate_seqs_kernel(
    enc_ptr, hid_ptr, out_ptr,
    B, T, I, H,
    enc_bs, enc_ts, enc_hs,
    hid_bs, hid_is, hid_hs,
    out_bs, out_ts, out_hs,
    BLOCK_T: tl.constexpr, BLOCK_I: tl.constexpr,
):
    # One program per batch
    b = tl.program_id(0)

    # Base pointers for this batch
    enc_b = enc_ptr + b * enc_bs
    hid_b = hid_ptr + b * hid_bs
    out_b = out_ptr + b * out_bs

    # Iterate over total sequence length T + I
    for l in range(0, T + I):
        is_encoder = l < T
        if is_encoder:
            src_ptr = enc_b
            idx = l
        else:
            src_ptr = hid_b
            idx = l - T
        # Vector over H columns
        offs_h = tl.arange(0, H)
        vals = tl.load(src_ptr + idx * enc_ts + offs_h * enc_hs, mask=offs_h < H, other=0.0)
        tl.store(out_b + l * out_ts + offs_h * out_hs, vals, mask=offs_h < H)


@triton.jit
def _batched_gemm_right_kernel(
    A_ptr, WT_ptr, C_ptr,
    M, K, N,
    A_stride_m, A_stride_k,
    WT_stride_row, WT_stride_col,
    C_stride_m, C_stride_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D tiling over output C of shape [M, N]
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Reduction over K
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)

        # Load A tile: shape [BLOCK_M, BLOCK_K]
        a_ptrs = A_ptr + m_offsets[:, None] * A_stride_m + k_offsets[None, :] * A_stride_k
        a = tl.load(a_ptrs, mask=(m_offsets[:, None] < M) & (k_offsets[None, :] < K), other=0.0)

        # Load W^T tile: shape [BLOCK_K, BLOCK_N] where W^T[n, k] = WT[k, n]
        wt_ptrs = WT_ptr + k_offsets[:, None] * WT_stride_row + n_offsets[None, :] * WT_stride_col
        wt = tl.load(wt_ptrs, mask=(k_offsets[:, None] < K) & (n_offsets[None, :] < N), other=0.0)

        # Accumulate: [BLOCK_M, BLOCK_K] @ [BLOCK_K, BLOCK_N] -> [BLOCK_M, BLOCK_N]
        acc += tl.dot(a, wt)

    # Store results to C
    c_ptrs = C_ptr + m_offsets[:, None] * C_stride_m + n_offsets[None, :] * C_stride_n
    tl.store(c_ptrs, acc, mask=(m_offsets[:, None] < M) & (n_offsets[None, :] < N))


@triton.jit
def _split_streams_kernel(
    C_ptr, out_encoder_ptr, out_hidden_ptr,
    M, T, I, H,
    C_stride_m, C_stride_n,
    enc_bs, enc_ts, enc_hs,  # not used but can be for future
    hid_bs, hid_ts, hid_hs,  # not used but can be for future
    out_encoder_bs, out_encoder_ts, out_encoder_hs,
    out_hidden_bs, out_hidden_is, out_hidden_hs,
    BLOCK_H: tl.constexpr,
):
    # One program per batch
    b = tl.program_id(0)

    # Compute number of rows for each stream
    rows_encoder = T
    rows_image = I

    # Copy encoder stream: rows 0..T-1
    for s in range(0, T):
        m = b * (T + I) + s
        # Vector over H
        offs_h = tl.arange(0, H)
        vals = tl.load(C_ptr + m * C_stride_m + offs_h * C_stride_n, mask=offs_h < H, other=0.0)
        # Store to encoder output
        tl.store(out_encoder_ptr + b * out_encoder_bs + s * out_encoder_ts + offs_h * out_encoder_hs, vals, mask=offs_h < H)

    # Copy image stream: rows T..T+I-1
    for s in range(0, I):
        m = b * (T + I) + T + s
        offs_h = tl.arange(0, H)
        vals = tl.load(C_ptr + m * C_stride_m + offs_h * C_stride_n, mask=offs_h < H, other=0.0)
        # Store to image output
        tl.store(out_hidden_ptr + b * out_hidden_bs + s * out_hidden_is + offs_h * out_hidden_hs, vals, mask=offs_h < H)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-only implementation:
        - Concatenate encoder_hidden_states and hidden_states along sequence dimension.
        - Apply linear projection via Triton GEMM: concatenated @ process_weight.T
        - Split back into encoder and image outputs.
        Returns (processed_encoder_hidden_states, processed_hidden_states).
        """
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "All tensors must be on CUDA for Triton"
        assert hidden_states.dim() == 3 and encoder_hidden_states.dim() == 3, "Inputs must be 3D tensors [B, L, H]"
        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        H = hidden_states.shape[2]
        assert encoder_hidden_states.shape[2] == H and process_weight.shape[0] == H and process_weight.shape[1] == H, "Dimension mismatch: hidden_dim must match"

        # Ensure contiguous for simple stride arithmetic
        enc = encoder_hidden_states.contiguous()
        hid = hidden_states.contiguous()
        wt = process_weight.t().contiguous()  # right-multiply: W_T of shape [H, H]

        # 1) Concatenate along sequence dimension into out_cat [B, T+I, H]
        out_cat = torch.empty((B, T + I, H), dtype=hidden_states.dtype, device=hidden_states.device)
        # Launch concatenation: one program per batch
        _concatenate_seqs_kernel[(B,)](
            enc, hid, out_cat,
            B, T, I, H,
            enc.stride(0), enc.stride(1), enc.stride(2),
            hid.stride(0), hid.stride(1), hid.stride(2),
            out_cat.stride(0), out_cat.stride(1), out_cat.stride(2),
            BLOCK_T=128, BLOCK_I=128,
            num_warps=1, num_stages=1,
        )

        # 2) GEMM: C = out_cat @ W_T, where out_cat [M,K] with M = B*(T+I), K = H
        M = B * (T + I)
        K = H
        N = H  # output columns = hidden_dim

        # Allocate output C [M, N]
        C = torch.empty((M, N), dtype=torch.float32, device=hidden_states.device)  # compute in fp32

        # Launch 2D GEMM kernel
        # Grid over tiles: (ceil_div(M, BLOCK_M), ceil_div(N, BLOCK_N))
        # Choose reasonable defaults; you can tune these for performance
        BLOCK_M = 128
        BLOCK_N = 128
        BLOCK_K = 32
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        _batched_gemm_right_kernel[grid](
            out_cat, wt, C,
            M, K, N,
            out_cat.stride(0), out_cat.stride(1),
            wt.stride(0), wt.stride(1),
            C.stride(0), C.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        # 3) Split C [M, H] back into [B, T, H] and [B, I, H]
        processed_encoder = torch.empty((B, T, H), dtype=torch.float32, device=hidden_states.device)
        processed_hidden = torch.empty((B, I, H), dtype=torch.float32, device=hidden_states.device)

        _split_streams_kernel[(B,)](
            C,
            processed_encoder, processed_hidden,
            M, T, I, H,
            C.stride(0), C.stride(1),
            0, 0, 0,  # unused
            0, 0, 0,  # unused
            processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
            processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
            BLOCK_H=256,
            num_warps=1, num_stages=1,
        )

        # Return as originally requested (matching shapes and dtypes of torch implementation)
        return processed_encoder, processed_hidden