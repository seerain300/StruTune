import torch
import triton
import triton.language as tl


@triton.jit
def _concat_sequences_kernel(
    out_ptr,           # *fp32, shape [N, L_total, K]
    enc_ptr,           # *fp32, shape [N, L_txt, K]
    hid_ptr,           # *fp32, shape [N, L_img, K]
    N, L_txt, L_img, K,
    STRIDE_OUT_N, STRIDE_OUT_T, STRIDE_OUT_K,
    STRIDE_ENC_N, STRIDE_ENC_T, STRIDE_ENC_K,
    STRIDE_HID_N, STRIDE_HID_T, STRIDE_HID_K,
    tiles_k: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # grid = (N, L_total, tiles_k)
    n = tl.program_id(0)
    t = tl.program_id(1)
    tile_k = tl.program_id(2)

    k_offsets = tile_k * BLOCK_K + tl.arange(0, BLOCK_K)
    k_mask = k_offsets < K

    # Determine source tensor: if t < L_txt, from enc; else from hid
    is_enc = t < L_txt

    # Compute addresses
    # out: out_ptr + n*STRIDE_OUT_N + t*STRIDE_OUT_T + k_offsets*STRIDE_OUT_K
    out_addrs = n * STRIDE_OUT_N + t * STRIDE_OUT_T + k_offsets * STRIDE_OUT_K

    if is_enc:
        # enc: enc_ptr + n*STRIDE_ENC_N + t*STRIDE_ENC_T + k_offsets*STRIDE_ENC_K
        enc_addrs = n * STRIDE_ENC_N + t * STRIDE_ENC_T + k_offsets * STRIDE_ENC_K
        vals = tl.load(enc_ptr + enc_addrs, mask=k_mask, other=0.0)
    else:
        # hid index: t - L_txt
        hid_t = t - L_txt
        # hid: hid_ptr + n*STRIDE_HID_N + hid_t*STRIDE_HID_T + k_offsets*STRIDE_HID_K
        hid_addrs = n * STRIDE_HID_N + hid_t * STRIDE_HID_T + k_offsets * STRIDE_HID_K
        vals = tl.load(hid_ptr + hid_addrs, mask=k_mask, other=0.0)

    tl.store(out_ptr + out_addrs, vals, mask=k_mask)


@triton.jit
def _matmul_row_kernel(
    C_ptr, A_ptr, B_ptr,       # *fp32
    N_rows, K,
    STRIDE_C_ROW, STRIDE_C_COL,
    STRIDE_A_ROW, STRIDE_A_COL,
    STRIDE_B_ROW, STRIDE_B_COL,
    BLOCK_K: tl.constexpr,
):
    # Each program handles one row i in [0, N_rows)
    i = tl.program_id(0)

    # Accumulator
    acc = tl.zeros([K], dtype=tl.float32)

    # Reduction over K in chunks of BLOCK_K
    k0 = 0
    while k0 < K:
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < K

        # A_row[i, k] = A_ptr[i*STRIDE_A_ROW + k*STRIDE_A_COL]
        a_addrs = i * STRIDE_A_ROW + k_offsets * STRIDE_A_COL
        a_vals = tl.load(A_ptr + a_addrs, mask=k_mask, other=0.0).to(tl.float32)

        # B[:, k] = B_ptr[k*STRIDE_B_ROW + k_offsets*STRIDE_B_COL]
        b_addrs = k_offsets * STRIDE_B_ROW + k_offsets * STRIDE_B_COL  # stride col should be 1
        b_vals = tl.load(B_ptr + b_addrs, mask=k_mask, other=0.0).to(tl.float32)

        # acc += a_vals * b_vals
        acc += tl.sum(a_vals[:, None] * b_vals[None, :], axis=0)

        k0 += BLOCK_K

    # Store acc to C[i, :]
    c_addrs = i * STRIDE_C_ROW + tl.arange(0, K) * STRIDE_C_COL
    tl.store(C_ptr + c_addrs, acc, mask=True)  # mask=True is fine since k in [0,K)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        hidden_states: [N, L_img, K]
        encoder_hidden_states: [N, L_txt, K]
        process_weight: [K, K]
        returns: (processed_encoder [N, L_txt, K], processed_hidden [N, L_img, K])
        """
        assert hidden_states.ndim == 3 and encoder_hidden_states.ndim == 3 and process_weight.ndim == 2
        assert hidden_states.shape[2] == process_weight.shape[0] == process_weight.shape[1], "K mismatch"

        N = hidden_states.shape[0]
        L_img = hidden_states.shape[1]
        L_txt = encoder_hidden_states.shape[1]
        K = hidden_states.shape[2]
        device = hidden_states.device

        # Ensure dtype is float32 for Triton kernels
        # We will compute in fp32 and cast outputs back to original dtype at the end.
        # Make tensors contiguous
        enc = encoder_hidden_states.contiguous()
        hid = hidden_states.contiguous()
        W = process_weight.contiguous()

        # 1) Concatenate sequences along sequence dimension: out [N, L_total, K]
        L_total = L_txt + L_img
        out = torch.empty((N, L_total, K), device=device, dtype=torch.float32)

        # Strides
        STRIDE_OUT_N = out.stride(0)
        STRIDE_OUT_T = out.stride(1)
        STRIDE_OUT_K = out.stride(2)

        STRIDE_ENC_N = enc.stride(0)
        STRIDE_ENC_T = enc.stride(1)
        STRIDE_ENC_K = enc.stride(2)

        STRIDE_HID_N = hid.stride(0)
        STRIDE_HID_T = hid.stride(1)
        STRIDE_HID_K = hid.stride(2)

        # Launch concat kernel
        BLOCK_K = 128
        tiles_k = triton.cdiv(K, BLOCK_K)
        grid_concat = (N, L_total, tiles_k)
        _concat_sequences_kernel[grid_concat](
            out, enc, hid,
            N, L_txt, L_img, K,
            STRIDE_OUT_N, STRIDE_OUT_T, STRIDE_OUT_K,
            STRIDE_ENC_N, STRIDE_ENC_T, STRIDE_ENC_K,
            STRIDE_HID_N, STRIDE_HID_T, STRIDE_HID_K,
            tiles_k=tiles_k, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        # 2) GEMM: out @ W.T -> [N, L_total, K] in fp32
        # Flatten rows: N_rows = N * L_total
        N_rows = N * L_total
        out_rows = out.reshape(N_rows, K).contiguous()  # [N_rows, K]
        W_T = W.t().contiguous()  # [K, K]

        # Allocate output rows
        C_rows = torch.empty((N_rows, K), device=device, dtype=torch.float32)

        # Strides for row-wise matmul
        STRIDE_C_ROW = C_rows.stride(0)
        STRIDE_C_COL = C_rows.stride(1)
        STRIDE_A_ROW = out_rows.stride(0)
        STRIDE_A_COL = out_rows.stride(1)
        STRIDE_B_ROW = W_T.stride(0)
        STRIDE_B_COL = W_T.stride(1)  # should be 1 for contiguous [K, K]

        grid_gemm = (N_rows,)
        BLOCK_K_GEMM = 128 if K >= 128 else 64
        _matmul_row_kernel[grid_gemm](
            C_rows, out_rows, W_T,
            N_rows, K,
            STRIDE_C_ROW, STRIDE_C_COL,
            STRIDE_A_ROW, STRIDE_A_COL,
            STRIDE_B_ROW, STRIDE_B_COL,
            BLOCK_K=BLOCK_K_GEMM,
            num_warps=4, num_stages=2,
        )

        # 3) Reshape back and split
        processed = C_rows.view(N, L_total, K)
        processed_encoder = processed[:, :L_txt, :]
        processed_hidden = processed[:, L_txt:, :]

        # Cast back to original dtype (match PyTorch behavior)
        if hidden_states.dtype != processed_encoder.dtype:
            processed_encoder = processed_encoder.to(hidden_states.dtype)
        if hidden_states.dtype != processed_hidden.dtype:
            processed_hidden = processed_hidden.to(hidden_states.dtype)

        return processed_encoder, processed_hidden