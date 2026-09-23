import torch
import triton
import triton.language as tl


@triton.jit
def concatenate_sequences_kernel(
    e_ptr, i_ptr, out_ptr,
    B: tl.int32, T: tl.int32, I: tl.int32, H: tl.int32,
    e_s0, e_s1, e_s2,  # strides for e_ptr
    i_s0, i_s1, i_s2,  # strides for i_ptr
    o_s0, o_s1, o_s2,  # strides for out_ptr
    BLOCK_l: tl.constexpr, BLOCK_h: tl.constexpr,
):
    # program ids
    pid_b = tl.program_id(0)
    pid_l = tl.program_id(1)
    pid_h = tl.program_id(2)

    # tile indices
    l = pid_l * BLOCK_l + tl.arange(0, BLOCK_l)  # [BLOCK_l]
    h = pid_h * BLOCK_h + tl.arange(0, BLOCK_h)  # [BLOCK_h]

    # form 2D grid
    L = T + I

    # broadcast to 2D [BLOCK_l, BLOCK_h]
    L_mat = l[:, None]
    H_mat = h[None, :]

    # bounds mask
    mask = (L_mat < L) & (H_mat < H) & (pid_b < B)

    # select source: first T rows from e, next I rows from i
    from_encoder = L_mat < T
    # for indices >= T, subtract T to map to i
    idx_i = L_mat - T

    # compute pointers
    e_ptrs = e_ptr + pid_b * e_s0 + l[:, None] * e_s1 + h[None, :] * e_s2
    i_ptrs = i_ptr + pid_b * i_s0 + (l[:, None] - T) * i_s1 + h[None, :] * i_s2
    out_ptrs = out_ptr + pid_b * o_s0 + L_mat * o_s1 + H_mat * o_s2

    # load from encoder or image depending on l
    # Note: when from_encoder is False, idx_i may be negative; masking ensures we don't read invalid elements.
    e_vals = tl.load(e_ptrs, mask=mask & from_encoder, other=0.0)
    i_vals = tl.load(i_ptrs, mask=mask & (~from_encoder), other=0.0)

    vals = tl.where(from_encoder, e_vals, i_vals)
    tl.store(out_ptrs, vals, mask=mask)


@triton.jit
def matmul_sequences_kernel(
    A_ptr, W_ptr, C_ptr,
    B: tl.int32, L: tl.int32, H: tl.int32,
    A_s0, A_s1, A_s2,  # A strides: [B, L, H]
    W_s0, W_s1, W_s2,  # W strides: [H, H]
    C_s0, C_s1, C_s2,  # C strides: [B, L, H]
    BLOCK_m: tl.constexpr, BLOCK_n: tl.constexpr, BLOCK_k: tl.constexpr,
):
    # program ids: flatten M = B*L
    pid_m = tl.program_id(0)  # over rows
    pid_n = tl.program_id(1)  # over columns

    # tile indices
    m = pid_m * BLOCK_m + tl.arange(0, BLOCK_m)  # rows in [0, B*L)
    n = pid_n * BLOCK_n + tl.arange(0, BLOCK_n)  # cols in [0, H)

    # compute b and l for each row m
    # m = b*(L) + l, but we don't have L in kernel; we reconstruct using m
    # We can iterate k over H; Triton will vectorize. For simplicity, use for loop over k.
    M = B * L

    # masks for bounds
    mask_m = m < M
    mask_n = n < H

    # Accumulator
    acc = tl.zeros((BLOCK_m, BLOCK_n), dtype=tl.float32)

    # Reduction over K = H
    for k0 in range(0, H, BLOCK_k):
        k = k0 + tl.arange(0, BLOCK_k)  # [BLOCK_k]
        mask_k = k < H

        # Map m to (b, l)
        # For each m, b = m // L, l = m % L. However, we only need l for reading A, not b for W.
        # We can compute b as m // L when L is known. Here, we reconstruct l using the fact that A rows are contiguous in (b, l).
        # Simpler: we load A using the m index and read its l = m % L. We need b to index A_s0; we can derive b = m // L.
        b_vec = m // L
        l_vec = m % L

        # Pointers for A: A[b, l, k]
        A_ptrs = A_ptr + b_vec[:, None] * A_s0 + l_vec[:, None] * A_s1 + k[None, :] * A_s2
        # Pointers for W: W[k, n]
        W_ptrs = W_ptr + k[:, None] * W_s0 + n[None, :] * W_s1

        # Valid mask combining m, n, k bounds
        mask_A = mask_m[:, None] & mask_k[None, :] & mask_n[None, :]
        a = tl.load(A_ptrs, mask=mask_A, other=0.0)  # [BLOCK_m, BLOCK_k]
        w = tl.load(W_ptrs, mask=(mask_k[:, None] & mask_n[None, :]), other=0.0)  # [BLOCK_k, BLOCK_n]

        # Accumulate
        acc += tl.dot(a, w)

    # Write result to C[b, l, n]
    # We need to map m back to b and l for storing: C[b, l, n]
    # For each m, b = m // L, l = m % L
    b_vec = m // L
    l_vec = m % L

    C_ptrs = C_ptr + b_vec[:, None] * C_s0 + l_vec[:, None] * C_s1 + n[None, :] * C_s2
    mask_out = mask_m[:, None] & mask_n[None, :]
    tl.store(C_ptrs, acc, mask=mask_out)


@triton.jit
def copy_rows_kernel(
    in_ptr, out_ptr,
    B: tl.int32, NUM_ROWS: tl.int32, H: tl.int32,
    in_s0, in_s1, in_s2,
    out_s0, out_s1, out_s2,
    ROW_START: tl.int32,  # starting row index in input
    BLOCK_l: tl.constexpr, BLOCK_h: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_l = tl.program_id(1)  # over rows
    pid_h = tl.program_id(2)  # over hidden dim

    l = pid_l * BLOCK_l + tl.arange(0, BLOCK_l)  # row indices in [0, NUM_ROWS)
    h = pid_h * BLOCK_h + tl.arange(0, BLOCK_h)  # hidden indices

    L = NUM_ROWS
    mask = (l < L) & (h < H) & (pid_b < B)

    # Input pointers: in[b, l+ROW_START, h]
    in_ptrs = in_ptr + pid_b * in_s0 + (l + ROW_START) * in_s1 + h * in_s2
    # Output pointers: out[b, l, h]
    out_ptrs = out_ptr + pid_b * out_s0 + l * out_s1 + h * out_s2

    vals = tl.load(in_ptrs, mask=mask, other=0.0)
    tl.store(out_ptrs, vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        # Shapes
        B, T, H = encoder_hidden_states.shape
        B2, I, H2 = hidden_states.shape
        assert B == B2, "Batch size mismatch"
        assert H == H2, "Hidden dim mismatch"
        # Ensure device/dtype consistency
        device = hidden_states.device
        dtype = hidden_states.dtype

        # 1) Concatenate along sequence dimension [B, T+I, H]
        L = T + I
        out = torch.empty((B, L, H), dtype=dtype, device=device)

        grid_concat = (B, triton.cdiv(L, 128), triton.cdiv(H, 128))
        concatenate_sequences_kernel[grid_concat](
            encoder_hidden_states, hidden_states, out,
            B, T, I, H,
            encoder_hidden_states.stride(0), encoder_hidden_states.stride(1), encoder_hidden_states.stride(2),
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            out.stride(0), out.stride(1), out.stride(2),
            BLOCK_l=128, BLOCK_h=128,
            num_warps=4, num_stages=2,
        )

        # 2) Linear projection: processed = out @ process_weight.T
        W_t = process_weight.t().to(device=device, dtype=dtype)  # ensure same dtype/device
        processed = torch.empty((B, L, H), dtype=dtype, device=device)

        grid_mm = (triton.cdiv(B * L, 64), triton.cdiv(H, 64))
        matmul_sequences_kernel[grid_mm](
            out, W_t, processed,
            B, L, H,
            out.stride(0), out.stride(1), out.stride(2),
            W_t.stride(0), W_t.stride(1), W_t.stride(2),
            processed.stride(0), processed.stride(1), processed.stride(2),
            BLOCK_m=64, BLOCK_n=64, BLOCK_k=64,
            num_warps=4, num_stages=2,
        )

        # 3) Split back into two streams
        processed_encoder = torch.empty((B, T, H), dtype=dtype, device=device)
        grid_copy_encoder = (B, triton.cdiv(T, 64), triton.cdiv(H, 64))
        copy_rows_kernel[grid_copy_encoder](
            processed, processed_encoder,
            B, T, H,
            processed.stride(0), processed.stride(1), processed.stride(2),
            processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
            ROW_START=0, BLOCK_l=64, BLOCK_h=64,
            num_warps=4, num_stages=2,
        )

        processed_hidden = torch.empty((B, I, H), dtype=dtype, device=device)
        grid_copy_hidden = (B, triton.cdiv(I, 64), triton.cdiv(H, 64))
        copy_rows_kernel[grid_copy_hidden](
            processed, processed_hidden,
            B, I, H,
            processed.stride(0), processed.stride(1), processed.stride(2),
            processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
            ROW_START=T, BLOCK_l=64, BLOCK_h=64,
            num_warps=4, num_stages=2,
        )

        return processed_encoder, processed_hidden