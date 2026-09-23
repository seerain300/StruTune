import torch
import triton
import triton.language as tl


@triton.jit
def concat_encoder_image_kernel(
    e_ptr,            # *const T: [B, T, H]
    i_ptr,            # *const T: [B, I, H]
    out_ptr,          # *T: [B, T+I, H]
    B: tl.int32, T: tl.int32, I: tl.int32, H: tl.int32,
    e_s0, e_s1, e_s2, # strides for e_ptr
    i_s0, i_s1, i_s2, # strides for i_ptr
    o_s0, o_s1, o_s2, # strides for out_ptr
    BLOCK_l: tl.constexpr, BLOCK_h: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_l = tl.program_id(1)
    pid_h = tl.program_id(2)

    b = pid_b
    l = pid_l * BLOCK_l + tl.arange(0, BLOCK_l)  # sequence indices in [0, T+I)
    h = pid_h * BLOCK_h + tl.arange(0, BLOCK_h)  # hidden indices in [0, H)

    # Masks for valid ranges
    mask_l = l < (T + I)
    mask_h = h < H
    mask = mask_l[:, None] & mask_h[None, :]

    # Compute base offsets
    # out[b, l, h] -> o_s0*b + o_s1*l + o_s2*h
    out_offsets = b * o_s0 + l[:, None] * o_s2 + h[None, :] * o_s1

    # Select source: if l < T -> e[b, l, h], else -> i[b, l - T, h]
    mask_e = mask & (l[:, None] < T)
    mask_i = mask & (l[:, None] >= T)

    # e offsets: e_s0*b + e_s1*l + e_s2*h
    e_offsets = b * e_s0 + l[:, None] * e_s2 + h[None, :] * e_s1
    # i offsets: i_s0*b + i_s1*(l - T) + i_s2*h
    i_offsets = b * i_s0 + (l[:, None] - T) * i_s1 + h[None, :] * i_s2

    # Load from encoder and image with masks; where mask_e is True, load e; where mask_i is True, load i
    # Triton doesn't support direct conditional load on tl.tensor; use masked load and combine.
    val_e = tl.load(e_ptr + e_offsets, mask=mask_e, other=0)
    val_i = tl.load(i_ptr + i_offsets, mask=mask_i, other=0)
    # Combine: for positions not in e, val_e is 0; for positions not in i, val_i is 0
    out_vals = val_e + val_i

    # Store to out
    tl.store(out_ptr + out_offsets, out_vals, mask=mask)


@triton.jit
def matmul_kernel(
    A_ptr,     # *const T: [B, L, H] (concatenated)
    W_ptr,     # *const T: [H, H] (process_weight, shape matches HxH)
    C_ptr,     # *T: [B, L, H] (output processed)
    B: tl.int32, T: tl.int32, I: tl.int32, H: tl.int32, L: tl.int32,
    A_s0, A_s1, A_s2,
    W_s0, W_s1, W_s2,
    C_s0, C_s1, C_s2,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Flatten M = B * L
    M = B * L

    pid_m = tl.program_id(0)  # tile index along M
    pid_n = tl.program_id(1)  # tile index along N (H)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # rows in [0, M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # columns in [0, H)

    # Masks for boundaries
    mask_m = m_offsets < M
    mask_n = n_offsets < H
    mask_n_2d = mask_m[:, None] & mask_n[None, :]

    # Map m_offsets to (b, l)
    b_idx = m_offsets // L
    l_idx = m_offsets % L

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Reduction over K = H
    for k in range(0, H, BLOCK_K):
        k_offsets = k + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < H

        # Load A[m, k] -> shape (BLOCK_M, BLOCK_K): A[b, l, k]
        a_offsets = b_idx[:, None] * A_s0 + l_idx[:, None] * A_s1 + k_offsets[None, :] * A_s2
        a_mask = mask_m[:, None] & mask_k[None, :]
        a = tl.load(A_ptr + a_offsets, mask=a_mask, other=0.0)
        a = a.to(tl.float32)  # accumulate in fp32 for stability

        # Load W^T[k, n] -> we need W[k, n] since W is [H, H]
        # W[k, n] offsets: W_s0*k + W_s1*n
        w_offsets = k_offsets[:, None] * W_s0 + n_offsets[None, :] * W_s1
        w_mask = mask_k[:, None] & mask_n[None, :]
        w = tl.load(W_ptr + w_offsets, mask=w_mask, other=0.0)
        w = w.to(tl.float32)

        # Accumulate: (BLOCK_M, BLOCK_K) @ (BLOCK_K, BLOCK_N) -> (BLOCK_M, BLOCK_N)
        acc += tl.dot(a, w)

    # Write back to C[b, l, n]
    # Compute c offsets: C_s0*b + C_s1*l + C_s2*n
    c_offsets = b_idx[:, None] * C_s0 + l_idx[:, None] * C_s1 + n_offsets[None, :] * C_s2
    tl.store(C_ptr + c_offsets, acc, mask=mask_n_2d)


@triton.jit
def copy_rows_kernel(
    src_ptr,            # *const T: [B, L, H]
    dst_ptr,            # *T: [B, L, H]
    B: tl.int32, L: tl.int32, H: tl.int32,
    src_s0, src_s1, src_s2,
    dst_s0, dst_s1, dst_s2,
    ROW_START: tl.int32,       # starting row index to copy (e.g., 0 for encoder, T for hidden)
    BLOCK_l: tl.constexpr, BLOCK_h: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_l = tl.program_id(1)
    pid_h = tl.program_id(2)

    b = pid_b
    l = pid_l * BLOCK_l + tl.arange(0, BLOCK_l)
    h = pid_h * BLOCK_h + tl.arange(0, BLOCK_h)

    mask_l = l < L
    mask_h = h < H
    mask = mask_l[:, None] & mask_h[None, :]

    # src offsets for rows starting at ROW_START
    src_offsets = b * src_s0 + (ROW_START + l)[:, None] * src_s1 + h[None, :] * src_s2
    # dst offsets for rows starting at ROW_START
    dst_offsets = b * dst_s0 + (ROW_START + l)[:, None] * dst_s1 + h[None, :] * dst_s2

    vals = tl.load(src_ptr + src_offsets, mask=mask, other=0)
    tl.store(dst_ptr + dst_offsets, vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,        # [B, I, H]
        encoder_hidden_states: torch.Tensor, # [B, T, H]
        process_weight: torch.Tensor,       # [H, H]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Ensure CUDA tensors
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Tensors must be on CUDA for Triton kernels"

        B = hidden_states.shape[0]
        I = hidden_states.shape[1]
        T = encoder_hidden_states.shape[1]
        H = hidden_states.shape[2]
        L = T + I

        # 1) Concatenate using Triton
        out = torch.empty((B, L, H), device=hidden_states.device, dtype=hidden_states.dtype)
        BLOCK_l = 128
        BLOCK_h = 64
        grid_concat = (B, triton.cdiv(L, BLOCK_l), triton.cdiv(H, BLOCK_h))
        concat_encoder_image_kernel[grid_concat](
            encoder_hidden_states, hidden_states, out,
            B, T, I, H,
            encoder_hidden_states.stride(0), encoder_hidden_states.stride(1), encoder_hidden_states.stride(2),
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            out.stride(0), out.stride(1), out.stride(2),
            BLOCK_l=BLOCK_l, BLOCK_h=BLOCK_h,
            num_warps=4, num_stages=2,
        )

        # 2) Matmul: out @ process_weight.T
        C = torch.empty((B, L, H), device=hidden_states.device, dtype=hidden_states.dtype)
        # Choose tile sizes; H can vary, so pick moderate sizes
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 64
        grid_matmul = (triton.cdiv(B * L, BLOCK_M), triton.cdiv(H, BLOCK_N))
        matmul_kernel[grid_matmul](
            out, process_weight, C,
            B, T, I, H, L,
            out.stride(0), out.stride(1), out.stride(2),
            process_weight.stride(0), process_weight.stride(1), process_weight.stride(2),
            C.stride(0), C.stride(1), C.stride(2),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        # 3) Split using Triton copy kernels
        processed_encoder = torch.empty((B, T, H), device=hidden_states.device, dtype=hidden_states.dtype)
        processed_hidden = torch.empty((B, I, H), device=hidden_states.device, dtype=hidden_states.dtype)

        # Copy first T rows
        grid_copy_encoder = (B, triton.cdiv(T, 64), triton.cdiv(H, 64))
        copy_rows_kernel[grid_copy_encoder](
            C, processed_encoder,
            B, L, H,
            C.stride(0), C.stride(1), C.stride(2),
            processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
            ROW_START=0,
            BLOCK_l=64, BLOCK_h=64,
            num_warps=4, num_stages=2,
        )

        # Copy next I rows (from index T to T+I)
        grid_copy_hidden = (B, triton.cdiv(I, 64), triton.cdiv(H, 64))
        copy_rows_kernel[grid_copy_hidden](
            C, processed_hidden,
            B, L, H,
            C.stride(0), C.stride(1), C.stride(2),
            processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
            ROW_START=T,
            BLOCK_l=64, BLOCK_h=64,
            num_warps=4, num_stages=2,
        )

        return processed_encoder, processed_hidden