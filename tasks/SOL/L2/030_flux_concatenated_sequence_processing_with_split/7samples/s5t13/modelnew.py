import torch
import triton
import triton.language as tl


@triton.jit
def concat_encoder_image_kernel(
    e_ptr,            # *const float32: [B, T, H]
    i_ptr,            # *const float32: [B, I, H]
    out_ptr,          # *float32: [B, T+I, H]
    B: tl.int32, T: tl.int32, I: tl.int32, H: tl.int32,
    e_s0, e_s1, e_s2, # strides for e_ptr
    i_s0, i_s1, i_s2, # strides for i_ptr
    o_s0, o_s1, o_s2, # strides for out_ptr
    BLOCK_l: tl.constexpr, BLOCK_h: tl.constexpr,
):
    # Program IDs: batch, tiles over sequence (T+I), tiles over hidden (H)
    pid_b = tl.program_id(0)
    pid_l = tl.program_id(1)
    pid_h = tl.program_id(2)

    # Indices within tile
    l = pid_l * BLOCK_l + tl.arange(0, BLOCK_l)  # sequence positions in [0, T+I)
    h = pid_h * BLOCK_h + tl.arange(0, BLOCK_h)  # hidden dim indices in [0, H)

    # Masks for bounds
    mask_l = l < (T + I)
    mask_h = h < H
    mask = mask_l[:, None] & mask_h[None, :]

    # Base pointers for current batch
    base_e = e_ptr + pid_b * e_s0
    base_i = i_ptr + pid_b * i_s0
    base_o = out_ptr + pid_b * o_s0

    # Compute source addresses for each output (b, l, h)
    # If l < T: source = e_ptr[b, l, h]; else: source = i_ptr[b, l - T, h]
    addr_e = base_e + l[:, None] * e_s1 + h[None, :] * e_s2
    addr_i = base_i + (l[:, None] - T) * i_s1 + h[None, :] * i_s2
    addr_out = base_o + l[:, None] * o_s1 + h[None, :] * o_s2

    # Select source: for l < T use e_ptr, else use i_ptr
    sel = l[:, None] < T
    src = tl.where(sel, addr_e, addr_i)

    # Load and store
    val = tl.load(src, mask=mask, other=0.0)
    tl.store(addr_out, val, mask=mask)


@triton.jit
def matmul_rows_cols_kernel(
    a_ptr,          # *const float32: [B*(T+I), H] (concatenated flattened as rows)
    w_ptr,          # *const float32: [H, H] (process_weight.T)
    c_ptr,          # *float32: [B*(T+I), H] (output processed)
    M: tl.int32, N: tl.int32, K: tl.int32,
    a_s0, a_s1,     # strides for a_ptr (row, col)
    w_s0, w_s1,     # strides for w_ptr (row, col)
    c_s0, c_s1,     # strides for c_ptr (row, col)
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)  # tiles over rows M
    pid_n = tl.program_id(1)  # tiles over cols N

    m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K in chunks
    for k in range(0, K, BLOCK_K):
        k_vec = k + tl.arange(0, BLOCK_K)
        # A tile: [BLOCK_M, BLOCK_K]
        a_ptrs = a_ptr + m[:, None] * a_s0 + k_vec[None, :] * a_s1
        a_mask = (m[:, None] < M) & (k_vec[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # W^T tile: we want W^T[k, n], which equals W[n, k] (since W is [H, H])
        w_ptrs = w_ptr + n[None, :] * w_s0 + k_vec[:, None] * w_s1  # shape (BLOCK_K, BLOCK_N)
        w_mask = (n[None, :] < N) & (k_vec[:, None] < K)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)  # shape (BLOCK_K, BLOCK_N)

        # Multiply-accumulate
        acc += tl.dot(a, w)  # a: (BLOCK_M, BLOCK_K), w: (BLOCK_K, BLOCK_N) -> (BLOCK_M, BLOCK_N)

    # Store results
    c_ptrs = c_ptr + m[:, None] * c_s0 + n[None, :] * c_s1
    c_mask = (m[:, None] < M) & (n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


@triton.jit
def copy_rows_kernel(
    src_ptr,         # *const float32: source tensor [B, L, H]
    dst_ptr,         # *float32: destination tensor [B, L, H]
    B: tl.int32, L: tl.int32, H: tl.int32,
    src_s0, src_s1, src_s2,
    dst_s0, dst_s1, dst_s2,
    ROW_START: tl.int32, BLOCK_l: tl.constexpr, BLOCK_h: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_l = tl.program_id(1)
    pid_h = tl.program_id(2)

    l = ROW_START + pid_l * BLOCK_l + tl.arange(0, BLOCK_l)  # row indices to copy
    h = pid_h * BLOCK_h + tl.arange(0, BLOCK_h)             # hidden dim indices

    mask_l = l < (ROW_START + L)
    mask_h = h < H
    mask = mask_l[:, None] & mask_h[None, :]

    base_src = src_ptr + pid_b * src_s0
    base_dst = dst_ptr + pid_b * dst_s0

    src_addrs = base_src + l[:, None] * src_s1 + h[None, :] * src_s2
    dst_addrs = base_dst + l[:, None] * dst_s1 + h[None, :] * dst_s2

    vals = tl.load(src_addrs, mask=mask, other=0.0)
    tl.store(dst_addrs, vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        process_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Concatenate encoder_hidden_states and hidden_states along sequence dimension,
        apply linear projection, and split back into two streams using Triton kernels.

        hidden_states: [B, I, H]
        encoder_hidden_states: [B, T, H]
        process_weight: [H, H]
        returns:
          processed_encoder: [B, T, H]
          processed_hidden: [B, I, H]
        """
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, \
            "All inputs must be CUDA tensors for Triton kernels."

        Bsz = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        H = hidden_states.shape[2]
        # Ensure dtype float32 for Triton kernels
        dtype = torch.float32

        # 1) Concatenate along sequence dimension: out [B, T+I, H]
        out = torch.empty((Bsz, T + I, H), device=hidden_states.device, dtype=dtype)
        grid_concat = (Bsz, triton.cdiv(T + I, 64), triton.cdiv(H, 64))
        concat_encoder_image_kernel[grid_concat](
            encoder_hidden_states, hidden_states, out,
            Bsz, T, I, H,
            encoder_hidden_states.stride(0), encoder_hidden_states.stride(1), encoder_hidden_states.stride(2),
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            out.stride(0), out.stride(1), out.stride(2),
            BLOCK_l=64, BLOCK_h=64,
            num_warps=4, num_stages=2,
        )

        # 2) Apply linear projection: processed = out @ process_weight.T
        # Flatten rows M = B*(T+I)
        M = Bsz * (T + I)
        N = H
        K = H
        a = out  # [B, T+I, H]
        # Prepare W^T = process_weight.T as [H, H] in float32
        w_t = process_weight.t().contiguous().to(dtype)  # [H, H]
        processed = torch.empty((Bsz, T + I, H), device=hidden_states.device, dtype=dtype)

        # For Triton, pass as row-major (M, N) but we'll use strides over [B, L, H] layout by flattening
        # Here, we treat 'a' as (M, K): reshape to (M, K) and use strides accordingly
        a_rows = a.view(M, K).contiguous()
        c_rows = processed.view(M, N).contiguous()

        grid_matmul = (triton.cdiv(M, 128), triton.cdiv(N, 128))
        matmul_rows_cols_kernel[grid_matmul](
            a_rows, w_t, c_rows,
            M, N, K,
            a_rows.stride(0), a_rows.stride(1),
            w_t.stride(0), w_t.stride(1),
            c_rows.stride(0), c_rows.stride(1),
            BLOCK_M=128, BLOCK_N=128, BLOCK_K=64,
            num_warps=4, num_stages=2,
        )
        # processed is already updated; no need to reassign

        # 3) Split into encoder and hidden streams
        processed_encoder = torch.empty((Bsz, T, H), device=hidden_states.device, dtype=dtype)
        processed_hidden = torch.empty((Bsz, I, H), device=hidden_states.device, dtype=dtype)

        grid_copy_enc = (Bsz, triton.cdiv(T, 64), triton.cdiv(H, 64))
        copy_rows_kernel[grid_copy_enc](
            processed, processed_encoder,
            Bsz, T, H,
            processed.stride(0), processed.stride(1), processed.stride(2),
            processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
            ROW_START=0, BLOCK_l=64, BLOCK_h=64,
            num_warps=4, num_stages=2,
        )

        grid_copy_img = (Bsz, triton.cdiv(I, 64), triton.cdiv(H, 64))
        copy_rows_kernel[grid_copy_img](
            processed, processed_hidden,
            Bsz, I, H,
            processed.stride(0), processed.stride(1), processed.stride(2),
            processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
            ROW_START=T, BLOCK_l=64, BLOCK_h=64,
            num_warps=4, num_stages=2,
        )

        return processed_encoder, processed_hidden