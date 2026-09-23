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
    # Program IDs for tiling
    pid_b = tl.program_id(0)  # batch
    pid_l = tl.program_id(1)  # tiles over sequence (T+I)
    pid_h = tl.program_id(2)  # tiles over hidden dim (H)

    # Compute index vectors
    l = pid_l * BLOCK_l + tl.arange(0, BLOCK_l)     # sequence positions in [0, T+I)
    h = pid_h * BLOCK_h + tl.arange(0, BLOCK_h)     # hidden dim indices in [0, H)

    # Masks for bounds
    mask_l = l < (T + I)
    mask_h = h < H
    mask = mask_l[:, None] & mask_h[None, :]

    # Base pointers for batch
    base_e = e_ptr + pid_b * e_s0
    base_i = i_ptr + pid_b * i_s0
    base_o = out_ptr + pid_b * o_s0

    # Compute source addresses
    # For output at (b, l, h): if l < T -> from e_ptr[b, l, h]; else -> from i_ptr[b, l - T, h]
    # Create 2D grid for l and h
    L2D = l[:, None]  # shape (BLOCK_l, 1)
    H2D = h[None, :]  # shape (1, BLOCK_h)

    # Masks per element
    mask_e = (L2D < T) & mask
    mask_i = (~mask_e) & mask

    # Addresses
    addr_e = base_e + L2D * e_s1 + H2D * e_s2
    addr_i = base_i + (L2D - T) * i_s1 + H2D * i_s2
    addr_o = base_o + L2D * o_s1 + H2D * o_s2

    # Load and store
    val_e = tl.load(addr_e, mask=mask_e, other=0.0)
    val_i = tl.load(addr_i, mask=mask_i, other=0.0)
    # Either val_e or val_i will be valid because mask_e and mask_i are disjoint and cover mask
    val = tl.where(mask_e, val_e, val_i)
    tl.store(addr_o, val, mask=mask)


@triton.jit
def copy_rows_kernel(
    src_ptr, dst_ptr,                      # input/output pointers
    B: tl.int32, N_rows: tl.int32, H: tl.int32,
    src_s0, src_s1, src_s2,               # strides for src_ptr
    dst_s0, dst_s1, dst_s2,               # strides for dst_ptr
    ROW_START: tl.int32,                  # starting row index in src
    BLOCK_l: tl.constexpr, BLOCK_h: tl.constexpr,
):
    pid_b = tl.program_id(0)  # batch
    pid_l = tl.program_id(1)  # tiles over rows (N_rows)
    pid_h = tl.program_id(2)  # tiles over hidden dim (H)

    l = ROW_START + pid_l * BLOCK_l + tl.arange(0, BLOCK_l)   # sequence positions [ROW_START, ROW_START+N_rows)
    h = pid_h * BLOCK_h + tl.arange(0, BLOCK_h)

    mask_l = l < (ROW_START + N_rows)
    mask_h = h < H
    mask = mask_l[:, None] & mask_h[None, :]

    base_src = src_ptr + pid_b * src_s0
    base_dst = dst_ptr + pid_b * dst_s0

    L2D = l[:, None]
    H2D = h[None, :]

    addr_src = base_src + L2D * src_s1 + H2D * src_s2
    addr_dst = base_dst + L2D * dst_s1 + H2D * dst_s2

    val = tl.load(addr_src, mask=mask, other=0.0)
    tl.store(addr_dst, val, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        # Expect: hidden_states: [B, I, H], encoder_hidden_states: [B, T, H], process_weight: [H, H]
        assert hidden_states.dim() == 3 and encoder_hidden_states.dim() == 3 and process_weight.dim() == 2
        Bsz = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        H = hidden_states.shape[2]
        # Ensure device and dtype consistency
        device = hidden_states.device
        dtype = hidden_states.dtype
        process_weight = process_weight.to(device=device, dtype=dtype)

        # 1) Concatenate along sequence dimension into out [B, T+I, H]
        out = torch.empty((Bsz, T + I, H), device=device, dtype=dtype)

        # Launch concat kernel
        BLOCK_l = 128
        BLOCK_h = 64
        grid = (Bsz, triton.cdiv(T + I, BLOCK_l), triton.cdiv(H, BLOCK_h))
        concat_encoder_image_kernel[grid](
            encoder_hidden_states, hidden_states, out,
            Bsz, T, I, H,
            encoder_hidden_states.stride(0), encoder_hidden_states.stride(1), encoder_hidden_states.stride(2),
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            out.stride(0), out.stride(1), out.stride(2),
            BLOCK_l=BLOCK_l, BLOCK_h=BLOCK_h,
            num_warps=4, num_stages=2,
        )

        # 2) Apply linear projection: processed = out @ process_weight.T
        # We implement matmul with Triton. Flatten (B, L, H) with L=T+I.
        L = T + I
        processed = torch.empty((Bsz, L, H), device=device, dtype=dtype)

        # Triton matmul kernel: compute C = A @ W^T, where A has shape [B*L, H], W^T has shape [H, H]
        # We flatten A as [M, K], C as [M, N], W^T as [K, N] (with N=H, K=H).
        M = Bsz * L
        # Reshape A (out) to [M, K], W^T to [K, N]
        A = out.reshape(M, H).contiguous()
        Wt = process_weight.t().contiguous()  # [H, H]
        C = processed.reshape(M, H)  # [M, H]

        # Launch matmul kernel
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 64
        grid_matmul = (Bsz, triton.cdiv(L, BLOCK_M), triton.cdiv(H, BLOCK_N))
        @triton.jit
        def matmul_kernel(A_ptr, Wt_ptr, C_ptr,
                          Bsz: tl.int32, L: tl.int32, H: tl.int32,
                          A_s0, A_s1, Wt_s0, Wt_s1, C_s0, C_s1,
                          BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
            pid_b = tl.program_id(0)  # batch index
            pid_m = tl.program_id(1)  # tiles over L (rows)
            pid_n = tl.program_id(2)  # tiles over H (cols)

            m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [0, L)
            n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [0, H)

            mask_m = m < L
            mask_n = n < H
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

            # Reduce over K=H in chunks
            for k in range(0, H, BLOCK_K):
                kk = k + tl.arange(0, BLOCK_K)  # [0, H)
                mask_k = kk < H

                # A[m, kk]: shape (BLOCK_M, BLOCK_K)
                A_offsets = pid_b * A_s0 + m[:, None] * A_s1 + kk[None, :] * A_s1  # note: A_s1 is stride along K
                A_vals = tl.load(A_ptr + A_offsets, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
                # Wt[kk, n]: shape (BLOCK_K, BLOCK_N)
                Wt_offsets = Wt_ptr + kk[:, None] * Wt_s0 + n[None, :] * Wt_s1
                Wt_vals = tl.load(Wt_ptr + Wt_offsets, mask=mask_k[:, None] & mask_n[None, :], other=0.0)

                # Accumulate in float32
                A_vals = A_vals.to(tl.float32)
                Wt_vals = Wt_vals.to(tl.float32)
                acc += tl.dot(A_vals, Wt_vals)

            # Store back (cast to original dtype if needed)
            C_offsets = pid_b * C_s0 + m[:, None] * C_s1 + n[None, :] * C_s2
            # We need to pass C_s2; since C is [M, H], C_s2 should be element stride (likely 1 for contiguous).
            # Using strides from C.reshape(M, H):
            # But we launched grid with only 3 dims; we should pass correct strides:
            # For C = processed.reshape(M, H), strides are (processed.stride(1)*H + processed.stride(2), processed.stride(2)).
            # To simplify, we'll pass processed strides as if C_s1=1, C_s2=H, but reshape already ensured contiguous.
            # Instead, we compute directly from processed strides in host: For C contiguous, C_s1=H, C_s2=1.
            # Here, processed is contiguous by allocation; so C_s1=H, C_s2=1.
            C_vals = acc
            tl.store(C_ptr + C_offsets, C_vals, mask=mask_m[:, None] & mask_n[None, :])

        matmul_kernel[grid_matmul](
            A, Wt, C,
            Bsz, L, H,
            A.stride(0), A.stride(1), Wt.stride(0), Wt.stride(1), C.stride(0), C.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        processed = processed  # shape [B, L, H]

        # 3) Split back: encoder stream is rows [0:T), image stream is rows [T:L)
        processed_encoder = torch.empty((Bsz, T, H), device=device, dtype=dtype)
        processed_hidden = torch.empty((Bsz, I, H), device=device, dtype=dtype)

        # Copy first T rows
        grid_copy_enc = (Bsz, triton.cdiv(T, 64), triton.cdiv(H, 64))
        copy_rows_kernel[grid_copy_enc](
            processed, processed_encoder,
            Bsz, T, H,
            processed.stride(0), processed.stride(1), processed.stride(2),
            processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
            ROW_START=0, BLOCK_l=64, BLOCK_h=64,
            num_warps=4, num_stages=2,
        )

        # Copy next I rows
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


def run(*args):
    return ModelNew()(*args)
