import torch
import triton
import triton.language as tl


@triton.jit
def concat_kernel(
    e_ptr,        # *const float: [B, T, H]
    i_ptr,        # *const float: [B, I, H]
    out_ptr,      # *float: [B, L, H], where L = T + I
    B: tl.int32, T: tl.int32, I: tl.int32, L: tl.int32, H: tl.int32,
    e_s0, e_s1, e_s2,       # strides for e_ptr
    i_s0, i_s1, i_s2,       # strides for i_ptr
    o_s0, o_s1, o_s2,       # strides for out_ptr
    BLOCK_l: tl.constexpr,  # tile over L
    BLOCK_h: tl.constexpr,  # tile over H
):
    pid_b = tl.program_id(0)
    pid_l = tl.program_id(1)
    pid_h = tl.program_id(2)

    l_idx = pid_l * BLOCK_l + tl.arange(0, BLOCK_l)  # [BLOCK_l]
    h_idx = pid_h * BLOCK_h + tl.arange(0, BLOCK_h)  # [BLOCK_h]

    # Broadcast to 2D
    L2, H2 = tl.meshgrid(l_idx, h_idx)
    mask = (L2 < L) & (H2 < H)

    b = pid_b
    # Select source based on l
    src_mask_e = L2 < T
    src_mask_i = ~(L2 < T)

    # Compute pointers
    # For e_ptr: address = b*e_s0 + l*e_s1 + h*e_s2
    ptr_e = e_ptr + b * e_s0 + L2 * e_s1 + H2 * e_s2
    # For i_ptr: address = b*i_s0 + (l - T)*i_s1 + h*i_s2
    ptr_i = i_ptr + b * i_s0 + (L2 - T) * i_s1 + H2 * i_s2

    # Masked loads: we cannot directly combine since masks are per source
    val_e = tl.load(ptr_e, mask=mask & src_mask_e, other=0.0)
    val_i = tl.load(ptr_i, mask=mask & src_mask_i, other=0.0)
    val = val_e + val_i  # either e or i contributes; the other is zero due to mask

    # Store to out_ptr: address = b*o_s0 + l*o_s1 + h*o_s2
    out_ptr_tile = out_ptr + b * o_s0 + L2 * o_s1 + H2 * o_s2
    tl.store(out_ptr_tile, val, mask=mask)


@triton.jit
def matmul_kernel(
    A_ptr,        # *const float: [M, N] where M = B*L, N = H
    WT_ptr,       # *const float: [K, N] where K = H, N = H (process_weight.T)
    C_ptr,        # *float: [M, N] where M = B*L, N = H
    M: tl.int32, N: tl.int32, K: tl.int32,
    a_s0, a_s1,   # strides for A: row-major => (N, 1)
    wt_s0, wt_s1, # strides for WT: (K, 1) in this case
    c_s0, c_s1,   # strides for C
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        k = k0 + tl.arange(0, BLOCK_K)  # [BLOCK_K]

        # Load A tile [BLOCK_M, BLOCK_K]
        a_ptrs = A_ptr + m[:, None] * a_s0 + k[None, :] * a_s1
        a_mask = (m[:, None] < M) & (k[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # Load WT tile [BLOCK_K, BLOCK_N] (WT is [K, N])
        wt_ptrs = WT_ptr + k[:, None] * wt_s0 + n[None, :] * wt_s1
        wt_mask = (k[:, None] < K) & (n[None, :] < N)
        wt = tl.load(wt_ptrs, mask=wt_mask, other=0.0)

        acc += tl.dot(a, wt)

    # Store C tile
    c_ptrs = C_ptr + m[:, None] * c_s0 + n[None, :] * c_s1
    c_mask = (m[:, None] < M) & (n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


@triton.jit
def copy_rows_kernel(
    X_ptr,        # *const float: input [B, L, H]
    Y_ptr,        # *float: output [B, ROWS, H]
    B: tl.int32, ROWS: tl.int32, L: tl.int32, H: tl.int32,
    X_s0, X_s1, X_s2,       # strides for X
    Y_s0, Y_s1, Y_s2,       # strides for Y
    ROW_START: tl.int32,    # starting row in X to copy into row 0 of Y
    BLOCK_l: tl.constexpr,  # tile over ROWS
    BLOCK_h: tl.constexpr,  # tile over H
):
    pid_b = tl.program_id(0)
    pid_rows = tl.program_id(1)
    pid_h = tl.program_id(2)

    r = pid_rows * BLOCK_rows + tl.arange(0, BLOCK_rows)  # [BLOCK_rows]
    h = pid_h * BLOCK_h + tl.arange(0, BLOCK_h)          # [BLOCK_h]

    R2, H2 = tl.meshgrid(r, h)
    mask = (R2 < ROWS) & (H2 < H)

    # Map r to X rows: src_row = ROW_START + r
    src_row = ROW_START + R2

    # Pointers for X and Y
    x_ptrs = X_ptr + pid_b * X_s0 + src_row * X_s1 + H2 * X_s2
    y_ptrs = Y_ptr + pid_b * Y_s0 + R2 * Y_s1 + H2 * Y_s2

    # Load from X and store to Y
    val = tl.load(x_ptrs, mask=mask, other=0.0)
    tl.store(y_ptrs, val, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        # Shapes
        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        H = hidden_states.shape[2]
        L = T + I

        # Allocate concatenated tensor
        out = torch.empty((B, L, H), device=hidden_states.device, dtype=hidden_states.dtype)

        # Launch concat_kernel
        BLOCK_l = 64
        BLOCK_h = 64
        grid_concat = (B, triton.cdiv(L, BLOCK_l), triton.cdiv(H, BLOCK_h))
        concat_kernel[grid_concat](
            encoder_hidden_states, hidden_states, out,
            B, T, I, L, H,
            encoder_hidden_states.stride(0), encoder_hidden_states.stride(1), encoder_hidden_states.stride(2),
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            out.stride(0), out.stride(1), out.stride(2),
            BLOCK_l=BLOCK_l, BLOCK_h=BLOCK_h,
            num_warps=4, num_stages=2,
        )

        # Matmul: C = out @ process_weight.T
        # out is [B, L, H], process_weight is [H, H]; WT = process_weight.T is [H, H]
        M = B * L
        N = H
        K = H

        # Ensure WT is contiguous [H, H]
        WT = process_weight.t().contiguous()  # [H, H]

        # Allocate C
        C = torch.empty((B, L, H), device=out.device, dtype=out.dtype)

        # Launch matmul_kernel
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 64
        grid_matmul = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        matmul_kernel[grid_matmul](
            out, WT, C,
            M, N, K,
            out.stride(0), out.stride(1),  # A is out viewed as [M, N], so strides are (N, 1)
            WT.stride(0), WT.stride(1),    # WT is [H, H], strides (H, 1)
            C.stride(0), C.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        # Split into encoder and hidden streams
        processed_encoder = torch.empty((B, T, H), device=out.device, dtype=out.dtype)
        processed_hidden = torch.empty((B, I, H), device=out.device, dtype=out.dtype)

        # Launch copy_rows_kernel for encoder part (first T rows)
        BLOCK_rows = 64
        BLOCK_h = 64
        grid_encoder = (B, triton.cdiv(T, BLOCK_rows), triton.cdiv(H, BLOCK_h))
        copy_rows_kernel[grid_encoder](
            C, processed_encoder,
            B, T, L, H,
            C.stride(0), C.stride(1), C.stride(2),
            processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
            ROW_START=0, BLOCK_rows=BLOCK_rows, BLOCK_h=BLOCK_h,
            num_warps=4, num_stages=2,
        )

        # Launch copy_rows_kernel for hidden part (next I rows)
        grid_hidden = (B, triton.cdiv(I, BLOCK_rows), triton.cdiv(H, BLOCK_h))
        copy_rows_kernel[grid_hidden](
            C, processed_hidden,
            B, I, L, H,
            C.stride(0), C.stride(1), C.stride(2),
            processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
            ROW_START=T, BLOCK_rows=BLOCK_rows, BLOCK_h=BLOCK_h,
            num_warps=4, num_stages=2,
        )

        return processed_encoder, processed_hidden