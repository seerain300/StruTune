import torch
import triton
import triton.language as tl


@triton.jit
def build_lower_tri_causal_kernel(
    A_ptr,             # *float32, [B, C, L, H]
    L_ptr,             # *float32, [B, C, 128, 128, H]
    B_size,            # int
    C_size,            # int
    H_size,            # int
    L_len,             # int  (actual chunk_size of hidden_states)
    sl_b, sl_c, sl_l, sl_h,            # strides for A (in elements)
    sl_b_L, sl_c_L, sl_i_L, sl_j_L, sl_h_L  # strides for L (in elements)
):
    b = tl.program_id(0)
    c = tl.program_id(1)
    h = tl.program_id(2)
    i = tl.program_id(3)  # row index in 128
    j = tl.program_id(4)  # col index in 128

    # Compute cumsum of A[b, c, :, h] up to L_len (actual chunk length)
    total = 0.0
    k = 0
    while k < L_len:
        addr = b * sl_b + c * sl_c + k * sl_l + h * sl_h
        a_val = tl.load(A_ptr + addr)
        total += a_val
        k += 1

    # Set L[i, j] = exp(total) if j <= i, else 0.0
    if j <= i:
        l_val = tl.exp(total)
    else:
        l_val = 0.0

    L_addr = b * sl_b_L + c * sl_c_L + i * sl_i_L + j * sl_j_L + h * sl_h_L
    tl.store(L_ptr + L_addr, l_val)


@triton.jit
def expand_groups_repeat_interleave(
    in_ptr,            # *float32, [B, C, L, groups, S]
    out_ptr,           # *float32, [B, C, L, H, S]
    B_size, C_size, L_len, groups, H_size, S_size,
    stride_in_b, stride_in_c, stride_in_l, stride_in_g, stride_in_s,
    stride_out_b, stride_out_c, stride_out_l, stride_out_h, stride_out_s
):
    # program_ids over (B, C, L, H, S)
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)  # chunk position
    h = tl.program_id(3)  # head index
    s = tl.program_id(4)  # state index

    # Map group index g to h: g in [0..groups-1], h in [0..H_size-1]
    # GROUP_EXPAND = 4 means H_size = groups * 4
    g = h // 4

    # Load from in_ptr[b, c, i, g, s]
    in_addr = b * stride_in_b + c * stride_in_c + i * stride_in_l + g * stride_in_g + s * stride_in_s
    val = tl.load(in_ptr + in_addr)

    # Store to out_ptr[b, c, i, h, s]
    out_addr = b * stride_out_b + c * stride_out_c + i * stride_out_l + h * stride_out_h + s * stride_out_s
    tl.store(out_ptr + out_addr, val)


@triton.jit
def g_outer_kernel(
    C_exp_ptr,     # *float32, [B, C, L, H, S]
    B_exp_ptr,     # *float32, [B, C, L, H, S]
    G_ptr,         # *float32, [B, C, L, L, H]
    B_size, C_size, L_len, H_size, S_size,
    sl_b_C, sl_c_C, sl_l_C, sl_h_C, sl_s_C,   # strides for C_exp
    sl_b_B, sl_c_B, sl_l_B, sl_h_B, sl_s_B,   # strides for B_exp
    sl_b_G, sl_c_G, sl_i_G, sl_j_G, sl_h_G    # strides for G
):
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)  # row index
    j = tl.program_id(3)  # col index
    h = tl.program_id(4)  # head index

    g_val = 0.0
    s = 0
    while s < S_size:
        C_addr = b * sl_b_C + c * sl_c_C + i * sl_l_C + h * sl_h_C + s * sl_s_C
        B_addr = b * sl_b_B + c * sl_c_B + j * sl_l_B + h * sl_h_B + s * sl_s_B
        C_val = tl.load(C_exp_ptr + C_addr)
        B_val = tl.load(B_exp_ptr + B_addr)
        g_val += C_val * B_val
        s += 1

    # Store G[b, c, i, j, h] = g_val
    G_addr = b * sl_b_G + c * sl_c_G + i * sl_i_G + j * sl_j_G + h * sl_h_G
    tl.store(G_ptr + G_addr, g_val)


@triton.jit
def y_diag_reduce_kernel(
    M_ptr,           # *float32, [B, C, L, L, H]
    hidden_ptr,      # *float32, [B, C, L, H, D]
    out_ptr,         # *float32, [B, C, L, H, D]
    B_size, C_size, L_len, H_size, D_size,
    sl_b_M, sl_c_M, sl_i_M, sl_j_M, sl_h_M,   # strides for M
    sl_b_h, sl_c_h, sl_l_h, sl_h_h, sl_d_h,   # strides for hidden
    sl_b_out, sl_c_out, sl_i_out, sl_h_out, sl_d_out  # strides for out
):
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)  # chunk position for output i
    h = tl.program_id(3)  # head index

    d = 0
    while d < D_size:
        acc = 0.0
        j = 0
        while j < L_len:
            M_addr = b * sl_b_M + c * sl_c_M + i * sl_i_M + j * sl_j_M + h * sl_h_M
            M_val = tl.load(M_ptr + M_addr)  # scalar
            h_addr = b * sl_b_h + c * sl_c_h + j * sl_l_h + h * sl_h_h + d * sl_d_h
            hidden_val = tl.load(hidden_ptr + h_addr)  # scalar
            acc += M_val * hidden_val
            j += 1
        out_addr = b * sl_b_out + c * sl_c_out + i * sl_i_out + h * sl_h_out + d * sl_d_out
        tl.store(out_ptr + out_addr, acc)
        d += 1


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Constants
        self.NUM_HEADS = 32
        self.N_GROUPS = 8
        self.GROUP_EXPAND = 4
        self.L_mat_size = 128  # as in original code

    def forward(self, hidden_states: torch.Tensor, A_cumsum: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        # Ensure inputs are float32
        hidden_f32 = hidden_states.to(torch.float32)
        A_f32 = A_cumsum.to(torch.float32)
        B_f32 = B.to(torch.float32)
        C_f32 = C.to(torch.float32)

        # Shapes
        batch_size, num_chunks, chunk_size, num_heads, head_dim = hidden_f32.shape
        # Output: Y_diag [B, C, L, H, D] (float32)
        Y = torch.empty((batch_size, num_chunks, chunk_size, num_heads, head_dim), dtype=torch.float32, device=hidden_f32.device)

        # 1) Build L [B, C, 128, 128, H]
        L = torch.empty((batch_size, num_chunks, self.L_mat_size, self.L_mat_size, num_heads), dtype=torch.float32, device=hidden_f32.device)

        # Strides for A and L
        sl_b, sl_c, sl_l, sl_h = A_f32.stride()
        sl_b_L, sl_c_L, sl_i_L, sl_j_L, sl_h_L = L.stride()

        grid_L = (batch_size, num_chunks, num_heads, self.L_mat_size, self.L_mat_size)
        build_lower_tri_causal_kernel[grid_L](
            A_f32, L, batch_size, num_chunks, num_heads, chunk_size,
            sl_b, sl_c, sl_l, sl_h,
            sl_b_L, sl_c_L, sl_i_L, sl_j_L, sl_h_L
        )

        # 2) Expand B and C from groups to heads
        # C_f32: [B, C, L, groups, S]; B_f32 same
        S_size = C_f32.shape[-1]
        B_exp = torch.empty((batch_size, num_chunks, chunk_size, num_heads, S_size), dtype=torch.float32, device=hidden_f32.device)
        C_exp = torch.empty((batch_size, num_chunks, chunk_size, num_heads, S_size), dtype=torch.float32, device=hidden_f32.device)

        stride_in_b, stride_in_c, stride_in_l, stride_in_g, stride_in_s = B_f32.stride()
        stride_out_b, stride_out_c, stride_out_l, stride_out_h, stride_out_s = B_exp.stride()

        grid_expand = (batch_size, num_chunks, chunk_size, num_heads, S_size)
        expand_groups_repeat_interleave[grid_expand](
            B_f32, B_exp, batch_size, num_chunks, chunk_size, self.N_GROUPS, num_heads, S_size,
            stride_in_b, stride_in_c, stride_in_l, stride_in_g, stride_in_s,
            stride_out_b, stride_out_c, stride_out_l, stride_out_h, stride_out_s
        )
        expand_groups_repeat_interleave[grid_expand](
            C_f32, C_exp, batch_size, num_chunks, chunk_size, self.N_GROUPS, num_heads, S_size,
            stride_in_b, stride_in_c, stride_in_l, stride_in_g, stride_in_s,
            stride_out_b, stride_out_c, stride_out_l, stride_out_h, stride_out_s
        )

        # 3) Compute G [B, C, L, L, H]
        G = torch.empty((batch_size, num_chunks, chunk_size, chunk_size, num_heads), dtype=torch.float32, device=hidden_f32.device)

        sl_b_C, sl_c_C, sl_l_C, sl_h_C, sl_s_C = C_exp.stride()
        sl_b_B, sl_c_B, sl_l_B, sl_h_B, sl_s_B = B_exp.stride()
        sl_b_G, sl_c_G, sl_i_G, sl_j_G, sl_h_G = G.stride()

        grid_G = (batch_size, num_chunks, chunk_size, chunk_size, num_heads)
        g_outer_kernel[grid_G](
            C_exp, B_exp, G,
            batch_size, num_chunks, chunk_size, num_heads, S_size,
            sl_b_C, sl_c_C, sl_l_C, sl_h_C, sl_s_C,
            sl_b_B, sl_c_B, sl_l_B, sl_h_B, sl_s_B,
            sl_b_G, sl_c_G, sl_i_G, sl_j_G, sl_h_G
        )

        # 4) Apply mask: M = G * L (element-wise). Permute L to match G: [B, C, L, L, H]
        # L already has shape [B, C, 128, 128, H]; we need to index using chunk_size as L dimension.
        # The original PyTorch code applies element-wise multiply with broadcasting: G: [B,C,L,L,H] and L: [B,C,L,L,H]
        # Here, G and L have the same shape, so element-wise multiply is fine.
        M = torch.empty_like(G)
        # We can do element-wise multiply directly in Triton by reading L and multiplying with G, but to avoid extra kernel,
        # we can use PyTorch for this simple elementwise multiply. However, to adhere to "TRITON-ONLY", we implement it in Triton.
        # Since we have G and L tensors, we can perform element-wise multiply using PyTorch; but to satisfy Triton-only, we'll implement
        # a simple Triton elementwise kernel.
        # Create a 1D grid over the flattened number of elements to perform elementwise multiply.
        total_elems = G.numel()
        grid_mul = (1,)
        # Elementwise multiply G * L
        # We need a kernel that reads G and L and writes to M.
        @triton.jit
        def elemwise_mul_kernel(G_ptr, L_ptr, M_ptr, num_elems):
            idx = tl.program_id(0)
            # compute linear index into flattened arrays
            if idx < num_elems:
                val_G = tl.load(G_ptr + idx)
                val_L = tl.load(L_ptr + idx)
                val_M = val_G * val_L
                tl.store(M_ptr + idx, val_M)
        M = torch.empty_like(G)  # M has same shape as G
        # Flatten G and L for elementwise multiply (Triton expects contiguous pointers)
        G_flat = G.view(-1)
        L_flat = L.view(-1)
        M_flat = M.view(-1)
        elemwise_mul_kernel[grid_mul](G_flat, L_flat, M_flat, total_elems)

        # 5) Compute Y_diag by contraction over j: Y[b, c, i, h, d] = sum_j M[b, c, i, j, h] * hidden_states[b, c, j, h, d]
        # hidden_f32: [B, C, L, H, D]
        Y = torch.empty((batch_size, num_chunks, chunk_size, num_heads, head_dim), dtype=torch.float32, device=hidden_f32.device)
        sl_b_M, sl_c_M, sl_i_M, sl_j_M, sl_h_M = M.stride()
        sl_b_h, sl_c_h, sl_l_h, sl_h_h, sl_d_h = hidden_f32.stride()
        sl_b_out, sl_c_out, sl_i_out, sl_h_out, sl_d_out = Y.stride()

        grid_reduce = (batch_size, num_chunks, chunk_size, num_heads)
        y_diag_reduce_kernel[grid_reduce](
            M, hidden_f32, Y,
            batch_size, num_chunks, chunk_size, num_heads, head_dim,
            sl_b_M, sl_c_M, sl_i_M, sl_j_M, sl_h_M,
            sl_b_h, sl_c_h, sl_l_h, sl_h_h, sl_d_h,
            sl_b_out, sl_c_out, sl_i_out, sl_h_out, sl_d_out
        )

        # Return in bfloat16 (matching original run signature which converts to bfloat16)
        return Y.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
