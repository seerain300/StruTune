import torch
import triton
import triton.language as tl


# Kernel 1: Build L_mat[b, h, c, i, j] = exp(cumsum(A_cumsum[b, :, :, h])) with lower-triangular mask (diagonal = -1)
# We process per (b, h). For each c, we build L for all i,j in [0..L-1]. L is stored as [B, H, C, L, L] float32.
@triton.jit
def build_L_mat_kernel(
    A_ptr,          # *float32, [B, C, L, H]
    L_ptr,          # *float32, [B, H, C, L, L] (expanded view will be used as [B, 1, C, L, L] via grid)
    B_size,         # int
    C_size,         # int
    L_len,          # int
    H_size,         # int
    groups,          # int (N_GROUPS)
    group_expand,    # int (H_size // groups) expected to be 4
    BLOCK_I: tl.constexpr,  # chunk size along i (rows)
    BLOCK_J: tl.constexpr,  # chunk size along j (cols)
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    # We need c index in Triton; pass it via program_id(2). To make things simple, launch grid over (B, H, C).
    # But Triton only allows 3 program_ids. We'll emulate by using grid as (B, H, C) directly.
    # In Triton, we can't get 4th pid, so we pass c separately. However, we can compute c from grid dimension by mapping:
    # We will use grid = (B, H, C). Hence, program_id(2) gives c.
    c = tl.program_id(2)

    # Accumulator for row sum s_i = sum_{k=0..i} A[b, c, k, h]
    # We will loop over i in chunks of BLOCK_I, and for each i, compute s_i.
    # Initialize s vector of size BLOCK_I, then for each i in chunk, compute sum over k=0..i.
    # For simplicity, we'll do scalar i loop; given typical L are small (workloads show 1..8), this is fine.

    # We will iterate i from 0 to L_len and set L[i, j] = exp(s_i) for j <= i.
    # We'll use nested loops in Triton: outer over i, inner over j.
    # To stay within Triton's loop limits, we use while loops.

    i = 0
    while i < L_len:
        # Compute sum s_i = sum_{k=0..i} A[b, c, k, h]
        s = 0.0
        k = 0
        while k <= i:
            # A_ptr has layout [B, C, L, H]; contiguous with strides (C*L*H, L*H, H, 1)
            # address = A_ptr + b*stride_b + c*stride_c + k*stride_l + h*stride_h
            # But since we passed A as [B, C, L, H], we can index as:
            a_val = tl.load(A_ptr + b * (C_size * L_len * H_size) + c * (L_len * H_size) + k * H_size + h, mask=k < L_len, other=0.0)
            s += a_val
            k += 1

        # Now set L[i, j] = exp(s) for j <= i
        j = 0
        while j < L_len:
            # If j > i, set 0 (upper triangle zero in mask), but cumsum already zeros it. Here we just skip storing.
            if j <= i:
                # L_ptr is [B, H, C, L, L]. Address = B*H*C*L*L + H*C*L*L + C*L*L + i*L + j
                # We need to compute base: offset for (b,h,c) = b*(H*C*L*L) + h*(C*L*L)
                L_base = b * (H_size * C_size * L_len * L_len) + h * (C_size * L_len * L_len) + c * (L_len * L_len)
                addr = L_base + i * L_len + j
                # Store exp(s)
                tl.store(L_ptr + addr, tl.exp(s))
            j += 1
        i += 1


# Kernel 2: Compute G[i, j, h] = sum_s C_expanded[b, c, i, h, s] * B_expanded[b, c, j, h, s]
# Inputs: B_expanded: [B, C, L, H, S], C_expanded: [B, C, L, H, S]
@triton.jit
def compute_G_kernel(
    B_exp_ptr,      # *float32, [B, C, L, H, S]
    C_exp_ptr,      # *float32, [B, C, L, H, S]
    G_ptr,          # *float32, [B, C, L, L, H]
    B_size,         # int
    C_size,         # int
    L_len,          # int
    H_size,         # int
    S_size,         # int
    groups,          # int
    group_expand,    # int
    BLOCK_I: tl.constexpr,
    BLOCK_J: tl.constexpr,
):
    b = tl.program_id(0)
    c = tl.program_id(1)
    h = tl.program_id(2)

    # Loop over i (rows) and j (cols) chunks
    i = 0
    while i < L_len:
        j = 0
        while j < L_len:
            # Accumulate G[i, j, h] over s
            g_val = 0.0
            s = 0
            while s < S_size:
                # Load B_exp[b, c, j, h, s], C_exp[b, c, i, h, s]
                # Addressing:
                # B_exp[b, c, j, h, s] -> base + b*stride_b + c*stride_c + j*stride_j + h*stride_h + s*stride_s
                # Strides: (C*L*H*S, L*H*S, H*S, S, 1)
                stride_s = 1
                stride_h = S_size
                stride_j = H_size * S_size
                stride_c = L_len * H_size * S_size
                stride_b = C_size * L_len * H_size * S_size

                B_addr = b * stride_b + c * stride_c + j * stride_j + h * stride_h + s * stride_s
                C_addr = b * stride_b + c * stride_c + i * stride_j + h * stride_h + s * stride_s

                B_val = tl.load(B_exp_ptr + B_addr)
                C_val = tl.load(C_exp_ptr + C_addr)
                g_val += C_val * B_val
                s += 1

            # Store G[b, c, i, j, h] in float32
            # G layout [B, C, L, L, H] with strides
            G_stride_H = 1
            G_stride_L = H_size
            G_stride_C = L_len * H_size
            G_stride_B = C_size * L_len * H_size

            G_addr = b * G_stride_B + c * G_stride_C + i * L_len + j * G_stride_L + h * G_stride_H
            tl.store(G_ptr + G_addr, g_val)
            j += 1
        i += 1


# Kernel 3: Compute Y_diag[b, c, i, h, d] = sum_j M[b, c, i, j, h] * hidden_states[b, c, j, h, d]
# hidden_states: [B, C, L, H, D] (we'll pass D as meta)
@triton.jit
def y_diag_reduce_kernel(
    M_ptr,          # *float32, [B, C, L, L, H]
    HS_ptr,         # *float32, [B, C, L, H, D]
    Y_ptr,          # *float32, [B, C, L, H, D]
    B_size,         # int
    C_size,         # int
    L_len,          # int
    H_size,         # int
    D_size,         # int
    groups,          # int
    group_expand,    # int
    BLOCK_D: tl.constexpr,
):
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)
    h = tl.program_id(3)

    # Accumulator for each d
    d_offsets = tl.arange(0, BLOCK_D)
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)

    # Loop over j
    j = 0
    while j < L_len:
        # Load M[b, c, i, j, h]
        M_addr = b * (C_size * L_len * L_len * H_size) + c * (L_len * L_len * H_size) + i * (L_len * H_size) + j * H_size + h
        m_val = tl.load(M_ptr + M_addr)
        # Load HS[b, c, j, h, d_offsets]
        HS_base = b * (C_size * L_len * H_size * D_size) + c * (L_len * H_size * D_size) + j * (H_size * D_size) + h * D_size
        HS_addr = HS_base + d_offsets
        hs_vec = tl.load(HS_ptr + HS_addr, mask=d_offsets < D_size, other=0.0)
        acc += m_val * hs_vec
        j += 1

    # Store acc to Y[b, c, i, h, d_offsets]
    Y_base = b * (C_size * L_len * H_size * D_size) + c * (L_len * H_size * D_size) + i * (H_size * D_size) + h * D_size
    tl.store(Y_ptr + Y_base + d_offsets, acc, mask=d_offsets < D_size)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                A_cumsum: torch.Tensor,
                B: torch.Tensor,
                C: torch.Tensor) -> torch.Tensor:
        """
        hidden_states: [B, C, L, H, D]
        A_cumsum:      [B, C, L, H]
        B:             [B, C, L, groups, S]
        C:             [B, C, L, groups, S]
        Returns:       [B, C, L, H, D] in bfloat16
        """
        B_size, C_size, L_len, H_size, D_size = hidden_states.shape
        assert A_cumsum.shape == (B_size, C_size, L_len, H_size), "A_cumsum shape mismatch"
        groups = 8  # N_GROUPS
        group_expand = H_size // groups
        assert group_expand == 4, "H must be 4 times N_GROUPS (NUM_HEADS=32, N_GROUPS=8 -> 4)"

        # Prepare outputs and buffers
        # We will perform computations in float32 and cast at the end.

        # Kernel 1: build L_mat[b, h, c, i, j] (note: original expands to [B, H, C, L, L]; we create it here)
        # Allocate L_mat in float32: [B, H, C, L, L]
        L_mat = torch.empty((B_size, H_size, C_size, L_len, L_len), dtype=torch.float32, device=hidden_states.device)

        # Launch build_L_mat_kernel: grid = (B, H, C)
        build_L_mat_kernel[(B_size, H_size, C_size)](
            A_cumsum, L_mat, B_size, C_size, L_len, H_size, groups, group_expand,
            BLOCK_I=1, BLOCK_J=1  # simple loops suffice given small L
        )

        # Kernel 2: compute B_expanded and C_expanded (host-side expansion), then G[b, c, i, j, h]
        # Create B_expanded and C_expanded by repeating along groups dimension.
        B_expanded = B.repeat_interleave(group_expand, dim=3)  # [B, C, L, H, S]
        C_expanded = C.repeat_interleave(group_expand, dim=3)  # [B, C, L, H, S]
        # Ensure contiguous
        B_expanded = B_expanded.contiguous()
        C_expanded = C_expanded.contiguous()

        G = torch.empty((B_size, C_size, L_len, L_len, H_size), dtype=torch.float32, device=hidden_states.device)

        compute_G_kernel[(B_size, C_size, H_size)](
            B_expanded, C_expanded, G,
            B_size, C_size, L_len, H_size, B_expanded.shape[4], groups, group_expand,
            BLOCK_I=1, BLOCK_J=1
        )

        # Compute M = G * L_permuted. L_permuted has shape [B, C, L, L, H] (permute dims 1->2, 2->1 for G)
        # Since G is [B, C, L, L, H] and L_mat is [B, H, C, L, L], we can broadcast multiply:
        # M = G * L_mat.permute(0, 2, 1, 3, 4) -> [B, C, L, L, H]
        M = G * L_mat.permute(0, 2, 1, 3, 4)

        # Kernel 3: compute Y_diag by reduction over j
        # Y: [B, C, L, H, D], float32
        Y = torch.empty((B_size, C_size, L_len, H_size, D_size), dtype=torch.float32, device=hidden_states.device)
        # Ensure hidden_states and Y are float32 for kernel
        HS_f32 = hidden_states.to(torch.float32)

        y_diag_reduce_kernel[(B_size, C_size, L_len, H_size)](
            M, HS_f32, Y,
            B_size, C_size, L_len, H_size, D_size, groups, group_expand,
            BLOCK_D=64  # D_size in provided workloads is 1 (too small). Using 64 covers it; mask handles remainder.
        )

        # Return in bfloat16, matching original function
        return Y.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
