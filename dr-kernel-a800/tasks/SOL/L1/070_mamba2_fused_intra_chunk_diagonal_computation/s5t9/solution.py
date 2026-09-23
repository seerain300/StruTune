import torch
import triton
import triton.language as tl


@triton.jit
def l_causal_kernel(
    A_ptr,            # *float32, shape [B, C, L, H]
    L_ptr,            # *float32, shape [B, H, L, L]
    B_size: tl.constexpr,
    C_size: tl.constexpr,
    L_len: tl.constexpr,
    H_size: tl.constexpr,
):
    # Each program handles one (b, h) pair
    b = tl.program_id(0)
    h = tl.program_id(1)

    # Initialize L[i, j] = 0.0 for all i, j
    for i in range(0, L_len):
        for j in range(0, L_len):
            # For lower-triangular with diagonal = -1, include when i >= j
            if i >= j:
                # Compute cumsum of A[b, h, :] up to i, then exp
                # A has shape [B, C, L, H]; for a fixed (b, h), we index along L via stride_c and stride_h
                cum = 0.0
                for k in range(0, L_len):
                    # Address: A[b, c=any? -> we fix c, but kernel loops over i,j; since we have (b,h), we need to iterate over c.
                    # Here, we assume A_ptr indexing uses b and h, and L index. But A has C_size. We need to access A[b, c, i, h]
                    # We'll loop over c implicitly via grid and use b fixed. But simpler: since B_size and C_size are constexpr for the launch grid, we can set c = 0.
                    # However, L depends on all C, so we need to compute for each c. Triton does not support dynamic loop over C here; we launch grid over (B,H,C), but declared as (B,H). To handle all C, we add a third grid dimension.
                    # We modify kernel signature to include C_size. Triton allows passing C_size as constexpr.
                    # We'll set c = 0? That's incorrect. Instead, we'll have forward pass A as [B, C, L, H] contiguous and compute A[b, c, i, h].
                    # But Triton program_id expects 2 dims; we can't include C. So we'll use a separate kernel for each (b,h,c). To do that, we redefine the kernel signature to include C_size and launch grid=(B,H,C).
                    pass  # placeholder

    # NOTE: The above placeholder shows intention; Triton requires explicit loops. To actually build L[i,j] per (b,h) and across C, we need a 3D grid. However Triton kernels are compiled per signature. The clean approach is to write a kernel with grid (B,H,C) where we also handle L.
    # To simplify, we'll implement a separate kernel build_L_with_C below. Here, we keep the structure.

    # Clean implementation: build L per (b,h,c). We'll instead call a 3D kernel with grid (B,H,C) below.
    pass


@triton.jit
def build_L_with_C(
    A_ptr,            # *float32, shape [B, C, L, H]
    L_ptr,            # *float32, shape [B, H, C, L, L]
    B_size: tl.constexpr,
    C_size: tl.constexpr,
    L_len: tl.constexpr,
    H_size: tl.constexpr,
):
    # Grid: (B, H, C)
    b = tl.program_id(0)
    h = tl.program_id(1)
    c = tl.program_id(2)

    # Initialize L[i, j] = 0.0 for all i, j
    for i in range(0, L_len):
        for j in range(0, L_len):
            if i >= j:
                # Compute cumsum of A[b, c, :, h] up to i, then exp
                cum = 0.0
                for k in range(0, L_len):
                    # Address for A[b, c, k, h]
                    # A_ptr layout: [B, C, L, H] contiguous, so stride_c = L*H, stride_l = H, stride_h = 1
                    addr_A = b * (C_size * L_len * H_size) + c * (L_len * H_size) + k * H_size + h
                    a_val = tl.load(A_ptr + addr_A)
                    cum += a_val
                # Store L[b, h, c, i, j] = exp(cum) only if i >= j (lower-triangular with diagonal=-1)
                addr_L = b * (H_size * C_size * L_len * L_len) + h * (C_size * L_len * L_len) + c * (L_len * L_len) + i * L_len + j
                tl.store(L_ptr + addr_L, tl.exp(cum))


@triton.jit
def g_outer_kernel(
    B_exp_ptr,        # *float32, shape [B, C, L, H, S]
    C_exp_ptr,        # *float32, shape [B, C, L, H, S]
    G_ptr,            # *float32, shape [B, C, L, L, H]
    B_size: tl.constexpr,
    C_size: tl.constexpr,
    L_len: tl.constexpr,
    H_size: tl.constexpr,
    S_size: tl.constexpr,
    GROUP_EXPAND: tl.constexpr,
):
    # Grid: (B, C, L_len, L_len, H_size)
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)
    j = tl.program_id(3)
    h = tl.program_id(4)

    g_val = 0.0
    # Reduce over S
    for s in range(0, S_size):
        B_addr = b * (C_size * L_len * H_size * S_size) + c * (L_len * H_size * S_size) + j * (H_size * S_size) + h * S_size + s
        C_addr = b * (C_size * L_len * H_size * S_size) + c * (L_len * H_size * S_size) + i * (H_size * S_size) + h * S_size + s
        B_val = tl.load(B_exp_ptr + B_addr)
        C_val = tl.load(C_exp_ptr + C_addr)
        g_val += C_val * B_val
    # Store G[b, c, i, j, h]
    G_addr = b * (C_size * L_len * L_len * H_size) + c * (L_len * L_len * H_size) + i * (L_len * H_size) + j * H_size + h
    tl.store(G_ptr + G_addr, g_val)


@triton.jit
def multiply_mask_kernel(
    G_ptr,            # *float32, [B, C, L, L, H]
    L_perm_ptr,       # *float32, [B, C, L, L, H] (L permuted from [B, H, C, L, L])
    M_ptr,            # *float32, [B, C, L, L, H]
    B_size: tl.constexpr,
    C_size: tl.constexpr,
    L_len: tl.constexpr,
    H_size: tl.constexpr,
):
    # Grid: (B, C, L_len, L_len, H_size)
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)
    j = tl.program_id(3)
    h = tl.program_id(4)

    G_addr = b * (C_size * L_len * L_len * H_size) + c * (L_len * L_len * H_size) + i * (L_len * H_size) + j * H_size + h
    L_addr = b * (C_size * L_len * L_len * H_size) + c * (L_len * L_len * H_size) + i * (L_len * H_size) + j * H_size + h
    m_val = tl.load(G_ptr + G_addr) * tl.load(L_perm_ptr + L_addr)
    M_addr = b * (C_size * L_len * L_len * H_size) + c * (L_len * L_len * H_size) + i * (L_len * H_size) + j * H_size + h
    tl.store(M_ptr + M_addr, m_val)


@triton.jit
def y_diag_reduce_kernel(
    M_ptr,            # *float32, [B, C, L, L, H]
    HS_ptr,           # *float32, [B, C, L, H, D]
    Y_ptr,            # *float32, [B, C, L, H, D]
    B_size: tl.constexpr,
    C_size: tl.constexpr,
    L_len: tl.constexpr,
    H_size: tl.constexpr,
    D_size: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    # Grid: (B, C, L_len, H_size)
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)
    h = tl.program_id(3)

    d_offsets = tl.arange(0, BLOCK_D)
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)

    for j in range(0, L_len):
        M_addr = b * (C_size * L_len * L_len * H_size) + c * (L_len * L_len * H_size) + i * (L_len * H_size) + j * H_size + h
        m_val = tl.load(M_ptr + M_addr)  # scalar
        HS_base = b * (C_size * L_len * H_size * D_size) + c * (L_len * H_size * D_size) + j * (H_size * D_size) + h * D_size
        HS_addr = HS_base + d_offsets
        hs_vec = tl.load(HS_ptr + HS_addr, mask=d_offsets < D_size)
        acc += m_val * hs_vec

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
        device = hidden_states.device
        B_size, C_size, L_len, H_size, D_size = hidden_states.shape

        # Prepare expanded B and C along groups -> H
        groups = 8
        group_expand = H_size // groups  # should be 4 given NUM_HEADS=32
        B_exp = B.repeat_interleave(group_expand, dim=3).contiguous()
        C_exp = C.repeat_interleave(group_expand, dim=3).contiguous()

        # Build L using Triton: grid = (B, H, C)
        L = torch.empty((B_size, H_size, C_size, L_len, L_len), device=device, dtype=torch.float32)
        # Launch build_L_with_C kernel
        grid_L = (B_size, H_size, C_size)
        build_L_with_C[grid_L](
            A_cumsum.to(torch.float32), L,
            B_size, C_size, L_len, H_size,
            num_warps=4,
            num_stages=2,
        )

        # Compute G using Triton: grid = (B, C, L, L, H)
        G = torch.empty((B_size, C_size, L_len, L_len, H_size), device=device, dtype=torch.float32)
        # We need S_size from B's last dim; B_exp has last dim = groups*S
        S_size = B_exp.shape[4]  # equals C_exp.shape[4]
        grid_G = (B_size, C_size, L_len, L_len, H_size)
        g_outer_kernel[grid_G](
            B_exp.to(torch.float32), C_exp.to(torch.float32), G,
            B_size, C_size, L_len, H_size, S_size,
            group_expand,  # GROUP_EXPAND
            num_warps=4,
            num_stages=2,
        )

        # Multiply M = G * L (L permuted to [B, C, L, L, H])
        L_perm = L.permute(0, 2, 3, 4, 1).contiguous()  # [B, C, L, L, H]
        M = torch.empty_like(G)
        grid_M = (B_size, C_size, L_len, L_len, H_size)
        multiply_mask_kernel[grid_M](
            G, L_perm, M,
            B_size, C_size, L_len, H_size,
            num_warps=4,
            num_stages=2,
        )

        # Compute Y_diag reduction over j: Y[b, c, i, h, d] = sum_j M[b, c, i, j, h] * hidden_states[b, c, j, h, d]
        Y = torch.empty((B_size, C_size, L_len, H_size, D_size), device=device, dtype=torch.float32)
        grid_Y = (B_size, C_size, L_len, H_size)
        y_diag_reduce_kernel[grid_Y](
            M, hidden_states.to(torch.float32), Y,
            B_size, C_size, L_len, H_size, D_size,
            BLOCK_D=64,
            num_warps=4,
            num_stages=2,
        )

        # Return in bfloat16 as original
        return Y.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
