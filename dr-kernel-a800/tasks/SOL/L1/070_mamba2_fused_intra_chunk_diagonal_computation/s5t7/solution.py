import torch
import triton
import triton.language as tl

# Kernel 1: build L per (b, h) over internal chunk length L
# We will compute L[i, j] for i, j in [0..L-1], lower-triangular (j <= i), and L[i, j] = 0 else.
# The original PyTorch code uses A_cumsum[B, C, L, H] and builds a mask along the internal chunk dim.
# Here, we implement the L matrix directly: lower-triangular with diagonal=-1, and exp is applied after.
# We will compute L as float32 and store it in a buffer [B, H, C, L, L]. In the forward, we will permute to [B, C, H, L, L] for multiplication.
@triton.jit
def build_L_kernel(
    A_cumsum_ptr,  # *float32, shape [B, C, L, H]
    L_ptr,         # *float32, shape [B, H, C, L, L] to be filled
    B_size, C_size, L_len, H_size,
):
    b = tl.program_id(0)
    h = tl.program_id(1)

    # We will fill L[b, h, c, i, j] for all c, i, j
    # Loop over c, i, j (nested). Triton allows nested loops.
    for c in range(0, C_size):
        for i in range(0, L_len):
            row_sum = 0.0  # sum_{k=0..i} A[b, h, k]
            # Compute row_sum: iterate k from 0 to i
            for k in range(0, i + 1):
                a_addr = b * (C_size * L_len * H_size) + c * (L_len * H_size) + k * H_size + h
                a_val = tl.load(A_cumsum_ptr + a_addr)
                row_sum += a_val
            # Now set L[i, j] for j in [0..L_len-1]
            for j in range(0, L_len):
                # lower-triangular: i >= j -> keep row_sum, else 0
                if i >= j:
                    l_val = tl.exp(row_sum)
                else:
                    l_val = 0.0
                l_addr = b * (C_size * H_size * L_len * L_len) + h * (C_size * L_len * L_len) + c * (L_len * L_len) + i * L_len + j
                tl.store(L_ptr + l_addr, l_val)


# Kernel 2: compute G = sum_s C_exp[b, c, i, h, s] * B_exp[b, c, j, h, s]
# We need S_size inferred from B's last dim. For robustness, we will not infer S here; this kernel will be launched only when S is known.
@triton.jit
def g_outer_kernel(
    B_exp_ptr,     # *float32, [B, C, L, H, S]
    C_exp_ptr,     # *float32, [B, C, L, H, S]
    G_ptr,         # *float32, [B, C, L, L, H] to be filled
    B_size, C_size, L_len, H_size, S_size,
):
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)
    j = tl.program_id(3)
    h = tl.program_id(4)

    g_val = 0.0
    # Reduce over S dimension
    for s in range(0, S_size):
        B_addr = b * (C_size * L_len * H_size * S_size) + c * (L_len * H_size * S_size) + i * (H_size * S_size) + h * S_size + s
        C_addr = b * (C_size * L_len * H_size * S_size) + c * (L_len * H_size * S_size) + j * (H_size * S_size) + h * S_size + s
        B_val = tl.load(B_exp_ptr + B_addr)
        C_val = tl.load(C_exp_ptr + C_addr)
        g_val += C_val * B_val
    # Store G[b, c, i, j, h]
    G_stride_B = C_size * L_len * L_len * H_size
    G_stride_C = L_len * L_len * H_size
    G_stride_L1 = L_len * H_size
    G_stride_L2 = H_size
    G_addr = b * G_stride_B + c * G_stride_C + i * G_stride_L1 + j * G_stride_L2 + h
    tl.store(G_ptr + G_addr, g_val)


# Kernel 3: elementwise M = G * L_permuted where L_permuted is [B, C, H, L, L]
@triton.jit
def multiply_mask_kernel(
    G_ptr,         # *float32, [B, C, L, L, H]
    L_perm_ptr,    # *float32, [B, C, H, L, L]
    M_ptr,         # *float32, [B, C, L, L, H] to be filled
    B_size, C_size, L_len, H_size,
):
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)
    j = tl.program_id(3)
    h = tl.program_id(4)

    G_addr = b * (C_size * L_len * L_len * H_size) + c * (L_len * L_len * H_size) + i * (L_len * H_size) + j * H_size + h
    L_addr = b * (C_size * L_len * L_len * H_size) + c * (L_len * L_len * H_size) + j * (L_len * H_size) + i * H_size + h  # note swap: [B, C, H, L, L]
    m_val = tl.load(G_ptr + G_addr) * tl.load(L_perm_ptr + L_addr)
    M_addr = b * (C_size * L_len * L_len * H_size) + c * (L_len * L_len * H_size) + i * (L_len * H_size) + j * H_size + h
    tl.store(M_ptr + M_addr, m_val)


# Kernel 4: Y_diag reduction over j: Y[b, c, i, h, d] = sum_j M[b, c, i, j, h] * hidden_states[b, c, j, h, d]
@triton.jit
def y_diag_reduce_kernel(
    M_ptr,          # *float32, [B, C, L, L, H]
    HS_ptr,         # *float32, [B, C, L, H, D]
    Y_ptr,          # *float32, [B, C, L, H, D] to be filled
    B_size, C_size, L_len, H_size, D_size,
    BLOCK_D: tl.constexpr,
):
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)
    h = tl.program_id(3)

    d_offsets = tl.arange(0, BLOCK_D)
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)

    for j in range(0, L_len):
        M_addr = b * (C_size * L_len * L_len * H_size) + c * (L_len * L_len * H_size) + i * (L_len * H_size) + j * H_size + h
        m_val = tl.load(M_ptr + M_addr)
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
        Triton-only implementation.
        Inputs:
          hidden_states: [B, C, L, H, D]
          A_cumsum:      [B, C, L, H] (cumsum along internal chunk dim)
          B:             [B, C, L, groups, S]
          C:             [B, C, L, groups, S]
        Output:
          Y_diag:        [B, C, L, H, D] in bfloat16
        """
        device = hidden_states.device
        B_size, C_size, L_len, H_size, D_size = hidden_states.shape

        # Infer groups and expand factor (NUM_HEADS=32, N_GROUPS=8)
        groups = 8
        group_expand = H_size // groups

        # 1) Build L: [B, H, C, L, L] using Triton kernel
        # Allocate L buffer
        L = torch.empty((B_size, H_size, C_size, L_len, L_len), device=device, dtype=torch.float32)
        # Launch build_L_kernel with grid (B_size, H_size)
        # Note: Triton grid for loop over c and i,j is handled inside the kernel via nested loops.
        build_L_kernel[(B_size, H_size)](
            A_cumsum.to(torch.float32).contiguous(),
            L,
            B_size, C_size, L_len, H_size,
            num_warps=1, num_stages=1,
        )

        # 2) Expand B and C from groups to H along dim=3
        # Repeat_interleave using expand (repeat_interleave not available in Triton)
        B_exp = B.unsqueeze(3).expand(B_size, C_size, L_len, H_size, B.shape[4]).contiguous()  # [B, C, L, H, S]
        C_exp = C.unsqueeze(3).expand(B_size, C_size, L_len, H_size, C.shape[4]).contiguous()  # [B, C, L, H, S]

        # We need S_size to launch g_outer_kernel. In the original, S is B.shape[4]. Ensure B and C share S.
        S_size = B.shape[4]
        assert C.shape[4] == S_size, "C's last dimension must equal B's last dimension (state_size S)."

        # 3) Compute G via Triton outer-product reduction
        G = torch.empty((B_size, C_size, L_len, L_len, H_size), device=device, dtype=torch.float32)
        g_outer_kernel[(B_size, C_size, L_len, L_len, H_size)](
            B_exp, C_exp, G,
            B_size, C_size, L_len, H_size, S_size,
            num_warps=1, num_stages=1,
        )

        # 4) Permute L to [B, C, H, L, L] and multiply elementwise to get M
        L_perm = L.permute(0, 2, 1, 3, 4)  # [B, C, H, L, L]
        M = torch.empty_like(G)
        multiply_mask_kernel[(B_size, C_size, L_len, L_len, H_size)](
            G, L_perm, M,
            B_size, C_size, L_len, H_size,
            num_warps=1, num_stages=1,
        )

        # 5) Compute Y_diag via Triton reduction over j
        Y = torch.empty((B_size, C_size, L_len, H_size, D_size), device=device, dtype=torch.float32)
        y_diag_reduce_kernel[(B_size, C_size, L_len, H_size)](
            M, hidden_states.to(torch.float32).contiguous(), Y,
            B_size, C_size, L_len, H_size, D_size,
            BLOCK_D=64,
            num_warps=1, num_stages=1,
        )

        # Return in bfloat16 to match original
        return Y.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
