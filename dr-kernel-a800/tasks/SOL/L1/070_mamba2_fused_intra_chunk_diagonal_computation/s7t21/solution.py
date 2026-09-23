import torch
import triton
import triton.language as tl


# Triton kernel 1: compute L = exp(masked_cumsum(A, dim=-2)) with lower-triangular mask (diagonal=0)
# A: [B, Csz, S, N_groups] float32
# L: [B, Csz, S, S, N_groups] float32
@triton.jit
def masked_cumsum_exp_kernel(
    A_ptr, L_ptr,
    B: tl.constexpr, Csz: tl.constexpr, S: tl.constexpr, N_groups: tl.constexpr,
    A_stride_b, A_stride_c, A_stride_i, A_stride_ng,
    L_stride_b, L_stride_c, L_stride_i, L_stride_j, L_stride_ng,
):
    # program ids
    b = tl.program_id(0)
    c = tl.program_id(1)
    n = tl.program_id(2)  # n_groups index
    i = tl.program_id(3)  # i in [0, S)

    # running sum across j from 0..S-1; include j <= i (diagonal=0 => include j==i)
    run_sum = 0.0
    for j in range(0, S):
        # load A[b, c, j, n]
        a_ptr = A_ptr + b * A_stride_b + c * A_stride_c + j * A_stride_i + n * A_stride_ng
        a_val = tl.load(a_ptr)
        # apply lower-triangular mask: include when j <= i
        include = j <= i
        run_sum = tl.where(include, run_sum + a_val, run_sum)
        # store exp(cumsum) to L[b, c, i, j, n]
        l_ptr = L_ptr + b * L_stride_b + c * L_stride_c + i * L_stride_i + j * L_stride_j + n * L_stride_ng
        tl.store(l_ptr, tl.exp(run_sum))


# Triton kernel 2: contract B_expanded and C_expanded to G[b, c, i, j, n] = sum_k C[b,c,i,n,k] * B[b,c,j,n,k]
# B_expanded: [B, Csz, S, N, D], C_expanded: [B, Csz, S, N, D]
@triton.jit
def contract_BC_to_G_kernel(
    B_ptr, C_ptr, G_ptr,
    Bsz: tl.constexpr, Csz: tl.constexpr, S: tl.constexpr, N: tl.constexpr, D: tl.constexpr,
    B_stride_b, B_stride_c, B_stride_j, B_stride_n, B_stride_d,
    C_stride_b, C_stride_c, C_stride_i, C_stride_n, C_stride_d,
    G_stride_b, G_stride_c, G_stride_i, G_stride_j, G_stride_n,
):
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)
    j = tl.program_id(3)
    n = tl.program_id(4)

    # accumulate over k from 0..D-1
    acc = 0.0
    for k in range(0, D):
        b_ptr = B_ptr + b * B_stride_b + c * B_stride_c + j * B_stride_j + n * B_stride_n + k * B_stride_d
        c_ptr = C_ptr + b * C_stride_b + c * C_stride_c + i * C_stride_i + n * C_stride_n + k * C_stride_d
        b_val = tl.load(b_ptr)
        c_val = tl.load(c_ptr)
        acc += b_val * c_val

    # store G[b, c, i, j, n]
    g_ptr = G_ptr + b * G_stride_b + c * G_stride_c + i * G_stride_i + j * G_stride_j + n * G_stride_n
    tl.store(g_ptr, acc)


# Triton kernel 3: diagonal contraction to produce Y[b, c, i, n, d] = sum_j M[b, c, i, j, n] * hidden_states[b, c, j, n, d]
# M: [B, Csz, S, S, N] (we pass M = G * L computed in PyTorch; to keep Triton use, we could multiply in Triton too, but original uses PyTorch elementwise)
# hidden_states: [B, Csz, S, N, D]
# Y: [B, Csz, S, N, D]
@triton.jit
def diag_contract_Y_kernel(
    M_ptr, HS_ptr, Y_ptr,
    Bsz: tl.constexpr, Csz: tl.constexpr, S: tl.constexpr, N: tl.constexpr, D: tl.constexpr,
    M_stride_b, M_stride_c, M_stride_i, M_stride_j, M_stride_n,
    HS_stride_b, HS_stride_c, HS_stride_j, HS_stride_n, HS_stride_d,
    Y_stride_b, Y_stride_c, Y_stride_i, Y_stride_n, Y_stride_d,
):
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)
    n = tl.program_id(3)
    d = tl.program_id(4)

    acc = 0.0
    for j in range(0, S):
        m_ptr = M_ptr + b * M_stride_b + c * M_stride_c + i * M_stride_i + j * M_stride_j + n * M_stride_n
        hs_ptr = HS_ptr + b * HS_stride_b + c * HS_stride_c + j * HS_stride_j + n * HS_stride_n + d * HS_stride_d
        m_val = tl.load(m_ptr)
        hs_val = tl.load(hs_ptr)
        acc += m_val * hs_val

    y_ptr = Y_ptr + b * Y_stride_b + c * Y_stride_c + i * Y_stride_i + n * Y_stride_n + d * Y_stride_d
    tl.store(y_ptr, acc)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                A_cumsum: torch.Tensor,
                B: torch.Tensor,
                C: torch.Tensor) -> torch.Tensor:

        # Extract dynamic axes
        device = hidden_states.device
        Bsz, Csz, S, N, D = hidden_states.shape  # num_heads is provided by hidden_states.shape[3]
        N_groups = A_cumsum.shape[3]  # 8

        # Ensure dtypes and device
        assert hidden_states.is_cuda and A_cumsum.is_cuda and B.is_cuda and C.is_cuda, "Tensors must be on CUDA."
        # Cast to float32 for computation
        A = A_cumsum.to(torch.float32)
        # Expand B and C to num_heads=32 via repeat_interleave by 4
        B_expanded = B.to(torch.float32).repeat_interleave(4, dim=3)  # [B, C, S, 32, D]
        C_expanded = C.to(torch.float32).repeat_interleave(4, dim=3)  # [B, C, S, 32, D]

        # 1) Compute L = exp(masked_cumsum(A, dim=-2)) with lower-triangular mask (diagonal=0) in Triton
        # Allocate L
        L = torch.empty((Bsz, Csz, S, S, N_groups), device=device, dtype=torch.float32)

        # Launch masked_cumsum_exp_kernel
        grid_L = (Bsz, Csz, N_groups, S)
        masked_cumsum_exp_kernel[grid_L](
            A, L,
            Bsz, Csz, S, N_groups,
            A.stride(0), A.stride(1), A.stride(2), A.stride(3),
            L.stride(0), L.stride(1), L.stride(2), L.stride(3), L.stride(4),
            num_warps=1, num_stages=1,
        )

        # 2) Compute G = contract(B_expanded, C_expanded) in Triton
        G = torch.empty((Bsz, Csz, S, S, 32), device=device, dtype=torch.float32)

        # Strides for G
        G_stride_b, G_stride_c, G_stride_i, G_stride_j, G_stride_n = G.stride()

        grid_G = (Bsz, Csz, S, S, 32)
        contract_BC_to_G_kernel[grid_G](
            B_expanded, C_expanded, G,
            Bsz, Csz, S, 32, D,  # D is head_dim from hidden_states
            B_expanded.stride(0), B_expanded.stride(1), B_expanded.stride(2), B_expanded.stride(3), B_expanded.stride(4),
            C_expanded.stride(0), C_expanded.stride(1), C_expanded.stride(2), C_expanded.stride(3), C_expanded.stride(4),
            G_stride_b, G_stride_c, G_stride_i, G_stride_j, G_stride_n,
            num_warps=1, num_stages=1,
        )

        # 3) Elementwise M = G * L using PyTorch (matches original). Note: we could write a Triton elementwise kernel, but original uses PyTorch here.
        M = G * L  # G: [B, C, S, S, 32], L: [B, C, S, S, N_groups] -> broadcasting over N_groups, but original L has N_groups, so we need to align. In original, num_heads is 32 and L has N_groups=8. The original code expands B/C to num_heads implicitly via repeat_interleave(4), but L is computed from A_cumsum which has shape [B, C, S, N_groups]. In the original, they then form L with N=32, implying L is expanded implicitly or computed for N=32. To match, we need L to be [B, C, S, S, 32]. The previous code computed L with N_groups=8 and then multiplied by G with N=32, which would be incompatible. Therefore, we recompute L with N=32 to match G's last dim. We'll do that now.

        # Recompute L with N=32 to match G's last dim. That means expanding A_cumsum across N=32 by duplicating values across 8 groups (since original has 8 groups).
        # We can create an expanded A for N=32 that duplicates each group 4 times (since 32/8=4). We'll build A_expanded [B, C, S, 32] where A_expanded[:, :, :, n] = A[:, :, :, n//4]. Then apply the same masked cumsum logic.

        # Build A_expanded: [B, C, S, 32]
        # A has shape [B, C, S, 8]; we map n in [0..31] to group = n // 4, so A_expanded[b, c, i, n] = A[b, c, i, n//4]
        A_expanded = torch.empty((Bsz, Csz, S, 32), device=device, dtype=torch.float32)
        # Fill using indexing: A_expanded[:, :, :, n] = A[:, :, :, n // 4]
        # Efficient way: A[:, :, :, :] broadcast along last dim and assign to expanded
        # We can do this by concatenation or by index fill:
        for b_i in range(Bsz):
            for c_i in range(Csz):
                for i_i in range(S):
                    for n_expanded in range(32):
                        n_group = n_expanded // 4
                        A_expanded[b_i, c_i, i_i, n_expanded] = A[b_i, c_i, i_i, n_group]

        # Now compute L with N=32
        L32 = torch.empty((Bsz, Csz, S, S, 32), device=device, dtype=torch.float32)
        grid_L32 = (Bsz, Csz, 32, S)
        masked_cumsum_exp_kernel[grid_L32](
            A_expanded, L32,
            Bsz, Csz, S, 32,
            A_expanded.stride(0), A_expanded.stride(1), A_expanded.stride(2), A_expanded.stride(3),
            L32.stride(0), L32.stride(1), L32.stride(2), L32.stride(3), L32.stride(4),
            num_warps=1, num_stages=1,
        )

        # Recompute G using the new A_expanded? Not necessary; original code already computed G using B_expanded and C_expanded, and L was computed from A_cumsum with N_groups=8. There seems to be a mismatch in original code: it defines num_heads=32 but A_cumsum has N_groups=8 and uses N_groups in L. The original code then multiplies G (N=32) with L (N=N_groups=8), which is not broadcastable. To ensure correctness, we will follow the original logic and compute M = G * L using L with N_groups=8, by broadcasting across heads. In original code, they multiply G (shape [B, C, S, S, 32]) with L (shape [B, C, S, S, 8]) using PyTorch; PyTorch broadcasts the last dim (8) to 32 by repeating. We will mimic this by computing M elementwise in Triton as M[b, c, i, j, n] = G[b, c, i, j, n] * L[b, c, i, j, n % 8].

        # Elementwise M = G * L (broadcast L over N=32): n in [0..31] maps to group n_group = n % 8
        M = torch.empty((Bsz, Csz, S, S, 32), device=device, dtype=torch.float32)
        grid_M = (Bsz, Csz, S, S, 32)
        # Triton elementwise multiply with broadcasting: use L32[..., n_group] where n_group = n % 8
        @triton.jit
        def elementwise_mul_broadcast_kernel(
            G_ptr, L_ptr, M_ptr,
            Bsz: tl.constexpr, Csz: tl.constexpr, S: tl.constexpr, N: tl.constexpr, N_groups: tl.constexpr,
            G_stride_b, G_stride_c, G_stride_i, G_stride_j, G_stride_n,
            L_stride_b, L_stride_c, L_stride_i, L_stride_j, L_stride_ng,
            M_stride_b, M_stride_c, M_stride_i, M_stride_j, M_stride_n,
        ):
            b = tl.program_id(0)
            c = tl.program_id(1)
            i = tl.program_id(2)
            j = tl.program_id(3)
            n = tl.program_id(4)
            # map n to group in [0..7]
            n_group = n % N_groups
            g_ptr = G_ptr + b * G_stride_b + c * G_stride_c + i * G_stride_i + j * G_stride_j + n * G_stride_n
            l_ptr = L_ptr + b * L_stride_b + c * L_stride_c + i * L_stride_i + j * L_stride_j + n_group * L_stride_ng
            m_ptr = M_ptr + b * M_stride_b + c * M_stride_c + i * M_stride_i + j * M_stride_j + n * M_stride_n
            g_val = tl.load(g_ptr)
            l_val = tl.load(l_ptr)
            tl.store(m_ptr, g_val * l_val)

        elementwise_mul_broadcast_kernel[grid_M](
            G, L, M,
            Bsz, Csz, S, 32, 8,
            G.stride(0), G.stride(1), G.stride(2), G.stride(3), G.stride(4),
            L.stride(0), L.stride(1), L.stride(2), L.stride(3), L.stride(4),
            M.stride(0), M.stride(1), M.stride(2), M.stride(3), M.stride(4),
            num_warps=1, num_stages=1,
        )

        # 4) Diagonal contraction Y[b, c, i, n, d] = sum_j M[b, c, i, j, n] * hidden_states[b, c, j, n, d]
        Y = torch.empty((Bsz, Csz, S, 32, D), device=device, dtype=torch.float32)

        grid_Y = (Bsz, Csz, S, 32, D)
        diag_contract_Y_kernel[grid_Y](
            M, hidden_states.to(torch.float32), Y,
            Bsz, Csz, S, 32, D,
            M.stride(0), M.stride(1), M.stride(2), M.stride(3), M.stride(4),
            hidden_states.to(torch.float32).stride(0), hidden_states.to(torch.float32).stride(1), hidden_states.to(torch.float32).stride(2), hidden_states.to(torch.float32).stride(3), hidden_states.to(torch.float32).stride(4),
            Y.stride(0), Y.stride(1), Y.stride(2), Y.stride(3), Y.stride(4),
            num_warps=1, num_stages=1,
        )

        # 5) Cast to bfloat16 to match original signature
        return Y.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
