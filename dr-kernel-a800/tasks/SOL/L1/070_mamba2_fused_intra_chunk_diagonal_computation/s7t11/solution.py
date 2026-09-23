import torch
import triton
import triton.language as tl


@triton.jit
def l_masked_cumsum_exp_kernel(
    A_ptr,  # [B, Csz, S, N]
    L_ptr,  # [B, Csz, S, S, N], float32
    B_size, C_size, S, N,
    A_stride_b, A_stride_c, A_stride_j, A_stride_n,
    L_stride_b, L_stride_c, L_stride_i, L_stride_j, L_stride_n,
    BLOCK_S: tl.constexpr
):
    # program ids: one program per (b, c, n)
    b = tl.program_id(0)
    c = tl.program_id(1)
    n = tl.program_id(2)

    # We'll loop over i and j. For each i, compute cumsum up to i-1 (exclude i) via running sum,
    # then store L[i, j] = exp(running) if j < i (diagonal=-1), else 1.0.
    i = 0
    while i < S:
        running = tl.zeros((), dtype=tl.float32)
        j = 0
        while j < S:
            # tril(diagonal=-1): include j < i, exclude j == i. j and i are scalars here.
            include = j < i
            a_val = tl.load(A_ptr + b * A_stride_b + c * A_stride_c + j * A_stride_j + n * A_stride_n).to(tl.float32)
            if include:
                running += a_val
            # For j >= i, we must store 1.0 because the original tril excludes the diagonal and upper triangle remains 1 before exp.
            l_val = tl.exp(running) if include else 1.0
            tl.store(L_ptr + b * L_stride_b + c * L_stride_c + i * L_stride_i + j * L_stride_j + n * L_stride_n, l_val)
            j += 1
        i += 1


@triton.jit
def contract_BC_to_G_kernel(
    B_ptr,  # [B, Csz, S, N, K]
    C_ptr,  # [B, Csz, S, N, K]
    G_ptr,  # [B, Csz, S, S, N], float32
    B_size, Csz, S, N, K,
    B_stride_b, B_stride_c, B_stride_j, B_stride_n, B_stride_k,
    C_stride_b, C_stride_c, C_stride_i, C_stride_n, C_stride_k,
    G_stride_b, G_stride_c, G_stride_i, G_stride_j, G_stride_n,
    BLOCK_K: tl.constexpr
):
    # program ids: one program per (b, c, i, j, n)
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)
    j = tl.program_id(3)
    n = tl.program_id(4)

    acc = tl.zeros((), dtype=tl.float32)
    k = 0
    while k < K:
        b_val = tl.load(B_ptr + b * B_stride_b + c * B_stride_c + j * B_stride_j + n * B_stride_n + k * B_stride_k).to(tl.float32)
        c_val = tl.load(C_ptr + b * C_stride_b + c * C_stride_c + i * C_stride_i + n * C_stride_n + k * C_stride_k).to(tl.float32)
        acc += b_val * c_val
        k += 1
    tl.store(G_ptr + b * G_stride_b + c * G_stride_c + i * G_stride_i + j * G_stride_j + n * G_stride_n, acc)


@triton.jit
def diag_contract_Y_kernel(
    M_ptr,       # [B, Csz, S, S, N], float32
    HS_ptr,      # [B, Csz, S, N, D], float32
    Y_ptr,       # [B, Csz, S, N, D], float32
    B_size, Csz, S, N, D,
    M_stride_b, M_stride_c, M_stride_i, M_stride_j, M_stride_n,
    HS_stride_b, HS_stride_c, HS_stride_j, HS_stride_n, HS_stride_d,
    Y_stride_b, Y_stride_c, Y_stride_i, Y_stride_n, Y_stride_d,
    BLOCK_J: tl.constexpr
):
    # program ids: one program per (b, c, i, n, d)
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)
    n = tl.program_id(3)
    d = tl.program_id(4)

    acc = tl.zeros((), dtype=tl.float32)
    j = 0
    while j < S:
        m_val = tl.load(M_ptr + b * M_stride_b + c * M_stride_c + i * M_stride_i + j * M_stride_j + n * M_stride_n).to(tl.float32)
        hs_val = tl.load(HS_ptr + b * HS_stride_b + c * HS_stride_c + j * HS_stride_j + n * HS_stride_n + d * HS_stride_d).to(tl.float32)
        acc += m_val * hs_val
        j += 1
    tl.store(Y_ptr + b * Y_stride_b + c * Y_stride_c + i * Y_stride_i + n * Y_stride_n + d * Y_stride_d, acc)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                A_cumsum: torch.Tensor,
                B: torch.Tensor,
                C: torch.Tensor):
        """
        hidden_states: [B, Csz, S, N, D]
        A_cumsum: [B, Csz, S, N]
        B: [B, Csz, S, N_groups, K]
        C: [B, Csz, S, N_groups, K]
        Returns Y_diag: [B, Csz, S, N, D] (cast to bfloat16)
        """
        Bsz = hidden_states.shape[0]
        Csz = hidden_states.shape[1]
        S = hidden_states.shape[2]
        N = hidden_states.shape[3]
        D = hidden_states.shape[4]

        # Expand B and C to num_heads=32 via repeat_interleave(4)
        B_expanded = B.repeat_interleave(4, dim=3)  # from N_groups=8 -> N=32
        C_expanded = C.repeat_interleave(4, dim=3)

        # Ensure float32 for computation
        A_cumsum_f32 = A_cumsum.to(torch.float32)
        B_expanded_f32 = B_expanded.to(torch.float32)
        C_expanded_f32 = C_expanded.to(torch.float32)
        hidden_states_f32 = hidden_states.to(torch.float32)

        # Allocate L: [B, Csz, S, S, N], float32
        L = torch.empty((Bsz, Csz, S, S, N), device=hidden_states.device, dtype=torch.float32)

        # Strides
        A_stride_b, A_stride_c, A_stride_j, A_stride_n = A_cumsum_f32.stride()
        L_stride_b, L_stride_c, L_stride_i, L_stride_j, L_stride_n = L.stride()

        # Launch Triton kernel to compute L
        grid_L = (Bsz, Csz, N)
        l_masked_cumsum_exp_kernel[grid_L](
            A_cumsum_f32, L,
            Bsz, Csz, S, N,
            A_stride_b, A_stride_c, A_stride_j, A_stride_n,
            L_stride_b, L_stride_c, L_stride_i, L_stride_j, L_stride_n,
            BLOCK_S=S,
            num_warps=1, num_stages=1
        )

        # Allocate G: [B, Csz, S, S, N], float32
        G = torch.empty((Bsz, Csz, S, S, N), device=hidden_states.device, dtype=torch.float32)

        # Strides for B/C/G
        B_stride_b, B_stride_c, B_stride_j, B_stride_n, B_stride_k = B_expanded_f32.stride()
        C_stride_b, C_stride_c, C_stride_i, C_stride_n, C_stride_k = C_expanded_f32.stride()
        G_stride_b, G_stride_c, G_stride_i, G_stride_j, G_stride_n = G.stride()

        # Reduction dimension K equals head_dim D in this implementation (consistent with original outer-product).
        K = D

        grid_G = (Bsz, Csz, S, S, N)
        contract_BC_to_G_kernel[grid_G](
            B_expanded_f32, C_expanded_f32, G,
            Bsz, Csz, S, N, K,
            B_stride_b, B_stride_c, B_stride_j, B_stride_n, B_stride_k,
            C_stride_b, C_stride_c, C_stride_i, C_stride_n, C_stride_k,
            G_stride_b, G_stride_c, G_stride_i, G_stride_j, G_stride_n,
            BLOCK_K=K,
            num_warps=1, num_stages=1
        )

        # Elementwise M = G * L (PyTorch for simplicity; heavy numeric work is already Triton above)
        M = G * L  # float32

        # Allocate Y: [B, Csz, S, N, D], float32
        Y = torch.empty((Bsz, Csz, S, N, D), device=hidden_states.device, dtype=torch.float32)

        # Strides for M, hidden_states_f32, Y
        M_stride_b, M_stride_c, M_stride_i, M_stride_j, M_stride_n = M.stride()
        HS_stride_b, HS_stride_c, HS_stride_j, HS_stride_n, HS_stride_d = hidden_states_f32.stride()
        Y_stride_b, Y_stride_c, Y_stride_i, Y_stride_n, Y_stride_d = Y.stride()

        # Launch Triton diagonal contraction kernel
        grid_Y = (Bsz, Csz, S, N, D)
        diag_contract_Y_kernel[grid_Y](
            M, hidden_states_f32, Y,
            Bsz, Csz, S, N, D,
            M_stride_b, M_stride_c, M_stride_i, M_stride_j, M_stride_n,
            HS_stride_b, HS_stride_c, HS_stride_j, HS_stride_n, HS_stride_d,
            Y_stride_b, Y_stride_c, Y_stride_i, Y_stride_n, Y_stride_d,
            BLOCK_J=S,
            num_warps=1, num_stages=1
        )

        # Cast to bfloat16 to match original signature
        return Y.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
