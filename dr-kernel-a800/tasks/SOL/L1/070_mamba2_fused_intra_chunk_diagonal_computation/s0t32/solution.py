import torch
import triton
import triton.language as tl


# Kernel 1: Build L[i, j, h] = exp(cumsum(A[b, h, c, :])[j]) if i >= j else 0
# A: [B, H, C, S]; L: [B, C, S, S, H]
@triton.jit
def build_L_kernel(
    A_ptr, L_ptr,
    B, C, H,
    aB, aH, aC, aS,
    lB, lC, lSi, lSj, lH,
    S_CONST: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_h = tl.program_id(2)

    # Compute base addresses using strides (in elements)
    # Note: S_CONST is constexpr, loops are unrolled at compile time.
    cumsum = tl.zeros([S_CONST], dtype=tl.float32)

    for i in range(S_CONST):
        # Compute cumsum up to position i: sum_{t=0..i} A[b, h, c, t]
        for t in range(S_CONST):
            a_idx = pid_b * aB + pid_h * aH + pid_c * aC + t * aS
            a_val = tl.load(A_ptr + a_idx)
            cumsum[t] += a_val

        # Fill L[i, j, h] for j in [0..S_CONST-1], only if i >= j
        for j in range(S_CONST):
            if i >= j:
                l_idx = pid_b * lB + pid_c * lC + i * lSi + j * lSj + pid_h * lH
                l_val = tl.exp(cumsum[j])
                tl.store(L_ptr + l_idx, l_val)
            # else store 0 implicitly


# Kernel 2: Compute G[i, j, h] = sum over n of C_exp[i, n, j, h] * B_exp[j, n, i, h]
# B_exp: [B, C, S, H, N]; C_exp: [B, C, S, H, N]; G: [B, C, S, S, H]
@triton.jit
def compute_G_kernel(
    Bexp_ptr, Cexp_ptr, G_ptr,
    B, C, H, N,
    beB, beC, beSi, beSh, beN,
    ceB, ceC, ceSi, ceSh, ceN,
    geB, geC, geSi, geSj, geH,
    S_CONST: tl.constexpr,
    N_CONST: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_i = tl.program_id(2)
    pid_j = tl.program_id(3)
    pid_h = tl.program_id(4)

    acc = tl.zeros([1], dtype=tl.float32)

    for n in range(N_CONST):
        # Load B_exp[b, c, j, h, n] and C_exp[b, c, i, n, j, h]
        b_idx = (pid_b * beB) + (pid_c * beC) + (pid_j * beSi) + (pid_h * beSh) + (n * beN)
        c_idx = (pid_b * ceB) + (pid_c * ceC) + (pid_i * ceSi) + (pid_h * ceSh) + (n * ceN)

        b_val = tl.load(Bexp_ptr + b_idx)
        c_val = tl.load(Cexp_ptr + c_idx)

        acc += b_val * c_val

    g_idx = (pid_b * geB) + (pid_c * geC) + (pid_i * geSi) + (pid_j * geSj) + (pid_h * geH)
    tl.store(G_ptr + g_idx, acc[0])


# Kernel 3: Compute Y_diag[b, c, i, h, d] = sum_j M[i, j, h] * hidden_states[b, c, j, h, d]
# M: [B, C, S, S, H]; hidden: [B, C, S, H, head_dim]; Y_diag: [B, C, S, H, head_dim]
@triton.jit
def compute_Y_diag_kernel(
    M_ptr, hidden_ptr, Y_ptr,
    B, C, H, HEAD_DIM,
    mB, mC, mSi, mSj, mH,
    hB, hC, hSi, hSh, hD,
    yB, yC, ySi, yH, yD,
    S_CONST: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_i = tl.program_id(2)
    pid_h = tl.program_id(3)
    pid_d = tl.program_id(4)

    acc = tl.zeros([1], dtype=tl.float32)

    for j in range(S_CONST):
        m_idx = (pid_b * mB) + (pid_c * mC) + (pid_i * mSi) + (j * mSj) + (pid_h * mH)
        h_idx = (pid_b * hB) + (pid_c * hC) + (j * hSi) + (pid_h * hSh) + (pid_d * hD)
        m_val = tl.load(M_ptr + m_idx)
        h_val = tl.load(hidden_ptr + h_idx)
        acc += m_val * h_val

    y_idx = (pid_b * yB) + (pid_c * yC) + (pid_i * ySi) + (pid_h * yH) + (pid_d * yD)
    tl.store(Y_ptr + y_idx, acc[0])


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Constants from reference
        self.S_CONST = 128
        self.H_CONST = 32
        self.N_GROUPS = 8
        self.NUM_HEADS = 32

    def forward(self, hidden_states: torch.Tensor, A_cumsum: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        # Ensure tensors are on CUDA and contiguous
        assert hidden_states.is_cuda and A_cumsum.is_cuda and B.is_cuda and C.is_cuda, "Tensors must be on CUDA device"
        hidden_states = hidden_states.contiguous()
        A_cumsum = A_cumsum.contiguous()
        B = B.contiguous()
        C = C.contiguous()

        # Shapes
        Bsz, Csz, S, H, head_dim = hidden_states.shape
        assert S == self.S_CONST, f"Expected S=128, got {S}"
        assert H == self.H_CONST, f"Expected H=32, got {H}"

        # 1) Build L[i, j, h] in Triton
        L = torch.empty((Bsz, Csz, S, S, H), device=hidden_states.device, dtype=torch.float32)

        # Strides for A and L
        aB, aH, aC, aS = A_cumsum.stride()
        lB, lC, lSi, lSj, lH = L.stride()

        grid_L = (Bsz, Csz, H)
        build_L_kernel[grid_L](
            A_cumsum, L,
            Bsz, Csz, H,
            aB, aH, aC, aS,
            lB, lC, lSi, lSj, lH,
            S_CONST=self.S_CONST,
            num_warps=4, num_stages=1,
        )

        # 2) Expand B and C along H: repeat_interleave by NUM_HEADS // N_GROUPS = 4
        N = B.size(-1)  # state size (typically 128)
        B_expanded = B.repeat_interleave(4, dim=3)  # [B, C, S, H, N]
        C_expanded = C.repeat_interleave(4, dim=3)  # [B, C, S, H, N]

        B_expanded = B_expanded.contiguous()
        C_expanded = C_expanded.contiguous()

        # 3) Compute G[i, j, h] via Triton kernel
        G = torch.empty((Bsz, Csz, S, S, H), device=hidden_states.device, dtype=torch.float32)

        beB, beC, beSi, beSh, beN = B_expanded.stride()
        ceB, ceC, ceSi, ceSh, ceN = C_expanded.stride()
        geB, geC, geSi, geSj, geH = G.stride()

        grid_G = (Bsz, Csz, S, S, H)
        compute_G_kernel[grid_G](
            B_expanded, C_expanded, G,
            Bsz, Csz, H, N,
            beB, beC, beSi, beSh, beN,
            ceB, ceC, ceSi, ceSh, ceN,
            geB, geC, geSi, geSj, geH,
            S_CONST=self.S_CONST,
            N_CONST=N,
            num_warps=4, num_stages=1,
        )

        # 4) Multiply M = G * L
        M = G * L  # elementwise

        # 5) Compute Y_diag via Triton kernel
        Y_diag = torch.empty((Bsz, Csz, S, H, head_dim), device=hidden_states.device, dtype=torch.float32)

        mB, mC, mSi, mSj, mH = M.stride()
        hB, hC, hSi, hSh, hD = hidden_states.stride()
        yB, yC, ySi, yH, yD = Y_diag.stride()

        grid_Y = (Bsz, Csz, S, H, head_dim)
        compute_Y_diag_kernel[grid_Y](
            M, hidden_states, Y_diag,
            Bsz, Csz, H, head_dim,
            mB, mC, mSi, mSj, mH,
            hB, hC, hSi, hSh, hD,
            yB, yC, ySi, yH, yD,
            S_CONST=self.S_CONST,
            num_warps=4, num_stages=1,
        )

        # Return in bfloat16 to match original model
        return Y_diag.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
