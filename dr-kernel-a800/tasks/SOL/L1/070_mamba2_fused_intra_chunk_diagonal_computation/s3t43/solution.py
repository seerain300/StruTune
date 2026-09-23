import torch
import triton
import triton.language as tl


@triton.jit
def kernel_A_to_L(
    A_ptr, L_ptr,
    Bsz, Csz, H,
    stride_A_b, stride_A_h, stride_A_c, stride_A_s,
    stride_L_b, stride_L_c, stride_L_i, stride_L_j, stride_L_h,
    S: tl.constexpr,  # 128
    num_warps=4, num_stages=2
):
    # Grid: (B, C, H)
    b = tl.program_id(0)
    c = tl.program_id(1)
    h = tl.program_id(2)

    # For each i, build cumsum across j and store L[i, j, h] = exp(cumsum) if i >= j else 0
    for i in range(S):
        # vector s[j] for j in [0..S-1]
        s = tl.zeros((S,), dtype=tl.float32)
        for j in range(S):
            a_off = b * stride_A_b + h * stride_A_h + c * stride_A_c + j * stride_A_s
            a_val = tl.load(A_ptr + a_off)
            s[j] = a_val
        # enforce triangular: for j > i, set s[j] = 0
        for j in range(S):
            if j > i:
                s[j] = 0.0
        # store L[i, j, h] = exp(s[j])
        for j in range(S):
            l_off = b * stride_L_b + c * stride_L_c + i * stride_L_i + j * stride_L_j + h * stride_L_h
            l_val = tl.exp(s[j])
            tl.store(L_ptr + l_off, l_val)


@triton.jit
def kernel_BC_to_G(
    B_ptr, C_ptr, G_ptr,
    Bsz, Csz,
    stride_B_b, stride_B_c, stride_B_s, stride_B_g, stride_B_k,
    stride_C_b, stride_C_c, stride_C_s, stride_C_g, stride_C_k,
    stride_G_b, stride_G_c, stride_G_i, stride_G_j, stride_G_h,
    S: tl.constexpr,          # 128
    G_CONST: tl.constexpr,    # 8
    K: tl.constexpr,          # 32
    num_warps=4, num_stages=2
):
    # Grid: (B, C)
    b = tl.program_id(0)
    c = tl.program_id(1)

    # Loop over i, j, and h; accumulate G[i, j, h] per (i, j, h)
    for i in range(S):
        for j in range(S):
            acc = 0.0
            # accumulate over groups and states
            for g in range(G_CONST):
                for k in range(K):
                    b_off = b * stride_B_b + c * stride_B_c + j * stride_B_s + g * stride_B_g + k * stride_B_k
                    c_off = b * stride_C_b + c * stride_C_c + i * stride_C_s + g * stride_C_g + k * stride_C_k
                    b_val = tl.load(B_ptr + b_off)
                    c_val = tl.load(C_ptr + c_off)
                    acc += b_val * c_val
            # store G[b, c, i, j, h] for each h
            # We write all H values; here H is passed but since we don't have grid dimension for H, we write h=0..31 if needed.
            # In this implementation, H is handled by calling ModelNew.forward for specific H=32.
            for hh in range(32):  # we assume H == 32; if different, adjust accordingly
                g_off = b * stride_G_b + c * stride_G_c + i * stride_G_i + j * stride_G_j + hh * stride_G_h
                tl.store(G_ptr + g_off, acc)


@triton.jit
def kernel_contract_M_with_hidden(
    G_ptr, L_ptr, hidden_ptr, Y_diag_ptr,
    Bsz, Csz, S, H, D,
    stride_G_b, stride_G_c, stride_G_i, stride_G_j, stride_G_h,
    stride_L_b, stride_L_c, stride_L_i, stride_L_j, stride_L_h,
    stride_hs_b, stride_hs_c, stride_hs_s, stride_hs_h, stride_hs_d,
    stride_y_b, stride_y_c, stride_y_i, stride_y_h, stride_y_d,
    num_warps=4, num_stages=2
):
    # Grid: (B, C, S, H, D)
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)
    h = tl.program_id(3)
    d = tl.program_id(4)

    acc = 0.0
    for j in range(S):
        # load G[b, c, i, j, h] and L[b, c, i, j, h]
        g_off = b * stride_G_b + c * stride_G_c + i * stride_G_i + j * stride_G_j + h * stride_G_h
        l_off = b * stride_L_b + c * stride_L_c + i * stride_L_i + j * stride_L_j + h * stride_L_h
        G_val = tl.load(G_ptr + g_off)
        L_val = tl.load(L_ptr + l_off)
        M_val = G_val * L_val

        # load hidden[b, c, j, h, d]
        hs_off = b * stride_hs_b + c * stride_hs_c + j * stride_hs_s + h * stride_hs_h + d * stride_hs_d
        hs_val = tl.load(hidden_ptr + hs_off)

        acc += M_val * hs_val

    # store Y_diag[b, c, i, h, d]
    y_off = b * stride_y_b + c * stride_y_c + i * stride_y_i + h * stride_y_h + d * stride_y_d
    tl.store(Y_diag_ptr + y_off, acc)


class ModelNew(torch.nn.Module):
    def __init__(self, H: int = 32, D: int = 32, S: int = 128, G_CONST: int = 8, K: int = 32):
        super().__init__()
        self.H = H
        self.D = D
        self.S = S
        self.G_CONST = G_CONST
        self.K = K

    def forward(self, hidden_states: torch.Tensor, A_cumsum: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        """
        Triton-only forward:
        - kernel_A_to_L: compute L from A_cumsum
        - kernel_BC_to_G: compute G from B and C
        - kernel_contract_M_with_hidden: compute Y_diag by contracting M=G*L with hidden_states
        Returns Y_diag in bfloat16, shape [B, C, S, H, D].
        """
        assert hidden_states.is_cuda and A_cumsum.is_cuda and B.is_cuda and C.is_cuda, "Inputs must be CUDA tensors"
        Bsz, Csz, S, H, D = hidden_states.shape
        assert S == self.S and H == self.H and D == self.D, "This Triton implementation specializes for S={}, H={}, D={}".format(self.S, self.H, self.D)
        assert A_cumsum.shape == (Bsz, H, Csz, S), f"Expected A_cumsum shape [B,H,C,S], got {A_cumsum.shape}"
        assert B.shape == (Bsz, Csz, S, self.G_CONST, self.K), f"Expected B shape [B,C,S,{self.G_CONST},{self.K}], got {B.shape}"
        assert C.shape == (Bsz, Csz, S, self.G_CONST, self.K), f"Expected C shape [B,C,S,{self.G_CONST},{self.K}], got {C.shape}"

        # Ensure contiguous
        A = A_cumsum.contiguous()
        B = B.contiguous()
        C = C.contiguous()
        hidden = hidden_states.contiguous()

        # Allocate outputs
        L = torch.empty((Bsz, Csz, S, S, H), dtype=torch.float32, device=hidden.device)
        G = torch.empty((Bsz, Csz, S, S, H), dtype=torch.float32, device=hidden.device)
        Y_diag = torch.empty((Bsz, Csz, S, H, D), dtype=torch.float32, device=hidden.device)

        # Launch kernel 1: compute L from A_cumsum
        grid_L = (Bsz, Csz, H)
        kernel_A_to_L[grid_L](
            A, L,
            Bsz, Csz, H,
            A_cumsum.stride(0), A_cumsum.stride(1), A_cumsum.stride(2), A_cumsum.stride(3),
            L.stride(0), L.stride(1), L.stride(2), L.stride(3), L.stride(4),
            S=self.S,
        )

        # Launch kernel 2: compute G from B and C
        grid_G = (Bsz, Csz)
        kernel_BC_to_G[grid_G](
            B, C, G,
            Bsz, Csz,
            B.stride(0), B.stride(1), B.stride(2), B.stride(3), B.stride(4),
            C.stride(0), C.stride(1), C.stride(2), C.stride(3), C.stride(4),
            G.stride(0), G.stride(1), G.stride(2), G.stride(3), G.stride(4),
            S=self.S, G_CONST=self.G_CONST, K=self.K,
        )

        # Launch kernel 3: compute Y_diag by contracting M = G * L with hidden
        grid_Y = (Bsz, Csz, self.S, self.H, self.D)
        kernel_contract_M_with_hidden[grid_Y](
            G, L, hidden, Y_diag,
            Bsz, Csz, self.S, self.H, self.D,
            G.stride(0), G.stride(1), G.stride(2), G.stride(3), G.stride(4),
            L.stride(0), L.stride(1), L.stride(2), L.stride(3), L.stride(4),
            hidden.stride(0), hidden.stride(1), hidden.stride(2), hidden.stride(3), hidden.stride(4),
            Y_diag.stride(0), Y_diag.stride(1), Y_diag.stride(2), Y_diag.stride(3), Y_diag.stride(4),
        )

        return Y_diag.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
