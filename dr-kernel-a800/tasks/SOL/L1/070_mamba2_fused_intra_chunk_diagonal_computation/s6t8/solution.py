import torch
import triton
import triton.language as tl


@triton.jit
def g_contract_kernel(
    B_ptr, C_ptr, G_ptr,
    Bsz: tl.constexpr, Nsz: tl.constexpr, S: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
    # strides for B: [B, N, S, H, D]
    B_stride_b, B_stride_n, B_stride_s, B_stride_h, B_stride_d,
    # strides for C: [B, N, S, H, D]
    C_stride_b, C_stride_n, C_stride_s, C_stride_h, C_stride_d,
    # strides for G: [B, N, S, S, H]
    G_stride_b, G_stride_n, G_stride_s1, G_stride_s2, G_stride_h
):
    # 5D grid: (b, n, i, j, h)
    b = tl.program_id(0)
    n = tl.program_id(1)
    i = tl.program_id(2)
    j = tl.program_id(3)
    h = tl.program_id(4)

    acc = 0.0
    # Loop over state dimension D to compute sum_d C[b,n,i,h,d] * B[b,n,j,h,d]
    for d in range(0, D):
        # Load C[b,n,i,h,d]
        ptr_C = B_ptr + b * B_stride_b + n * B_stride_n + i * B_stride_s + h * B_stride_h + d * B_stride_d
        C_val = tl.load(ptr_C)
        # Load B[b,n,j,h,d]
        ptr_B = C_ptr + b * C_stride_b + n * C_stride_n + j * C_stride_s + h * C_stride_h + d * C_stride_d
        B_val = tl.load(ptr_B)
        acc += C_val * B_val

    # Store G[b,n,i,j,h] = acc
    ptr_G = G_ptr + b * G_stride_b + n * G_stride_n + i * G_stride_s1 + j * G_stride_s2 + h * G_stride_h
    tl.store(ptr_G, acc)


@triton.jit
def y_diag_reduce_kernel(
    M_ptr, hidden_ptr, Y_ptr,
    Bsz: tl.constexpr, Nsz: tl.constexpr, S: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
    # strides for M: [B, N, S, S, H]
    M_stride_b, M_stride_n, M_stride_s1, M_stride_s2, M_stride_h,
    # strides for hidden: [B, N, S, H, D]
    hidden_stride_b, hidden_stride_n, hidden_stride_s, hidden_stride_h, hidden_stride_d,
    # strides for Y: [B, N, S, H]
    Y_stride_b, Y_stride_n, Y_stride_s1, Y_stride_h
):
    # 4D grid: (b, n, i, h)
    b = tl.program_id(0)
    n = tl.program_id(1)
    i = tl.program_id(2)
    h = tl.program_id(3)

    acc = 0.0
    # Loop over j (chunk dimension) to reduce
    for j in range(0, S):
        # Load M[b,n,i,j,h]
        ptr_M = M_ptr + b * M_stride_b + n * M_stride_n + i * M_stride_s1 + j * M_stride_s2 + h * M_stride_h
        M_val = tl.load(ptr_M)
        # Load hidden[b,n,j,h,0..D-1], then sum over D
        # We'll compute dot over D: sum_d hidden[b,n,j,h,d]
        hidden_dot = 0.0
        for d in range(0, D):
            ptr_hidden = hidden_ptr + b * hidden_stride_b + n * hidden_stride_n + j * hidden_stride_s + h * hidden_stride_h + d * hidden_stride_d
            hidden_val = tl.load(ptr_hidden)
            hidden_dot += hidden_val
        acc += M_val * hidden_dot

    # Store Y[b,n,i,h] = acc
    ptr_Y = Y_ptr + b * Y_stride_b + n * Y_stride_n + i * Y_stride_s1 + h * Y_stride_h
    tl.store(ptr_Y, acc)


class ModelNew(torch.nn.Module):
    def __init__(self, NUM_HEADS: int = 32, N_GROUPS: int = 8):
        super().__init__()
        self.NUM_HEADS = NUM_HEADS
        self.N_GROUPS = N_GROUPS

    def forward(self, hidden_states: torch.Tensor,
                A_cumsum: torch.Tensor,
                B: torch.Tensor,
                C: torch.Tensor) -> torch.Tensor:
        # Shapes
        Bsz, Nsz, S, H, D = hidden_states.shape
        assert A_cumsum.shape == (Bsz, H, Nsz, S), "A_cumsum must have shape [B, H, N, S]"
        assert B.shape == (Bsz, Nsz, S, H, D), "B must have shape [B, N, S, H, D]"
        assert C.shape == (Bsz, Nsz, S, H, D), "C must have shape [B, N, S, H, D]"

        device = hidden_states.device

        # Compute L = exp(cumsum(masked A)) exactly as original:
        # A_cumsum: [B, H, N, S]
        # Expand to [B, H, N, S, S] and apply tril(diagonal=-1), then cumsum along dim=3 (S axis), then exp.
        A_expanded = A_cumsum.unsqueeze(-1).expand(Bsz, H, Nsz, S, S).to(torch.float32)  # [B,H,N,S,S]
        # Lower-triangular mask with diagonal=-1 (keep j <= i)
        j = torch.arange(S, device=device)
        i = torch.arange(S, device=device).unsqueeze(0)  # [1, S]
        lower_mask = (j <= i)[0]  # [S, S], bool
        lower_mask_exp = lower_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(Bsz, H, Nsz, S, S).to(torch.float32)
        A_masked = A_expanded * lower_mask_exp  # mask j>i to 0
        A_masked_cum = torch.cumsum(A_masked, dim=3)  # [B,H,N,S,S]
        L = torch.exp(A_masked_cum)  # [B,H,N,S,S], causal mask with exp

        # 1) Triton: G contraction G[b, n, i, j, h] = sum_d C[b, n, i, h, d] * B[b, n, j, h, d]
        G = torch.empty((Bsz, Nsz, S, S, H), dtype=torch.float32, device=device)

        B_contig = B.contiguous()
        C_contig = C.contiguous()

        B_stride_b, B_stride_n, B_stride_s, B_stride_h, B_stride_d = B_contig.stride()
        C_stride_b, C_stride_n, C_stride_s, C_stride_h, C_stride_d = C_contig.stride()
        G_stride_b, G_stride_n, G_stride_s1, G_stride_s2, G_stride_h = G.stride()

        grid_g = (Bsz, Nsz, S, S, H)
        g_contract_kernel[grid_g](
            B_contig, C_contig, G,
            Bsz, Nsz, S, H, D,
            B_stride_b, B_stride_n, B_stride_s, B_stride_h, B_stride_d,
            C_stride_b, C_stride_n, C_stride_s, C_stride_h, C_stride_d,
            G_stride_b, G_stride_n, G_stride_s1, G_stride_s2, G_stride_h,
            num_warps=1, num_stages=1
        )

        # 2) Triton: M = G * L. Since L is [B,H,N,S,S], we need to permute to [B,N,S,S,H] and multiply.
        L_perm = L.permute(0, 2, 3, 4, 1)  # [B, N, S, S, H]
        M = torch.empty((Bsz, Nsz, S, S, H), dtype=torch.float32, device=device)

        G_stride_b, G_stride_n, G_stride_s1, G_stride_s2, G_stride_h = G.stride()
        Lp_stride_b, Lp_stride_n, Lp_stride_s1, Lp_stride_s2, Lp_stride_h = L_perm.stride()
        M_stride_b, M_stride_n, M_stride_s1, M_stride_s2, M_stride_h = M.stride()

        grid_m = (Bsz, Nsz, S, S, H)
        # Triton kernel for elementwise multiply: M = G * L_perm
        # Note: Triton elementwise multiply can be done via tl.load + multiply, store.
        # Implement a simple elementwise kernel using 5D grid.
        @triton.jit
        def mul_elementwise_kernel(G_ptr, L_ptr, M_ptr,
                                    Bsz: tl.constexpr, Nsz: tl.constexpr, S: tl.constexpr, H: tl.constexpr):
            b = tl.program_id(0)
            n = tl.program_id(1)
            i = tl.program_id(2)
            j = tl.program_id(3)
            h = tl.program_id(4)
            ptr_G = G_ptr + b * G_stride_b + n * G_stride_n + i * G_stride_s1 + j * G_stride_s2 + h * G_stride_h
            ptr_L = L_ptr + b * Lp_stride_b + n * Lp_stride_n + i * Lp_stride_s1 + j * Lp_stride_s2 + h * Lp_stride_h
            G_val = tl.load(ptr_G)
            L_val = tl.load(ptr_L)
            tl.store(M_ptr + b * M_stride_b + n * M_stride_n + i * M_stride_s1 + j * M_stride_s2 + h * M_stride_h, G_val * L_val)

        mul_elementwise_kernel[grid_m](
            G, L_perm, M,
            Bsz, Nsz, S, H,
            G_stride_b, G_stride_n, G_stride_s1, G_stride_s2, G_stride_h,
            Lp_stride_b, Lp_stride_n, Lp_stride_s1, Lp_stride_s2, Lp_stride_h,
            M_stride_b, M_stride_n, M_stride_s1, M_stride_s2, M_stride_h,
            num_warps=1, num_stages=1
        )

        # 3) Triton: Y_diag reduction Y[b, n, i, h] = sum_j M[b, n, i, j, h] * hidden[b, n, j, h]
        hidden = hidden_states.contiguous()  # [B, N, S, H, D]
        Y = torch.empty((Bsz, Nsz, S, H), dtype=torch.float32, device=device)

        M_stride_b, M_stride_n, M_stride_s1, M_stride_s2, M_stride_h = M.stride()
        hidden_stride_b, hidden_stride_n, hidden_stride_s, hidden_stride_h, hidden_stride_d = hidden.stride()
        Y_stride_b, Y_stride_n, Y_stride_s1, Y_stride_h = Y.stride()

        grid_y = (Bsz, Nsz, S, H)
        y_diag_reduce_kernel[grid_y](
            M, hidden, Y,
            Bsz, Nsz, S, H, D,
            M_stride_b, M_stride_n, M_stride_s1, M_stride_s2, M_stride_h,
            hidden_stride_b, hidden_stride_n, hidden_stride_s, hidden_stride_h, hidden_stride_d,
            Y_stride_b, Y_stride_n, Y_stride_s1, Y_stride_h,
            num_warps=1, num_stages=1
        )

        return Y.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
