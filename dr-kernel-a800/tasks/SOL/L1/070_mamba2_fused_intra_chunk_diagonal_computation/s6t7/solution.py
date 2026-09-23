import torch
import triton
import triton.language as tl


@triton.jit
def tril_mask_1d_kernel(M: tl.pointer_type(tl.int8), S: tl.constexpr):
    # Build a 1D lower-triangular mask M_lower[k] where k = i*S + j, j <= i -> 1 else 0
    pid = tl.program_id(0)
    S_i32 = tl.cast(S, tl.int32)
    i = pid // S
    j = pid % S
    if j <= i:
        tl.store(M + pid, 1)
    else:
        tl.store(M + pid, 0)


@triton.jit
def cumsum_exp_tril_kernel(
    A_ptr, L_ptr,
    Bsz, Hsz, Nsz, S: tl.constexpr,
    A_stride_b, A_stride_h, A_stride_n, A_stride_s1, A_stride_s2,
    Out_stride_b, Out_stride_h, Out_stride_n, Out_stride_s1, Out_stride_s2,
):
    # Grid is (Bsz, Hsz, Nsz). Each program computes L for one (b,h,n).
    b = tl.program_id(0)
    h = tl.program_id(1)
    n = tl.program_id(2)

    # Initialize prefix sum for j = 0
    prefix = 0.0
    for i in range(S):  # i is the "source" chunk index (expanded dim)
        # Process j from 0..S-1: L[b,h,n,i,j] = exp(prefix) if j <= i else 0
        for j in range(S):
            # Load A[b, h, n, i, j]
            a_val = tl.load(A_ptr + b * A_stride_b + h * A_stride_h + n * A_stride_n + i * A_stride_s1 + j * A_stride_s2)
            # Apply mask: j <= i? if yes, add; else keep prefix as it was (0 contribution)
            if j <= i:
                prefix += a_val
            # Store L[b, h, n, i, j] = exp(prefix)
            tl.store(L_ptr + b * Out_stride_b + h * Out_stride_h + n * Out_stride_n + i * Out_stride_s1 + j * Out_stride_s2, tl.exp(prefix))


@triton.jit
def g_contract_kernel(
    B_ptr, C_ptr, G_ptr,
    Bsz, Nsz, S, H, D: tl.constexpr,
    B_stride_b, B_stride_n, B_stride_s, B_stride_h, B_stride_d,
    C_stride_b, C_stride_n, C_stride_s, C_stride_h, C_stride_d,
    G_stride_b, G_stride_n, G_stride_s1, G_stride_s2, G_stride_h,
):
    # Grid is (Bsz, Nsz, S, S, H): compute G[i,j,h] for all (i,j,h)
    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_s1 = tl.program_id(2)  # i
    pid_s2 = tl.program_id(3)  # j
    pid_h = tl.program_id(4)

    acc = 0.0
    # Loop over state dimension D
    for d in range(D):
        b_val = tl.load(B_ptr + pid_b * B_stride_b + pid_n * B_stride_n + pid_s2 * B_stride_s + pid_h * B_stride_h + d * B_stride_d)
        c_val = tl.load(C_ptr + pid_b * C_stride_b + pid_n * C_stride_n + pid_s1 * C_stride_s + pid_h * C_stride_h + d * C_stride_d)
        acc += b_val * c_val

    tl.store(G_ptr + pid_b * G_stride_b + pid_n * G_stride_n + pid_s1 * G_stride_s1 + pid_s2 * G_stride_s2 + pid_h * G_stride_h, acc)


@triton.jit
def m_mul_kernel(
    G_ptr, L_ptr, M_ptr,
    Bsz, Nsz, S, H,
    G_stride_b, G_stride_n, G_stride_s1, G_stride_s2, G_stride_h,
    L_stride_b, L_stride_h, L_stride_n, L_stride_s1, L_stride_s2,
    M_stride_b, M_stride_n, M_stride_s1, M_stride_s2, M_stride_h,
):
    # Grid is (Bsz, Nsz, S, H): for each (b,n,i,h), compute M[b,n,i,j,h] = G[b,n,i,j,h] * L[b,h,n,j,i] over j
    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_s1 = tl.program_id(2)  # i
    pid_h = tl.program_id(3)   # h

    for j in range(S):
        g = tl.load(G_ptr + pid_b * G_stride_b + pid_n * G_stride_n + pid_s1 * G_stride_s1 + j * G_stride_s2 + pid_h * G_stride_h)
        # L[b,h,n,j,i]
        l = tl.load(L_ptr + pid_b * L_stride_b + pid_h * L_stride_h + pid_n * L_stride_n + j * L_stride_s1 + pid_s1 * L_stride_s2)
        m_val = g * l
        tl.store(M_ptr + pid_b * M_stride_b + pid_n * M_stride_n + pid_s1 * M_stride_s1 + j * M_stride_s2 + pid_h * M_stride_h, m_val)


@triton.jit
def y_diag_reduce_kernel(
    M_ptr, hidden_ptr, Y_ptr,
    Bsz, Nsz, S, H, D: tl.constexpr,
    M_stride_b, M_stride_n, M_stride_s1, M_stride_s2, M_stride_h,
    hidden_stride_b, hidden_stride_n, hidden_stride_s, hidden_stride_h, hidden_stride_d,
    Y_stride_b, Y_stride_n, Y_stride_s1, Y_stride_h,
):
    # Grid is (Bsz, Nsz, S, H): compute Y[b,n,i,h] = sum_j M[b,n,i,j,h] * hidden[b,n,j,h]
    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_s1 = tl.program_id(2)  # i
    pid_h = tl.program_id(3)   # h

    y_acc = 0.0
    for j in range(S):
        m = tl.load(M_ptr + pid_b * M_stride_b + pid_n * M_stride_n + pid_s1 * M_stride_s1 + j * M_stride_s2 + pid_h * M_stride_h)
        # hidden[b,n,j,h] is a vector over D; accumulate dot over d
        for d in range(D):
            h_val = tl.load(hidden_ptr + pid_b * hidden_stride_b + pid_n * hidden_stride_n + j * hidden_stride_s + pid_h * hidden_stride_h + d * hidden_stride_d)
            y_acc += m * h_val

    tl.store(Y_ptr + pid_b * Y_stride_b + pid_n * Y_stride_n + pid_s1 * Y_stride_s1 + pid_h * Y_stride_h, y_acc)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.NUM_HEADS = 32
        self.N_GROUPS = 8

    def forward(self, hidden_states: torch.Tensor, A_cumsum: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        # Ensure we are on CUDA for Triton
        assert hidden_states.is_cuda and A_cumsum.is_cuda and B.is_cuda and C.is_cuda, "All inputs must be CUDA tensors for Triton kernels"

        # Shapes
        Bsz, Nsz, S, H, D = hidden_states.shape
        assert A_cumsum.shape == (Bsz, H, Nsz, S), "A_cumsum must have shape [B, H, N, S]"
        # B/C are expected to be expanded to H already (NUM_HEADS // N_GROUPS == 4). We assume that in caller.
        assert B.shape[3] == H and C.shape[3] == H, "B/C must be expanded to num_heads (H)"

        device = hidden_states.device

        # 1) Triton: 1D Lower-triangular mask M_lower [S*S] (j <= i -> 1 else 0)
        M_lower = torch.empty(S * S, dtype=torch.int8, device=device)
        grid_mask = (S * S,)
        tril_mask_1d_kernel[grid_mask](M_lower, S, num_warps=1, num_stages=1)

        # 2) Triton: Compute L = exp(cumsum(masked A)) along the source dimension
        # Expand A_cumsum to [B, H, N, S, S]
        A_expanded = A_cumsum.unsqueeze(-1).expand(Bsz, H, Nsz, S, S).to(torch.float32).contiguous()
        L = torch.empty((Bsz, H, Nsz, S, S), dtype=torch.float32, device=device)

        A_stride_b, A_stride_h, A_stride_n, A_stride_s1, A_stride_s2 = A_expanded.stride()
        Out_stride_b, Out_stride_h, Out_stride_n, Out_stride_s1, Out_stride_s2 = L.stride()

        grid = (Bsz, H, Nsz)
        cumsum_exp_tril_kernel[grid](
            A_expanded, L,
            Bsz, H, Nsz, S,
            A_stride_b, A_stride_h, A_stride_n, A_stride_s1, A_stride_s2,
            Out_stride_b, Out_stride_h, Out_stride_n, Out_stride_s1, Out_stride_s2,
            num_warps=1, num_stages=1
        )

        # 3) Triton: G contraction G[b, n, i, j, h] = sum_d C[b, n, i, h, d] * B[b, n, j, h, d]
        B_contig = B.contiguous()
        C_contig = C.contiguous()
        G = torch.empty((Bsz, Nsz, S, S, H), dtype=torch.float32, device=device)

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

        # 4) Triton: M = G * L (note: L is [B,H,N,S,S], M is [B,N,S,S,H])
        M = torch.empty((Bsz, Nsz, S, S, H), dtype=torch.float32, device=device)

        L_stride_b, L_stride_h, L_stride_n, L_stride_s1, L_stride_s2 = L.stride()
        M_stride_b, M_stride_n, M_stride_s1, M_stride_s2, M_stride_h = M.stride()

        grid_m = (Bsz, Nsz, S, H)
        m_mul_kernel[grid_m](
            G, L, M,
            Bsz, Nsz, S, H,
            G_stride_b, G_stride_n, G_stride_s1, G_stride_s2, G_stride_h,
            L_stride_b, L_stride_h, L_stride_n, L_stride_s1, L_stride_s2,
            M_stride_b, M_stride_n, M_stride_s1, M_stride_s2, M_stride_h,
            num_warps=1, num_stages=1
        )

        # 5) Triton: Y_diag reduction Y[b, n, i, h] = sum_j M[b, n, i, j, h] * hidden[b, n, j, h]
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

        # Return bfloat16 to match original behavior
        return Y.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
