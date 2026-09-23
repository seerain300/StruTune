import torch
import triton
import triton.language as tl

# Triton kernel 1: masked cumsum along j (source) for each (b, c, n, i), then exp to produce L.
# A: [B, C, S, N] (A_cumsum permuted to match)
# L: [B, C, S, S, N] float32
@triton.jit
def cumsum_lower_exp(
    A_ptr, L_ptr,
    B_size: tl.constexpr, C_size: tl.constexpr, S: tl.constexpr, N: tl.constexpr,
    A_stride_b, A_stride_c, A_stride_i, A_stride_n,
    L_stride_b, L_stride_c, L_stride_i, L_stride_j, L_stride_n,
):
    b = tl.program_id(0)
    c = tl.program_id(1)
    n = tl.program_id(2)
    i = tl.program_id(3)

    running = 0.0
    for j in range(0, S):
        # lower-triangular mask with diagonal=-1: include j < i
        include = j < i
        a_val = tl.load(A_ptr + b * A_stride_b + c * A_stride_c + j * A_stride_i + n * A_stride_n, mask=True, other=0.0)
        a_val = tl.where(include, a_val, 0.0)
        running += a_val
        tl.store(L_ptr + b * L_stride_b + c * L_stride_c + i * L_stride_i + j * L_stride_j + n * L_stride_n, tl.exp(running))

# Triton kernel 2: contract B_expanded and C_expanded to form G[b, c, i, j, n]
# Shapes:
#   B_expanded: [B, C, S, N, D]
#   C_expanded: [B, C, S, N, D]
#   G: [B, C, S, S, N]
@triton.jit
def contract_BC_to_G(
    B_ptr, C_ptr, G_ptr,
    B_size: tl.constexpr, C_size: tl.constexpr, S: tl.constexpr, N: tl.constexpr, D: tl.constexpr,
    B_stride_b, B_stride_c, B_stride_i, B_stride_n, B_stride_k,
    C_stride_b, C_stride_c, C_stride_i, C_stride_n, C_stride_k,
    G_stride_b, G_stride_c, G_stride_i, G_stride_j, G_stride_n,
):
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)
    j = tl.program_id(3)
    n = tl.program_id(4)

    acc = 0.0
    for k in range(0, D):
        b_val = tl.load(B_ptr + b * B_stride_b + c * B_stride_c + j * B_stride_i + n * B_stride_n + k * B_stride_k)
        c_val = tl.load(C_ptr + b * C_stride_b + c * C_stride_c + i * C_stride_i + n * C_stride_n + k * C_stride_k)
        acc += b_val * c_val
    tl.store(G_ptr + b * G_stride_b + c * G_stride_c + i * G_stride_i + j * G_stride_j + n * G_stride_n, acc)

# Triton kernel 3: diagonal contraction Y[b, c, i, n, d] = sum_j M[b, c, i, j, n] * hidden_states[b, c, j, n, d]
# M: [B, C, S, S, N] (computed as G * L in PyTorch)
# hidden_states: [B, C, S, N, D]
# Y: [B, C, S, N, D]
@triton.jit
def diag_contract_Y(
    M_ptr, HS_ptr, Y_ptr,
    B_size: tl.constexpr, C_size: tl.constexpr, S: tl.constexpr, N: tl.constexpr, D: tl.constexpr,
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
        m_val = tl.load(M_ptr + b * M_stride_b + c * M_stride_c + i * M_stride_i + j * M_stride_j + n * M_stride_n)
        hs_val = tl.load(HS_ptr + b * HS_stride_b + c * HS_stride_c + j * HS_stride_j + n * HS_stride_n + d * HS_stride_d)
        acc += m_val * hs_val
    tl.store(Y_ptr + b * Y_stride_b + c * Y_stride_c + i * Y_stride_i + n * Y_stride_n + d * Y_stride_d, acc)

class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, A_cumsum: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        # Infer shapes
        Bsz, Csz, S, N, D = hidden_states.shape

        # Ensure dtypes and contiguity
        hidden_states_f32 = hidden_states.to(torch.float32).contiguous()

        # A_cumsum: [B, N, Csz, S] -> permute to [B, Csz, S, N]
        A = A_cumsum.permute(0, 3, 2, 1).contiguous().to(torch.float32)

        # Allocate L: [B, C, S, S, N] float32
        L = torch.empty((Bsz, Csz, S, S, N), device=hidden_states.device, dtype=torch.float32)

        # Launch Triton kernel for cumsum_lower_exp
        A_stride_b, A_stride_c, A_stride_i, A_stride_n = A.stride()
        L_stride_b, L_stride_c, L_stride_i, L_stride_j, L_stride_n = L.stride()
        grid_L = (Bsz, Csz, N, S)  # (b, c, n, i)
        cumsum_lower_exp[grid_L](
            A, L,
            B_size=Bsz, C_size=Csz, S=S, N=N,
            A_stride_b=A_stride_b, A_stride_c=A_stride_c, A_stride_i=A_stride_i, A_stride_n=A_stride_n,
            L_stride_b=L_stride_b, L_stride_c=L_stride_c, L_stride_i=L_stride_i, L_stride_j=L_stride_j, L_stride_n=L_stride_n,
            num_warps=1, num_stages=1
        )

        # Expand B and C from n_groups=8 to num_heads=32 by repeat_interleave(NUM_HEADS // N_GROUPS) = 4.
        # B: [B, C, S, 8, D], C: [B, C, S, 8, D] -> B_expanded: [B, C, S, 32, D], C_expanded: [B, C, S, 32, D]
        # We need to split the 8 groups into 32 slots. Each original group index g (0..7) maps to head index n = g * 4 + h_local,
        # where h_local in {0,1,2,3}. We replicate each group's vectors 4 times across N.
        # We can construct expanded tensors by repeating each group's contribution 4 times into the N dimension.
        # To avoid reshaping complexity, we compute B_expanded and C_expanded using tensor.expand/repeat_interleave with actual
        # mapping. Since Triton expects strides, we materialize expanded tensors and keep them contiguous.

        # Materialize B_expanded and C_expanded
        # We'll build B_expanded of shape [B, C, S, N, D]:
        # For each (b, c, s, d), we take B[b, c, s, g, d] and place it into B_expanded at n = g*4 + h_local for h_local=0..3.
        # Do the same for C.
        B_expanded = torch.empty((Bsz, Csz, S, N, D), device=hidden_states.device, dtype=torch.float32)
        C_expanded = torch.empty((Bsz, Csz, S, N, D), device=hidden_states.device, dtype=torch.float32)

        # Populate expanded tensors: replicate each of the 8 groups across 4 head slots
        for g in range(8):
            # n positions for this group
            # We need to distribute g equally across 32 heads: n = g * 4 + h_local for h_local in 0..3
            # Build a mask for n
            # Loop over h_local = 0..3
            for h_local in range(4):
                n_idx = g * 4 + h_local
                b_sel = B[:, g, :, d]  # B is [B, C, S, 8, D] -> select g-th group
                # Assign to B_expanded[..., n_idx, :]
                B_expanded[:, :, :, n_idx, :] = b_sel
                C_expanded[:, :, :, n_idx, :] = C[:, g, :, d]

        # Ensure contiguous
        B_expanded = B_expanded.contiguous()
        C_expanded = C_expanded.contiguous()

        # Allocate G: [B, C, S, S, N]
        G = torch.empty((Bsz, Csz, S, S, N), device=hidden_states.device, dtype=torch.float32)

        # Launch Triton kernel for contraction
        B_stride_b, B_stride_c, B_stride_i, B_stride_n, B_stride_k = B_expanded.stride()
        C_stride_b, C_stride_c, C_stride_i, C_stride_n, C_stride_k = C_expanded.stride()
        G_stride_b, G_stride_c, G_stride_i, G_stride_j, G_stride_n = G.stride()
        grid_G = (Bsz, Csz, S, S, N)
        contract_BC_to_G[grid_G](
            B_expanded, C_expanded, G,
            B_size=Bsz, C_size=Csz, S=S, N=N, D=D,
            B_stride_b=B_stride_b, B_stride_c=B_stride_c, B_stride_i=B_stride_i, B_stride_n=B_stride_n, B_stride_k=B_stride_k,
            C_stride_b=C_stride_b, C_stride_c=C_stride_c, C_stride_i=C_stride_i, C_stride_n=C_stride_n, C_stride_k=C_stride_k,
            G_stride_b=G_stride_b, G_stride_c=G_stride_c, G_stride_i=G_stride_i, G_stride_j=G_stride_j, G_stride_n=G_stride_n,
            num_warps=1, num_stages=1
        )

        # Elementwise multiply M = G * L
        # Use PyTorch for elementwise multiply (lightweight)
        M = G * L  # [B, C, S, S, N], float32

        # Allocate Y: [B, C, S, N, D]
        Y = torch.empty((Bsz, Csz, S, N, D), device=hidden_states.device, dtype=torch.float32)

        # Launch Triton kernel for diagonal contraction
        M_stride_b, M_stride_c, M_stride_i, M_stride_j, M_stride_n = M.stride()
        HS_stride_b, HS_stride_c, HS_stride_j, HS_stride_n, HS_stride_d = hidden_states_f32.stride()
        Y_stride_b, Y_stride_c, Y_stride_i, Y_stride_n, Y_stride_d = Y.stride()
        grid_Y = (Bsz, Csz, S, N, D)
        diag_contract_Y[grid_Y](
            M, hidden_states_f32, Y,
            B_size=Bsz, C_size=Csz, S=S, N=N, D=D,
            M_stride_b=M_stride_b, M_stride_c=M_stride_c, M_stride_i=M_stride_i, M_stride_j=M_stride_j, M_stride_n=M_stride_n,
            HS_stride_b=HS_stride_b, HS_stride_c=HS_stride_c, HS_stride_j=HS_stride_j, HS_stride_n=HS_stride_n, HS_stride_d=HS_stride_d,
            Y_stride_b=Y_stride_b, Y_stride_c=Y_stride_c, Y_stride_i=Y_stride_i, Y_stride_n=Y_stride_n, Y_stride_d=Y_stride_d,
            num_warps=1, num_stages=1
        )

        # Cast to bfloat16 to match original


def run(*args):
    return ModelNew()(*args)
