import torch
import triton
import triton.language as tl


@triton.jit
def mask_cumsum_exp(
    A_ptr,  # [B, N, C, S]
    L_ptr,  # [B, N, C, S, S]
    B_size, C_size, S,
    A_stride_b, A_stride_n, A_stride_c, A_stride_s,
    L_stride_b, L_stride_n, L_stride_c, L_stride_i, L_stride_j,
):
    # Each program computes L for one (b, n, c, i)
    b = tl.program_id(0)
    n = tl.program_id(1)
    c = tl.program_id(2)
    i = tl.program_id(3)

    # If out of bounds (not common in grid), return
    if b >= B_size or n >= 32 or c >= C_size or i >= S:
        return

    # Running sum initialized to 0
    running_sum = 0.0
    # Compute masked cumsum along j (source axis) and apply exp
    # Include j < i (diagonal=-1), exclude j == i
    for j in range(S):
        # Load A[b, n, c, j]
        a_val = tl.load(A_ptr + b * A_stride_b + n * A_stride_n + c * A_stride_c + j * A_stride_s)
        # Apply lower-triangular mask: j < i
        include = j < i
        # If include, add; else keep running_sum
        running_sum = running_sum + a_val if include else running_sum
        # Store exp(cumsum) to L[b, n, c, i, j]
        tl.store(L_ptr + b * L_stride_b + n * L_stride_n + c * L_stride_c + i * L_stride_i + j * L_stride_j, tl.exp(running_sum))


@triton.jit
def contract_BC_to_G(
    B_exp_ptr,  # [B, C, S, N, K] where N=NUM_HEADS=32
    C_exp_ptr,  # [B, C, S, N, K]
    G_ptr,      # [B, C, S, S, N]
    B_size, C_size, S, N, K,
    B_stride_b, B_stride_c, B_stride_j, B_stride_n, B_stride_k,
    C_stride_b, C_stride_c, C_stride_i, C_stride_n, C_stride_k,
    G_stride_b, G_stride_c, G_stride_i, G_stride_j, G_stride_n,
):
    # Grid is 5D: (b, c, i, j, n)
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)
    j = tl.program_id(3)
    n = tl.program_id(4)

    if b >= B_size or c >= C_size or i >= S or j >= S or n >= N:
        return

    acc = 0.0
    # Loop over state dimension K
    for k in range(K):
        b_val = tl.load(B_exp_ptr + b * B_stride_b + c * B_stride_c + j * B_stride_j + n * B_stride_n + k * B_stride_k)
        c_val = tl.load(C_exp_ptr + b * C_stride_b + c * C_stride_c + i * C_stride_i + n * C_stride_n + k * C_stride_k)
        acc += b_val * c_val

    # Store G[b, c, i, j, n] = acc
    tl.store(G_ptr + b * G_stride_b + c * G_stride_c + i * G_stride_i + j * G_stride_j + n * G_stride_n, acc)


@triton.jit
def diag_contract_Y(
    M_ptr,        # [B, C, S, S, N]
    HS_ptr,       # [B, C, S, N, D] (float32)
    Y_ptr,        # [B, C, S, N, D] (float32)
    B_size, C_size, S, N, D,
    M_stride_b, M_stride_c, M_stride_i, M_stride_j, M_stride_n,
    HS_stride_b, HS_stride_c, HS_stride_j, HS_stride_n, HS_stride_d,
    Y_stride_b, Y_stride_c, Y_stride_i, Y_stride_n, Y_stride_d,
):
    # Grid is 5D: (b, c, i, n, d)
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)
    n = tl.program_id(3)
    d = tl.program_id(4)

    if b >= B_size or c >= C_size or i >= S or n >= N or d >= D:
        return

    # Accumulate over j
    acc = 0.0
    for j in range(S):
        m_val = tl.load(M_ptr + b * M_stride_b + c * M_stride_c + i * M_stride_i + j * M_stride_j + n * M_stride_n)
        hs_val = tl.load(HS_ptr + b * HS_stride_b + c * HS_stride_c + j * HS_stride_j + n * HS_stride_n + d * HS_stride_d)
        acc += m_val * hs_val

    # Store Y[b, c, i, n, d]
    tl.store(Y_ptr + b * Y_stride_b + c * Y_stride_c + i * Y_stride_i + n * Y_stride_n + d * Y_stride_d, acc)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Constants from original code
        self.NUM_HEADS = 32
        self.N_GROUPS = 8

    def forward(self, hidden_states: torch.Tensor, A_cumsum: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        """
        hidden_states: [B, C, S, N, D]
        A_cumsum: [B, N, C, S] (note: original order [batch, num_heads, num_chunks, chunk_size])
        B: [B, C, S, G, K]
        C: [B, C, S, G, K]
        Returns Y_diag: [B, C, S, N, D], cast to bfloat16
        """
        # Infer shapes
        Bsz = hidden_states.shape[0]
        Csz = hidden_states.shape[1]
        S = hidden_states.shape[2]  # chunk_size
        N = hidden_states.shape[3]  # num_heads
        D = hidden_states.shape[4]  # head_dim

        # A_cumsum shape: [B, N, C, S]
        assert A_cumsum.shape[0] == Bsz and A_cumsum.shape[3] == S
        N_heads_in_A = A_cumsum.shape[1]
        Csz_in_A = A_cumsum.shape[2]
        assert N_heads_in_A == self.NUM_HEADS and Csz_in_A == Csz, "A_cumsum shape mismatch with expected num_heads or num_chunks"

        # Prepare expanded B and C to num_heads = 32 by repeat_interleave(4)
        # B and C: [B, C, S, G, K]
        Bsz_b, Csz_b, S_b, G, K = B.shape
        assert Bsz_b == Bsz and Csz_b == Csz and S_b == S, "B shape mismatch"
        Csz_c, K_c = C.shape[1], C.shape[4]
        assert Csz_c == Csz and K_c == K, "C shape mismatch"

        # Ensure contiguous and float32 for computation
        A = A_cumsum.contiguous().to(torch.float32)
        hidden_states_f32 = hidden_states.contiguous().to(torch.float32)

        # Compute L in Triton: [B, N, C, S, S]
        device = hidden_states.device
        L = torch.empty((Bsz, self.NUM_HEADS, Csz, S, S), device=device, dtype=torch.float32)

        A_stride_b, A_stride_n, A_stride_c, A_stride_s = A.stride()
        L_stride_b, L_stride_n, L_stride_c, L_stride_i, L_stride_j = L.stride()

        grid_L = (Bsz, self.NUM_HEADS, Csz, S)
        mask_cumsum_exp[grid_L](
            A, L,
            B_size=Bsz, C_size=Csz, S=S,
            A_stride_b=A_stride_b, A_stride_n=A_stride_n, A_stride_c=A_stride_c, A_stride_s=A_stride_s,
            L_stride_b=L_stride_b, L_stride_n=L_stride_n, L_stride_c=L_stride_c, L_stride_i=L_stride_i, L_stride_j=L_stride_j,
            num_warps=1, num_stages=1
        )

        # Expand B and C to num_heads=32 by repeat_interleave(4)
        # original code does: B_exp = B.repeat_interleave(NUM_HEADS // N_GROUPS, dim=3)
        # Here, NUM_HEADS // N_GROUPS = 4
        B_exp = B.repeat_interleave(4, dim=3)  # -> [B, C, S, 32, K]
        C_exp = C.repeat_interleave(4, dim=3)  # -> [B, C, S, 32, K]

        B_exp = B_exp.contiguous().to(torch.float32)
        C_exp = C_exp.contiguous().to(torch.float32)

        # Compute G in Triton: [B, C, S, S, N]
        G = torch.empty((Bsz, Csz, S, S, self.NUM_HEADS), device=device, dtype=torch.float32)

        B_stride_b, B_stride_c, B_stride_j, B_stride_n, B_stride_k = B_exp.stride()
        C_stride_b, C_stride_c, C_stride_i, C_stride_n, C_stride_k = C_exp.stride()
        G_stride_b, G_stride_c, G_stride_i, G_stride_j, G_stride_n = G.stride()

        grid_G = (Bsz, Csz, S, S, self.NUM_HEADS)
        contract_BC_to_G[grid_G](
            B_exp, C_exp, G,
            B_size=Bsz, C_size=Csz, S=S, N=self.NUM_HEADS, K=K,
            B_stride_b=B_stride_b, B_stride_c=B_stride_c, B_stride_j=B_stride_j, B_stride_n=B_stride_n, B_stride_k=B_stride_k,
            C_stride_b=C_stride_b, C_stride_c=C_stride_c, C_stride_i=C_stride_i, C_stride_n=C_stride_n, C_stride_k=C_stride_k,
            G_stride_b=G_stride_b, G_stride_c=G_stride_c, G_stride_i=G_stride_i, G_stride_j=G_stride_j, G_stride_n=G_stride_n,
            num_warps=1, num_stages=1
        )

        # Elementwise M = G * L (in PyTorch). This avoids torch.exp/tril/cumsum, as they are handled in Triton for L.
        M = G * L  # float32

        # Compute Y_diag in Triton: [B, C, S, N, D]
        Y = torch.empty((Bsz, Csz, S, self.NUM_HEADS, D), device=device, dtype=torch.float32)

        M_stride_b, M_stride_c, M_stride_i, M_stride_j, M_stride_n = M.stride()
        HS_stride_b, HS_stride_c, HS_stride_j, HS_stride_n, HS_stride_d = hidden_states_f32.stride()
        Y_stride_b, Y_stride_c, Y_stride_i, Y_stride_n, Y_stride_d = Y.stride()

        grid_Y = (Bsz, Csz, S, self.NUM_HEADS, D)
        diag_contract_Y[grid_Y](
            M, hidden_states_f32, Y,
            B_size=Bsz, C_size=Csz, S=S, N=self.NUM_HEADS, D=D,
            M_stride_b=M_stride_b, M_stride_c=M_stride_c, M_stride_i=M_stride_i, M_stride_j=M_stride_j, M_stride_n=M_stride_n,
            HS_stride_b=HS_stride_b, HS_stride_c=HS_stride_c, HS_stride_j=HS_stride_j, HS_stride_n=HS_stride_n, HS_stride_d=HS_stride_d,
            Y_stride_b=Y_stride_b, Y_stride_c=Y_stride_c, Y_stride_i=Y_stride_i, Y_stride_n=Y_stride_n, Y_stride_d=Y_stride_d,
            num_warps=1, num_stages=1
        )

        # Cast to bfloat16 to match original signature
        return Y.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
