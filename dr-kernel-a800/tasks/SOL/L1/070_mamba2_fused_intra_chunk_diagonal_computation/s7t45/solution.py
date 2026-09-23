import torch
import triton
import triton.language as tl

@triton.jit
def contract_BC_to_G(
    B_ptr,  # float32 [B, C, S, N, K]
    C_ptr,  # float32 [B, C, S, N, K]
    G_ptr,  # float32 [B, C, S, S, N]
    B_size: tl.constexpr,  # batch
    C_size: tl.constexpr,  # num_chunks
    S: tl.constexpr,       # chunk_size
    N: tl.constexpr,       # num_heads
    K: tl.constexpr,       # state_size (hidden_states.last_dim)
    # strides
    B_stride_b, B_stride_c, B_stride_s, B_stride_n, B_stride_k,
    C_stride_b, C_stride_c, C_stride_s, C_stride_n, C_stride_k,
    G_stride_b, G_stride_c, G_stride_i, G_stride_j, G_stride_n,
):
    # One program per (b, c, i, j, n)
    pid = tl.program_id(0)
    b = pid // (C_size * S * S * N)
    tmp = pid % (C_size * S * S * N)
    n = tmp // (C_size * S * S)
    tmp = tmp % (C_size * S * S)
    j = tmp // (C_size * S)
    c = tmp // S
    i = tmp % S

    acc = tl.zeros((), dtype=tl.float32)
    for k in range(0, K):
        b_val = tl.load(B_ptr + b * B_stride_b + c * B_stride_c + i * B_stride_s + n * B_stride_n + k * B_stride_k)
        c_val = tl.load(C_ptr + b * C_stride_b + c * C_stride_c + j * C_stride_s + n * C_stride_n + k * C_stride_k)
        acc += b_val * c_val

    tl.store(G_ptr + b * G_stride_b + c * G_stride_c + i * G_stride_i + j * G_stride_j + n * G_stride_n, acc)


@triton.jit
def diag_contract_Y(
    M_ptr,  # float32 [B, C, S, S, N]
    HS_ptr,  # float32 [B, C, S, N, D]
    Y_ptr,  # float32 [B, C, S, N, D]
    B_size: tl.constexpr,
    C_size: tl.constexpr,
    S: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    # strides
    M_stride_b, M_stride_c, M_stride_i, M_stride_j, M_stride_n,
    HS_stride_b, HS_stride_c, HS_stride_j, HS_stride_n, HS_stride_d,
    Y_stride_b, Y_stride_c, Y_stride_i, Y_stride_n, Y_stride_d,
):
    pid = tl.program_id(0)
    # Map pid to (b, c, i, n, d)
    b = pid // (C_size * S * N * D)
    tmp = pid % (C_size * S * N * D)
    n = tmp // (C_size * S * D)
    tmp = tmp % (C_size * S * D)
    i = tmp // (C_size * D)
    d = tmp % D

    acc = tl.zeros((), dtype=tl.float32)
    for j in range(0, S):
        m_val = tl.load(M_ptr + b * M_stride_b + c * M_stride_c + i * M_stride_i + j * M_stride_j + n * M_stride_n)
        hs_val = tl.load(HS_ptr + b * HS_stride_b + c * HS_stride_c + j * HS_stride_j + n * HS_stride_n + d * HS_stride_d)
        acc += m_val * hs_val

    tl.store(Y_ptr + b * Y_stride_b + c * Y_stride_c + i * Y_stride_i + n * Y_stride_n + d * Y_stride_d, acc)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.NUM_HEADS = 32
        self.N_GROUPS = 8
        self.EXPAND_FACTOR = self.NUM_HEADS // self.N_GROUPS  # 4

    def forward(self, hidden_states: torch.Tensor, A_cumsum: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        # hidden_states: [B, C, S, N, D]
        Bsz, Csz, S, N, D = hidden_states.shape
        device = hidden_states.device

        # Compute L using PyTorch to match original masked cumsum + exp exactly
        # A_cumsum: [B, N, C, S] (same as original)
        # Lower-triangular mask with diagonal=-1 over (i, j) where i=row, j=col
        # We need L: [B, N, C, S, S] = exp of masked cumsum along j for each i
        # Build mask for SxS
        mask = torch.tril(torch.ones(S, S, device=device, dtype=torch.bool), diagonal=-1)
        # segment_sum: [B, N, C, S, S]
        # For each (b, n, c, i), inclusive sum up to j, then apply mask (exclude j==i)
        # Using PyTorch cumsum for correctness; this is the original heavy masked cumsum step.
        # Note: A_cumsum is [B, N, C, S]; we want to form A_cumsum[i, j] = A_cumsum[b, n, c, j].
        # We expand along i via cumsum and masking.
        L = torch.empty((Bsz, N, Csz, S, S), device=device, dtype=torch.float32)

        # For each (b, n, c), compute the inclusive cumsum along j and apply mask
        for b_idx in range(Bsz):
            for n_idx in range(N):
                for c_idx in range(Csz):
                    # Create A_ij for this (b, n, c): shape [S]
                    A_ij = A_cumsum[b_idx, n_idx, c_idx, :]  # [S]
                    # Inclusive cumsum along j
                    seg_sum = torch.cumsum(A_ij, dim=0)  # [S]
                    # Apply mask (exclude j==i). We need to set positions (i, j) where j == i to -inf then exp -> 0.
                    # Construct a [S, S] matrix where rows are i and cols are j.
                    # For each i, we zero out the diagonal element.
                    for i in range(S):
                        # L[b, n, c, i, :] = exp of masked cumsum up to i
                        # We need per-row segment sums up to i for j<=i. That is just seg_sum[:i+1].
                        if i >= 0:
                            # Build L rows: [S]
                            # For j <= i, take exp(seg_sum[j]); for j == i, set to 0 (exclude diagonal).
                            l_row = torch.empty(S, device=device, dtype=torch.float32)
                            for j in range(S):
                                if j <= i:
                                    if j == i:
                                        l_row[j] = 0.0  # exp(-inf) -> 0, but we can directly set to 0
                                    else:
                                        l_row[j] = torch.exp(seg_sum[j].float())
                                else:
                                    l_row[j] = 0.0
                            L[b_idx, n_idx, c_idx, i, :] = l_row
        # Now L is constructed as per original code: lower-triangular with diagonal=-1.

        # Expand B and C to NUM_HEADS=32
        B_expanded = B.repeat_interleave(self.EXPAND_FACTOR, dim=3).to(torch.float32)
        C_expanded = C.repeat_interleave(self.EXPAND_FACTOR, dim=3).to(torch.float32)

        # Compute G via Triton: [B, C, S, S, N], float32
        G = torch.empty((Bsz, Csz, S, S, N), device=device, dtype=torch.float32)

        B_stride_b, B_stride_c, B_stride_s, B_stride_n, B_stride_k = B_expanded.stride()
        C_stride_b, C_stride_c, C_stride_s, C_stride_n, C_stride_k = C_expanded.stride()
        G_stride_b, G_stride_c, G_stride_i, G_stride_j, G_stride_n = G.stride()

        # Grid: one program per (b, c, i, j, n) -> total Bsz*Csz*S*S*N
        grid_G = (Bsz * Csz * S * S * N,)
        K = D  # hidden_states.shape[4]
        contract_BC_to_G[grid_G](
            B_expanded, C_expanded, G,
            B_size=Bsz, C_size=Csz, S=S, N=N, K=K,
            B_stride_b=B_stride_b, B_stride_c=B_stride_c, B_stride_s=B_stride_s, B_stride_n=B_stride_n, B_stride_k=B_stride_k,
            C_stride_b=C_stride_b, C_stride_c=C_stride_c, C_stride_s=C_stride_s, C_stride_n=C_stride_n, C_stride_k=C_stride_k,
            G_stride_b=G_stride_b, G_stride_c=G_stride_c, G_stride_i=G_stride_i, G_stride_j=G_stride_j, G_stride_n=G_stride_n,
            num_warps=1, num_stages=1
        )

        # Elementwise M = G * L (PyTorch), float32
        L_perm = L.permute(0, 3, 2, 1, 4).contiguous()  # [B, C, S, S, N]
        M = G * L_perm

        # Compute Y_diag via Triton: [B, C, S, N, D], float32
        Y = torch.empty((Bsz, Csz, S, N, D), device=device, dtype=torch.float32)

        M_stride_b, M_stride_c, M_stride_i, M_stride_j, M_stride_n = M.stride()
        HS_stride_b, HS_stride_c, HS_stride_j, HS_stride_n, HS_stride_d = hidden_states.to(torch.float32).stride()
        Y_stride_b, Y_stride_c, Y_stride_i, Y_stride_n, Y_stride_d = Y.stride()

        grid_Y = (Bsz * Csz * S * N * D,)
        diag_contract_Y[grid_Y](
            M, hidden_states.to(torch.float32), Y,
            B_size=Bsz, C_size=Csz, S=S, N=N, D=D,
            M_stride_b=M_stride_b, M_stride_c=M_stride_c, M_stride_i=M_stride_i, M_stride_j=M_stride_j, M_stride_n=M_stride_n,
            HS_stride_b=HS_stride_b, HS_stride_c=HS_stride_c, HS_stride_j=HS_stride_j, HS_stride_n=HS_stride_n, HS_stride_d=HS_stride_d,
            Y_stride_b=Y_stride_b, Y_stride_c=Y_stride_c, Y_stride_i=Y_stride_i, Y_stride_n=Y_stride_n, Y_stride_d=Y_stride_d,
            num_warps=1, num_stages=1
        )

        # Cast to bfloat16 to match original signature
        return Y.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
