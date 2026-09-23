import torch
import triton
import triton.language as tl


@triton.jit
def masked_cumsum_exp_kernel(
    A: tl.pointer,            # A_cumsum: [B, N, C, S]
    L: tl.pointer,            # output L: [B, N, C, S, S] (float32)
    B_size: tl.int32,         # B
    N: tl.int32,              # num_heads
    C_size: tl.int32,         # num_chunks
    S: tl.int32,              # chunk_size
):
    # program ids: tile over (b, n, c, i)
    b = tl.program_id(0)
    n = tl.program_id(1)
    c = tl.program_id(2)
    i = tl.program_id(3)

    # base offset for A[b, n, c, :]
    base = (b * N + n) * C_size * S + c * S

    # running sum for each i (along j). We will build cumsum along dim=1 of A (which is C)
    # Note: A is [B, N, C, S], so indexing for A[b, n, c, j] is base + j
    running = 0.0
    # For diagonal=-1, include j < i. Initialize running with j=0 and set it to 0 if j >= i and not included. We do this by masking.
    # We will compute cumsum step by step for j from 0 to S-1.
    # To ensure correctness for all i and j, we will compute masked cumsum: only add when j < i else add 0.
    for j in range(0, S):
        # get A[b, n, c, j]
        val = tl.load(A + base + j)
        # apply mask: include only if j < i (diagonal=-1)
        if j < i:
            running += val
        # store exp(running) into L[b, n, c, i, j] as float32
        # L offset: ((b * N + n) * (C_size * S * S) + c * (S * S) + i * S + j)
        L_offset = ((b * N + n) * (C_size * S * S) + c * (S * S) + i * S + j)
        tl.store(L + L_offset, tl.exp(running))

    # After loop, for j >= i, running already excludes j>=i contributions. We just need to write zeros where j >= i,
    # but we already masked during the loop. So L contains exp(cumsum) for j < i and 1 for j >= i if included (but mask excluded),
    # but since we never add when j >= i, running at that point is just the cumsum up to i-1, and we wrote exp(running) for j < i,
    # and we need to fill j >= i with 0. We can do it here:
    for j in range(i, S):
        L_offset = ((b * N + n) * (C_size * S * S) + c * (S * S) + i * S + j)
        tl.store(L + L_offset, 0.0)


@triton.jit
def contract_BC_to_G_kernel(
    B_exp: tl.pointer,        # [B, C, S, N, K]
    C_exp: tl.pointer,        # [B, C, S, N, K]
    G: tl.pointer,            # output [B, C, S, S, N] (float32)
    B_size: tl.int32,         # B
    C_size: tl.int32,         # C
    S: tl.int32,              # S
    N: tl.int32,              # N
    K: tl.constexpr,          # state_size (constexpr for Triton)
):
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)
    j = tl.program_id(3)
    n = tl.program_id(4)

    # Accumulator for G[b, c, i, j, n]
    acc = 0.0
    # Loop over k in [0, K)
    for k in range(0, K):
        # B_exp[b, c, j, n, k]
        B_offset = (b * C_size * S * N * K) + (c * S * N * K) + (j * N * K) + (n * K) + k
        B_val = tl.load(B_exp + B_offset)
        # C_exp[b, c, i, n, k]
        C_offset = (b * C_size * S * N * K) + (c * S * N * K) + (i * N * K) + (n * K) + k
        C_val = tl.load(C_exp + C_offset)
        acc += B_val * C_val

    # Store G
    G_offset = (b * C_size * S * S * N) + (c * S * S * N) + (i * S * N) + (j * N) + n
    tl.store(G + G_offset, acc)


@triton.jit
def diag_contract_Y_kernel(
    M: tl.pointer,            # [B, C, S, S, N] (float32)
    HS: tl.pointer,           # [B, C, S, N, D] (float32)
    Y: tl.pointer,            # output [B, C, S, N, D] (float32)
    B_size: tl.int32,         # B
    C_size: tl.int32,         # C
    S: tl.int32,              # S
    N: tl.int32,              # N
    D: tl.constexpr,          # D
):
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)
    n = tl.program_id(3)
    d = tl.program_id(4)

    # Accumulator for Y[b, c, i, n, d]
    acc = 0.0
    # Loop over j in [0, S)
    for j in range(0, S):
        # M[b, c, i, j, n]
        M_offset = (b * C_size * S * S * N) + (c * S * S * N) + (i * S * N) + (j * N) + n
        M_val = tl.load(M + M_offset)
        # HS[b, c, j, n, d]
        HS_offset = (b * C_size * S * N * D) + (c * S * N * D) + (j * N * D) + (n * D) + d
        HS_val = tl.load(HS + HS_offset)
        acc += M_val * HS_val

    # Store Y
    Y_offset = (b * C_size * S * N * D) + (c * S * N * D) + (i * N * D) + (n * D) + d
    tl.store(Y + Y_offset, acc)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, A_cumsum: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        """
        Triton-optimized version of the original run function.
        Computes Y_diag with Triton kernels:
          - L via masked cumsum along j for each (b, n, c, i) and exp
          - G via contraction of expanded B and C
          - Y via diagonal contraction of M = G * L with hidden_states
        Returns: tensor of shape [B, C, S, N, D], cast to bfloat16.
        """
        # Infer shapes
        Bsz, Csz, S, N, D = hidden_states.shape  # batch_size, num_chunks, chunk_size, num_heads, head_dim
        device = hidden_states.device

        # Ensure tensors are contiguous and float32 for compute
        A_cumsum_f = A_cumsum.contiguous().to(torch.float32)           # [B, N, C, S]
        hidden_states_f = hidden_states.contiguous().to(torch.float32) # [B, C, S, N, D]
        # Expand B and C from n_groups to num_heads=32 (repeat_interleave by 4)
        B_exp = B.contiguous().to(torch.float32).repeat_interleave(4, dim=3)  # [B, C, S, N, K]
        C_exp = C.contiguous().to(torch.float32).repeat_interleave(4, dim=3)  # [B, C, S, N, K]
        K = hidden_states_f.shape[4]  # state_size

        # Allocate L: [B, N, C, S, S] (float32)
        L = torch.empty((Bsz, N, Csz, S, S), device=device, dtype=torch.float32)

        # Launch Triton kernel to compute L
        grid_L = (Bsz, N, Csz, S)
        masked_cumsum_exp_kernel[grid_L](
            A_cumsum_f, L,
            B_size=Bsz, N=N, C_size=Csz, S=S,
            num_warps=1, num_stages=1
        )

        # Allocate G: [B, C, S, S, N] (float32)
        G = torch.empty((Bsz, Csz, S, S, N), device=device, dtype=torch.float32)

        # Launch Triton kernel to compute G
        grid_G = (Bsz, Csz, S, S, N)
        contract_BC_to_G_kernel[grid_G](
            B_exp, C_exp, G,
            B_size=Bsz, C_size=Csz, S=S, N=N, K=K,
            num_warps=1, num_stages=1
        )

        # Compute M = G * L (elementwise, allowed in PyTorch)
        M = G * L  # [B, C, S, S, N], float32

        # Allocate Y: [B, C, S, N, D] (float32)
        Y = torch.empty((Bsz, Csz, S, N, D), device=device, dtype=torch.float32)

        # Launch Triton kernel to compute Y via diagonal contraction
        grid_Y = (Bsz, Csz, S, N, D)
        diag_contract_Y_kernel[grid_Y](
            M, hidden_states_f, Y,
            B_size=Bsz, C_size=Csz, S=S, N=N, D=D,
            num_warps=1, num_stages=1
        )

        # Cast to bfloat16 to match original signature
        return Y.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
