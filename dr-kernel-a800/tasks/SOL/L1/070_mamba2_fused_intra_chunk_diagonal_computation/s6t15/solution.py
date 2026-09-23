import torch
import triton
import triton.language as tl


# Kernel: Compute L = exp(cumsum(masked A)) with lower triangle (j <= i).
# Input:
#   A: [B, H, N, S] float32 (pointer; we logically treat as [B,H,N,S,S] by masking j>i)
# Output:
#   L_out: [B, H, N, S, S] float32
@triton.jit
def cumsum_exp_tril_kernel(
    A: tl.pointer_type(tl.float32),
    L_out: tl.pointer_type(tl.float32),
    Bsz: tl.constexpr, H: tl.constexpr, Nsz: tl.constexpr, S: tl.constexpr
):
    # Program ids map to (b, h, n)
    b = tl.program_id(0)
    h = tl.program_id(1)
    n = tl.program_id(2)

    # We process i from 0..S-1
    for i in range(S):
        prefix = 0.0
        # Compute prefix sum over j <= i: A[b, h, n, j]
        for j in range(S):
            a_ptr = A + ((b * H + h) * Nsz + n) * S + j  # linearized [B,H,N,S]
            a_val = tl.load(a_ptr)
            prefix += a_val
        # Write L[b, h, n, i, j] = exp(prefix) for j <= i, else 0
        for j in range(S):
            out_ptr = L_out + (((b * H + h) * Nsz + n) * S) * S + i * S + j
            if j <= i:
                tl.store(out_ptr, tl.exp(prefix))
            else:
                tl.store(out_ptr, 0.0)


# Kernel: G contraction: G[i,j,h] = sum_d C[i,h,d] * B[j,h,d]
# Inputs:
#   B_exp: [B, N, S, H, D] float32
#   C_exp: [B, N, S, H, D] float32
#   G_out: [B, N, S, S, H] float32
@triton.jit
def g_contract_kernel(
    B_exp: tl.pointer_type(tl.float32),
    C_exp: tl.pointer_type(tl.float32),
    G_out: tl.pointer_type(tl.float32),
    Bsz: tl.constexpr, Nsz: tl.constexpr, S: tl.constexpr, H: tl.constexpr, D: tl.constexpr
):
    b = tl.program_id(0)
    n = tl.program_id(1)
    i = tl.program_id(2)
    j = tl.program_id(3)
    h = tl.program_id(4)

    acc = 0.0
    for d in range(D):
        B_ptr = B_exp + (((b * Nsz) + n) * S + i) * H * D + h * D + d
        C_ptr = C_exp + (((b * Nsz) + n) * S + j) * H * D + h * D + d
        b_val = tl.load(B_ptr)
        c_val = tl.load(C_ptr)
        acc += b_val * c_val
    base = (((b * Nsz) + n) * S) * S * H
    out_ptr = G_out + base + i * S * H + j * H + h
    tl.store(out_ptr, acc)


# Kernel: Y_diag reduction: Y[b, n, i, h] = sum_j G[b, n, i, j, h] * hidden[b, n, j, h]
# Inputs:
#   G: [B, N, S, S, H] float32
#   hidden: [B, N, S, H, D] float32
#   Y_out: [B, N, S, H] float32
@triton.jit
def y_diag_reduce_kernel(
    G: tl.pointer_type(tl.float32),
    hidden: tl.pointer_type(tl.float32),
    Y_out: tl.pointer_type(tl.float32),
    Bsz: tl.constexpr, Nsz: tl.constexpr, S: tl.constexpr, H: tl.constexpr, D: tl.constexpr
):
    b = tl.program_id(0)
    n = tl.program_id(1)
    i = tl.program_id(2)
    h = tl.program_id(3)

    acc = 0.0
    for j in range(S):
        g_ptr = G + (((b * Nsz) + n) * S) * S * H + i * S * H + j * H + h
        g_val = tl.load(g_ptr)
        hid_ptr = hidden + (((b * Nsz) + n) * S) * H * D + j * H * D + h * D
        v = tl.load(hid_ptr)  # vector of length D
        acc += tl.sum(v * g_val)
    out_ptr = Y_out + (((b * Nsz) + n) * S) * H + i * H + h
    tl.store(out_ptr, acc)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states: torch.Tensor, A_cumsum: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        """
        Compute Y_diag = sum_j (G * L)[i,j] * hidden_states[..., j] over j, where:
          - L = exp(cumsum(masked A)) with lower triangle (diagonal=-1)
          - G[b,n,i,j,h] = sum_d C[b,n,i,h,d] * B[b,n,j,h,d]
          - hidden_states: [B, N, S, H, D]
          - A_cumsum:      [B, H, N, S]
          - B:             [B, N, S, H, D]
          - C:             [B, N, S, H, D]
        Returns: [B, N, S, H] in bfloat16.
        """
        Bsz, Nsz, S, H, D = hidden_states.shape
        assert A_cumsum.shape == (Bsz, H, Nsz, S), "A_cumsum must have shape [B, H, N, S]"
        # Ensure B/C are expanded to H as in the original: expand from G=N_GROUPS to H=NUM_HEADS
        assert B.shape[3] == H and C.shape[3] == H, "B/C must be expanded to num_heads (H)"

        device = hidden_states.device

        # Ensure inputs are contiguous and float32 for kernels
        A = A_cumsum.to(torch.float32).contiguous()  # [B, H, N, S]
        B_exp = B.to(torch.float32).contiguous()    # [B, N, S, H, D]
        C_exp = C.to(torch.float32).contiguous()    # [B, N, S, H, D]

        # 1) Triton: Compute L = exp(cumsum(masked A)) with tril(diagonal=-1)
        L = torch.empty((Bsz, H, Nsz, S, S), dtype=torch.float32, device=device)
        cumsum_exp_tril_kernel[(Bsz, H, Nsz)](A, L, Bsz, H, Nsz, S)

        # 2) Triton: G contraction
        G = torch.empty((Bsz, Nsz, S, S, H), dtype=torch.float32, device=device)


def run(*args):
    return ModelNew()(*args)
