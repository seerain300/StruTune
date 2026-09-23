import torch
import triton
import triton.language as tl


# 1) Triton: generate lower-triangular mask M_lower of shape [S, S] with diagonal=-1 (int8: 1 for True, 0 for False).
# We create a 1D buffer of length S*S and interpret index k as (i,j) via i=k//S, j=k%S.
@triton.jit
def tril_mask_kernel(M: tl.pointer_type(tl.int8), S: tl.int32):
    pid = tl.program_id(0)
    if pid >= S * S:
        return
    j = pid % S
    i = pid // S
    val = 1 if j <= i else 0
    tl.store(M + pid, val)


# 2) Triton: compute L = exp(cumsum(masked A)) for each (b, h, n, i, j) with j <= i (diagonal=-1).
# A: [B, H, N, S], L: [B, H, N, S, S].
# We avoid actual 5D expand; in-kernel, for each i, compute prefix sum over j of A[b,h,n,j], zero for j > i, then exp.
@triton.jit
def cumsum_exp_tril_kernel(A: tl.pointer_type(tl.float32), L: tl.pointer_type(tl.float32),
                           Bsz: tl.int32, H: tl.int32, N: tl.int32, S: tl.int32):
    pid = tl.program_id(0)  # linear index over B*H*N
    if pid >= Bsz * H * N:
        return
    b = pid // (H * N)
    rem = pid % (H * N)
    h = rem // N
    n = rem % N

    # For each i in [0..S-1], compute prefix sum over j of A[b,h,n,j], zero when j > i, then exp(prefix)
    # We store to L[b,h,n,i,j].
    for i in range(0, S):
        acc = 0.0
        for j in range(0, S):
            # A[b,h,n,j] at linearized pointer: ((b*H + h)*N + n)*S + j
            ptr = ((b * H + h) * N + n) * S + j
            a = tl.load(A + ptr)  # float32
            # j <= i ensures lower-triangular mask; we include j > i by masking it to 0 via condition.
            include = 1 if j <= i else 0
            acc += a * include
            # L[b,h,n,i,j] = exp(acc)
            out_ptr = ((b * H + h) * N * S) + (i * S) + j
            tl.store(L + out_ptr, tl.exp(acc))


# 3) Triton: G contraction G[b,n,i,j,h] = sum_d C[b,n,i,h,d] * B[b,n,j,h,d].
# We launch with grid=(B,N,H,S*S). Each program handles one (i,j) pair for all (b,n,h).
@triton.jit
def g_contract_kernel(B: tl.pointer_type(tl.float32), C: tl.pointer_type(tl.float32), G: tl.pointer_type(tl.float32),
                      Bsz: tl.int32, N: tl.int32, S: tl.int32, H: tl.int32, D: tl.int32):
    pid = tl.program_id(0)  # over B*N*H*(S*S)
    if pid >= Bsz * N * H * (S * S):
        return
    rem = pid % (N * H * S * S)
    h = rem % H
    ij = rem // (N * H * S * S)  # but we want (i,j) from pid; rem structure is pid = b*(N*H*S*S) + n*(H*S*S) + h*(S*S) + ij
    # Recompute in proper grouping:
    b = pid // (N * H * S * S)
    nn = (pid // (H * S * S)) % N
    tmp = pid % (H * S * S)
    h = tmp // (S * S)
    ij = tmp % (S * S)
    i = ij // S
    j = ij % S

    # Accumulate over D
    acc = 0.0
    # Loop over D dimension
    for d in range(0, D):
        # Pointer for B[b,n,j,h,d]: (((b*Bsz + b)*N + nn)*S + j)*H*D + h*D + d
        # Note: B shape [B,N,S,H,D] in row-major contiguous, but pointer math uses strides logic.
        # We need to read B[b,n,j,h,d]. Assume B is contiguous [B,N,S,H,D], then linear index:
        # ((b*Bsz + b)*N*S*H*D) + (nn*N*S*H*D) + (j*S*H*D) + (h*H*D) + d
        # To keep it simple, we pass B as contiguous [B,N,S,H,D] and compute linear index accordingly.
        # Compute base excluding d
        base = ((b * Bsz + b) * N + nn) * S * H * D + j * H * D
        # Add h contribution
        base += h * H * D
        # Add d
        val_B = tl.load(B + base + d)
        # Similarly for C[b,n,i,h,d]
        base_C = ((b * Bsz + b) * N + nn) * S * H * D + i * H * D
        base_C += h * H * D
        val_C = tl.load(C + base_C + d)
        acc += val_B * val_C

    # Store G[b,n,i,j,h] at linearized index: ((b*Bsz + b)*N*S*S*H) + (nn*N*S*S) + (i*S*S) + (j*S) + h
    out_idx = ((b * Bsz + b) * N + nn) * S * S * H + (i * S * S) + (j * S) + h
    tl.store(G + out_idx, acc)


# 4) Triton: M = G * L (elementwise). L is [B,H,N,S,S], G is [B,N,S,S,H].
# We permute L to [B,N,S,S,H] and multiply.
@triton.jit
def m_mul_kernel(G: tl.pointer_type(tl.float32), L: tl.pointer_type(tl.float32), M: tl.pointer_type(tl.float32),
                 Bsz: tl.int32, N: tl.int32, S: tl.int32, H: tl.int32):
    pid = tl.program_id(0)  # over B*N*H*S*S
    if pid >= Bsz * N * H * (S * S):
        return
    b = pid // (N * H * S * S)
    nn = (pid // (H * S * S)) % N
    h = (pid // (S * S)) % H
    ij = pid % (S * S)
    i = ij // S
    j = ij % S

    # Load G[b,n,i,j,h] and L[b,h,n,i,j]
    g_ptr = ((b * Bsz + b) * N + nn) * S * S * H + (i * S * S) + (j * S) + h
    g_val = tl.load(G + g_ptr)

    l_ptr = ((b * H + h) * N * S * S) + (nn * S * S) + (i * S) + j
    l_val = tl.load(L + l_ptr)

    out_ptr = ((b * N + nn) * S * S * H) + (i * S * S) + (j * S) + h
    tl.store(M + out_ptr, g_val * l_val)


# 5) Triton: Y_diag reduction. Compute Y[b,n,i,h] = sum_j M[b,n,i,j,h] * hidden[b,n,j,h].
# hidden: [B,N,S,H,D], M: [B,N,S,S,H].
@triton.jit
def y_diag_reduce_kernel(M: tl.pointer_type(tl.float32), hidden: tl.pointer_type(tl.float32), Y: tl.pointer_type(tl.float32),
                         Bsz: tl.int32, N: tl.int32, S: tl.int32, H: tl.int32, D: tl.int32):
    pid = tl.program_id(0)  # over B*N*H*S
    if pid >= Bsz * N * H * S:
        return
    b = pid // (N * H * S)
    n = (pid // (H * S)) % N
    h = (pid // S) % H
    i = pid % S

    acc = 0.0
    # Loop over j in tiles of size BLOCK_J (e.g., 128)
    BLOCK_J = 128
    for j0 in range(0, S, BLOCK_J):
        for j in range(0, BLOCK_J):
            jj = j0 + j
            if jj >= S:
                break
            # M[b,n,i,j,h]
            m_ptr = ((b * N + n) * S * S * H) + (i * S * S) + (jj * S) + h
            m_val = tl.load(M + m_ptr)
            # hidden[b,n,jj,h,:] dot with 1 (no scaling from M), effectively m_val * sum(hidden[b,n,jj,h,:])
            # But since M is scalar multiplier per j, we compute hidden vector and multiply.
            sum_hid = 0.0
            for dd in range(0, D):
                hid_ptr = ((b * N + n) * S * H * D) + (jj * H * D) + (h * H * D) + dd
                val = tl.load(hidden + hid_ptr)
                sum_hid += val
            acc += m_val * sum_hid

    # Store Y[b,n,i,h]
    y_ptr = ((b * N + n) * S * H) + (i * H) + h
    tl.store(Y + y_ptr, acc)


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
        # Ensure inputs are contiguous and float32 for Triton kernels
        Bsz, N, S, H, D = hidden_states.shape
        assert A_cumsum.shape == (Bsz, H, N, S), "A_cumsum must have shape [B, H, N, S]"
        assert B.shape == (Bsz, N, S, H, D), "B must have shape [B, N, S, H, D]"
        assert C.shape == (Bsz, N, S, H, D), "C must have shape [B, N, S, H, D]"

        device = hidden_states.device
        A = A_cumsum.to(torch.float32).contiguous()  # [B,H,N,S]
        B_exp = B.to(torch.float32).contiguous()     # [B,N,S,H,D]
        C_exp = C.to(torch.float32).contiguous()     # [B,N,S,H,D]

        # 1) Triton: generate mask M_lower (int8, length S*S)
        M_lower = torch.empty(S * S, dtype=torch.int8, device=device)
        tril_mask_kernel[(S * S,)](M_lower, S)

        # 2) Triton: compute L = exp(cumsum(masked A)) with diagonal=-1
        L = torch.empty((Bsz, H, N, S, S), dtype=torch.float32, device=device)
        cumsum_exp_tril_kernel[(Bsz * H * N,)](A, L, Bsz, H, N, S)

        # 3) Triton: G contraction G[b,n,i,j,h] = sum_d C[b,n,i,h,d] * B[b,n,j,h,d]
        G = torch.empty((Bsz, N, S, S, H), dtype=torch.float32, device=device)
        g_contract_kernel[(Bsz * N * H * (S * S),)](B_exp, C_exp, G, Bsz, N, S, H, D)

        # 4) Triton: M = G * L
        # permute L to [B,N,S,S,H] for multiplication
        L_perm = L.permute(0, 2, 3, 4, 1).contiguous()  # [B,N,S,S,H]
        M = torch.empty((Bsz, N, S, S, H), dtype=torch.float32, device=device)
        m_mul_kernel[(Bsz * N * H * (S * S),)](G, L_perm, M, Bsz, N, S, H)

        # 5) Triton: Y_diag reduction
        hidden = hidden_states.to(torch.float32).contiguous()  # [B,N,S,H,D]
        Y = torch.empty((Bsz, N, S, H), dtype=torch.float32, device=device)
        y_diag_reduce_kernel[(Bsz * N * H * S,)](M, hidden, Y, Bsz, N, S, H, D)

        # Return in bfloat16 to match original behavior
        return Y.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
