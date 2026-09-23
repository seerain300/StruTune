import torch
import triton
import triton.language as tl

# Constants as per original code
CHUNK_SIZE = 128   # hidden.shape[2] and B/C.shape[2]
NUM_HEADS = 32     # hidden.shape[3]
N_GROUPS = 8       # B/C.shape[3]
STATE_SIZE = 128   # B/C.shape[4]
HEAD_DIM = 128     # hidden.shape[4]

# 1) Triton kernel to build L: lower-triangular causal mask from A_cumsum
#    L[b, i, k, j, h] = exp(cumsum(A[b, h, i, 0:k+1])) if i >= j else 0
@triton.jit
def build_L_kernel(
    A_ptr, L_ptr,
    A_stride0, A_stride1, A_stride2, A_stride3,
    L_stride0, L_stride1, L_stride2, L_stride3, L_stride4,
    B_batch, num_chunks
):
    b = tl.program_id(0)
    i = tl.program_id(1)
    h = tl.program_id(2)  # head index
    # We will compute L[i, :, :, h] as a 2D matrix [CHUNK_SIZE, CHUNK_SIZE] for this (b, i, h)
    for j in range(CHUNK_SIZE):  # source index
        # compute cumulative sum over A[b, h, i, 0:j] and then exp
        cum = tl.zeros((), dtype=tl.float32)
        for t in range(j + 1):  # j starts at 0; j+1 elements up to j
            a_val = tl.load(A_ptr + b * A_stride0 + h * A_stride1 + i * A_stride2 + t * A_stride3)
            cum += a_val
        exp_val = tl.exp(cum)
        # For lower-triangular, write exp_val at (i=j', j=j)
        for j2 in range(CHUNK_SIZE):  # target index
            if j2 <= j:
                L_ptr_addr = L_ptr + b * L_stride0 + i * L_stride1 + j2 * L_stride2 + j * L_stride3 + h * L_stride4
                tl.store(L_ptr_addr, exp_val)
            else:
                # upper triangle: zero
                L_ptr_addr = L_ptr + b * L_stride0 + i * L_stride1 + j2 * L_stride2 + j * L_stride3 + h * L_stride4
                tl.store(L_ptr_addr, 0.0)

# Helper to allocate L and launch build_L_kernel
def build_L(A_cumsum: torch.Tensor) -> torch.Tensor:
    A = A_cumsum.contiguous()
    L = torch.empty((A.shape[0], A.shape[2], CHUNK_SIZE, CHUNK_SIZE, A.shape[1]),
                    device=A.device, dtype=torch.float32)
    grid = (A.shape[0], A.shape[2], A.shape[1])
    build_L_kernel[grid](
        A, L,
        A.stride(0), A.stride(1), A.stride(2), A.stride(3),
        L.stride(0), L.stride(1), L.stride(2), L.stride(3), L.stride(4),
        A.shape[0], A.shape[2],
        num_warps=4,
        num_stages=2
    )
    return L

# 2) Triton kernel to compute G[i, j, h] = sum_k sum_s C[i, k, h, s] * B[j, k, h, s]
#    We expand B and C across heads: NUM_HEADS // N_GROUPS = 4. We pass expanded tensors.
@triton.jit
def compute_G_per_i_h_kernel(
    B_ptr, C_ptr, G_ptr,
    B_shape, C_shape,
    B_stride0, B_stride1, B_stride2, B_stride3, B_stride4,
    C_stride0, C_stride1, C_stride2, C_stride3, C_stride4,
    G_stride0, G_stride1, G_stride2, G_stride3,  # G: [B, num_chunks, CHUNK_SIZE, NUM_HEADS]
    B_batch, num_chunks
):
    b = tl.program_id(0)
    i = tl.program_id(1)
    h = tl.program_id(2)  # h in 0..NUM_HEADS-1

    # Initialize G[j, h] = 0 for all j
    G_jH = tl.zeros([CHUNK_SIZE], dtype=tl.float32)

    # Loop over k (chunk positions) and s (state dimension) in compile-time ranges
    for k in range(CHUNK_SIZE):
        for s in range(STATE_SIZE):
            # Compute B[j, k, h, s] for all j as a vector
            j_offsets = tl.arange(0, CHUNK_SIZE)
            base_B = B_ptr + b * B_stride0 + i * B_stride1 + k * B_stride2 + h * B_stride3 + s * B_stride4
            B_vals = tl.load(base_B + j_offsets * B_stride2)

            # Compute C[i, k, h, s] as scalar
            C_scalar = tl.load(C_ptr + b * C_stride0 + i * C_stride1 + k * C_stride2 + h * C_stride3 + s * C_stride4)

            # Accumulate G_jH[j] += B_vals[j] * C_scalar
            G_jH += B_vals * C_scalar

    # Store G[i, :, h] across j dimension
    for j in range(CHUNK_SIZE):
        G_ptr_addr = G_ptr + b * G_stride0 + i * G_stride1 + j * G_stride2 + h * G_stride3
        tl.store(G_ptr_addr, G_jH[j])

def compute_G_expanded(B_exp: torch.Tensor, C_exp: torch.Tensor) -> torch.Tensor:
    """
    Compute G[b, i, j, h] for all (b, i, j, h), return [B, num_chunks, CHUNK_SIZE, NUM_HEADS]
    """
    B_exp = B_exp.contiguous()
    C_exp = C_exp.contiguous()
    G = torch.empty((B_exp.shape[0], B_exp.shape[1], CHUNK_SIZE, NUM_HEADS),
                    device=B_exp.device, dtype=torch.float32)

    grid = (B_exp.shape[0], B_exp.shape[1], NUM_HEADS)
    compute_G_per_i_h_kernel[grid](
        B_exp, C_exp, G,
        B_exp.shape, C_exp.shape,
        B_exp.stride(0), B_exp.stride(1), B_exp.stride(2), B_exp.stride(3), B_exp.stride(4),
        C_exp.stride(0), C_exp.stride(1), C_exp.stride(2), C_exp.stride(3), C_exp.stride(4),
        G.stride(0), G.stride(1), G.stride(2), G.stride(3),
        B_exp.shape[0], B_exp.shape[1],
        num_warps=4,
        num_stages=2
    )
    return G

# 3) Triton kernel to compute M = G * L element-wise: M[i, j, h] = G[i, j, h] * L[i, j, h]
@triton.jit
def multiply_G_L_kernel(
    G_ptr, L_ptr, M_ptr,
    G_shape, L_shape, M_shape,
    G_stride0, G_stride1, G_stride2, G_stride3,  # G: [B, num_chunks, CHUNK_SIZE, NUM_HEADS]
    L_stride0, L_stride1, L_stride2, L_stride3, L_stride4,  # L: [B, num_chunks, CHUNK_SIZE, CHUNK_SIZE, NUM_HEADS]
    M_stride0, M_stride1, M_stride2, M_stride3, M_stride4,  # M: [B, num_chunks, CHUNK_SIZE, CHUNK_SIZE, NUM_HEADS]
    B_batch, num_chunks
):
    b = tl.program_id(0)
    i = tl.program_id(1)
    h = tl.program_id(2)
    for j in range(CHUNK_SIZE):
        G_val = tl.load(G_ptr + b * G_stride0 + i * G_stride1 + j * G_stride2 + h * G_stride3)
        for j2 in range(CHUNK_SIZE):
            L_val = tl.load(L_ptr + b * L_stride0 + i * L_stride1 + j2 * L_stride2 + j * L_stride3 + h * L_stride4)
            M_ptr_addr = M_ptr + b * M_stride0 + i * M_stride1 + j2 * M_stride2 + j * M_stride3 + h * M_stride4
            tl.store(M_ptr_addr, G_val * L_val)

def multiply_G_L(G: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
    """
    Compute M[b, i, j, j2, h] = G[b, i, j, h] * L[b, i, j2, j, h]
    Returns M of shape [B, num_chunks, CHUNK_SIZE, CHUNK_SIZE, NUM_HEADS], dtype float32.
    """
    B, num_chunks, _, NUM_HEADS = G.shape
    M = torch.empty((B, num_chunks, CHUNK_SIZE, CHUNK_SIZE, NUM_HEADS), device=G.device, dtype=torch.float32)

    grid = (B, num_chunks, NUM_HEADS)
    multiply_G_L_kernel[grid](
        G, L, M,
        G.shape, L.shape, M.shape,
        G.stride(0), G.stride(1), G.stride(2), G.stride(3),
        L.stride(0), L.stride(1), L.stride(2), L.stride(3), L.stride(4),
        M.stride(0), M.stride(1), M.stride(2), M.stride(3), M.stride(4),
        B, num_chunks,
        num_warps=4,
        num_stages=2
    )
    return M

# 4) Triton kernel to contract M with hidden to produce Y_diag
#    Y_diag[b, i, k, h, d] = sum_j M[b, i, j, k, h] * hidden[b, i, k, j, h, d]
@triton.jit
def contract_M_hidden_kernel(
    M_ptr, hidden_ptr, Y_ptr,
    M_shape, hidden_shape, Y_shape,
    M_stride0, M_stride1, M_stride2, M_stride3, M_stride4,  # M: [B, num_chunks, CHUNK_SIZE, CHUNK_SIZE, NUM_HEADS]
    hidden_stride0, hidden_stride1, hidden_stride2, hidden_stride3, hidden_stride4, hidden_stride5,
    Y_stride0, Y_stride1, Y_stride2, Y_stride3, Y_stride4,  # Y: [B, num_chunks, CHUNK_SIZE, NUM_HEADS, HEAD_DIM]
    B_batch, num_chunks
):
    b = tl.program_id(0)
    i = tl.program_id(1)
    k = tl.program_id(2)  # chunk index for hidden
    h = tl.program_id(3)  # head index
    d = tl.program_id(4)  # head_dim index

    acc = tl.zeros((), dtype=tl.float32)

    for j in range(CHUNK_SIZE):
        # M[b, i, j, k, h]
        M_val = tl.load(M_ptr + b * M_stride0 + i * M_stride1 + j * M_stride2 + k * M_stride3 + h * M_stride4)
        # hidden[b, i, k, j, h, d]
        hidden_addr = hidden_ptr + b * hidden_stride0 + i * hidden_stride1 + k * hidden_stride2 + j * hidden_stride3 + h * hidden_stride4 + d * hidden_stride5
        hidden_val = tl.load(hidden_addr)
        acc += M_val * hidden_val

    Y_addr = Y_ptr + b * Y_stride0 + i * Y_stride1 + k * Y_stride2 + h * Y_stride3 + d * Y_stride4
    tl.store(Y_addr, acc.to(tl.bfloat16))

def contract_to_Ydiag(M: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
    """
    Compute Y_diag[b, i, k, h, d] = sum_j M[b, i, j, k, h] * hidden[b, i, k, j, h, d]
    Returns tensor of shape [B, num_chunks, CHUNK_SIZE, NUM_HEADS, HEAD_DIM], dtype bfloat16.
    """
    hidden = hidden_states.contiguous()
    Y = torch.empty((hidden.shape[0], hidden.shape[1], CHUNK_SIZE, NUM_HEADS, HEAD_DIM),
                    device=hidden.device, dtype=torch.bfloat16)

    grid = (hidden.shape[0], hidden.shape[1], CHUNK_SIZE, NUM_HEADS, HEAD_DIM)
    contract_M_hidden_kernel[grid](
        M, hidden, Y,
        M.shape, hidden.shape, Y.shape,
        M.stride(0), M.stride(1), M.stride(2), M.stride(3), M.stride(4),
        hidden.stride(0), hidden.stride(1), hidden.stride(2), hidden.stride(3), hidden.stride(4), hidden.stride(5),
        Y.stride(0), Y.stride(1), Y.stride(2), Y.stride(3), Y.stride(4),
        hidden.shape[0], hidden.shape[1],
        num_warps=4,
        num_stages=2
    )
    return Y

class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                A_cumsum: torch.Tensor,
                B: torch.Tensor,
                C: torch.Tensor) -> torch.Tensor:
        # Step 1: Build L using Triton
        L = build_L(A_cumsum)  # [B, num_chunks, CHUNK_SIZE, CHUNK_SIZE, NUM_HEADS], float32

        # Step 2: Expand B and C across heads and compute G in Triton
        # NUM_HEADS // N_GROUPS = 4
        B_exp = B.repeat_interleave(NUM_HEADS // N_GROUPS, dim=3)  # [B, num_chunks, CHUNK_SIZE, NUM_HEADS, STATE_SIZE]
        C_exp = C.repeat_interleave(NUM_HEADS // N_GROUPS, dim=3)  # [B, num_chunks, CHUNK_SIZE, NUM_HEADS, STATE_SIZE]
        G = compute_G_expanded(B_exp, C_exp)  # [B, num_chunks, CHUNK_SIZE, NUM_HEADS], float32

        # Step 3: Multiply G and L in Triton to get M
        M = multiply_G_L(G, L)  # [B, num_chunks, CHUNK_SIZE, CHUNK_SIZE, NUM_HEADS], float32

        # Step 4: Contract M with hidden in Triton to produce Y_diag
        Y_diag = contract_to_Ydiag(M, hidden_states)  # [B, num_chunks, CHUNK_SIZE, NUM_HEADS, HEAD_DIM], bfloat16

        return Y_diag


def run(*args):
    return ModelNew()(*args)
