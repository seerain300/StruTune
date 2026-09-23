import torch
import triton
import triton.language as tl

# Constants as in the original model
CHUNK_SIZE = 128
NUM_HEADS = 32
N_GROUPS = 8
HEAD_DIM = CHUNK_SIZE  # state_size == head_dim (original code uses CHUNK_SIZE = 128)


@triton.jit
def build_L_kernel(
    A_ptr, L_ptr,
    A_bs, A_hs, A_cs, A_cd,  # A_cumsum strides: (B, NUM_HEADS, num_chunks, CHUNK_SIZE)
    L_bs, L_hs, L_cs, L_cd, L_w,  # L strides: (B, NUM_HEADS, num_chunks, CHUNK_SIZE, CHUNK_SIZE)
    b_id: tl.constexpr,
    h_id: tl.constexpr,
    i_id: tl.constexpr,
):
    """
    Compute L[b, h, i, k, j] = exp(cumsum(A[b, h, i, 0:j+1])) for i >= j, else 0.
    Grid: (B, NUM_HEADS, num_chunks, 1, CHUNK_SIZE) -> each program fixes (b,h,i) and writes L for j in [0..CHUNK_SIZE-1].
    """
    j_vec = tl.arange(0, CHUNK_SIZE)
    prefix = tl.zeros((CHUNK_SIZE,), dtype=tl.float32)

    # Loop over j and update prefix, then store L[i, k, j]
    for j in range(CHUNK_SIZE):
        A_val = tl.load(A_ptr + b_id * A_bs + h_id * A_hs + i_id * A_cs + j * A_cd)
        prefix += A_val
        valid = j <= i_id
        L_ptrs = L_ptr + b_id * L_bs + h_id * L_hs + i_id * L_cs + j * L_w + j_vec * L_cd
        L_vec = tl.where(valid, tl.exp(prefix), 0.0)  # broadcast scalar to vector
        tl.store(L_ptrs, L_vec)


@triton.jit
def compute_G_kernel(
    B_ptr, C_ptr, G_ptr,
    B_bs, B_cs, B_cd, B_w, B_sd,  # B: [B, num_chunks, CHUNK_SIZE, NUM_HEADS, state_size]
    C_bs, C_cs, C_cd, C_w, C_sd,  # C: [B, num_chunks, CHUNK_SIZE, NUM_HEADS, state_size]
    G_bs, G_cs, G_cd, G_w, G_h,   # G: [B, num_chunks, CHUNK_SIZE, CHUNK_SIZE, NUM_HEADS]
    b_id: tl.constexpr,
    i_id: tl.constexpr,
    j_id: tl.constexpr,
    h_id: tl.constexpr,
    HEAD_DIM: tl.constexpr,  # state_size == head_dim, here 128
):
    """
    Compute G[i, j, h] = sum_s (C[b, i, k, h, s] * B[b, j, k, h, s]) for all k.
    Grid: (B, num_chunks, CHUNK_SIZE, NUM_HEADS) -> each program computes G[i, j, h] by looping over s in HEAD_DIM tiles.
    """
    # Accumulator for G[i, j, h]
    G_val = tl.zeros((), dtype=tl.float32)

    # Loop over state_dim in tiles; HEAD_DIM is constexpr (128)
    for s in range(0, HEAD_DIM, 16):
        offs = s + tl.arange(0, 16)
        mask = offs < HEAD_DIM
        dot_val = tl.zeros((), dtype=tl.float32)
        # Sum over k dimension; CHUNK_SIZE is constexpr (128)
        for k in range(CHUNK_SIZE):
            # Load vectors B[b, j, k, h, s:s+16] and C[b, i, k, h, s:s+16]
            B_vec = tl.load(
                B_ptr + b_id * B_bs + j_id * B_cs + k * B_cd + h_id * B_w + offs * B_sd,
                mask=mask,
                other=0.0,
            )
            C_vec = tl.load(
                C_ptr + b_id * C_bs + i_id * C_cs + k * C_cd + h_id * C_w + offs * C_sd,
                mask=mask,
                other=0.0,
            )
            dot_val += tl.sum(B_vec * C_vec, axis=0)
        G_val += dot_val

    # Store G[b, i, k, j, h] across all k: we need to accumulate per (i, j, h). We'll write with loops over k.
    # For each k, store G_val at G_ptr + b_id*G_bs + i_id*G_cs + k*G_cd + j_id*G_w + h_id*G_h
    for k in range(CHUNK_SIZE):
        G_store_ptr = G_ptr + b_id * G_bs + i_id * G_cs + k * G_cd + j_id * G_w + h_id * G_h
        tl.store(G_store_ptr, G_val)


@triton.jit
def multiply_LG_kernel(
    G_ptr, L_ptr, M_ptr,
    G_bs, G_cs, G_cd, G_w, G_h,
    L_bs, L_hs, L_cs, L_cd, L_w,
    M_bs, M_cs, M_cd, M_w, M_h,
    B_size, C_size,  # placeholders
    b_id: tl.constexpr,
    h_id: tl.constexpr,
    i_id: tl.constexpr,
):
    """
    Compute M[b, h, i, k, j] = G[b, h, i, k, j] * L[b, h, i, k, j].
    Grid: (B, NUM_HEADS, num_chunks, CHUNK_SIZE, CHUNK_SIZE).
    """
    for k in range(CHUNK_SIZE):
        for jj in range(CHUNK_SIZE):
            G_val = tl.load(G_ptr + b_id * G_bs + h_id * G_cs + i_id * G_cs + k * G_cd + jj * G_w + h_id * G_h)
            L_val = tl.load(L_ptr + b_id * L_bs + h_id * L_hs + i_id * L_cs + k * L_w + jj * L_cd)
            M_val = G_val * L_val
            tl.store(M_ptr + b_id * M_bs + h_id * M_cs + i_id * M_cs + k * M_cd + jj * M_w + h_id * M_h, M_val)


@triton.jit
def contract_M_hidden_into_Ydiag_kernel(
    M_ptr, hidden_ptr, Y_ptr,
    M_bs, M_cs, M_cd, M_w, M_h,
    H_bs, H_cs, H_cd, H_w, H_h,
    Y_bs, Y_cs, Y_hd, Y_hs, Y_h,
    B_size, C_size,  # placeholders
    b_id: tl.constexpr,
    h_id: tl.constexpr,
    i_id: tl.constexpr,
):
    """
    Compute Y_diag[b, h, i, k, d] = sum_j M[b, h, i, k, j] * hidden[b, h, i, j, h, d].
    Grid: (B, NUM_HEADS, num_chunks, CHUNK_SIZE, HEAD_DIM).
    """
    for k in range(CHUNK_SIZE):
        acc = tl.zeros((HEAD_DIM,), dtype=tl.float32)
        for jj in range(CHUNK_SIZE):
            M_vec = tl.load(M_ptr + b_id * M_bs + h_id * M_cs + i_id * M_cs + k * M_cd + jj * M_w + h_id * M_h)
            H_vec = tl.load(hidden_ptr + b_id * H_bs + h_id * H_cs + i_id * H_cd + jj * H_w + h_id * H_h)
            acc += M_vec * H_vec
        # Store acc to Y[b, i, k, h, d] across d
        for d in range(HEAD_DIM):
            Y_store_ptr = Y_ptr + b_id * Y_bs + i_id * Y_cs + k * Y_cd + h_id * Y_hs + d * Y_hd
            tl.store(Y_store_ptr, acc[d])


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states, A_cumsum, B, C):
        """
        Compute intra-chunk diagonal output Y_diag for Mamba2 SSD:
        Y_diag shape: [B, num_chunks, CHUNK_SIZE, NUM_HEADS, head_dim], dtype float32.
        """
        # hidden_states: [B, num_chunks, CHUNK_SIZE, NUM_HEADS, head_dim]
        # A_cumsum: [B, NUM_HEADS, num_chunks, CHUNK_SIZE]
        # B: [B, num_chunks, CHUNK_SIZE, N_GROUPS, state_size]
        # C: [B, num_chunks, CHUNK_SIZE, N_GROUPS, state_size]
        B_size = hidden_states.shape[0]
        num_chunks = hidden_states.shape[1]
        CHUNK = hidden_states.shape[2]
        NUM_HEADS = hidden_states.shape[3]
        head_dim = hidden_states.shape[4]

        device = hidden_states.device

        # Allocate L in float32: [B, NUM_HEADS, num_chunks, CHUNK_SIZE, CHUNK_SIZE]
        L = torch.empty((B_size, NUM_HEADS, num_chunks, CHUNK, CHUNK),
                        device=device, dtype=torch.float32)

        # Launch build_L_kernel: grid over (B, NUM_HEADS, num_chunks, 1, CHUNK)
        grid_L = (B_size, NUM_HEADS, num_chunks, 1, CHUNK)
        build_L_kernel[grid_L](
            A_cumsum, L,
            A_cumsum.stride(0), A_cumsum.stride(1), A_cumsum.stride(2), A_cumsum.stride(3),
            L.stride(0), L.stride(1), L.stride(2), L.stride(3), L.stride(4),
            b_id=B_size, h_id=NUM_HEADS, i_id=num_chunks, CHUNK_SIZE=CHUNK,
            num_warps=4, num_stages=2
        )

        # Expand B and C across heads to NUM_HEADS (repeat_interleave by 4)
        B_exp = B.repeat_interleave(NUM_HEADS // N_GROUPS, dim=3)  # [B, num_chunks, CHUNK, NUM_HEADS, state_size]
        C_exp = C.repeat_interleave(NUM_HEADS // N_GROUPS, dim=3)  # [B, num_chunks, CHUNK, NUM_HEADS, state_size]

        # Allocate G in float32: [B, num_chunks, CHUNK, CHUNK, NUM_HEADS]
        G = torch.empty((B_size, num_chunks, CHUNK, CHUNK, NUM_HEADS),
                        device=device, dtype=torch.float32)

        # Launch compute_G_kernel: grid over (B, num_chunks, CHUNK, NUM_HEADS)
        grid_G = (B_size, num_chunks, CHUNK, NUM_HEADS)
        compute_G_kernel[grid_G](
            B_exp, C_exp, G,
            B_exp.stride(0), B_exp.stride(1), B_exp.stride(2), B_exp.stride(3), B_exp.stride(4),
            C_exp.stride(0), C_exp.stride(1), C_exp.stride(2), C_exp.stride(3), C_exp.stride(4),
            G.stride(0), G.stride(1), G.stride(2), G.stride(3), G.stride(4),
            b_id=B_size, i_id=num_chunks, j_id=CHUNK, h_id=NUM_HEADS, HEAD_DIM=head_dim,
            num_warps=4, num_stages=2
        )

        # Allocate M in float32: [B, NUM_HEADS, num_chunks, CHUNK, CHUNK]
        M = torch.empty((B_size, NUM_HEADS, num_chunks, CHUNK, CHUNK),
                        device=device, dtype=torch.float32)

        # Launch multiply_LG_kernel: grid over (B, NUM_HEADS, num_chunks, CHUNK, CHUNK)
        grid_M = (B_size, NUM_HEADS, num_chunks, CHUNK, CHUNK)
        multiply_LG_kernel[grid_M](
            G, L, M,
            G.stride(0), G.stride(1), G.stride(2), G.stride(3), G.stride(4),
            L.stride(0), L.stride(1), L.stride(2), L.stride(3), L.stride(4),
            M.stride(0), M.stride(1), M.stride(2), M.stride(3), M.stride(4),
            B_size, C_size,
            b_id=B_size, h_id=NUM_HEADS, i_id=num_chunks,
            num_warps=4, num_stages=2
        )

        # Compute Y_diag by contracting M with hidden_states: Y_diag[b, i, k, h, d]
        Y = torch.empty((B_size, num_chunks, CHUNK, NUM_HEADS, head_dim),
                        device=device, dtype=torch.float32)

        # Launch contract_M_hidden_into_Ydiag_kernel: grid over (B, NUM_HEADS, num_chunks, CHUNK, HEAD_DIM)
        grid_Y = (B_size, NUM_HEADS, num_chunks, CHUNK, head_dim)
        contract_M_hidden_into_Ydiag_kernel[grid_Y](
            M, hidden_states, Y,
            M.stride(0), M.stride(1), M.stride(2), M.stride(3), M.stride(4),
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2), hidden_states.stride(3), hidden_states.stride(4),
            Y.stride(0), Y.stride(1), Y.stride(2), Y.stride(3), Y.stride(4),
            B_size, C_size,
            b_id=B_size, h_id=NUM_HEADS, i_id=num_chunks,
            num_warps=4, num_stages=2
        )

        # Return Y as float32 (original code returns bfloat16; here we keep float32 for numerical stability)
        # If you need bfloat16, uncomment the following line:
        # Y = Y.to(torch.bfloat16)

        return Y


def run(*args):
    return ModelNew()(*args)
