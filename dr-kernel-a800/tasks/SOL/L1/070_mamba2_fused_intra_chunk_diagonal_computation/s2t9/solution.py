import torch
import triton
import triton.language as tl

# Constants as in the original model
CHUNK_SIZE = 128
NUM_HEADS = 32
N_GROUPS = 8

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
    Compute G[i, j, h] = sum_s (C[b, i, k, h, s] * B[b, j, k, h, s]) for all k. Since k dimension is CHUNK_SIZE, we loop over k and accumulate.
    Grid: (B, num_chunks, CHUNK_SIZE, NUM_HEADS) -> each program computes G[i, j, h] by looping over s in HEAD_DIM tiles and k in CHUNK_SIZE.
    """
    G_val = tl.zeros((), dtype=tl.float32)

    # Loop over k
    for k in range(CHUNK_SIZE):
        # Accumulate dot product over state_dim in tiles
        for s_start in range(0, HEAD_DIM, 16):
            offs = s_start + tl.arange(0, 16)
            mask = offs < HEAD_DIM
            # Load C and B vectors for this (i, k, h) and s tile, compute dot
            dot_val = tl.zeros((), dtype=tl.float32)
            for s_idx in range(16):
                s = s_start + s_idx
                C_ptr_j = C_ptr + b_id * C_bs + i_id * C_cs + k * C_cd + h_id * C_w + s * C_sd
                B_ptr_j = B_ptr + b_id * B_bs + j_id * B_cs + k * B_cd + h_id * B_w + s * B_sd
                C_val = tl.load(C_ptr_j)
                B_val = tl.load(B_ptr_j)
                dot_val += C_val * B_val
            G_val += dot_val

    # Store G_val into G[b, i, j, h]
    G_ptrs = G_ptr + b_id * G_bs + i_id * G_cs + j_id * G_cs + h_id * G_h
    tl.store(G_ptrs, G_val)

@triton.jit
def multiply_LG_kernel(
    G_ptr, L_ptr, M_ptr,
    G_bs, G_cs, G_cd, G_w, G_h,  # G strides: (B, num_chunks, CHUNK_SIZE, CHUNK_SIZE, NUM_HEADS)
    L_bs, L_hs, L_cs, L_cd, L_w,  # L strides: (B, NUM_HEADS, num_chunks, CHUNK_SIZE, CHUNK_SIZE)
    M_bs, M_cs, M_cd, M_w, M_h,   # M strides: (B, num_chunks, CHUNK_SIZE, CHUNK_SIZE, NUM_HEADS)
    b_id: tl.constexpr,
    i_id: tl.constexpr,
    h_id: tl.constexpr,
):
    """
    Compute M[b, i, k, j, h] = G[b, i, k, j, h] * L[b, h, i, k, j].
    Grid: (B, num_chunks, CHUNK_SIZE, CHUNK_SIZE, NUM_HEADS) -> each program computes one element M[b, i, k, j, h].
    """
    j = tl.program_id(1)  # j index
    k = tl.program_id(2)  # k index

    # Load G[b, i, k, j, h] and L[b, h, i, k, j]
    G_val = tl.load(G_ptr + b_id * G_bs + i_id * G_cs + k * G_cd + j * G_w + h_id * G_h)
    L_val = tl.load(L_ptr + b_id * L_bs + h_id * L_hs + i_id * L_cs + k * L_cd + j * L_w)
    M_val = G_val * L_val

    # Store M[b, i, k, j, h]
    M_ptrs = M_ptr + b_id * M_bs + i_id * M_cs + k * M_cd + j * M_w + h_id * M_h
    tl.store(M_ptrs, M_val)

@triton.jit
def contract_M_hidden_into_Ydiag_kernel(
    M_ptr, hidden_ptr, Y_ptr,
    M_bs, M_cs, M_cd, M_w, M_h,  # M strides: (B, num_chunks, CHUNK_SIZE, CHUNK_SIZE, NUM_HEADS)
    H_bs, H_cs, H_cd, H_hs, H_d,  # hidden strides: (B, num_chunks, CHUNK_SIZE, NUM_HEADS, HEAD_DIM)
    Y_bs, Y_cs, Y_hd, Y_hs, Y_d,  # output strides: (B, num_chunks, CHUNK_SIZE, NUM_HEADS, HEAD_DIM)
    b_id: tl.constexpr,
    i_id: tl.constexpr,
    h_id: tl.constexpr,
):
    """
    Compute Y_diag[b, i, k, h, d] = sum_j ( M[b, i, k, j, h] * hidden[b, i, j, h, d] ).
    We vectorize across d in tiles. Since Triton can't easily load/store vectors across d in mixed dims, we implement a simple scalar loop over d.
    Grid: (B, num_chunks, CHUNK_SIZE, NUM_HEADS)
    """
    k = tl.program_id(1)
    h = tl.program_id(2)

    # Loop over d in tiles; HEAD_DIM=128, constexpr
    for d_start in range(0, 128, 16):
        d_offs = d_start + tl.arange(0, 16)
        acc = tl.zeros((16,), dtype=tl.float32)
        # Loop over j to contract
        for j in range(CHUNK_SIZE):
            # Load M[b, i, k, j, h] for each d_off
            for t in range(16):
                d_idx = d_start + t
                if d_idx < 128:
                    M_val = tl.load(M_ptr + b_id * M_bs + i_id * M_cs + k * M_cd + j * M_w + h * M_h)
                    H_val = tl.load(hidden_ptr + b_id * H_bs + i_id * H_cs + j * H_cd + h * H_hs + d_idx * H_d)
                    acc[t] += M_val * H_val
        # Store acc into Y[b, i, k, h, d_offs]
        for t in range(16):
            d_idx = d_start + t
            if d_idx < 128:
                Y_ptrs = Y_ptr + b_id * Y_bs + i_id * Y_cs + k * Y_hd + h * Y_hs + d_idx * Y_d
                tl.store(Y_ptrs, acc[t])

# Example helper to expand heads (repeat_interleave) — will be done in host before kernel call
def expand_to_heads(x: torch.Tensor, num_heads: int):
    # x: [B, num_chunks, CHUNK_SIZE, N_GROUPS, HEAD_DIM], expand groups to num_heads by repeating N_GROUPS -> NUM_HEADS
    # NUM_HEADS // N_GROUPS = 4, so repeat_interleave by 4 along group dim
    return x.repeat_interleave(4, dim=3)

class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, A_cumsum: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        """
        hidden_states: [B, num_chunks, CHUNK_SIZE, NUM_HEADS, HEAD_DIM]
        A_cumsum: [B, NUM_HEADS, num_chunks, CHUNK_SIZE]
        B: [B, num_chunks, CHUNK_SIZE, N_GROUPS, HEAD_DIM]
        C: [B, num_chunks, CHUNK_SIZE, N_GROUPS, HEAD_DIM]
        Output: [B, num_chunks, CHUNK_SIZE, NUM_HEADS, HEAD_DIM]
        """
        B_size, num_chunks, CHUNK_SIZE, NUM_HEADS, HEAD_DIM = hidden_states.shape
        # Ensure tensors are on CUDA (forward assumes inputs are on GPU as per requirement)
        device = hidden_states.device

        # Expand B and C to NUM_HEADS by repeating groups (repeat_interleave by 4)
        B_exp = expand_to_heads(B, NUM_HEADS)
        C_exp = expand_to_heads(C, NUM_HEADS)

        # Allocate L: [B, NUM_HEADS, num_chunks, CHUNK_SIZE, CHUNK_SIZE] float32
        L = torch.empty((B_size, NUM_HEADS, num_chunks, CHUNK_SIZE, CHUNK_SIZE), device=device, dtype=torch.float32)

        # Launch build_L_kernel
        grid_L = (B_size, NUM_HEADS, num_chunks, 1, CHUNK_SIZE)
        build_L_kernel[grid_L](
            A_cumsum, L,
            A_cumsum.stride(0), A_cumsum.stride(1), A_cumsum.stride(2), A_cumsum.stride(3),
            L.stride(0), L.stride(1), L.stride(2), L.stride(3), L.stride(4),
            b_id=B_size, h_id=NUM_HEADS, i_id=num_chunks, CHUNK_SIZE=CHUNK_SIZE,
            num_warps=4, num_stages=2
        )

        # Allocate G: [B, num_chunks, CHUNK_SIZE, CHUNK_SIZE, NUM_HEADS] float32
        G = torch.empty((B_size, num_chunks, CHUNK_SIZE, CHUNK_SIZE, NUM_HEADS), device=device, dtype=torch.float32)

        # Launch compute_G_kernel: grid over (B, num_chunks, CHUNK_SIZE, NUM_HEADS)
        grid_G = (B_size, num_chunks, CHUNK_SIZE, NUM_HEADS)
        compute_G_kernel[grid_G](
            B_exp, C_exp, G,
            B_exp.stride(0), B_exp.stride(1), B_exp.stride(2), B_exp.stride(3), B_exp.stride(4),
            C_exp.stride(0), C_exp.stride(1), C_exp.stride(2), C_exp.stride(3), C_exp.stride(4),
            G.stride(0), G.stride(1), G.stride(2), G.stride(3), G.stride(4),
            b_id=B_size, i_id=num_chunks, j_id=CHUNK_SIZE, h_id=NUM_HEADS, HEAD_DIM=HEAD_DIM,
            num_warps=4, num_stages=2
        )

        # Allocate M: [B, num_chunks, CHUNK_SIZE, CHUNK_SIZE, NUM_HEADS] float32
        M = torch.empty_like(G)

        # Launch multiply_LG_kernel: grid over (B, num_chunks, CHUNK_SIZE, CHUNK_SIZE, NUM_HEADS)
        grid_M = (B_size, num_chunks, CHUNK_SIZE, CHUNK_SIZE, NUM_HEADS)
        multiply_LG_kernel[grid_M](
            G, L, M,
            G.stride(0), G.stride(1), G.stride(2), G.stride(3), G.stride(4),
            L.stride(0), L.stride(1), L.stride(2), L.stride(3), L.stride(4),
            M.stride(0), M.stride(1), M.stride(2), M.stride(3), M.stride(4),
            b_id=B_size, i_id=num_chunks, h_id=NUM_HEADS,
            num_warps=4, num_stages=2
        )

        # Allocate Y output: [B, num_chunks, CHUNK_SIZE, NUM_HEADS, HEAD_DIM] float32
        Y = torch.empty((B_size, num_chunks, CHUNK_SIZE, NUM_HEADS, HEAD_DIM), device=device, dtype=torch.float32)

        # Launch contract_M_hidden_into_Ydiag_kernel: grid over (B, num_chunks, CHUNK_SIZE, NUM_HEADS)
        grid_Y = (B_size, num_chunks, CHUNK_SIZE, NUM_HEADS)
        contract_M_hidden_into_Ydiag_kernel[grid_Y](
            M, hidden_states, Y,
            M.stride(0), M.stride(1), M.stride(2), M.stride(3), M.stride(4),
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2), hidden_states.stride(3), hidden_states.stride(4),
            Y.stride(0), Y.stride(1), Y.stride(2), Y.stride(3), Y.stride(4),
            b_id=B_size, i_id=num_chunks, h_id=NUM_HEADS,
            num_warps=4, num_stages=2
        )

        # Return Y (float32), matching the intent of the original code’s computation. If strict bfloat16 is required, cast at the end.
        # Y shape: [B, num_chunks, CHUNK_SIZE, NUM_HEADS, HEAD_DIM], dtype: float32
        return Y


def run(*args):
    return ModelNew()(*args)
