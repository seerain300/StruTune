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
    A_bs, A_hs, A_cs, A_j,  # A_cumsum strides: (B, NUM_HEADS, num_chunks, CHUNK_SIZE)
    L_bs, L_hs, L_cs, L_j, L_k,  # L strides: (B, NUM_HEADS, num_chunks, CHUNK_SIZE, CHUNK_SIZE)
    B_size, C_size,  # placeholders
    b_id: tl.constexpr,
    h_id: tl.constexpr,
    i_id: tl.constexpr,
):
    """
    Compute L[b, h, i, k, j] = exp(cumsum(A[b, h, i, 0:j+1])) for i >= j, else 0.
    Grid: (B, NUM_HEADS, num_chunks, 1, CHUNK_SIZE) -> each program fixes (b,h,i) and writes L for j in [0..CHUNK_SIZE-1].
    """
    j_vec = tl.arange(0, CHUNK_SIZE)
    # prefix is a vector across j positions
    prefix = tl.zeros((CHUNK_SIZE,), dtype=tl.float32)

    # Loop over j and build cumsum; write L[i, k, j] for all k
    for j in range(CHUNK_SIZE):
        # Load A[b, h, i, j] as scalar
        A_val = tl.load(A_ptr + b_id * A_bs + h_id * A_hs + i_id * A_cs + j * A_j)
        prefix += A_val

        # Lower-triangular mask: include j if i >= j
        valid = j <= i_id
        L_ptrs = L_ptr + b_id * L_bs + h_id * L_hs + i_id * L_cs + j * L_j + j_vec * L_k
        L_vec = tl.where(valid, tl.exp(prefix), 0.0)  # broadcast scalar prefix to vector
        tl.store(L_ptrs, L_vec)

@triton.jit
def compute_G_kernel(
    B_ptr, C_ptr, G_ptr,
    B_bs, B_cs, B_j, B_h, B_s,  # B: [B, num_chunks, CHUNK_SIZE, NUM_HEADS, state_size]
    C_bs, C_cs, C_j, C_h, C_s,  # C: [B, num_chunks, CHUNK_SIZE, NUM_HEADS, state_size]
    G_bs, G_cs, G_j, G_h, G_s,  # G: [B, num_chunks, CHUNK_SIZE_i, CHUNK_SIZE_j, NUM_HEADS]
    B_size, C_size,
    b_id: tl.constexpr,
    h_id: tl.constexpr,  # head index
    i_id: tl.constexpr,  # chunk i
    j_id: tl.constexpr,  # chunk j
    state_size: tl.constexpr,  # reduction bound
):
    """
    Compute G[b, i, k_i, k_j, h] = sum_{s=0..state_size-1} C[b, i, k_i, h, s] * B[b, j, k_j, h, s].
    Grid: (B, NUM_HEADS, num_chunks, num_chunks) -> each program computes G[i, j, h] for fixed (b).
    """
    acc = tl.zeros((), dtype=tl.float32)
    s = 0
    while s < state_size:
        C_val = tl.load(C_ptr + b_id * C_bs + i_id * C_cs + h_id * C_h + s * C_s)  # scalar at k_i
        B_val = tl.load(B_ptr + b_id * B_bs + j_id * B_cs + h_id * B_h + s * B_s)  # scalar at k_j
        acc += C_val * B_val
        s += 1
    G_ptr_elt = G_ptr + b_id * G_bs + i_id * G_cs + j_id * G_cs + h_id * G_h
    tl.store(G_ptr_elt, acc)

@triton.jit
def contract_M_hidden_kernel(
    L_ptr, G_ptr, hidden_ptr, out_ptr,
    L_bs, L_hs, L_cs, L_j, L_k,
    G_bs, G_cs, G_j, G_h, G_s,
    H_bs, H_cs, H_j, H_h, H_d,
    Out_bs, Out_cs, Out_k, Out_h, Out_d,
    B_size, C_size,
    b_id: tl.constexpr,
    h_id: tl.constexpr,  # head index
    i_id: tl.constexpr,  # chunk i
    k_id: tl.constexpr,  # chunk k
    CHUNK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    """
    Compute out[b, i, k, h, d] = sum_{j=0..CHUNK_SIZE-1} G[b, i, k, j, h] * L[b, h, i, k, j] * hidden[b, i, j, h, d].
    Grid: (B, NUM_HEADS, num_chunks, CHUNK_SIZE, HEAD_DIM) -> each program computes one element for fixed (b, h, i, k, d).
    """
    d = 0  # Triton expects program_id over dims; we handle d via grid's last dim.
    while d < HEAD_DIM:
        acc = tl.zeros((), dtype=tl.float32)
        j = 0
        while j < CHUNK_SIZE:
            G_val = tl.load(G_ptr + b_id * G_bs + i_id * G_cs + k_id * G_cs + j * G_j + h_id * G_h)
            L_val = tl.load(L_ptr + b_id * L_bs + h_id * L_hs + i_id * L_cs + j * L_j + k_id * L_k)
            H_val = tl.load(hidden_ptr + b_id * H_bs + i_id * H_cs + j * H_j + h_id * H_h + d * H_d)
            acc += G_val * L_val * H_val
            j += 1
        out_ptr_elt = out_ptr + b_id * Out_bs + i_id * Out_cs + k_id * Out_k + h_id * Out_h + d * Out_d
        tl.store(out_ptr_elt, acc)
        d += 1

class ModelNew(torch.nn.Module):
    def forward(self, hidden_states, A_cumsum, B, C):
        """
        Compute Y_diag using Triton kernels:
        Inputs:
          hidden_states: [B, num_chunks, CHUNK_SIZE, NUM_HEADS, head_dim]
          A_cumsum: [B, NUM_HEADS, num_chunks, CHUNK_SIZE]
          B: [B, num_chunks, CHUNK_SIZE, N_GROUPS, state_size]
          C: [B, num_chunks, CHUNK_SIZE, N_GROUPS, state_size]
        Output:
          Y_diag: [B, num_chunks, CHUNK_SIZE, NUM_HEADS, head_dim] in bfloat16
        """
        assert hidden_states.is_cuda and A_cumsum.is_cuda and B.is_cuda and C.is_cuda, "Triton kernels require CUDA tensors."
        hidden = hidden_states.contiguous()
        A = A_cumsum.contiguous()
        B = B.contiguous()
        C = C.contiguous()

        B_size = B.size()  # [B, num_chunks, CHUNK_SIZE, NUM_HEADS, state_size]
        C_size = C.size()
        H_size = hidden.size()  # [B, num_chunks, CHUNK_SIZE, NUM_HEADS, head_dim]

        # Strides
        A_bs, A_hs, A_cs, A_j = A.stride()  # (B, NUM_HEADS, num_chunks, CHUNK_SIZE)
        H_bs, H_cs, H_j, H_hs, H_d = hidden.stride()  # (B, num_chunks, CHUNK_SIZE, NUM_HEADS, head_dim)
        B_bs, B_cs, B_j, B_hs, B_s = B.stride()
        C_bs, C_cs, C_j, C_hs, C_s = C.stride()

        # Allocate L (float32)
        L = torch.empty((B_size[0], NUM_HEADS, B_size[2], CHUNK_SIZE, CHUNK_SIZE), dtype=torch.float32, device=hidden.device)
        L_bs, L_hs, L_cs, L_j, L_k = L.stride()

        # Launch build_L_kernel: grid over (B, NUM_HEADS, num_chunks, 1, CHUNK_SIZE)
        grid_L = (B_size[0], NUM_HEADS, B_size[2], 1, CHUNK_SIZE)
        build_L_kernel[grid_L](
            A, L,
            A_bs, A_hs, A_cs, A_j,
            L_bs, L_hs, L_cs, L_j, L_k,
            B_size[0], B_size[2],
            b_id=0, h_id=0, i_id=0  # indices are provided; Triton will use them per program
        )

        # Allocate G (float32): [B, num_chunks, CHUNK_SIZE, CHUNK_SIZE, NUM_HEADS]
        G = torch.empty((B_size[0], B_size[2], CHUNK_SIZE, CHUNK_SIZE, NUM_HEADS), dtype=torch.float32, device=hidden.device)
        G_bs, G_cs, G_j, G_h, G_s = G.stride()

        # Launch compute_G_kernel: grid over (B, NUM_HEADS, num_chunks, num_chunks)
        grid_G = (B_size[0], NUM_HEADS, B_size[2], B_size[2])
        compute_G_kernel[grid_G](
            B, C, G,
            B_bs, B_cs, B_j, B_hs, B_s,
            C_bs, C_cs, C_j, C_hs, C_s,
            G_bs, G_cs, G_j, G_h, G_s,
            B_size[0], B_size[2],
            b_id=0, h_id=0, i_id=0, j_id=0, state_size=B_s
        )

        # Output tensor: [B, num_chunks, CHUNK_SIZE, NUM_HEADS, head_dim], bfloat16
        out = torch.empty((B_size[0], B_size[2], CHUNK_SIZE, NUM_HEADS, H_d), dtype=torch.bfloat16, device=hidden.device)
        out_bs, out_cs, out_k, out_h, out_d = out.stride()

        # Launch contract_M_hidden_kernel: grid over (B, NUM_HEADS, num_chunks, CHUNK_SIZE, head_dim)
        grid_out = (B_size[0], NUM_HEADS, B_size[2], CHUNK_SIZE, H_d)
        contract_M_hidden_kernel[grid_out](
            L, G, hidden, out,
            L_bs, L_hs, L_cs, L_j, L_k,
            G_bs, G_cs, G_j, G_h, G_s,
            H_bs, H_cs, H_j, H_hs, H_d,
            out_bs, out_cs, out_k, out_h, out_d,
            B_size[0], B_size[2],
            b_id=0, h_id=0, i_id=0, k_id=0, CHUNK_SIZE=CHUNK_SIZE, HEAD_DIM=H_d
        )

        return out


def run(*args):
    return ModelNew()(*args)
