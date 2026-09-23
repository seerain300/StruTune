import torch
import triton
import triton.language as tl

# Constants used in the original model (compile-time for kernels)
CHUNK_SIZE = 128
NUM_HEADS = 32
N_GROUPS = 8

@triton.jit
def build_L_kernel(
    A_ptr, L_ptr,
    A_bs, A_hs, A_cs, A_cd,   # A strides: (B, NUM_HEADS, num_chunks, CHUNK_SIZE)
    L_bs, L_hs, L_cs, L_cd, L_w,  # L strides: (B, NUM_HEADS, num_chunks, CHUNK_SIZE, CHUNK_SIZE)
    b_id: tl.constexpr,
    h_id: tl.constexpr,
    i_id: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
):
    """
    Compute L[b, h, i, k, j] = exp(cumsum(A[b, h, i, 0:j+1])) for i >= j, else 0.
    Grid: (B, NUM_HEADS, num_chunks, 1, CHUNK_SIZE) -> each program fixes (b,h,i) and writes L for j in [0..CHUNK_SIZE-1].
    """
    j_vec = tl.arange(0, CHUNK_SIZE)
    # inclusive cumsum along j; keep as vector
    prefix = tl.zeros((CHUNK_SIZE,), dtype=tl.float32)

    # For each j, update prefix and store L[i, k, j]
    for j in range(CHUNK_SIZE):
        # Load A[b, h, i, j] as a scalar
        A_val = tl.load(A_ptr + b_id * A_bs + h_id * A_hs + i_id * A_cs + j * A_cd)
        prefix += A_val

        # L[i, k, j] = exp(prefix) if i >= j, else 0
        valid = j <= i_id
        L_ptrs = L_ptr + b_id * L_bs + h_id * L_hs + i_id * L_cs + j * L_w + j_vec * L_cd
        L_vec = tl.where(valid, tl.exp(prefix), 0.0)  # broadcast scalar to vector
        tl.store(L_ptrs, L_vec)

@triton.jit
def compute_G_kernel(
    B_ptr, C_ptr, G_ptr,
    B_bs, B_cs, B_cd, B_w, B_sd,    # B strides: (B, num_chunks, CHUNK_SIZE, NUM_HEADS, state_size)
    C_bs, C_cs, C_cd, C_w, C_sd,    # C strides: (B, num_chunks, CHUNK_SIZE, NUM_HEADS, state_size)
    G_bs, G_cs, G_cd, G_w, G_hs,    # G strides: (B, num_chunks, CHUNK_SIZE, CHUNK_SIZE, NUM_HEADS)
    b_id: tl.constexpr,
    i_id: tl.constexpr,
    j_id: tl.constexpr,
    h_id: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
):
    """
    Compute G[b, i, j, head] = sum over state_dim (HEAD_DIM) of C[b, i, :, head, s] * B[b, j, :, head, s].
    Grid: (B, num_chunks, CHUNK_SIZE, NUM_HEADS) -> each program computes one G element for fixed (b,i,j,head).
    """
    acc = 0.0
    for s in range(HEAD_DIM):
        # Loop over k dimension (CHUNK_SIZE) and accumulate per (i,j,head)
        for k in range(CHUNK_SIZE):
            B_val = tl.load(B_ptr + b_id * B_bs + j_id * B_cs + k * B_cd + h_id * B_w + s * B_sd)
            C_val = tl.load(C_ptr + b_id * C_bs + i_id * C_cs + k * C_cd + h_id * C_w + s * C_sd)
            acc += C_val * B_val
    # Store G[b, i, j, head]
    G_ptr_offset = b_id * G_bs + i_id * G_cs + j_id * G_w + h_id * G_hs
    tl.store(G_ptr + G_ptr_offset, acc)

@triton.jit
def contract_M_hidden_kernel(
    M_ptr, hidden_ptr, Y_ptr,
    M_bs, M_cs, M_cd, M_w, M_hs,   # M strides: (B, num_chunks, CHUNK_SIZE, CHUNK_SIZE, NUM_HEADS)
    H_bs, H_cs, H_cd, H_w, H_hs, H_hd,  # hidden strides: (B, num_chunks, CHUNK_SIZE, NUM_HEADS, head_dim)
    Y_bs, Y_cs, Y_hd, Y_hs, Y_bdim,     # Y strides: (B, num_chunks, CHUNK_SIZE, NUM_HEADS, head_dim)
    b_id: tl.constexpr,
    i_id: tl.constexpr,
    k_id: tl.constexpr,
    h_id: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
):
    """
    Compute Y[b, i, k, h, :] = sum_j (M[b, i, k, j, h] * hidden[b, i, j, h, :]).
    Grid: (B, num_chunks, CHUNK_SIZE, NUM_HEADS) -> each program computes one head-vector across j for fixed (b,i,k,h).
    """
    j_vec = tl.arange(0, CHUNK_SIZE)
    acc_vec = tl.zeros((HEAD_DIM,), dtype=tl.float32)

    for j in range(CHUNK_SIZE):
        M_ptrs = M_ptr + b_id * M_bs + i_id * M_cs + k_id * M_cd + j * M_w + h_id * M_hs
        M_val = tl.load(M_ptrs)  # scalar
        H_ptrs = hidden_ptr + b_id * H_bs + i_id * H_cs + j * H_cd + h_id * H_hs + j_vec * H_hd
        H_vec = tl.load(H_ptrs)  # vector of length HEAD_DIM
        acc_vec += M_val * H_vec

    Y_ptrs = Y_ptr + b_id * Y_bs + i_id * Y_cs + k_id * Y_cd + h_id * Y_hs + j_vec * Y_hd
    tl.store(Y_ptrs, acc_vec)

class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, A_cumsum: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        """
        Compute Y_diag with Triton kernels.
        hidden_states: [B, num_chunks, CHUNK_SIZE, NUM_HEADS, head_dim] (float32/float16)
        A_cumsum: [B, NUM_HEADS, num_chunks, CHUNK_SIZE] (float32)
        B: [B, num_chunks, CHUNK_SIZE, N_GROUPS, state_size]
        C: same shape as B
        Output: [B, num_chunks, CHUNK_SIZE, NUM_HEADS, head_dim] (bfloat16, matching original)
        """
        # Ensure tensors are on CUDA; forward does not move tensors, only launches kernels
        B_size, num_chunks, CHUNK_SIZE, NUM_HEADS, head_dim = hidden_states.shape

        # Allocate L as float32 (causal mask from A_cumsum)
        L = torch.empty((B_size, NUM_HEADS, num_chunks, CHUNK_SIZE, CHUNK_SIZE),
                        device=hidden_states.device, dtype=torch.float32)

        # Launch build_L_kernel: grid over (B, NUM_HEADS, num_chunks, 1, CHUNK_SIZE)
        grid_L = (B_size, NUM_HEADS, num_chunks, 1, CHUNK_SIZE)
        build_L_kernel[grid_L](
            A_cumsum, L,
            A_cumsum.stride(0), A_cumsum.stride(1), A_cumsum.stride(2), A_cumsum.stride(3),
            L.stride(0), L.stride(1), L.stride(2), L.stride(3), L.stride(4),
            b_id=B_size, h_id=NUM_HEADS, i_id=num_chunks, CHUNK_SIZE=CHUNK_SIZE,
            num_warps=4, num_stages=2
        )

        # Allocate G as float32: [B, num_chunks, CHUNK_SIZE, CHUNK_SIZE, NUM_HEADS]
        G = torch.empty((B_size, num_chunks, CHUNK_SIZE, CHUNK_SIZE, NUM_HEADS),
                        device=hidden_states.device, dtype=torch.float32)

        # Launch compute_G_kernel: grid over (B, num_chunks, CHUNK_SIZE, NUM_HEADS)
        grid_G = (B_size, num_chunks, CHUNK_SIZE, NUM_HEADS)
        compute_G_kernel[grid_G](
            B, C, G,
            B.stride(0), B.stride(1), B.stride(2), B.stride(3), B.stride(4),
            C.stride(0), C.stride(1), C.stride(2), C.stride(3), C.stride(4),
            G.stride(0), G.stride(1), G.stride(2), G.stride(3), G.stride(4),
            b_id=B_size, i_id=num_chunks, j_id=CHUNK_SIZE, h_id=NUM_HEADS, HEAD_DIM=head_dim, CHUNK_SIZE=CHUNK_SIZE,
            num_warps=4, num_stages=2
        )

        # Compute M = G * L
        M = G * L  # elementwise multiply, stays in Triton? Torch op. To strictly keep Triton, implement mul in Triton.
        # Since Triton kernels must be used, define a simple elementwise multiply in Triton: M_ptr = torch.empty_like(G)
        # Then implement a Triton kernel that copies G to M, multiplying by L. To avoid an extra kernel, we can perform
        # M = G * L via torch (disallowed). Therefore, implement mul in Triton:
        M = torch.empty_like(G)
        # Triton multiply kernel (simple cast): elementwise multiply using torch is not allowed; we will


def run(*args):
    return ModelNew()(*args)
