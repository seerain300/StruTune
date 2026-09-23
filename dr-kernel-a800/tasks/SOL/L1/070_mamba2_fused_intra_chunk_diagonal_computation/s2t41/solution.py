import torch
import triton
import triton.language as tl

# Constants used by Triton kernels (compile-time)
CHUNK_SIZE: tl.constexpr = 128
NUM_HEADS: tl.constexpr = 32
HEAD_DIM: tl.constexpr = 128
N_GROUPS: tl.constexpr = 8
REPEAT: tl.constexpr = NUM_HEADS // N_GROUPS  # 4
REPEAT_INV: tl.constexpr = N_GROUPS           # 8


@triton.jit
def build_L_kernel(
    A_ptr,            # *f32, [B, num_heads, num_chunks, chunk_size]
    L_ptr,            # *f32, [B, num_chunks, chunk_size, chunk_size, num_heads]
    A_stride_b, A_stride_h, A_stride_c, A_stride_k,   # int32
    L_stride_b, L_stride_c, L_stride_k, L_stride_j, L_stride_h  # int32
):
    # Grid: (b, c, h, j) where j is the target index along chunk
    b = tl.program_id(0)
    c = tl.program_id(1)
    j = tl.program_id(3)
    h = tl.program_id(2)

    # Vector of source indices (0..127)
    src = tl.arange(0, CHUNK_SIZE)
    mask_src = src < CHUNK_SIZE

    # Compute cumsum along k for A[b, h, c, :]
    cum = tl.zeros([1], dtype=tl.float32)
    for s in range(CHUNK_SIZE):
        # Only consider s < j for lower-triangular: i >= j means we include s when s < j
        i_ok = s < j
        A_off = b * A_stride_b + h * A_stride_h + c * A_stride_c + s * A_stride_k
        # If not ok, treat as zero
        A_val = tl.load(A_ptr + A_off, mask=i_ok, other=0.0)
        cum += A_val
        # Store L[b, c, s, j, h] = exp(cum) for s < j (lower triangle)
        if i_ok:
            L_off = b * L_stride_b + c * L_stride_c + s * L_stride_k + j * L_stride_j + h * L_stride_h
            tl.store(L_ptr + L_off, tl.exp(cum))

    # For s >= j, L values are implicitly 0 because we didn't store them above.


@triton.jit
def compute_G_kernel(
    B_ptr,            # *f32, [B, num_chunks, chunk_size, num_heads, state_size]
    C_ptr,            # *f32, [B, num_chunks, chunk_size, num_heads, state_size]
    G_ptr,            # *f32, [B, num_chunks, chunk_size, chunk_size, num_heads]
    B_stride_b, B_stride_c, B_stride_k, B_stride_h, B_stride_s,
    C_stride_b, C_stride_c, C_stride_k, C_stride_h, C_stride_s,
    G_stride_b, G_stride_c, G_stride_k, G_stride_j, G_stride_h,
    num_chunks, num_heads
):
    # Grid: (b, c, h, j)
    b = tl.program_id(0)
    c = tl.program_id(1)
    j = tl.program_id(3)  # destination position along chunk
    h = tl.program_id(2)

    # Compute G[i=j, j, h] = sum over state s of C[b, c, j, h, s] * B[b, c, j, h, s]
    G_val = tl.zeros([1], dtype=tl.float32)
    for s in range(HEAD_DIM):
        C_off = b * C_stride_b + c * C_stride_c + j * C_stride_k + h * C_stride_h + s * C_stride_s
        B_off = b * B_stride_b + c * B_stride_c + j * B_stride_k + h * B_stride_h + s * B_stride_s
        C_val = tl.load(C_ptr + C_off)
        B_val = tl.load(B_ptr + B_off)
        G_val += C_val * B_val

    # Store G[b, c, j, j, h]
    G_off = b * G_stride_b + c * G_stride_c + j * G_stride_k + j * G_stride_j + h * G_stride_h
    tl.store(G_ptr + G_off, G_val)


@triton.jit
def multiply_LG_kernel(
    L_ptr,            # *f32, [B, num_chunks, chunk_size, chunk_size, num_heads]
    G_ptr,            # *f32, [B, num_chunks, chunk_size, chunk_size, num_heads]
    M_ptr,            # *f32, [B, num_chunks, chunk_size, chunk_size, num_heads]
    L_stride_b, L_stride_c, L_stride_k, L_stride_j, L_stride_h,
    G_stride_b, G_stride_c, G_stride_k, G_stride_j, G_stride_h,
    M_stride_b, M_stride_c, M_stride_k, M_stride_j, M_stride_h,
    num_chunks
):
    # Grid over (b, c, h), then iterate over j,k inside kernel
    b = tl.program_id(0)
    c = tl.program_id(1)
    h = tl.program_id(2)

    for j in range(CHUNK_SIZE):
        for k in range(CHUNK_SIZE):
            L_off = b * L_stride_b + c * L_stride_c + k * L_stride_k + j * L_stride_j + h * L_stride_h
            G_off = b * G_stride_b + c * G_stride_c + j * G_stride_k + k * G_stride_j + h * G_stride_h
            M_off = b * M_stride_b + c * M_stride_c + k * M_stride_k + j * M_stride_j + h * M_stride_h
            L_val = tl.load(L_ptr + L_off)
            G_val = tl.load(G_ptr + G_off)
            M_val = L_val * G_val
            tl.store(M_ptr + M_off, M_val)


@triton.jit
def contract_to_Ydiag_kernel(
    M_ptr,            # *f32, [B, num_chunks, chunk_size, chunk_size, num_heads]
    hidden_ptr,       # *f32, [B, num_chunks, chunk_size, num_heads, head_dim]
    Y_ptr,            # *bf16, [B, num_chunks, chunk_size, num_heads, head_dim]
    M_stride_b, M_stride_c, M_stride_k, M_stride_j, M_stride_h,
    hidden_stride_b, hidden_stride_c, hidden_stride_k, hidden_stride_h, hidden_stride_d,
    Y_stride_b, Y_stride_c, Y_stride_k, Y_stride_h, Y_stride_d,
    num_chunks
):
    # Grid: (b, c, k, h, d)
    b = tl.program_id(0)
    c = tl.program_id(1)
    k = tl.program_id(2)
    h = tl.program_id(3)
    d = tl.program_id(4)

    acc = tl.zeros([1], dtype=tl.float32)
    for j in range(CHUNK_SIZE):
        M_off = b * M_stride_b + c * M_stride_c + k * M_stride_k + j * M_stride_j + h * M_stride_h
        M_val = tl.load(M_ptr + M_off)
        hidden_off = b * hidden_stride_b + c * hidden_stride_c + k * hidden_stride_k + h * hidden_stride_h + d * hidden_stride_d
        hidden_val = tl.load(hidden_ptr + hidden_off)
        acc += M_val * hidden_val

    Y_off = b * Y_stride_b + c * Y_stride_c + k * Y_stride_k + h * Y_stride_h + d * Y_stride_d
    tl.store(Y_ptr + Y_off, acc.to(tl.bfloat16))


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                A_cumsum: torch.Tensor,
                B: torch.Tensor,
                C: torch.Tensor) -> torch.Tensor:
        # Shapes:
        # hidden_states: [B, num_chunks, chunk_size, num_heads, head_dim] = [B, NC, 128, 32, 128]
        # A_cumsum: [B, num_heads, num_chunks, chunk_size] = [B, 32, NC, 128]
        # B: [B, num_chunks, chunk_size, n_groups, state_size] = [B, NC, 128, 8, 128]
        # C: same as B

        # Expand B and C across heads: from n_groups to num_heads
        B_expanded = B.repeat_interleave(REPEAT, dim=3).contiguous()
        C_expanded = C.repeat_interleave(REPEAT, dim=3).contiguous()

        B_size = hidden_states.shape[0]
        num_chunks = hidden_states.shape[1]
        chunk_size = hidden_states.shape[2]
        num_heads = hidden_states.shape[3]
        head_dim = hidden_states.shape[4]

        # Allocate outputs
        L = torch.empty(
            (B_size, num_chunks, chunk_size, chunk_size, num_heads),
            device=hidden_states.device,
            dtype=torch.float32
        )
        G = torch.empty(
            (B_size, num_chunks, chunk_size, chunk_size, num_heads),
            device=hidden_states.device,
            dtype=torch.float32
        )
        M = torch.empty_like(G)
        Y = torch.empty(
            (B_size, num_chunks, chunk_size, num_heads, head_dim),
            device=hidden_states.device,
            dtype=torch.bfloat16
        )

        # Launch build_L_kernel: grid over (b, c, h, j)
        grid_L = (B_size, num_chunks, num_heads, chunk_size)
        build_L_kernel[grid_L](
            A_cumsum, L,
            A_cumsum.stride(0), A_cumsum.stride(1), A_cumsum.stride(2), A_cumsum.stride(3),
            L.stride(0), L.stride(1), L.stride(2), L.stride(3), L.stride(4)
        )

        # Launch compute_G_kernel: grid over (b, c, h, j)
        grid_G = (B_size, num_chunks, num_heads, chunk_size)
        compute_G_kernel[grid_G](
            B_expanded, C_expanded, G,
            B_expanded.stride(0), B_expanded.stride(1), B_expanded.stride(2), B_expanded.stride(3), B_expanded.stride(4),
            C_expanded.stride(0), C_expanded.stride(1), C_expanded.stride(2), C_expanded.stride(3), C_expanded.stride(4),
            G.stride(0), G.stride(1), G.stride(2), G.stride(3), G.stride(4),
            num_chunks, num_heads
        )

        # Launch multiply_LG_kernel: grid over (b, c, h)
        grid_M = (B_size, num_chunks, num_heads)
        multiply_LG_kernel[grid_M](
            L, G, M,
            L.stride(0), L.stride(1), L.stride(2), L.stride(3), L.stride(4),
            G.stride(0), G.stride(1), G.stride(2), G.stride(3), G.stride(4),
            M.stride(0), M.stride(1), M.stride(2), M.stride(3), M.stride(4),
            num_chunks
        )

        # Launch contract_to_Ydiag_kernel: grid over (b, c, k, h, d)
        grid_Y = (B_size, num_chunks, chunk_size, num_heads, head_dim)
        contract_to_Ydiag_kernel[grid_Y](
            M, hidden_states, Y,
            M.stride(0), M.stride(1), M.stride(2), M.stride(3), M.stride(4),
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2), hidden_states.stride(3), hidden_states.stride(4),
            Y.stride(0), Y.stride(1), Y.stride(2), Y.stride(3), Y.stride(4),
            num_chunks
        )

        return Y


def run(*args):
    return ModelNew()(*args)
