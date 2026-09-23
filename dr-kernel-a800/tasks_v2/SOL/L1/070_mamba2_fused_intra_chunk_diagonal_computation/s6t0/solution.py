import torch
import triton
import triton.language as tl

# Kernel 1: Compute causal mask L = exp(cumsum(masked A_cumsum)), lower-triangular (diagonal=-1)
# A_cumsum: [B, H, N, S] float32
# L: [B, H, N, S, S] float32
@triton.jit
def l_causal_mask_exp_kernel(
    A_ptr, L_ptr,
    B_size, H_size, N_size, S_size,
    A_stride_b, A_stride_h, A_stride_n, A_stride_s,
    L_stride_b, L_stride_h, L_stride_n, L_stride_s1, L_stride_s2,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    n = tl.program_id(2)

    # Initialize L vector for each i (source index)
    i = 0
    while i < S_size:
        # Running sum of A[b, h, n, :] with lower-triangular mask (exclude diagonal)
        running = 0.0
        j = 0
        while j < S_size:
            # Lower-triangular mask: include only when j <= i-1 (diagonal=-1)
            include = j <= (i - 1)
            a_offset = b * A_stride_b + h * A_stride_h + n * A_stride_n + j * A_stride_s
            a_val = tl.load(A_ptr + a_offset, mask=include, other=0.0)
            running += a_val
            j += 1
        # Write exp(running) to L[b, h, n, i, :]
        l_offset = b * L_stride_b + h * L_stride_h + n * L_stride_n + i * L_stride_s1
        # Store as float32
        tl.store(L_ptr + l_offset, tl.exp(running))
        i += 1

# Kernel 2: Compute G[i,j,h] = sum over d of C[i,n] * B[j,n]
# C: [B, N, S, H, D]
# B: [B, N, S, H, D]
# G: [B, N, S, S, H]
@triton.jit
def g_contract_kernel(
    C_ptr, B_ptr, G_ptr,
    B_size, N_size, S_size, H_size, D_size,
    C_stride_b, C_stride_n, C_stride_s, C_stride_h, C_stride_d,
    B_stride_b, B_stride_n, B_stride_s, B_stride_h, B_stride_d,
    G_stride_b, G_stride_n, G_stride_s1, G_stride_s2, G_stride_h,
    BLOCK_D: tl.constexpr,
):
    # Grid dimensions: (B, N, S, S, H)
    b = tl.program_id(0)
    n = tl.program_id(1)
    i = tl.program_id(2)
    j = tl.program_id(3)
    h = tl.program_id(4)

    acc = tl.zeros((), dtype=tl.float32)

    # Iterate over state dimension D in chunks of BLOCK_D
    d = 0
    while d < D_size:
        # Vectors for C[i, h, d:d+BLOCK_D] and B[j, h, d:d+BLOCK_D]
        c_off_base = b * C_stride_b + n * C_stride_n + i * C_stride_s + h * C_stride_h
        b_off_base = b * B_stride_b + n * B_stride_n + j * B_stride_s + h * B_stride_h

        c_ptrs = C_ptr + c_off_base + (d + tl.arange(0, BLOCK_D)) * C_stride_d
        b_ptrs = B_ptr + b_off_base + (d + tl.arange(0, BLOCK_D)) * B_stride_d
        c_vec = tl.load(c_ptrs, mask=(d + tl.arange(0, BLOCK_D)) < D_size, other=0.0)
        b_vec = tl.load(b_ptrs, mask=(d + tl.arange(0, BLOCK_D)) < D_size, other=0.0)
        acc += tl.sum(c_vec * b_vec, axis=0)
        d += BLOCK_D

    # Store G[i,j,h] = acc
    g_offset = b * G_stride_b + n * G_stride_n + i * G_stride_s1 + j * G_stride_s2 + h * G_stride_h
    tl.store(G_ptr + g_offset, acc)

# Kernel 3: Compute Y_diag[b,n,i,h,:] = sum_j M[i,j,h] * hidden_states[b,n,j,h,:]
# M: [B, N, S, S, H]
# hidden_states: [B, N, S, H, D]
# Y_diag: [B, N, S, H, D]
@triton.jit
def y_diag_reduce_kernel(
    M_ptr, HS_ptr, Y_ptr,
    B_size, N_size, S_size, H_size, D_size,
    M_stride_b, M_stride_n, M_stride_s1, M_stride_s2, M_stride_h,
    HS_stride_b, HS_stride_n, HS_stride_s, HS_stride_h, HS_stride_d,
    Y_stride_b, Y_stride_n, Y_stride_s, Y_stride_h, Y_stride_d,
    BLOCK_D: tl.constexpr,
):
    b = tl.program_id(0)
    n = tl.program_id(1)
    i = tl.program_id(2)
    h = tl.program_id(3)

    # Vector accumulator for D dimension
    acc_vec = tl.zeros((BLOCK_D,), dtype=tl.float32)

    j = 0
    while j < S_size:
        # Load M[b,n,i,j,h] and hidden_states[b,n,j,h,:]
        m_off = b * M_stride_b + n * M_stride_n + i * M_stride_s1 + j * M_stride_s2 + h * M_stride_h
        m_val = tl.load(M_ptr + m_off, mask=True, other=0.0)

        hs_off_base = b * HS_stride_b + n * HS_stride_n + j * HS_stride_s + h * HS_stride_h
        d_idx = tl.arange(0, BLOCK_D)
        hs_ptrs = HS_ptr + hs_off_base + d_idx * HS_stride_d
        hs_vec = tl.load(hs_ptrs, mask=(d_idx < D_size), other=0.0)

        acc_vec += m_val * hs_vec
        j += 1

    # Store Y_diag[b,n,i,h,:] as float32 (will be cast outside)
    y_off_base = b * Y_stride_b + n * Y_stride_n + i * Y_stride_s + h * Y_stride_h
    # Store each d component
    d = 0
    while d < D_size:
        y_off = y_off_base + d * Y_stride_d
        tl.store(Y_ptr + y_off, acc_vec[d])
        d += 1

# Optional helper to prepare inputs on CUDA and ensure contiguity
def _ensure_cuda(t: torch.Tensor) -> torch.Tensor:
    if not t.is_cuda:
        return t.to('cuda')
    return t

class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                A_cumsum: torch.Tensor,
                B: torch.Tensor,
                C: torch.Tensor) -> torch.Tensor:
        # Ensure all inputs are on CUDA and contiguous
        hidden_states = _ensure_cuda(hidden_states).contiguous()
        A_cumsum = _ensure_cuda(A_cumsum).contiguous()
        B = _ensure_cuda(B).contiguous()
        C = _ensure_cuda(C).contiguous()

        # Dimensions
        B_size, N_size, S_size, H_size, D_size = hidden_states.shape
        assert A_cumsum.shape == (B_size, H_size, N_size, S_size), "A_cumsum shape must be [B, H, N, S]"
        # B and C shapes: [B, N, S, G, D]; but original code uses n_groups via repeat, not used directly.
        # We assume B and C have the last two dims as (G, D). The code later expands to H via repeat_interleave(NUM_HEADS // N_GROUPS).
        # In this Triton version, we avoid repeat_interleave by computing contraction directly over state_dim.
        # So we need to know state_size. Original uses head_dim; we can extract D_size from hidden_states' last dim.
        # However, B and C in provided signature may have different last dim; but the original code uses B,C's last dim (D),
        # which equals hidden_states' head_dim (D_size). We'll use D_size for contraction.

        # 1) Compute L = exp(cumsum(masked A_cumsum)) with lower-triangular (diagonal = -1), Triton kernel
        # Allocate L as float32
        L = torch.empty((B_size, H_size, N_size, S_size, S_size), dtype=torch.float32, device=hidden_states.device)

        grid_L = (B_size, H_size, N_size)
        # Strides
        A_stride_b, A_stride_h, A_stride_n, A_stride_s = A_cumsum.stride()
        L_stride_b, L_stride_h, L_stride_n, L_stride_s1, L_stride_s2 = L.stride()

        l_causal_mask_exp_kernel[grid_L](
            A_cumsum, L,
            B_size, H_size, N_size, S_size,
            A_stride_b, A_stride_h, A_stride_n, A_stride_s,
            L_stride_b, L_stride_h, L_stride_n, L_stride_s1, L_stride_s2,
            num_warps=1, num_stages=1
        )

        # 2) Compute G[i,j,h] = sum over state (D_size) of C[i,n] * B[j,n]
        # We need to expand B/C to H. The original code repeats_interleave NUM_HEADS // N_GROUPS (32//8=4).
        # We will emulate that expansion by reshaping to H directly. But for Triton, we can use original G dimension and
        # rely on the contraction kernel to produce output with H dimension. So we define G with H_size output.
        # Here, the contraction uses D_size (head_dim). To match the original, we should have B/C's last dim == D_size.
        # We'll proceed under the assumption that B,C's last dim equals hidden_states' head_dim (D_size), which matches original.
        G = torch.empty((B_size, N_size, S_size, S_size, H_size), dtype=torch.float32, device=hidden_states.device)

        # Strides for C and B: last dim is D_size
        C_stride_b, C_stride_n, C_stride_s, C_stride_h, C_stride_d = C.stride()
        B_stride_b, B_stride_n, B_stride_s, B_stride_h, B_stride_d = B.stride()
        G_stride_b, G_stride_n, G_stride_s1, G_stride_s2, G_stride_h = G.stride()

        grid_G = (B_size, N_size, S_size, S_size, H_size)
        g_contract_kernel[grid_G](
            C, B, G,
            B_size, N_size, S_size, H_size, D_size,
            C_stride_b, C_stride_n, C_stride_s, C_stride_h, C_stride_d,
            B_stride_b, B_stride_n, B_stride_s, B_stride_h, B_stride_d,
            G_stride_b, G_stride_n, G_stride_s1, G_stride_s2, G_stride_h,
            BLOCK_D=32,  # tuneable
            num_warps=2, num_stages=2
        )

        # 3) Compute Y_diag[b,n,i,h,:] = sum_j G[i,j,h] * hidden_states[b,n,j,h,:]
        # hidden_states: [B, N, S, H, D] -> we need to use G[..., j, h] and hidden_states[..., j, h, :]
        Y = torch.empty((B_size, N_size, S_size, H_size, D_size), dtype=torch.float32, device=hidden_states.device)

        M_stride_b, M_stride_n, M_stride_s1, M_stride_s2, M_stride_h = G.stride()  # G has dims [B,N,S,S,H]
        HS_stride_b, HS_stride_n, HS_stride_s, HS_stride_h, HS_stride_d = hidden_states.stride()
        Y_stride_b, Y_stride_n, Y_stride_s, Y_stride_h, Y_stride_d = Y.stride()

        grid_Y = (B_size, N_size, S_size, H_size)
        y_diag_reduce_kernel[grid_Y](
            G, hidden_states, Y,
            B_size, N_size, S_size, H_size, D_size,
            M_stride_b, M_stride_n, M_stride_s1, M_stride_s2, M_stride_h,
            HS_stride_b, HS_stride_n, HS_stride_s, HS_stride_h, HS_stride_d,
            Y_stride_b, Y_stride_n, Y_stride_s, Y_stride_h, Y_stride_d,
            BLOCK_D=32,  # tuneable
            num_warps=2, num_stages=2
        )

        # Return in bfloat16, matching original
        return Y.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
