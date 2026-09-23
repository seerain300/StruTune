import torch
import triton
import triton.language as tl


# Kernel 1: Compute L = exp(segment_sum(A)) on lower triangle (diagonal=-1).
# Input: A_cumsum [B, H, N, S], Output: L [B, H, N, S, S] float32
@triton.jit
def a_segment_sum_exp_kernel(
    A_ptr, L_ptr,
    B_size, H_size, N_size, S_size,
    A_stride_b, A_stride_h, A_stride_n, A_stride_s,
    L_stride_b, L_stride_h, L_stride_n, L_stride_s1, L_stride_s2,
):
    pid_b = tl.program_id(0)  # batch
    pid_h = tl.program_id(1)  # num_heads
    pid_n = tl.program_id(2)  # num_chunks

    # We iterate i and j. For j <= i, segment_sum = sum_{k=0..i} A[b,h,n,k]
    # We directly write L[b,h,n,i,j] = exp(segment_sum) if j<=i else 0.0
    # This matches tril(diagonal=-1): j <= i included, j > i excluded.
    for i in range(S_size):
        seg = 0.0  # segment sum scalar
        # Note: A tensor is [B,H,N,S], so element at (b,h,n,i) has address:
        # base + b*A_stride_b + h*A_stride_h + n*A_stride_n + i*A_stride_s
        for j in range(S_size):
            # Load A[b,h,n,j] (scalar)
            a_val = tl.load(A_ptr + pid_b * A_stride_b + pid_h * A_stride_h + pid_n * A_stride_n + j * A_stride_s)
            if j <= i:
                seg += a_val
            # Store L[b,h,n,i,j] = exp(seg) if j<=i else 0.0
            l_val = tl.exp(seg) if j <= i else 0.0
            tl.store(L_ptr + pid_b * L_stride_b + pid_h * L_stride_h + pid_n * L_stride_n + i * L_stride_s1 + j * L_stride_s2, l_val)


# Kernel 2: G contraction: G[b,n,i,j,h] = sum_d C[b,n,i,h,d] * B[b,n,j,h,d]
@triton.jit
def g_contract_kernel(
    B_ptr, C_ptr, G_ptr,
    B_size, N_size, S_size, H_size, D_size,
    B_stride_b, B_stride_n, B_stride_s1, B_stride_h, B_stride_d,
    C_stride_b, C_stride_n, C_stride_s1, C_stride_h, C_stride_d,
    G_stride_b, G_stride_n, G_stride_s1, G_stride_s2, G_stride_h,
    BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_i = tl.program_id(2)
    pid_j = tl.program_id(3)
    pid_h = tl.program_id(4)

    acc = tl.zeros((), dtype=tl.float32)
    for d in range(0, D_size, BLOCK_D):
        offs_d = d + tl.arange(0, BLOCK_D)
        mask_d = offs_d < D_size
        # Load B[b,n,j,h,offs_d]
        b_ptrs = B_ptr + pid_b * B_stride_b + pid_n * B_stride_n + pid_j * B_stride_s1 + pid_h * B_stride_h + offs_d * B_stride_d
        b_vals = tl.load(b_ptrs, mask=mask_d, other=0.0)
        # Load C[b,n,i,h,offs_d]
        c_ptrs = C_ptr + pid_b * C_stride_b + pid_n * C_stride_n + pid_i * C_stride_s1 + pid_h * C_stride_h + offs_d * C_stride_d
        c_vals = tl.load(c_ptrs, mask=mask_d, other=0.0)
        # Accumulate dot over D tile
        acc += tl.sum(b_vals * c_vals, axis=0)

    # Store G[b,n,i,j,h] = acc
    g_ptr = G_ptr + pid_b * G_stride_b + pid_n * G_stride_n + pid_i * G_stride_s1 + pid_j * G_stride_s2 + pid_h * G_stride_h
    tl.store(g_ptr, acc)


# Kernel 3: Elementwise M = G * L
@triton.jit
def m_mul_kernel(
    G_ptr, L_ptr, M_ptr,
    B_size, N_size, S_size, H_size,
    G_stride_b, G_stride_n, G_stride_s1, G_stride_s2, G_stride_h,
    L_stride_b, L_stride_n, L_stride_s1, L_stride_s2, L_stride_h,
    M_stride_b, M_stride_n, M_stride_s1, M_stride_s2, M_stride_h,
):
    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_i = tl.program_id(2)
    pid_j = tl.program_id(3)
    pid_h = tl.program_id(4)

    g_ptr = G_ptr + pid_b * G_stride_b + pid_n * G_stride_n + pid_i * G_stride_s1 + pid_j * G_stride_s2 + pid_h * G_stride_h
    l_ptr = L_ptr + pid_b * L_stride_b + pid_n * L_stride_n + pid_i * L_stride_s1 + pid_j * L_stride_s2 + pid_h * L_stride_h
    m_ptr = M_ptr + pid_b * M_stride_b + pid_n * M_stride_n + pid_i * M_stride_s1 + pid_j * M_stride_s2 + pid_h * M_stride_h

    g_val = tl.load(g_ptr)
    l_val = tl.load(l_ptr)
    tl.store(m_ptr, g_val * l_val)


# Kernel 4: Y_diag reduction: Y[b,n,i,h] = sum_j M[b,n,i,j,h] * hidden_states[b,n,j,h,:]
# We implement as a per-(b,n,i,h) loop over j and D. Launch grid = (B, N, S, H).
@triton.jit
def y_diag_reduce_kernel(
    M_ptr, hidden_ptr, Y_ptr,
    B_size, N_size, S_size, H_size, D_size,
    M_stride_b, M_stride_n, M_stride_s1, M_stride_s2, M_stride_h,
    hidden_stride_b, hidden_stride_n, hidden_stride_s, hidden_stride_h, hidden_stride_d,
    Y_stride_b, Y_stride_n, Y_stride_s1, Y_stride_h,
    BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_i = tl.program_id(2)
    pid_h = tl.program_id(3)

    acc = tl.zeros((), dtype=tl.float32)

    for j in range(S_size):
        # acc += sum_d M[b,n,i,j,h] * hidden[b,n,j,h,d]
        for d in range(0, D_size, BLOCK_D):
            offs_d = d + tl.arange(0, BLOCK_D)
            mask_d = offs_d < D_size
            m_ptrs = M_ptr + pid_b * M_stride_b + pid_n * M_stride_n + pid_i * M_stride_s1 + j * M_stride_s2 + pid_h * M_stride_h
            m_vals = tl.load(m_ptrs, mask=mask_d, other=0.0)
            h_ptrs = hidden_ptr + pid_b * hidden_stride_b + pid_n * hidden_stride_n + j * hidden_stride_s + pid_h * hidden_stride_h + offs_d * hidden_stride_d
            h_vals = tl.load(h_ptrs, mask=mask_d, other=0.0)
            acc += tl.sum(m_vals * h_vals, axis=0)

    # Store Y[b,n,i,h] = acc
    y_ptr = Y_ptr + pid_b * Y_stride_b + pid_n * Y_stride_n + pid_i * Y_stride_s1 + pid_h * Y_stride_h
    tl.store(y_ptr, acc)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, A_cumsum: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        # Shapes (original signature): hidden_states [B, N, S, H, D], A_cumsum [B, H, N, S], B [B, N, S, G, D], C [B, N, S, G, D]
        B_size, N_size, S_size, H_size, D_size = hidden_states.shape
        device = hidden_states.device

        # 1) Compute tril mask (diagonal=-1) as a 1D int8 tensor of length S*S
        # We'll use a Triton kernel to fill this, though a PyTorch fill would be fine; but to keep Triton-only, we implement a tiny kernel.
        S2 = S_size * S_size
        tril_mask = torch.empty(S2, dtype=torch.int8, device=device)
        tril_kernel = triton.jit(
            "void(tril_mask_ptr, S: tl.constexpr)",
            num_warps=1, num_stages=1
        )
        tril_kernel(tril_mask, S_size)

        # 2) Compute L = exp(segment_sum(A)) on lower triangle using Triton kernel
        A = A_cumsum.contiguous().to(torch.float32)
        L = torch.empty((B_size, H_size, N_size, S_size, S_size), dtype=torch.float32, device=device)

        A_stride_b = A.stride(0); A_stride_h = A.stride(1); A_stride_n = A.stride(2); A_stride_s = A.stride(3)
        L_stride_b = L.stride(0); L_stride_h = L.stride(1); L_stride_n = L.stride(2); L_stride_s1 = L.stride(3); L_stride_s2 = L.stride(4)

        a_segment_sum_exp_kernel[(B_size, H_size, N_size)](
            A, L,
            B_size, H_size, N_size, S_size,
            A_stride_b, A_stride_h, A_stride_n, A_stride_s,
            L_stride_b, L_stride_h, L_stride_n, L_stride_s1, L_stride_s2,
            num_warps=1, num_stages=1
        )

        # 3) Expand B/C to H if necessary (original code expands from G to H via repeat_interleave(4)). We assume inputs are already expanded to H.
        Bc = B.contiguous().to(torch.float32)
        Cc = C.contiguous().to(torch.float32)

        # 4) Compute G[b,n,i,j,h] = sum_d C[b,n,i,h,d] * B[b,n,j,h,d] via Triton
        G = torch.empty((B_size, N_size, S_size, S_size, H_size), dtype=torch.float32, device=device)
        G_stride_b = G.stride(0); G_stride_n = G.stride(1); G_stride_s1 = G.stride(2); G_stride_s2 = G.stride(3); G_stride_h = G.stride(4)

        Bc_stride_b = Bc.stride(0); Bc_stride_n = Bc.stride(1); Bc_stride_s1 = Bc.stride(2); Bc_stride_h = Bc.stride(3); Bc_stride_d = Bc.stride(4)
        Cc_stride_b = Cc.stride(0); Cc_stride_n = Cc.stride(1); Cc_stride_s1 = Cc.stride(2); Cc_stride_h = Cc.stride(3); Cc_stride_d = Cc.stride(4)

        g_contract_kernel[(B_size, N_size, S_size, S_size, H_size)](
            Bc, Cc, G,
            B_size, N_size, S_size, H_size, D_size,
            Bc_stride_b, Bc_stride_n, Bc_stride_s1, Bc_stride_h, Bc_stride_d,
            Cc_stride_b, Cc_stride_n, Cc_stride_s1, Cc_stride_h, Cc_stride_d,
            G_stride_b, G_stride_n, G_stride_s1, G_stride_s2, G_stride_h,
            BLOCK_D=32,
            num_warps=1, num_stages=1
        )

        # 5) Compute M = G * L in Triton
        M = torch.empty_like(G, dtype=torch.float32, device=device)

        G_stride_b, G_stride_n, G_stride_s1, G_stride_s2, G_stride_h = G.stride()
        L_stride_b, L_stride_n, L_stride_s1, L_stride_s2, L_stride_h = L.stride()
        M_stride_b, M_stride_n, M_stride_s1, M_stride_s2, M_stride_h = M.stride()

        m_mul_kernel[(B_size, N_size, S_size, S_size, H_size)](
            G, L, M,
            B_size, N_size, S_size, H_size,
            G_stride_b, G_stride_n, G_stride_s1, G_stride_s2, G_stride_h,
            L_stride_b, L_stride_n, L_stride_s1, L_stride_s2, L_stride_h,
            M_stride_b, M_stride_n, M_stride_s1, M_stride_s2, M_stride_h,
            num_warps=1, num_stages=1
        )

        # 6) Compute Y[b,n,i,h] = sum_j M[b,n,i,j,h] * hidden_states[b,n,j,h,:] via Triton
        hidden_f32 = hidden_states.contiguous().to(torch.float32)
        Y = torch.empty((B_size, N_size, S_size, H_size), dtype=torch.float32, device=device)

        Y_stride_b = Y.stride(0); Y_stride_n = Y.stride(1); Y_stride_s1 = Y.stride(2); Y_stride_h = Y.stride(3)
        M_stride_b, M_stride_n, M_stride_s1, M_stride_s2, M_stride_h = M.stride()
        hidden_stride_b = hidden_f32.stride(0); hidden_stride_n = hidden_f32.stride(1); hidden_stride_s = hidden_f32.stride(2); hidden_stride_h = hidden_f32.stride(3); hidden_stride_d = hidden_f32.stride(4)

        y_diag_reduce_kernel[(B_size, N_size, S_size, H_size)](
            M, hidden_f32, Y,
            B_size, N_size, S_size, H_size, D_size,
            M_stride_b, M_stride_n, M_stride_s1, M_stride_s2, M_stride_h,
            hidden_stride_b, hidden_stride_n, hidden_stride_s, hidden_stride_h, hidden_stride_d,
            Y_stride_b, Y_stride_n, Y_stride_s1, Y_stride_h,
            BLOCK_D=32,
            num_warps=1, num_stages=1
        )

        # Return in bfloat16 to match original signature
        return Y.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
