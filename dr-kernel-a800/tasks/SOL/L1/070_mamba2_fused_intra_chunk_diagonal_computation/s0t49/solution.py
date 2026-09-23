import torch
import triton
import triton.language as tl

# Triton kernel: build L with causal mask and exp on cumsum
@triton.jit
def build_L_kernel(
    A_ptr,  # *float32, [B, H, C, S]
    L_ptr,  # *float32, [B, C, S, S, H]
    Bsz: tl.constexpr, Csz: tl.constexpr, Hsz: tl.constexpr, S: tl.constexpr,
    A_bs, A_hs, A_cs, A_s,        # strides for A
    L_bs, L_cs, L_i, L_j, L_h,    # strides for L
):
    b = tl.program_id(0)
    c = tl.program_id(1)
    h = tl.program_id(2)

    # Precompute base pointers
    A_base = A_ptr + b * A_bs + h * A_hs + c * A_cs
    L_base = L_ptr + b * L_bs + c * L_cs

    # Cumulative sum along j for each i
    for i in range(0, S):
        # Initialize cumsum for this i
        cumsum = tl.zeros((), dtype=tl.float32)
        # For causal lower-triangular: we need cumsum_j[j] = sum_{t=0..j} A[b,h,c,t]
        # We compute cumsum up to i (we only use when i >= j), but we precompute cumsum[j] for all j at the end of i-loop.
        # Instead, we recompute cumsum up to j for each j: not optimal but simple and correct.
        # Better approach: precompute cumsum_j for all j at the beginning using a second loop over j, but Triton allows nested loops over constants.
        # Here, we compute cumsum_j[j] by accumulating from t=0 to j:
        cumsum_j = [0.0 for _ in range(0, S)]  # not a tensor, but we emulate via assignment
        # This emulation is tricky in Triton; to be safe, we implement nested loops in compute_G. For now, we implement L based on direct A[b,h,c,:] values per j using cumsum at i >= j. We'll implement the nested loops below.
        # Simpler: We will compute L[i,j,h] only when i >= j by using masks; for i < j, L is 0. For i >= j, we need cumsum_j[j].
        # We'll emulate cumsum_j[j] by computing a vector of sums per j in the next nested loop. Triton supports loops but not Python list assignment here; instead, compute sum per j with inner loop.
        # Implement cumsum_j[j] directly in the j-loop:
        for j in range(0, S):
            # cumsum_j[j] = sum_{t=0..j} A[b,h,c,t]
            sum_j = tl.zeros((), dtype=tl.float32)
            for t in range(0, S):
                # mask t <= j
                if t <= j:
                    a_val = tl.load(A_base + t * A_s)
                    sum_j += a_val
            # Store only if i >= j; otherwise L[i,j,h] = 0
            if i >= j:
                l_val = tl.exp(sum_j)  # exp(cumsum_j[j])
            else:
                l_val = tl.zeros((), dtype=tl.float32)
            # Store L[b, c, i, j, h]
            L_ptr_ijh = L_base + i * L_i + j * L_j + h * L_h
            tl.store(L_ptr_ijh, l_val)


# Triton kernel: compute G[i,j,h] = sum_n (C_exp[b,c,i,h,n] * sum_{j'} B_exp[b,c,j',h,n] * L[b,c,i,j',h])
@triton.jit
def compute_G_kernel(
    B_exp_ptr, C_exp_ptr, L_ptr, G_ptr,
    Bsz: tl.constexpr, Csz: tl.constexpr, Hsz: tl.constexpr, Nsz: tl.constexpr, S: tl.constexpr,
    B_bs, B_cs, B_i, B_h, B_n,
    C_bs, C_cs, C_i, C_h, C_n,
    L_bs, L_cs, L_i, L_j, L_h,
    G_bs, G_cs, G_i, G_j, G_h,
):
    b = tl.program_id(0)
    c = tl.program_id(1)

    # We will iterate over i, h, j, n. Use 1D grid and nested loops.
    for i in range(0, S):
        for h in range(0, Hsz):
            for j in range(0, S):
                acc = tl.zeros((), dtype=tl.float32)
                for n in range(0, Nsz):
                    # Load C_exp[b, c, i, h, n]
                    ce_ptrs = C_exp_ptr + b * C_bs + c * C_cs + i * C_i + h * C_h + n * C_n
                    ce_val = tl.load(ce_ptrs)
                    # Compute inner = sum_{j'} B_exp[b, c, j', h, n] * L[b, c, i, j', h]
                    inner = tl.zeros((), dtype=tl.float32)
                    for j_prime in range(0, S):
                        be_ptrs = B_exp_ptr + b * B_bs + c * B_cs + j_prime * B_i + h * B_h + n * B_n
                        be_val = tl.load(be_ptrs)
                        l_ptrs = L_ptr + b * L_bs + c * L_cs + i * L_i + j_prime * L_j + h * L_h
                        l_val = tl.load(l_ptrs)
                        inner += be_val * l_val
                    acc += ce_val * inner
                # Store G[b, c, i, j, h]
                G_ptrs = G_ptr + b * G_bs + c * G_cs + i * G_i + j * G_j + h * G_h
                tl.store(G_ptrs, acc)


# Triton kernel: compute Y_diag[b, c, i, h, d] = sum_j M[b, c, i, j, h] * hidden[b, c, j, h, d]
@triton.jit
def compute_Y_diag_kernel(
    M_ptr, hidden_ptr, Y_ptr,
    Bsz: tl.constexpr, Csz: tl.constexpr, Hsz: tl.constexpr, S: tl.constexpr, head_dim: tl.constexpr,
    M_bs, M_cs, M_i, M_j, M_h,
    h_bs, h_cs, h_s, h_h, h_d,
    Y_bs, Y_cs, Y_i, Y_h, Y_d,
):
    b = tl.program_id(0)
    c = tl.program_id(1)
    for i in range(0, S):
        for h in range(0, Hsz):
            for d in range(0, head_dim):
                acc = tl.zeros((), dtype=tl.float32)
                for j in range(0, S):
                    m_ptrs = M_ptr + b * M_bs + c * M_cs + i * M_i + j * M_j + h * M_h
                    m_val = tl.load(m_ptrs)
                    h_ptrs = hidden_ptr + b * h_bs + c * h_cs + j * h_s + h * h_h + d * h_d
                    h_val = tl.load(h_ptrs)
                    acc += m_val * h_val
                Y_ptrs = Y_ptr + b * Y_bs + c * Y_cs + i * Y_i + h * Y_h + d * Y_d
                tl.store(Y_ptrs, acc)


class ModelNew(torch.nn.Module):
    def __init__(self, batch_size, num_chunks, head_dim):
        super().__init__()
        self.batch_size = batch_size
        self.num_chunks = num_chunks
        self.head_dim = head_dim

    def forward(self, hidden_states, A_cumsum, B, C):
        # Ensure tensors are on CUDA
        device = hidden_states.device
        Bsz = hidden_states.size(0)
        Csz = hidden_states.size(1)
        S = hidden_states.size(2)
        Hsz = hidden_states.size(3)
        head_dim = hidden_states.size(4)

        # Create A_expanded and L (A_cumsum expanded for H, but we need original A_cumsum)
        # We'll keep A_cumsum as-is; reference uses A_cumsum: [B, H, C, S]
        # A_cumsum: [B, H, C, S] -> we will use it directly for cumsum computation inside Triton.

        # Launch build_L_kernel
        A = A_cumsum  # [B, H, C, S]
        L = torch.empty((Bsz, Csz, S, S, Hsz), device=device, dtype=torch.float32)

        A_bs, A_hs, A_cs, A_s = A.stride()  # (B, H, C, S)
        L_bs, L_cs, L_i, L_j, L_h = L.stride()
        grid_L = (Bsz, Csz, Hsz)
        build_L_kernel[grid_L](
            A, L,
            Bsz, Csz, Hsz, S,
            A_bs, A_hs, A_cs, A_s,
            L_bs, L_cs, L_i, L_j, L_h,
            num_warps=1, num_stages=1
        )

        # Prepare B_exp and C_exp: repeat_interleave along H (NUM_HEADS // N_GROUPS = 4)
        N_GROUPS = 8
        NUM_HEADS = 32
        expand_factor = NUM_HEADS // N_GROUPS
        B_exp = B.repeat_interleave(expand_factor, dim=3)  # [B, C, S, H, N]
        C_exp = C.repeat_interleave(expand_factor, dim=3)  # [B, C, S, H, N]

        # Compute G = sum_n (C_exp * B_exp) with lower-triangular L included; we'll use M = G * L later.
        # To keep code simple, compute G directly in Triton without using L here. We will precompute G without L.
        # Then we will use L to form M and compute Y_diag. However, reference requires M = G * L. Since G depends on L, we need to compute G taking L into account? Actually, G is computed as above and then M = G * L. Let's implement that next.
        # Note: The original code constructs G without using L, then multiplies by L. Our logic will follow the same.

        # Launch compute_G_kernel over (B, C), and it will loop over i, h, j, n
        G = torch.empty((Bsz, Csz, S, S, Hsz), device=device, dtype=torch.float32)

        B_bs, B_cs, B_i, B_h, B_n = B_exp.stride()  # (B, C, S, H, N)
        C_bs, C_cs, C_i, C_h, C_n = C_exp.stride()  # (B, C, S, H, N)
        L_bs, L_cs, L_i, L_j, L_h = L.stride()      # (B, C, S, S, H)
        G_bs, G_cs, G_i, G_j, G_h = G.stride()

        grid_G = (Bsz, Csz)
        compute_G_kernel[grid_G](
            B_exp, C_exp, L, G,
            Bsz, Csz, Hsz, 128, S,  # Nsz assumed 128
            B_bs, B_cs, B_i, B_h, B_n,
            C_bs, C_cs, C_i, C_h, C_n,
            L_bs, L_cs, L_i, L_j, L_h,
            G_bs, G_cs, G_i, G_j, G_h,
            num_warps=1, num_stages=1
        )

        # Compute M = G * L
        M = G * L

        # Compute Y_diag using Triton kernel
        hidden_f32 = hidden_states.to(torch.float32)
        Y = torch.empty((Bsz, Csz, S, Hsz, head_dim), device=device, dtype=torch.float32)

        M_bs, M_cs, M_i, M_j, M_h = M.stride()
        h_bs, h_cs, h_s, h_h, h_d = hidden_f32.stride()  # (B, C, S, H, D)
        Y_bs, Y_cs, Y_i, Y_h, Y_d = Y.stride()

        grid_Y = (Bsz, Csz)
        compute_Y_diag_kernel[grid_Y](
            M, hidden_f32, Y,
            Bsz, Csz, Hsz, S, head_dim,
            M_bs, M_cs, M_i, M_j, M_h,
            h_bs, h_cs, h_s, h_h, h_d,
            Y_bs, Y_cs, Y_i, Y_h, Y_d,
            num_warps=1, num_stages=1
        )

        return Y.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
