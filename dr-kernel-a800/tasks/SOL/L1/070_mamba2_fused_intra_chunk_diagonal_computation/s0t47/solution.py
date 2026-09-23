import torch
import triton
import triton.language as tl

# Triton kernel: build L with causal mask L[i, j, h] = exp(cumsum(A[b, h, c, j])) if i >= j, else 0
@triton.jit
def build_L_kernel(
    A_ptr, L_ptr,
    Bsz, C, H, S,
    A_bs, A_hs, A_cs, A_s,
    L_b, L_c, L_i, L_j, L_h,
):
    b = tl.program_id(0)
    c = tl.program_id(1)
    h = tl.program_id(2)

    # Precompute base offsets
    base_A = b * A_bs + h * A_hs + c * A_cs

    # Compute cumsum along j for each i: cumsum_j[j] = sum_{t=0..j} A[b, h, c, t]
    # Then set L[i, j, h] = exp(cumsum_j[j]) if i >= j else 0
    for i in tl.range(0, 128):  # i in [0..127]
        cumsum = tl.zeros((), dtype=tl.float32)
        for j in tl.range(0, 128):  # j in [0..127]
            a_ptrs = A_ptr + base_A + j * A_s
            a_val = tl.load(a_ptrs)
            cumsum += a_val
            # If i >= j, set L[i, j, h] = exp(cumsum), else 0
            if i >= j:
                l_ptrs = L_ptr + b * L_b + c * L_c + i * L_i + j * L_j + h * L_h
                tl.store(l_ptrs, tl.exp(cumsum))
            else:
                l_ptrs = L_ptr + b * L_b + c * L_c + i * L_i + j * L_j + h * L_h
                tl.store(l_ptrs, 0.0)


# Triton kernel: compute G[i, j, h] = sum_n C_exp[b, c, i, n, j, h] * B_exp[b, c, j, n, i, h]
@triton.jit
def compute_G_kernel(
    B_exp_ptr, C_exp_ptr, G_ptr,
    Bsz, C, H, S, N,
    B_bs, B_cs, B_s, B_h, B_n,  # strides for B_exp
    C_bs, C_cs, C_s, C_h, C_n,  # strides for C_exp
    G_bs, G_cs, G_i, G_j, G_h,  # strides for G
):
    b = tl.program_id(0)
    c = tl.program_id(1)
    h = tl.program_id(2)

    base_B = b * B_bs + c * B_cs + h * B_h
    base_C = b * C_bs + c * C_cs + h * C_h

    for i in tl.range(0, 128):
        for j in tl.range(0, 128):
            acc = tl.zeros((), dtype=tl.float32)
            for n in tl.range(0, 128):  # N assumed up to 128; actual N can be <128, but loops must be static
                # C_exp[b, c, i, n, j, h] and B_exp[b, c, j, n, i, h]
                ce_ptrs = C_exp_ptr + base_C + i * C_s + n * C_n + j * C_j + h * C_h
                be_ptrs = B_exp_ptr + base_B + j * B_s + n * B_n + i * B_i + h * B_h
                ce_val = tl.load(ce_ptrs)
                be_val = tl.load(be_ptrs)
                acc += ce_val * be_val
            # Store G[b, c, i, j, h]
            g_ptrs = G_ptr + b * G_bs + c * G_cs + i * G_i + j * G_j + h * G_h
            tl.store(g_ptrs, acc)


# Triton kernel: compute Y_diag[b, c, i, h, d] = sum_j M[b, c, i, j, h] * hidden[b, c, j, h, d]
@triton.jit
def compute_Y_diag_kernel(
    M_ptr, hidden_ptr, Y_ptr,
    Bsz, C, S, H, D,
    M_b, M_c, M_i, M_j, M_h,
    hidden_b, hidden_c, hidden_s, hidden_h, hidden_d,
    Y_b, Y_c, Y_i, Y_h, Y_d,
):
    b = tl.program_id(0)
    c = tl.program_id(1)

    for i in tl.range(0, 128):
        for h in tl.range(0, 32):
            for d in tl.range(0, D):
                acc = tl.zeros((), dtype=tl.float32)
                for j in tl.range(0, 128):
                    m_ptrs = M_ptr + b * M_b + c * M_c + i * M_i + j * M_j + h * M_h
                    h_ptrs = hidden_ptr + b * hidden_b + c * hidden_c + j * hidden_s + h * hidden_h + d * hidden_d
                    m_val = tl.load(m_ptrs)
                    h_val = tl.load(h_ptrs)
                    acc += m_val * h_val
                y_ptrs = Y_ptr + b * Y_b + c * Y_c + i * Y_i + h * Y_h + d * Y_d
                tl.store(y_ptrs, acc)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Constants
        self.S = 128
        self.H = 32
        self.N_GROUPS = 8

    def forward(self, hidden_states: torch.Tensor, A_cumsum: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        """
        Compute Y_diag following Mamba2 SSD logic:
        1) Build L[i, j, h] with lower-triangular causal mask: L[i,j,h] = exp(cumsum(A[b,h,c,0:j])[j]) if i>=j else 0
        2) Expand B and C along H by repeat_interleave(NUM_HEADS // N_GROUPS = 4).
        3) Compute G[i, j, h] = sum_n(C_exp[b,c,i,n,j,h] * B_exp[b,c,j,n,i,h]).
        4) M = G * L (element-wise multiply).
        5) Y_diag[b, c, i, h, d] = sum_j M[b, c, i, j, h] * hidden_states[b, c, j, h, d].
        Output dtype: bfloat16, shape [B, C, S, H, head_dim].
        """
        Bsz = hidden_states.size(0)
        C_num = hidden_states.size(1)
        S = hidden_states.size(2)
        H = hidden_states.size(3)
        head_dim = hidden_states.size(4)

        # Ensure dtype float32 for computation
        A = A_cumsum.to(torch.float32)
        B_t = B.to(torch.float32)
        C_t = C.to(torch.float32)
        hidden = hidden_states.to(torch.float32)

        # Allocate L: [B, C, S, S, H]
        L = torch.empty((Bsz, C_num, self.S, self.S, H), dtype=torch.float32, device=hidden.device)

        # Allocate B_expanded and C_expanded: repeat along H dimension (repeat_interleave(NUM_HEADS//N_GROUPS=4))
        repeat_factor = self.H // self.N_GROUPS  # 4
        B_exp = B_t.repeat_interleave(repeat_factor, dim=3)  # [B, C, S, H, N]
        C_exp = C_t.repeat_interleave(repeat_factor, dim=3)  # [B, C, S, H, N]

        # Allocate G: [B, C, S, S, H]
        G = torch.empty((Bsz, C_num, self.S, self.S, H), dtype=torch.float32, device=hidden.device)

        # Launch Triton kernel: build_L_kernel
        grid_L = (Bsz, C_num, H)
        build_L_kernel[grid_L](
            A, L,
            Bsz, C_num, H, self.S,
            A.stride(0), A.stride(1), A.stride(2), A.stride(3),
            L.stride(0), L.stride(1), L.stride(2), L.stride(3), L.stride(4),
        )

        # Launch Triton kernel: compute_G_kernel
        N = B_exp.size(-1)  # state_size (e.g., 128)
        grid_G = (Bsz, C_num, H)
        compute_G_kernel[grid_G](
            B_exp, C_exp, G,
            Bsz, C_num, H, self.S, N,
            B_exp.stride(0), B_exp.stride(1), B_exp.stride(2), B_exp.stride(3), B_exp.stride(4),
            C_exp.stride(0), C_exp.stride(1), C_exp.stride(2), C_exp.stride(3), C_exp.stride(4),
            G.stride(0), G.stride(1), G.stride(2), G.stride(3), G.stride(4),
        )

        # Compute M = G * L
        # Use elementwise multiplication (torch) here; since G and L are Triton outputs, ensure they are on device.
        M = G * L

        # Allocate Y_diag: [B, C, S, H, head_dim]
        Y = torch.empty((Bsz, C_num, self.S, H, head_dim), dtype=torch.float32, device=hidden.device)

        # Launch Triton kernel: compute_Y_diag_kernel
        grid_Y = (Bsz, C_num)
        compute_Y_diag_kernel[grid_Y](
            M, hidden, Y,
            Bsz, C_num, self.S, H, head_dim,
            M.stride(0), M.stride(1), M.stride(2), M.stride(3), M.stride(4),
            hidden.stride(0), hidden.stride(1), hidden.stride(2), hidden.stride(3), hidden.stride(4),
            Y.stride(0), Y.stride(1), Y.stride(2), Y.stride(3), Y.stride(4),
        )

        # Return in bfloat16 to match original model
        return Y.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
