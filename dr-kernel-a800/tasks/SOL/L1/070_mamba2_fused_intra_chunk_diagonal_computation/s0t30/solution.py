import torch
import triton
import triton.language as tl

# Kernel 1: Build L = exp(cumsum(A)) with lower-triangular mask per (b, c, h)
@triton.jit
def build_L_kernel(
    A_ptr,            # *float32, [B, H, C, S]
    L_ptr,            # *float32, [B, C, S, S, H]
    Bsz: tl.constexpr, Csz: tl.constexpr, H: tl.constexpr, S: tl.constexpr,
    stride_A_b, stride_A_h, stride_A_c, stride_A_s,
    stride_L_b, stride_L_c, stride_L_i, stride_L_j, stride_L_h,
):
    b = tl.program_id(0)
    c = tl.program_id(1)
    h = tl.program_id(2)

    # We'll compute L across all i and j for this (b, c, h). One Triton program handles the full SxS tile.
    # Loop over j (source positions) to build cumsum.
    cumsum = tl.zeros((S,), dtype=tl.float32)
    for j in range(0, S):
        # Load A[b, h, c, j] as scalar
        A_off = b * stride_A_b + h * stride_A_h + c * stride_A_c + j * stride_A_s
        a_val = tl.load(A_ptr + A_off)
        cumsum[j] = cumsum[j - 1] + a_val if j > 0 else a_val
        # For each i >= j, store exp(cumsum[j]) in L
        for i in range(j, S):
            L_off = b * stride_L_b + c * stride_L_c + i * stride_L_i + j * stride_L_j + h * stride_L_h
            # Zero for masked positions (i < j) are already zero-initialized in PyTorch before launch.
            L_val = tl.exp(cumsum[j])
            tl.store(L_ptr + L_off, L_val)

# Kernel 2: Compute G[i, j, h] = sum_n C_expanded[b, c, i, n, j, h] * B_expanded[b, c, j, n, i, h]
# B_expanded: [B, C, S, H, N], C_expanded: [B, C, S, H, N]
# Output G: [B, C, S, S, H]
@triton.jit
def compute_G_kernel(
    B_exp_ptr,        # *float32, [B, C, S, H, N]
    C_exp_ptr,        # *float32, [B, C, S, H, N]
    G_ptr,            # *float32, [B, C, S, S, H]
    Bsz: tl.constexpr, Csz: tl.constexpr, H: tl.constexpr, S: tl.constexpr, N: tl.constexpr,
    stride_B_b, stride_B_c, stride_B_i, stride_B_h, stride_B_n,
    stride_C_b, stride_C_c, stride_C_i, stride_C_h, stride_C_n,
    stride_G_b, stride_G_c, stride_G_i, stride_G_j, stride_G_h,
):
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)
    j = tl.program_id(3)
    h = tl.program_id(4)

    # Sum over n: contraction
    g_val = tl.zeros((), dtype=tl.float32)
    for n in range(0, N):
        # Load B_exp[b, c, j, h, n] and C_exp[b, c, i, h, n]
        B_off = b * stride_B_b + c * stride_B_c + j * stride_B_i + h * stride_B_h + n * stride_B_n
        C_off = b * stride_C_b + c * stride_C_c + i * stride_C_i + h * stride_C_h + n * stride_C_n
        b_val = tl.load(B_exp_ptr + B_off)
        c_val = tl.load(C_exp_ptr + C_off)
        g_val += c_val * b_val
    # Store G[b, c, i, j, h]
    G_off = b * stride_G_b + c * stride_G_c + i * stride_G_i + j * stride_G_j + h * stride_G_h
    tl.store(G_ptr + G_off, g_val)

# Kernel 3: Compute Y_diag[b, c, i, h, d] = sum_j M[b, c, i, j, h] * hidden_states[b, c, j, h, d]
@triton.jit
def compute_Y_diag_kernel(
    M_ptr,            # *float32, [B, C, S, S, H]
    hidden_ptr,       # *float32, [B, C, S, H, D]
    Y_ptr,            # *float32, [B, C, S, H, D]
    Bsz: tl.constexpr, Csz: tl.constexpr, H: tl.constexpr, S: tl.constexpr, D: tl.constexpr,
    stride_M_b, stride_M_c, stride_M_i, stride_M_j, stride_M_h,
    stride_hidden_b, stride_hidden_c, stride_hidden_j, stride_hidden_h, stride_hidden_d,
    stride_Y_b, stride_Y_c, stride_Y_i, stride_Y_h, stride_Y_d,
):
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)
    h = tl.program_id(3)
    d = tl.program_id(4)

    # Reduce over j
    y_val = tl.zeros((), dtype=tl.float32)
    for j in range(0, S):
        M_off = b * stride_M_b + c * stride_M_c + i * stride_M_i + j * stride_M_j + h * stride_M_h
        hidden_off = b * stride_hidden_b + c * stride_hidden_c + j * stride_hidden_j + h * stride_hidden_h + d * stride_hidden_d
        M_val = tl.load(M_ptr + M_off)
        hidden_val = tl.load(hidden_ptr + hidden_off)
        y_val += M_val * hidden_val
    # Store Y[b, c, i, h, d]
    Y_off = b * stride_Y_b + c * stride_Y_c + i * stride_Y_i + h * stride_Y_h + d * stride_Y_d
    tl.store(Y_ptr + Y_off, y_val)

def launch_build_L(A, L):
    B, H, C, S = A.shape
    # Ensure A is contiguous and float32
    A_ = A.contiguous().to(torch.float32)
    # Allocate L as float32 and zero-initialize (we only store i>=j positions)
    L_ = torch.zeros((B, C, S, S, H), dtype=torch.float32, device=A.device)
    # Strides
    stride_A_b, stride_A_h, stride_A_c, stride_A_s = A_.stride()
    stride_L_b, stride_L_c, stride_L_i, stride_L_j, stride_L_h = L_.stride()
    # Launch one Triton program per (b, c, h)
    grid = (B, C, H)
    build_L_kernel[grid](
        A_, L_,
        B, C, H, S,
        stride_A_b, stride_A_h, stride_A_c, stride_A_s,
        stride_L_b, stride_L_c, stride_L_i, stride_L_j, stride_L_h,
        num_warps=4, num_stages=2,
    )

def launch_compute_G(B_exp, C_exp, G):
    B, C, S, H, N = B_exp.shape
    # Ensure inputs are contiguous float32
    B_exp_ = B_exp.contiguous().to(torch.float32)
    C_exp_ = C_exp.contiguous().to(torch.float32)
    G_ = torch.empty((B, C, S, S, H), dtype=torch.float32, device=B_exp.device)
    # Strides
    stride_B_b, stride_B_c, stride_B_i, stride_B_h, stride_B_n = B_exp_.stride()
    stride_C_b, stride_C_c, stride_C_i, stride_C_h, stride_C_n = C_exp_.stride()
    stride_G_b, stride_G_c, stride_G_i, stride_G_j, stride_G_h = G_.stride()
    # Grid: one program per (b, c, i, j, h)
    grid = (B, C, S, S, H)
    compute_G_kernel[grid](
        B_exp_, C_exp_, G_,
        B, C, H, S, N,
        stride_B_b, stride_B_c, stride_B_i, stride_B_h, stride_B_n,
        stride_C_b, stride_C_c, stride_C_i, stride_C_h, stride_C_n,
        stride_G_b, stride_G_c, stride_G_i, stride_G_j, stride_G_h,
        num_warps=4, num_stages=2,
    )

def launch_compute_Y_diag(M, hidden_states, Y, head_dim):
    B, C, S, H, D = hidden_states.shape
    # Ensure inputs are contiguous float32
    M_ = M.contiguous().to(torch.float32)
    hidden_ = hidden_states.contiguous().to(torch.float32)
    Y_ = torch.empty((B, C, S, H, D), dtype=torch.float32, device=M.device)
    # Strides
    stride_M_b, stride_M_c, stride_M_i, stride_M_j, stride_M_h = M_.stride()
    stride_hidden_b, stride_hidden_c, stride_hidden_j, stride_hidden_h, stride_hidden_d = hidden_.stride()
    stride_Y_b, stride_Y_c, stride_Y_i, stride_Y_h, stride_Y_d = Y_.stride()
    # Grid: one program per (b, c, i, h, d)
    grid = (B, C, S, H, D)
    compute_Y_diag_kernel[grid](
        M_, hidden_, Y_,
        B, C, H, S, D,
        stride_M_b, stride_M_c, stride_M_i, stride_M_j, stride_M_h,
        stride_hidden_b, stride_hidden_c, stride_hidden_j, stride_hidden_h, stride_hidden_d,
        stride_Y_b, stride_Y_c, stride_Y_i, stride_Y_h, stride_Y_d,
        num_warps=4, num_stages=2,
    )

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Constants
        self.S = 128
        self.H = 32
        self.N_GROUPS = 8
        self.NUM_HEADS = 32
        self.INTERLEAVE = self.NUM_HEADS // self.N_GROUPS  # 4

    def forward(self, hidden_states: torch.Tensor, A_cumsum: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        # Shapes
        Bsz, Csz, S, H, head_dim = hidden_states.shape
        # Ensure consistent S (reference uses 128). If not, we can't exactly match; but the provided workloads use S=128.
        assert S == self.S, f"Expected S=128, got {S}"

        # 1) Build L in Triton
        L = torch.empty((Bsz, Csz, self.S, self.S, H), dtype=torch.float32, device=hidden_states.device)
        launch_build_L(A_cumsum, L)  # Triton kernel

        # 2) Expand B and C along H (repeat_interleave by 4) and compute G in Triton
        # Ensure B and C are float32
        B_f = B.to(torch.float32)
        C_f = C.to(torch.float32)
        B_exp = B_f.repeat_interleave(self.INTERLEAVE, dim=3)  # [B, C, S, H, N]
        C_exp = C_f.repeat_interleave(self.INTERLEAVE, dim=3)  # [B, C, S, H, N]
        assert B_exp.shape == (Bsz, Csz, self.S, self.NUM_HEADS, B_f.shape[-1])
        N = B_exp.shape[-1]
        G = torch.empty((Bsz, Csz, self.S, self.S, H), dtype=torch.float32, device=hidden_states.device)
        launch_compute_G(B_exp, C_exp, G)

        # 3) Multiply G and L to get M
        M = G * L  # broadcast-safe elementwise multiply

        # 4) Compute Y_diag by contracting M with hidden_states over j and d, then cast to bfloat16
        Y = torch.empty((Bsz, Csz, self.S, H, head_dim), dtype=torch.float32, device=hidden_states.device)
        launch_compute_Y_diag(M, hidden_states.to(torch.float32), Y, head_dim)
        return Y.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
