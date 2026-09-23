import torch
import triton
import triton.language as tl

# Kernel 1: Build L_flat as a 1D vector of length B*C*S*S*H
# L_flat index ordering: for fixed (b,c), iterate over i in [0..S-1], then j in [0..S-1], then h in [0..H-1]
@triton.jit
def build_L_flat_kernel(
    A_ptr,            # [B, H, C, S]
    L_flat_ptr,       # [B*C*S*S*H], float32
    Bsz: tl.constexpr,
    Csz: tl.constexpr,
    H: tl.constexpr,
    S: tl.constexpr,
):
    total = Bsz * Csz * S * S * H
    pid = tl.program_id(0)
    # Decompose pid into (b, c, i, j, h)
    # Note: S and H are compile-time constants (128 and 32). We compute b,c from pid // (S*S*H),
    # then i from remaining // (S*H), j from remaining // H, h from remaining % H.
    # But Triton expects int divisions; better approach: restructure grid as 3D to avoid confusion.
    # Instead of relying on complex decomposition, we switch to a 3D launch below.
    # For now, assume 1D launch and compute indices via fixed S=128, H=32.
    # To keep code simple and correct, we will not rely on this 1D kernel in forward; see compute_L below.
    # Placeholder (not used in forward).
    pass


# 3D kernel to compute L per (b,c,i,h) row: for each i, build vector L[i, j, h] for j in [0..S-1]
@triton.jit
def build_L_rows_kernel(
    A_ptr,            # [B, H, C, S]
    L_ptr,            # [B, C, S, S, H] float32
    Bsz: tl.constexpr,
    Csz: tl.constexpr,
    H: tl.constexpr,
    S: tl.constexpr,
):
    # Grid: (B*C, S, H)
    # program_id(0) iterates over (b,c) pairs, program_id(1) over i, program_id(2) over h
    bc_id = tl.program_id(0)
    i = tl.program_id(1)
    h = tl.program_id(2)
    b = bc_id // Csz
    c = bc_id % Csz

    # Base offset for (b,h,c) in A_ptr
    base_A = b * (H * Csz * S) + h * (Csz * S) + c * S

    # Initialize cumulative sum vector cum[j]
    cum = tl.zeros([S], dtype=tl.float32)

    # Compute L row L[i, :, h]
    # For each j, include A[b,h,c,j] only if j <= i
    for j in range(0, S):
        a_val = tl.load(A_ptr + base_A + j)
        if j <= i:
            cum[j] = cum[j - 1] + a_val  # cum[j] = sum_{k=0..j} A[b,h,c,k], only if k<=i contributes
        else:
            cum[j] = cum[j - 1]  # maintain previous cum for j>i (no contribution)
        # L[b, c, i, j, h] = exp(cum[j]) if j <= i else 0
        L_val = tl.exp(cum[j])
        L_offset = b * (Csz * S * S * H) + c * (S * S * H) + i * (S * H) + j * H + h
        tl.store(L_ptr + L_offset, L_val)  # For j>i, cum[j]=cum[j-1]; exp is 0 at j=i but we store exp(cum[j]) with j>i would be exp(cum[j]) where cum[j] unchanged. Better to mask:
        # We can set L_val = 0 for j > i. However, cum[j] remains correct. To avoid confusion, we directly compute exp(cum[j]) and rely on j>i yielding exp of previous cum which is fine for lower-triangular mask since L is only defined when j<=i. So we don't need to set zero explicitly; the lower-triangular mask is applied by construction in forward by only using M=G*L where G has j>=i. Here L is computed row; for j>i we can set L_val=0.
        if j > i:
            L_val = 0.0
        tl.store(L_ptr + L_offset, L_val)


# Kernel 2: Compute G_flat as a 1D vector of length B*C*S*S*H
# G_flat index ordering: for fixed (b,c), iterate over i,j,h such that i varies fastest
@triton.jit
def compute_G_flat_kernel(
    B_exp_ptr,        # [B, C, S, H, N], float32
    C_exp_ptr,        # [B, C, S, H, N], float32
    G_flat_ptr,       # [B*C*S*S*H], float32
    Bsz: tl.constexpr,
    Csz: tl.constexpr,
    H: tl.constexpr,
    S: tl.constexpr,
    N: tl.constexpr,
):
    # Grid: (B*C, S, S, H) => total = B*C*S*S*H
    pid = tl.program_id(0)
    # Decompose pid into (b,c,i,j,h)
    # Use a simple mapping: pid -> (((b*c) * S*S*H) + (i*S*S*H) + (j*S*H) + h)
    # But Triton prefers simple grids. Instead, launch 1D and compute manually. To keep it robust, we'll use a 3D grid above.
    pass


# 3D kernel to compute G per (b,c,i,h) row across j: G[i, j, h]
@triton.jit
def compute_G_rows_kernel(
    B_exp_ptr,        # [B, C, S, H, N]
    C_exp_ptr,        # [B, C, S, H, N]
    G_ptr,            # [B, C, S, S, H], float32
    Bsz: tl.constexpr,
    Csz: tl.constexpr,
    H: tl.constexpr,
    S: tl.constexpr,
    N: tl.constexpr,
):
    # Grid: (B*C, S, H) => each program handles a fixed (b,c,i,h), loops over j
    bc_id = tl.program_id(0)
    i = tl.program_id(1)
    h = tl.program_id(2)
    b = bc_id // Csz
    c = bc_id % Csz

    # We need to compute G[i, j, h] for j in [0..S-1]
    # G[i, j, h] = sum_n C[b, c, i, n, j, h] * B[b, c, j, n, i, h]
    # We'll loop over j within the kernel and accumulate.
    for j in range(0, S):
        acc = 0.0
        # Loop over n from 0 to N-1 (N=128)
        for n in range(0, N):
            # B_exp: [B, C, S, H, N] => offset = b*(C*S*H*N) + c*(S*H*N) + i*(H*N) + h*N + n
            B_off = b * (Csz * S * H * N) + c * (S * H * N) + i * (H * N) + h * N + n
            # C_exp: [B, C, S, H, N] => offset = b*(C*S*H*N) + c*(S*H*N) + i*(H*N) + j*(H*N) + n
            C_off = b * (Csz * S * H * N) + c * (S * H * N) + i * (H * N) + j * (H * N) + n
            # Load B and C values for this (n)
            B_val = tl.load(B_exp_ptr + B_off)
            C_val = tl.load(C_exp_ptr + C_off)
            acc += C_val * B_val
        # Store G[b, c, i, j, h] = acc
        G_off = b * (Csz * S * S * H) + c * (S * S * H) + i * (S * H) + j * H + h
        tl.store(G_ptr + G_off, acc)


# Kernel 3: Compute Y_diag_flat as a 1D vector of length B*C*S*H*head_dim
# Y_diag_flat index ordering: for fixed (b,c,i,h), iterate d in [0..head_dim-1]
@triton.jit
def compute_Y_diag_kernel(
    G_ptr,            # [B, C, S, S, H], float32
    hidden_ptr,       # [B, C, S, H, head_dim], float32
    Y_flat_ptr,       # [B*C*S*H*head_dim], float32
    Bsz: tl.constexpr,
    Csz: tl.constexpr,
    H: tl.constexpr,
    S: tl.constexpr,
    head_dim: tl.constexpr,
):
    # Grid: (B*C*S*H, head_dim)
    pid_bc = tl.program_id(0)
    d = tl.program_id(1)
    # Decompose pid_bc into (b, c, i, h)
    bc = pid_bc // (S * H)
    rem = pid_bc % (S * H)
    i = rem // H
    h = rem % H
    b = bc // Csz
    c = bc % Csz

    # Compute Y_diag[b, c, i, h, d] = sum_j G[b, c, i, j, h] * hidden[b, c, j, h, d]
    total = 0.0
    for j in range(0, S):
        G_off = b * (Csz * S * S * H) + c * (S * S * H) + i * (S * H) + j * H + h
        G_val = tl.load(G_ptr + G_off)
        # hidden offset: b*(C*S*H*head_dim) + c*(S*H*head_dim) + j*(H*head_dim) + h*head_dim + d
        hidden_off = b * (Csz * S * H * head_dim) + c * (S * H * head_dim) + j * (H * head_dim) + h * head_dim + d
        hidden_val = tl.load(hidden_ptr + hidden_off)
        total += G_val * hidden_val
    # Store to Y_flat[pid_bc * head_dim + d]
    Y_idx = pid_bc * head_dim + d
    tl.store(Y_flat_ptr + Y_idx, total)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Constants from original code
        self.CHUNK_SIZE = 128  # S
        self.NUM_HEADS = 32     # H
        self.N_GROUPS = 8       # groups

    def forward(self, hidden_states: torch.Tensor, A_cumsum: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        """
        Triton-based forward:
        - hidden_states: [B, C, S, H, head_dim]
        - A_cumsum: [B, H, C, S]
        - B: [B, C, S, N_GROUPS, N]
        - C: [B, C, S, N_GROUPS, N]
        Returns Y_diag: [B, C, S, H, head_dim] in bfloat16.
        """
        assert hidden_states.is_cuda and A_cumsum.is_cuda and B.is_cuda and C.is_cuda, "All tensors must be on CUDA for Triton."
        Bsz, Csz, S, H, head_dim = hidden_states.shape
        assert S == self.CHUNK_SIZE, f"hidden_states chunk size must be {self.CHUNK_SIZE}, got {S}"
        assert H == self.NUM_HEADS, f"num_heads must be {self.NUM_HEADS}, got {H}"
        # Ensure contiguity
        hidden_states = hidden_states.contiguous()
        A = A_cumsum.contiguous()  # [B, H, C, S]
        B = B.contiguous()
        C = C.contiguous()

        # Permute A to [B, C, H, S] for easier indexing in Triton (we'll pass as is; kernels use dims)
        # No need to permute; we use A as [B, H, C, S] with strides.

        # Expand B and C along H by repeat_interleave(NUM_HEADS // N_GROUPS = 4, dim=3) to [B, C, S, H, N]
        B_exp = B.repeat_interleave(self.NUM_HEADS // self.N_GROUPS, dim=3).contiguous()
        C_exp = C.repeat_interleave(self.NUM_HEADS // self.N_GROUPS, dim=3).contiguous()

        # Allocate float32 buffers for computation
        L = torch.empty((Bsz, Csz, S, S, H), dtype=torch.float32, device=hidden_states.device)
        G = torch.empty((Bsz, Csz, S, S, H), dtype=torch.float32, device=hidden_states.device)
        Y_flat = torch.empty(Bsz * Csz * S * H * head_dim, dtype=torch.float32, device=hidden_states.device)

        # Launch build_L_rows_kernel (3D grid: (B*C, S, H))
        grid_L = (Bsz * Csz, S, H)
        build_L_rows_kernel[grid_L](
            A, L,
            Bsz=Bsz, Csz=Csz, H=H, S=S
        )

        # Launch compute_G_rows_kernel (3D grid: (B*C, S, H))
        N = C.shape[-1]  # state_size, typically 128
        grid_G = (Bsz * Csz, S, H)
        compute_G_rows_kernel[grid_G](
            B_exp, C_exp, G,
            Bsz=Bsz, Csz=Csz, H=H, S=S, N=N
        )

        # Compute M = G * L
        # Since L is float32, this is elementwise multiply. To keep everything in Triton, we can inline this in Y kernel
        # by loading G and L and multiplying. However, Triton kernels don't take other tensors as inputs for multiply here.
        # We'll prepare M = G * L in Python for the next kernel (small tensor), or multiply inside compute_Y_diag by loading both.
        # To avoid extra Python tensor, we can pass M as G; hidden_states and L are not required for Y_diag because Y_diag uses M
        # as G*L? Actually we do need M. Given Triton limitations across tensors, we'll compute Y_diag directly using G
        # and hidden_states by reconstructing L via same cumsum logic per j (but that would be redundant and slow).
        # Instead, we'll do M in Python: M = G * L (elementwise), then pass M to a Triton kernel for Y_diag.
        M = G * L  # float32

        # Launch compute_Y_diag_kernel (2D grid: (B*C*S*H, head_dim))
        grid_Y = (Bsz * Csz * S * H, head_dim)
        compute_Y_diag_kernel[grid_Y](
            M, hidden_states.float(),  # cast hidden_states to float32 for compute
            Y_flat,
            Bsz=Bsz, Csz=Csz, H=H, S=S, head_dim=head_dim
        )

        # Reshape and cast to bfloat16 to match original return type
        Y = Y_flat.view(Bsz, Csz, S, H, head_dim).to(torch.bfloat16)
        return Y


# If you want to keep the original run interface:
# def run(...):
#     return ModelNew().forward(...)


def run(*args):
    return ModelNew()(*args)
