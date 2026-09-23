import torch
import triton
import triton.language as tl

# Kernel 1: Build L_flat = exp(cumsum(A_lower_tri)) linearized as [B*C*S*S*H]
@triton.jit
def build_L_flat_kernel(
    A_ptr,  # A: [B, H, C, S]
    L_flat_ptr,  # [B*C*S*S*H], float32
    Bsz: tl.constexpr, Csz: tl.constexpr, H: tl.constexpr, S: tl.constexpr
):
    pid = tl.program_id(axis=0)
    total = Bsz * Csz * S * S * H
    if pid >= total:
        return

    # Decode (b, c, i, j, h) from linear pid
    tmp = pid
    h = tmp % H
    tmp = tmp // H
    j = tmp % S
    tmp = tmp // S
    i = tmp % S
    tmp = tmp // S
    c = tmp % Csz
    b = tmp // Csz

    # Only lower-triangular positions (i >= j) contribute
    if i < j:
        idx = ((b * Csz + c) * S + i) * (S * H) + (j * H + h)
        tl.store(L_flat_ptr + idx, 0.0)
        return

    # Compute cumsum of A[b, h, c, :] up to j
    cum = 0.0
    for jj in range(0, S):  # S is constexpr, so loop is unrolled
        if jj > i:
            break
        a_idx = ((b * H + h) * Csz * S + c * S + jj)
        a_val = tl.load(A_ptr + a_idx)
        cum += a_val
    l_val = tl.exp(cum)
    idx = ((b * Csz + c) * S + i) * (S * H) + (j * H + h)
    tl.store(L_flat_ptr + idx, l_val)


# Kernel 2: Compute G_flat[i, j, h] = sum_n C_exp[b, c, i, n, j, h] * B_exp[b, c, j, n, i, h]
# B_exp and C_exp are [B, C, S, H, N]
@triton.jit
def compute_G_flat_kernel(
    B_exp_ptr, C_exp_ptr, G_flat_ptr,
    Bsz: tl.constexpr, Csz: tl.constexpr, S: tl.constexpr, H: tl.constexpr, N: tl.constexpr
):
    pid = tl.program_id(axis=0)
    total = Bsz * Csz * S * S * H
    if pid >= total:
        return

    # Decode (b, c, i, j, h)
    tmp = pid
    h = tmp % H
    tmp = tmp // H
    j = tmp % S
    tmp = tmp // S
    i = tmp % S
    tmp = tmp // S
    c = tmp % Csz
    b = tmp // Csz

    sum_val = 0.0
    # Loop over n in 0..N-1 (N=128)
    for n in range(0, N):
        # C_exp[b, c, i, n, j, h] linear index:
        # ((b*Csz + c)*S*H*N + i*H*N + n*H + j)*H + h
        C_off = ((b * Csz + c) * S * H * N + i * (H * N) + n * H + j) * H + h
        # B_exp[b, c, j, n, i, h] linear index:
        # ((b*Csz + c)*S*H*N + j*H*N + n*H + i)*H + h
        B_off = ((b * Csz + c) * S * H * N + j * (H * N) + n * H + i) * H + h
        c_val = tl.load(C_exp_ptr + C_off)
        b_val = tl.load(B_exp_ptr + B_off)
        sum_val += c_val * b_val

    # Store to G_flat at linear index: ((b*Csz + c)*S + i)*S*H + (j*H + h)
    G_idx = ((b * Csz + c) * S + i) * (S * H) + (j * H + h)
    tl.store(G_flat_ptr + G_idx, sum_val)


# Kernel 3: Compute Y_diag_flat[b, c, i, h, d] = sum_j G[b, c, i, j, h] * hidden_states[b, c, j, h, d]
# hidden_states: [B, C, S, H, head_dim]
@triton.jit
def compute_Y_diag_kernel(
    G_flat_ptr, hidden_ptr, Y_flat_ptr,
    Bsz: tl.constexpr, Csz: tl.constexpr, S: tl.constexpr, H: tl.constexpr, head_dim: tl.constexpr
):
    pid = tl.program_id(axis=0)
    total = Bsz * Csz * S * H * head_dim
    if pid >= total:
        return

    # Decode (b, c, i, h, d)
    d = pid % head_dim
    tmp = pid // head_dim
    h = tmp % H
    tmp = tmp // H
    i = tmp % S
    tmp = tmp // S
    c = tmp % Csz
    b = tmp // Csz

    sum_y = 0.0
    for j in range(0, S):
        # G_flat index: ((b*Csz + c)*S + i)*S*H + (j*H + h)
        G_idx = ((b * Csz + c) * S + i) * (S * H) + (j * H + h)
        g = tl.load(G_flat_ptr + G_idx)
        # hidden_ptr[b, c, j, h, d] linear index:
        # ((b*Csz + c)*S*H*head_dim + j*H*head_dim + h*head_dim + d)
        h_off = ((b * Csz + c) * S * H * head_dim + j * (H * head_dim) + h * head_dim + d)
        hs_val = tl.load(hidden_ptr + h_off)
        sum_y += g * hs_val

    # Store to Y_flat at linear index: ((b*Csz + c)*S*H*head_dim + i*(H*head_dim) + h*head_dim + d)
    Y_idx = ((b * Csz + c) * S * H * head_dim + i * (H * head_dim) + h * head_dim + d)
    tl.store(Y_flat_ptr + Y_idx, sum_y)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.CHUNK_SIZE = 128  # S
        self.NUM_HEADS = 32    # H
        self.N_GROUPS = 8      # groups in B/C

    def forward(self, hidden_states: torch.Tensor, A_cumsum: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        """
        Compute Y_diag = sum_j M * hidden_states, where:
        - hidden_states: [B, C, S, H, head_dim]
        - A_cumsum: [B, H, C, S]
        - B: [B, C, S, N_GROUPS, N]
        - C: [B, C, S, N_GROUPS, N]
        Returns: [B, C, S, H, head_dim] in bfloat16.
        """
        assert hidden_states.is_cuda and A_cumsum.is_cuda and B.is_cuda and C.is_cuda, "All tensors must be on CUDA for Triton."
        Bsz, Csz, S, H, head_dim = hidden_states.shape
        assert S == self.CHUNK_SIZE, f"hidden_states chunk size must be {self.CHUNK_SIZE}, got {S}"
        assert H == self.NUM_HEADS, f"num_heads must be {self.NUM_HEADS}, got {H}"
        # Ensure contiguity
        hidden_states = hidden_states.contiguous()
        A = A_cumsum.contiguous()
        B = B.contiguous()
        C = C.contiguous()

        # Expand B and C along H by repeat_interleave(NUM_HEADS // N_GROUPS = 4) to match H=32
        # This is necessary for computing G.
        B_exp = B.repeat_interleave(self.NUM_HEADS // self.N_GROUPS, dim=3)  # [B, C, S, H, N]
        C_exp = C.repeat_interleave(self.NUM_HEADS // self.N_GROUPS, dim=3)  # [B, C, S, H, N]

        # Allocate flat buffers (float32 for computation)
        L_flat = torch.empty(Bsz * Csz * S * S * H, dtype=torch.float32, device=hidden_states.device)
        G_flat = torch.empty(Bsz * Csz * S * S * H, dtype=torch.float32, device=hidden_states.device)
        Y_flat = torch.empty(Bsz * Csz * S * H * head_dim, dtype=torch.float32, device=hidden_states.device)

        # Launch build_L_flat_kernel
        total_L = Bsz * Csz * S * S * H
        grid_L = (total_L,)
        build_L_flat_kernel[grid_L](
            A, L_flat,
            Bsz=Bsz, Csz=Csz, H=H, S=S
        )

        # Launch compute_G_flat_kernel
        total_G = Bsz


def run(*args):
    return ModelNew()(*args)
