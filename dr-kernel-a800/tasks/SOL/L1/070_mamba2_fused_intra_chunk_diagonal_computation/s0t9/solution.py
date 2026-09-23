import torch
import triton
import triton.language as tl

# Kernel 1: Build L_flat with causal lower-triangular exp(cumsum) over j for each i
# L_flat layout: flatten [B, C, S, S, H] => total = B*C*S*S*H
@triton.jit
def build_L_flat_kernel(
    A_ptr,        # [B, H, C, S], float32
    L_flat_ptr,   # [total], float32, where total = B*C*S*S*H
    Bsz: tl.constexpr,
    Csz: tl.constexpr,
    H: tl.constexpr,
    S: tl.constexpr,
):
    pid = tl.program_id(0)  # over total elements
    total = Bsz * Csz * S * S * H
    # Derive (b, c, i, j, h) from pid
    # Layout: h varies fastest, then j, then i, then c, then b
    h = pid % H
    tmp = pid // H
    j = tmp % S
    tmp = tmp // S
    i = tmp % S
    tmp = tmp // S
    c = tmp % Csz
    b = tmp // Csz

    # Only compute if i >= j (lower-triangular)
    if i >= j:
        # cumsum across j positions up to i
        cum = 0.0
        for k in range(0, S):
            if k <= i:
                a_off = b * (H * Csz * S) + h * (Csz * S) + c * S + k
                a_val = tl.load(A_ptr + a_off)  # float32
                cum += a_val
            else:
                cum += 0.0
        l_val = tl.exp(cum)
    else:
        l_val = 0.0

    off = (b * (Csz * S * S * H)) + (c * (S * S * H)) + (i * (S * H)) + (j * H) + h
    tl.store(L_flat_ptr + off, l_val)


# Kernel 2: Compute G_flat where G[i, j, h] = sum_n C_exp[b, c, i, n, j, h] * B_exp[b, c, j, n, i, h]
# Inputs: B_exp and C_exp are [B, C, S, H, N], float32
@triton.jit
def compute_G_flat_kernel(
    B_exp_ptr,    # [B, C, S, H, N], float32
    C_exp_ptr,    # [B, C, S, H, N], float32
    G_flat_ptr,   # [total], float32, total = B*C*S*S*H
    Bsz: tl.constexpr,
    Csz: tl.constexpr,
    H: tl.constexpr,
    S: tl.constexpr,
    N: tl.constexpr,  # typically 128
):
    pid = tl.program_id(0)  # over total elements
    total = Bsz * Csz * S * S * H
    h = pid % H
    tmp = pid // H
    j = tmp % S
    tmp = tmp // S
    i = tmp % S
    tmp = tmp // S
    c = tmp % Csz
    b = tmp // Csz

    acc = 0.0
    # Sum over n from 0 to N-1
    for n in range(0, N):
        # B_exp[b, c, j, h, n]
        B_off = b * (Csz * S * H * N) + c * (S * H * N) + j * (H * N) + h * N + n
        # C_exp[b, c, i, n, j, h]
        C_off = b * (Csz * S * H * N) + c * (S * H * N) + i * (H * N) + j * (H * N) + n
        B_val = tl.load(B_exp_ptr + B_off)
        C_val = tl.load(C_exp_ptr + C_off)
        acc += C_val * B_val

    off = (b * (Csz * S * S * H)) + (c * (S * S * H)) + (i * (S * H)) + (j * H) + h
    tl.store(G_flat_ptr + off, acc)


# Kernel 3: Compute Y_diag flat from G_flat and hidden_states
# hidden_states_flat layout: [B, C, S, H, head_dim], we flatten as B*C*S*H*head_dim
# Each program handles one (b, c, i, h, d), loops over j, accumulates sum
@triton.jit
def compute_Y_diag_kernel(
    G_flat_ptr,            # [B*C*S*S*H], float32
    hidden_flat_ptr,       # [B*C*S*H*head_dim], float32
    Y_flat_ptr,            # [B*C*S*H*head_dim], float32
    Bsz: tl.constexpr,
    Csz: tl.constexpr,
    H: tl.constexpr,
    S: tl.constexpr,
    head_dim: tl.constexpr,
):
    pid = tl.program_id(0)  # over planes (B*C*S*H)
    d = tl.program_id(1)    # over head_dim
    total_planes = Bsz * Csz * S * H
    if pid >= total_planes:
        return
    # derive (b, c, i, h) from pid
    h = pid % H
    tmp = pid // H
    i = tmp % S
    tmp = tmp // S
    c = tmp % Csz
    b = tmp // Csz

    acc = 0.0
    for j in range(0, S):
        g_off = (b * (Csz * S * S * H)) + (c * (S * S * H)) + (i * (S * H)) + (j * H) + h
        g_val = tl.load(G_flat_ptr + g_off)
        hs_off = (b * (Csz * S * H * head_dim)) + (c * (S * H * head_dim)) + (i * (H * head_dim)) + (h * head_dim) + d
        hs_val = tl.load(hidden_flat_ptr + hs_off)
        acc += g_val * hs_val

    y_off = (b * (Csz * S * H * head_dim)) + (c * (S * H * head_dim)) + (i * (H * head_dim)) + (h * head_dim) + d
    tl.store(Y_flat_ptr + y_off, acc)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.CHUNK_SIZE = 128
        self.NUM_HEADS = 32
        self.N_GROUPS = 8

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

        # Ensure contiguity and correct dtype
        hidden_states = hidden_states.contiguous().to(torch.float32)
        A = A_cumsum.contiguous().to(torch.float32)  # [B, H, C, S]
        B_t = B.contiguous().to(torch.float32)       # [B, C, S, N_GROUPS, N]
        C_t = C.contiguous().to(torch.float32)       # [B, C, S, N_GROUPS, N]

        # Expand B and C along H by repeat_interleave(NUM_HEADS // N_GROUPS = 4) to match H=32
        B_exp = B_t.repeat_interleave(self.NUM_HEADS // self.N_GROUPS, dim=3)  # [B, C, S, H, N]
        C_exp = C_t.repeat_interleave(self.NUM_HEADS // self.N_GROUPS, dim=3)  # [B, C, S, H, N]

        # Allocate flat buffers (float32 for computation)
        total_L = Bsz * Csz * S * S * H
        L_flat = torch.empty(total_L, dtype=torch.float32, device=hidden_states.device)

        total_G = total_L
        G_flat = torch.empty(total_G, dtype=torch.float32, device=hidden_states.device)

        # Launch build_L_flat_kernel
        grid_L = (total_L,)
        build_L_flat_kernel[grid_L](A, L_flat, Bsz=Bsz, Csz=Csz, H=H, S=S)

        # Launch compute_G_flat_kernel
        grid_G = (total_G,)
        compute_G_flat_kernel[grid_G](B_exp, C_exp, G_flat, Bsz=Bsz, Csz=Csz, H=H, S=S, N=128)

        # Compute Y_diag: flatten hidden as [B, C, S, H, head_dim]
        hidden_flat = hidden_states.view(-1).contiguous()  # float32
        total_Y = Bsz * Csz * S * H * head_dim
        Y_flat = torch.empty(total_Y, dtype=torch.float32, device=hidden_states.device)

        # Launch compute_Y_diag_kernel
        grid_Y = (Bsz * Csz * S * H, head_dim)
        compute_Y_diag_kernel[grid_Y](G_flat, hidden_flat, Y_flat, Bsz=Bsz, Csz=Csz, H=H, S=S, head_dim=head_dim)

        # Reshape and cast to bfloat16 for return
        Y = Y_flat.view(Bsz, Csz, S, H, head_dim).to(torch.bfloat16)
        return Y


def run(*args):
    return ModelNew()(*args)
