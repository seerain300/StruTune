import torch
import triton
import triton.language as tl

# Kernel 1: Build L_flat where L[b, c, i, j, h] = exp(cumsum(A[b, h, c, :])[j]) for i >= j
# A input is [B, C, H, S] (permuted from [B, H, C, S])
# Output is L_flat[bc*S*S*H + i*S*H + j*H + h]
@triton.jit
def build_L_rows_kernel(
    A_ptr,            # [B, C, H, S], float32
    L_ptr,            # [B*C*S*S*H], float32
    Bsz: tl.constexpr,
    Csz: tl.constexpr,
    H: tl.constexpr,
    S: tl.constexpr,
):
    bc = tl.program_id(0)  # over B*C
    i = tl.program_id(1)   # row i in [0..S-1]
    h = tl.program_id(2)   # head h in [0..H-1]
    b = bc // Csz
    c = bc % Csz

    for j in range(0, S):
        if j <= i:
            sum_val = 0.0
            for k in range(0, j + 1):
                A_off = b * (Csz * H * S) + c * (H * S) + h * S + k
                a_val = tl.load(A_ptr + A_off)
                sum_val += a_val
            l_val = tl.exp(sum_val)
        else:
            l_val = 0.0
        L_off = bc * (S * S * H) + i * (S * H) + j * H + h
        tl.store(L_ptr + L_off, l_val)


# Kernel 2: Compute G[b, c, i, j, h] = sum over n of C_exp[b, c, i, n, j, h] * B_exp[b, c, j, n, i, h]
# B_exp and C_exp are [B, C, S, H, N] (float32). We loop j and n in-kernel.
@triton.jit
def compute_G_rows_kernel(
    B_exp_ptr,        # [B, C, S, H, N], float32
    C_exp_ptr,        # [B, C, S, H, N], float32
    G_ptr,            # [B, C, S, S, H], float32
    Bsz: tl.constexpr,
    Csz: tl.constexpr,
    H: tl.constexpr,
    S: tl.constexpr,
    N: tl.constexpr,  # state_size (128 typical)
):
    bc = tl.program_id(0)  # over B*C
    i = tl.program_id(1)   # row i in [0..S-1]
    h = tl.program_id(2)   # head h in [0..H-1]
    b = bc // Csz
    c = bc % Csz

    for j in range(0, S):
        acc = 0.0
        for n in range(0, N):
            # B_exp[b, c, j, h, n] offset = b*(Csz*S*H*N) + c*(S*H*N) + j*(H*N) + h*N + n
            B_off = b * (Csz * S * H * N) + c * (S * H * N) + j * (H * N) + h * N + n
            # C_exp[b, c, i, n, j, h] offset = b*(Csz*S*H*N) + c*(S*H*N) + i*(H*N) + j*(H*N) + n
            C_off = b * (Csz * S * H * N) + c * (S * H * N) + i * (H * N) + j * (H * N) + n
            B_val = tl.load(B_exp_ptr + B_off)
            C_val = tl.load(C_exp_ptr + C_off)
            acc += C_val * B_val
        G_off = b * (Csz * S * S * H) + c * (S * S * H) + i * (S * H) + j * H + h
        tl.store(G_ptr + G_off, acc)


# Kernel 3: Compute Y_diag[b, c, i, h, d] = sum_j G[b, c, i, j, h] * hidden_states[b, c, j, h, d]
# Grid over (B*C, S, H, head_dim). Each program handles one (b,c,i,h,d) and loops over j.
@triton.jit
def compute_Y_diag_rows_kernel(
    G_ptr,            # [B, C, S, S, H], float32
    hidden_ptr,       # [B, C, S, H, head_dim], float32
    Y_flat_ptr,       # [B*C*S*H*head_dim], float32
    Bsz: tl.constexpr,
    Csz: tl.constexpr,
    H: tl.constexpr,
    S: tl.constexpr,
    head_dim: tl.constexpr,
):
    bc = tl.program_id(0)  # over B*C
    i = tl.program_id(1)   # row i in [0..S-1]
    h = tl.program_id(2)   # head h in [0..H-1]
    d = tl.program_id(3)   # feature dim d in [0..head_dim-1]
    b = bc // Csz
    c = bc % Csz

    acc = 0.0
    for j in range(0, S):
        G_off = b * (Csz * S * S * H) + c * (S * S * H) + i * (S * H) + j * H + h
        g = tl.load(G_ptr + G_off)
        # hidden[b, c, j, h, d] offset = b*(Csz*S*H*head_dim) + c*(S*H*head_dim) + j*(H*head_dim) + h*head_dim + d
        hidden_off = b * (Csz * S * H * head_dim) + c * (S * H * head_dim) + j * (H * head_dim) + h * head_dim + d
        h_val = tl.load(hidden_ptr + hidden_off)
        acc += g * h_val

    # Write to flat buffer: [B*C*S*H*head_dim]
    Y_off = bc * (S * H * head_dim) + i * (H * head_dim) + h * head_dim + d
    tl.store(Y_flat_ptr + Y_off, acc)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.CHUNK_SIZE = 128
        self.NUM_HEADS = 32
        self.N_GROUPS = 8
        self.STATE_SIZE = 128  # N

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

        # Permute A_cumsum to [B, C, H, S] to match kernel indexing
        A = A_cumsum.permute(0, 2, 1, 3).contiguous()  # [B, C, H, S]
        hidden = hidden_states.contiguous()
        B_contig = B.contiguous()
        C_contig = C.contiguous()

        # Expand B and C along H by repeat_interleave(NUM_HEADS // N_GROUPS = 4) to match H=32
        B_exp = B_contig.repeat_interleave(self.NUM_HEADS // self.N_GROUPS, dim=3)  # [B, C, S, H, N]
        C_exp = C_contig.repeat_interleave(self.NUM_HEADS // self.N_GROUPS, dim=3)  # [B, C, S, H, N]

        # Allocate flat buffers (float32 for computation)
        total_L = Bsz * Csz * S * S * H
        L_flat = torch.empty(total_L, dtype=torch.float32, device=hidden.device)

        total_G = Bsz * Csz * S * S * H
        G_flat = torch.empty(total_G, dtype=torch.float32, device=hidden.device)

        total_Y = Bsz * Csz * S * H * head_dim
        Y_flat = torch.empty(total_Y, dtype=torch.float32, device=hidden.device)

        # Launch build_L_rows_kernel: grid over (B*C, S, H)
        grid_L = (Bsz * Csz, S, H)
        build_L_rows_kernel[grid_L](
            A, L_flat,
            Bsz=Bsz, Csz=Csz, H=H, S=S
        )

        # Launch compute_G_rows_kernel: grid over (B*C, S, H)
        grid_G = (Bsz * Csz, S, H)
        compute_G_rows_kernel[grid_G](
            B_exp, C_exp, G_flat,
            Bsz=Bsz, Csz=Csz, H=H, S=S, N=self.STATE_SIZE
        )

        # Launch compute_Y_diag_rows_kernel: grid over (B*C, S, H, head_dim)
        grid_Y = (Bsz * Csz, S, H, head_dim)
        compute_Y_diag_rows_kernel[grid_Y](
            G_flat, hidden, Y_flat,
            Bsz=Bsz, Csz=Csz, H=H, S=S, head_dim=head_dim
        )

        # Reshape Y_flat to [B, C, S, H, head_dim] and cast to bfloat16
        Y_diag = Y_flat.view(Bsz, Csz, S, H, head_dim)
        Y_diag = Y_diag.to(torch.bfloat16)
        return Y_diag


def run(*args):
    return ModelNew()(*args)
