import torch
import triton
import triton.language as tl


@triton.jit
def compute_L_kernel(
    A_cumsum_ptr,  # *float32, shape [B, H, C, S]
    L_ptr,         # *float32, shape [B, C, S, S, H]
    B: tl.constexpr, C: tl.constexpr, H: tl.constexpr, S: tl.constexpr,
):
    # grid = (B, C, H): each program handles one (b, c, h)
    b = tl.program_id(0)
    c = tl.program_id(1)
    h = tl.program_id(2)

    # Loop over i and j in [0, S)
    for i in range(0, S):
        for j in range(0, S):
            # Lower-triangular: i >= j
            lower = i >= j
            # Load A_cumsum[b, h, c, j]
            a_off = b * (H * C * S) + h * (C * S) + c * S + j
            A_val = tl.load(A_cumsum_ptr + a_off)
            L_val = tl.where(lower, tl.exp(A_val), 0.0)
            # Store to L[b, c, i, j, h]
            L_off = b * (C * S * H * S) + c * (S * S * H) + i * (S * H) + j * H + h
            tl.store(L_ptr + L_off, L_val)


@triton.jit
def compute_G_kernel(
    B_exp_ptr,     # *float32, shape [B, C, S, H, N], N=128
    C_exp_ptr,     # *float32, shape [B, C, S, H, N], N=128
    G_ptr,         # *float32, shape [B, C, S, S, H]
    B_size, C_size, H, S, N,  # integers
):
    # grid = (B*C*H, S, S): one program per (b,c,h) and per (i,j)
    pid0 = tl.program_id(0)
    i = tl.program_id(1)
    j = tl.program_id(2)

    # Recover (b, c, h) from pid0
    CH = C_size * H
    b = pid0 // CH
    rem = pid0 % CH
    c = rem // H
    h = rem % H

    # Accumulator for G[i, j, h]
    acc = 0.0
    # Sum over n in [0, N)
    for n in range(0, N):
        # B_exp[b, c, j, h, n] -> offset = b*(C*S*H*N) + c*(S*H*N) + j*(H*N) + h*N + n
        B_off = b * (C_size * S * H * N) + c * (S * H * N) + j * (H * N) + h * N + n
        B_val = tl.load(B_exp_ptr + B_off)
        # C_exp[b, c, i, h, n] -> offset = b*(C*S*H*N) + c*(S*H*N) + i*(H*N) + h*N + n
        C_off = b * (C_size * S * H * N) + c * (S * H * N) + i * (H * N) + h * N + n
        C_val = tl.load(C_exp_ptr + C_off)
        acc += C_val * B_val

    # Store G[b, c, i, j, h]
    G_off = b * (C_size * S * H * S) + c * (S * S * H) + i * (S * H) + j * H + h
    tl.store(G_ptr + G_off, acc)


@triton.jit
def compute_Y_diag_kernel(
    M_ptr,         # *float32, shape [B, C, S, S, H]
    hidden_ptr,    # *float32, shape [B, C, S, H, head_dim]
    Y_ptr,         # *float32, shape [B, C, S, H, head_dim]
    B_size, C_size, H, S, head_dim,
):
    # grid = (B*C*H, S): one program per (b,c,h) and per i
    pid0 = tl.program_id(0)
    i = tl.program_id(1)

    # Recover (b, c, h) from pid0
    CH = C_size * H
    b = pid0 // CH
    rem = pid0 % CH
    c = rem // H
    h = rem % H

    # Accumulator per d
    for d in range(0, head_dim):
        acc = 0.0
        # Sum over j in [0, S)
        for j in range(0, S):
            # Load M[b, c, i, j, h]
            M_off = b * (C_size * S * H * S) + c * (S * S * H) + i * (S * H) + j * H + h
            M_val = tl.load(M_ptr + M_off)
            # Load hidden_states[b, c, j, h, d]
            # hidden shape [B, C, S, H, head_dim] => offset = b*(C*S*H*head_dim) + c*(S*H*head_dim) + j*(H*head_dim) + h*head_dim + d
            hidden_off = b * (C_size * S * H * head_dim) + c * (S * H * head_dim) + j * (H * head_dim) + h * head_dim + d
            hidden_val = tl.load(hidden_ptr + hidden_off)
            acc += M_val * hidden_val

        # Store Y[b, c, i, h, d]
        Y_off = b * (C_size * S * H * head_dim) + c * (S * H * head_dim) + i * (H * head_dim) + h * head_dim + d
        tl.store(Y_ptr + Y_off, acc)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, A_cumsum: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        # Ensure tensors are on CUDA and contiguous
        assert hidden_states.is_cuda and A_cumsum.is_cuda and B.is_cuda and C.is_cuda, "Tensors must be on CUDA"
        hidden_states = hidden_states.contiguous()
        A_cumsum = A_cumsum.contiguous()
        B = B.contiguous()
        C = C.contiguous()

        # Cast to float32 for compute
        hidden_states = hidden_states.to(torch.float32)
        A_cumsum = A_cumsum.to(torch.float32)
        B = B.to(torch.float32)
        C = C.to(torch.float32)

        # Dimensions
        B_size, C_size, S, H, head_dim = hidden_states.shape
        # This implementation assumes S=128 and H=32 (matches given workloads)
        assert S == 128 and H == 32, "This Triton implementation assumes S=128 and H=32"
        N = B.size(-1)  # state_size, typically 128
        N_GROUPS = 8
        assert B.size(2) == S and B.size(3) == N_GROUPS and C.size(2) == S and C.size(3) == N_GROUPS, "B/C shapes must be [B, C, S, N_GROUPS, N]"
        assert A_cumsum.shape == (B_size, H, C_size, S), "A_cumsum must be [B, H, C, S]"

        # Allocate outputs
        L = torch.empty((B_size, C_size, S, S, H), device=hidden_states.device, dtype=torch.float32)
        G = torch.empty((B_size, C_size, S, S, H), device=hidden_states.device, dtype=torch.float32)

        # Build L in Triton
        grid_L = (B_size, C_size, H)
        compute_L_kernel[grid_L](
            A_cumsum, L,
            B=B_size, C=C_size, H=H, S=S,
            num_warps=4, num_stages=2,
        )

        # Expand B and C along H by repeat_interleave(NUM_HEADS // N_GROUPS = 4)
        B_exp = B.repeat_interleave(4, dim=3)  # [B, C, S, H, N]
        C_exp = C.repeat_interleave(4, dim=3)  # [B, C, S, H, N]

        # Compute G in Triton
        grid_G = (B_size * C_size * H, S, S)
        compute_G_kernel[grid_G](
            B_exp, C_exp, G,
            B_size, C_size, H, S, N,
            num_warps=4, num_stages=2,
        )

        # Compute M = G * L
        M = G * L

        # Compute Y_diag in Triton
        Y = torch.empty((B_size, C_size, S, H, head_dim), device=hidden_states.device, dtype=torch.float32)
        grid_Y = (B_size * C_size * H, S)
        compute_Y_diag_kernel[grid_Y](
            M, hidden_states, Y,
            B_size, C_size, H, S, head_dim,
            num_warps=4, num_stages=2,
        )

        # Return in bfloat16 to match original
        return Y.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
