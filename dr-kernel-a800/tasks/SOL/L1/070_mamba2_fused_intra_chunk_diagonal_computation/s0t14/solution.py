import torch
import triton
import triton.language as tl


@triton.jit
def compute_L_kernel(A_ptr, L_ptr,
                     B_SIZE, C_SIZE, H, S):
    # Each program handles one (b, c, h)
    b = tl.program_id(0)
    c = tl.program_id(1)
    h = tl.program_id(2)

    # A_ptr layout: [B, H, C, S] -> flattened
    # L_ptr layout: [B, C, S, S, H] -> flattened
    base_A = b * (H * C_SIZE * S) + h * (C_SIZE * S)
    base_L = b * (C_SIZE * S * S * H) + c * (S * S * H) + h * (S * S)

    # Loop over j
    for j in range(0, S):
        val = tl.load(A_ptr + base_A + c * S + j)
        # lower-triangular mask: only write when i >= j
        # For each i >= j, store exp(val) at L[i, j, h]
        # Unrolled simple loop for i
        for i in range(j, S):
            tl.store(L_ptr + base_L + i * (S * H) + j * H, tl.exp(val))


@triton.jit
def compute_G_kernel(B_exp_ptr, C_exp_ptr, G_ptr,
                     B_SIZE, C_SIZE, H, S, N):
    # Each program handles one (b, c, h)
    b = tl.program_id(0)
    c = tl.program_id(1)
    h = tl.program_id(2)

    # B_exp: [B, C, S, H, N] flattened
    # C_exp: [B, C, S, H, N] flattened
    # G: [B, C, S, S, H] flattened
    base_B = b * (C_SIZE * S * H * N) + c * (S * H * N)
    base_C = b * (C_SIZE * S * H * N) + c * (S * H * N)
    base_G = b * (C_SIZE * S * S * H) + c * (S * S * H) + h * (S * S)

    # Loop over i and j
    for i in range(0, S):
        for j in range(0, S):
            # sum over n: C_exp[b, c, i, n, j, h] * B_exp[b, c, j, n, i, h]
            acc = 0.0
            for n in range(0, N):
                # B_exp offset: ((i*S + h)*N + n)*H + j
                off_B = base_B + i * (H * N) + h * N + n * H + j
                # C_exp offset: ((j*S + h)*N + n)*i
                off_C = base_C + j * (H * N) + h * N + n * H + i
                b_val = tl.load(B_exp_ptr + off_B)
                c_val = tl.load(C_exp_ptr + off_C)
                acc += c_val * b_val
            tl.store(G_ptr + base_G + i * (S * H) + j * H, acc)


@triton.jit
def compute_Y_diag_kernel(M_ptr, hidden_ptr, Y_ptr,
                          B_SIZE, C_SIZE, H, S, HEAD_DIM):
    # Each program handles one (b, c, h, d)
    b = tl.program_id(0)
    c = tl.program_id(1)
    h = tl.program_id(2)
    d = tl.program_id(3)

    # M: [B, C, S, S, H] flattened
    # hidden: [B, C, S, H, HEAD_DIM] flattened
    # Y: [B, C, S, H, HEAD_DIM] flattened
    base_M = b * (C_SIZE * S * S * H) + c * (S * S * H) + h * (S * S)
    # hidden and Y are laid out with head_dim as the last dim. We'll compute offsets accordingly.

    # We need to iterate over i and sum over j: Y[b, c, i, h, d] += sum_j M[b, c, i, j, h] * hidden[b, c, j, h, d]
    for i in range(0, S):
        acc = 0.0
        for j in range(0, S):
            M_off = base_M + i * (S * H) + j * H
            M_val = tl.load(M_ptr + M_off)
            # hidden offset: ((j*S + h)*HEAD_DIM + d)
            hidden_off = b * (C_SIZE * S * H * HEAD_DIM) + c * (S * H * HEAD_DIM) + j * (H * HEAD_DIM) + h * HEAD_DIM + d
            hidden_val = tl.load(hidden_ptr + hidden_off)
            acc += M_val * hidden_val
        # Store Y[b, c, i, h, d]
        Y_off = b * (C_SIZE * S * H * HEAD_DIM) + c * (S * H * HEAD_DIM) + i * (H * HEAD_DIM) + h * HEAD_DIM + d
        tl.store(Y_ptr + Y_off, acc)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, A_cumsum: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        # Ensure CUDA device
        assert hidden_states.is_cuda and A_cumsum.is_cuda and B.is_cuda and C.is_cuda, "Tensors must be on CUDA device"

        # Cast to float32 for computation
        hidden_states = hidden_states.to(torch.float32)
        A_cumsum = A_cumsum.to(torch.float32)
        B = B.to(torch.float32)
        C = C.to(torch.float32)

        # Dimensions
        B_size, C_size, S, H, head_dim = hidden_states.shape
        # This implementation assumes S=128 and H=32 as per the reference
        assert S == 128 and H == 32, "This implementation assumes S=128 and H=32"
        N = B.size(-1)  # state_size, typically 128
        N_GROUPS = 8
        assert B.size(2) == S and B.size(3) == N_GROUPS and C.size(2) == S and C.size(3) == N_GROUPS, "B/C shapes must be [B, C, S, N_GROUPS, N]"
        assert A_cumsum.shape == (B_size, H, C_size, S), "A_cumsum must be [B, H, C, S]"

        # Build L in Triton
        L = torch.empty((B_size, C_size, S, S, H), device=hidden_states.device, dtype=torch.float32)
        grid_L = (B_size, C_size, H)
        compute_L_kernel[grid_L](
            A_cumsum, L,
            B_SIZE=B_size, C_SIZE=C_size, H=H, S=S,
            num_warps=4, num_stages=2,
        )

        # Expand B and C along H by repeat_interleave(NUM_HEADS // N_GROUPS = 4)
        B_exp = B.repeat_interleave(4, dim=3)  # [B, C, S, H, N]
        C_exp = C.repeat_interleave(4, dim=3)  # [B, C, S, H, N]

        # Compute G in Triton
        G = torch.empty((B_size, C_size, S, S, H), device=hidden_states.device, dtype=torch.float32)
        grid_G = (B_size, C_size, H)
        compute_G_kernel[grid_G](
            B_exp, C_exp, G,
            B_SIZE=B_size, C_SIZE=C_size, H=H, S=S, N=N,
            num_warps=4, num_stages=2,
        )

        # Multiply M = G * L
        M = G * L

        # Compute Y_diag in Triton
        Y = torch.empty((B_size, C_size, S, H, head_dim), device=hidden_states.device, dtype=torch.float32)
        grid_Y = (B_size, C_size, H, head_dim)
        compute_Y_diag_kernel[grid_Y](
            M, hidden_states, Y,
            B_SIZE=B_size, C_SIZE=C_size, H=H, S=S, HEAD_DIM=head_dim,
            num_warps=2, num_stages=2,
        )

        # Return in bfloat16 to match original model output dtype
        return Y.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
