import torch
import triton
import triton.language as tl


@triton.jit
def build_L_kernel(
    A_ptr,              # *float32, shape [B, H, C, S] (note: indexing uses pid0, pid2, pid3)
    L_ptr,              # *float32, shape [B, C, S, S, H]
    B: tl.constexpr,    # int
    C: tl.constexpr,    # int
    H: tl.constexpr,    # int
    S: tl.constexpr,    # int (assumed 128)
):
    # program ids: b over dim0, c over dim1, h over dim2
    b = tl.program_id(0)
    c = tl.program_id(1)
    h = tl.program_id(2)

    # vector of indices j
    j = tl.arange(0, S)  # shape (S,)
    j_vec = j  # for offsets

    # For each i, compute L[i, j] = exp(A[b, h, c, j]) if i >= j else 0
    for i in range(0, S):
        i_val = i
        # mask for lower-triangular: i_val >= j
        mask = (i_val >= j_vec)  # shape (S,)
        # Load A[b, h, c, j]
        a_offsets = b * H * C * S + h * C * S + c * S + j_vec  # vector of offsets
        a_vals = tl.load(A_ptr + a_offsets)  # vector of length S
        l_vals = tl.exp(a_vals)  # exp of cumulative sums
        # Apply mask: for j > i, set to 0
        l_vals = tl.where(mask, l_vals, 0.0)

        # Store L[b, c, i, j, h]
        # L layout: [B, C, S, S, H] => offset = b*(C*S*S*H) + c*(S*S*H) + i*(S*H) + j*H + h
        for jj in range(0, S):
            j_val = jj
            L_offset = b * (C * S * S * H) + c * (S * S * H) + i_val * (S * H) + j_val * H + h
            tl.store(L_ptr + L_offset, l_vals[jj])


@triton.jit
def compute_G_kernel(
    C_exp_ptr,          # *float32, shape [B, C, S, H, N]
    B_exp_ptr,          # *float32, shape [B, C, S, H, N]
    G_ptr,              # *float32, shape [B, C, S, S, H]
    B: tl.constexpr,    # int
    C: tl.constexpr,    # int
    H: tl.constexpr,    # int
    S: tl.constexpr,    # int (assumed 128)
    N: tl.constexpr,    # int (assumed 128)
):
    b = tl.program_id(0)
    c = tl.program_id(1)
    h = tl.program_id(2)

    # We compute G[i, j, h] = sum_n C_exp[b, c, i, n, j, h] * B_exp[b, c, j, n, i, h]
    for i in range(0, S):
        i_val = i
        for j in range(0, S):
            j_val = j
            acc = 0.0
            # sum over n
            for n in range(0, N):
                # C_exp offset: [B, C, S, H, N] => b*(C*S*H*N) + c*(S*H*N) + i*H*N + h*N + n
                C_exp_offset = b * (C * S * H * N) + c * (S * H * N) + i_val * (H * N) + h * N + n
                # B_exp offset: [B, C, S, H, N] => b*(C*S*H*N) + c*(S*H*N) + j_val*(H*N) + h*N + n
                B_exp_offset = b * (C * S * H * N) + c * (S * H * N) + j_val * (H * N) + h * N + n
                c_val = tl.load(C_exp_ptr + C_exp_offset)
                b_val = tl.load(B_exp_ptr + B_exp_offset)
                acc += c_val * b_val
            # Store G[b, c, i, j, h] = acc
            G_offset = b * (C * S * S * H) + c * (S * S * H) + i_val * (S * H) + j_val * H + h
            tl.store(G_ptr + G_offset, acc)


@triton.jit
def compute_Y_diag_kernel(
    M_ptr,              # *float32, shape [B, C, S, S, H]
    hidden_ptr,         # *float32, shape [B, C, S, H, head_dim]
    Y_ptr,              # *float32, shape [B, C, S, H, head_dim]
    B: tl.constexpr,    # int
    C: tl.constexpr,    # int
    H: tl.constexpr,    # int
    S: tl.constexpr,    # int (assumed 128)
    head_dim: tl.constexpr,  # dynamic but constexpr for Triton
):
    b = tl.program_id(0)
    c = tl.program_id(1)
    h = tl.program_id(2)

    # Compute Y_diag[i, h, d] = sum_j M[b, c, i, j, h] * hidden[b, c, j, h, d]
    for i in range(0, S):
        acc = 0.0
        # vector of j
        j = tl.arange(0, S)
        for jj in range(0, S):
            j_val = jj
            # Load M[i, j, h]
            M_offset = b * (C * S * S * H) + c * (S * S * H) + i * (S * H) + j_val * H + h
            M_val = tl.load(M_ptr + M_offset)
            # Load hidden[b, c, j, h, d] for d in 0..head_dim-1
            for d in range(0, head_dim):
                hidden_offset = b * (C * S * H * head_dim) + c * (S * H * head_dim) + j_val * (H * head_dim) + h * head_dim + d
                hidden_val = tl.load(hidden_ptr + hidden_offset)
                acc += M_val * hidden_val
        # Store Y[b, c, i, h, d] for all d
        for d in range(0, head_dim):
            Y_offset = b * (C * S * H * head_dim) + c * (S * H * head_dim) + i * (H * head_dim) + h * head_dim + d
            tl.store(Y_ptr + Y_offset, acc)


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
        build_L_kernel[grid_L](
            A_cumsum, L,
            B=B_size, C=C_size, H=H, S=S,
            num_warps=4, num_stages=2,
        )

        # Expand B and C along H by repeat_interleave(NUM_HEADS // N_GROUPS = 4)
        B_exp = B.repeat_interleave(4, dim=3)  # [B, C, S, H, N]
        C_exp = C.repeat_interleave(4, dim=3)  # [B, C, S, H, N


def run(*args):
    return ModelNew()(*args)
