import torch
import triton
import triton.language as tl


@triton.jit
def mask_cumsum_exp_kernel(
    A_cumsum_ptr, L_ptr,
    Bsz, N, Csz, S,
    A_stride_b, A_stride_n, A_stride_c, A_stride_s,
    L_stride_b, L_stride_n, L_stride_c, L_stride_i, L_stride_j,
    num_warps: tl.constexpr, num_stages: tl.constexpr
):
    # Grid: (Bsz, N, Csz, ceil(S / BLOCK_I))
    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_c = tl.program_id(2)
    pid_i_block = tl.program_id(3)

    # Destination index vector
    start_i = pid_i_block * 64
    i_idx = start_i + tl.arange(0, 64)
    i_mask = i_idx < S

    # Running sum per i
    running = tl.zeros([64], dtype=tl.float32)

    # Loop over source j
    for j in range(0, S):
        # Include only when j < i (diagonal=-1)
        include = j < i_idx
        load_mask = i_mask & include
        a_ptrs = A_cumsum_ptr + pid_b * A_stride_b + pid_n * A_stride_n + pid_c * A_stride_c + j * A_stride_s
        vals = tl.load(a_ptrs, mask=load_mask, other=0.0)
        running += vals
        # exp of masked cumsum (excluding diagonal)
        exp_vals = tl.exp(running)
        L_ptrs = L_ptr + pid_b * L_stride_b + pid_n * L_stride_n + pid_c * L_stride_c + i_idx * L_stride_i + j * L_stride_j
        tl.store(L_ptrs, exp_vals, mask=i_mask & include)


@triton.jit
def contract_BC_to_G_kernel(
    B_expanded_ptr, C_expanded_ptr, G_ptr,
    Bsz, Csz, S, N, K,
    B_stride_b, B_stride_c, B_stride_j, B_stride_n, B_stride_k,
    C_stride_b, C_stride_c, C_stride_i, C_stride_n, C_stride_k,
    G_stride_b, G_stride_c, G_stride_i, G_stride_j, G_stride_n,
    num_warps: tl.constexpr, num_stages: tl.constexpr
):
    # Grid: (Bsz, Csz, S, S, N)
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_i = tl.program_id(2)  # destination i
    pid_j = tl.program_id(3)  # source j
    pid_n = tl.program_id(4)

    acc = tl.zeros((), dtype=tl.float32)
    # Reduce over state dimension K
    for k in range(0, K):
        B_val = tl.load(
            B_expanded_ptr + pid_b * B_stride_b + pid_c * B_stride_c + pid_j * B_stride_j + pid_n * B_stride_n + k * B_stride_k
        )
        C_val = tl.load(
            C_expanded_ptr + pid_b * C_stride_b + pid_c * C_stride_c + pid_i * C_stride_i + pid_n * C_stride_n + k * C_stride_k
        )
        acc += C_val * B_val
    # Store G[b, c, i, j, n]
    G_ptr_out = G_ptr + pid_b * G_stride_b + pid_c * G_stride_c + pid_i * G_stride_i + pid_j * G_stride_j + pid_n * G_stride_n
    tl.store(G_ptr_out, acc)


@triton.jit
def diag_contract_Y_kernel(
    M_ptr, hidden_ptr, Y_ptr,
    Bsz, Csz, S, N, D,
    M_stride_b, M_stride_c, M_stride_i, M_stride_j, M_stride_n,
    HS_stride_b, HS_stride_c, HS_stride_j, HS_stride_n, HS_stride_d,
    Y_stride_b, Y_stride_c, Y_stride_i, Y_stride_n, Y_stride_d,
    num_warps: tl.constexpr, num_stages: tl.constexpr
):
    # Grid: (Bsz, Csz, S, N, D)
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_i = tl.program_id(2)
    pid_n = tl.program_id(3)
    pid_d = tl.program_id(4)

    acc = tl.zeros((), dtype=tl.float32)
    # Loop over j (source positions)
    for j in range(0, S):
        M_val = tl.load(
            M_ptr + pid_b * M_stride_b + pid_c * M_stride_c + pid_i * M_stride_i + j * M_stride_j + pid_n * M_stride_n
        )
        HS_val = tl.load(
            hidden_ptr + pid_b * HS_stride_b + pid_c * HS_stride_c + j * HS_stride_j + pid_n * HS_stride_n + pid_d * HS_stride_d
        )
        acc += M_val * HS_val

    # Store Y[b, c, i, n, d]
    Y_ptr_out = Y_ptr + pid_b * Y_stride_b + pid_c * Y_stride_c + pid_i * Y_stride_i + pid_n * Y_stride_n + pid_d * Y_stride_d
    tl.store(Y_ptr_out, acc)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, A_cumsum: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        # Ensure device consistency
        device = hidden_states.device

        # Shapes
        Bsz, Csz, S, N, D = hidden_states.shape

        # Cast A_cumsum to float32 and make contiguous
        A = A_cumsum.to(torch.float32).contiguous()

        # Expand B and C to num_heads=32 from n_groups=8 (repeat_interleave by 4)
        B_expanded = B.to(torch.float32).repeat_interleave(N // 8, dim=3).contiguous()
        C_expanded = C.to(torch.float32).repeat_interleave(N // 8, dim=3).contiguous()

        # 1) Compute L via Triton: L[b, n, c, i, j] = exp(masked cumsum along j, exclude j==i)
        L = torch.empty((Bsz, N, Csz, S, S), device=device, dtype=torch.float32)

        A_stride_b, A_stride_n, A_stride_c, A_stride_s = A.stride()
        L_stride_b, L_stride_n, L_stride_c, L_stride_i, L_stride_j = L.stride()

        grid_mask = (Bsz, N, Csz, triton.cdiv(S, 64))
        mask_cumsum_exp_kernel[grid_mask](
            A, L,
            Bsz, N, Csz, S,
            A_stride_b, A_stride_n, A_stride_c, A_stride_s,
            L_stride_b, L_stride_n, L_stride_c, L_stride_i, L_stride_j,
            num_warps=4, num_stages=2
        )

        # 2) Compute G via Triton contraction: G[b, c, i, j, n] = sum_k C_expanded[b,c,i,n,k] * B_expanded[b,c,j,n,k]
        K = D  # state_size equals head_dim
        G = torch.empty((Bsz, Csz, S, S, N), device=device, dtype=torch.float32)

        B_exp_stride_b, B_exp_stride_c, B_exp_stride_j, B_exp_stride_n, B_exp_stride_k = B_expanded.stride()
        C_exp_stride_b, C_exp_stride_c, C_exp_stride_i, C_exp_stride_n, C_exp_stride_k = C_expanded.stride()
        G_stride_b, G_stride_c, G_stride_i, G_stride_j, G_stride_n = G.stride()

        grid_G = (Bsz, Csz, S, S, N)
        contract_BC_to_G_kernel[grid_G](
            B_expanded, C_expanded, G,
            Bsz, Csz, S, N, K,
            B_exp_stride_b, B_exp_stride_c, B_exp_stride_j, B_exp_stride_n, B_exp_stride_k,
            C_exp_stride_b, C_exp_stride_c, C_exp_stride_i, C_exp_stride_n, C_exp_stride_k,
            G_stride_b, G_stride_c, G_stride_i, G_stride_j, G_stride_n,
            num_warps=4, num_stages=2
        )

        # 3) Elementwise multiply M = G * L (PyTorch)
        M = G * L  # float32

        # 4) Diagonal contraction to compute Y_diag: Y[b, c, i, n, d] = sum_j M[b, c, i, j, n] * hidden_states[b, c, j, n, d]
        Y = torch.empty((Bsz, Csz, S, N, D), device=device, dtype=torch.float32)

        # hidden_states is already in the input dtype; cast to float32 for computation
        hidden_states_f32 = hidden_states.to(torch.float32)

        M_stride_b, M_stride_c, M_stride_i, M_stride_j, M_stride_n = M.stride()
        HS_stride_b, HS_stride_c, HS_stride_j, HS_stride_n, HS_stride_d = hidden_states_f32.stride()
        Y_stride_b, Y_stride_c, Y_stride_i, Y_stride_n, Y_stride_d = Y.stride()

        grid_Y = (Bsz, Csz, S, N, D)
        diag_contract_Y_kernel[grid_Y](
            M, hidden_states_f32, Y,
            Bsz, Csz, S, N, D,
            M_stride_b, M_stride_c, M_stride_i, M_stride_j, M_stride_n,
            HS_stride_b, HS_stride_c, HS_stride_j, HS_stride_n, HS_stride_d,
            Y_stride_b, Y_stride_c, Y_stride_i, Y_stride_n, Y_stride_d,
            num_warps=4, num_stages=2
        )

        # Cast to bfloat16 to match original signature
        return Y.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
