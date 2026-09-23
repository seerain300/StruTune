import torch
import triton
import triton.language as tl


@triton.jit
def masked_cumsum_lower_exp(
    A_ptr, L_ptr,
    B_size, C_size, S, N,
    A_stride_b, A_stride_c, A_stride_i, A_stride_j, A_stride_n,
    L_stride_b, L_stride_c, L_stride_i, L_stride_j, L_stride_n,
    num_warps: tl.constexpr, num_stages: tl.constexpr
):
    # Grid over (b, c, i)
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)

    # NUM_HEADS is 32 in original signature
    for n in range(0, 32):
        running = 0.0
        # CHUNK_SIZE is 128 in original; we loop j from 0 to 127
        for j in range(0, 128):
            include = j < i  # diagonal=-1: include j < i
            # Load A[b, c, i, j, n]
            a = tl.load(
                A_ptr + b * A_stride_b + c * A_stride_c + i * A_stride_i + j * A_stride_j + n * A_stride_n,
                mask=True, other=0.0
            )
            # Apply mask
            a = tl.where(include, a, 0.0)
            running += a
            out = tl.exp(running)
            # Store L[b, c, i, j, n]
            tl.store(
                L_ptr + b * L_stride_b + c * L_stride_c + i * L_stride_i + j * L_stride_j + n * L_stride_n,
                out
            )


@triton.jit
def contract_BC_to_G(
    B_ptr, C_ptr, G_ptr,
    B_size, C_size, S, N, K, D,
    B_stride_b, B_stride_c, B_stride_j, B_stride_n, B_stride_k,
    C_stride_b, C_stride_c, C_stride_i, C_stride_n, C_stride_k,
    G_stride_b, G_stride_c, G_stride_i, G_stride_j, G_stride_n,
    num_warps: tl.constexpr, num_stages: tl.constexpr
):
    # Grid over (b, c, i, j, n)
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)
    j = tl.program_id(3)
    n = tl.program_id(4)

    acc = 0.0
    # Loop over state_size K (compile-time constant)
    for k in range(0, K):
        bval = tl.load(
            B_ptr + b * B_stride_b + c * B_stride_c + j * B_stride_j + n * B_stride_n + k * B_stride_k,
            mask=True, other=0.0
        )
        cval = tl.load(
            C_ptr + b * C_stride_b + c * C_stride_c + i * C_stride_i + n * C_stride_n + k * C_stride_k,
            mask=True, other=0.0
        )
        acc += bval * cval

    # Store G[b, c, i, j, n] as float32
    tl.store(
        G_ptr + b * G_stride_b + c * G_stride_c + i * G_stride_i + j * G_stride_j + n * G_stride_n,
        acc
    )


@triton.jit
def diag_contract_Y(
    M_ptr, HS_ptr, Y_ptr,
    B_size, C_size, S, N, D,
    M_stride_b, M_stride_c, M_stride_i, M_stride_j, M_stride_n,
    HS_stride_b, HS_stride_c, HS_stride_j, HS_stride_n, HS_stride_d,
    Y_stride_b, Y_stride_c, Y_stride_i, Y_stride_n, Y_stride_d,
    num_warps: tl.constexpr, num_stages: tl.constexpr
):
    # Grid over (b, c, i, n, d) and loop over j
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)
    n = tl.program_id(3)
    d = tl.program_id(4)

    acc = 0.0
    for j in range(0, S):
        mval = tl.load(
            M_ptr + b * M_stride_b + c * M_stride_c + i * M_stride_i + j * M_stride_j + n * M_stride_n,
            mask=True, other=0.0
        )
        hsv = tl.load(
            HS_ptr + b * HS_stride_b + c * HS_stride_c + j * HS_stride_j + n * HS_stride_n + d * HS_stride_d,
            mask=True, other=0.0
        )
        acc += mval * hsv

    # Store Y[b, c, i, n, d]
    tl.store(Y_ptr + b * Y_stride_b + c * Y_stride_c + i * Y_stride_i + n * Y_stride_n + d * Y_stride_d, acc)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, A_cumsum: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        # Shapes (original code uses these):
        # hidden_states: [batch, num_chunks, chunk_size, num_heads, head_dim]
        # A_cumsum: [batch, num_heads, num_chunks, chunk_size]
        # B: [batch, num_chunks, chunk_size, n_groups, state_size]
        # C: [batch, num_chunks, chunk_size, n_groups, state_size]
        batch_size, num_chunks, chunk_size, num_heads, head_dim = hidden_states.shape
        n_groups = 8
        num_heads_const = 32  # from original signature
        K = hidden_states.shape[4]  # state_size (original uses head_dim, but is K)

        # Prepare expanded B and C to num_heads=32 via repeat_interleave(NUM_HEADS//N_GROUPS)=4
        B_expanded = B.repeat_interleave(4, dim=3)  # [B, C, S, N, K]
        C_expanded = C.repeat_interleave(4, dim=3)  # [B, C, S, N, K]

        # Ensure float32 for numeric stability
        A_cumsum_f32 = A_cumsum.to(torch.float32)
        B_expanded_f32 = B_expanded.to(torch.float32)
        C_expanded_f32 = C_expanded.to(torch.float32)
        hidden_states_f32 = hidden_states.to(torch.float32)

        # Allocate L: [B, C, S, S, N], float32
        L = torch.empty((batch_size, num_chunks, chunk_size, chunk_size, num_heads_const), device=hidden_states.device, dtype=torch.float32)

        # Strides for A_cumsum -> L
        A_stride_b, A_stride_c, A_stride_i, A_stride_j, A_stride_n = A_cumsum_f32.stride()
        L_stride_b, L_stride_c, L_stride_i, L_stride_j, L_stride_n = L.stride()

        # Launch masked_cumsum_lower_exp
        grid_L = (batch_size, num_chunks, chunk_size)
        masked_cumsum_lower_exp[grid_L](
            A_cumsum_f32, L,
            batch_size, num_chunks, chunk_size, num_heads_const,
            A_stride_b, A_stride_c, A_stride_i, A_stride_j, A_stride_n,
            L_stride_b, L_stride_c, L_stride_i, L_stride_j, L_stride_n,
            num_warps=1, num_stages=1
        )

        # Allocate G: [B, C, S, S, N], float32
        G = torch.empty((batch_size, num_chunks, chunk_size, chunk_size, num_heads_const), device=hidden_states.device, dtype=torch.float32)

        # Strides for B_expanded and C_expanded -> G
        B_stride_b, B_stride_c, B_stride_j, B_stride_n, B_stride_k = B_expanded_f32.stride()
        C_stride_b, C_stride_c, C_stride_i, C_stride_n, C_stride_k = C_expanded_f32.stride()
        G_stride_b, G_stride_c, G_stride_i, G_stride_j, G_stride_n = G.stride()

        grid_G = (batch_size, num_chunks, chunk_size, chunk_size, num_heads_const)
        contract_BC_to_G[grid_G](
            B_expanded_f32, C_expanded_f32, G,
            batch_size, num_chunks, chunk_size, num_heads_const, K, head_dim,
            B_stride_b, B_stride_c, B_stride_j, B_stride_n, B_stride_k,
            C_stride_b, C_stride_c, C_stride_i, C_stride_n, C_stride_k,
            G_stride_b, G_stride_c, G_stride_i, G_stride_j, G_stride_n,
            num_warps=1, num_stages=1
        )

        # Elementwise M = G * L
        M = G * L  # float32

        # Allocate Y: [B, C, S, N, D]
        Y = torch.empty((batch_size, num_chunks, chunk_size, num_heads_const, head_dim), device=hidden_states.device, dtype=torch.float32)

        # Strides for M, hidden_states_f32, and Y
        M_stride_b, M_stride_c, M_stride_i, M_stride_j, M_stride_n = M.stride()
        HS_stride_b, HS_stride_c, HS_stride_j, HS_stride_n, HS_stride_d = hidden_states_f32.stride()
        Y_stride_b, Y_stride_c, Y_stride_i, Y_stride_n, Y_stride_d = Y.stride()

        grid_Y = (batch_size, num_chunks, chunk_size, num_heads_const, head_dim)
        diag_contract_Y[grid_Y](
            M, hidden_states_f32, Y,
            batch_size, num_chunks, chunk_size, num_heads_const, head_dim,
            M_stride_b, M_stride_c, M_stride_i, M_stride_j, M_stride_n,
            HS_stride_b, HS_stride_c, HS_stride_j, HS_stride_n, HS_stride_d,
            Y_stride_b, Y_stride_c, Y_stride_i, Y_stride_n, Y_stride_d,
            num_warps=1, num_stages=1
        )

        # Cast to bfloat16 to match original signature
        return Y.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
