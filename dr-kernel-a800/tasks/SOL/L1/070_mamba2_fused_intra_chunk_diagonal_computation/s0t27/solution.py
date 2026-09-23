import torch
import triton
import triton.language as tl

# Kernel 1: Build L with lower-triangular causal mask using exp(cumsum(A))
# L shape: [B, C, S, S, H]; L[i, j, h] = exp(sum_{t=0..j} A[b,h,c,t]) if i >= j else 0
@triton.jit
def build_L_kernel(
    A_ptr,               # *float32, A_cumsum: [B, H, C, S]
    L_ptr,               # *float32, L: [B, C, S, S, H]
    Bsz, Csz, S, H,      # ints
    a_stride_b, a_stride_h, a_stride_c, a_stride_s,  # strides for A
    l_stride_b, l_stride_c, l_stride_i, l_stride_j, l_stride_h,  # strides for L
    CHUNK_SIZE: tl.constexpr
):
    # grid = (B, C)
    b = tl.program_id(0)
    c = tl.program_id(1)

    # Loop over target j, source i, and head h
    for j in range(CHUNK_SIZE):
        # compute cumsum up to j for A[b, :, c, :]
        cumsum = 0.0
        for t in range(CHUNK_SIZE):
            # mask t <= j to accumulate
            include = t <= j
            # A layout is [B, H, C, S]; we index with strides
            # Note: A_ptr + b*a_stride_b + t*a_stride_s + c*a_stride_c + h*a_stride_h
            # We need h index for A, but we are summing across h for fixed (b, c, t)
            # Since we want sum over h: A[b, h, c, t], we loop over h using stride
            # However, we don't have H stride here; instead, we load scalar per h in outer loop below.
            # To avoid this, we compute cumsum for each h by iterating h:
            # Re-compute cumsum per h outside, then set L per h. But Triton kernel prefers structured loops.
            # So we restructure: we compute cumsum once by loading A[b, 0, c, t], ..., A[b, H-1, c, t] and summing.
            # Here, A is [B, H, C, S]; load per h using A_ptr + b*a_stride_b + h*a_stride_h + c*a_stride_c + t*a_stride_s
            # We'll create a per-h cumsum vector.
            cumsum_vec = tl.zeros([H], dtype=tl.float32)
            for h_index in range(H):
                val = tl.load(A_ptr + b * a_stride_b + h_index * a_stride_h + c * a_stride_c + t * a_stride_s, mask=include, other=0.0)
                cumsum_vec += val
            # Now for each i >= j, set L[i, j, h] = exp(cumsum_vec[h])
            for i in range(CHUNK_SIZE):
                if i >= j:
                    for h in range(H):
                        ptr = L_ptr + b * l_stride_b + c * l_stride_c + i * l_stride_i + j * l_stride_j + h * l_stride_h
                        tl.store(ptr, tl.exp(cumsum_vec[h]))

# Kernel 2: Compute G[i, j, h] = sum_n C_exp[b, c, i, n, j, h] * B_exp[b, c, j, n, i, h]
# Shapes: B_exp, C_exp: [B, C, S, H, N]; G: [B, C, S, S, H]
@triton.jit
def compute_G_kernel_simple(
    B_exp_ptr, C_exp_ptr, G_ptr,
    Bsz, Csz, S, H, N,
    B_exp_stride0, B_exp_stride1, B_exp_stride2, B_exp_stride3, B_exp_stride4,
    C_exp_stride0, C_exp_stride1, C_exp_stride2, C_exp_stride3, C_exp_stride4,
    G_stride_b, G_stride_c, G_stride_i, G_stride_j, G_stride_h,
    total_elems: tl.constexpr
):
    pid = tl.program_id(0)
    # We iterate over all (b, c, i, j, h) sequentially to keep it simple and correct
    # total_elems = Bsz * Csz * S * S * H
    tmp = pid
    h = tmp % H
    tmp = tmp // H
    j = tmp % S
    tmp = tmp // S
    i = tmp % S
    tmp = tmp // S
    c = tmp % Csz
    b = tmp // Csz

    # Accumulate over n (state dimension)
    acc = 0.0
    for n in range(N):
        b_val = tl.load(
            B_exp_ptr + b * B_exp_stride0 + c * B_exp_stride1 + i * B_exp_stride2 + h * B_exp_stride3 + n * B_exp_stride4
        )
        c_val = tl.load(
            C_exp_ptr + b * C_exp_stride0 + c * C_exp_stride1 + i * C_exp_stride2 + h * C_exp_stride3 + n * C_exp_stride4
        )
        acc += b_val * c_val
    # Store G[b, c, i, j, h] = acc
    tl.store(
        G_ptr + b * G_stride_b + c * G_stride_c + i * G_stride_i + j * G_stride_j + h * G_stride_h,
        acc
    )

# Kernel 3: Compute Y_diag[b, c, i, h, d] = sum_j (G[b, c, i, j, h] * L[b, c, i, j, h] * hidden[b, c, j, h, d])
# hidden shape: [B, C, S, H, head_dim]; output Y: [B, C, S, H, head_dim], cast to bfloat16
@triton.jit
def compute_Y_diag_kernel(
    G_ptr, L_ptr, hidden_ptr, Y_ptr,
    Bsz, Csz, S, H, head_dim,
    G_stride_b, G_stride_c, G_stride_i, G_stride_j, G_stride_h,
    L_stride_b, L_stride_c, L_stride_i, L_stride_j, L_stride_h,
    hidden_stride_b, hidden_stride_c, hidden_stride_s, hidden_stride_h, hidden_stride_d,
    Y_stride_b, Y_stride_c, Y_stride_i, Y_stride_h, Y_stride_d,
    BLOCK_D: tl.constexpr
):
    # grid over (b, c, i, h, d-block)
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)
    h = tl.program_id(3)
    d_block = tl.program_id(4)

    d_offsets = d_block * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = d_offsets < head_dim

    # Initialize accumulator per d
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)

    # Loop over j in [0..S-1]
    for j in range(S):
        # Load G and L
        G_val = tl.load(
            G_ptr + b * G_stride_b + c * G_stride_c + i * G_stride_i + j * G_stride_j + h * G_stride_h
        )
        L_val = tl.load(
            L_ptr + b * L_stride_b + c * L_stride_c + i * L_stride_i + j * L_stride_j + h * L_stride_h
        )
        # Load hidden for all d in block
        hidden_ptrs = hidden_ptr + b * hidden_stride_b + c * hidden_stride_c + j * hidden_stride_s + h * hidden_stride_h + d_offsets * hidden_stride_d
        hidden_vals = tl.load(hidden_ptrs, mask=mask_d, other=0.0)
        # Accumulate: acc += G_val * L_val * hidden_vals
        acc += G_val * L_val * hidden_vals

    # Store results
    Y_ptrs = Y_ptr + b * Y_stride_b + c * Y_stride_c + i * Y_stride_i + h * Y_stride_h + d_offsets * Y_stride_d
    tl.store(Y_ptrs, acc, mask=mask_d)

class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                A_cumsum: torch.Tensor,
                B: torch.Tensor,
                C: torch.Tensor) -> torch.Tensor:
        """
        Computes Y_diag as in the original run function, but using Triton kernels for:
        - Building L (causal mask with exp(cumsum(A))).
        - Computing G contraction.
        - Computing Y_diag via reduction over j and combining with hidden.
        Returns Y_diag in bfloat16.
        """
        # Shapes
        Bsz, Csz, S, H, head_dim = hidden_states.shape
        device = hidden_states.device

        # Prepare expanded B and C to [B, C, S, H, N], where N is state size (B.size(-1))
        N_GROUPS = 8
        N = B.size(-1)
        B_exp = B.repeat_interleave(NUM_HEADS // N_GROUPS, dim=3)  # [B, C, S, H, N]
        C_exp = C.repeat_interleave(NUM_HEADS // N_GROUPS, dim=3)  # [B, C, S, H, N]

        # Allocate L: [B, C, S, S, H] as float32
        L = torch.zeros((Bsz, Csz, S, S, H), device=device, dtype=torch.float32)
        # Ensure A_cumsum is float32
        A = A_cumsum.to(torch.float32)

        # Launch build_L_kernel: grid = (Bsz, Csz)
        build_L_kernel[(Bsz, Csz)](
            A, L,
            Bsz, Csz, S, H,
            A.stride(0), A.stride(1), A.stride(2), A.stride(3),
            L.stride(0), L.stride(1), L.stride(2), L.stride(3), L.stride(4),
            CHUNK_SIZE=S
        )

        # Allocate G: [B, C, S, S, H] as float32
        G = torch.empty((Bsz, Csz, S, S, H), device=device, dtype=torch.float32)

        # Launch compute_G_kernel_simple: grid over all elements
        total = Bsz * Csz * S * S * H
        compute_G_kernel_simple[(total,)](
            B_exp, C_exp, G,
            Bsz, Csz, S, H, N,
            B_exp.stride(0), B_exp.stride(1), B_exp.stride(2), B_exp.stride(3), B_exp.stride(4),
            C_exp.stride(0), C_exp.stride(1), C_exp.stride(2), C_exp.stride(3), C_exp.stride(4),
            G.stride(0), G.stride(1), G.stride(2), G.stride(3), G.stride(4),
            total_elems=total
        )

        # Compute Y_diag: [B, C, S, H, head_dim] in float32, then cast to bfloat16
        Y = torch.empty((Bsz, Csz, S, H, head_dim), device=device, dtype=torch.float32)

        # Launch compute_Y_diag_kernel with grid over (B, C, S, H, blocks of head_dim)
        BLOCK_D = 32  # tune as needed
        grid_d = (head_dim + BLOCK_D - 1) // BLOCK_D
        compute_Y_diag_kernel[(Bsz, Csz, S, H, grid_d)](
            G, L, hidden_states.to(torch.float32), Y,
            Bsz, Csz, S, H, head_dim,
            G.stride(0), G.stride(1), G.stride(2), G.stride(3), G.stride(4),
            L.stride(0), L.stride(1), L.stride(2), L.stride(3), L.stride(4),
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2), hidden_states.stride(3), hidden_states.stride(4),
            Y.stride(0), Y.stride(1), Y.stride(2), Y.stride(3), Y.stride(4),
            BLOCK_D=BLOCK_D
        )

        # Return in bfloat16 as per original
        return Y.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
