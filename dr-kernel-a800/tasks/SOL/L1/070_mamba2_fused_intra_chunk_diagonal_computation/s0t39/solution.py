import torch
import triton
import triton.language as tl


# Kernel: build L with lower-triangular causal mask:
# L[b, c, i, j, h] = exp(sum_{t=0..j} A[b, h, c, t]) if i >= j else 0
@triton.jit
def build_L_1d(A_ptr, L_ptr,
               total_elems,  # B*C*S*S*H
               Bsz, C, S, H,
               A_s0, A_s1, A_s2, A_s3,  # strides for A: (B,H,C,S)
               L_s0, L_s1, L_s2, L_s3, L_s4,  # strides for L: (B,C,S,S,H)
               S_const: tl.constexpr, H_const: tl.constexpr):
    pid = tl.program_id(0)
    if pid >= total_elems:
        return

    # Map pid to (b, c, i, j, h)
    tmp = pid
    h = tmp % H_const
    tmp = tmp // H_const
    j = tmp % S_const
    tmp = tmp // S_const
    i = tmp % S_const
    tmp = tmp // S_const
    c = tmp % C
    b = tmp // C

    # Only compute if i >= j; else L = 0
    if i < j:
        # store 0.0 in L
        L_idx = b * L_s0 + c * L_s1 + i * L_s2 + j * L_s3 + h * L_s4
        tl.store(L_ptr + L_idx, 0.0)
        return

    # Load A[b, h, c, :] and compute cumsum up to j
    cumsum = 0.0
    for t in range(0, S_const):
        A_idx = b * A_s0 + h * A_s1 + c * A_s2 + t * A_s3
        a_val = tl.load(A_ptr + A_idx)
        # assume A is float32
        cumsum += tl.exp(a_val)
    L_val = cumsum  # since i >= j, L[i, j, h] = exp(cumsum_j[j])
    L_idx = b * L_s0 + c * L_s1 + i * L_s2 + j * L_s3 + h * L_s4
    tl.store(L_ptr + L_idx, L_val)


# Kernel: compute G[i, j, h] = sum_n C_exp[b, c, i, n, j, h] * B_exp[b, c, j, n, i, h]
@triton.jit
def compute_G_1d(C_exp_ptr, B_exp_ptr, G_ptr,
                 total_elems,  # B*C*S*S*H
                 Bsz, C, S, H, N,
                 Cexp_s0, Cexp_s1, Cexp_s2, Cexp_s3, Cexp_s4,  # strides for C_exp: (B,C,S,H,N)
                 Bexp_s0, Bexp_s1, Bexp_s2, Bexp_s3, Bexp_s4,  # strides for B_exp: (B,C,S,H,N)
                 G_s0, G_s1, G_s2, G_s3, G_s4,  # strides for G: (B,C,S,S,H)
                 S_const: tl.constexpr, H_const: tl.constexpr, N_const: tl.constexpr):
    pid = tl.program_id(0)
    if pid >= total_elems:
        return

    # Map pid to (b, c, i, j, h)
    tmp = pid
    h = tmp % H_const
    tmp = tmp // H_const
    j = tmp % S_const
    tmp = tmp // S_const
    i = tmp % S_const
    tmp = tmp // S_const
    c = tmp % C
    b = tmp // C

    acc = 0.0
    for n in range(0, N_const):
        Cexp_idx = b * Cexp_s0 + c * Cexp_s1 + i * Cexp_s2 + h * Cexp_s3 + n * Cexp_s4
        Bexp_idx = b * Bexp_s0 + c * Bexp_s1 + j * Bexp_s2 + h * Bexp_s3 + n * Bexp_s4
        Cexp_val = tl.load(Cexp_ptr + Cexp_idx)
        Bexp_val = tl.load(Bexp_ptr + Bexp_idx)
        acc += Cexp_val * Bexp_val

    G_idx = b * G_s0 + c * G_s1 + i * G_s2 + j * G_s3 + h * G_s4
    tl.store(G_ptr + G_idx, acc)


# Kernel: compute Y_diag[b, c, i, h, d] = sum_j (M[b, c, i, j, h] * hidden[b, c, j, h, d]),
# where M = G * L. We pass G and L and hidden to compute Y_diag.
# We vectorize over i in tiles and loop over j and d.
@triton.jit
def compute_Y_diag_2d(G_ptr, L_ptr, hidden_ptr, Y_ptr,
                      Bsz, C, S, H, head_dim,
                      G_s0, G_s1, G_s2, G_s3, G_s4,  # strides for G: (B,C,S,S,H)
                      L_s0, L_s1, L_s2, L_s3, L_s4,  # strides for L: (B,C,S,S,H)
                      hidden_s0, hidden_s1, hidden_s2, hidden_s3, hidden_s4,  # strides for hidden: (B,C,S,H,head_dim)
                      Y_s0, Y_s1, Y_s2, Y_s3, Y_s4,  # strides for Y: (B,C,S,H,head_dim)
                      S_const: tl.constexpr, H_const: tl.constexpr, head_const: tl.constexpr):
    # grid: (B*C*H, ceil(S/Tile_i))
    grid0 = tl.program_id(0)
    grid1 = tl.program_id(1)

    # recover (b, c, h) from grid0
    h = grid0 % H_const
    tmp = grid0 // H_const
    c = tmp % C
    b = tmp // C

    # tile over i
    Tile_i = 64  # tile size for i
    i_start = grid1 * Tile_i
    i_offsets = i_start + tl.arange(0, Tile_i)
    mask_i = i_offsets < S_const

    # initialize Y for this (b, c, tile)
    # Y shape: [B, C, S, H, head_dim]; we process one h at a time
    for d in range(0, head_const):
        Y_vec = tl.zeros((Tile_i,), dtype=tl.float32)
        for j in range(0, S_const):
            # load M[j, i] = G[b, c, j, i, h] * L[b, c, j, i, h]
            # accumulate Y_vec[i] += M[j, i] * hidden[b, c, j, h, d]
            for i_idx in range(0, Tile_i):
                if mask_i[i_idx]:
                    i_i = i_offsets[i_idx]
                    G_idx = b * G_s0 + c * G_s1 + j * G_s2 + i_i * G_s3 + h * G_s4
                    L_idx = b * L_s0 + c * L_s1 + j * L_s2 + i_i * L_s3 + h * L_s4
                    hidden_idx = b * hidden_s0 + c * hidden_s1 + j * hidden_s2 + h * hidden_s3 + d * hidden_s4
                    G_val = tl.load(G_ptr + G_idx)
                    L_val = tl.load(L_ptr + L_idx)
                    hidden_val = tl.load(hidden_ptr + hidden_idx)
                    M_val = G_val * L_val
                    Y_vec[i_idx] += M_val * hidden_val
        # store Y_vec to Y[b, c, :, h, d] over the tile i
        for i_idx in range(0, Tile_i):
            if mask_i[i_idx]:
                i_i = i_offsets[i_idx]
                Y_idx = b * Y_s0 + c * Y_s1 + i_i * Y_s2 + h * Y_s3 + d * Y_s4
                tl.store(Y_ptr + Y_idx, Y_vec[i_idx])


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, A_cumsum: torch.Tensor, B: torch.Tensor, C: torch.Tensor):
        # Ensure contiguous and float32 for computation
        device = hidden_states.device
        Bsz = hidden_states.shape[0]
        Csz = hidden_states.shape[1]
        S = hidden_states.shape[2]
        H = hidden_states.shape[3]
        head_dim = hidden_states.shape[4]

        A_cumsum = A_cumsum.contiguous().to(torch.float32)
        B = B.contiguous().to(torch.float32)
        C = C.contiguous().to(torch.float32)
        hidden_states = hidden_states.contiguous().to(torch.float32)

        # Expand B and C along H by repeat_interleave(NUM_HEADS // N_GROUPS = 4, dim=3)
        N_GROUPS = 8
        NUM_HEADS = 32
        repeat_factor = NUM_HEADS // N_GROUPS  # 4
        B_expanded = B.repeat_interleave(repeat_factor, dim=3)  # [B, C, S, H, N]
        C_expanded = C.repeat_interleave(repeat_factor, dim=3)  # [B, C, S, H, N]
        N = B_expanded.shape[4]

        # Allocate outputs
        L = torch.empty((Bsz, Csz, S, S, H), device=device, dtype=torch.float32)
        G = torch.empty((Bsz, Csz, S, S, H), device=device, dtype=torch.float32)
        Y = torch.empty((Bsz, Csz, S, H, head_dim), device=device, dtype=torch.float32)

        # Launch Triton kernels

        # 1) Build L: 1D grid over total elements
        total_L = Bsz * Csz * S * S * H
        grid_L = (total_L,)
        build_L_1d[grid_L](
            A_cumsum, L,
            total_L,
            Bsz, Csz, S, H,
            A_cumsum.stride(0), A_cumsum.stride(1), A_cumsum.stride(2), A_cumsum.stride(3),
            L.stride(0), L.stride(1), L.stride(2), L.stride(3), L.stride(4),
            S_const=S, H_const=H,
        )

        # 2) Compute G: 1D grid over total elements
        total_G = Bsz * Csz * S * S * H
        grid_G = (total_G,)
        compute_G_1d[grid_G](
            C_expanded, B_expanded, G,
            total_G,
            Bsz, Csz, S, H, N,
            C_expanded.stride(0), C_expanded.stride(1), C_expanded.stride(2), C_expanded.stride(3), C_expanded.stride(4),
            B_expanded.stride(0), B_expanded.stride(1), B_expanded.stride(2), B_expanded.stride(3), B_expanded.stride(4),
            G.stride(0), G.stride(1), G.stride(2), G.stride(3), G.stride(4),
            S_const=S, H_const=H, N_const=N,
        )

        # 3) Compute Y_diag: 2D grid over (B*C*H, ceil(S/Tile_i))
        Tile_i = 64
        grid0 = Bsz * Csz * H
        grid1 = (S + Tile_i - 1) // Tile_i
        grid_Y = (grid0, grid1)
        compute_Y_diag_2d[grid_Y](
            G, L, hidden_states, Y,
            Bsz, Csz, S, H, head_dim,
            G.stride(0), G.stride(1), G.stride(2), G.stride(3), G.stride(4),
            L.stride(0), L.stride(1), L.stride(2), L.stride(3), L.stride(4),
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2), hidden_states.stride(3), hidden_states.stride(4),
            Y.stride(0), Y.stride(1), Y.stride(2), Y.stride(3), Y.stride(4),
            S_const=S, H_const=H, head_const=head_dim,
        )

        # Return in bfloat16 to match original Model output dtype
        return Y.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
