import torch
import triton
import triton.language as tl


# Kernel: build L with lower-triangular causal mask:
# L[b, c, i, j, h] = exp(sum_{t=0..j} A[b, h, c, t]) if i >= j else 0
@triton.jit
def build_L_1d(A_ptr, L_ptr,
               total_elems,  # B*C*S*S*H
               Bsz, C, S, H,
               A_s0, A_s1, A_s2, A_s3,
               L_s0, L_s1, L_s2, L_s3, L_s4,
               diagonal: tl.constexpr):
    pid = tl.program_id(0)
    if pid >= total_elems:
        return

    # Map pid -> (b, c, i, j, h)
    tmp = pid
    h = tmp % H
    tmp = tmp // H
    j = tmp % S
    tmp = tmp // S
    i = tmp % S
    tmp = tmp // S
    c = tmp % C
    b = tmp // C

    # If i < j, L = 0 (lower-triangular mask with diagonal=0)
    if i < j:
        L_val = 0.0
    else:
        # Compute cumsum along j positions up to j
        cumsum = 0.0
        for t in range(0, S):
            A_idx = b * A_s0 + h * A_s1 + c * A_s2 + t * A_s3
            a_val = tl.load(A_ptr + A_idx)  # A is float32
            cumsum += a_val
        L_val = tl.exp(cumsum)

    # Store to L
    L_idx = b * L_s0 + c * L_s1 + i * L_s2 + j * L_s3 + h * L_s4
    tl.store(L_ptr + L_idx, L_val)


# Kernel: compute G via contraction over N (state dimension):
# G[b, c, i, j, h] = sum_{n} C_exp[b, c, i, n, j, h] * B_exp[b, c, j, n, i, h]
@triton.jit
def compute_G_1d(C_exp_ptr, B_exp_ptr, G_ptr,
                 total_elems,  # B*C*S*S*H
                 Bsz, C, S, H, N,
                 C_s0, C_s1, C_s2, C_s3, C_s4,
                 B_s0, B_s1, B_s2, B_s3, B_s4,
                 G_s0, G_s1, G_s2, G_s3, G_s4):
    pid = tl.program_id(0)
    if pid >= total_elems:
        return

    # Map pid -> (b, c, i, j, h)
    tmp = pid
    h = tmp % H
    tmp = tmp // H
    j = tmp % S
    tmp = tmp // S
    i = tmp % S
    tmp = tmp // S
    c = tmp % C
    b = tmp // C

    acc = 0.0
    for n in range(0, N):
        C_idx = b * C_s0 + c * C_s1 + i * C_s2 + n * C_s3 + j * C_s4
        B_idx = b * B_s0 + c * B_s1 + j * B_s2 + n * B_s3 + i * B_s4
        C_val = tl.load(C_exp_ptr + C_idx)
        B_val = tl.load(B_exp_ptr + B_idx)
        acc += C_val * B_val

    G_idx = b * G_s0 + c * G_s1 + i * G_s2 + j * G_s3 + h * G_s4
    tl.store(G_ptr + G_idx, acc)


# Kernel: compute Y_diag by contracting M with hidden_states over j, for a fixed d:
# Y[b, c, i, h, d] = sum_{j} M[b, c, i, j, h] * hidden_states[b, c, j, h, d]
# We vectorize over i and loop over j in a separate launcher by splitting the grid over i-blocks.
@triton.jit
def compute_Y_diag_j_loop(M_ptr, hidden_ptr, Y_ptr,
                          Bsz, C, S, H, head_dim,
                          M_s0, M_s1, M_s2, M_s3, M_s4,
                          hidden_s0, hidden_s1, hidden_s2, hidden_s3, hidden_s4,
                          Y_s0, Y_s1, Y_s2, Y_s3, Y_s4,
                          b, c, h, d):
    # Single program handles one (i) for a given (b, c, h, d). We will launch a 1D grid over i.
    # To support vectorization, we can change this to a 1D vector of i and use a loop over j.
    # Here we keep it simple and robust: 1D over i.
    # We will vectorize over i via a separate launcher which sets this grid accordingly.
    i = tl.program_id(0)
    if i >= S:
        return

    acc = 0.0
    for j in range(0, S):
        M_idx = b * M_s0 + c * M_s1 + i * M_s2 + j * M_s3 + h * M_s4
        hid_idx = b * hidden_s0 + c * hidden_s1 + j * hidden_s2 + h * hidden_s3 + d * hidden_s4
        M_val = tl.load(M_ptr + M_idx)
        hid_val = tl.load(hidden_ptr + hid_idx)
        acc += M_val * hid_val

    Y_idx = b * Y_s0 + c * Y_s1 + i * Y_s2 + h * Y_s3 + d * Y_s4
    tl.store(Y_ptr + Y_idx, acc)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, A_cumsum: torch.Tensor, B: torch.Tensor, C: torch.Tensor):
        # Ensure contiguous and float32 for stable accumulation
        device = hidden_states.device
        A_cumsum = A_cumsum.contiguous().to(torch.float32)
        B = B.contiguous().to(torch.float32)
        C = C.contiguous().to(torch.float32)
        hidden_states = hidden_states.contiguous().to(torch.float32)

        # Shape extraction
        Bsz = hidden_states.shape[0]
        Csz = hidden_states.shape[1]
        S = hidden_states.shape[2]
        H = hidden_states.shape[3]
        head_dim = hidden_states.shape[4]

        # Constants from reference
        N_GROUPS = 8
        NUM_HEADS = 32
        repeat_factor = NUM_HEADS // N_GROUPS  # 4
        # Typical N from input C/B: if not available, default to 128
        N = B.shape[4] if B.dim() == 5 else 128

        # 1) Compute L in Triton: [B, C, S, S, H]
        L = torch.empty((Bsz, Csz, S, S, H), device=device, dtype=torch.float32)
        total_L = Bsz * Csz * S * S * H
        grid_L = (total_L,)
        build_L_1d[grid_L](
            A_cumsum, L,
            total_L,
            Bsz, Csz, S, H,
            A_cumsum.stride(0), A_cumsum.stride(1), A_cumsum.stride(2), A_cumsum.stride(3),
            L.stride(0), L.stride(1), L.stride(2), L.stride(3), L.stride(4),
            diagonal=0
        )

        # 2) Expand B and C along H by repeat_interleave(NUM_HEADS // N_GROUPS = 4, dim=3)
        B_expanded = B.repeat_interleave(repeat_factor, dim=3)  # [B, C, S, H, N]
        C_expanded = C.repeat_interleave(repeat_factor, dim=3)  # [B, C, S, H, N]

        # 3) Compute G in Triton: [B, C, S, S, H]
        G = torch.empty((Bsz, Csz, S, S, H), device=device, dtype=torch.float32)
        total_G = Bsz * Csz * S * S * H
        grid_G = (total_G,)
        compute_G_1d[grid_G](
            C_expanded, B_expanded, G,
            total_G,
            Bsz, Csz, S, H, N,
            C_expanded.stride(0), C_expanded.stride(1), C_expanded.stride(2), C_expanded.stride(3), C_expanded.stride(4),
            B_expanded.stride(0), B_expanded.stride(1), B_expanded.stride(2), B_expanded.stride(3), B_expanded.stride(4),
            G.stride(0), G.stride(1), G.stride(2), G.stride(3), G.stride(4),
        )

        # 4) Compute M = G * L
        M = G * L  # element-wise multiply

        # 5) Compute Y_diag via Triton: [B, C, S, H, head_dim], return in bfloat16
        Y = torch.empty((Bsz, Csz, S, H, head_dim), device=device, dtype=torch.float32)

        # Launch compute_Y_diag_j_loop for each (b, c, h, d)
        for b in range(0, Bsz):
            for c in range(0, Csz):
                for h in range(0, H):
                    for d in range(0, head_dim):
                        grid_Y = (S,)
                        compute_Y_diag_j_loop[grid_Y](
                            M, hidden_states, Y,
                            Bsz, Csz, S, H, head_dim,
                            M.stride(0), M.stride(1), M.stride(2), M.stride(3), M.stride(4),
                            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2), hidden_states.stride(3), hidden_states.stride(4),
                            Y.stride(0), Y.stride(1), Y.stride(2), Y.stride(3), Y.stride(4),
                            b, c, h, d
                        )

        # Return in bfloat16 to match original behavior
        return Y.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
