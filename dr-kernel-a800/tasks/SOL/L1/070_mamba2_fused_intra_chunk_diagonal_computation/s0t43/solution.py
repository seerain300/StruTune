import torch
import triton
import triton.language as tl


@triton.jit
def compute_G_triton(
    B_ptr, C_ptr, G_ptr,
    B_stride0, B_stride1, B_stride2, B_stride3, B_stride4,
    C_stride0, C_stride1, C_stride2, C_stride3, C_stride4,
    G_stride0, G_stride1, G_stride2, G_stride3, G_stride4,
    Bsz, Csz, Ssz, Hsz, N,  # N is 128
):
    # Total number of programs = Bsz * Csz * Ssz * Ssz * Hsz
    pid = tl.program_id(axis=0)
    total = Bsz * Csz * Ssz * Ssz * Hsz
    # Decode pid -> (b, c, i, j, h)
    iS = Ssz
    hS = Hsz
    b = pid // (Csz * iS * iS * hS)
    tmp = pid % (Csz * iS * iS * hS)
    c = tmp // (iS * iS * hS)
    tmp = tmp % (iS * iS * hS)
    i = tmp // (iS * hS)
    tmp = tmp % (iS * hS)
    j = tmp // hS
    h = tmp % hS

    # Accumulator for G[i, j, h]
    g_val = 0.0

    # Loop over n (state dimension)
    for n in tl.static_range(N):
        # B[b, c, j, n, i, h]
        b_idx = b * B_stride0
        c_idx = c * B_stride1
        j_idx = j * B_stride2
        n_idx = n * B_stride3
        i_idx = i * B_stride4
        # C[b, c, i, n, j, h]
        c_idx2 = c * C_stride0
        i_idx2 = i * C_stride2
        j_idx2 = j * C_stride3
        h_idx2 = h * C_stride4  # h dim is last, but C has N dim before H
        # Note: We expanded along H before calling this kernel, so C_ptr has H at the end.

        # Gather pointers with strides
        b_val = tl.load(B_ptr + b_idx + c_idx + j_idx + n_idx + i_idx, mask=True)
        c_val = tl.load(C_ptr + c_idx2 + i_idx2 + n_idx + j_idx2 + h_idx2, mask=True)
        g_val += b_val * c_val

    # Store G[b, c, i, j, h]
    g_idx = b * G_stride0 + c * G_stride1 + i * G_stride2 + j * G_stride3 + h * G_stride4
    tl.store(G_ptr + g_idx, g_val)


@triton.jit
def compute_Y_diag_triton(
    G_ptr, hidden_ptr, Y_ptr,
    B_stride0, B_stride1, B_stride2, B_stride3, B_stride4,
    hidden_stride0, hidden_stride1, hidden_stride2, hidden_stride3, hidden_stride4,
    Y_stride0, Y_stride1, Y_stride2, Y_stride3, Y_stride4,
    Bsz, Csz, Ssz, Hsz, head_dim: tl.constexpr,
):
    # axis 0 over B*C*S*H, axis 1 over head_dim vectorization
    axis0 = tl.program_id(axis=0)
    h_idx = tl.program_id(axis=1)

    # Decode axis0 -> (b, c, i, h)
    tmp = axis0
    c = tmp // (Ssz * Hsz)
    tmp = tmp % (Ssz * Hsz)
    i = tmp // Hsz
    h = tmp % Hsz
    b = axis0 // (Csz * Ssz * Hsz)

    # Accumulator for Y[b, c, i, h, h_idx]
    total = 0.0

    # Loop over j
    for j in tl.static_range(Ssz):
        # Load G[b, c, i, j, h]
        g_idx = b * B_stride0 + c * B_stride1 + i * B_stride2 + j * B_stride3 + h * B_stride4
        g_val = tl.load(G_ptr + g_idx)

        # Load hidden[b, c, j, h, h_idx]
        hidden_idx = b * hidden_stride0 + c * hidden_stride1 + j * hidden_stride2 + h * hidden_stride3 + h_idx * hidden_stride4
        h_val = tl.load(hidden_ptr + hidden_idx)

        total += g_val * h_val

    # Store Y[b, c, i, h, h_idx]
    y_idx = b * Y_stride0 + c * Y_stride1 + i * Y_stride2 + h * Y_stride3 + h_idx * Y_stride4
    tl.store(Y_ptr + y_idx, total)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # No parameters; everything is computed in Triton

    def forward(self, hidden_states: torch.Tensor, A_cumsum: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        """
        Compute Y_diag with Triton kernels. A_cumsum is unused to match the original run function's output.
        """
        # Shapes
        Bsz, Csz, Ssz, Hsz, head_dim = hidden_states.shape
        # As per reference, CHUNK_SIZE = 128 and NUM_HEADS = 32, N_GROUPS = 8 -> repeat_interleave 4
        # Expand B and C along H by repeat_interleave(NUM_HEADS // N_GROUPS)
        # Note: repeat_interleave along dim=3 (H), repeating factor = 4
        B_exp = B.repeat_interleave(4, dim=3)
        C_exp = C.repeat_interleave(4, dim=3)

        # Ensure contiguous for Triton
        B_exp = B_exp.contiguous()
        C_exp = C_exp.contiguous()

        # Prepare G: [B, C, S, S, H] as float32
        G = torch.empty((Bsz, Csz, Ssz, Ssz, Hsz), device=hidden_states.device, dtype=torch.float32)

        # Launch compute_G_triton
        grid_G = (Bsz * Csz * Ssz * Ssz * Hsz,)
        compute_G_triton[grid_G](
            B_exp, C_exp, G,
            B_exp.stride(0), B_exp.stride(1), B_exp.stride(2), B_exp.stride(3), B_exp.stride(4),
            C_exp.stride(0), C_exp.stride(1), C_exp.stride(2), C_exp.stride(3), C_exp.stride(4),
            G.stride(0), G.stride(1), G.stride(2), G.stride(3), G.stride(4),
            Bsz, Csz, Ssz, Hsz, 128,
        )

        # Prepare hidden (float32)
        hidden_f32 = hidden_states.contiguous().to(torch.float32)

        # Prepare Y: [B, C, S, H, head_dim] float32
        Y = torch.empty((Bsz, Csz, Ssz, Hsz, head_dim), device=hidden_states.device, dtype=torch.float32)

        # Launch compute_Y_diag_triton over (B*C*S*H, head_dim) grid
        grid_Y = (Bsz * Csz * Ssz * Hsz, head_dim)
        compute_Y_diag_triton[grid_Y](
            G, hidden_f32, Y,
            G.stride(0), G.stride(1), G.stride(2), G.stride(3), G.stride(4),  # we use G strides to index (b,c,i,j,h)
            hidden_f32.stride(0), hidden_f32.stride(1), hidden_f32.stride(2), hidden_f32.stride(3), hidden_f32.stride(4),
            Y.stride(0), Y.stride(1), Y.stride(2), Y.stride(3), Y.stride(4),
            Bsz, Csz, Ssz, Hsz, head_dim,
        )

        # Return in bfloat16 to match original
        return Y.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
