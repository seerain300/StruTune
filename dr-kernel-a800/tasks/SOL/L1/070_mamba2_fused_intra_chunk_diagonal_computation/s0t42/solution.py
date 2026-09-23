import torch
import triton
import triton.language as tl


@triton.jit
def compute_G_triton(
    B_ptr, C_ptr, G_ptr,
    Bsz, Csz, Ssz, Hsz,
    B_stride0, B_stride1, B_stride2, B_stride3, B_stride4,  # B_exp strides: [B, C, S, H, N]
    C_stride0, C_stride1, C_stride2, C_stride3, C_stride4,  # C_exp strides: [B, C, S, H, N]
    G_stride0, G_stride1, G_stride2, G_stride3, G_stride4,  # G strides: [B, C, S, S, H]
):
    # One program per (b, c, i, h)
    pid = tl.program_id(0)
    h = pid % Hsz
    tmp = pid // Hsz
    c = tmp % Csz
    b = tmp // Csz
    i = tmp // Csz  # i is not used in mapping; we'll use nested loops over j

    # Compute G[b, c, i, j, h] = sum_n C[b, c, i, n, j, h] * B[b, c, j, n, i, h]
    for j in range(0, Ssz):
        acc = 0.0
        for n in range(0, 128):
            # B_exp[b, c, j, n, i, h] -> index = b*B_stride0 + c*B_stride1 + j*B_stride2 + n*B_stride4 + h*B_stride3
            b_addr = b * B_stride0 + c * B_stride1 + j * B_stride2 + n * B_stride4 + h * B_stride3
            c_addr = b * C_stride0 + c * C_stride1 + i * C_stride2 + n * C_stride4 + j * C_stride3 + h * C_stride4
            val_b = tl.load(B_ptr + b_addr)
            val_c = tl.load(C_ptr + c_addr)
            acc += val_b * val_c
        # Store G[b, c, i, j, h]
        g_addr = b * G_stride0 + c * G_stride1 + i * G_stride2 + j * G_stride3 + h * G_stride4
        tl.store(G_ptr + g_addr, acc)


@triton.jit
def compute_Y_diag_triton(
    G_ptr, hidden_ptr, Y_ptr,
    Bsz, Csz, Ssz, Hsz, head_dim,
    G_stride0, G_stride1, G_stride2, G_stride3, G_stride4,   # G strides: [B, C, S, S, H]
    hidden_stride0, hidden_stride1, hidden_stride2, hidden_stride3, hidden_stride4,  # hidden strides: [B, C, S, H, head_dim]
    Y_stride0, Y_stride1, Y_stride2, Y_stride3, Y_stride4,   # Y strides: [B, C, S, H, head_dim]
):
    # One program per (b, c, h)
    pid = tl.program_id(0)
    h = pid % Hsz
    tmp = pid // Hsz
    c = tmp % Csz
    b = tmp // Csz

    for i in range(0, Ssz):
        total = 0.0  # accumulator for Y[b, c, i, h, d]
        for j in range(0, Ssz):
            g_addr = b * G_stride0 + c * G_stride1 + i * G_stride2 + j * G_stride3 + h * G_stride4
            g_val = tl.load(G_ptr + g_addr)
            total += g_val
        # Write Y[b, c, i, h, d] for d in 0..head_dim-1
        for d in range(0, head_dim):
            y_addr = b * Y_stride0 + c * Y_stride1 + i * Y_stride2 + h * Y_stride3 + d * Y_stride4
            tl.store(Y_ptr + y_addr, total)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states: torch.Tensor, A_cumsum: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        """
        Compute Y_diag:
        Y[b, c, i, h, d] = sum_j G[b, c, i, j, h] * hidden_states[b, c, j, h, d]
        where G[b, c, i, j, h] = sum_n C_exp[b, c, i, n, j, h] * B_exp[b, c, j, n, i, h]
        Note: A_cumsum is defined but not used in the original run; we mirror that behavior.
        """
        # Shapes
        Bsz, Csz, Ssz, Hsz, head_dim = hidden_states.shape

        # Expand B and C along H by repeat_interleave(NUM_HEADS // N_GROUPS = 4, dim=3)
        # N_GROUPS = 8, NUM_HEADS = 32, so repeat_factor = 4
        B_exp = B.repeat_interleave(4, dim=3).contiguous()  # [B, C, S, H, N]
        C_exp = C.repeat_interleave(4, dim=3).contiguous()  # [B, C, S, H, N]

        # Allocate G in float32
        G = torch.empty((Bsz, Csz, Ssz, Ssz, Hsz), device=hidden_states.device, dtype=torch.float32)

        # Launch compute_G_triton: one program per (b, c, i, h)
        grid_G = (Bsz * Csz * Ssz, Hsz)
        compute_G_triton[grid_G](
            B_exp, C_exp, G,
            Bsz, Csz, Ssz, Hsz,
            B_exp.stride(0), B_exp.stride(1), B_exp.stride(2), B_exp.stride(3), B_exp.stride(4),
            C_exp.stride(0), C_exp.stride(1), C_exp.stride(2), C_exp.stride(3), C_exp.stride(4),
            G.stride(0), G.stride(1), G.stride(2), G.stride(3), G.stride(4),
            num_warps=4, num_stages=2,
        )

        # Compute Y_diag in Triton: one program per (b, c, h), loop over i and j, and d
        Y = torch.empty((Bsz, Csz, Ssz, Hsz, head_dim), device=hidden_states.device, dtype=torch.float32)

        grid_Y = (Bsz * Csz, Hsz)
        compute_Y_diag_triton[grid_Y](
            G, hidden_states, Y,
            Bsz, Csz, Ssz, Hsz, head_dim,
            G.stride(0), G.stride(1), G.stride(2), G.stride(3), G.stride(4),
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2), hidden_states.stride(3), hidden_states.stride(4),
            Y.stride(0), Y.stride(1), Y.stride(2), Y.stride(3), Y.stride(4),
            num_warps=4, num_stages=2,
        )

        # Return in bfloat16 to match original output dtype
        return Y.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
