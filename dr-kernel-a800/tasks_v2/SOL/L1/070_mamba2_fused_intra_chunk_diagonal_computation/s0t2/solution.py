import torch
import triton
import triton.language as tl


# Kernel 1: Build L = exp(cumsum(A_lower_tri)) per (b, h, c, i), rows i in [0..S-1]
# A_perm: [B, C, H, S] float32, contiguous
# L_flat: [B, C, S, S, H] float32, contiguous (linearized as [B, C, S, S, H])
@triton.jit
def build_L_kernel(
    A_ptr, L_ptr,
    B: tl.constexpr, C: tl.constexpr, S: tl.constexpr, H: tl.constexpr
):
    # Grid over (B, C, H, S): each program computes L vector for one i
    b = tl.program_id(0)
    c = tl.program_id(1)
    h = tl.program_id(2)
    i = tl.program_id(3)

    # Running cumsum for row i
    cum = 0.0
    for j in range(0, S):
        # Load A[b, h, c, j]
        A_idx = b * (C * H * S) + c * (H * S) + h * S + j
        a_val = tl.load(A_ptr + A_idx)
        # Include only if j <= i (lower-triangular)
        include = j <= i
        # Update cumsum; include is a scalar predicate. Triton will cast it to 0.0/1.0
        cum = cum + a_val * include
        # Store exp(cum) into L[b, c, i, j, h]
        L_idx = b * (C * S * S * H) + c * (S * S * H) + i * (S * H) + j * H + h
        L_val = tl.exp(cum)
        tl.store(L_ptr + L_idx, L_val)


# Kernel 2: Compute G = sum_n C[:, :, :, n, :] * B[:, :, :, n, :] -> [B, C, S, S, H]
# B_expanded: [B, C, S, H, N] float32
# C_expanded: [B, C, S, H, N] float32
# G: [B, C, S, S, H] float32
@triton.jit
def compute_G_kernel(
    B_ptr, C_ptr, G_ptr,
    Bsz: tl.constexpr, Csz: tl.constexpr, S: tl.constexpr, H: tl.constexpr, N: tl.constexpr
):
    # Grid over (B, C, S, S, H): each program computes one element G[b, c, i, j, h]
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)
    j = tl.program_id(3)
    h = tl.program_id(4)

    acc = 0.0
    for n in range(0, N):
        B_idx = b * (Bsz * Csz * S * H * N) + c * (S * H * N) + j * (H * N) + h * N + n
        C_idx = b * (Bsz * Csz * S * H * N) + c * (S * H * N) + i * (H * N) + h * N + n
        b_val = tl.load(B_ptr + B_idx)
        c_val = tl.load(C_ptr + C_idx)
        acc += b_val * c_val

    G_idx = b * (Csz * S * S * H) + c * (S * S * H) + i * (S * H) + j * H + h
    tl.store(G_ptr + G_idx, acc)


# Kernel 3: Compute Y_diag = sum_j G[i, j, h] * hidden_states[j, h, d], per (b, c, i, h, d)
# G: [B, C, S, S, H] float32
# hidden_states: [B, C, S, H, head_dim] float32
# Y_diag: [B, C, S, H, head_dim] float32
@triton.jit
def compute_Y_diag_kernel(
    G_ptr, hidden_ptr, Y_ptr,
    Bsz: tl.constexpr, Csz: tl.constexpr, S: tl.constexpr, H: tl.constexpr, HEAD_DIM: tl.constexpr
):
    # Grid over (B*C*S, H, HEAD_DIM): each program computes one output element for fixed (b, c, i, h, d)
    pid0 = tl.program_id(0)
    h = tl.program_id(1)
    d = tl.program_id(2)

    b = pid0 // (Csz * S)
    tmp = pid0 % (Csz * S)
    c = tmp // S
    i = tmp % S

    acc = 0.0
    for j in range(0, S):
        G_idx = b * (Csz * S * S * H) + c * (S * S * H) + i * (S * H) + j * H + h
        hs_idx = b * (Bsz * Csz * S * H * HEAD_DIM) + c * (S * H * HEAD_DIM) + j * (H * HEAD_DIM) + h * HEAD_DIM + d
        g_val = tl.load(G_ptr + G_idx)
        hs_val = tl.load(hidden_ptr + hs_idx)
        acc += g_val * hs_val

    Y_idx = b * (Bsz * Csz * S * H * HEAD_DIM) + c * (S * H * HEAD_DIM) + i * (H * HEAD_DIM) + h * HEAD_DIM + d
    tl.store(Y_ptr + Y_idx, acc)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                A_cumsum: torch.Tensor,
                B: torch.Tensor,
                C: torch.Tensor):
        """
        Triton-optimized implementation of the original run function.
        Assumptions:
          - CHUNK_SIZE (S) = 128 (compile-time constant for Triton loops)
          - NUM_HEADS (H) = 32 (compile-time constant)
          - state_size (N) = 128 (compile-time constant for G contraction)
        Returns:
          Y_diag with dtype bfloat16, shape [B, C, S, H, head_dim]
        """
        assert hidden_states.is_cuda and A_cumsum.is_cuda and B.is_cuda and C.is_cuda, "Tensors must be on CUDA"

        # Shapes
        Bsz, Csz, S, H, head_dim = hidden_states.shape
        # The original code uses CHUNK_SIZE=128; we follow that here.
        assert S == 128, "This Triton implementation assumes S=128"
        assert H == 32, "This Triton implementation assumes H=32"
        # A_cumsum: [B, H, C, S]
        assert A_cumsum.shape == (Bsz, H, Csz, S), "A_cumsum must have shape [B, H, C, S]"

        # Ensure contiguous and dtype float32 for kernel math
        hidden_states = hidden_states.contiguous().to(torch.float32)
        A_cumsum = A_cumsum.contiguous().to(torch.float32)

        # Permute A_cumsum to [B, C, H, S] for kernel indexing
        A_perm = A_cumsum.permute(0, 2, 1, 3).contiguous()

        # Expand B and C along heads (repeat_interleave by 4 to get 32 from 8 groups)
        B_expanded = B.repeat_interleave(4, dim=3).contiguous().to(torch.float32)
        C_expanded = C.repeat_interleave(4, dim=3).contiguous().to(torch.float32)
        # After expansion, shapes should be [B, C, S, H, N] with N=128.

        # Allocate outputs
        # L_flat: [B, C, S, S, H] as float32
        L_flat = torch.empty((Bsz, Csz, S, S, H), dtype=torch.float32, device=hidden_states.device)
        # G: [B, C, S, S, H] float32
        G = torch.empty((Bsz, Csz, S, S, H), dtype=torch.float32, device=hidden_states.device)
        # Y_diag: [B, C, S, H, head_dim] float32
        Y_diag = torch.empty((Bsz, Csz, S, H, head_dim), dtype=torch.float32, device=hidden_states.device)

        # Launch build_L_kernel
        grid_L = (Bsz, Csz, H, S)
        build_L_kernel[grid_L](
            A_perm, L_flat,
            Bsz, Csz, S, H
        )

        # Launch compute_G_kernel
        grid_G = (Bsz, Csz, S, S, H)
        compute_G_kernel[grid_G](
            B_expanded, C_expanded, G,
            Bsz, Csz, S, H, 128  # N = state_size = 128
        )

        # Launch compute_Y_diag_kernel
        grid_Y = (Bsz * Csz * S, H, head_dim)
        compute_Y_diag_kernel[grid_Y](
            G, hidden_states, Y_diag,
            Bsz, Csz, S, H, head_dim
        )

        # Return in bfloat16 to match original
        return Y_diag.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
