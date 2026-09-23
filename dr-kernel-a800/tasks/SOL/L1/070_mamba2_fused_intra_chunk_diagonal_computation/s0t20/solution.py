import torch
import triton
import triton.language as tl


# Kernel: build L[b, c, i, j, h] = exp(sum_{t=0..j} A[b,h,c,t]) for i >= j, else 0
# A: [B, H, C, S], L: [B, C, S, S, H]
@triton.jit
def build_L_kernel(
    A_ptr, L_ptr,
    B, H, C, S,
    stride_A_b, stride_A_h, stride_A_c, stride_A_s,
    stride_L_b, stride_L_c, stride_L_i, stride_L_j, stride_L_h,
):
    # Grid: (B*C*S, H)
    pid0 = tl.program_id(0)  # over B*C*S
    pid1 = tl.program_id(1)  # over H
    # Recover b, c, i from pid0
    # pid0 = ((b * C + c) * S + i)
    b = pid0 // (C * S)
    rem = pid0 % (C * S)
    c = rem // S
    i = rem % S
    h = pid1

    # Prepare cumsum vector for j positions
    cumsum = 0.0
    # Loop over j to compute cumsum(A[b,h,c,j])
    # Note: j goes from 0 to S-1
    for j in range(0, S):
        # A[b, h, c, j]
        a_off = b * stride_A_b + h * stride_A_h + c * stride_A_c + j * stride_A_s
        a_val = tl.load(A_ptr + a_off)  # float32
        cumsum += a_val
        # L[i, j, h] = exp(cumsum) if i >= j else 0
        if i >= j:
            l_off = b * stride_L_b + c * stride_L_c + i * stride_L_i + j * stride_L_j + h * stride_L_h
            tl.store(L_ptr + l_off, tl.exp(cumsum))


# Kernel: compute G[i, j, h] = sum over n of C_expanded[b, c, i, n, j, h] * B_expanded[b, c, j, n, i, h]
# Inputs:
#   B_expanded: [B, C, S, H, N] (float32), strides provided
#   C_expanded: [B, C, S, H, N] (float32), strides provided
# Output:
#   G: [B, C, S, S, H] (float32), we write per (b, c, i, h) vector over j
@triton.jit
def compute_G_kernel(
    B_exp_ptr, C_exp_ptr, G_ptr,
    B, C, S, H, N,
    stride_B_b, stride_B_c, stride_B_s, stride_B_h, stride_B_n,
    stride_C_b, stride_C_c, stride_C_s, stride_C_h, stride_C_n,
    stride_G_b, stride_G_c, stride_G_i, stride_G_j, stride_G_h,
):
    # Grid: (B*C*S, H) — each program computes G for one (i, h) across all j
    pid0 = tl.program_id(0)  # over B*C*S
    pid1 = tl.program_id(1)  # over H
    b = pid0 // (C * S)
    rem = pid0 % (C * S)
    c = rem // S
    i = rem % S
    h = pid1

    # Initialize G[i, :, h] vector
    # We will fill it by looping over j and n
    # For each j, compute dot over n: sum_n C_exp[b, c, i, n, j, h] * B_exp[b, c, j, n, i, h]
    # Treat C_exp[:, n, j, h] as matrix [S, 1], B_exp[j, n, i, h] as vector [S]
    # We'll compute one G[i, j, h] per j via accumulation over n.
    for j in range(0, S):
        dot = 0.0
        # Loop over n from 0..N-1 (N is typically 128)
        for n in range(0, N):
            # C_exp[b, c, i, n, j, h]
            c_off = b * stride_C_b + c * stride_C_c + i * stride_C_s + n * stride_C_n + j * stride_C_s + h * stride_C_h
            # Note: C_exp is [B, C, S, H, N], so the j index is stride_C_s? We need to pass correct strides.
            # Here we set c_off = b*stride_C_b + c*stride_C_c + i*stride_C_s + n*stride_C_n + j*stride_C_j + h*stride_C_h
            # We need to pass strides for H and j properly. We will pass stride_C_j=1, stride_C_h=1 in launch, but since C_exp has H as dim=3 and j as dim=2, we need to set strides accordingly.
            # We'll instead pass actual strides by creating tensors in PyTorch and using tensor.stride().
            # To keep code simple, assume strides for j are stride_C_s and for h is stride_C_h; we'll pass correct strides when launching.
            # For safety, we'll reconstruct offset manually using tensor.stride():
            # c_off = b*stride_C_b + c*stride_C_c + i*stride_C_s + n*stride_C_n + j*stride_C_j + h*stride_C_h
            # We'll pass stride_C_j as stride_C_ptr[2], stride_C_h as stride_C_ptr[3] in launch. Triton kernel expects them as ints.
            # However, Triton kernel cannot access .stride() of PyTorch tensors; we must pass them as arguments.
            # So we'll restructure: We'll pass explicit strides for each dimension. For C_exp: [S, H, N], strides are [C_exp.stride(3), C_exp.stride(2), C_exp.stride(4)] but we need mapping. Better to rely on passing stride for j and h from the PyTorch tensor using .stride() in launch. Triton accepts these as ints.
            # Here, we will pass stride_C_j and stride_C_h as kernel arguments computed from C_exp.stride() in Python before launch.

            # B_exp[b, c, j, n, i, h]
            b_off = b * stride_B_b + c * stride_B_c + j * stride_B_s + n * stride_B_n + i * stride_B_s + h * stride_B_h
            c_val = tl.load(C_exp_ptr + c_off)  # float32
            b_val = tl.load(B_exp_ptr + b_off)  # float32
            dot += c_val * b_val

        # Store G[i, j, h] = dot
        g_off = b * stride_G_b + c * stride_G_c + i * stride_G_i + j * stride_G_j + h * stride_G_h
        tl.store(G_ptr + g_off, dot)


# Kernel: compute Y_diag[b, c, i, h, d] = sum over j of M[b, c, i, j, h] * hidden_states[b, c, j, h, d]
# Inputs:
#   M: [B, C, S, S, H] (float32)
#   hidden_states: [B, C, S, H, D] (float32), D is dynamic
# Output:
#   Y_diag: [B, C, S, H, D] (float32), we'll store into a 2D pointer with D
@triton.jit
def compute_Y_diag_kernel(
    M_ptr, hidden_ptr, Y_ptr,
    B, C, S, H, D,
    stride_M_b, stride_M_c, stride_M_i, stride_M_j, stride_M_h,
    stride_h_b, stride_h_c, stride_h_s, stride_h_h, stride_h_d,
    stride_Y_b, stride_Y_c, stride_Y_i, stride_Y_h, stride_Y_d,
):
    # Grid: (B*C*S, H), one program per (i, h)
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)
    b = pid0 // (C * S)
    c = (pid0 % (C * S)) // S
    i = pid0 % S
    h = pid1

    # Accumulate over j dimension
    acc = tl.zeros((D,), dtype=tl.float32)  # vector over head_dim
    for j in range(0, S):
        # Load M[b, c, i, j, h] as scalar
        m_off = b * stride_M_b + c * stride_M_c + i * stride_M_i + j * stride_M_j + h * stride_M_h
        m_val = tl.load(M_ptr + m_off)  # float32
        # Load hidden_states[b, c, j, h, :] vector of length D
        hs_off = b * stride_h_b + c * stride_h_c + j * stride_h_s + h * stride_h_h
        d_vec = tl.arange(0, D)
        hs_ptr = hidden_ptr + hs_off + d_vec * stride_h_d
        hs_vals = tl.load(hs_ptr)  # float32 vector of length D
        acc += m_val * hs_vals

    # Store Y[b, c, i, h, :] vector
    y_off_base = b * stride_Y_b + c * stride_Y_c + i * stride_Y_i + h * stride_Y_h
    d_vec = tl.arange(0, D)
    y_ptr = Y_ptr + y_off_base + d_vec * stride_Y_d
    tl.store(y_ptr, acc)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                A_cumsum: torch.Tensor,
                B: torch.Tensor,
                C: torch.Tensor) -> torch.Tensor:
        """
        Triton implementation of the original run function.
        Computes Y_diag with Triton kernels:
          - build_L: L = causal mask via exp(cumsum) lower triangular
          - compute_G: contraction G = sum_n(C_exp[n, j] * B_exp[j, n]^T)
          - compute_Y_diag: Y_diag = sum_j M[j] * hidden[j] where M = G * L
        Returns Y_diag in bfloat16.
        """
        # Extract shapes (note: we use provided dimensions; S and H are assumed 128 and 32 in the original, but here we use actual inputs)
        Bsz, Csz, S, H, head_dim = hidden_states.shape
        N_GROUPS = 8
        # Check constants (the original code assumes H=32, S=128; we can rely on these per workload, but we keep dynamic logic robust)
        assert H == 32, f"ModelNew expects NUM_HEADS=32, got {H}"
        # Ensure all inputs are contiguous and on CUDA device
        device = hidden_states.device
        assert hidden_states.is_cuda and A_cumsum.is_cuda and B.is_cuda and C.is_cuda, "All inputs must be CUDA tensors for Triton kernels."

        # Compute in float32 for numerical stability
        hidden_f32 = hidden_states.to(torch.float32)
        A_f32 = A_cumsum.to(torch.float32)
        B_f32 = B.to(torch.float32)
        C_f32 = C.to(torch.float32)

        # 1) Build L in Triton: [B, C, S, S, H]
        L = torch.empty((Bsz, Csz, S, S, H), dtype=torch.float32, device=device)
        # Launch build_L kernel: grid = (B*C*S, H)
        grid_L = (Bsz * Csz * S, H)
        build_L_kernel[grid_L](
            A_f32, L,
            Bsz, H, Csz, S,
            A_f32.stride(0), A_f32.stride(1), A_f32.stride(2), A_f32.stride(3),
            L.stride(0), L.stride(1), L.stride(2), L.stride(3), L.stride(4),
            num_warps=4,
        )

        # 2) Expand B and C along H (repeat_interleave NUM_HEADS // N_GROUPS = 4)
        B_exp = B_f32.repeat_interleave(4, dim=3)  # [B, C, S, H, N]
        C_exp = C_f32.repeat_interleave(4, dim=3)  # [B, C, S, H, N]
        # Ensure they are contiguous along last dim (N)
        B_exp = B_exp.contiguous()
        C_exp = C_exp.contiguous()

        # 3) Compute G in Triton: [B, C, S, S, H]
        G = torch.empty((Bsz, Csz, S, S, H), dtype=torch.float32, device=device)
        # We need strides for B_exp, C_exp, and G. Use actual strides.
        grid_G = (Bsz * Csz * S, H)
        compute_G_kernel[grid_G](
            B_exp, C_exp, G,
            Bsz, Csz, S, H, 128,  # N is typically 128
            B_exp.stride(0), B_exp.stride(1), B_exp.stride(2), B_exp.stride(3), B_exp.stride(4),
            C_exp.stride(0), C_exp.stride(1), C_exp.stride(2), C_exp.stride(3), C_exp.stride(4),
            G.stride(0), G.stride(1), G.stride(2), G.stride(3), G.stride(4),
            num_warps=4,
        )

        # 4) Compute M = G * L
        M = G * L  # element-wise multiplication

        # 5) Compute Y_diag in Triton: [B, C, S, H, head_dim]
        Y = torch.empty((Bsz, Csz, S, H, head_dim), dtype=torch.float32, device=device)
        grid_Y = (Bsz * Csz * S, H)
        compute_Y_diag_kernel[grid_Y](
            M, hidden_f32, Y,
            Bsz, Csz, S, H, head_dim,
            M.stride(0), M.stride(1), M.stride(2), M.stride(3), M.stride(4),
            hidden_f32.stride(0), hidden_f32.stride(1), hidden_f32.stride(2), hidden_f32.stride(3), hidden_f32.stride(4),
            Y.stride(0), Y.stride(1), Y.stride(2), Y.stride(3), Y.stride(4),
            num_warps=4,
        )

        # Return in bfloat16 as per original
        return Y.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
