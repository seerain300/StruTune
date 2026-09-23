import torch

# Triton imports
import triton
import triton.language as tl


# Kernel 1: Compute L = exp(cumsum(A, along target j)) with lower-triangular mask (including diagonal), i.e., j <= i.
# A: [B, H, C, S] where C = num_chunks, S = chunk_size, H = num_heads
# L: [B, C, S, S, H]
@triton.jit
def cumsum_mask_exp_kernel(
    A_ptr, L_ptr,
    Bsz, Csz, S, H,
    stride_A_b, stride_A_h, stride_A_c, stride_A_s,
    stride_L_b, stride_L_c, stride_L_i, stride_L_j, stride_L_h,
    num_warps=4, num_stages=2
):
    # One program per (b, c, i)
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)

    # Initialize cumsum vector of length S
    cumsum = tl.zeros((S,), dtype=tl.float32)

    # For each j from 0 to S-1
    # Note: Triton supports while loops; we use them here.
    j = 0
    while j < S:
        # Load A[b, :, c, i, j] which is a vector across H
        # We need to compute pointer for A[b, h, c, i, j]
        # Using a small loop over h to load each element
        A_vals = tl.zeros((H,), dtype=tl.float32)
        h = 0
        while h < H:
            ptr = A_ptr + b * stride_A_b + h * stride_A_h + c * stride_A_c + i * stride_A_s + j
            A_vals[h] = tl.load(ptr)
            h += 1

        # Apply lower-triangular mask (include diagonal): j <= i -> keep, else 0
        # If j > i, set A_vals to 0
        if j > i:
            A_vals = tl.zeros((H,), dtype=tl.float32)

        # Update cumsum for each h: cumsum[h] += A_vals[h]
        h = 0
        while h < H:
            cumsum[h] += A_vals[h]
            h += 1

        # Exponentiate and store to L[b, c, i, j, :]
        h = 0
        while h < H:
            val = tl.exp(cumsum[h])
            ptr_L = L_ptr + b * stride_L_b + c * stride_L_c + i * stride_L_i + j * stride_L_j + h * stride_L_h
            tl.store(ptr_L, val)
            h += 1

        j += 1


# Kernel 2: Contract B and C to form G: G[i, j, h] = sum over groups g and state_size k of C[b, c, i, g, k] * B[b, c, j, g, k]
# We avoid materializing B_expanded by folding repeat_interleave: NUM_HEADS = H = 32, N_GROUPS = G_const = 8, so repeat_factor = H // G_const = 4.
# Inputs:
#   B: [B, C, S, G_const, K] where K is state_dim (head_dim in original). We assume K == H for simplicity in this context (matches original usage).
#   C: [B, C, S, G_const, K]
# Outputs:
#   G: [B, C, S, S, H]
@triton.jit
def contract_BC_to_G_kernel(
    B_ptr, C_ptr, G_ptr,
    Bsz, Csz, S, H, G_const, K,
    stride_B_b, stride_B_c, stride_B_s, stride_B_g, stride_B_k,
    stride_C_b, stride_C_c, stride_C_s, stride_C_g, stride_C_k,
    stride_G_b, stride_G_c, stride_G_i, stride_G_j, stride_G_h,
    num_warps=4, num_stages=2
):
    # One program per (b, c, i) and store G[i, :, :, :]
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)

    # Accumulator for G[i, j, h] for all j, h
    # We will fill G for all j and h by iterating over j and computing per j.
    # Launch a nested grid over j and h. However, Triton allows only one grid; we instead compute per j and h inside one program by iterating j and h.
    # We implement a loop over j from 0..S-1, and for each j, compute G[i, j, :] and store it.

    j = 0
    while j < S:
        # Compute G[i, j, :] across all heads h
        # G[i, j, h] = sum over g in [0..G_const-1] and k in [0..K-1] of C[b, c, i, g, k] * B[b, c, j, g, k]
        G_vals = tl.zeros((H,), dtype=tl.float32)
        h = 0
        while h < H:
            # Initialize G_vals[h] = 0
            G_vals[h] = 0.0
            g = 0
            while g < G_const:
                k = 0
                # Accumulate over k: K is assumed equal to H, but we pass K as a runtime param; we loop over k
                while k < K:
                    # Load B[b, c, j, g, k] and C[b, c, i, g, k]
                    ptr_B = B_ptr + b * stride_B_b + c * stride_B_c + j * stride_B_s + g * stride_B_g + k * stride_B_k
                    ptr_C = C_ptr + b * stride_C_b + c * stride_C_c + i * stride_C_s + g * stride_C_g + k * stride_C_k
                    b_val = tl.load(ptr_B)
                    c_val = tl.load(ptr_C)
                    G_vals[h] += b_val * c_val
                    k += 1
                g += 1
            h += 1

        # Store G[i, j, :]
        h = 0
        while h < H:
            ptr_G = G_ptr + b * stride_G_b + c * stride_G_c + i * stride_G_i + j * stride_G_j + h * stride_G_h
            tl.store(ptr_G, G_vals[h])
            h += 1
        j += 1


# Kernel 3: Final contraction to compute Y_diag = sum_j M[b, c, i, j, h] * hidden_states[b, c, j, h, d]
# Here, M = G * L elementwise. We compute M on-the-fly from G and L and contract with hidden.
# Inputs:
#   G: [B, C, S, S, H]
#   L: [B, C, S, S, H]
#   hidden_states: [B, C, S, H, D]
# Output:
#   Y_diag: [B, C, S, H, D] (compute in float32, convert to bfloat16 in forward before store)
@triton.jit
def contract_M_with_hidden_kernel(
    G_ptr, L_ptr, hidden_ptr, Y_ptr,
    Bsz, Csz, S, H, D,
    stride_G_b, stride_G_c, stride_G_i, stride_G_j, stride_G_h,
    stride_L_b, stride_L_c, stride_L_i, stride_L_j, stride_L_h,
    stride_hs_b, stride_hs_c, stride_hs_s, stride_hs_h, stride_hs_d,
    stride_Y_b, stride_Y_c, stride_Y_i, stride_Y_h, stride_Y_d,
    num_warps=4, num_stages=2
):
    # Grid over (b, c, i, h, d_tile). Use d_tile = 1 since D is small (128), but we can keep it general.
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)
    h = tl.program_id(3)
    d_tile = tl.program_id(4)

    # Accumulator for Y_diag[b, c, i, h, d] across all j
    # We'll compute Y for a single d index per program (d_tile = 0 typically), since D is passed as constexpr and we want one output per program for simplicity.
    # However, Triton expects the grid to cover all D; we can implement accumulation over j for each d index. To keep it simple and fast, we assume D is small and handle single d per program.

    d = d_tile  # For single-d per program; if D > 1, launch multiple programs over d axis.

    # Accumulator scalar
    acc = 0.0

    j = 0
    while j < S:
        # Load G[b, c, i, j, h] and L[b, c, i, j, h]
        ptr_G = G_ptr + b * stride_G_b + c * stride_G_c + i * stride_G_i + j * stride_G_j + h * stride_G_h
        ptr_L = L_ptr + b * stride_L_b + c * stride_L_c + i * stride_L_i + j * stride_L_j + h * stride_L_h
        G_val = tl.load(ptr_G)
        L_val = tl.load(ptr_L)

        M_val = G_val * L_val  # elementwise

        # Load hidden_states[b, c, j, h, d]
        ptr_hs = hidden_ptr + b * stride_hs_b + c * stride_hs_c + j * stride_hs_s + h * stride_hs_h + d * stride_hs_d
        hs_val = tl.load(ptr_hs)

        acc += M_val * hs_val
        j += 1

    # Store result to Y[b, c, i, h, d]
    ptr_Y = Y_ptr + b * stride_Y_b + c * stride_Y_c + i * stride_Y_i + h * stride_Y_h + d * stride_Y_d
    tl.store(ptr_Y, acc)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, A_cumsum: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        """
        Triton-optimized forward that:
          1) computes L via cumsum_mask_exp_kernel
          2) computes G via contract_BC_to_G_kernel
          3) computes Y_diag via contract_M_with_hidden_kernel
        All computation is done inside Triton kernels; no torch ops in host code except for tensor creation/allocations.
        Returns Y_diag in bfloat16 to match original behavior.
        """
        # Ensure CUDA tensors and compute in float32
        assert hidden_states.is_cuda and A_cumsum.is_cuda and B.is_cuda and C.is_cuda, "Inputs must be CUDA tensors"
        Bsz = hidden_states.shape[0]
        Csz = hidden_states.shape[1]
        S = hidden_states.shape[2]
        H = hidden_states.shape[3]
        D = hidden_states.shape[4]  # head_dim

        # A_cumsum shape: [Bsz, H, Csz, S]
        # Constants used in kernels
        NUM_HEADS = 32  # must match original
        N_GROUPS = 8    # must match original
        REPEAT_FACTOR = NUM_HEADS // N_GROUPS  # 4

        # Allocate outputs in float32
        L = torch.empty((Bsz, Csz, S, S, H), dtype=torch.float32, device=hidden_states.device)
        G = torch.empty((Bsz, Csz, S, S, H), dtype=torch.float32, device=hidden_states.device)
        Y_diag = torch.empty((Bsz, Csz, S, H, D), dtype=torch.float32, device=hidden_states.device)

        # Launch kernel 1: compute L
        cumsum_mask_exp_kernel[(Bsz, Csz, S)](
            A_cumsum, L,
            Bsz, Csz, S, H,
            A_cumsum.stride(0), A_cumsum.stride(1), A_cumsum.stride(2), A_cumsum.stride(3),
            L.stride(0), L.stride(1), L.stride(2), L.stride(3), L.stride(4),
            num_warps=4, num_stages=2
        )

        # Launch kernel 2: compute G from B and C
        # Note: We assume B and C have shape [B, C, S, N_GROUPS, K]. In the original, K == H (head_dim). We pass K=H.
        G_const = N_GROUPS
        K = H  # assumption: head_dim equals state_size used in contraction
        contract_BC_to_G_kernel[(Bsz, Csz, S)](
            B, C, G,
            Bsz, Csz, S, H, G_const, K,
            B.stride(0), B.stride(1), B.stride(2), B.stride(3), B.stride(4),
            C.stride(0), C.stride(1), C.stride(2), C.stride(3), C.stride(4),
            G.stride(0), G.stride(1), G.stride(2), G.stride(3), G.stride(4),
            num_warps=4, num_stages=2
        )

        # Launch kernel 3: compute Y_diag = sum_j (G * L) * hidden over j
        # We need to pass a grid that covers all d=0..D-1. Use a 5D grid: (Bsz, Csz, S, H, ceil(D/BLOCK_D))
        # For simplicity, assume D <= 128 and use BLOCK_D=1 (one d per program). If D > 128, adjust BLOCK_D to 1 (i.e., one program per d).
        BLOCK_D = 1
        grid = (Bsz, Csz, S, H, triton.cdiv(D, BLOCK_D))
        contract_M_with_hidden_kernel[grid](
            G, L, hidden_states, Y_diag,
            Bsz, Csz, S, H, D,
            G.stride(0), G.stride(1), G.stride(2), G.stride(3), G.stride(4),
            L.stride(0), L.stride(1), L.stride(2), L.stride(3), L.stride(4),
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2), hidden_states.stride(3), hidden_states.stride(4),
            Y_diag.stride(0), Y_diag.stride(1), Y_diag.stride(2), Y_diag.stride(3), Y_diag.stride(4),
            num_warps=4, num_stages=2
        )

        # Return in bfloat16 as original code does
        return Y_diag.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
