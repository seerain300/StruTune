import torch
import triton
import triton.language as tl


# Triton kernel 1: Compute L = exp(cumsum(A, along target j)) with lower-triangular mask (include diagonal), i.e., j <= i.
# Inputs:
#   A_expanded: [B, H, C, S, S] (float32)
#   L: [B, C, S, S, H] (float32)
# Grid: (Bsz, Csz, H) -> one program per (b, c, h)
@triton.jit
def cumsum_mask_exp_kernel(
    A_ptr, L_ptr,
    Bsz, Csz, S, H,
    stride_A_b, stride_A_h, stride_A_c, stride_A_i, stride_A_j,
    stride_L_b, stride_L_c, stride_L_i, stride_L_j, stride_L_h,
    num_warps=4, num_stages=2
):
    b = tl.program_id(0)
    c = tl.program_id(1)
    h = tl.program_id(2)

    # Loop over source i and compute cumsum for target j
    i = 0
    while i < S:
        # cumsum vector for this (b, c, h, i)
        cumsum = tl.zeros((S,), dtype=tl.float32)

        j = 0
        while j < S:
            # Load A[b, h, c, i, j]
            ptr_A = A_ptr + b * stride_A_b + h * stride_A_h + c * stride_A_c + i * stride_A_i + j * stride_A_j
            val = tl.load(ptr_A)

            # Apply lower-triangular mask (include diagonal): j <= i
            # If j > i, set val = 0
            # Triton supports scalar if; we compare j and i as scalars
            if j > i:
                val = 0.0

            # Update cumsum
            cumsum += val

            # Store exp(cumsum) to L[b, c, i, j, h]
            ptr_L = L_ptr + b * stride_L_b + c * stride_L_c + i * stride_L_i + j * stride_L_j + h * stride_L_h
            tl.store(ptr_L, tl.exp(cumsum))

            j += 1
        i += 1


# Triton kernel 2: Contract B and C to form G: G[i, j, h] = sum over groups g and state_size k of C[b, c, i, g, k] * B[b, c, j, g, k]
# Assumes K == H (head_dim). We fold repeat_interleave by repeat_factor = NUM_HEADS // N_GROUPS = 4.
# Inputs:
#   B: [B, C, S, G, K]
#   C: [B, C, S, G, K]
# Output:
#   G: [B, C, S, S, H]
@triton.jit
def contract_BC_to_G_kernel(
    B_ptr, C_ptr, G_ptr,
    Bsz, Csz, S, H, G_CONST: tl.constexpr, K: tl.constexpr,
    stride_B_b, stride_B_c, stride_B_s, stride_B_g, stride_B_k,
    stride_C_b, stride_C_c, stride_C_s, stride_C_g, stride_C_k,
    stride_G_b, stride_G_c, stride_G_i, stride_G_j, stride_G_h,
    num_warps=4, num_stages=2
):
    # Grid: (Bsz, Csz, S) -> one program per (b, c, i), compute G[i, :, :]
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)

    repeat_factor = H // G_CONST  # 4

    # For each target j, compute G[i, j, :] across all heads
    for j in range(S):
        # Accumulator for all heads
        G_vals = tl.zeros((H,), dtype=tl.float32)
        # Loop over groups and state k
        for g in range(G_CONST):
            for k in range(K):
                # For each head h_rep, accumulate B[b, c, j, g, k] * C[b, c, i, g, k]
                for h_rep in range(H):
                    # Compute pointers
                    ptr_B = B_ptr + b * stride_B_b + c * stride_B_c + j * stride_B_s + g * stride_B_g + k * stride_B_k
                    ptr_C = C_ptr + b * stride_C_b + c * stride_C_c + i * stride_C_s + g * stride_C_g + k * stride_C_k
                    b_val = tl.load(ptr_B)
                    c_val = tl.load(ptr_C)
                    G_vals[h_rep] += b_val * c_val
        # Store G[i, j, :]
        for h in range(H):
            ptr_G = G_ptr + b * stride_G_b + c * stride_G_c + i * stride_G_i + j * stride_G_j + h * stride_G_h
            tl.store(ptr_G, G_vals[h])


# Triton kernel 3: Final contraction to compute Y_diag = sum_j M[b, c, i, j, h] * hidden_states[b, c, j, h, d]
# M = G * L elementwise. We compute M on-the-fly from G and L and contract with hidden.
# Inputs:
#   G: [B, C, S, S, H]
#   L: [B, C, S, S, H]
#   hidden_states: [B, C, S, H, D]
# Output:
#   Y_diag: [B, C, S, H, D] (float32)
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
    # Grid: (Bsz, Csz, S, H, D) one program per element
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)
    h = tl.program_id(3)
    d = tl.program_id(4)

    acc = 0.0
    for j in range(S):
        ptr_G = G_ptr + b * stride_G_b + c * stride_G_c + i * stride_G_i + j * stride_G_j + h * stride_G_h
        ptr_L = L_ptr + b * stride_L_b + c * stride_L_c + i * stride_L_i + j * stride_L_j + h * stride_L_h
        G_val = tl.load(ptr_G)
        L_val = tl.load(ptr_L)
        M_val = G_val * L_val

        ptr_hs = hidden_ptr + b * stride_hs_b + c * stride_hs_c + j * stride_hs_s + h * stride_hs_h + d * stride_hs_d
        hs_val = tl.load(ptr_hs)
        acc += M_val * hs_val

    ptr_Y = Y_ptr + b * stride_Y_b + c * stride_Y_c + i * stride_Y_i + h * stride_Y_h + d * stride_Y_d
    tl.store(ptr_Y, acc)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, A_cumsum: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        """
        Triton-optimized forward that:
          1) computes L via cumsum_mask_exp_kernel
          2) computes G via contract_BC_to_G_kernel
          3) computes Y_diag via contract_M_with_hidden_kernel
        All heavy computation is done inside Triton kernels; no torch ops in host code for elementwise operations.
        Returns Y_diag in bfloat16 to match original behavior.
        """
        assert hidden_states.is_cuda and A_cumsum.is_cuda and B.is_cuda and C.is_cuda, "Inputs must be CUDA tensors"

        Bsz, Csz, S, H


def run(*args):
    return ModelNew()(*args)
