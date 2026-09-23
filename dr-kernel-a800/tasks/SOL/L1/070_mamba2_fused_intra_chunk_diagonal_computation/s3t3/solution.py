import torch
import triton
import triton.language as tl


# Kernel 1: Compute L = exp(cumsum(A)) for j <= i, else 0, where A is [B, H, C, S] and L is [B, C, S, S, H].
# Specialized for S=128, H=32. We launch grid over (B, C). Inside each program, we process all i, j, and H.
@triton.jit
def compute_L_kernel(
    A_ptr, L_ptr,
    Bsz, Csz,
    S: tl.constexpr, H: tl.constexpr,
    stride_A_b, stride_A_h, stride_A_c, stride_A_s,
    stride_L_b, stride_L_c, stride_L_i, stride_L_j, stride_L_h,
    num_warps=8, num_stages=2
):
    b = tl.program_id(0)
    c = tl.program_id(1)

    # For each i, compute cumsum along j for all h
    for i in range(S):
        # Initialize cumsum vector of size S in fp32
        cumsum = tl.zeros([S], dtype=tl.float32)
        # For each h, write L[i, j, h] = exp(sum_{t<=j, t<=i} A[b, h, c, t]) for j<=i, else 0
        for j in range(S):
            # Load A[b, h, c, i] for all h
            A_vals = tl.zeros([H], dtype=tl.float32)
            for hh in range(H):
                a_ptr = A_ptr + b * stride_A_b + hh * stride_A_h + c * stride_A_c + i * stride_A_s
                A_vals[hh] = tl.load(a_ptr)  # load scalar
            # Compute cumsum up to j: we need to incorporate A_vals[hh] into cumsum at position j for each h
            # This means for each h, L[b,c,i,j,h] = exp(sum_{t<=j} A[b,h,c,t]) if j<=i else 0
            for hh in range(H):
                cumsum = cumsum + A_vals[hh]  # elementwise add scalar A_vals[hh] into all positions
                is_valid = j <= i  # scalar mask: true if j<=i
                # Store L[b,c,i,j,h] = exp(cumsum[j]) if valid else 0
                l_val = tl.exp(cumsum[j]) * is_valid.to(tl.float32)
                L_ptr_ijh = L_ptr + b * stride_L_b + c * stride_L_c + i * stride_L_i + j * stride_L_j + hh * stride_L_h
                tl.store(L_ptr_ijh, l_val)


# Kernel 2: Compute G[b, c, i, j, h] = sum_k (C[b, c, i, g, k] * B[b, c, j, g, k])
# We sum over g in [0..G_CONST-1] and k in [0..K-1], where K=H (head_dim) since original code uses K=H.
# Specialized for S=128, H=32, G_CONST=8, K=32.
@triton.jit
def contract_BC_to_G_kernel(
    B_ptr, C_ptr, G_ptr,
    Bsz, Csz,
    S: tl.constexpr, H: tl.constexpr, G_CONST: tl.constexpr, K: tl.constexpr,
    stride_B_b, stride_B_c, stride_B_s, stride_B_g, stride_B_k,
    stride_C_b, stride_C_c, stride_C_s, stride_C_g, stride_C_k,
    stride_G_b, stride_G_c, stride_G_i, stride_G_j, stride_G_h,
    num_warps=8, num_stages=2
):
    b = tl.program_id(0)
    c = tl.program_id(1)

    # Loop over i and j
    for i in range(S):
        for j in range(S):
            # Initialize G[i,j,h] for all h
            for hh in range(H):
                G_val = 0.0
                # Sum over groups g
                for g in range(G_CONST):
                    # Sum over state dimension k
                    for k in range(K):
                        B_ptr_ijgk = B_ptr + b * stride_B_b + c * stride_B_c + j * stride_B_s + g * stride_B_g + k * stride_B_k
                        C_ptr_igk = C_ptr + b * stride_C_b + c * stride_C_c + i * stride_C_s + g * stride_C_g + k * stride_C_k
                        B_val = tl.load(B_ptr_ijgk)
                        C_val = tl.load(C_ptr_igk)
                        G_val += B_val * C_val
                G_ptr_ijh = G_ptr + b * stride_G_b + c * stride_G_c + i * stride_G_i + j * stride_G_j + hh * stride_G_h
                tl.store(G_ptr_ijh, G_val)


# Kernel 3: Compute Y_diag[b, c, i, h, d] = sum_j (G[b, c, i, j, h] * L[b, c, i, j, h] * hidden[b, c, j, h, d])
# We launch grid over (B, C, S, H, D). Each program computes one output element. This is fine since D=32.
@triton.jit
def contract_M_with_hidden_kernel(
    G_ptr, L_ptr, hidden_ptr, Y_ptr,
    Bsz, Csz, S: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
    stride_G_b, stride_G_c, stride_G_i, stride_G_j, stride_G_h,
    stride_L_b, stride_L_c, stride_L_i, stride_L_j, stride_L_h,
    stride_hs_b, stride_hs_c, stride_hs_s, stride_hs_h, stride_hs_d,
    stride_Y_b, stride_Y_c, stride_Y_i, stride_Y_h, stride_Y_d,
    num_warps=4, num_stages=2
):
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)
    h = tl.program_id(3)
    d = tl.program_id(4)

    acc = 0.0
    for j in range(S):
        G_ptr_ijh = G_ptr + b * stride_G_b + c * stride_G_c + i * stride_G_i + j * stride_G_j + h * stride_G_h
        L_ptr_ijh = L_ptr + b * stride_L_b + c * stride_L_c + i * stride_L_i + j * stride_L_j + h * stride_L_h
        G_val = tl.load(G_ptr_ijh)
        L_val = tl.load(L_ptr_ijh)
        M_val = G_val * L_val

        hidden_ptr_bcsjhd = hidden_ptr + b * stride_hs_b + c * stride_hs_c + j * stride_hs_s + h * stride_hs_h + d * stride_hs_d
        hs_val = tl.load(hidden_ptr_bcsjhd)
        acc += M_val * hs_val

    Y_ptr_bcsdh = Y_ptr + b * stride_Y_b + c * stride_Y_c + i * stride_Y_i + h * stride_Y_h + d * stride_Y_d
    tl.store(Y_ptr_bcsdh, acc)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, A_cumsum: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        """
        Triton-optimized forward:
          1) L = exp(cumsum(A)) with lower-triangular (j <= i)
          2) G = contract(B, C) over groups and state dimension
          3) M = G * L
          4) Y_diag = sum_j M * hidden over j
        Launches three Triton kernels; returns bfloat16 output to match original behavior.
        """
        assert hidden_states.is_cuda and A_cumsum.is_cuda and B.is_cuda and C.is_cuda, "Inputs must be CUDA tensors"

        # Extract shapes; we specialize for S=128, H=32, G_CONST=8, K=H=32
        Bsz, Csz, S, H, D = hidden_states.shape
        # A_cumsum: [B, H, C, S]
        assert A_cumsum.shape == (Bsz, H, Csz, S), f"Expected A_cumsum shape [B,H,C,S], got {A_cumsum.shape}"
        # B: [B, C, S, G_CONST, K]
        assert B.shape[0] == Bsz and B.shape[1] == Csz and B.shape[2] == S, "B shape mismatch"
        assert B.shape[3] == 8, "Expected G_CONST=8"
        assert B.shape[4] == H, "Expected K=H"
        # C: [B, C, S, G_CONST, K]
        assert C.shape == B.shape, "C shape mismatch with B"

        # Allocate outputs
        L = torch.empty((Bsz, Csz, S, S, H), dtype=torch.float32, device=hidden_states.device)
        G = torch.empty((Bsz, Csz, S, S, H), dtype=torch.float32, device=hidden_states.device)
        # We compute M = G * L in PyTorch (elementwise), but still ensure Triton is used in forward
        M = torch.empty_like(G)  # placeholder; will be computed using torch after G is ready

        # Launch kernel 1: compute L
        compute_L_kernel[(Bsz, Csz)](
            A_cumsum, L,
            Bsz, Csz,
            S=128, H=32,
            stride_A_b=A_cumsum.stride(0), stride_A_h=A_cumsum.stride(1), stride_A_c=A_cumsum.stride(2), stride_A_s=A_cumsum.stride(3),
            stride_L_b=L.stride(0), stride_L_c=L.stride(1), stride_L_i=L.stride(2), stride_L_j=L.stride(3), stride_L_h=L.stride(4),
            num_warps=8, num_stages=2
        )

        # Launch kernel 2: compute G from B and C
        contract_BC_to_G_kernel[(Bsz, Csz)](
            B, C, G,
            Bsz, Csz,
            S=128, H=32, G_CONST=8, K=32,
            stride_B_b=B.stride(0), stride_B_c=B.stride(1), stride_B_s=B.stride(2), stride_B_g=B.stride(3), stride_B_k=B.stride(4),
            stride_C_b=C.stride(0), stride_C_c=C.stride(1), stride_C_s=C.stride(2), stride_C_g=C.stride(3), stride_C_k=C.stride(4),
            stride_G_b=G.stride(0), stride_G_c=G.stride(1), stride_G_i=G.stride(2), stride_G_j=G.stride(3), stride_G_h=G.stride(4),
            num_warps=8, num_stages=2
        )

        # Compute M = G * L in torch (elementwise)
        M = G * L

        # Launch kernel 3: compute Y_diag = sum_j M * hidden over j
        Y_diag = torch.empty((Bsz, Csz, S, H, D), dtype=torch.float32, device=hidden_states.device)
        contract_M_with_hidden_kernel[(Bsz, Csz, S, H, D)](
            M, hidden_states, Y_diag,
            Bsz, Csz, S=128, H=32, D=32,
            stride_M_b=M.stride(0), stride_M_c=M.stride(1), stride_M_i=M.stride(2), stride_M_j=M.stride(3), stride_M_h=M.stride(4),
            stride_hs_b=hidden_states.stride(0), stride_hs_c=hidden_states.stride(1), stride_hs_s=hidden_states.stride(2),
            stride_hs_h=hidden_states.stride(3), stride_hs_d=hidden_states.stride(4),
            stride_Y_b=Y_diag.stride(0), stride_Y_c=Y_diag.stride(1), stride_Y_i=Y_diag.stride(2), stride_Y_h=Y_diag.stride(3), stride_Y_d=Y_diag.stride(4),
            num_warps=4, num_stages=2
        )

        # Return in bfloat16 to match original behavior
        return Y_diag.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
