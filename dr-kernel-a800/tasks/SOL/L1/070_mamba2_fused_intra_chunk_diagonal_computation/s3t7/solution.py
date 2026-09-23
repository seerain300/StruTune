import torch
import triton
import triton.language as tl


@triton.jit
def compute_L_exp_cumsum_kernel(
    A_ptr, L_ptr,
    Bsz, Csz,
    stride_A_b, stride_A_h, stride_A_c, stride_A_s,
    stride_L_b, stride_L_c, stride_L_i, stride_L_j, stride_L_h,
    S: tl.constexpr,  # chunk_size = 128
    H: tl.constexpr,  # num_heads = 32
    num_warps=4, num_stages=2
):
    # Grid: (Bsz, Csz)
    b = tl.program_id(0)
    c = tl.program_id(1)

    # For each i, build cumsum across j and write L[i, j, h] with lower-triangular mask (i >= j)
    for i in range(S):
        for hh in range(H):
            # cumsum across j for head hh
            s_j = 0.0
            for j in range(S):
                a_off = b * stride_A_b + hh * stride_A_h + c * stride_A_c + j * stride_A_s
                a_val = tl.load(A_ptr + a_off)
                s_j += a_val
            # Write L[i, j, hh] = exp(s_j) for j <= i; zero otherwise
            for j in range(S):
                l_off = b * stride_L_b + c * stride_L_c + i * stride_L_i + j * stride_L_j + hh * stride_L_h
                if j <= i:
                    tl.store(L_ptr + l_off, tl.exp(s_j))
                else:
                    tl.store(L_ptr + l_off, 0.0)


@triton.jit
def contract_BC_to_G_kernel(
    B_ptr, C_ptr, G_ptr,
    Bsz, Csz,
    stride_B_b, stride_B_c, stride_B_s, stride_B_g, stride_B_k,
    stride_C_b, stride_C_c, stride_C_s, stride_C_g, stride_C_k,
    stride_G_b, stride_G_c, stride_G_i, stride_G_j, stride_G_h,
    S: tl.constexpr,  # 128
    G_CONST: tl.constexpr,  # 8
    K: tl.constexpr,  # 32
    num_warps=4, num_stages=2
):
    # Grid: (Bsz, Csz)
    b = tl.program_id(0)
    c = tl.program_id(1)

    # Accumulate G[i, j, h] for all h
    for i in range(S):
        for j in range(S):
            acc = 0.0
            # Sum over groups g and state k
            for g in range(G_CONST):
                for k in range(K):
                    b_off = b * stride_B_b + c * stride_B_c + j * stride_B_s + g * stride_B_g + k * stride_B_k
                    c_off = b * stride_C_b + c * stride_C_c + i * stride_C_s + g * stride_C_g + k * stride_C_k
                    b_val = tl.load(B_ptr + b_off)
                    c_val = tl.load(C_ptr + c_off)
                    acc += b_val * c_val
            # Write G[i, j, h] for all h (h index implied by strides)
            for hh in range(S):
                g_off = b * stride_G_b + c * stride_G_c + i * stride_G_i + j * stride_G_j + hh * stride_G_h
                tl.store(G_ptr + g_off, acc)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, A_cumsum: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        """
        Triton-optimized forward that:
          1) computes L via compute_L_exp_cumsum_kernel
          2) computes G via contract_BC_to_G_kernel
          3) computes Y_diag using PyTorch elementwise multiply and sum over j
        Returns Y_diag in bfloat16 to match original behavior.
        """
        assert hidden_states.is_cuda and A_cumsum.is_cuda and B.is_cuda and C.is_cuda, "Inputs must be CUDA tensors"

        Bsz, Csz, S, H, D = hidden_states.shape
        # Assert fixed constants for correctness in Triton loops
        assert S == 128 and H == 32 and D == 32, "This Triton implementation specializes for S=128, H=32, D=32"
        assert A_cumsum.shape == (Bsz, H, Csz, S), f"Expected A_cumsum shape [B,H,C,S], got {A_cumsum.shape}"
        assert B.shape == (Bsz, Csz, S, 8, 32), "Expected B shape [B,C,S,8,32]"
        assert C.shape == (Bsz, Csz, S, 8, 32), "Expected C shape [B,C,S,8,32]"

        # Ensure tensors are contiguous
        A_cumsum = A_cumsum.contiguous()
        B = B.contiguous()
        C = C.contiguous()
        hidden_states = hidden_states.contiguous()

        # Allocate outputs
        L = torch.empty((Bsz, Csz, S, S, H), dtype=torch.float32, device=hidden_states.device)
        G = torch.empty((Bsz, Csz, S, S, H), dtype=torch.float32, device=hidden_states.device)

        # Launch Triton kernel 1: compute L
        compute_L_exp_cumsum_kernel[(Bsz, Csz)](
            A_cumsum, L,
            Bsz, Csz,
            stride_A_b=A_cumsum.stride(0), stride_A_h=A_cumsum.stride(1), stride_A_c=A_cumsum.stride(2), stride_A_s=A_cumsum.stride(3),
            stride_L_b=L.stride(0), stride_L_c=L.stride(1), stride_L_i=L.stride(2), stride_L_j=L.stride(3), stride_L_h=L.stride(4),
            S=128, H=32,
            num_warps=4, num_stages=2
        )

        # Launch Triton kernel 2: compute G
        contract_BC_to_G_kernel[(Bsz, Csz)](
            B, C, G,
            Bsz, Csz,
            stride_B_b=B.stride(0), stride_B_c=B.stride(1), stride_B_s=B.stride(2), stride_B_g=B.stride(3), stride_B_k=B.stride(4),
            stride_C_b=C.stride(0), stride_C_c=C.stride(1), stride_C_s=C.stride(2), stride_C_g=C.stride(3), stride_C_k=C.stride(4),
            stride_G_b=G.stride(0), stride_G_c=G.stride(1), stride_G_i=G.stride(2), stride_G_j=G.stride(3), stride_G_h=G.stride(4),
            S=128, G_CONST=8, K=32,
            num_warps=4, num_stages=2
        )

        # Compute M = G * L using PyTorch
        M = G * L  # elementwise multiply

        # Compute Y_diag = sum over j of M * hidden_states
        # hidden_states: [B, C, S, H, D]
        # M: [B, C, S, S, H]
        # Y: [B, C, S, H, D]
        Y = torch.zeros((Bsz, Csz, S, H, D), dtype=torch.float32, device=hidden_states.device)
        # Sum over j dimension of M across hidden_states j
        for j in range(S):
            # For each (b, c, i, h, d), accumulate M[:, :, i, j, :] * hidden_states[:, :, j, :, :] along d -> sum across j
            # Use einsum for clarity: contract M over j with hidden along j
            Y += torch.einsum('bcsH, bcjHd->bcsHd', M[:, :, j, :, :], hidden_states)

        # Convert to bfloat16 to match original behavior
        return Y.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
