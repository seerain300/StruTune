import torch
import triton
import triton.language as tl


@triton.jit
def compute_L_exp_cumsum_kernel(
    A_ptr, L_ptr,
    Bsz, Csz,
    S: tl.constexpr, H: tl.constexpr,
    stride_A_b, stride_A_h, stride_A_c, stride_A_s,
    stride_L_b, stride_L_c, stride_L_i, stride_L_j, stride_L_h,
    num_warps=8, num_stages=2
):
    # Each program handles one (b, c) pair
    b = tl.program_id(0)
    c = tl.program_id(1)

    for h_idx in range(H):
        for i_idx in range(S):
            cumsum = tl.zeros((S,), dtype=tl.float32)
            for j_idx in range(S):
                if j_idx <= i_idx:
                    ptr_A = A_ptr + b * stride_A_b + h_idx * stride_A_h + c * stride_A_c + j_idx * stride_A_s
                    a_val = tl.load(ptr_A)
                    cumsum[j_idx] = cumsum[j_idx] + a_val
            for j_idx in range(S):
                if j_idx <= i_idx:
                    exp_val = tl.exp(cumsum[j_idx])
                    ptr_L = L_ptr + b * stride_L_b + c * stride_L_c + i_idx * stride_L_i + j_idx * stride_L_j + h_idx * stride_L_h
                    tl.store(ptr_L, exp_val)
                else:
                    ptr_L = L_ptr + b * stride_L_b + c * stride_L_c + i_idx * stride_L_i + j_idx * stride_L_j + h_idx * stride_L_h
                    tl.store(ptr_L, 0.0)


@triton.jit
def contract_BC_to_G_kernel(
    B_ptr, C_ptr, G_ptr,
    Bsz, Csz,
    S: tl.constexpr, H: tl.constexpr,
    G_CONST: tl.constexpr, K: tl.constexpr,
    stride_B_b, stride_B_c, stride_B_s, stride_B_g, stride_B_k,
    stride_C_b, stride_C_c, stride_C_s, stride_C_g, stride_C_k,
    stride_G_b, stride_G_c, stride_G_i, stride_G_j, stride_G_h,
    num_warps=8, num_stages=2
):
    # Each program handles one (b, c)
    b = tl.program_id(0)
    c = tl.program_id(1)

    for h_idx in range(H):
        for i_idx in range(S):
            for j_idx in range(S):
                acc = 0.0
                for g_idx in range(G_CONST):
                    for k_idx in range(K):
                        ptr_B = B_ptr + b * stride_B_b + c * stride_B_c + j_idx * stride_B_s + g_idx * stride_B_g + k_idx * stride_B_k
                        ptr_C = C_ptr + b * stride_C_b + c * stride_C_c + i_idx * stride_C_s + g_idx * stride_C_g + k_idx * stride_C_k
                        b_val = tl.load(ptr_B)
                        c_val = tl.load(ptr_C)
                        acc += b_val * c_val
                ptr_G = G_ptr + b * stride_G_b + c * stride_G_c + i_idx * stride_G_i + j_idx * stride_G_j + h_idx * stride_G_h
                tl.store(ptr_G, acc)


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
    # Grid covers (b, c, i, h, d) -> one program per output element
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
          1) computes L via compute_L_exp_cumsum_kernel
          2) computes G via contract_BC_to_G_kernel
          3) computes Y_diag via contract_M_with_hidden_kernel
        All heavy computation is done inside Triton kernels; no torch ops in host code for elementwise operations.
        Returns Y_diag in bfloat16 to match original behavior.
        """
        assert hidden_states.is_cuda and A_cumsum.is_cuda and B.is_cuda and C.is_cuda, "Inputs must be CUDA tensors"

        # Shapes and specialization for fixed sizes used in original code
        Bsz, Csz, S, H, D = hidden_states.shape
        # The original code fixes S=128, H=32, and K=state_size=H=32. We will use these to simplify kernels.
        assert S == 128 and H == 32 and D == 32, "This Triton implementation specializes for S=128, H=32, D=32"
        assert A_cumsum.shape == (Bsz, H, Csz, S), f"Expected A_cumsum shape [B,H,C,S], got {A_cumsum.shape}"
        # B and C are expected to match original semantics with G_CONST=8, K=H=32
        assert B.shape == (Bsz, Csz, S, 8, 32), "Expected B shape [B,C,S,8,32]"
        assert C.shape == (Bsz, Csz, S, 8, 32), "Expected C shape [B,C,S,8,32]"

        # Allocate outputs
        L = torch.empty((Bsz, Csz, S, S, H), dtype=torch.float32, device=hidden_states.device)
        G = torch.empty((Bsz, Csz, S, S, H), dtype=torch.float32, device=hidden_states.device)
        Y_diag = torch.empty((Bsz, Csz, S, H, D), dtype=torch.float32, device=hidden_states.device)

        # Launch kernel 1: compute L
        compute_L_exp_cumsum_kernel[(Bsz, Csz)](
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
            S=128, H=32,
            G_CONST=8, K=32,
            stride_B_b=B.stride(0), stride_B_c=B.stride(1), stride_B_s=B.stride(2), stride_B_g=B.stride(3), stride_B_k=B.stride(4),
            stride_C_b=C.stride(0), stride_C_c=C.stride(1), stride_C_s=C.stride(2), stride_C_g=C.stride(3), stride_C_k=C.stride(4),
            stride_G_b=G.stride(0), stride_G_c=G.stride(1), stride_G_i=G.stride(2), stride_G_j=G.stride(3), stride_G_h=G.stride(4),
            num_warps=8, num_stages=2
        )

        # Launch kernel 3: compute Y_diag = sum_j (G * L) * hidden over j
        contract_M_with_hidden_kernel[(Bsz, Csz, S, H, D)](
            G, L, hidden_states, Y_diag,
            Bsz, Csz, S=128, H=32, D=32,
            stride_G_b=G.stride(0), stride_G_c=G.stride(1), stride_G_i=G.stride(2), stride_G_j=G.stride(3), stride_G_h=G.stride(4),
            stride_L_b=L.stride(0), stride_L_c=L.stride(1), stride_L_i=L.stride(2), stride_L_j=L.stride(3), stride_L_h=L.stride(4),
            stride_hs_b=hidden_states.stride(0), stride_hs_c=hidden_states.stride(1), stride_hs_s=hidden_states.stride(2), stride_hs_h=hidden_states.stride(3), stride_hs_d=hidden_states.stride(4),
            stride_Y_b=Y_diag.stride(0), stride_Y_c=Y_diag.stride(1), stride_Y_i=Y_diag.stride(2), stride_Y_h=Y_diag.stride(3), stride_Y_d=Y_diag.stride(4),
            num_warps=4, num_stages=2
        )

        # Return in bfloat16 to match original behavior
        return Y_diag.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
