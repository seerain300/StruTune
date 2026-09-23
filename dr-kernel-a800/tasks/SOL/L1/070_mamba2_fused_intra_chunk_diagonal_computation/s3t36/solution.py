import torch
import triton
import triton.language as tl


@triton.jit
def compute_L_exp_cumsum_kernel(
    A_ptr, L_ptr,
    Bsz, Csz,
    stride_A_b, stride_A_h, stride_A_c, stride_A_s,
    stride_L_b, stride_L_c, stride_L_i, stride_L_j, stride_L_h,
    S: tl.constexpr,  # chunk_size (128)
    H: tl.constexpr,  # num_heads (32)
    num_warps=8, num_stages=2
):
    # One program per (b, c)
    b = tl.program_id(0)
    c = tl.program_id(1)

    for hh in range(H):
        for i in range(S):
            # cumsum across j for this i and head hh
            s = 0.0
            for j in range(S):
                a_off = b * stride_A_b + hh * stride_A_h + c * stride_A_c + j * stride_A_s
                a_val = tl.load(A_ptr + a_off)
                s += a_val
                if j <= i:
                    l_off = b * stride_L_b + c * stride_L_c + i * stride_L_i + j * stride_L_j + hh * stride_L_h
                    tl.store(L_ptr + l_off, tl.exp(s))
                else:
                    l_off = b * stride_L_b + c * stride_L_c + i * stride_L_i + j * stride_L_j + hh * stride_L_h
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
    num_warps=8, num_stages=2
):
    # One program per (b, c)
    b = tl.program_id(0)
    c = tl.program_id(1)

    for hh in range(H):  # H is 32
        for i in range(S):
            for j in range(S):
                acc = 0.0
                for g in range(G_CONST):
                    for k in range(K):
                        b_off = b * stride_B_b + c * stride_B_c + j * stride_B_s + g * stride_B_g + k * stride_B_k
                        c_off = b * stride_C_b + c * stride_C_c + i * stride_C_s + g * stride_C_g + k * stride_C_k
                        b_val = tl.load(B_ptr + b_off)
                        c_val = tl.load(C_ptr + c_off)
                        acc += b_val * c_val
                g_off = b * stride_G_b + c * stride_G_c + i * stride_G_i + j * stride_G_j + hh * stride_G_h
                tl.store(G_ptr + g_off, acc)


@triton.jit
def contract_M_and_hidden_to_Y_diag_kernel(
    G_ptr, L_ptr, hidden_ptr, Y_ptr,
    Bsz, Csz, S, H, D,
    stride_G_b, stride_G_c, stride_G_i, stride_G_j, stride_G_h,
    stride_L_b, stride_L_c, stride_L_i, stride_L_j, stride_L_h,
    stride_hidden_b, stride_hidden_c, stride_hidden_j, stride_hidden_h, stride_hidden_d,
    stride_Y_b, stride_Y_c, stride_Y_i, stride_Y_h, stride_Y_d,
    num_warps=8, num_stages=2
):
    # Grid: (Bsz, Csz, S) -> each program handles one fixed (b, c, i)
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)

    # Accumulate over j for each head h and each feature d
    for d in range(D):
        # Compute scalar acc independent of h
        acc = 0.0
        for j in range(S):
            # For each j, compute sum over h of (G[i, j, h] * L[i, j, h]) and multiply by hidden[b, c, j, h, d]
            for hh in range(H):
                g_off = b * stride_G_b + c * stride_G_c + i * stride_G_i + j * stride_G_j + hh * stride_G_h
                l_off = b * stride_L_b + c * stride_L_c + i * stride_L_i + j * stride_L_j + hh * stride_L_h
                hidden_off = b * stride_hidden_b + c * stride_hidden_c + j * stride_hidden_j + hh * stride_hidden_h + d * stride_hidden_d
                g_val = tl.load(G_ptr + g_off)
                l_val = tl.load(L_ptr + l_off)
                hidden_val = tl.load(hidden_ptr + hidden_off)
                acc += (g_val * l_val) * hidden_val

        # Store acc to Y[b, c, i, h, d] for all h (since acc is scalar)
        # We store the same acc across all h; output dtype is float32 (will be converted to bfloat16 after forward).
        for hh in range(H):
            y_off = b * stride_Y_b + c * stride_Y_c + i * stride_Y_i + hh * stride_Y_h + d * stride_Y_d
            tl.store(Y_ptr + y_off, acc)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, A_cumsum: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        """
        Triton-optimized forward that:
          1) computes L via compute_L_exp_cumsum_kernel
          2) computes G via contract_BC_to_G_kernel
          3) computes Y_diag via contract_M_and_hidden_to_Y_diag_kernel
        All kernels are launched from forward; no PyTorch tensor ops for heavy compute.
        Returns bfloat16 output to match original behavior.
        """
        assert hidden_states.is_cuda and A_cumsum.is_cuda and B.is_cuda and C.is_cuda, "Inputs must be CUDA tensors"

        # Shapes (specialized to original constants)
        Bsz, Csz, S, H, D = hidden_states.shape
        # Ensure specialization matches original assumptions
        assert S == 128 and H == 32 and D == 32, "This Triton implementation specializes for S=128, H=32, D=32"

        # Allocate outputs (float32 for compute)
        L = torch.empty((Bsz, Csz, S, S, H), dtype=torch.float32, device=hidden_states.device)
        G = torch.empty((Bsz, Csz, S, S, H), dtype=torch.float32, device=hidden_states.device)
        Y_diag = torch.empty((Bsz, Csz, S, H, D), dtype=torch.float32, device=hidden_states.device)

        # Launch kernel 1: compute L
        compute_L_exp_cumsum_kernel[(Bsz, Csz)](
            A_cumsum, L,
            Bsz, Csz,
            stride_A_b=A_cumsum.stride(0), stride_A_h=A_cumsum.stride(1), stride_A_c=A_cumsum.stride(2), stride_A_s=A_cumsum.stride(3),
            stride_L_b=L.stride(0), stride_L_c=L.stride(1), stride_L_i=L.stride(2), stride_L_j=L.stride(3), stride_L_h=L.stride(4),
            S=S, H=H,
            num_warps=8, num_stages=2
        )

        # Launch kernel 2: compute G from B and C
        contract_BC_to_G_kernel[(Bsz, Csz)](
            B, C, G,
            Bsz, Csz,
            stride_B_b=B.stride(0), stride_B_c=B.stride(1), stride_B_s=B.stride(2), stride_B_g=B.stride(3), stride_B_k=B.stride(4),
            stride_C_b=C.stride(0), stride_C_c=C.stride(1), stride_C_s=C.stride(2), stride_C_g=C.stride(3), stride_C_k=C.stride(4),
            stride_G_b=G.stride(0), stride_G_c=G.stride(1), stride_G_i=G.stride(2), stride_G_j=G.stride(3), stride_G_h=G.stride(4),
            S=S, G_CONST=8, K=32,
            num_warps=8, num_stages=2
        )

        # Launch kernel 3: final contraction to Y_diag
        # We will compute Y_diag in float32 here; the original returns bfloat16, so we cast at the end.
        contract_M_and_hidden_to_Y_diag_kernel[(Bsz, Csz, S)](
            G, L, hidden_states.to(torch.float32), Y_diag,
            Bsz, Csz, S, H, D,
            stride_G_b=G.stride(0), stride_G_c=G.stride(1), stride_G_i=G.stride(2), stride_G_j=G.stride(3), stride_G_h=G.stride(4),
            stride_L_b=L.stride(0), stride_L_c=L.stride(1), stride_L_i=L.stride(2), stride_L_j=L.stride(3), stride_L_h=L.stride(4),
            stride_hidden_b=hidden_states.to(torch.float32).stride(0),
            stride_hidden_c=hidden_states.to(torch.float32).stride(1),
            stride_hidden_j=hidden_states.to(torch.float32).stride(2),
            stride_hidden_h=hidden_states.to(torch.float32).stride(3),
            stride_hidden_d=hidden_states.to(torch.float32).stride(4),
            stride_Y_b=Y_diag.stride(0), stride_Y_c=Y_diag.stride(1), stride_Y_i=Y_diag.stride(2), stride_Y_h=Y_diag.stride(3), stride_Y_d=Y_diag.stride(4),
            num_warps=8, num_stages=2
        )

        # Return in bfloat16 to match original behavior
        return Y_diag.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
