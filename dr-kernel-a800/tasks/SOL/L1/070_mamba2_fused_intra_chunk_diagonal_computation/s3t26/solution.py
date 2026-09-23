import torch
import triton
import triton.language as tl


@triton.jit
def contract_BC_to_G_kernel(
    B_ptr, C_ptr, G_ptr,
    Bsz, Csz,
    stride_B_b, stride_B_c, stride_B_s, stride_B_g, stride_B_k,
    stride_C_b, stride_C_c, stride_C_s, stride_C_g, stride_C_k,
    stride_G_b, stride_G_c, stride_G_i, stride_G_j, stride_G_h,
    S: tl.constexpr,       # 128
    G_CONST: tl.constexpr, # 8
    K: tl.constexpr,       # 32
    H: tl.constexpr,       # 32
    num_warps=8, num_stages=2
):
    # One program per (b, c)
    b = tl.program_id(0)
    c = tl.program_id(1)

    # Loop over i and j and accumulate over groups and states, writing for all heads h
    for i in range(S):
        for j in range(S):
            acc = 0.0
            # Sum over groups and state dimension
            for g in range(G_CONST):
                for k in range(K):
                    b_off = b * stride_B_b + c * stride_B_c + j * stride_B_s + g * stride_B_g + k * stride_B_k
                    c_off = b * stride_C_b + c * stride_C_c + i * stride_C_s + g * stride_C_g + k * stride_C_k
                    b_val = tl.load(B_ptr + b_off)
                    c_val = tl.load(C_ptr + c_off)
                    acc += b_val * c_val
            # Write G[i, j, h] = acc for all h
            for h in range(H):
                g_off = b * stride_G_b + c * stride_G_c + i * stride_G_i + j * stride_G_j + h * stride_G_h
                tl.store(G_ptr + g_off, acc)


@triton.jit
def contract_M_with_hidden_kernel(
    G_ptr, L_ptr, hidden_ptr, Y_ptr,
    Bsz, Csz,
    stride_G_b, stride_G_c, stride_G_i, stride_G_j, stride_G_h,
    stride_L_b, stride_L_c, stride_L_i, stride_L_j, stride_L_h,
    stride_hs_b, stride_hs_c, stride_hs_s, stride_hs_h, stride_hs_d,
    stride_Y_b, stride_Y_c, stride_Y_s, stride_Y_h, stride_Y_d,
    S: tl.constexpr,   # 128
    H: tl.constexpr,   # 32
    D: tl.constexpr,   # hidden dim
    num_warps=8, num_stages=2
):
    # One program per (b, c, i, h, d)
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)
    h = tl.program_id(3)
    d = tl.program_id(4)

    total = 0.0
    for j in range(S):
        g_off = b * stride_G_b + c * stride_G_c + i * stride_G_i + j * stride_G_j + h * stride_G_h
        l_off = b * stride_L_b + c * stride_L_c + i * stride_L_i + j * stride_L_j + h * stride_L_h
        g_val = tl.load(G_ptr + g_off)
        l_val = tl.load(L_ptr + l_off)
        # hidden[b, c, j, h, d]
        hs_off = b * stride_hs_b + c * stride_hs_c + j * stride_hs_s + h * stride_hs_h + d * stride_hs_d
        hs_val = tl.load(hidden_ptr + hs_off)
        total += g_val * l_val * hs_val

    y_off = b * stride_Y_b + c * stride_Y_c + i * stride_Y_s + h * stride_Y_h + d * stride_Y_d
    tl.store(Y_ptr + y_off, total)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                A_cumsum: torch.Tensor,
                B: torch.Tensor,
                C: torch.Tensor) -> torch.Tensor:
        """
        Triton-optimized forward that:
          - builds L via PyTorch (cumsum + exp of masked A_cumsum)
          - computes G via Triton kernel contracting B and C
          - computes Y_diag via Triton kernel contracting M = G * L with hidden_states
        Returns output cast to bfloat16 to match original behavior.
        """
        assert hidden_states.is_cuda and A_cumsum.is_cuda and B.is_cuda and C.is_cuda, "Inputs must be CUDA tensors"

        # Shapes: specialize for S=128, H=32, G_CONST=8, K=32
        Bsz, Csz, S, H, D = hidden_states.shape
        # The provided workloads use D=32 (original model), but keep D general
        assert S == 128 and H == 32, "This Triton implementation specializes for S=128, H=32"
        assert B.shape == (Bsz, Csz, S, 8, 32), "Expected B shape [B,C,S,8,32]"
        assert C.shape == (Bsz, Csz, S, 8, 32), "Expected C shape [B,C,S,8,32]"

        # Ensure contiguous tensors
        A = A_cumsum.contiguous()
        Bc = B.contiguous()
        Cc = C.contiguous()
        hidden = hidden_states.contiguous()

        # Step A: Build L via PyTorch to ensure correctness:
        # L[b,h,c,i,j] = exp(cumsum_j(A[b,h,c,:])) if j <= i else 0
        A_exp = A.unsqueeze(-1).expand(Bsz, H, Csz, S, S).contiguous()  # [B,H,C,S,S]
        i_idx = torch.arange(S, device=A.device).view(1, 1, 1, S, 1)   # [1,1,1,S,1]
        j_idx = torch.arange(S, device=A.device).view(1, 1, 1, 1, S)   # [1,1,1,1,S]
        mask = (j_idx <= i_idx).to(A.dtype)  # broadcast to [1,1,1,S,S]
        A_cum = torch.cumsum(A_exp, dim=-1)  # [B,H,C,S,S]
        A_masked = A_cum * mask
        L = torch.exp(A_masked)  # [B,H,C,S,S], float32
        L = L.to(torch.float32).contiguous()

        # Allocate G and Y_diag
        G = torch.empty((Bsz, Csz, S, S, H), dtype=torch.float32, device=hidden.device)
        Y = torch.empty((Bsz, Csz, S, H, D), dtype=torch.float32, device=hidden.device)

        # Launch Triton kernel 1: contract B and C to G
        contract_BC_to_G_kernel[(Bsz, Csz)](
            Bc, Cc, G,
            Bsz, Csz,
            Bc.stride(0), Bc.stride(1), Bc.stride(2), Bc.stride(3), Bc.stride(4),
            Cc.stride(0), Cc.stride(1), Cc.stride(2), Cc.stride(3), Cc.stride(4),
            G.stride(0), G.stride(1), G.stride(2), G.stride(3), G.stride(4),
            S=128, G_CONST=8, K=32, H=32,
            num_warps=8, num_stages=2
        )

        # Launch Triton kernel 2: contract M = G * L with hidden to get Y_diag
        # Note: M is implicit in this kernel via elementwise multiply of loaded G and L
        contract_M_with_hidden_kernel[(Bsz, Csz, S, H, D)](
            G, L, hidden, Y,
            Bsz, Csz,
            G.stride(0), G.stride(1), G.stride(2), G.stride(3), G.stride(4),
            L.stride(0), L.stride(1), L.stride(2), L.stride(3), L.stride(4),
            hidden.stride(0), hidden.stride(1), hidden.stride(2), hidden.stride(3), hidden.stride(4),
            Y.stride(0), Y.stride(1), Y.stride(2), Y.stride(3), Y.stride(4),
            S=128, H=32, D=D,
            num_warps=8, num_stages=2
        )

        # Return in bfloat16 to match original
        return Y.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
