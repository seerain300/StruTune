import torch
import triton
import triton.language as tl


@triton.jit
def compute_L_exp_cumsum_kernel(
    A_ptr, L_ptr,
    Bsz, Csz,
    stride_A_b, stride_A_h, stride_A_c, stride_A_s,
    stride_L_b, stride_L_c, stride_L_i, stride_L_j, stride_L_h,
    S: tl.constexpr,  # chunk_size (e.g., 128)
    H: tl.constexpr,  # num_heads (e.g., 32)
    num_warps=8, num_stages=2
):
    # One program per (b, c)
    b = tl.program_id(0)
    c = tl.program_id(1)

    for i in range(S):
        # Initialize cumsum vector s[j]
        s = tl.zeros([S], dtype=tl.float32)
        for j in range(S):
            for hh in range(H):
                a_off = b * stride_A_b + hh * stride_A_h + c * stride_A_c + j * stride_A_s
                a_val = tl.load(A_ptr + a_off, mask=True, other=0.0)
                s[j] += a_val
            for hh in range(H):
                l_off = b * stride_L_b + c * stride_L_c + i * stride_L_i + j * stride_L_j + hh * stride_L_h
                if j <= i:
                    tl.store(L_ptr + l_off, tl.exp(s[j]))
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
    num_warps=8, num_stages=2
):
    # One program per (b, c)
    b = tl.program_id(0)
    c = tl.program_id(1)

    # Loop over all heads h; we write G[i, j, h] for all h by accumulating into a 5D tensor
    for i in range(S):
        for j in range(S):
            acc = 0.0
            # Sum over groups and states
            for g in range(G_CONST):
                for k in range(K):
                    b_off = b * stride_B_b + c * stride_B_c + j * stride_B_s + g * stride_B_g + k * stride_B_k
                    c_off = b * stride_C_b + c * stride_C_c + i * stride_C_s + g * stride_C_g + k * stride_C_k
                    b_val = tl.load(B_ptr + b_off)
                    c_val = tl.load(C_ptr + c_off)
                    acc += b_val * c_val
            # Store G[i, j, h] for all h (we assume H == K in this specialization; adjust as needed)
            # If H != K, you would need to pass H as constexpr and loop over it. Here, we follow the original model's H=32, K=32.
            for hh in range(K):
                g_off = b * stride_G_b + c * stride_G_c + i * stride_G_i + j * stride_G_j + hh * stride_G_h
                tl.store(G_ptr + g_off, acc)


@triton.jit
def contract_M_with_hidden_reduce_j_kernel(
    G_ptr, L_ptr, hidden_ptr, Y_ptr,
    Bsz, Csz, S: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
    stride_G_b, stride_G_c, stride_G_i, stride_G_j, stride_G_h,
    stride_L_b, stride_L_c, stride_L_i, stride_L_j, stride_L_h,
    stride_hs_b, stride_hs_c, stride_hs_j, stride_hs_h, stride_hs_d,
    stride_Y_b, stride_Y_c, stride_Y_i, stride_Y_h, stride_Y_d,
    num_warps=8, num_stages=2
):
    # One program per output (b, c, i, h). We tile over d dimension for better throughput.
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)
    h = tl.program_id(3)
    # d is handled via tiling
    d_base = tl.program_id(4) * 8  # tile size 8
    d_vec = d_base + tl.arange(0, 8)
    # Initialize accumulator for Y_diag[b, c, i, h, d]
    Y_acc = tl.zeros([8], dtype=tl.float32)

    # Reduction over j
    for j in range(S):
        # Load G[i, j, h]
        g_off = b * stride_G_b + c * stride_G_c + i * stride_G_i + j * stride_G_j + h * stride_G_h
        g_val = tl.load(G_ptr + g_off)
        # Load L[i, j, h]
        l_off = b * stride_L_b + c * stride_L_c + i * stride_L_i + j * stride_L_j + h * stride_L_h
        l_val = tl.load(L_ptr + l_off)
        M_val = g_val * l_val  # elementwise multiply
        # Load hidden[b, c, j, h, d] for d in tile
        hs_off = b * stride_hs_b + c * stride_hs_c + j * stride_hs_j + h * stride_hs_h + d_vec * stride_hs_d
        hs_mask = d_vec < D
        hs_val = tl.load(hidden_ptr + hs_off, mask=hs_mask, other=0.0)
        Y_acc += M_val * hs_val

    # Store results for each d in the tile
    # Note: Y is float32 as per original model; convert later to bfloat16 if needed
    for dd in range(8):
        if d_base + dd < D:
            y_off = b * stride_Y_b + c * stride_Y_c + i * stride_Y_i + h * stride_Y_h + (d_base + dd) * stride_Y_d
            tl.store(Y_ptr + y_off, Y_acc[dd])


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, A_cumsum: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        """
        Triton-optimized forward:
          - compute_L_exp_cumsum_kernel: build L
          - contract_BC_to_G_kernel: build G
          - contract_M_with_hidden_reduce_j_kernel: compute Y_diag
        Returns: [B, C, S, H, D] in float32 (converted to bfloat16 at the end to match original behavior).
        """
        assert hidden_states.is_cuda and A_cumsum.is_cuda and B.is_cuda and C.is_cuda, "Inputs must be CUDA tensors"

        Bsz, Csz, S, H, D = hidden_states.shape
        # Specialize for the original model's constants; if different, you can adapt loops.
        assert S == 128 and H == 32 and D == 32, "This Triton implementation specializes for S=128, H=32, D=32"

        # Ensure contiguous for predictable strides
        hidden_states = hidden_states.contiguous()
        A_cumsum = A_cumsum.contiguous()
        B = B.contiguous()
        C = C.contiguous()

        # Allocate outputs
        L = torch.empty((Bsz, Csz, S, S, H), dtype=torch.float32, device=hidden_states.device)
        G = torch.empty((Bsz, Csz, S, S, H), dtype=torch.float32, device=hidden_states.device)
        # Tile output Y along D to improve Triton performance
        Y = torch.empty((Bsz, Csz, S, H, D), dtype=torch.float32, device=hidden_states.device)

        # Launch kernel 1: compute L
        compute_L_exp_cumsum_kernel[(Bsz, Csz)](
            A_cumsum, L,
            Bsz, Csz,
            A_cumsum.stride(0), A_cumsum.stride(1), A_cumsum.stride(2), A_cumsum.stride(3),
            L.stride(0), L.stride(1), L.stride(2), L.stride(3), L.stride(4),
            S=S, H=H,
            num_warps=8, num_stages=2
        )

        # Launch kernel 2: compute G
        contract_BC_to_G_kernel[(Bsz, Csz)](
            B, C, G,
            Bsz, Csz,
            B.stride(0), B.stride(1), B.stride(2), B.stride(3), B.stride(4),
            C.stride(0), C.stride(1), C.stride(2), C.stride(3), C.stride(4),
            G.stride(0), G.stride(1), G.stride(2), G.stride(3), G.stride(4),
            S=S, G_CONST=8, K=32,
            num_warps=8, num_stages=2
        )

        # Build M = G * L using PyTorch elementwise multiply (lightweight)
        M = G * L

        # Launch kernel 3: reduce over j to produce Y_diag
        # Grid: (Bsz, Csz, S, H, tiles along D). Use 8-way tiling along D.
        grid = (Bsz, Csz, S, H, (D + 7) // 8)
        contract_M_with_hidden_reduce_j_kernel[grid](
            M, L, hidden_states, Y,
            Bsz, Csz, S, H, D,
            M.stride(0), M.stride(1), M.stride(2), M.stride(3), M.stride(4),
            L.stride(0), L.stride(1), L.stride(2), L.stride(3), L.stride(4),
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2), hidden_states.stride(3), hidden_states.stride(4),
            Y.stride(0), Y.stride(1), Y.stride(2), Y.stride(3), Y.stride(4),
            num_warps=8, num_stages=2
        )

        # Return in bfloat16 to match original model behavior
        return Y.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
