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

    # Loop over i (source positions), then for each i, compute cumsum over j and write L[i, j, h]
    for i in range(S):
        # cumsum vector for j in [0..S-1], one per h
        # We will iterate j and update cumsum[h], then write L[i, j, h]
        for j in range(S):
            # Update cumsum across h
            # For each h, L[i, j, h] = exp(sum_{t<=j} A[b, h, c, t])
            # Loop over h
            for hh in range(H):
                # Accumulate A[b, h, c, j] into cumsum[h]
                ptr_A = A_ptr + b * stride_A_b + hh * stride_A_h + c * stride_A_c + j * stride_A_s
                a_val = tl.load(ptr_A).to(tl.float32)
                # We keep a global cumsum vector across h using tl.arange
                # Since Triton doesn't support vector indexing like s[hh] += a_val directly, we implement cumsum scalar per h.
                # Therefore, we need to recompute cumsum for each h via a loop over j: inefficient.
                # To fix this, we instead compute cumsum per h inside the loop below.
                # That means we will not use vector s here; instead compute sum on the fly per h.
                # So, for each j, compute exp of sum up to j for all h by recomputing sum per h.
                # However, this is unavoidable without a vector cumsum buffer; we'll implement as below:
                # We'll use scalar cumsum for simplicity and correctness.
                # Note: Triton doesn't allow dynamic vector accumulation across h easily here.
                # So we'll recompute the sum per h each j, which is acceptable for these sizes.
                # (We'll do cumsum per h by recomputation; see below implementation detail.)

                # Now write L[i, j, h] if j <= i; otherwise 0
                if j <= i:
                    # We need cumsum value at j for this h. We'll compute it by recomputing sum from 0 to j for this h.
                    # Initialize sum for this h
                    sum_val = 0.0
                    for t in range(j + 1):
                        ptr_At = A_ptr + b * stride_A_b + hh * stride_A_h + c * stride_A_c + t * stride_A_s
                        at = tl.load(ptr_At).to(tl.float32)
                        sum_val += at
                    l_val = tl.exp(sum_val)
                else:
                    l_val = 0.0

                ptr_L = L_ptr + b * stride_L_b + c * stride_L_c + i * stride_L_i + j * stride_L_j + hh * stride_L_h
                tl.store(ptr_L, l_val)


@triton.jit
def contract_BC_to_G_kernel(
    B_ptr, C_ptr, G_ptr,
    Bsz, Csz,
    stride_B_b, stride_B_c, stride_B_s, stride_B_g, stride_B_k,
    stride_C_b, stride_C_c, stride_C_s, stride_C_g, stride_C_k,
    stride_G_b, stride_G_c, stride_G_i, stride_G_j, stride_G_h,
    S: tl.constexpr, H: tl.constexpr, G_CONST: tl.constexpr, K: tl.constexpr,
    num_warps=8, num_stages=2
):
    # One program per (b, c)
    b = tl.program_id(0)
    c = tl.program_id(1)

    # Accumulator G[i, j, h] per (i, j, h)
    for i in range(S):
        for j in range(S):
            g_val = tl.zeros((), dtype=tl.float32)
            # Sum over groups and states
            for g in range(G_CONST):
                for k in range(K):
                    ptr_B = B_ptr + b * stride_B_b + c * stride_B_c + j * stride_B_s + g * stride_B_g + k * stride_B_k
                    b_val = tl.load(ptr_B).to(tl.float32)
                    ptr_C = C_ptr + b * stride_C_b + c * stride_C_c + i * stride_C_s + g * stride_C_g + k * stride_C_k
                    c_val = tl.load(ptr_C).to(tl.float32)
                    g_val += b_val * c_val

            # Store G[i, j, h] for all h
            for hh in range(H):
                ptr_G = G_ptr + b * stride_G_b + c * stride_G_c + i * stride_G_i + j * stride_G_j + hh * stride_G_h
                tl.store(ptr_G, g_val)


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
    # Grid: (Bsz, Csz, S, H, D) one program per output element
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
        Launches three Triton kernels; returns bfloat16 output to match original behavior.
        """
        assert hidden_states.is_cuda and A_cumsum.is_cuda and B.is_cuda and C.is_cuda, "Inputs must be CUDA tensors"

        # Shapes and specialization
        Bsz, Csz, S, H, D = hidden_states.shape
        # Enforce specialization constants for correctness
        assert S == 128 and H == 32 and D == 32, "This Triton implementation specializes for S=128, H=32, D=32"

        # A_cumsum: [B, H, C, S]
        assert A_cumsum.shape == (Bsz, H, Csz, S), f"Expected A_cumsum shape [B,H,C,S], got {A_cumsum.shape}"

        # B and C: [B, C, S, 8, 32]
        G_CONST = 8
        K = 32  # state_size, matches H in original model
        assert B.shape == (Bsz, Csz, S, G_CONST, K), f"Expected B shape [B,C,S,{G_CONST},{K}], got {B.shape}"
        assert C.shape == (Bsz, Csz, S, G_CONST, K), f"Expected C shape [B,C,S,{G_CONST},{K}], got {C.shape}"

        # Ensure contiguous for simpler stride handling
        A_cumsum = A_cumsum.contiguous()
        B = B.contiguous()
        C = C.contiguous()
        hidden_states = hidden_states.contiguous()

        # Allocate outputs in float32 for compute
        L = torch.empty((Bsz, Csz, S, S, H), dtype=torch.float32, device=hidden_states.device)
        G = torch.empty((Bsz, Csz, S, S, H), dtype=torch.float32, device=hidden_states.device)
        Y_diag = torch.empty((Bsz, Csz, S, H, D), dtype=torch.float32, device=hidden_states.device)

        # Launch Triton kernel 1: compute L
        compute_L_exp_cumsum_kernel[(Bsz, Csz)](
            A_cumsum, L,
            Bsz, Csz,
            stride_A_b=A_cumsum.stride(0), stride_A_h=A_cumsum.stride(1), stride_A_c=A_cumsum.stride(2), stride_A_s=A_cumsum.stride(3),
            stride_L_b=L.stride(0), stride_L_c=L.stride(1), stride_L_i=L.stride(2), stride_L_j=L.stride(3), stride_L_h=L.stride(4),
            S=128, H=32,
            num_warps=8, num_stages=2
        )

        # Launch Triton kernel 2: compute G
        contract_BC_to_G_kernel[(Bsz, Csz)](
            B, C, G,
            Bsz, Csz,
            stride_B_b=B.stride(0), stride_B_c=B.stride(1), stride_B_s=B.stride(2), stride_B_g=B.stride(3), stride_B_k=B.stride(4),
            stride_C_b=C.stride(0), stride_C_c=C.stride(1), stride_C_s=C.stride(2), stride_C_g=C.stride(3), stride_C_k=C.stride(4),
            stride_G_b=G.stride(0), stride_G_c=G.stride(1), stride_G_i=G.stride(2), stride_G_j=G.stride(3), stride_G_h=G.stride(4),
            S=128, H=32, G_CONST=8, K=32,
            num_warps=8, num_stages=2
        )

        # Launch Triton kernel 3: compute Y_diag = sum_j (G * L) * hidden over j
        contract_M_with_hidden_kernel[(Bsz, Csz, S, H, D)](
            G, L, hidden_states, Y_diag,
            Bsz, Csz, S, H, D,
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
