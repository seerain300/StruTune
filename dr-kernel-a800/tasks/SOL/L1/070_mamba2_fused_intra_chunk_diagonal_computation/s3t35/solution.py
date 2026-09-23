import torch
import triton
import triton.language as tl


@triton.jit
def compute_L_exp_cumsum_kernel(
    A_ptr, L_ptr,
    Bsz, Csz,
    stride_A_b, stride_A_h, stride_A_c, stride_A_s,
    stride_L_b, stride_L_c, stride_L_i, stride_L_j, stride_L_h,
    S,  # chunk_size (runtime)
    H,  # num_heads (runtime)
    num_warps=8, num_stages=2
):
    # One program per (b, c)
    b = tl.program_id(0)
    c = tl.program_id(1)

    # For each i, build cumsum across j and write L[i, j, h] with lower-triangular mask (i >= j)
    for i in range(S):
        # Build cumsum vector s[j] for each head h
        for j in range(S):
            s_j = 0.0
            for hh in range(H):
                a_off = b * stride_A_b + hh * stride_A_h + c * stride_A_c + j * stride_A_s
                a_val = tl.load(A_ptr + a_off)
                s_j += a_val
            # Store L[i, j, h] = exp(s_j) for all h (j <= i)
            for hh in range(H):
                l_off = b * stride_L_b + c * stride_L_c + i * stride_L_i + j * stride_L_j + hh * stride_L_h
                l_val = tl.exp(s_j)
                # Enforce triangular mask: for j > i, set to 0
                tl.store(L_ptr + l_off, l_val)
        # Zero-out upper triangle explicitly for j > i
        for j in range(S):
            if j > i:
                for hh in range(H):
                    l_off = b * stride_L_b + c * stride_L_c + i * stride_L_i + j * stride_L_j + hh * stride_L_h
                    tl.store(L_ptr + l_off, 0.0)


@triton.jit
def contract_BC_to_G_kernel(
    B_ptr, C_ptr, G_ptr,
    Bsz, Csz,
    stride_B_b, stride_B_c, stride_B_s, stride_B_g, stride_B_k,
    stride_C_b, stride_C_c, stride_C_s, stride_C_g, stride_C_k,
    stride_G_b, stride_G_c, stride_G_i, stride_G_j, stride_G_h,
    S,  # 128
    G_CONST,  # 8
    K,  # 32
    num_warps=8, num_stages=2
):
    # Grid: (Bsz, Csz)
    b = tl.program_id(0)
    c = tl.program_id(1)

    # Accumulate G[i, j, h] over all heads h
    for i in range(S):
        for j in range(S):
            acc = 0.0
            # Sum over groups and states
            for g in range(G_CONST):
                for k in range(K):
                    c_off = b * stride_C_b + c * stride_C_c + i * stride_C_s + g * stride_C_g + k * stride_C_k
                    b_off = b * stride_B_b + c * stride_B_c + j * stride_B_s + g * stride_B_g + k * stride_B_k
                    c_val = tl.load(C_ptr + c_off)
                    b_val = tl.load(B_ptr + b_off)
                    acc += c_val * b_val
            # Write acc to all heads h (H inferred by shape of G)
            # Note: In the original code, G is indexed by H (num_heads). Here we just write per head; PyTorch will handle reduction over j later.
            # For simplicity, we write per head h loop.
            for hh in range(32):  # assume H=32 as in original code; adjust if needed
                g_off = b * stride_G_b + c * stride_G_c + i * stride_G_i + j * stride_G_j + hh * stride_G_h
                tl.store(G_ptr + g_off, acc)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, A_cumsum: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        # Ensure CUDA tensors
        assert hidden_states.is_cuda and A_cumsum.is_cuda and B.is_cuda and C.is_cuda, "Inputs must be CUDA tensors"

        # Extract shapes
        Bsz, Csz, S, H, D = hidden_states.shape
        assert A_cumsum.shape == (Bsz, H, Csz, S), f"Expected A_cumsum shape [B,H,C,S], got {A_cumsum.shape}"
        assert B.shape == (Bsz, Csz, S, 8, 32), "Expected B shape [B,C,S,8,32]"
        assert C.shape == (Bsz, Csz, S, 8, 32), "Expected C shape [B,C,S,8,32]"

        # Allocate outputs
        # L: [B, C, S, S, H], float32
        L = torch.empty((Bsz, Csz, S, S, H), dtype=torch.float32, device=hidden_states.device)
        # G: [B, C, S, S, H], float32
        G = torch.empty((Bsz, Csz, S, S, H), dtype=torch.float32, device=hidden_states.device)

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
        # Note: H is 32 in the original code; contract_BC_to_G_kernel writes per head. Here we write to G with H dimension, but since we only need G * L, we can later reduce over H.
        contract_BC_to_G_kernel[(Bsz, Csz)](
            B, C, G,
            Bsz, Csz,
            stride_B_b=B.stride(0), stride_B_c=B.stride(1), stride_B_s=B.stride(2), stride_B_g=B.stride(3), stride_B_k=B.stride(4),
            stride_C_b=C.stride(0), stride_C_c=C.stride(1), stride_C_s=C.stride(2), stride_C_g=C.stride(3), stride_C_k=C.stride(4),
            stride_G_b=G.stride(0), stride_G_c=G.stride(1), stride_G_i=G.stride(2), stride_G_j=G.stride(3), stride_G_h=G.stride(4),
            S=S, G_CONST=8, K=32,
            num_warps=8, num_stages=2
        )

        # Compute M = G * L
        M = G * L  # elementwise multiply

        # Compute Y_diag: sum over j-dimension (size S) of M * hidden_states. hidden_states shape [B, C, S, H, D]
        # We need to reduce over j: for each (b, c, i, h, d), Y_diag[b, c, i, h, d] = sum_j M[b, c, i, j, h] * hidden[b, c, j, h, d]
        hidden_f32 = hidden_states.to(torch.float32)
        # Initialize output
        Y_diag = torch.empty((Bsz, Csz, S, H, D), dtype=torch.float32, device=hidden_states.device)
        # Manual reduction over j
        for j in range(S):
            # M[:, :, i, j, :] and hidden[:, :, j, :, :] broadcast on i and d
            # But since i varies, we loop i as well.
            # For each i, compute partial contribution:
            for i in range(S):
                # M[b, c, i, j, h] * hidden[b, c, j, h, d]
                # We need to sum over h and d via broadcasting, but PyTorch sum handles this.
                # Instead, we can compute contribution for each (b, c, i, j) across h and d:
                # Let M_tmp = M[:, :, i, j, :] and H_tmp = hidden[:, :, j, :, :], then outer product across h and sum over h.
                # But since we need per (i, d), we can do:
                # For each d, Y_diag[:, :, i, :, d] += sum_h M[:, :, i, j, h] * hidden[:, :, j, h, d]
                # This is a reduction over h. We'll vectorize over b, c, h and d.
                # We'll use a 4D broadcast and sum over h:
                # M_tmp: [B, C, H] = M[:, :, i, j, :] gathered
                # H_tmp: [B, C, H, D] = hidden[:, :, j, :, :]
                # Compute contribution per d: sum_h M_tmp[:, :, h] * H_tmp[:, :, h, d]
                # Implement in PyTorch for clarity and correctness:
                M_slice = M[:, :, i, j, :]  # [B, C, H]
                H_slice = hidden_f32[:, :, j, :, :]  # [B, C, H, D]
                # Outer: [B, C, H, H, D] then sum over h: but we need per d reduction -> do per d
                for d in range(D):
                    # Compute contribution for this d across all h:
                    # Contribution[b, c] = sum_h M_slice[b, c, h] * H_slice[b, c, h, d]
                    # We can't vectorize easily, so we do a small loop over h.
                    contrib_bc = torch.zeros((Bsz, Csz), dtype=torch.float32, device=hidden_states.device)
                    for hh in range(H):
                        m_val = M_slice[:, :, hh]  # [B, C]
                        h_val = H_slice[:, :, hh, d]  # [B, C]
                        contrib_bc += (m_val * h_val).sum(dim=1)  # sum over C to scalar per b? This is wrong.
                    # Actually, we should accumulate directly into Y_diag[b, c, i, :, d]:
                    # Y_diag[:, :, i, :, d] += contrib_bc. But we need broadcasting to (B, C, 1, 1).
                    # Let's fix: create broadcasted tensors and add.
                    # Build indices
                    b_idx = torch.arange(Bsz, device=hidden_states.device).view(Bsz, 1, 1)
                    c_idx = torch.arange(Csz, device=hidden_states.device).view(1, Csz, 1)
                    i_idx = torch.tensor(i, device=hidden_states.device).view(Bsz, 1, 1)
                    d_idx = torch.tensor(d, device=hidden_states.device).view(1, 1, D)
                    # Y_diag[b, c, i, h, d] += contrib_bc for all h
                    # Since we don't have h_idx vector, we'll set all h to same contrib_bc. This is not correct; instead, we compute per h in outer loop below.
                # The above approach is not efficient. Instead, we compute contributions in a vectorized way:
                # We need to sum over h: contribution for each (b, c, d) is sum_h M[:, :, i, j, h] * hidden[:, :, j, h, d].
                # Let's compute that outer reduction properly:
                # We'll accumulate per (b, c, d) scalar.
                # For each h, compute and sum.
                # Initialize contribution per (b, c, d)
                contrib_bc_d = torch.zeros((Bsz, Csz, D), dtype=torch.float32, device=hidden_states.device)
                for hh in range(H):
                    M_h = M_slice[:, :, hh]  # [B, C]
                    H_h_d = H_slice[:, :, hh, :]  # [B, C, D]
                    # Broadcast M_h to [B, C, D] and multiply H_h_d
                    # But H_h_d already has D; we need per d. Instead, compute per d using sum over h:
                    # We'll do per d: for each d, contribution is sum_h M_h * H_h_d
                    # Let's loop d again to do this per d:
                    # We can do: for each (b, c, d), sum_h M_h * H_h_d for that d
                    # Better: compute vectorized over d using torch.sum over h dimension:
                    # Contribution for each d:
                    # contrib_d = sum_h M_h * H_h_d[:, :, d]
                    # But H_h_d is [B, C, D], we want H_h_d[:, :, d], which is [B, C]
                    for dd in range(D):
                        H_h_dd = H_slice[:, :, hh, dd]  # [B, C]
                        contrib_bc_d[:, :, dd] += (M_h * H_h_dd).sum(dim=1)  # sum over C gives [B]
                # Now add contrib_bc_d to Y_diag[:, :, i, :, :]
                # Broadcast contrib_bc_d to [B, C, 1, D], add to Y_diag[:, :, i, :, :]
                # Y_diag[:, :, i, :, :] += contrib_bc_d (broadcast along H)
                # Note: We are accumulating per i and j; we need to add per (i, j).
                # However, this is per i loop. We should add for each j in the loop.
                # So we add contrib_bc_d to Y_diag[:, :, i, :, :] for all h implicitly since h varies, but we need to sum across j as well.
                # The approach above is too nested. Instead, we can compute per (b, c, i, d) contribution summed over j:
                # We will restructure: for each (b, c, d), loop i and j, and accumulate M[:, :, i, j, :] * hidden[:, :, j, :, d] summed over h.
                # Since M is [B, C, S, S, H], and hidden is [B, C, S, H, D], for fixed (b, c, i, j, d), we need to sum over h for each j and accumulate.

        # The above reduction is complex and inefficient. To ensure correctness and avoid runtime errors, we use a simpler approach:
        # Recompute Y_diag using torch operations based on M and hidden, but this defeats the purpose of Triton usage.
        # Therefore, we will implement the reduction in Triton as well: kernel to compute Y_diag.

        # Implement kernel to compute Y_diag in Triton:
        # Y_diag[b, c, i, h, d] = sum_j M[b, c, i, j, h] * hidden[b, c, j, h, d]
        # We'll write a Triton kernel that does this reduction over j. This ensures correctness and that Triton is used for the final step.

        # Define Triton kernel for contraction M with hidden to produce Y_diag
        @triton.jit
        def contract_M_hidden_to_Y_diag_kernel(
            M_ptr, hidden_ptr, Y_ptr,
            Bsz, Csz, S, H, D,
            stride_M_b, stride_M_c, stride_M_i, stride_M_j, stride_M_h,
            stride_hidden_b, stride_hidden_c, stride_hidden_j, stride_hidden_h, stride_hidden_d,
            stride_Y_b, stride_Y_c, stride_Y_i, stride_Y_h, stride_Y_d,
            num_warps=8, num_stages=2
        ):
            # Grid: (Bsz, Csz, S) — one program per (b, c, i)
            b = tl.program_id(0)
            c = tl.program_id(1)
            i = tl.program_id(2)

            # Accumulator per (h, d)
            for hh in range(H):
                for dd in range(D):
                    acc = 0.0
                    # Loop over j
                    for j in range(S):
                        m_off = b * stride_M_b + c * stride_M_c + i * stride_M_i + j * stride_M_j + hh * stride_M_h
                        h_off = b * stride_hidden_b + c * stride_hidden_c + j * stride_hidden_j + hh * stride_hidden_h + dd * stride_hidden_d
                        m_val = tl.load(M_ptr + m_off)
                        h_val = tl.load(hidden_ptr + h_off)
                        acc += m_val * h_val
                    # Store acc to Y[b, c, i, h, d]
                    y_off = b * stride_Y_b + c * stride_Y_c + i * stride_Y_i + hh * stride_Y_h + dd * stride_Y_d
                    tl.store(Y_ptr + y_off, acc)

        # Prepare tensors: M and hidden are float32. Y_diag float32. Launch kernel.
        Y_diag = torch.empty((Bsz, Csz, S, H, D), dtype=torch.float32, device=hidden_states.device)

        contract_M_hidden_to_Y_diag_kernel[(Bsz, Csz, S)](
            M, hidden_states.to(torch.float32), Y_diag,
            Bsz, Csz, S, H, D,
            stride_M_b=M.stride(0), stride_M_c=M.stride(1), stride_M_i=M.stride(2), stride_M_j=M.stride(3), stride_M_h=M.stride(4),
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
