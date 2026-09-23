class ModelNew(torch.nn.Module):
    def forward(self, hidden_states, A_cumsum, B, C):
        # hidden_states: [B, N, S, H, D]
        # A_cumsum: [B, H, N, S]
        # B, C: [B, N, S, H, D] (already expanded from G to H in caller)

        assert hidden_states.is_cuda and A_cumsum.is_cuda and B.is_cuda and C.is_cuda, "Triton kernels require CUDA tensors."
        hidden = hidden_states.contiguous()
        A = A_cumsum.contiguous()
        Bx = B.contiguous()
        Cx = C.contiguous()

        Bsz, Nsz, S, H, D = hidden.shape
        assert A.shape == (Bsz, H, Nsz, S), "A_cumsum must be [B, H, N, S]"

        # 1) Triton: Compute L = exp(cumsum(masked A_expanded)) with tril(diagonal=-1) over (S,S)
        L = torch.empty((Bsz, H, Nsz, S, S), dtype=torch.float32, device=hidden.device)

        A_stride_b, A_stride_h, A_stride_n, A_stride_s = A.stride()
        L_stride_b, L_stride_h, L_stride_n, L_stride_s1, L_stride_s2 = L.stride()

        a_tril_cumsum_exp_kernel[(Bsz, H, Nsz)](
            A, L,
            Bsz, H, Nsz, S,
            A_stride_b, A_stride_h, A_stride_n, A_stride_s,
            L_stride_b, L_stride_h, L_stride_n, L_stride_s1, L_stride_s2,
            num_warps=1, num_stages=1
        )

        # 2) Triton: G contraction: G[i,j,h] = sum_d C[i,h]*B[j,h]
        G = torch.empty((Bsz, Nsz, S, S, H), dtype=torch.float32, device=hidden.device)

        B_stride_b, B_stride_n, B_stride_s, B_stride_h, B_stride_d = Bx.stride()
        C_stride_b, C_stride_n, C_stride_s, C_stride_h, C_stride_d = Cx.stride()
        G_stride_b, G_stride_n, G_stride_s1, G_stride_s2, G_stride_h = G.stride()

        g_contract_kernel[(Bsz, Nsz, S, S, H)](
            Bx, Cx, G,
            B_stride_b, B_stride_n, B_stride_s, B_stride_h, B_stride_d,
            C_stride_b, C_stride_n, C_stride_s, C_stride_h, C_stride_d,
            G_stride_b, G_stride_n, G_stride_s1, G_stride_s2, G_stride_h,
            BLOCK_D=32,
            num_warps=1, num_stages=1
        )

        # 3) Triton: M = G * L
        M = torch.empty_like(G, dtype=torch.float32, device=hidden.device)

        g_stride_b, g_stride_n, g_stride_s1, g_stride_s2, g_stride_h = G.stride()
        l_stride_b, l_stride_h, l_stride_n, l_stride_s1, l_stride_s2 = L.stride()
        m_stride_b, m_stride_n, m_stride_s1, m_stride_s2, m_stride_h = M.stride()

        m_mul_kernel[(Bsz, Nsz, S, S, H)](
            G, L, M,
            g_stride_b, g_stride_n, g_stride_s1, g_stride_s2, g_stride_h,
            l_stride_b, l_stride_h, l_stride_n, l_stride_s1, l_stride_s2,
            m_stride_b, m_stride_n, m_stride_s1, m_stride_s2, m_stride_h,
            num_warps=1, num_stages=1
        )

        # 4) Triton: Y[b,n,i,h] = sum_j M[b,n,i,j,h] * hidden[b,n,j,h]
        Y = torch.empty((Bsz, Nsz, S, H), dtype=torch.float32, device=hidden.device)

        M_stride_b, M_stride_n, M_stride_s1, M_stride_s2, M_stride_h = M.stride()
        hidden_stride_b, hidden_stride_n, hidden_stride_s, hidden_stride_h, hidden_stride_d = hidden.stride()
        Y_stride_b, Y_stride_n, Y_stride_s1, Y_stride_h = Y.stride()

        y_diag_reduce_kernel[(Bsz, Nsz, S, H)](
            M, hidden, Y,
            Bsz, Nsz, S, H, D,
            M_stride_b, M_stride_n, M_stride_s1, M_stride_s2, M_stride_h,
            hidden_stride_b, hidden_stride_n, hidden_stride_s, hidden_stride_h, hidden_stride_d,
            Y_stride_b, Y_stride_n, Y_stride_s1, Y_stride_h,
            BLOCK_D=32,
            num_warps=1, num_stages=1
        )

        return Y.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
