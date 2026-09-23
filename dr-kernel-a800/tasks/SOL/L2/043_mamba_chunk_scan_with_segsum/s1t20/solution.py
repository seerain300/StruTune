import torch
import triton
import triton.language as tl


@triton.jit
def cumsum_exp_diff(A_ptr, Out_ptr,
                    Bsz, NC, H, N,
                    a_stride_b, a_stride_nc, a_stride_t, a_stride_h,
                    out_stride_b, out_stride_nc, out_stride_t, out_stride_h):
    # Grid over (b, nc, h)
    b = tl.program_id(0)
    nc = tl.program_id(1)
    h = tl.program_id(2)

    # Inclusive scan along t in [0, N)
    acc = tl.zeros((), dtype=tl.float32)
    for t in range(0, N):
        val = tl.load(A_ptr + b * a_stride_b + nc * a_stride_nc + t * a_stride_t + h * a_stride_h)
        acc = acc + val
        tl.store(Out_ptr + b * out_stride_b + nc * out_stride_nc + t * out_stride_t + h * out_stride_h, acc)

    # Compute exp( last - current ) for each t
    last = tl.load(Out_ptr + b * out_stride_b + nc * out_stride_nc + (N - 1) * out_stride_t + h * out_stride_h)
    for t in range(0, N):
        curr = tl.load(Out_ptr + b * out_stride_b + nc * out_stride_nc + t * out_stride_t + h * out_stride_h)
        diff = last - curr
        tl.store(Out_ptr + b * out_stride_b + nc * out_stride_nc + t * out_stride_t + h * out_stride_h, tl.exp(diff))


@triton.jit
def segment_sum_lower_tri_scan(A_ptr, L_ptr,
                                Bsz, NC, H, N,
                                a_stride_b, a_stride_nc, a_stride_i, a_stride_h,
                                l_stride_b, l_stride_nc, l_stride_i, l_stride_j, l_stride_h):
    # Grid over (b, nc, h, i) with i in [0, N)
    b = tl.program_id(0)
    nc = tl.program_id(1)
    h = tl.program_id(2)
    i = tl.program_id(3)

    # Masked inclusive scan along j in [0, N), only j <= i contributes
    acc = tl.zeros((), dtype=tl.float32)
    for j in range(0, N):
        contrib = tl.where(j <= i, tl.load(A_ptr + b * a_stride_b + nc * a_stride_nc + j * a_stride_i + h * a_stride_h), 0.0)
        acc = acc + contrib
        exp_acc = tl.exp(acc)
        tl.store(L_ptr + b * l_stride_b + nc * l_stride_nc + i * l_stride_i + j * l_stride_j + h * l_stride_h, exp_acc)


@triton.jit
def contraction_CxB(C_ptr, B_ptr, G_ptr,
                    Bsz, NC, H, N, S,
                    c_stride_b, c_stride_nc, c_stride_i, c_stride_h, c_stride_s,
                    b_stride_b, b_stride_nc, b_stride_j, b_stride_h, b_stride_s,
                    g_stride_b, g_stride_nc, g_stride_i, g_stride_j, g_stride_h):
    # Grid over (b, nc, i, j, h)
    b = tl.program_id(0)
    nc = tl.program_id(1)
    i = tl.program_id(2)
    j = tl.program_id(3)
    h = tl.program_id(4)

    acc = tl.zeros((), dtype=tl.float32)
    for s in range(0, S):
        c_val = tl.load(C_ptr + b * c_stride_b + nc * c_stride_nc + i * c_stride_i + h * c_stride_h + s * c_stride_s)
        b_val = tl.load(B_ptr + b * b_stride_b + nc * b_stride_nc + j * b_stride_j + h * b_stride_h + s * b_stride_s)
        acc += c_val * b_val

    g_ptrs = G_ptr + b * g_stride_b + nc * g_stride_nc + i * g_stride_i + j * g_stride_j + h * g_stride_h
    tl.store(g_ptrs, acc)


@triton.jit
def diagonal_output(M_ptr, hidden_ptr, Y_ptr,
                    Bsz, NC, H, N, D,
                    m_stride_b, m_stride_nc, m_stride_i, m_stride_j, m_stride_h,
                    h_stride_b, h_stride_nc, h_stride_j, h_stride_h, h_stride_d,
                    y_stride_b, y_stride_nc, y_stride_i, y_stride_h, y_stride_d):
    # Grid over (b, nc, i, h, d)
    b = tl.program_id(0)
    nc = tl.program_id(1)
    i = tl.program_id(2)
    h = tl.program_id(3)
    d = tl.program_id(4)

    acc = tl.zeros((), dtype=tl.float32)
    for j in range(0, N):
        m_val = tl.load(M_ptr + b * m_stride_b + nc * m_stride_nc + i * m_stride_i + j * m_stride_j + h * m_stride_h)
        h_val = tl.load(hidden_ptr + b * h_stride_b + nc * h_stride_nc + j * h_stride_j + h * h_stride_h + d * h_stride_d)
        acc += m_val * h_val
    y_ptrs = Y_ptr + b * y_stride_b + nc * y_stride_nc + i * y_stride_i + h * y_stride_h + d * y_stride_d
    tl.store(y_ptrs, acc)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,
                hidden_states: torch.Tensor,
                A: torch.Tensor,
                B: torch.Tensor,
                C: torch.Tensor,
                D: torch.Tensor,
                initial_states: torch.Tensor):
        # Shapes as in original code
        Bsz, seq_len, num_heads, head_dim = hidden_states.shape
        state_size = 256
        chunk_size = 256

        # Pad seq_len to multiple of chunk_size
        pad_size = (chunk_size - seq_len % chunk_size) % chunk_size
        seq_len_padded = seq_len + pad_size
        num_chunks = seq_len_padded // chunk_size

        # Cast to float32
        hidden_f = hidden_states.to(torch.float32)
        A_f = A.to(torch.float32)
        B_f = B.to(torch.float32)
        C_f = C.to(torch.float32)
        D_f = D.to(torch.float32)
        initial_f = initial_states.to(torch.float32)  # [B, 16, 64, 256]

        # Pad hidden along last dim (seq_len)
        hidden_padded = torch.nn.functional.pad(
            hidden_f, (0, 0, 0, 0, 0, pad_size, 0, 0),
            mode='constant', value=0
        )  # [B, seq_len_padded, 16, 64]

        # Expand B and C to match num_heads
        B_expanded = B_f.expand(Bsz, seq_len_padded, num_heads, state_size)  # [B, S, 16, 256]
        C_expanded = C_f.expand(Bsz, seq_len_padded, num_heads, state_size)  # [B, S, 16, 256]

        # Reshape into chunks
        hidden_chunked = hidden_padded.reshape(Bsz, num_chunks, chunk_size, num_heads, head_dim)
        B_chunked = B_expanded.reshape(Bsz, num_chunks, chunk_size, num_heads, state_size)
        C_chunked = C_expanded.reshape(Bsz, num_chunks, chunk_size, num_heads, state_size)

        # A handling: [B, S, 16] -> A_perm [B, NC, N, 16]
        A_transposed = A_f.transpose(1, 2).contiguous()  # [B, S, 16]
        A_perm = A_transposed.reshape(Bsz, num_chunks, chunk_size, num_heads)  # [B, NC, N, H]

        # 1) cumsum_exp_diff: computes exp(A_cumsum[:, :, :, -1:] - A_cumsum)
        A_cum_out = torch.empty((Bsz, num_chunks, chunk_size, num_heads), dtype=torch.float32, device=hidden_f.device)
        grid_cd = (Bsz, num_chunks, num_heads)
        cumsum_exp_diff[grid_cd](
            A_perm,
            A_cum_out,
            Bsz, num_chunks, num_heads, chunk_size,
            A_perm.stride(0), A_perm.stride(1), A_perm.stride(2), A_perm.stride(3),
            A_cum_out.stride(0), A_cum_out.stride(1), A_cum_out.stride(2), A_cum_out.stride(3),
            num_warps=4
        )

        # 2) segment_sum_lower_tri_scan: L[b, nc, i, j, h] = exp(sum_{k<=i} A_perm[b, nc, k, h])
        L = torch.empty((Bsz, num_chunks, chunk_size, chunk_size, num_heads), dtype=torch.float32, device=hidden_f.device)
        grid_sl = (Bsz, num_chunks, num_heads, chunk_size)
        segment_sum_lower_tri_scan[grid_sl](
            A_perm,
            L,
            Bsz, num_chunks, num_heads, chunk_size,
            A_perm.stride(0), A_perm.stride(1), A_perm.stride(2), A_perm.stride(3),
            L.stride(0), L.stride(1), L.stride(2), L.stride(3), L.stride(4),
            num_warps=4
        )

        # 3) Contraction G = sum_s C[i, s] * B[j, s]
        G = torch.empty((Bsz, num_chunks, chunk_size, chunk_size, num_heads), dtype=torch.float32, device=hidden_f.device)
        grid_cx = (Bsz, num_chunks, chunk_size, chunk_size, num_heads)
        contraction_CxB[grid_cx](
            C_chunked, B_chunked, G,
            Bsz, num_chunks, num_heads, chunk_size, state_size,
            C_chunked.stride(0), C_chunked.stride(1), C_chunked.stride(2), C_chunked.stride(3), C_chunked.stride(4),
            B_chunked.stride(0), B_chunked.stride(1), B_chunked.stride(2), B_chunked.stride(3), B_chunked.stride(4),
            G.stride(0), G.stride(1), G.stride(2), G.stride(3), G.stride(4),
            num_warps=4
        )

        # 4) Diagonal output: Y[b, nc, i, h, d] = sum_j G[b, nc, i, j,


def run(*args):
    return ModelNew()(*args)
