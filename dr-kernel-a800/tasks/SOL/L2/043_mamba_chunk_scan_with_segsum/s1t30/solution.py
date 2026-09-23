import torch
import triton
import triton.language as tl


@triton.jit
def pad_last_dim_1D(inp_ptr, out_ptr,
                    Bsz, S, H, D,
                    inp_stride_b, inp_stride_s, inp_stride_h, inp_stride_d,
                    out_stride_b, out_stride_s, out_stride_h, out_stride_d):
    # Grid over (b, s, h, d)
    b = tl.program_id(0)
    s = tl.program_id(1)
    h = tl.program_id(2)
    d = tl.program_id(3)

    pad = (256 - S % 256) % 256  # chunk_size
    S_padded = S + pad
    out_idx = tl.where(s < S, s, s - S + pad)  # map padded index
    val = tl.load(inp_ptr + b * inp_stride_b + s * inp_stride_s + h * inp_stride_h + d * inp_stride_d)
    tl.store(out_ptr + b * out_stride_b + out_idx * out_stride_s + h * out_stride_h + d * out_stride_d, val)


@triton.jit
def cumsum_exp_diff(A_ptr, Out_ptr,
                    Bsz, NC, H, N,
                    a_stride_b, a_stride_nc, a_stride_t, a_stride_h,
                    out_stride_b, out_stride_nc, out_stride_t, out_stride_h):
    # Grid over (b, nc, h)
    b = tl.program_id(0)
    nc = tl.program_id(1)
    h = tl.program_id(2)

    acc = tl.zeros((), dtype=tl.float32)
    for t in range(0, N):
        val = tl.load(A_ptr + b * a_stride_b + nc * a_stride_nc + t * a_stride_t + h * a_stride_h)
        acc = acc + val
        tl.store(Out_ptr + b * out_stride_b + nc * out_stride_nc + t * out_stride_t + h * out_stride_h, acc)

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
    # Grid over (b, nc, i, h)
    b = tl.program_id(0)
    nc = tl.program_id(1)
    i = tl.program_id(2)
    h = tl.program_id(3)

    acc = tl.zeros((), dtype=tl.float32)
    for j in range(0, N):
        if j <= i:
            val = tl.load(A_ptr + b * a_stride_b + nc * a_stride_nc + j * a_stride_i + h * a_stride_h)
            acc = acc + val
            tl.store(L_ptr + b * l_stride_b + nc * l_stride_nc + i * l_stride_i + j * l_stride_j + h * l_stride_h,
                     tl.exp(acc))
        else:
            tl.store(L_ptr + b * l_stride_b + nc * l_stride_nc + i * l_stride_i + j * l_stride_j + h * l_stride_h,
                     1.0)


@triton.jit
def contraction_CxB(C_ptr, B_ptr, G_ptr,
                    Bsz, NC, N, H, S,
                    c_stride_b, c_stride_nc, c_stride_t, c_stride_h, c_stride_s,
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
        c_val = tl.load(C_ptr + b * c_stride_b + nc * c_stride_nc + i * c_stride_t + h * c_stride_h + s * c_stride_s)
        b_val = tl.load(B_ptr + b * b_stride_b + nc * b_stride_nc + j * b_stride_j + h * b_stride_h + s * b_stride_s)
        acc += c_val * b_val
    tl.store(G_ptr + b * g_stride_b + nc * g_stride_nc + i * g_stride_i + j * g_stride_j + h * g_stride_h, acc)


@triton.jit
def diagonal_output(G_ptr, Hid_ptr, Y_ptr,
                    Bsz, NC, N, H, D,
                    g_stride_b, g_stride_nc, g_stride_i, g_stride_j, g_stride_h,
                    h_stride_b, h_stride_nc, h_stride_t, h_stride_h, h_stride_d,
                    y_stride_b, y_stride_nc, y_stride_i, y_stride_h, y_stride_d):
    # Grid over (b, nc, i, h, d)
    b = tl.program_id(0)
    nc = tl.program_id(1)
    i = tl.program_id(2)
    h = tl.program_id(3)
    d = tl.program_id(4)

    acc = tl.zeros((), dtype=tl.float32)
    for j in range(0, N):
        G_val = tl.load(G_ptr + b * g_stride_b + nc * g_stride_nc + i * g_stride_i + j * g_stride_j + h * g_stride_h)
        H_val = tl.load(Hid_ptr + b * h_stride_b + nc * h_stride_nc + j * h_stride_t + h * h_stride_h + d * h_stride_d)
        acc += G_val * H_val
    tl.store(Y_ptr + b * y_stride_b + nc * y_stride_nc + i * y_stride_i + h * y_stride_h + d * y_stride_d, acc)


@triton.jit
def inter_chunk_propagate(Decay_ptr, States_ptr, NewStates_ptr,
                           Bsz, NC, H, S, N,
                           d_stride_b, d_stride_h, d_stride_i, d_stride_j,
                           st_stride_b, st_stride_nc, st_stride_t, st_stride_h, st_stride_d, st_stride_s,
                           ns_stride_b, ns_stride_nc, ns_stride_t, ns_stride_h, ns_stride_d, ns_stride_s):
    # Grid over (b, i, h, d, s) — reduction over j
    b = tl.program_id(0)
    i = tl.program_id(1)
    h = tl.program_id(2)
    d = tl.program_id(3)
    s = tl.program_id(4)

    acc = tl.zeros((), dtype=tl.float32)
    for j in range(0, N):
        decay_val = tl.load(Decay_ptr + b * d_stride_b + h * d_stride_h + i * d_stride_i + j * d_stride_j)
        # states_with_init: initial at nc=0, t=j
        St_val = tl.load(States_ptr + b * st_stride_b + 0 * st_stride_nc + j * st_stride_t + h * st_stride_h +
                         d * st_stride_d + s * st_stride_s)
        acc += decay_val * St_val
    tl.store(NewStates_ptr + b * ns_stride_b + 0 * ns_stride_nc + i * ns_stride_t + h * ns_stride_h + d * ns_stride_d +
            s * ns_stride_s, acc)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, D: torch.Tensor,
                initial_states: torch.Tensor):
        # Shapes from the original problem
        Bsz = hidden_states.shape[0]
        S = hidden_states.shape[1]
        H = 16  # num_heads
        D = 64  # head_dim
        Sstate = 256  # state_size
        N = 256  # chunk_size

        device = hidden_states.device
        # Convert to float32
        hidden_states = hidden_states.to(device).to(torch.float32)
        A = A.to(device).to(torch.float32)
        B = B.to(device).to(torch.float32)
        C = C.to(device).to(torch.float32)
        D = D.to(device).to(torch.float32)
        initial_states = initial_states.to(device).to(torch.float32)

        # 1) Pad hidden states along seq_len to multiple of chunk_size
        S_padded = (S + (N - S % N) % N)
        hidden_padded = torch.empty((Bsz, S_padded, H, D), device=device, dtype=torch.float32)

        # Launch Triton pad_last_dim_1D
        grid_pad = (Bsz, S_padded, H, D)
        pad_last_dim_1D[grid_pad](
            hidden_states, hidden_padded,
            Bsz, S, H, D,
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2), hidden_states.stride(3),
            hidden_padded.stride(0), hidden_padded.stride(1), hidden_padded.stride(2), hidden_padded.stride(3)
        )

        # Reshape into chunks: [B, NC, N, H, D]
        NC = (S_padded + N - 1) // N
        hidden_chunked = hidden_padded.reshape(Bsz, NC, N, H, D)
        A_transposed = A.transpose(1, 2)  # [B, S, H]
        A_chunked = A_transposed.reshape(Bsz, NC, N, H)  # [B, NC, N, H]

        # 2) A_cumsum via Triton: grid over (B, NC, H)


def run(*args):
    return ModelNew()(*args)
