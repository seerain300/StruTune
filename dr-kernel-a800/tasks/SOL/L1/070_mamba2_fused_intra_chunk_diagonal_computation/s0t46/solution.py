import torch
import triton
import triton.language as tl


@triton.jit
def build_L_kernel(
    A_cumsum_ptr,  # [B, H, C, S]
    L_ptr,         # [B, C, S, S, H]
    Bsz, C, S, H,
    a_bs, a_h, a_cs, a_s,      # strides for A_cumsum
    l_b, l_c, l_i, l_j, l_h,   # strides for L
):
    # 1D grid over (b, c, i)
    pid = tl.program_id(0)
    i = pid % S
    tmp = pid // S
    c = tmp % C
    b = tmp // C

    for h in range(0, H):
        for j in range(0, S):
            include = i >= j
            # Compute cumsum_j[j] = sum_{t=0..j} A[b, h, c, t]
            cumsum_j = tl.zeros((), dtype=tl.float32)
            for t in range(0, S):
                if t <= j:
                    a_ptrs = A_cumsum_ptr + b * a_bs + h * a_h + c * a_cs + t * a_s
                    a_val = tl.load(a_ptrs)
                    cumsum_j += a_val
            l_val = 0.0
            if include:
                l_val = tl.exp(cumsum_j)
            l_ptrs = L_ptr + b * l_b + c * l_c + i * l_i + j * l_j + h * l_h
            tl.store(l_ptrs, l_val)


@triton.jit
def compute_G_kernel(
    B_exp_ptr,    # [B, C, S, H, N] (N is typically 128)
    C_exp_ptr,    # [B, C, S, H, N]
    G_ptr,        # [B, C, S, S, H]
    Bsz, C, S, H, N,
    be_bs, be_cs, be_s, be_h, be_n,
    ce_bs, ce_cs, ce_s, ce_h, ce_n,
    g_bs, g_cs, g_i, g_j, g_h,
):
    # 1D grid over (b, c, i, j)
    pid = tl.program_id(0)
    j = pid % S
    tmp = pid // S
    i = tmp % S
    tmp2 = tmp // S
    c = tmp2 % C
    b = tmp2 // C

    for h in range(0, H):
        acc = tl.zeros((), dtype=tl.float32)
        for n in range(0, N):  # N=128 by design
            inner = tl.zeros((), dtype=tl.float32)
            for jprime in range(0, S):
                be_ptrs = B_exp_ptr + b * be_bs + c * be_cs + jprime * be_s + h * be_h + n * be_n
                be_val = tl.load(be_ptrs)
                l_ptrs = L_ptr + b * l_bs + c * l_c + i * l_i + jprime * l_j + h * l_h
                l_val = tl.load(l_ptrs)
                inner += be_val * l_val
            ce_ptrs = C_exp_ptr + b * ce_bs + c * ce_cs + i * ce_s + h * ce_h + n * ce_n
            ce_val = tl.load(ce_ptrs)
            acc += ce_val * inner
        g_ptrs = G_ptr + b * g_bs + c * g_cs + i * g_i + j * g_j + h * g_h
        tl.store(g_ptrs, acc)


@triton.jit
def compute_Y_diag_kernel(
    G_ptr,        # [B, C, S, S, H]
    hidden_ptr,   # [B, C, S, H, D]
    Y_ptr,        # [B, C, S, H, D] (float32)
    Bsz, C, S, H, D,
    g_bs, g_cs, g_i, g_j, g_h,
    h_bs, h_cs, h_s, h_h, h_d,
    y_bs, y_cs, y_i, y_h, y_d,
    d_const: tl.constexpr,
):
    # 1D grid over (b, c, i, h)
    pid = tl.program_id(0)
    h = pid % H
    tmp = pid // H
    i = tmp % S
    tmp2 = tmp // S
    c = tmp2 % C
    b = tmp2 // C

    acc = tl.zeros((), dtype=tl.float32)
    for j in range(0, S):
        g_ptrs = G_ptr + b * g_bs + c * g_cs + i * g_i + j * g_j + h * g_h
        g_val = tl.load(g_ptrs)
        h_ptrs = hidden_ptr + b * h_bs + c * h_cs + j * h_s + h * h_h + d_const * h_d
        h_val = tl.load(h_ptrs)
        acc += g_val * h_val

    y_ptrs = Y_ptr + b * y_bs + c * y_cs + i * y_i + h * y_h + d_const * y_d
    tl.store(y_ptrs, acc)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states: torch.Tensor, A_cumsum: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        # Ensure CUDA and contiguous
        assert hidden_states.is_cuda and A_cumsum.is_cuda and B.is_cuda and C.is_cuda, "All inputs must be on CUDA"
        hidden_states = hidden_states.contiguous()
        A_cumsum = A_cumsum.contiguous()
        B = B.contiguous()
        C = C.contiguous()

        Bsz = hidden_states.shape[0]
        Cdim = hidden_states.shape[1]
        S = hidden_states.shape[2]
        H = hidden_states.shape[3]
        D = hidden_states.shape[4]

        # Parameters
        N_GROUPS = 8
        # The original code uses NUM_HEADS = 32 and N_GROUPS = 8, so H // N_GROUPS = 4
        # We expand along H by repeat_interleave(H // N_GROUPS)
        B_exp = B.repeat_interleave(H // N_GROUPS, dim=3)
        C_exp = C.repeat_interleave(H // N_GROUPS, dim=3)

        # Allocate L, G, Y
        L = torch.empty((Bsz, Cdim, S, S, H), device=hidden_states.device, dtype=torch.float32)
        G = torch.empty((Bsz, Cdim, S, S, H), device=hidden_states.device, dtype=torch.float32)
        Y = torch.empty((Bsz, Cdim, S, H, D), device=hidden_states.device, dtype=torch.float32)

        # Strides
        a_bs, a_h, a_cs, a_s = A_cumsum.stride()
        l_b, l_c, l_i, l_j, l_h = L.stride()
        be_bs, be_cs, be_s, be_h, be_n = B_exp.stride()
        ce_bs, ce_cs, ce_s, ce_h, ce_n = C_exp.stride()
        g_bs, g_cs, g_i, g_j, g_h = G.stride()
        h_bs, h_cs, h_s, h_h, h_d = hidden_states.stride()
        y_bs, y_cs, y_i, y_h, y_d = Y.stride()

        # 1) Build L = exp(cumsum(A_cumsum)) with causal mask
        grid_L = (Bsz * Cdim * S,)
        build_L_kernel[grid_L](A_cumsum, L, Bsz, Cdim, S, H, a_bs, a_h, a_cs, a_s, l_b, l_c, l_i, l_j, l_h, num_warps=1)

        # 2) Compute G
        grid_G = (Bsz * Cdim * S * S,)
        compute_G_kernel[grid_G](B_exp, C_exp, G, Bsz, Cdim, S, H, H, be_bs, be_cs, be_s, be_h, be_n, ce_bs, ce_cs, ce_s, ce_h, ce_n, g_bs, g_cs, g_i, g_j, g_h, num_warps=1)

        # 3) Compute Y_diag: sum_j G * hidden over j and d
        for d in range(D):
            grid_Y = (Bsz * Cdim * S * H,)
            compute_Y_diag_kernel[grid_Y](G, hidden_states, Y, Bsz, Cdim, S, H, D, g_bs, g_cs, g_i, g_j, g_h, h_bs, h_cs, h_s, h_h, h_d, y_bs, y_cs, y_i, y_h, y_d, d_const=d, num_warps=1)

        # Return in bfloat16
        return Y.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
