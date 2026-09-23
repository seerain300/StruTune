import torch
import triton
import triton.language as tl


@triton.jit
def masked_cumsum_lower_exp(
    A: tl.pointer_type(tl.float32),  # [B, C, S, N]
    L: tl.pointer_type(tl.float32),  # [B, C, S, S, N]
    Bsz: tl.int32, Csz: tl.int32, S: tl.int32, N: tl.int32,
    A_stride_b: tl.int32, A_stride_c: tl.int32, A_stride_s: tl.int32, A_stride_n: tl.int32,
    L_stride_b: tl.int32, L_stride_c: tl.int32, L_stride_i: tl.int32, L_stride_j: tl.int32, L_stride_n: tl.int32,
):
    # program ids
    b = tl.program_id(0)  # batch
    c = tl.program_id(1)  # chunk
    n = tl.program_id(2)  # head
    i = tl.program_id(3)  # target index along S (i in [0..S-1])
    # running cumsum
    running_sum = 0.0
    for j in range(S):  # source index along S
        include = j < i  # lower-triangular with diagonal=-1 (exclude j == i)
        a_val = tl.load(
            A + b * A_stride_b + c * A_stride_c + j * A_stride_s + n * A_stride_n,
            mask=True,
            other=0.0
        )
        running_sum = running_sum + a_val if include else running_sum
        tl.store(
            L + b * L_stride_b + c * L_stride_c + i * L_stride_i + j * L_stride_j + n * L_stride_n,
            tl.exp(running_sum),
            mask=True
        )


@triton.jit
def contract_BC_to_G(
    B_exp: tl.pointer_type(tl.float32),  # [B, C, S, N, D]
    C_exp: tl.pointer_type(tl.float32),  # [B, C, S, N, D]
    G: tl.pointer_type(tl.float32),      # [B, C, S, S, N]
    Bsz: tl.int32, Csz: tl.int32, S: tl.int32, N: tl.int32, D: tl.int32,
    K: tl.constexpr,                      # head_dim, compile-time constant for loop
    B_exp_stride_b: tl.int32, B_exp_stride_c: tl.int32, B_exp_stride_s: tl.int32, B_exp_stride_n: tl.int32, B_exp_stride_d: tl.int32,
    C_exp_stride_b: tl.int32, C_exp_stride_c: tl.int32, C_exp_stride_s: tl.int32, C_exp_stride_n: tl.int32, C_exp_stride_d: tl.int32,
    G_stride_b: tl.int32, G_stride_c: tl.int32, G_stride_i: tl.int32, G_stride_j: tl.int32, G_stride_n: tl.int32,
):
    b = tl.program_id(0)  # batch
    c = tl.program_id(1)  # chunk
    i = tl.program_id(2)  # i in [0..S-1]
    j = tl.program_id(3)  # j in [0..S-1]
    n = tl.program_id(4)  # head
    acc = 0.0
    for k in range(K):
        bk = tl.load(
            B_exp + b * B_exp_stride_b + c * B_exp_stride_c + j * B_exp_stride_s + n * B_exp_stride_n + k * B_exp_stride_d,
            mask=True,
            other=0.0
        )
        ck = tl.load(
            C_exp + b * C_exp_stride_b + c * C_exp_stride_c + i * C_exp_stride_s + n * C_exp_stride_n + k * C_exp_stride_d,
            mask=True,
            other=0.0
        )
        acc += bk * ck
    tl.store(G + b * G_stride_b + c * G_stride_c + i * G_stride_i + j * G_stride_j + n * G_stride_n, acc)


@triton.jit
def diag_contract_Y(
    M: tl.pointer_type(tl.float32),      # [B, C, S, S, N]
    hidden: tl.pointer_type(tl.float32), # [B, C, S, N, D]
    Y: tl.pointer_type(tl.float32),      # [B, C, S, N, D]
    Bsz: tl.int32, Csz: tl.int32, S: tl.int32, N: tl.int32, D: tl.int32,
    M_stride_b: tl.int32, M_stride_c: tl.int32, M_stride_i: tl.int32, M_stride_j: tl.int32, M_stride_n: tl.int32,
    hidden_stride_b: tl.int32, hidden_stride_c: tl.int32, hidden_stride_j: tl.int32, hidden_stride_n: tl.int32, hidden_stride_d: tl.int32,
    Y_stride_b: tl.int32, Y_stride_c: tl.int32, Y_stride_i: tl.int32, Y_stride_n: tl.int32, Y_stride_d: tl.int32,
):
    b = tl.program_id(0)  # batch
    c = tl.program_id(1)  # chunk
    i = tl.program_id(2)  # i in [0..S-1]
    n = tl.program_id(3)  # head
    d = tl.program_id(4)  # feature dim
    acc = 0.0
    for j in range(S):
        m_val = tl.load(M + b * M_stride_b + c * M_stride_c + i * M_stride_i + j * M_stride_j + n * M_stride_n, mask=True, other=0.0)
        h_val = tl.load(hidden + b * hidden_stride_b + c * hidden_stride_c + j * hidden_stride_j + n * hidden_stride_n + d * hidden_stride_d, mask=True, other=0.0)
        acc += m_val * h_val
    tl.store(Y + b * Y_stride_b + c * Y_stride_c + i * Y_stride_i + n * Y_stride_n + d * Y_stride_d, acc)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.NUM_HEADS = 32
        self.N_GROUPS = 8

    def forward(self, hidden_states: torch.Tensor, A_cumsum: torch.Tensor, B: torch.Tensor, C: torch.Tensor):
        # Ensure device and dtype handling
        device = hidden_states.device
        # Infer shapes
        Bsz, Csz, S, N, D = hidden_states.shape

        # Prepare expanded B and C to num_heads=32 by repeat_interleave(NUM_HEADS // N_GROUPS) = 4
        B_expanded = B.repeat_interleave(self.NUM_HEADS // self.N_GROUPS, dim=3).to(torch.float32)
        C_expanded = C.repeat_interleave(self.NUM_HEADS // self.N_GROUPS, dim=3).to(torch.float32)

        # Allocate L: [B, C, S, S, N] float32
        L = torch.empty((Bsz, Csz, S, S, N), device=device, dtype=torch.float32)
        # Allocate G: [B, C, S, S, N] float32
        G = torch.empty((Bsz, Csz, S, S, N), device=device, dtype=torch.float32)
        # Allocate Y: [B, C, S, N, D] float32
        Y = torch.empty((Bsz, Csz, S, N, D), device=device, dtype=torch.float32)

        # Launch masked_cumsum_lower_exp
        A = A_cumsum.to(torch.float32)
        A_stride_b, A_stride_c, A_stride_s, A_stride_n = A.stride()
        L_stride_b, L_stride_c, L_stride_i, L_stride_j, L_stride_n = L.stride()
        grid_L = (Bsz, Csz, N, S)  # program over (b, c, n, i)
        masked_cumsum_lower_exp[grid_L](
            A, L,
            Bsz, Csz, S, N,
            A_stride_b, A_stride_c, A_stride_s, A_stride_n,
            L_stride_b, L_stride_c, L_stride_i, L_stride_j, L_stride_n,
            num_warps=1, num_stages=1
        )

        # Launch contract_BC_to_G
        B_exp_stride_b, B_exp_stride_c, B_exp_stride_s, B_exp_stride_n, B_exp_stride_d = B_expanded.stride()
        C_exp_stride_b, C_exp_stride_c, C_exp_stride_s, C_exp_stride_n, C_exp_stride_d = C_expanded.stride()
        G_stride_b, G_stride_c, G_stride_i, G_stride_j, G_stride_n = G.stride()
        grid_G = (Bsz, Csz, S, S, N)
        contract_BC_to_G[grid_G](
            B_expanded, C_expanded, G,
            Bsz, Csz, S, N, D,
            K=D,
            B_exp_stride_b=B_exp_stride_b, B_exp_stride_c=B_exp_stride_c, B_exp_stride_s=B_exp_stride_s, B_exp_stride_n=B_exp_stride_n, B_exp_stride_d=B_exp_stride_d,
            C_exp_stride_b=C_exp_stride_b, C_exp_stride_c=C_exp_stride_c, C_exp_stride_s=C_exp_stride_s, C_exp_stride_n=C_exp_stride_n, C_exp_stride_d=C_exp_stride_d,
            G_stride_b=G_stride_b, G_stride_c=G_stride_c, G_stride_i=G_stride_i, G_stride_j=G_stride_j, G_stride_n=G_stride_n,
            num_warps=1, num_stages=1
        )

        # Elementwise M = G * L (PyTorch)
        M = G * L  # float32

        # Launch diag_contract_Y
        M_stride_b, M_stride_c, M_stride_i, M_stride_j, M_stride_n = M.stride()
        hidden_f32 = hidden_states.to(torch.float32)
        hidden_stride_b, hidden_stride_c, hidden_stride_j, hidden_stride_n, hidden_stride_d = hidden_f32.stride()
        Y_stride_b, Y_stride_c, Y_stride_i, Y_stride_n, Y_stride_d = Y.stride()
        grid_Y = (Bsz, Csz, S, N, D)
        diag_contract_Y[grid_Y](
            M, hidden_f32, Y,
            Bsz, Csz, S, N, D,
            M_stride_b, M_stride_c, M_stride_i, M_stride_j, M_stride_n,
            hidden_stride_b, hidden_stride_c, hidden_stride_j, hidden_stride_n, hidden_stride_d,
            Y_stride_b, Y_stride_c, Y_stride_i, Y_stride_n, Y_stride_d,
            num_warps=1, num_stages=1
        )

        # Cast to bfloat16 to match original signature
        return Y.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
