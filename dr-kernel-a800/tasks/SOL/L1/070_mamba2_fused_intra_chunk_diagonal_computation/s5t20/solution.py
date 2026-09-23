import torch
import triton
import triton.language as tl

# Constants (must match original implementation)
NUM_HEADS = 32
N_GROUPS = 8
GROUP_EXPAND = 4
CHUNK_SIZE = 128  # original code uses 128 for triangular mask size

# Kernel 1: Build L matrix with lower-triangular (diagonal=-1) over 128x128
# We implement this via simple 1D kernels over i and j (rows/cols), with grid over (b, c, h, i_group, j_group)
# Each program writes a single element L[b, c, i, j, h] = exp(total) if j <= i else 0
@triton.jit
def build_lower_tri_diag_minus1(
    A_ptr,              # *f32, [B, C, L, H]
    L_ptr,              # *f32, [B, C, 128, 128, H]
    B, C, H,            # int: dimensions
    A_stride_b, A_stride_c, A_stride_l, A_stride_h,  # strides of A
    L_stride_b, L_stride_c, L_stride_i, L_stride_j, L_stride_h,  # strides of L
):
    b = tl.program_id(0)
    c = tl.program_id(1)
    h = tl.program_id(2)
    i_group = tl.program_id(3)  # 0..127
    j_group = tl.program_id(4)  # 0..127

    i = i_group
    j = j_group

    # Compute total = sum_{k=0..L-1} A[b, c, k, h]
    total = 0.0
    L_dim = C  # hidden_states.shape[2] == C == L in original code
    for k in range(0, L_dim):
        a_ptr = A_ptr + b * A_stride_b + c * A_stride_c + k * A_stride_l + h * A_stride_h
        a_val = tl.load(a_ptr)  # scalar
        total += a_val

    # Build L[i, j] = exp(total) if j <= i, else 0
    is_lower = j <= i
    val = tl.where(is_lower, tl.exp(total), 0.0)

    # Store to L[b, c, i, j, h]
    l_ptr = L_ptr + b * L_stride_b + c * L_stride_c + i * L_stride_i + j * L_stride_j + h * L_stride_h
    tl.store(l_ptr, val)

# Kernel 2: Expand groups to heads for B and C
# Input: [B, C, L, G, S], Output: [B, C, L, H, S]
@triton.jit
def expand_groups_repeat_interleave(
    In_ptr,             # *f32, [B, C, L, G, S]
    Out_ptr,            # *f32, [B, C, L, H, S]
    B, C, L, G, S, H, GROUP_EXPAND,
    In_stride_b, In_stride_c, In_stride_l, In_stride_g, In_stride_s,
    Out_stride_b, Out_stride_c, Out_stride_l, Out_stride_h, Out_stride_s,
):
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)  # position along chunk dimension (L)
    h = tl.program_id(3)  # head index
    s = tl.program_id(4)  # state index

    # Determine which group g this head maps to
    # Original code maps: h = g * GROUP_EXPAND + k, so g = h // GROUP_EXPAND
    g = h // GROUP_EXPAND

    in_ptr = In_ptr + b * In_stride_b + c * In_stride_c + i * In_stride_l + g * In_stride_g + s * In_stride_s
    out_ptr = Out_ptr + b * Out_stride_b + c * Out_stride_c + i * Out_stride_l + h * Out_stride_h + s * Out_stride_s

    val = tl.load(in_ptr)
    tl.store(out_ptr, val)

# Kernel 3: Compute G[b, c, i, j, h] = sum_s C_exp[b, c, i, h, s] * B_exp[b, c, j, h, s]
# This kernel loops over s and accumulates into a per-(b,c,h) output tensor of shape [L, L, H]
@triton.jit
def compute_G_reduce_s(
    Bexp_ptr,           # *f32, [B, C, L, H, S]
    Cexp_ptr,           # *f32, [B, C, L, H, S]
    G_ptr,              # *f32, [B, C, L, L, H] (we write element-wise)
    B, C, L, H, S,
    Bexp_stride_b, Bexp_stride_c, Bexp_stride_l, Bexp_stride_h, Bexp_stride_s,
    Cexp_stride_b, Cexp_stride_c, Cexp_stride_l, Cexp_stride_h, Cexp_stride_s,
    G_stride_b, G_stride_c, G_stride_i, G_stride_j, G_stride_h,
):
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)  # row in chunk
    j = tl.program_id(3)  # col in chunk
    h = tl.program_id(4)  # head index

    acc = 0.0
    for s in range(0, S):
        bexp_ptr = Bexp_ptr + b * Bexp_stride_b + c * Bexp_stride_c + j * Bexp_stride_l + h * Bexp_stride_h + s * Bexp_stride_s
        cexp_ptr = Cexp_ptr + b * Cexp_stride_b + c * Cexp_stride_c + i * Cexp_stride_l + h * Cexp_stride_h + s * Cexp_stride_s

        b_val = tl.load(bexp_ptr)
        c_val = tl.load(cexp_ptr)
        acc += b_val * c_val

    # Store G[b, c, i, j, h] = acc
    g_ptr = G_ptr + b * G_stride_b + c * G_stride_c + i * G_stride_i + j * G_stride_j + h * G_stride_h
    tl.store(g_ptr, acc)

# Kernel 4: Apply mask M = G * L element-wise
# We implement a 1D kernel that loops over (b, c, h) and tiles over (i, j)
@triton.jit
def apply_mask_elementwise(
    G_ptr,              # *f32, [B, C, L, L, H]
    L_ptr,              # *f32, [B, C, 128, 128, H]
    M_ptr,              # *f32, [B, C, L, L, H]
    B, C, L, H,
    G_stride_b, G_stride_c, G_stride_i, G_stride_j, G_stride_h,
    L_stride_b, L_stride_c, L_stride_i, L_stride_j, L_stride_h,
    M_stride_b, M_stride_c, M_stride_i, M_stride_j, M_stride_h,
):
    # Grid: (BC, i_group, j_group, H)
    bc_id = tl.program_id(0)
    i_group = tl.program_id(1)
    j_group = tl.program_id(2)
    h = tl.program_id(3)

    b = bc_id // C
    c = bc_id % C

    i = i_group
    j = j_group

    g_ptr = G_ptr + b * G_stride_b + c * G_stride_c + i * G_stride_i + j * G_stride_j + h * G_stride_h
    l_ptr = L_ptr + b * L_stride_b + c * L_stride_c + i * L_stride_i + j * L_stride_j + h * L_stride_h
    m_ptr = M_ptr + b * M_stride_b + c * M_stride_c + i * M_stride_i + j * M_stride_j + h * M_stride_h

    g_val = tl.load(g_ptr)
    l_val = tl.load(l_ptr)
    tl.store(m_ptr, g_val * l_val)

# Kernel 5: Compute Y_diag[b, c, i, h, d] = sum_j M[b, c, i, j, h] * hidden_states[b, c, j, h, d]
# We implement a 1D kernel over d and loop over j.
@triton.jit
def contract_j_and_hidden(
    M_ptr,              # *f32, [B, C, L, L, H]
    hidden_ptr,         # *f32, [B, C, L, H, D]
    out_ptr,            # *f32, [B, C, L, H, D]
    B, C, L, H, D,
    M_stride_b, M_stride_c, M_stride_i, M_stride_j, M_stride_h,
    hidden_stride_b, hidden_stride_c, hidden_stride_l, hidden_stride_h, hidden_stride_d,
    out_stride_b, out_stride_c, out_stride_l, out_stride_h, out_stride_d,
):
    # Grid over (b, c, i, h, d)
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)
    h = tl.program_id(3)
    d = tl.program_id(4)

    acc = 0.0
    for j in range(0, L):
        m_ptr = M_ptr + b * M_stride_b + c * M_stride_c + i * M_stride_i + j * M_stride_j + h * M_stride_h
        h_ptr = hidden_ptr + b * hidden_stride_b + c * hidden_stride_c + j * hidden_stride_l + h * hidden_stride_h + d * hidden_stride_d
        m_val = tl.load(m_ptr)
        h_val = tl.load(h_ptr)
        acc += m_val * h_val

    out_ptr = out_ptr + b * out_stride_b + c * out_stride_c + i * out_stride_l + h * out_stride_h + d * out_stride_d
    tl.store(out_ptr, acc)

def _launch_build_lower_tri_diag_minus1(A_f32, L_ptr):
    B, C, L, H = A_f32.shape  # A_cumsum: [B, C, L, H]
    grid = (B, C, H, CHUNK_SIZE, CHUNK_SIZE)
    triton.run(
        build_lower_tri_diag_minus1,
        grid=grid,
        num_warps=1,
        num_stages=1,
        A_ptr=A_f32,
        L_ptr=L_ptr,
        B=B, C=C, H=H,
        A_stride_b=A_f32.stride(0), A_stride_c=A_f32.stride(1), A_stride_l=A_f32.stride(2), A_stride_h=A_f32.stride(3),
        L_stride_b=L_ptr.stride(0), L_stride_c=L_ptr.stride(1), L_stride_i=L_ptr.stride(2), L_stride_j=L_ptr.stride(3), L_stride_h=L_ptr.stride(4),
    )

def _launch_expand_groups_repeat_interleave(B_f32, C_f32, B_exp, C_exp):
    B, C, L, G, S = B_f32.shape
    H = NUM_HEADS
    grid = (B, C, L, H, S)
    triton.run(
        expand_groups_repeat_interleave,
        grid=grid,
        num_warps=1,
        num_stages=1,
        In_ptr=B_f32,
        Out_ptr=B_exp,
        B=B, C=C, L=L, G=G, S=S, H=H, GROUP_EXPAND=GROUP_EXPAND,
        In_stride_b=B_f32.stride(0), In_stride_c=B_f32.stride(1), In_stride_l=B_f32.stride(2), In_stride_g=B_f32.stride(3), In_stride_s=B_f32.stride(4),
        Out_stride_b=B_exp.stride(0), Out_stride_c=B_exp.stride(1), Out_stride_l=B_exp.stride(2), Out_stride_h=B_exp.stride(3), Out_stride_s=B_exp.stride(4),
    )
    triton.run(
        expand_groups_repeat_interleave,
        grid=grid,
        num_warps=1,
        num_stages=1,
        In_ptr=C_f32,
        Out_ptr=C_exp,
        B=B, C=C, L=L, G=G, S=S, H=H, GROUP_EXPAND=GROUP_EXPAND,
        In_stride_b=C_f32.stride(0), In_stride_c=C_f32.stride(1), In_stride_l=C_f32.stride(2), In_stride_g=C_f32.stride(3), In_stride_s=C_f32.stride(4),
        Out_stride_b=C_exp.stride(0), Out_stride_c=C_exp.stride(1), Out_stride_l=C_exp.stride(2), Out_stride_h=C_exp.stride(3), Out_stride_s=C_exp.stride(4),
    )

def _launch_compute_G_reduce_s(B_exp, C_exp, G):
    B, C, L, H, S = B_exp.shape
    grid = (B, C, L, L, H)
    triton.run(
        compute_G_reduce_s,
        grid=grid,
        num_warps=1,
        num_stages=1,
        Bexp_ptr=B_exp,
        Cexp_ptr=C_exp,
        G_ptr=G,
        B=B, C=C, L=L, H=H, S=S,
        Bexp_stride_b=B_exp.stride(0), Bexp_stride_c=B_exp.stride(1), Bexp_stride_l=B_exp.stride(2), Bexp_stride_h=B_exp.stride(3), Bexp_stride_s=B_exp.stride(4),
        Cexp_stride_b=C_exp.stride(0), Cexp_stride_c=C_exp.stride(1), Cexp_stride_l=C_exp.stride(2), Cexp_stride_h=C_exp.stride(3), Cexp_stride_s=C_exp.stride(4),
        G_stride_b=G.stride(0), G_stride_c=G.stride(1), G_stride_i=G.stride(2), G_stride_j=G.stride(3), G_stride_h=G.stride(4),
    )

def _launch_apply_mask_elementwise(G, L, M):
    B, C, L, Lj, H = G.shape
    grid = (B * C, L, L, H)
    triton.run(
        apply_mask_elementwise,
        grid=grid,
        num_warps=1,
        num_stages=1,
        G_ptr=G, L_ptr=L, M_ptr=M,
        B=B, C=C, L=L, H=H,
        G_stride_b=G.stride(0), G_stride_c=G.stride(1), G_stride_i=G.stride(2), G_stride_j=G.stride(3), G_stride_h=G.stride(4),
        L_stride_b=L.stride(0), L_stride_c=L.stride(1), L_stride_i=L.stride(2), L_stride_j=L.stride(3), L_stride_h=L.stride(4),
        M_stride_b=M.stride(0), M_stride_c=M.stride(1), M_stride_i=M.stride(2), M_stride_j=M.stride(3), M_stride_h=M.stride(4),
    )

def _launch_contract_j_and_hidden(M, hidden_f32, out):
    B, C, L, H, D = hidden_f32.shape
    grid = (B, C, L, H, D)
    triton.run(
        contract_j_and_hidden,
        grid=grid,
        num_warps=1,
        num_stages=1,
        M_ptr=M, hidden_ptr=hidden_f32, out_ptr=out,
        B=B, C=C, L=L, H=H, D=D,
        M_stride_b=M.stride(0), M_stride_c=M.stride(1), M_stride_i=M.stride(2), M_stride_j=M.stride(3), M_stride_h=M.stride(4),
        hidden_stride_b=hidden_f32.stride(0), hidden_stride_c=hidden_f32.stride(1), hidden_stride_l=hidden_f32.stride(2), hidden_stride_h=hidden_f32.stride(3), hidden_stride_d=hidden_f32.stride(4),
        out_stride_b=out.stride(0), out_stride_c=out.stride(1), out_stride_l=out.stride(2), out_stride_h=out.stride(3), out_stride_d=out.stride(4),
    )

class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, A_cumsum: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        # Ensure device and dtype
        device = hidden_states.device
        # The original code uses float32 for computations; we'll keep float32 to match behavior
        hidden_f32 = hidden_states.to(torch.float32)
        A_f32 = A_cumsum.to(torch.float32)
        B_f32 = B.to(torch.float32)
        C_f32 = C.to(torch.float32)

        # Shapes
        batch_size, num_chunks, chunk_size, num_heads, head_dim = hidden_f32.shape
        # Enforce constants
        H = NUM_HEADS
        G = N_GROUPS
        GROUP_EXPAND = 4
        L = chunk_size  # same as original

        # 1) Build L [B, C, 128, 128, H] with diagonal=-1
        L_tensor = torch.empty((batch_size, num_chunks, CHUNK_SIZE, CHUNK_SIZE, num_heads), dtype=torch.float32, device=device)
        _launch_build_lower_tri_diag_minus1(A_f32, L_tensor)

        # 2) Expand B and C to heads
        S = B_f32.shape[-1]
        B_exp = torch.empty((batch_size, num_chunks, chunk_size, num_heads, S), dtype=torch.float32, device=device)
        C_exp = torch.empty((batch_size, num_chunks, chunk_size, num_heads, S), dtype=torch.float32, device=device)
        _launch_expand_groups_repeat_interleave(B_f32, C_f32, B_exp, C_exp)

        # 3) Compute G = sum_s B_exp * C_exp
        G_tensor = torch.empty((batch_size, num_chunks, chunk_size, chunk_size, num_heads), dtype=torch.float32, device=device)
        _launch_compute_G_reduce_s(B_exp, C_exp, G_tensor)

        # 4) Apply mask L: M = G * L
        M_tensor = torch.empty_like(G_tensor)
        _launch_apply_mask_elementwise(G_tensor, L_tensor, M_tensor)

        # 5) Compute Y_diag = sum_j M * hidden_states
        Y = torch.empty((batch_size, num_chunks, chunk_size, num_heads, head_dim), dtype=torch.float32, device=device)
        _launch_contract_j_and_hidden(M_tensor, hidden_f32, Y)

        # Match original return dtype (original returns bfloat16)
        return Y.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
