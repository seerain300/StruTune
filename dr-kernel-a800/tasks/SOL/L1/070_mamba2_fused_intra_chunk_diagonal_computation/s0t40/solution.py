import torch
import triton
import triton.language as tl

# Kernel 1: Build L with causal mask: L[b, c, i, j, h] = exp(sum_{t=0..j} A[b, h, c, t]) if i >= j, else 0
@triton.jit
def build_L_triton(
    A_ptr,            # *float32, shape [B, H, C, S]
    L_ptr,            # *float32, shape [B, C, S, S, H]
    Bsz: tl.int32,    # batch size
    Hsz: tl.int32,    # num heads (32)
    Csz: tl.int32,    # num chunks (C)
    Ssz: tl.int32,    # chunk size (128)
    stride_Ab: tl.int64,
    stride_Ah: tl.int64,
    stride_Ac: tl.int64,
    stride_As: tl.int64,
    stride_Lb: tl.int64,
    stride_Lc: tl.int64,
    stride_Li: tl.int64,
    stride_Lj: tl.int64,
    stride_Lh: tl.int64,
):
    # Each program handles one (b, h) pair and iterates over (c, i, j)
    b = tl.program_id(0) % Bsz
    h = tl.program_id(0) // Bsz

    for c in range(0, Csz):
        base_A = b * stride_Ab + h * stride_Ah + c * stride_Ac
        for i in range(0, Ssz):
            cum = 0.0
            for j in range(0, Ssz):
                a = tl.load(A_ptr + base_A + j * stride_As)
                cum += a
                val = tl.exp(cum)
                if i >= j:
                    base_L = b * stride_Lb + c * stride_Lc + i * stride_Li + j * stride_Lj + h * stride_Lh
                    tl.store(L_ptr + base_L, val)

# Kernel 2: Compute G[i, j, h] = sum_n C_exp[b, c, i, n, j, h] * B_exp[b, c, j, n, i, h]
@triton.jit
def compute_G_triton(
    B_ptr,            # *float32, shape [B, C, S, H, N]
    C_ptr,            # *float32, shape [B, C, S, H, N]
    G_ptr,            # *float32, shape [B, C, S, S, H]
    Bsz: tl.int32,
    Csz: tl.int32,
    Ssz: tl.int32,
    Hsz: tl.int32,
    Nsz: tl.int32,    # typically 128 (state_size)
    stride_Bb: tl.int64,
    stride_Bc: tl.int64,
    stride_Bs: tl.int64,
    stride_Bh: tl.int64,
    stride_Bn: tl.int64,
    stride_Cb: tl.int64,
    stride_Cc: tl.int64,
    stride_Cs: tl.int64,
    stride_Ch: tl.int64,
    stride_Cn: tl.int64,
    stride_Gb: tl.int64,
    stride_Gc: tl.int64,
    stride_Gi: tl.int64,
    stride_Gj: tl.int64,
    stride_Gh: tl.int64,
):
    # Each program handles one (b, c)
    bc = tl.program_id(0)
    b = bc // Csz
    c = bc % Csz

    for i in range(0, Ssz):
        for j in range(0, Ssz):
            acc = 0.0
            # n_groups = 8, repeat = 4 (NUM_HEADS // N_GROUPS = 4)
            for g in range(0, 8):
                for r in range(0, 4):
                    h_src = g * 4 + r  # expanded H index
                    # B_exp[b, c, j, h_src, n, i]
                    base_B = b * stride_Bb + c * stride_Bc + j * stride_Bs + h_src * stride_Bh
                    for n in range(0, Nsz):
                        b_elem = tl.load(B_ptr + base_B + n * stride_Bn + i * stride_Bn)  # + i*stride_Bn
                        # C_exp[b, c, i, h_src, n, j]
                        base_C = b * stride_Cb + c * stride_Cc + i * stride_Cs + h_src * stride_Ch
                        c_elem = tl.load(C_ptr + base_C + n * stride_Cn + j * stride_Cs)
                        acc += b_elem * c_elem
            # Store G[b, c, i, j, h_src]
            for h in range(0, Hsz):
                base_G = b * stride_Gb + c * stride_Gc + i * stride_Gi + j * stride_Gj + h * stride_Gh
                tl.store(G_ptr + base_G, acc)

# Kernel 3: Compute Y_diag[b, c, i, h, d] = sum_j M[b, c, i, j, h] * hidden_states[b, c, j, h, d]
@triton.jit
def compute_Y_diag_triton(
    M_ptr,            # *float32, shape [B, C, S, S, H]
    hidden_ptr,       # *float32, shape [B, C, S, H, D] (D = head_dim)
    Y_ptr,            # *float32, shape [B, C, S, H, D]
    Bsz: tl.int32,
    Csz: tl.int32,
    Ssz: tl.int32,
    Hsz: tl.int32,
    Dsz: tl.int32,    # head_dim (dynamic)
    stride_Mb: tl.int64,
    stride_Mc: tl.int64,
    stride_Mi: tl.int64,
    stride_Mj: tl.int64,
    stride_Mh: tl.int64,
    stride_hb: tl.int64,
    stride_hc: tl.int64,
    stride_hs: tl.int64,
    stride_hh: tl.int64,
    stride_hd: tl.int64,
    stride_Yb: tl.int64,
    stride_Yc: tl.int64,
    stride_Yi: tl.int64,
    stride_Yh: tl.int64,
    stride_Yd: tl.int64,
):
    # Each program handles one (b, c)
    bc = tl.program_id(0)
    b = bc // Csz
    c = bc % Csz

    for i in range(0, Ssz):
        for d in range(0, Dsz):
            acc = 0.0
            for j in range(0, Ssz):
                for h in range(0, Hsz):
                    base_M = b * stride_Mb + c * stride_Mc + i * stride_Mi + j * stride_Mj + h * stride_Mh
                    m = tl.load(M_ptr + base_M)
                    base_h = b * stride_hb + c * stride_hc + j * stride_hs + h * stride_hh + d * stride_hd
                    hs = tl.load(hidden_ptr + base_h)
                    acc += m * hs
            # Store Y[b, c, i, h, d]
            for h in range(0, Hsz):
                base_Y = b * stride_Yb + c * stride_Yc + i * stride_Yi + h * stride_Yh + d * stride_Yd
                tl.store(Y_ptr + base_Y, acc)

class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, A_cumsum: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        """
        Triton-optimized implementation of the original run function:
        Returns Y_diag with shape [B, C, S, H, head_dim], cast to bfloat16.
        """
        # Shapes
        Bsz, Csz, Ssz, Hsz, head_dim = hidden_states.shape
        # Ensure inputs are contiguous and float32 for kernels
        A = A_cumsum.contiguous().to(torch.float32)           # [B, H, C, S]
        B_ = B.contiguous().to(torch.float32)                 # [B, C, S, 8, N]
        C_ = C.contiguous().to(torch.float32)                 # [B, C, S, 8, N]

        # Allocate L, G, and Y in float32
        L = torch.empty((Bsz, Csz, Ssz, Ssz, Hsz), dtype=torch.float32, device=hidden_states.device)  # [B, C, S, S, H]
        G = torch.empty((Bsz, Csz, Ssz, Ssz, Hsz), dtype=torch.float32, device=hidden_states.device)  # [B, C, S, S, H]
        Y = torch.empty((Bsz, Csz, Ssz, Hsz, head_dim), dtype=torch.float32, device=hidden_states.device)

        # Launch build_L_triton
        grid_L = (Bsz * Hsz,)
        build_L_triton[grid_L](
            A, L,
            Bsz, Hsz, Csz, Ssz,
            A.stride(0), A.stride(1), A.stride(2), A.stride(3),
            L.stride(0), L.stride(1), L.stride(2), L.stride(3), L.stride(4),
        )

        # Launch compute_G_triton
        grid_G = (Bsz * Csz,)
        Nsz = B_.shape[4]  # 128
        compute_G_triton[grid_G](
            B_, C_, G,
            Bsz, Csz, Ssz, Hsz, Nsz,
            B_.stride(0), B_.stride(1), B_.stride(2), B_.stride(3), B_.stride(4),
            C_.stride(0), C_.stride(1), C_.stride(2), C_.stride(3), C_.stride(4),
            G.stride(0), G.stride(1), G.stride(2), G.stride(3), G.stride(4),
        )

        # Compute M = G * L
        M = G * L  # element-wise multiply

        # Prepare hidden states for Y_diag (contiguous, float32)
        hidden_f32 = hidden_states.contiguous().to(torch.float32)  # [B, C, S, H, head_dim]

        # Launch compute_Y_diag_triton
        grid_Y = (Bsz * Csz,)
        compute_Y_diag_triton[grid_Y](
            M, hidden_f32, Y,
            Bsz, Csz, Ssz, Hsz, head_dim,
            M.stride(0), M.stride(1), M.stride(2), M.stride(3), M.stride(4),
            hidden_f32.stride(0), hidden_f32.stride(1), hidden_f32.stride(2), hidden_f32.stride(3), hidden_f32.stride(4),
            Y.stride(0), Y.stride(1), Y.stride(2), Y.stride(3), Y.stride(4),
        )

        # Return in bfloat16 to match original output dtype
        return Y.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
