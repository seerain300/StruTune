import torch
import triton
import triton.language as tl


# Kernel: Compute G[b, c, i, j, h] = sum_s C_exp[b, c, i, h, s] * B_exp[b, c, j, h, s]
# Grid: (B, C, 128, 128, 32)
@triton.jit
def compute_G_kernel(B_exp_ptr, C_exp_ptr, G_ptr,
                     B_exp_stride_b, B_exp_stride_c, B_exp_stride_l, B_exp_stride_h, B_exp_stride_s,
                     C_exp_stride_b, C_exp_stride_c, C_exp_stride_l, C_exp_stride_h, C_exp_stride_s,
                     G_stride_b, G_stride_c, G_stride_i, G_stride_j, G_stride_h,
                     S: tl.int32):
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)
    j = tl.program_id(3)
    h = tl.program_id(4)

    acc = 0.0
    for s in range(0, S):
        bval = tl.load(B_exp_ptr + b * B_exp_stride_b + c * B_exp_stride_c + j * B_exp_stride_l + h * B_exp_stride_h + s * B_exp_stride_s)
        cval = tl.load(C_exp_ptr + b * C_exp_stride_b + c * C_exp_stride_c + i * C_exp_stride_l + h * C_exp_stride_h + s * C_exp_stride_s)
        acc += bval * cval
    tl.store(G_ptr + b * G_stride_b + c * G_stride_c + i * G_stride_i + j * G_stride_j + h * G_stride_h, acc)


# Kernel: Element-wise multiply M = G * L
# L is [B, C, 128, 128, H], G and M are [B, C, 128, 128, H]
@triton.jit
def elementwise_M_kernel(G_ptr, L_ptr, M_ptr,
                          G_stride_b, G_stride_c, G_stride_i, G_stride_j, G_stride_h,
                          L_stride_b, L_stride_c, L_stride_i, L_stride_j, L_stride_h,
                          M_stride_b, M_stride_c, M_stride_i, M_stride_j, M_stride_h):
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)
    j = tl.program_id(3)
    h = tl.program_id(4)

    gval = tl.load(G_ptr + b * G_stride_b + c * G_stride_c + i * G_stride_i + j * G_stride_j + h * G_stride_h)
    lval = tl.load(L_ptr + b * L_stride_b + c * L_stride_c + i * L_stride_i + j * L_stride_j + h * L_stride_h)
    mval = gval * lval
    tl.store(M_ptr + b * M_stride_b + c * M_stride_c + i * M_stride_i + j * M_stride_j + h * M_stride_h, mval)


# Kernel: Compute Y[b, c, i, h, s] = sum_j M[b, c, i, j, h] * hidden[b, c, j, h, s]
# Output shape: [B, C, 128, 32, S]
@triton.jit
def compute_Y_kernel(M_ptr, hidden_ptr, Y_ptr,
                     M_stride_b, M_stride_c, M_stride_i, M_stride_j, M_stride_h,
                     hidden_stride_b, hidden_stride_c, hidden_stride_l, hidden_stride_h, hidden_stride_s,
                     Y_stride_b, Y_stride_c, Y_stride_i, Y_stride_h, Y_stride_s,
                     L_len: tl.int32, S: tl.int32):
    b = tl.program_id(0)
    c = tl.program_id(1)
    i = tl.program_id(2)
    h = tl.program_id(3)
    s = tl.program_id(4)

    acc = 0.0
    for j in range(0, L_len):  # L_len is hidden's L dimension, e.g., 128
        mval = tl.load(M_ptr + b * M_stride_b + c * M_stride_c + i * M_stride_i + j * M_stride_j + h * M_stride_h)
        hval = tl.load(hidden_ptr + b * hidden_stride_b + c * hidden_stride_c + j * hidden_stride_l + h * hidden_stride_h + s * hidden_stride_s)
        acc += mval * hval
    tl.store(Y_ptr + b * Y_stride_b + c * Y_stride_c + i * Y_stride_i + h * Y_stride_h + s * Y_stride_s, acc)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, A_cumsum: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        """
        Triton-optimized forward implementing the original logic.
        Computes output with Triton kernels for G, M, and Y.
        Returns tensor of shape [B, C, 128, 32, S], matching original Model's output.
        """
        # Shapes
        Bsz, Csz, L, H, D = hidden_states.shape  # hidden_states: [B, C, L, H, D]
        S = B.shape[-1] if len(B.shape) == 5 else B.shape[-1]  # original S is state_size; here B has shape [B, C, L, G, S], but we need S for Y output
        # We can infer S from B's last dim (B is [B, C, L, G, S]). We'll use B's S.
        # However, the original signature passes B and C with S known from their last dim.
        # So we directly take S from B's last dim.
        S = B.shape[-1]
        num_chunks = Csz  # as in original
        num_heads = 32
        chunk_size = 128  # CHUNK_SIZE in original

        # Ensure device consistency and float32 compute
        device = hidden_states.device
        hidden_f32 = hidden_states.to(torch.float32)
        A_f32 = A_cumsum.to(torch.float32)
        B_f32 = B.to(torch.float32)
        C_f32 = C.to(torch.float32)

        # 1) Build expanded B and C to heads (GROUP_EXPAND=4)
        # We repeat_interleave groups to heads via slicing or repeat, but using Triton:
        # B_exp: [B, C, L, H, S], where H=32
        B_exp = torch.empty((Bsz, Csz, L, num_heads, S), dtype=torch.float32, device=device)
        C_exp = torch.empty((Bsz, Csz, L, num_heads, S), dtype=torch.float32, device=device)

        # Fill B_exp and C_exp by mapping groups -> heads
        # N_GROUPS=8, so each group maps to 4 heads.
        # For each (b, c, l, s), copy to 4 head positions: h in [0..31], step 4.
        # We'll do this via simple torch indexing for correctness, then keep Triton for heavy ops.
        # For each s:
        for s_idx in range(S):
            b_c_l = (Bsz, Csz, L)
            for g in range(8):
                bval = B_f32[:, :, :, g, s_idx]  # [B, C, L]
                cval = C_f32[:, :, :, g, s_idx]  # [B, C, L]
                # Place into B_exp/C_exp at head positions h = g*4 + 0,1,2,3
                for k in range(4):
                    h_idx = g * 4 + k
                    B_exp[:, :, :, h_idx, s_idx] = bval
                    C_exp[:, :, :, h_idx, s_idx] = cval

        # 2) Compute G in Triton: G[b, c, i, j, h] = sum_s C_exp[b, c, i, h, s] * B_exp[b, c, j, h, s]
        G = torch.empty((Bsz, Csz, chunk_size, chunk_size, num_heads), dtype=torch.float32, device=device)
        grid_G = (Bsz, Csz, chunk_size, chunk_size, num_heads)
        compute_G_kernel[grid_G](
            B_exp, C_exp, G,
            B_exp.stride(0), B_exp.stride(1), B_exp.stride(2), B_exp.stride(3), B_exp.stride(4),
            C_exp.stride(0), C_exp.stride(1), C_exp.stride(2), C_exp.stride(3), C_exp.stride(4),
            G.stride(0), G.stride(1), G.stride(2), G.stride(3), G.stride(4),
            S
        )

        # 3) Build L (causal mask) with torch.tril: L[b, c, i, j, h] = exp(cumsum(A_cumsum[b, c, :, h])) if j<=i else 0
        # Treat A_cumsum as [B, C, L, H], then build 128x128 lower-triangular mask.
        A_mat = A_f32[:, :, :, :, None]  # [B, C, L, H, 1]
        L_mat = torch.empty((Bsz, Csz, chunk_size, chunk_size, num_heads), dtype=torch.float32, device=device)
        # For each (b, c, h), cumsum along L dimension
        for b in range(Bsz):
            for c in range(Csz):
                for h in range(num_heads):
                    # cumsum over L
                    cumsum = torch.cumsum(A_mat[b, c, :, h], dim=0)  # [L]
                    # pad to 128: cumsum[k] if k<L else 0
                    cumsum_128 = torch.zeros((chunk_size,), dtype=torch.float32, device=device)
                    cumsum_128[:L] = cumsum
                    L_lower = torch.tril(torch.ones((chunk_size, chunk_size), dtype=torch.float32, device=device), diagonal=-1)
                    L_mat[b, c, :, :, h] = L_lower * torch.exp(cumsum_128[:, None])

        # 4) Compute M = G * L element-wise in Triton
        M = torch.empty((Bsz, Csz, chunk_size, chunk_size, num_heads), dtype=torch.float32, device=device)
        grid_M = (Bsz, Csz, chunk_size, chunk_size, num_heads)
        elementwise_M_kernel[grid_M](
            G, L_mat, M,
            G.stride(0), G.stride(1), G.stride(2), G.stride(3), G.stride(4),
            L_mat.stride(0), L_mat.stride(1), L_mat.stride(2), L_mat.stride(3), L_mat.stride(4),
            M.stride(0), M.stride(1), M.stride(2), M.stride(3), M.stride(4)
        )

        # 5) Compute Y_diag[b, c, i, h, s] = sum_j M[b, c, i, j, h] * hidden[b, c, j, h, s] in Triton
        Y = torch.empty((Bsz, Csz, chunk_size, num_heads, S), dtype=torch.float32, device=device)
        grid_Y = (Bsz, Csz, chunk_size, num_heads, S)
        compute_Y_kernel[grid_Y](
            M, hidden_f32, Y,
            M.stride(0), M.stride(1), M.stride(2), M.stride(3), M.stride(4),
            hidden_f32.stride(0), hidden_f32.stride(1), hidden_f32.stride(2), hidden_f32.stride(3), hidden_f32.stride(4),
            Y.stride(0), Y.stride(1), Y.stride(2), Y.stride(3), Y.stride(4),
            L, S
        )

        # Return in bfloat16 as in original signature
        return Y.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
