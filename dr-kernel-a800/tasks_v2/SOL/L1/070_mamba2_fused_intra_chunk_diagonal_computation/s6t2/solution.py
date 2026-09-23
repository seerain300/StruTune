import torch
import triton
import triton.language as tl

# Kernel 1: tril_mask_kernel
# Produces a lower-triangular boolean mask of shape [S, S] with diagonal = -1
# Output: out_mask of shape [S, S] (int1). The forward will convert to torch.bool on host.
@triton.jit
def tril_mask_kernel(out_ptr, S_size: tl.constexpr):
    i = tl.program_id(0)
    j = tl.program_id(1)
    # Write 1 where j <= i else 0
    val = (j <= i).to(tl.int1)
    out_ptr[i * S_size + j] = val

# Kernel 2: cumsum_axis3_kernel
# Input: X_ptr of shape [B, H, N, S, S] float32
# Output: Out_ptr of shape [B, H, N, S, S] float32 where Out[b,h,n,i,j] = cumsum(X[b,h,n, i, j]) along axis=3 (i)
# Here, axis=3 is the source dimension S (i). We loop over i to compute cumsum for each j.
@triton.jit
def cumsum_axis3_kernel(X_ptr, Out_ptr,
                         B_size, H_size, N_size, S_size,
                         X_stride_b, X_stride_h, X_stride_n, X_stride_s, X_stride_s2,
                         Out_stride_b, Out_stride_h, Out_stride_n, Out_stride_s, Out_stride_s2):
    b = tl.program_id(0)
    h = tl.program_id(1)
    n = tl.program_id(2)
    j = tl.program_id(3)  # fixed target index for cumsum along i

    # Compute cumsum along i (axis=3)
    acc = tl.zeros([1], dtype=tl.float32)
    for i in range(0, S_size):
        x_ptr = X_ptr + b * X_stride_b + h * X_stride_h + n * X_stride_n + i * X_stride_s + j * X_stride_s2
        x_val = tl.load(x_ptr).to(tl.float32)
        acc += x_val
        out_ptr = Out_ptr + b * Out_stride_b + h * Out_stride_h + n * Out_stride_n + i * Out_stride_s + j * Out_stride_s2
        tl.store(out_ptr, acc)

# Kernel 3: g_contract_kernel
# Compute G[b,n,i,j,h] = sum_d C[b,n,i,h,d] * B[b,n,j,h,d]
# Inputs:
#   C_ptr: [B, N, S, H, D] float32
#   B_ptr: [B, N, S, H, D] float32
# Output:
#   G_ptr: [B, N, S, S, H] float32
@triton.jit
def g_contract_kernel(C_ptr, B_ptr, G_ptr,
                      B_size, N_size, S_size, H_size, D_size,
                      C_stride_b, C_stride_n, C_stride_s, C_stride_h, C_stride_d,
                      B_stride_b, B_stride_n, B_stride_s, B_stride_h, B_stride_d,
                      G_stride_b, G_stride_n, G_stride_s1, G_stride_s2, G_stride_h,
                      BLOCK_D: tl.constexpr):
    b = tl.program_id(0)
    n = tl.program_id(1)
    i = tl.program_id(2)
    j = tl.program_id(3)
    h = tl.program_id(4)

    # Accumulate dot product over D in chunks of BLOCK_D
    acc = tl.zeros([1], dtype=tl.float32)
    for d0 in range(0, D_size, BLOCK_D):
        d_offsets = d0 + tl.arange(0, BLOCK_D)
        d_mask = d_offsets < D_size
        c_ptrs = C_ptr + b * C_stride_b + n * C_stride_n + i * C_stride_s + h * C_stride_h + d_offsets * C_stride_d
        b_ptrs = B_ptr + b * B_stride_b + n * B_stride_n + j * B_stride_s + h * B_stride_h + d_offsets * B_stride_d
        c_vals = tl.load(c_ptrs, mask=d_mask, other=0.0).to(tl.float32)
        b_vals = tl.load(b_ptrs, mask=d_mask, other=0.0).to(tl.float32)
        prod = c_vals * b_vals
        acc += tl.sum(prod, axis=0)
    g_ptr = G_ptr + b * G_stride_b + n * G_stride_n + i * G_stride_s1 + j * G_stride_s2 + h * G_stride_h
    tl.store(g_ptr, acc)

# Kernel 4: y_diag_reduce_kernel
# Compute Y[b,n,i,h,:] = sum_j G[b,n,i,j,h] * hidden_states[b,n,j,h,:]
# Inputs:
#   G_ptr: [B, N, S, S, H] float32
#   HS_ptr: [B, N, S, H, D] float32 (hidden_states)
# Output:
#   Y_ptr: [B, N, S, H, D] float32
@triton.jit
def y_diag_reduce_kernel(G_ptr, HS_ptr, Y_ptr,
                         B_size, N_size, S_size, H_size, D_size,
                         G_stride_b, G_stride_n, G_stride_s1, G_stride_s2, G_stride_h,
                         HS_stride_b, HS_stride_n, HS_stride_s, HS_stride_h, HS_stride_d,
                         Y_stride_b, Y_stride_n, Y_stride_s, Y_stride_h, Y_stride_d,
                         BLOCK_D: tl.constexpr):
    b = tl.program_id(0)
    n = tl.program_id(1)
    i = tl.program_id(2)
    h = tl.program_id(3)

    acc_vec = tl.zeros([D_size], dtype=tl.float32)
    for j in range(0, S_size):
        g_ptr = G_ptr + b * G_stride_b + n * G_stride_n + i * G_stride_s1 + j * G_stride_s2 + h * G_stride_h
        g_val = tl.load(g_ptr).to(tl.float32)
        hs_ptrs = HS_ptr + b * HS_stride_b + n * HS_stride_n + j * HS_stride_s + h * HS_stride_h + tl.arange(0, D_size) * HS_stride_d
        hs_mask = tl.arange(0, D_size) < D_size
        hs_vals = tl.load(hs_ptrs, mask=hs_mask, other=0.0).to(tl.float32)
        acc_vec += g_val * hs_vals

    y_ptrs = Y_ptr + b * Y_stride_b + n * Y_stride_n + i * Y_stride_s + h * Y_stride_h + tl.arange(0, D_size) * Y_stride_d
    y_mask = tl.arange(0, D_size) < D_size
    tl.store(y_ptrs, acc_vec, mask=y_mask)

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Constants from original code
        self.NUM_HEADS = 32
        self.N_GROUPS = 8
        self.CHUNK_SIZE = 128  # used for mask shape
        # We'll rely on inputs to provide shapes; ensure D_size equals head_dim of hidden_states

    def forward(self, hidden_states: torch.Tensor, A_cumsum: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        # Ensure CUDA tensors and contiguous
        if not hidden_states.is_cuda:
            hidden_states = hidden_states.cuda()
        if not A_cumsum.is_cuda:
            A_cumsum = A_cumsum.cuda()
        if not B.is_cuda:
            B = B.cuda()
        if not C.is_cuda:
            C = C.cuda()

        hidden_states = hidden_states.contiguous()
        A_cumsum = A_cumsum.contiguous()
        B = B.contiguous()
        C = C.contiguous()

        # Shapes
        B_size, N_size, S_size, H_size, D_size = hidden_states.shape
        assert A_cumsum.shape == (B_size, H_size, N_size, S_size), "A_cumsum must have shape [B, H, N, S]"
        # Ensure B/C expanded to H: original repeats by factor NUM_HEADS//N_GROUPS = 4
        expand_factor = self.NUM_HEADS // self.N_GROUPS
        if B.shape[3] != H_size:
            B = B.repeat_interleave(expand_factor, dim=3)
        if C.shape[3] != H_size:
            C = C.repeat_interleave(expand_factor, dim=3)

        device = hidden_states.device

        # 1) Compute lower-triangular mask M_lower of shape [S, S] with diagonal=-1 using Triton
        M_lower = torch.empty((S_size, S_size), dtype=torch.int1, device=device)
        grid_mask = (S_size, S_size)
        tril_mask_kernel[grid_mask](M_lower, S_size)

        # 2) Expand A_cumsum to [B, H, N, S, S] and perform cumsum along axis=3 (S) using Triton
        A_expanded = A_cumsum.unsqueeze(-1).expand(B_size, H_size, N_size, S_size, S_size).to(torch.float32)
        A_cumsum_out = torch.empty_like(A_expanded, dtype=torch.float32, device=device)

        X_stride_b, X_stride_h, X_stride_n, X_stride_s, X_stride_s2 = A_expanded.stride()
        Out_stride_b, Out_stride_h, Out_stride_n, Out_stride_s, Out_stride_s2 = A_cumsum_out.stride()

        grid_cumsum = (B_size, H_size, N_size, S_size)
        cumsum_axis3_kernel[grid_cumsum](
            A_expanded, A_cumsum_out,
            B_size, H_size, N_size, S_size,
            X_stride_b, X_stride_h, X_stride_n, X_stride_s, X_stride_s2,
            Out_stride_b, Out_stride_h, Out_stride_n, Out_stride_s, Out_stride_s2,
            num_warps=1, num_stages=1
        )

        # 3) Compute L = exp(A_cumsum_out) where upper triangle is excluded by M_lower
        # Build full L: initialize zeros, then fill lower triangle positions
        L = torch.zeros((B_size, H_size, N_size, S_size, S_size), dtype=torch.float32, device=device)
        # Apply M_lower: for positions (i,j) where j <= i, L[i,j] = exp(A_cumsum_out[b,h,n,i,j]); else 0
        # We can directly fill using mask: M_lower is 1 where j<=i, 0 otherwise.
        # For efficiency, we just use A_cumsum_out for lower-triangular positions; upper triangle remains zero.
        # The original code applies exp to the cumsum result; since upper triangle is zero, exp(0)=1, but original uses masked cumsum,
        # so we must ensure upper triangle is not exp of anything; we set upper triangle to 0 explicitly after exp.
        L = A_cumsum_out.clone()
        # Apply mask: set upper triangle to 0
        # We can't index with mask in PyTorch efficiently here; we simply zero-out upper triangle using broadcasting:
        for b in range(B_size):
            for h in range(H_size):
                for n in range(N_size):
                    # L[b,h,n] has shape [S_size, S_size]
                    # Zero out upper triangle: j > i
                    # We can use torch.where with M_lower
                    # Create indices
                    i_idx = torch.arange(S_size, device=device).view(S_size, 1)
                    j_idx = torch.arange(S_size, device=device).view(1, S_size)
                    # Broadcast M_lower to [S,S] boolean; but we already have M_lower tensor; we'll construct mask from it
                    # Simpler: we can't directly use tensor mask here; instead, we zero upper triangle by checking j > i
                    # Since we cannot write back in Triton here, we do it in PyTorch:
                    pass
        # Now zero upper triangle explicitly
        for b in range(B_size):
            for h in range(H_size):
                for n in range(N_size):
                    for i in range(S_size):
                        for j in range(S_size):
                            if j > i:
                                L[b, h, n, i, j] = 0.0

        # Compute exp: L = exp(L); upper triangle remains 0 -> exp(0)=1? No: we want L[i,j] = exp(cumsum for i>=j). Above we used A_cumsum_out for lower, zeros for upper. So exp of zeros -> 1, but original logic excludes upper entirely? Wait, original code applies exp to the cumsum after masked fill, which keeps lower triangle and zeros upper. Therefore, exp of zeros is 1, but original desired L upper to be 0, not 1. We must correct this.
        # Conclusion: We should not leave upper triangle as zeros before exp. The original L has upper triangle as 0 after exp of masked cumsum, but in our code, we need to explicitly set upper triangle to 0 after exp of lower triangle.
        # To fix, recompute L correctly:
        # L = exp(A_cumsum_out) but we need to set upper triangle to 0. Since A_cumsum_out for upper was masked to 0 in cumsum step, exp(0)=1, but original intended upper to be 0. Therefore, after exp, zero out upper triangle.
        L = torch.exp(A_cumsum_out)
        # Zero upper triangle
        for b in range(B_size):
            for h in range(H_size):
                for n in range(N_size):
                    for i in range(S_size):
                        for j in range(S_size):
                            if j > i:
                                L[b, h, n, i, j] = 0.0

        # 4) Compute G = sum over state (D) of C * B using Triton
        G = torch.empty((B_size, N_size, S_size, S_size, H_size), dtype=torch.float32, device=device)

        C_stride_b, C_stride_n, C_stride_s, C_stride_h, C_stride_d = C.stride()
        B_stride_b, B_stride_n, B_stride_s, B_stride_h, B_stride_d = B.stride()
        G_stride_b, G_stride_n, G_stride_s1, G_stride_s2, G_stride_h = G.stride()

        grid_G = (B_size, N_size, S_size, S_size, H_size)
        g_contract_kernel[grid_G](
            C, B, G,
            B_size, N_size, S_size, H_size, D_size,
            C_stride_b, C_stride_n, C_stride_s, C_stride_h, C_stride_d,
            B_stride_b, B_stride_n, B_stride_s, B_stride_h, B_stride_d,
            G_stride_b, G_stride_n, G_stride_s1, G_stride_s2, G_stride_h,
            BLOCK_D=64,  # D_size is typically 64
            num_warps=2, num_stages=2
        )

        # 5) Compute Y_diag via Triton reduction: Y[b,n,i,h,:] = sum_j G[b,n,i,j,h] * hidden_states[b,n,j,h,:]
        Y = torch.empty((B_size, N_size, S_size, H_size, D_size), dtype=torch.float32, device=device)

        M_stride_b, M_stride_n, M_stride_s1, M_stride_s2, M_stride_h = G.stride()
        HS_stride_b, HS_stride_n, HS_stride_s, HS_stride_h, HS_stride_d = hidden_states.stride()
        Y_stride_b, Y_stride_n, Y_stride_s, Y_stride_h, Y_stride_d = Y.stride()

        grid_Y = (B_size, N_size, S_size, H_size)
        y_diag_reduce_kernel[grid_Y](
            G, hidden_states, Y,
            B_size, N_size, S_size, H_size, D_size,
            M_stride_b, M_stride_n, M_stride_s1, M_stride_s2, M_stride_h,
            HS_stride_b, HS_stride_n, HS_stride_s, HS_stride_h, HS_stride_d,
            Y_stride_b, Y_stride_n, Y_stride_s, Y_stride_h, Y_stride_d,
            BLOCK_D=64,
            num_warps=2, num_stages=2
        )

        # Return in bfloat16 to match original
        return Y.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
