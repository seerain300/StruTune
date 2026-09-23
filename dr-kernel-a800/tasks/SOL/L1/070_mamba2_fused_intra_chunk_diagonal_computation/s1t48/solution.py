import torch
import triton
import triton.language as tl

# Constants matching the original code's hardcoded logic
CHUNK_SIZE = 128
NUM_HEADS = 32
N_GROUPS = 8
REPEAT = NUM_HEADS // N_GROUPS  # 4
STATE_SIZE = 64
HEAD_DIM = 64

# Triton kernel: compute L = exp(cumsum(masked A))
# A: [B, H, N, K, K], L: [B, H, N, K, K]
# Mask is lower-triangular with diagonal=-1: only j <= i contributes
@triton.jit
def masked_cumsum_tril_exp(A_ptr, L_ptr,
                            B_batch, B_heads, B_n,
                            A_stride_b, A_stride_h, A_stride_n, A_stride_i, A_stride_j,
                            L_stride_b, L_stride_n, L_stride_i, L_stride_j, L_stride_h,
                            CHUNK: tl.constexpr):
    b = tl.program_id(0)
    h = tl.program_id(1)
    n = tl.program_id(2)
    i = tl.program_id(3)
    cumsum = 0.0
    for j in range(CHUNK):
        if j <= i:
            a_off = b * A_stride_b + h * A_stride_h + n * A_stride_n + i * A_stride_i + j * A_stride_j
            val = tl.load(A_ptr + a_off)
            cumsum += val
        l_off = b * L_stride_b + h * L_stride_h + n * L_stride_n + i * L_stride_i + j * L_stride_j
        tl.store(L_ptr + l_off, tl.exp(cumsum))

# Triton kernel: contract B and C to produce G: [B, N, K, K, H]
# B: [B, N, K, n_groups, STATE_SIZE], C: [B, N, K, n_groups, STATE_SIZE]
# g = h // REPEAT (REPEAT=4), so groups=N_GROUPS=8
@triton.jit
def contract_BC_to_G(B_ptr, C_ptr, G_ptr,
                     B_batch, B_n, B_K, B_ng, B_STATE,
                     G_stride_b, G_stride_n, G_stride_i, G_stride_j, G_stride_h,
                     B_stride_b, B_stride_n, B_stride_k, B_stride_g, B_stride_s,
                     C_stride_b, C_stride_n, C_stride_k, C_stride_g, C_stride_s,
                     CHUNK: tl.constexpr, BLOCK_S: tl.constexpr, H: tl.constexpr):
    b = tl.program_id(0)
    n = tl.program_id(1)
    h = tl.program_id(2)
    for i in range(CHUNK):
        for j in range(CHUNK):
            acc = 0.0
            for s_start in range(0, B_STATE, BLOCK_S):
                s = s_start + tl.arange(0, BLOCK_S)
                mask_s = s < B_STATE
                g = h // REPEAT  # 4
                B_off = b * B_stride_b + n * B_stride_n + j * B_stride_k + g * B_stride_g + s * B_stride_s
                C_off = b * C_stride_b + n * C_stride_n + i * C_stride_k + g * C_stride_g + s * C_stride_s
                B_vals = tl.load(B_ptr + B_off, mask=mask_s, other=0.0)
                C_vals = tl.load(C_ptr + C_off, mask=mask_s, other=0.0)
                acc += tl.sum(B_vals * C_vals, axis=0)
            G_off = b * G_stride_b + n * G_stride_n + i * G_stride_i + j * G_stride_j + h * G_stride_h
            tl.store(G_ptr + G_off, acc)

# Triton kernel: final reduction to produce Y_diag: [B, N, K, H, D]
# G: [B, N, K, K, H], L: [B, H, N, K, K] (note: last dim is H), hidden: [B, N, K, H, D]
# M = G * L, Y_diag[b, n, i, h, d] = sum_j M[b, n, i, j, h] * hidden[b, n, j, h, d]
@triton.jit
def reduce_M_with_hidden(G_ptr, L_ptr, hidden_ptr, out_ptr,
                          B_batch, B_n, B_K, B_H, B_D,
                          G_stride_b, G_stride_n, G_stride_i, G_stride_j, G_stride_h,
                          L_stride_b, L_stride_n, L_stride_i, L_stride_j, L_stride_h,
                          hidden_stride_b, hidden_stride_n, hidden_stride_k, hidden_stride_h, hidden_stride_d,
                          out_stride_b, out_stride_n, out_stride_k, out_stride_h, out_stride_d,
                          K: tl.constexpr, D: tl.constexpr, BLOCK_D: tl.constexpr):
    b = tl.program_id(0)
    n = tl.program_id(1)
    i = tl.program_id(2)
    h = tl.program_id(3)
    # Accumulator over D
    for d_start in range(0, D, BLOCK_D):
        d = d_start + tl.arange(0, BLOCK_D)
        mask_d = d < D
        # acc: [BLOCK_D]
        acc = tl.zeros([BLOCK_D], dtype=tl.float32)
        # Sum over j
        for j in range(K):
            G_off = b * G_stride_b + n * G_stride_n + i * G_stride_i + j * G_stride_j + h * G_stride_h
            L_off = b * L_stride_b + h * L_stride_h + n * L_stride_n + i * L_stride_i + j * L_stride_j
            G_val = tl.load(G_ptr + G_off)
            L_val = tl.load(L_ptr + L_off)
            M_val = G_val * L_val
            h_off = b * hidden_stride_b + n * hidden_stride_n + j * hidden_stride_k + h * hidden_stride_h + d * hidden_stride_d
            h_vals = tl.load(hidden_ptr + h_off, mask=mask_d, other=0.0)
            acc += M_val * h_vals
        # Store acc into out[b, n, i, h, d]
        out_off = b * out_stride_b + n * out_stride_n + i * out_stride_k + h * out_stride_h + d * out_stride_d
        tl.store(out_ptr + out_off, acc, mask=mask_d)

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states: torch.Tensor,
                A_cumsum: torch.Tensor,
                B: torch.Tensor,
                C: torch.Tensor) -> torch.Tensor:
        """
        Triton-only implementation matching the original Model.run logic:
        - hidden_states: [B, N, K, H, D]
        - A_cumsum: [B, H, N, K, K]
        - B: [B, N, K, n_groups, STATE_SIZE]
        - C: [B, N, K, n_groups, STATE_SIZE]
        Returns Y_diag: [B, N, K, H, D] in bfloat16.
        """
        # Ensure we are on CUDA for Triton
        assert hidden_states.is_cuda and A_cumsum.is_cuda and B.is_cuda and C.is_cuda, "Inputs must be CUDA tensors"
        B_batch, N, K, H, D = hidden_states.shape
        # Enforce constants to match original code; if mismatch, fallback to PyTorch path
        if not (H == NUM_HEADS and K == CHUNK_SIZE and D == HEAD_DIM and B.shape[1] == N and B.shape[2] == K and B.shape[4] == STATE_SIZE and
                C.shape[1] == N and C.shape[2] == K and C.shape[4] == STATE_SIZE and
                A_cumsum.shape[0] == B_batch and A_cumsum.shape[3] == K and A_cumsum.shape[4] == K and A_cumsum.shape[1] == H and A_cumsum.shape[2] == N):
            # Fallback to pure PyTorch for safety
            # Compute as in original Model.run:
            # 1) A_cumsum with lower-triangular mask (diagonal=-1) via cumsum
            # 2) L = exp(masked cumsum)
            # 3) G = contract(B, C) with head expansion
            # 4) M = G * L
            # 5) Y_diag = sum over j of M * hidden
            # Note: we'll use the original code logic but with PyTorch ops.
            # Step 1: build mask and masked A
            # A: [B, H, N, K, K]
            A = A_cumsum
            mask = torch.tril(torch.ones(K, K, dtype=torch.bool, device=A.device), diagonal=-1)  # [K, K]
            # expand to [B,H,N,K,K]
            mask_4d = mask[None, :, None, :, :].expand(B_batch, H, N, K, K).contiguous()
            A_masked = A.masked_fill(~mask_4d, 0.0)
            # Step 2: cumsum along last dim, then exp
            A_cumsum_seg = torch.cumsum(A_masked, dim=-1)  # last dim j
            # Keep only lower triangle, set others to -inf for exp; but since we compute on masked, exp of zeros above is 1?
            # Better: recompute L: exp of cumsum, but above should be 0? We need L to be 0 above, since original masked cumsum excludes upper.
            # Simpler: we will not rely on this here; instead, compute L directly in PyTorch for correctness:
            L = torch.exp(torch.cumsum(A_masked, dim=-1))
            # Step 3: contract B and C -> G
            # B: [B,N,K,n_groups,STATE_SIZE], C: [B,N,K,n_groups,STATE_SIZE]
            # original code expands heads: repeat_interleave(NUM_HEADS // N_GROUPS) -> here REPEAT=4
            g = torch.arange(H, device=B.device) // REPEAT  # [H]
            # G[i,j,h] = sum_s C[b,n,i,g,h,s] * B[b,n,j,g,h,s]
            # Let's reshape: gather per (b,n,i,h) then dot over s
            B_expanded = B  # already per group
            C_expanded = C
            # Build G with broadcasting and sum over s
            # This is inefficient; but correctness first.
            G = torch.zeros((B_batch, N, K, K, H), dtype=torch.float32, device=hidden_states.device)
            # Loop h to fill G: compute per h
            for h_idx in range(H):
                g_idx = (h_idx // REPEAT)  # if REPEAT=4, groups=N_GROUPS=8
                B_s = B[:, :, :, g_idx, :]   # [B,N,K,STATE_SIZE]
                C_s = C[:, :, :, g_idx, :]   # [B,N,K,STATE_SIZE]
                # Compute G[:, :, :, :, h_idx] via outer-product over s
                # G[i,j,h_idx] = sum_s C_s[i,s] * B_s[j,s]
                # Implement with broadcasting:
                # C_s expanded over j: [B,N,K,1,STATE_SIZE], B_s expanded over i: [B,N,1,K,STATE_SIZE]
                # But we need C_s per i and B_s per j. Simpler: compute per (i,j) pair in loop.
                # For performance, we can use einsum; but since this is fallback, any method works.
                # We'll use torch.einsum:
                # Prepare B_s_j: [B,N,1,K,STATE_SIZE], C_s_i: [B,N,K,1,STATE_SIZE]
                # But einsum needs aligned dims; use outer-product:
                # G[:, :, i, j, h_idx] = dot(C_s[:, :, i, :], B_s[:, :, j, :]) over s
                # This is B(N,K) x C(N,K) per (i,j). Since STATE_SIZE=64, fine.
                # We can compute via broadcasting:
                # Construct indices
                # Note: einsum 'ij,js->is' is sum over s; but here per (i,j), we need to loop i and j to fill G.
                # Let's do explicit:
                for i_idx in range(K):
                    for j_idx in range(K):
                        # sum over s: C[:, :, i_idx, s] * B[:, :, j_idx, s]
                        sum_s = 0.0
                        for s in range(STATE_SIZE):
                            c = C[:, :, i_idx, g_idx, s]  # [B,N]
                            b = B[:, :, j_idx, g_idx, s]  # [B,N]
                            sum_s += (c[:, :, None] * b[:, :, None]).sum(dim=1)  # [B,N]
                        # Write into G at fixed i_idx, j_idx, h_idx
                        G[:, :, i_idx, j_idx, h_idx] = sum_s
            # Step 4: M = G * L
            M = G * L
            # Step 5: Y_diag[b,n,i,h,d] = sum_j M[b,n,i,j,h] * hidden[b,n,j,h,d]
            out = torch.empty((B_batch, N, K, H, D), dtype=torch.float32, device=hidden_states.device)
            hidden_f = hidden_states.to(torch.float32)
            for i_idx in range(K):
                for h_idx in range(H):
                    M_ih = M[:, :, i_idx, :, h_idx]  # [B,N,K]
                    for j_idx in range(K):
                        M_val = M_ih[:, :, j_idx]  # [B,N]
                        # broadcast over d: hidden[:, :, j_idx, h_idx, :]
                        # Compute sum over d
                        # We need to accumulate over j for each d. Since hidden is [B,N,K,H,D], and D is 64 here,
                        # we can do per d:
                        # out[:, :, i_idx, h_idx, d] = sum_j M_val * hidden[:, :, j, h_idx, d]
                        # Let's vectorize over d
                        for d_start in range(0, D, 16):
                            d = d_start + torch.arange(16, device=hidden_states.device)
                            mask_d = d < D
                            acc = torch.zeros((B_batch, N), dtype=torch.float32, device=hidden_states.device)
                            for j in range(K):
                                # hidden[:, :, j, h_idx, d]
                                hidden_j = hidden_f[:, :, j, h_idx, d]
                                acc += M_val[:, :, None] * hidden_j
                            out[:, :, i_idx, h_idx, d] = acc
            return out.to(torch.bfloat16)
        # Triton path: enforce shapes match constants
        # Prepare L: [B, H, N, K, K] float32
        L = torch.empty((B_batch, H, N, K, K), dtype=torch.float32, device=hidden_states.device)

        # Strides and grid for masked cumsum + exp
        A_stride_b, A_stride_h, A_stride_n, A_stride_i, A_stride_j = A_cumsum.stride()
        L_stride_b, L_stride_n, L_stride_i, L_stride_j, L_stride_h = L.stride()
        grid1 = (B_batch, H, N, K)
        masked_cumsum_tril_exp[grid1](
            A_cumsum, L,
            B_batch, H, N,
            A_stride_b, A_stride_h, A_stride_n, A_stride_i, A_stride_j,
            L_stride_b, L_stride_n, L_stride_i, L_stride_j, L_stride_h,
            CHUNK=CHUNK_SIZE, num_warps=2, num_stages=2
        )

        # Prepare G: [B, N, K, K, H] float32
        B_ng = N_GROUPS
        G = torch.empty((B_batch, N, K, K, H), dtype=torch.float32, device=hidden_states.device)

        # Strides for B, C, G
        B_stride_b, B_stride_n, B_stride_k, B_stride_g, B_stride_s = B.stride(0), B.stride(1), B.stride(2), B.stride(3), B.stride(4)
        C_stride_b, C_stride_n, C_stride_k, C_stride_g, C_stride_s = C.stride(0), C.stride(1), C.stride(2), C.stride(3), C.stride(4)
        G_stride_b, G_stride_n, G_stride_i, G_stride_j, G_stride_h = G.stride(0), G.stride(1), G.stride(2), G.stride(3), G.stride(4)

        # Launch contraction kernel
        grid2 = (B_batch, N, H)
        contract_BC_to_G[grid2](
            B, C, G,
            B_batch, N, K, B_ng, STATE_SIZE,
            G_stride_b, G_stride_n, G_stride_i, G_stride_j, G_stride_h,
            B_stride_b, B_stride_n, B_stride_k, B_stride_g, B_stride_s,
            C_stride_b, C_stride_n, C_stride_k, C_stride_g, C_stride_s,
            CHUNK=CHUNK_SIZE, BLOCK_S=64, H=H, num_warps=4, num_stages=2
        )

        # Output buffer: [B, N, K, H, D] float32
        out = torch.empty((B_batch, N, K, H, D), dtype=torch.float32, device=hidden_states.device)

        # Strides for hidden and out
        hidden_stride_b, hidden_stride_n, hidden_stride_k, hidden_stride_h, hidden_stride_d = hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2), hidden_states.stride(3), hidden_states.stride(4)
        out_stride_b, out_stride_n, out_stride_k, out_stride_h, out_stride_d = out.stride(0), out.stride(1), out.stride(2), out.stride(3), out.stride(4)

        # Launch final reduction kernel
        grid3 = (B_batch, N, K, H)
        reduce_M_with_hidden[grid3](
            G, L, hidden_states, out,
            B_batch, N, K, H, D,
            G_stride_b, G_stride_n, G_stride_i, G_stride_j, G_stride_h,
            L_stride_b, L_stride_n, L_stride_i, L_stride_j, L_stride_h,
            hidden_stride_b, hidden_stride_n, hidden_stride_k, hidden_stride_h, hidden_stride_d,
            out_stride_b, out_stride_n, out_stride_k, out_stride_h, out_stride_d,
            K=CHUNK_SIZE, D=D, BLOCK_D=64, num_warps=4, num_stages=2
        )

        return out.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
