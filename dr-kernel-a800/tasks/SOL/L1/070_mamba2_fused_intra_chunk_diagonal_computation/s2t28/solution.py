import torch
import triton
import triton.language as tl


# Constants (compile-time for Triton)
CHUNK_SIZE = 128
NUM_HEADS = 32
HEAD_DIM = 128


# Kernel 1: build L_seg (segment cumulative sum with lower-triangular mask and exp)
# A: [B, num_heads, num_chunks, chunk_size]
# L_seg: [B, num_heads, num_chunks, chunk_size] (we compute per (b,h,i,k))
@triton.jit
def build_L_seg_kernel(
    A_ptr, L_seg_ptr,
    A_stride_b, A_stride_h, A_stride_i, A_stride_k,
    L_stride_b, L_stride_h, L_stride_i, L_stride_k,
    num_chunks,
    BLOCK_K: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    i = tl.program_id(2)

    acc = 0.0
    for k_start in range(0, CHUNK_SIZE, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < CHUNK_SIZE
        include = mask_k & (k_offsets <= i)
        A_addr = A_ptr + b * A_stride_b + h * A_stride_h + i * A_stride_i + k_offsets * A_stride_k
        A_vals = tl.load(A_addr, mask=include, other=0.0)
        tile_sum = tl.sum(A_vals, axis=0)
        acc += tile_sum
        L_addr = L_seg_ptr + b * L_stride_b + h * L_stride_h + i * L_stride_i + k_offsets * L_stride_k
        tl.store(L_addr, acc, mask=mask_k)

    # Exponentiate segment sums: L_seg = exp(L_seg)
    for k in range(CHUNK_SIZE):
        val = tl.load(L_seg_ptr + b * L_stride_b + h * L_stride_h + i * L_stride_i + k * L_stride_k)
        val = tl.exp(val)
        tl.store(L_seg_ptr + b * L_stride_b + h * L_stride_h + i * L_stride_i + k * L_stride_k, val)


# Kernel 2: compute G[i, j, h] = sum over k and s of C[i, k, h, s] * B[j, k, h, s]
# B: [B, num_chunks, chunk_size, num_heads, state_size]
# C: [B, num_chunks, chunk_size, num_heads, state_size]
# G: [B, num_chunks, chunk_size, num_heads]
@triton.jit
def compute_G_kernel(
    B_ptr, C_ptr, G_ptr,
    B_stride_b, B_stride_ci, B_stride_ck, B_stride_ch, B_stride_cs,
    C_stride_b, C_stride_ci, C_stride_ck, C_stride_ch, C_stride_cs,
    G_stride_b, G_stride_ci, G_stride_ck, G_stride_ch,
    num_chunks,
    BLOCK_K: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    b = tl.program_id(0)
    ci = tl.program_id(1)  # i
    ch = tl.program_id(2)  # h
    j = tl.program_id(3)   # j

    # We store G[ci, j, ch] for this (b, ci, ch, j)
    # Accumulate G_val in float32
    G_val = 0.0
    for k_start in range(0, CHUNK_SIZE, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < CHUNK_SIZE
        for s_start in range(0, NUM_HEADS, BLOCK_S):
            s_offsets = s_start + tl.arange(0, BLOCK_S)
            mask_s = s_offsets < NUM_HEADS
            # Load B[j, k, h, s] and C[ci, k, h, s]
            B_addr = B_ptr + b * B_stride_b + ci * B_stride_ci + k_offsets[:, None] * B_stride_ck + ch * B_stride_ch + s_offsets[None, :] * B_stride_cs
            C_addr = C_ptr + b * C_stride_b + ci * C_stride_ci + k_offsets[:, None] * C_stride_ck + ch * C_stride_ch + s_offsets[None, :] * C_stride_cs
            B_vals = tl.load(B_addr, mask=mask_k[:, None] & mask_s[None, :], other=0.0)
            C_vals = tl.load(C_addr, mask=mask_k[:, None] & mask_s[None, :], other=0.0)
            # Outer product and sum over s then k tiles
            # First sum over s axis (rows)
            prod = B_vals * C_vals
            # Reduce over s (axis=1)
            tile_sum = tl.sum(prod, axis=1)  # shape [BLOCK_K]
            G_val += tl.sum(tile_sum, axis=0)  # scalar
    # Store G[ci, j, ch]
    G_addr = G_ptr + b * G_stride_b + ci * G_stride_ci + j * G_stride_ck + ch * G_stride_ch
    tl.store(G_addr, G_val)


# Kernel 3: M = G * L_seg elementwise for each j and h
# G: [B, num_chunks, chunk_size, num_heads]
# L_seg: [B, num_heads, num_chunks, chunk_size]
# M: [B, num_chunks, chunk_size, num_heads]
@triton.jit
def multiply_LG_kernel(
    G_ptr, L_seg_ptr, M_ptr,
    G_stride_b, G_stride_ci, G_stride_ck, G_stride_ch,
    L_stride_b, L_stride_h, L_stride_i, L_stride_k,
    M_stride_b, M_stride_ci, M_stride_ck, M_stride_ch,
    num_chunks,
    BLOCK_J: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    b = tl.program_id(0)
    ci = tl.program_id(1)  # i
    ch = tl.program_id(2)  # h

    # Loop over j in tiles of BLOCK_J
    for j_start in range(0, CHUNK_SIZE, BLOCK_J):
        j_offsets = j_start + tl.arange(0, BLOCK_J)
        mask_j = j_offsets < CHUNK_SIZE
        # Loop over h in tiles of BLOCK_H (here h is fixed by program_id, but we keep loop for generality)
        for h_start in range(0, NUM_HEADS, BLOCK_H):
            h_offsets = h_start + tl.arange(0, BLOCK_H)
            mask_h = h_offsets < NUM_HEADS
            # G[ci, j, h] and L_seg[j, h, ci, k]
            G_addr = G_ptr + b * G_stride_b + ci * G_stride_ci + j_offsets[:, None] * G_stride_ck + h_offsets[None, :] * G_stride_ch
            M_addr = M_ptr + b * M_stride_b + ci * M_stride_ci + j_offsets[:, None] * M_stride_ck + h_offsets[None, :] * M_stride_ch

            # We need to load L_seg[j, h, ci, k] where j is the first dim (broadcast), h is second dim (broadcast), ci fixed, k over chunk_size
            # We'll compute per j and h scalar to avoid large 2D loads; since j_offsets and h_offsets are small tile, we handle one j at a time
            for jj in range(BLOCK_J):
                j_j = j_start + jj
                if j_j < CHUNK_SIZE:
                    # For each h in the tile
                    for hh in range(BLOCK_H):
                        h_h = h_start + hh
                        if h_h < NUM_HEADS:
                            G_val = tl.load(G_addr + jj * G_stride_ck + hh * G_stride_ch, mask=mask_j[jj] & mask_h[hh], other=0.0)
                            # For L_seg[j, h, ci, k], we need to load for all k; we loop over k in tiles
                            L_seg_tile = tl.zeros([BLOCK_K], dtype=tl.float32)
                            for k_start in range(0, CHUNK_SIZE, BLOCK_K):
                                k_offsets = k_start + tl.arange(0, BLOCK_K)
                                mask_k = k_offsets < CHUNK_SIZE
                                L_addr = L_seg_ptr + b * L_stride_b + h_h * L_stride_h + ci * L_stride_i + k_offsets * L_stride_k
                                L_vals = tl.load(L_addr, mask=mask_k, other=0.0)
                                L_seg_tile += L_vals
                            # Multiply
                            M_val = G_val * tl.exp(L_seg_tile[0])  # simple placeholder; need to fix below
                            # Store M[b, ci, j, h]
                            M_addr_val = M_addr + jj * M_stride_ck + hh * M_stride_ch
                            tl.store(M_addr_val, M_val, mask=(mask_j[jj] & mask_h[hh]))

    # Note: The above simple placeholder for M_val is incorrect because it uses only k=0. We need to restructure kernel to compute full M tile.
    # Below is a corrected version that computes the full M tile by loading L_seg per (j, h) and multiplying with G per (j, h).

    # Correct implementation: loop j and h explicitly
    for j in range(CHUNK_SIZE):
        for h in range(NUM_HEADS):
            # G_val = G[b, ci, j, h]
            G_addr_jh = G_ptr + b * G_stride_b + ci * G_stride_ci + j * G_stride_ck + h * G_stride_ch
            G_val = tl.load(G_addr_jh)
            # L_seg_val = L_seg[b, h, ci, j] (we already computed L_seg in step 1)
            L_addr_jh = L_seg_ptr + b * L_stride_b + h * L_stride_h + ci * L_stride_i + j * L_stride_k
            L_val = tl.load(L_addr_jh)
            M_addr_jh = M_ptr + b * M_stride_b + ci * M_stride_ci + j * M_stride_ck + h * M_stride_ch
            tl.store(M_addr_jh, G_val * L_val)


# Kernel 4: contract M with hidden_states to produce Y_diag
# hidden_states: [B, num_chunks, chunk_size, num_heads, head_dim]
# M: [B, num_chunks, chunk_size, num_heads]
# Y_diag: [B, num_chunks, chunk_size, num_heads, head_dim]
@triton.jit
def contract_M_hidden_kernel(
    M_ptr, hidden_ptr, Y_ptr,
    M_stride_b, M_stride_ci, M_stride_ck, M_stride_ch,
    hidden_stride_b, hidden_stride_ci, hidden_stride_ck, hidden_stride_ch, hidden_stride_cd,
    Y_stride_b, Y_stride_ci, Y_stride_ck, Y_stride_ch, Y_stride_cd,
    num_chunks,
    BLOCK_J: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_CD: tl.constexpr,
):
    b = tl.program_id(0)
    ci = tl.program_id(1)  # i
    ck = tl.program_id(2)  # k
    ch = tl.program_id(3)  # h

    # For each cd in tiles of BLOCK_CD, compute Y[b, ci, ck, ch, cd] = sum_j M[b, ci, j, ch] * hidden[b, ci, ck, j, ch, cd]
    for cd_start in range(0, HEAD_DIM, BLOCK_CD):
        cd_offsets = cd_start + tl.arange(0, BLOCK_CD)
        mask_cd = cd_offsets < HEAD_DIM
        Y_addr_base = Y_ptr + b * Y_stride_b + ci * Y_stride_ci + ck * Y_stride_ck + ch * Y_stride_ch + cd_offsets * Y_stride_cd
        # Accumulate across j
        acc = tl.zeros([BLOCK_CD], dtype=tl.float32)
        for j_start in range(0, CHUNK_SIZE, BLOCK_J):
            j_offsets = j_start + tl.arange(0, BLOCK_J)
            mask_j = j_offsets < CHUNK_SIZE
            # For each j in tile, accumulate M[b, ci, j, ch] * hidden[b, ci, ck, j, ch, cd]
            for jj in range(BLOCK_J):
                j_j = j_start + jj
                if j_j < CHUNK_SIZE:
                    M_val = tl.load(M_ptr + b * M_stride_b + ci * M_stride_ci + j_j * M_stride_ck + ch * M_stride_ch)
                    # hidden[b, ci, ck, j_j, ch, cd_offsets]
                    hidden_addr = hidden_ptr + b * hidden_stride_b + ci * hidden_stride_ci + ck * hidden_stride_ck + j_j * hidden_stride_ck + ch * hidden_stride_ch + cd_offsets * hidden_stride_cd
                    hidden_vals = tl.load(hidden_addr, mask=mask_cd, other=0.0)
                    acc += M_val * hidden_vals
        # Store Y[b, ci, ck, ch, cd_offsets]
        tl.store(Y_addr_base, acc, mask=mask_cd)

    # Since we used tiles, cast output to bfloat16
    # Here we assume M and hidden are float32; Triton will produce float32. We can cast on the host after this kernel.
    # However, ModelNew returns bfloat16; we cast after all kernel runs (host side) to avoid Triton dtype issues.


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                A_cumsum: torch.Tensor,
                B: torch.Tensor,
                C: torch.Tensor) -> torch.Tensor:
        # Shapes
        B_b, B_ci, B_ck, B_ch, B_cs = B.shape
        C_b, C_ci, C_ck, C_ch, C_cs = C.shape
        assert B_b == C_b and B_ci == C_ci and B_ck == C_ck and B_ch == C_ch and B_cs == C_cs, "B and C must have same shape"

        # Output Y_diag (we compute in float32 then cast to bfloat16 at end)
        # hidden_states: [B, num_chunks, chunk_size, num_heads, head_dim]
        # Allocate M as float32: [B, num_chunks, chunk_size, num_heads]
        M = torch.empty((B_b, B_ci, B_ck, B_ch), dtype=torch.float32, device=hidden_states.device)

        # Allocate L_seg as float32: [B, num_heads, num_chunks, chunk_size]
        L_seg = torch.empty((B_b, B_ch, B_ci, B_ck), dtype=torch.float32, device=hidden_states.device)

        # Kernel 1: build L_seg
        grid_L = (B_b, B_ch, B_ci)
        build_L_seg_kernel[grid_L](
            A_cumsum, L_seg,
            A_cumsum.stride(0), A_cumsum.stride(1), A_cumsum.stride(2), A_cumsum.stride(3),
            L_seg.stride(0), L_seg.stride(1), L_seg.stride(2), L_seg.stride(3),
            B_ci,
            BLOCK_K=CHUNK_SIZE,  # one-pass
        )

        # Kernel 2: compute G
        G = torch.empty((B_b, B_ci, B_ck, B_ch), dtype=torch.float32, device=hidden_states.device)

        grid_G = (B_b, B_ci, B_ck, B_ch)
        compute_G_kernel[grid_G](
            B, C, G,
            B.stride(0), B.stride(1), B.stride(2), B.stride(3), B.stride(4),
            C.stride(0), C.stride(1), C.stride(2), C.stride(3), C.stride(4),
            G.stride(0), G.stride(1), G.stride(2), G.stride(3),
            B_ci,
            BLOCK_K=CHUNK_SIZE,
            BLOCK_S=NUM_HEADS,
        )

        # Kernel 3: M = G * L_seg (elementwise)
        grid_M = (B_b, B_ci, B_ch)
        multiply_LG_kernel[grid_M](
            G, L_seg, M,
            G.stride(0), G.stride(1), G.stride(2), G.stride(3),
            L_seg.stride(0), L_seg.stride(1), L_seg.stride(2), L_seg.stride(3),
            M.stride(0), M.stride(1), M.stride(2), M.stride(3),
            B_ci,
            BLOCK_J=CHUNK_SIZE,
            BLOCK_H=NUM_HEADS,
        )

        # Kernel 4: contract M with hidden to produce Y_diag in float32
        Y_diag = torch.empty((B_b, B_ci, B_ck, B_ch, HEAD_DIM), dtype=torch.float32, device=hidden_states.device)
        grid_Y = (B_b, B_ci, B_ck, B_ch)
        contract_M_hidden_kernel[grid_Y](
            M, hidden_states, Y_diag,
            M.stride(0), M.stride(1), M.stride(2), M.stride(3),
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2), hidden_states.stride(3), hidden_states.stride(4),
            Y_diag.stride(0), Y_diag.stride(1), Y_diag.stride(2), Y_diag.stride(3), Y_diag.stride(4),
            B_ci,
            BLOCK_J=CHUNK_SIZE,
            BLOCK_H=NUM_HEADS,
            BLOCK_CD=HEAD_DIM,
        )

        # Cast to bfloat16 to match original return type
        return Y_diag.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
