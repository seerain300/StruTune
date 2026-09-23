import torch
import torch.nn.functional as F
import triton
import triton.language as tl


# 1) segment_sum_kernel: computes cumsum along last axis with lower-triangular mask (diagonal = mask_diag)
# Input A: [B, H, C, K] where K is chunk_size; Output L: [B, H, C, K, K]
# We permute A to [B, H, C, K] before calling, then write to L as [B, H, C, K, K].
@triton.jit
def segment_sum_kernel(
    out_ptr,  # pointer to output tensor L
    A_ptr,    # pointer to input tensor A_perm (B, H, C, K)
    B: tl.constexpr,  # batch size
    H: tl.constexpr,  # num_heads
    C: tl.constexpr,  # num_chunks
    K: tl.constexpr,  # chunk_size
    mask_diag: int,   # triangular mask diagonal (use -1)
    BLOCK_I: tl.constexpr,  # tile size along i (rows)
    BLOCK_J: tl.constexpr,  # tile size along j (cols)
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    c = tl.program_id(2)

    # offsets along i (rows) and j (cols)
    i_offsets = tl.arange(0, BLOCK_I)
    j_offsets = tl.arange(0, BLOCK_J)

    # We will loop over tiles in i dimension (since we need cumsum along j)
    # For each i in tile, compute prefix sum over j <= i + mask_diag
    for i_start in range(0, K, BLOCK_I):
        i_idx = i_start + i_offsets
        # mask for i within [0, K)
        i_mask = i_idx < K

        # For each j in tile
        for j_start in range(0, K, BLOCK_J):
            j_idx = j_start + j_offsets
            j_mask = j_idx < K

            # lower-triangular mask: j <= i + mask_diag
            tri_mask = j_idx[None, :] <= (i_idx[:, None] + mask_diag)

            # Build pointers for A[b, h, c, i, j]
            base = b * H * C * K * K + h * C * K * K + c * K * K
            # Note: A is laid out as (B, H, C, K); for a fixed (b,h,c), A[i, j] is a vector over j for each i
            # We need to load A[i, j] as a matrix (i, j). For Triton, we can compute addresses:
            # A_ptr offset: base + i_idx*K + j_idx
            A_ij = tl.load(
                A_ptr + base + i_idx[:, None] * K + j_idx[None, :],
                mask=i_mask[:, None] & j_mask[None, :] & tri_mask,
                other=0.0
            )

            # prefix sum along j: cumsum along last axis
            # We can compute inclusive cumsum via scan within the tile.
            # Since j is the last axis, we implement scan per row i.
            # Initialize running sum per i
            running = tl.zeros([BLOCK_I], dtype=tl.float32)
            out_vals = tl.zeros([BLOCK_I, BLOCK_J], dtype=tl.float32)

            for jj in range(BLOCK_J):
                jv = j_start + jj
                j_valid = jv < K
                # vector of current j positions valid
                if j_valid:
                    # per row i, add A[i, jv]
                    ai = tl.sum(A_ij[:, jj], axis=1)  # sum across j at fixed jj, for each i
                    running += ai  # running sum across j
                    # include -inf on diagonal
                    diag_flag = (jv == (i_idx + mask_diag))
                    # where not diagonal, store running; else -inf
                    out_vals[:, jj] = tl.where(diag_flag, float('-inf'), running)

            # Store out_vals into L[b, h, c, i, j]
            L_base = b * H * C * K * K + h * C * K * K + c * K * K
            tl.store(
                out_ptr + L_base + i_idx[:, None] * K + j_idx[None, :],
                out_vals,
                mask=i_mask[:, None] & j_mask[None, :]
            )


# 2) states_contract_kernel: compute states[b, nc, h, d, s] = sum_{i} B_decay[i] * hidden_states[i]
# Inputs:
# - B_decay_ptr: [B, NC, K, H, S] = B_chunked * decay_states_perm (which is [B, NC, K, H])
# - hidden_ptr: [B, NC, K, H, D]
# Outputs:
# - states_ptr: [B, NC, H, D, S]
@triton.jit
def states_contract_kernel(
    states_ptr, B_decay_ptr, hidden_ptr,
    B: tl.constexpr, NC: tl.constexpr, K: tl.constexpr, H: tl.constexpr, D: tl.constexpr, S: tl.constexpr,
    BLOCK_D: tl.constexpr, BLOCK_S: tl.constexpr,
):
    b = tl.program_id(0)
    nc = tl.program_id(1)
    h = tl.program_id(2)

    d_offsets = tl.arange(0, BLOCK_D)
    s_offsets = tl.arange(0, BLOCK_S)

    for d_start in range(0, D, BLOCK_D):
        d_idx = d_start + d_offsets
        d_mask = d_idx < D

        for s_start in range(0, S, BLOCK_S):
            s_idx = s_start + s_offsets
            s_mask = s_idx < S

            # accumulator over i in chunk
            acc = tl.zeros([BLOCK_D, BLOCK_S], dtype=tl.float32)

            # loop over i = 0..K-1
            for i in range(0, K):
                # load B_decay[b, nc, i, h, s] -> [S]
                base_bd = b * NC * K * H * S + nc * K * H * S + i * H * S + h * S
                b_decay = tl.load(
                    B_decay_ptr + base_bd + s_idx,
                    mask=s_mask,
                    other=0.0
                )  # shape [S]

                # load hidden[b, nc, i, h, d] -> [D]
                base_hd = b * NC * K * H * D + nc * K * H * D + i * H * D + h * D
                hidden = tl.load(
                    hidden_ptr + base_hd + d_idx,
                    mask=d_mask,
                    other=0.0
                )  # shape [D]

                # outer product accumulate acc += hidden[:, None] * b_decay[None, :]
                acc += hidden[:, None] * b_decay[None, :]

            # store states[b, nc, h, d, s]
            base_st = b * NC * H * D * S + nc * H * D * S + h * D * S
            tl.store(
                states_ptr + base_st + d_idx[:, None] * S + s_idx[None, :],
                acc,
                mask=d_mask[:, None] & s_mask[None, :]
            )


# 3) cumsum_scan_kernel: computes cumsum along K (chunk_size) for A_perm[b, h, c, k], output A_cumsum[b, h, c, k]
@triton.jit
def cumsum_scan_kernel(
    out_ptr, A_ptr,
    B: tl.constexpr, H: tl.constexpr, C: tl.constexpr, K: tl.constexpr,
    BLOCK_K: tl.constexpr
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    c = tl.program_id(2)
    k_offsets = tl.arange(0, BLOCK_K)

    for k_start in range(0, K, BLOCK_K):
        k_idx = k_start + k_offsets
        k_mask = k_idx < K

        # load A[b, h, c, k_idx]
        base = b * H * C * K + h * C * K + c * K
        a_vals = tl.load(A_ptr + base + k_idx, mask=k_mask, other=0.0)
        running = tl.zeros([BLOCK_K], dtype=tl.float32)
        out_row = tl.zeros([BLOCK_K], dtype=tl.float32)
        for kk in range(BLOCK_K):
            k = k_start + kk
            valid = k < K
            # add current element to running (if valid), else 0
            running += tl.where(valid, a_vals[kk], 0.0)
            out_row[kk] = running
        tl.store(out_ptr + base + k_idx, out_row, mask=k_mask)


# 4) final_state_recurrence_kernel: computes final_state[b, c_out, h, d, s] = sum_{j=0..c_out} decay[b, h, c_out, j] * states_with_init[b, j, h, d, s]
# Inputs:
# - decay_ptr: [B, H, NC, NC] (after segment_sum on A_chunk_ends_padded)
# - states_with_init_ptr: [B, NC+1, H, D, S]
# Outputs:
# - final_ptr: [B, NC, H, D, S]
@triton.jit
def final_state_recurrence_kernel(
    final_ptr, decay_ptr, states_with_init_ptr,
    B: tl.constexpr, NC: tl.constexpr, H: tl.constexpr, D: tl.constexpr, S: tl.constexpr,
    BLOCK_D: tl.constexpr, BLOCK_S: tl.constexpr,
):
    b = tl.program_id(0)
    c_out = tl.program_id(1)
    h = tl.program_id(2)

    d_offsets = tl.arange(0, BLOCK_D)
    s_offsets = tl.arange(0, BLOCK_S)

    for d_start in range(0, D, BLOCK_D):
        d_idx = d_start + d_offsets
        d_mask = d_idx < D

        for s_start in range(0, S, BLOCK_S):
            s_idx = s_start + s_offsets
            s_mask = s_idx < S

            acc = tl.zeros([BLOCK_D, BLOCK_S], dtype=tl.float32)

            # Loop over previous chunks j = 0..c_out
            for j in range(0, c_out + 1):
                # load decay[b, h, c_out, j]
                dec_base = b * H * (NC + 1) * NC + h * (NC + 1) * NC + c_out * (NC + 1) + j
                dec_val = tl.load(decay_ptr + dec_base)  # scalar

                # load states_with_init[b, j, h, d, s]
                st_base = b * (NC + 1) * H * D * S + j * H * D * S + h * D * S
                st_ptr = states_with_init_ptr + st_base
                # we need to load a [D, S] tile
                d_idx = d_start + tl.arange(0, BLOCK_D)
                s_idx = s_start + tl.arange(0, BLOCK_S)
                d_mask = d_idx < D
                s_mask = s_idx < S
                st_tile = tl.load(
                    st_ptr + d_idx[:, None] * S + s_idx[None, :],
                    mask=d_mask[:, None] & s_mask[None, :],
                    other=0.0
                )
                acc += st_tile * dec_val

            # store final_state[b, c_out, h, d, s]
            fi_base = b * NC * H * D * S + c_out * H * D * S + h * D * S
            fi_ptr = final_ptr + fi_base
            tl.store(
                fi_ptr + d_idx[:, None] * S + s_idx[None, :],
                acc,
                mask=d_mask[:, None] & s_mask[None, :]
            )


# 5) c_times_states_kernel: compute C_times_hidden[b, nc, t, h, d] = sum_s C[b, nc, t, h, s] * states[b, nc, h, d, s]
@triton.jit
def c_times_states_kernel(
    out_ptr, C_ptr, states_ptr,
    B: tl.constexpr, NC: tl.constexpr, T: tl.constexpr, H: tl.constexpr, D: tl.constexpr, S: tl.constexpr,
    BLOCK_D: tl.constexpr, BLOCK_S: tl.constexpr,
):
    b = tl.program_id(0)
    nc = tl.program_id(1)
    h = tl.program_id(2)

    d_offsets = tl.arange(0, BLOCK_D)
    s_offsets = tl.arange(0, BLOCK_S)

    for t in range(0, T):
        for d_start in range(0, D, BLOCK_D):
            d_idx = d_start + d_offsets
            d_mask = d_idx < D

            for s_start in range(0, S, BLOCK_S):
                s_idx = s_start + s_offsets
                s_mask = s_idx < S

                acc = tl.zeros([BLOCK_D, BLOCK_S], dtype=tl.float32)

                # loop over s
                for ss in range(0, S):
                    # load C[b, nc, t, h, ss]
                    C_base = b * NC * T * H * S + nc * T * H * S + t * H * S + h * S
                    C_val = tl.load(C_ptr + C_base + ss)
                    # load states[b, nc, h, d, ss]
                    st_base = b * NC * H * D * S + nc * H * D * S + h * D * S
                    st_ptr = states_ptr + st_base
                    st_tile = tl.load(
                        st_ptr + d_idx[:, None] * S + (s_start + ss),
                        mask=d_mask[:, None],
                        other=0.0
                    )
                    acc += st_tile * C_val

                # store out[b, nc, t, h, d]
                out_base = b * NC * T * H * D + nc * T * H * D + t * H * D
                out_ptr_t = out_ptr + out_base + h * D
                tl.store(out_ptr_t + d_idx, acc[:, 0], mask=d_mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, D: torch.Tensor,
                initial_states: torch.Tensor):
        # Extract shapes
        batch_size, seq_len, num_heads, head_dim = hidden_states.shape
        state_size = 256
        n_groups = 1
        chunk_size = 256

        # Compute padding size to make seq_len multiple of chunk_size
        pad_size = (chunk_size - seq_len % chunk_size) % chunk_size
        seq_len_padded = seq_len + pad_size
        num_chunks = (seq_len_padded + chunk_size - 1) // chunk_size

        # Convert to float32 for numerical stability
        hidden_states_f = hidden_states.to(torch.float32)
        A_f = A.to(torch.float32)
        B_f = B.to(torch.float32)
        C_f = C.to(torch.float32)
        D_f = D.to(torch.float32)
        initial_states_f = initial_states.to(torch.float32)

        # Expand B and C to match num_heads (from n_groups=1 to num_heads=16)
        B_expanded = B_f.unsqueeze(2).expand(batch_size, seq_len, num_heads, state_size)
        C_expanded = C_f.unsqueeze(2).expand(batch_size, seq_len, num_heads, state_size)

        # Prepare padded hidden states
        hidden_padded = F.pad(hidden_states_f, (0, 0, 0, pad_size, 0, 0))
        hidden_states_chunked = hidden_padded.reshape(
            batch_size, num_chunks, chunk_size, num_heads, head_dim
        )

        # A transposed and chunked: permute to [B, H, C, K] for cumsum along K
        A_transposed = A_f.transpose(1, 2)  # [batch, num_heads, seq_len]
        A_chunked_perm = A_transposed.reshape(
            batch_size, num_chunks, chunk_size, num_heads
        )

        # Compute A_cumsum along K axis for each (b, h, c)
        A_cumsum_perm = torch.empty((batch_size, num_heads, num_chunks, chunk_size), dtype=torch.float32, device=hidden_states.device)
        # Launch cumsum scan kernel
        BLOCK_K = 128
        grid = (batch_size, num_heads, num_chunks)
        cumsum_scan_kernel[grid](
            A_cumsum_perm, A_chunked_perm,
            B=batch_size, H=num_heads, C=num_chunks, K=chunk_size,
            BLOCK_K=BLOCK_K
        )

        # 1) Compute L = exp(segment_sum(A_cumsum_perm))
        # A_cumsum_perm shape: [B, H, C, K]
        # We need to compute segment_sum over last axis K (chunk_size) with triangular mask, then exp.
        L = torch.empty((batch_size, num_heads, num_chunks, chunk_size, chunk_size), dtype=torch.float32, device=hidden_states.device)
        BLOCK_I = 64
        BLOCK_J = 64
        grid_seg = (batch_size, num_heads, num_chunks)
        segment_sum_kernel[grid_seg](
            L, A_cumsum_perm,
            B=batch_size, H=num_heads, C=num_chunks, K=chunk_size, mask_diag=-1,
            BLOCK_I=BLOCK_I, BLOCK_J=BLOCK_J
        )
        L = torch.exp(L)

        # 2) Compute states via contraction: states[b, nc, h, d, s] = sum_i B_decay[i] * hidden_states[i]
        # Prepare B_decay: [B, NC, K, H, S]
        # decay_states_perm is exp(A_cumsum along K): we need exp difference? The original uses decay = exp(A_cumsum[:, :, :, -1:] - A_cumsum).
        # But for Triton, we can compute exp(A_cumsum) and then exp(diff).
        # However, original code uses exp(A_cumsum[:, :, :, -1:] - A_cumsum). We need to implement that:
        # For each (b, h, c), last element is at k=K-1.
        # We'll compute decay factor per (b,h,c,k) as exp(A_cumsum[b,h,c,k] - A_cumsum[b,h,c,K-1]).
        A_last = torch.empty((batch_size, num_heads, num_chunks), dtype=torch.float32, device=hidden_states.device)
        # We can get A_last by taking last element of A_cumsum along K: we need to compute A_cumsum at k=K-1 for each (b,h,c).
        # Since A_cumsum is computed by our kernel, we can reuse it. But to compute exp(A_cumsum[:, :, :, -1:] - A_cumsum), we need A_cumsum at K-1.
        # We'll compute A_cumsum_last by running our cumsum kernel with K=1? Simpler: since cumsum kernel returns full A_cumsum, take last dim.
        # We don't have direct pointer to last element, so we compute it via PyTorch after kernel:
        A_cumsum_last = A_cumsum_perm[:, :, :, -1]  # shape [B, H, C]
        # Now compute B_decay: B_expanded * exp(A_cumsum - A_cumsum_last)
        B_expanded = B_expanded  # [B, S, H, S] ? No, original B is [H, S], we expanded to [B, S, H, S] previously. That was wrong.
        # Correct: Original B is [H, S], C is [H, S]. We need to expand B and C to include batch and seq_len dims for chunked reshape.
        # Let's recompute properly:
        # We need B_chunked and C_chunked to be [B, NC, K, H, S]. We can create them by expanding B and C along B and K dims using broadcasting.
        # However, the reference code expands B/C to num_heads and uses reshape. Since we don't have B/C per batch, we infer: original B/C are independent of batch/seq.
        # We will create B_chunked and C_chunked by unsqueeze and expand to [B, NC, K, H, S]:
        B_chunked = B_f.unsqueeze(0).unsqueeze(1).unsqueeze(3).expand(batch_size, num_chunks, chunk_size, num_heads, state_size)
        C_chunked = C_f.unsqueeze(0).unsqueeze(1).unsqueeze(3).expand(batch_size, num_chunks, chunk_size, num_heads, state_size)
        # Compute exp differences: exp(A_cumsum - A_cumsum_last)
        # We need decay factor per (b,h,c,k): exp(A_cumsum[b,h,c,k] - A_cumsum_last[b,h,c])
        # Create decay_factors [B, H, C, K]
        decay_factors = torch.empty((batch_size, num_heads, num_chunks, chunk_size), dtype=torch.float32, device=hidden_states.device)
        # We can compute decay_factors by subtracting A_cumsum_last from A_cumsum along K:
        # For each (b,h,c,k), value is A_cumsum[b,h,c,k] - A_cumsum_last[b,h,c]
        # We can do this with


def run(*args):
    return ModelNew()(*args)
