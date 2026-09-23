import torch
import triton
import triton.language as tl

# Constants as in the original model
CHUNK_SIZE = 128  # constexpr for Triton
NUM_HEADS = 32
N_GROUPS = 8
HEAD_DIM = 128  # state_size == head_dim

@triton.jit
def compute_G_partial_triton(
    B_ptr, C_ptr, Gout_ptr,
    B_bs, B_cs, B_cd, B_w, B_sd,  # B strides: (B, num_chunks, CHUNK_SIZE, NUM_HEADS, HEAD_DIM)
    C_bs, C_cs, C_cd, C_w, C_sd,  # C strides: (B, num_chunks, CHUNK_SIZE, NUM_HEADS, HEAD_DIM)
    G_bs, G_cs, G_cd, G_w, G_j, G_h,  # Gout strides: (B, num_chunks, CHUNK_SIZE, CHUNK_SIZE, NUM_HEADS)
    b_id: tl.constexpr,
    i_id: tl.constexpr,
):
    """
    Compute G[i, j, h] = sum over s of C[b, i, k, h, s] * B[b, j, k, h, s], for all k, j, h.
    Grid: (B, num_chunks). Each program computes the full G for one (b, i) over j in [0..CHUNK_SIZE-1] and h in [0..NUM_HEADS-1].
    We loop over k, s, and write to Gout with constexpr bounds to avoid dynamic indexing.
    """
    # We will compute G for all j, h across k and s, and store in Gout.
    # Note: Gout is a 5D tensor. Triton supports vectorized operations; we avoid dynamic indexing by using loops.
    # Outer: loop over k
    for k in range(CHUNK_SIZE):
        # For each j, compute G[j, :] vector over h, and store into Gout[b, i, k, j, :]
        for j in range(CHUNK_SIZE):
            # Accumulator for G[j, :]
            G_vec = tl.zeros((NUM_HEADS,), dtype=tl.float32)
            # Sum over state dimension in tiles of 16
            for s0 in range(0, HEAD_DIM, 16):
                offs = s0 + tl.arange(0, 16)
                mask = offs < HEAD_DIM
                acc = tl.zeros((16,), dtype=tl.float32)
                # For each h_off (head index), compute dot over s lanes and accumulate
                for h_off in range(NUM_HEADS):
                    # B[b, j, k, h_off, s] as vector over s
                    B_ptrs = B_ptr + b_id * B_bs + i_id * B_cs + j * B_cd + h_off * B_w + k * B_sd + offs
                    B_vals = tl.load(B_ptrs, mask=mask, other=0.0)  # shape [16]
                    # C[b, i, k, h_off, s]
                    C_ptrs = C_ptr + b_id * C_bs + i_id * C_cs + k * C_cd + h_off * C_w + offs
                    C_vals = tl.load(C_ptrs, mask=mask, other=0.0)  # shape [16]
                    acc += C_vals * B_vals
                # Accumulate acc into G_vec
                G_vec += acc  # acc is 16-lane; add to G_vec components
            # Store G_vec into Gout[b, i, k, j, :]
            for h_off in range(NUM_HEADS):
                G_ptrs = Gout_ptr + b_id * G_bs + i_id * G_cs + k * G_cd + j * G_j + h_off * G_h
                # We computed G_vec over h_off positions; store each lane
                # Note: G_vec is a tensor across h_off; here we access lanes via offsets.
                # However, Triton does not support indirect indexing of vector lanes in this pattern reliably.
                # Therefore, we store acc[h_off] per iteration; but acc is over s, not h.
                # Instead, we store the entire vector by writing each element:
                # Since we cannot index vector lanes, we instead store scalar by recomputing for h_off.
                # Compute scalar contribution for this h_off:
                # Reuse acc[h_off] computed above by summing acc where offs == h_off is invalid.
                # Better: recompute per h_off:
                per_h = tl.zeros((), dtype=tl.float32)
                for s0 in range(0, HEAD_DIM, 16):
                    offs = s0 + tl.arange(0, 16)
                    mask = offs < HEAD_DIM
                    acc = tl.zeros((16,), dtype=tl.float32)
                    for kk in range(CHUNK_SIZE):
                        B_ptrs = B_ptr + b_id * B_bs + i_id * B_cs + j * B_cd + h_off * B_w + kk * B_sd + offs
                        B_vals = tl.load(B_ptrs, mask=mask, other=0.0)
                        C_ptrs = C_ptr + b_id * C_bs + i_id * C_cs + kk * C_cd + h_off * C_w + offs
                        C_vals = tl.load(C_ptrs, mask=mask, other=0.0)
                        acc += C_vals * B_vals
                    # Extract scalar per_h for this h_off:
                    # Triton doesn't support arbitrary element extraction from vector; workaround:
                    # Since we loop h_off last, we can recompute per_h directly without storing vectors.
                    per_h = 0.0
                    for s in range(HEAD_DIM):
                        # Manual scalar accumulation
                        B_ptr_s = B_ptr + b_id * B_bs + i_id * B_cs + j * B_cd + h_off * B_w + k * B_sd + s
                        C_ptr_s = C_ptr + b_id * C_bs + i_id * C_cs + k * C_cd + h_off * C_w + s
                        B_s = tl.load(B_ptr_s)
                        C_s = tl.load(C_ptr_s)
                        per_h += C_s * B_s
                tl.store(G_ptrs, per_h)
    # The above nested loops with tl.constexpr bounds are the safest way to avoid dynamic indexing and runtime errors.
    # Despite complexity, Triton will unroll them because CHUNK_SIZE, NUM_HEADS, HEAD_DIM are constexpr.

@triton.jit
def contract_Y_triton(
    Gout_ptr, hidden_ptr, Y_ptr,
    G_bs, G_cs, G_cd, G_w, G_j, G_h,  # Gout strides
    hidden_bs, hidden_cs, hidden_cd, hidden_w, hidden_d,  # hidden strides: (B, num_chunks, CHUNK_SIZE, NUM_HEADS, HEAD_DIM)
    Y_bs, Y_cs, Y_k, Y_h, Y_d,  # Y strides: (B, num_chunks, CHUNK_SIZE, NUM_HEADS, HEAD_DIM)
    b_id: tl.constexpr,
    i_id: tl.constexpr,
    k_id: tl.constexpr,
    h_id: tl.constexpr,
    d_id: tl.constexpr,
):
    """
    Compute Y[b, i, k, h, d] = sum_j sum_s Gout[b, i, k, j, h] * hidden[b, i, j, h, d]
    Grid: (B, num_chunks, CHUNK_SIZE, NUM_HEADS) covers all (b, i, k, h) slices; d is handled via constexpr d_id.
    We loop over j and s with tl.constexpr bounds and store Y for the fixed d_id.
    """
    acc = 0.0
    # Loop over j (chunk position)
    for j in range(CHUNK_SIZE):
        # Sum over s (state dimension) in tiles of 16
        for s0 in range(0, HEAD_DIM, 16):
            offs = s0 + tl.arange(0, 16)
            mask = offs < HEAD_DIM
            # Load Gout[b, i, k, j, h]
            G_ptrs = Gout_ptr + b_id * G_bs + i_id * G_cs + k_id * G_cd + j * G_j + h_id * G_h
            G_vals = tl.load(G_ptrs, mask=mask, other=0.0)  # shape [16]
            # Load hidden[b, i, j, h, d] for d_id (single scalar)
            hidden_ptrs = hidden_ptr + b_id * hidden_bs + i_id * hidden_cs + j * hidden_cd + h_id * hidden_w + d_id * hidden_d
            hidden_val = tl.load(hidden_ptrs)  # scalar
            acc += tl.sum(G_vals * hidden_val, axis=0)
    # Store Y[b, i, k, h, d]
    Y_ptrs = Y_ptr + b_id * Y_bs + i_id * Y_cs + k_id * Y_k + h_id * Y_h + d_id * Y_d
    tl.store(Y_ptrs, acc)

class ModelNew(torch.nn.Module):
    def forward(self, hidden_states, A_cumsum, B, C):
        """
        Compute Y_diag per the original logic, launching Triton kernels for core computations.
        Returns tensor of shape [B, num_chunks, CHUNK_SIZE, NUM_HEADS, HEAD_DIM] in bfloat16.
        """
        # Ensure inputs are contiguous float32
        B = B.contiguous().to(torch.float32)
        C = C.contiguous().to(torch.float32)
        hidden = hidden_states.contiguous().to(torch.float32)

        B_expanded = B.repeat_interleave(NUM_HEADS // N_GROUPS, dim=3)  # [B, num_chunks, CHUNK_SIZE, NUM_HEADS, HEAD_DIM]
        C_expanded = C.repeat_interleave(NUM_HEADS // N_GROUPS, dim=3)

        # Allocate Gout for partial G (float32): [B, num_chunks, CHUNK_SIZE, CHUNK_SIZE, NUM_HEADS]
        Gout = torch.empty((B.shape[0], B.shape[1], CHUNK_SIZE, CHUNK_SIZE, NUM_HEADS), dtype=torch.float32, device=B.device)

        # Launch Triton kernel to compute Gout (no dynamic indexing into 5D tensors)
        grid = (B.shape[0], B.shape[1])
        compute_G_partial_triton[grid](
            B_expanded, C_expanded, Gout,
            B_expanded.stride(0), B_expanded.stride(1), B_expanded.stride(2), B_expanded.stride(3), B_expanded.stride(4),
            C_expanded.stride(0), C_expanded.stride(1), C_expanded.stride(2), C_expanded.stride(3), C_expanded.stride(4),
            Gout.stride(0), Gout.stride(1), Gout.stride(2), Gout.stride(3), Gout.stride(4),  # dummy j and h strides
            0, 0  # b_id, i_id
        )

        # Allocate Y: [B, num_chunks, CHUNK_SIZE, NUM_HEADS, HEAD_DIM] float32
        Y = torch.empty((B.shape[0], B.shape[1], CHUNK_SIZE, NUM_HEADS, HEAD_DIM), dtype=torch.float32, device=B.device)

        # Launch Triton kernel to contract Gout with hidden and produce Y
        grid2 = (B.shape[0], B.shape[1], CHUNK_SIZE, NUM_HEADS)
        for d_id in range(HEAD_DIM):
            contract_Y_triton[grid2](
                Gout, hidden, Y,
                Gout.stride(0), Gout.stride(1), Gout.stride(2), Gout.stride(3), Gout.stride(4),
                hidden.stride(0), hidden.stride(1), hidden.stride(2), hidden.stride(3), hidden.stride(4),
                Y.stride(0), Y.stride(1), Y.stride(2), Y.stride(3), Y.stride(4),
                0, 0, 0, 0, d_id  # b_id, i_id, k_id, h_id, d_id
            )

        # Return Y cast to bfloat16 to match original interface expectations
        return Y.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
