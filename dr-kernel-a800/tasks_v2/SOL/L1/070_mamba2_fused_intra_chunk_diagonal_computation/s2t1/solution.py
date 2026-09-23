import torch
import triton
import triton.language as tl

# Constants matching the original model
CHUNK_SIZE = 128
NUM_HEADS = 32
N_GROUPS = 8
HEAD_DIM = 64  # In the original, hidden_states has head_dim=64; we use that here

# Triton kernels for contractions and elementwise ops go here.
# We will create a kernel that computes Y_diag = sum_j (M[..., j, :] * hidden[..., j, :])
# M is provided as G * L, where G and L are precomputed with torch on host (this is allowed).

# Kernel: contract M and hidden along chunk_size_j and reduce into Y_diag
@triton.jit
def contract_m_hidden_kernel(
    M_ptr, hidden_ptr, out_ptr,
    # M shape: [B, num_chunks, CHUNK_SIZE, CHUNK_SIZE, NUM_HEADS]
    # hidden shape: [B, num_chunks, CHUNK_SIZE, NUM_HEADS, HEAD_DIM]
    # out shape: [B, num_chunks, CHUNK_SIZE, NUM_HEADS, HEAD_DIM]
    # Strides:
    M_b_stride, M_c_stride, M_i_stride, M_j_stride, M_h_stride,
    hidden_b_stride, hidden_c_stride, hidden_j_stride, hidden_h_stride, hidden_hd_stride,
    out_b_stride, out_c_stride, out_j_stride, out_h_stride, out_hd_stride,
    B: tl.constexpr, C: tl.constexpr, CHUNK: tl.constexpr, HEADS: tl.constexpr, HD: tl.constexpr,
    BLOCK_K: tl.constexpr,  # tile size over CHUNK_SIZE dimension (j)
):
    """
    Each program handles a block of j (chunk_size) for a fixed (b, chunk_i, head_block).
    We loop over j and for each j, we multiply M[b, chunk_i, k, j, head] with hidden[b, chunk_i, j, head, h]
    and accumulate into out[b, chunk_i, k, head, h].
    """
    # Grid is (B, C, ceil(CHUNK/BLOCK_K), ceil(HEADS/BLOCK_H))
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_j_blk = tl.program_id(2)
    pid_h_blk = tl.program_id(3)

    # Offsets
    j_off = pid_j_blk * BLOCK_K + tl.arange(0, BLOCK_K)  # vector of j indices
    h_off = pid_h_blk * BLOCK_K + tl.arange(0, BLOCK_K)  # vector of head indices

    # Masks
    j_mask = j_off < CHUNK
    h_mask = h_off < HEADS

    # Initialize accumulator for this (b, chunk_i, head_block) across k and h
    # We'll accumulate into a [BLOCK_K, HEADS] float32 tensor
    acc = tl.zeros((BLOCK_K, HEADS), dtype=tl.float32)

    # Loop over k (chunk_size) dimension explicitly by scalar since we need to store per k
    # Note: We cannot vectorize over k easily in Triton without more elaborate kernel; we will iterate k.
    # However, Triton kernels are best when vectors are used; here we implement per-k computation.
    # To keep Triton happy, we'll unroll k loop up to CHUNK_SIZE (small).
    for k in range(CHUNK):
        # For each k, load M[b, chunk_i, k, j, h] and hidden[b, chunk_i, j, h, h]
        # Build address vectors for M and hidden
        M_offsets = (
            pid_b * M_b_stride
            + pid_c * M_c_stride
            + k * M_i_stride
            + j_off * M_j_stride
            + h_off * M_h_stride
        )
        hidden_offsets = (
            pid_b * hidden_b_stride
            + pid_c * hidden_c_stride
            + j_off * hidden_j_stride
            + h_off * hidden_h_stride
            + 0 * hidden_hd_stride  # we'll vary h via h_off
        )

        # Load M and hidden with masks
        M_vals = tl.load(M_ptr + M_offsets, mask=j_mask[:, None] & h_mask[None, :], other=0.0)
        hidden_vals = tl.load(hidden_ptr + hidden_offsets, mask=j_mask[:, None] & h_mask[None, :], other=0.0)

        # acc[:, :] += M_vals * hidden_vals
        acc += M_vals * hidden_vals

    # Now store acc to out[b, chunk_i, k, h, 0..HEAD_DIM-1]
    # We'll store per h and per k. We can loop over k again or store acc as out for h=h_off, k=j_off.
    # We store acc into out for each k (k is known implicitly per iteration above). Triton requires explicit iteration.
    # Instead, we can store acc into out by building out offsets for each k and writing acc[:, h_off].
    # We'll compute out offsets and store acc[:, h_off] for each k.

    # For each k in CHUNK (we already computed acc), we write acc[:, h_off] into out
    # out_offsets_k = pid_b*out_b_stride + pid_c*out_c_stride + k*out_j_stride + h_off*out_h_stride + 0*out_hd_stride
    for k in range(CHUNK):
        out_offsets_k = (
            pid_b * out_b_stride
            + pid_c * out_c_stride
            + k * out_j_stride
            + h_off * out_h_stride
            + 0 * out_hd_stride
        )
        # Store acc[:, h_off] into out
        # acc[:, h_off] shape: [BLOCK_K, HEADS] but we only want to store columns h_off. We can select via h_mask.
        # Triton supports storing with mask and broadcasting along h axis.
        tl.store(out_ptr + out_offsets_k, acc[:, h_off], mask=h_mask[None, :])

# For simplicity and correctness, we will provide a ModelNew that:
# 1) Computes L on host using torch (allowed), with the exact original semantics: exp(cumsum with lower-tri mask diagonal=-1)
# 2) Computes G on host using torch (allowed): G = sum over state_dim of C * B expanded across heads
# 3) Creates M = G * L
# 4) Launches Triton kernel to compute Y_diag = sum_j (M[..., j, :] * hidden[..., j, :]).

class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                A_cumsum: torch.Tensor,
                B: torch.Tensor,
                C: torch.Tensor) -> torch.Tensor:
        """
        Triton-optimized forward that:
        - Computes L using torch (host) with exact original semantics: causal mask with tril(diagonal=-1) on cumsum(exp).
        - Computes G using torch (host): contraction of B and C expanded to heads.
        - Produces Y_diag using Triton contraction kernel.
        Note: The Triton kernels do not perform torch elementwise ops; host uses torch only for building L and G and for final casting.
        """
        # Ensure CUDA tensors and contiguity
        assert hidden_states.is_cuda and A_cumsum.is_cuda and B.is_cuda and C.is_cuda, "All tensors must be CUDA tensors."
        hidden_states = hidden_states.contiguous()
        A_cumsum = A_cumsum.contiguous()
        B = B.contiguous()
        C = C.contiguous()

        # Shapes
        B_bs, B_cs, B_cd, B_hs, B_hd = hidden_states.shape  # [B, num_chunks, CHUNK_SIZE, NUM_HEADS, HEAD_DIM]
        A_bs, A_hs, A_cd, A_cs = A_cumsum.shape              # [B, NUM_HEADS, num_chunks, CHUNK_SIZE]
        B2_bs, B2_cs, B2_cd, B2_gs, B2_sd = B.shape          # [B, num_chunks, CHUNK_SIZE, N_GROUPS, state_size]
        C2_bs, C2_cs, C2_cd, C2_gs, C2_sd = C.shape          # [B, num_chunks, CHUNK_SIZE, N_GROUPS, state_size]
        assert B_bs == A_bs == C2_bs == B2_bs, "Batch sizes must match."
        assert B2_cs == C2_cs, "num_chunks must match for B and C."
        assert B2_cd == CHUNK_SIZE, "B/C chunk_size must equal CHUNK_SIZE."
        assert B2_gs == N_GROUPS and C2_gs == N_GROUPS, "N_GROUPS must match."
        assert B2_sd == C2_sd, "State size must match for B and C."
        assert B_hs == NUM_HEADS, "NUM_HEADS must match hidden heads."
        assert B_hd == HEAD_DIM, "HEAD_DIM must match hidden head_dim."

        # 1) Compute L on host using torch: L = exp(cumsum with lower-tri mask diagonal=-1), shape: [B, NUM_HEADS, num_chunks, CHUNK_SIZE, CHUNK_SIZE]
        # Initialize L
        L = torch.zeros(
            (B_bs, NUM_HEADS, B_cs, CHUNK_SIZE, CHUNK_SIZE),
            dtype=torch.float32,
            device=hidden_states.device,
        )

        # Build lower-triangular mask (diagonal = -1) on the fly for each (b, chunk_i)
        for b in range(B_bs):
            for h in range(NUM_HEADS):
                for chunk_i in range(B_cs):
                    # A_cumsum[b, h, chunk_i, :] is the vector we need
                    vec = A_cumsum[b, h, chunk_i, :]  # [CHUNK_SIZE]
                    # Create index vector
                    j = torch.arange(CHUNK_SIZE, device=vec.device)
                    # Lower-triangular mask (diagonal=-1): i >= j means i == chunk_i >= j
                    mask = (j >= 0) & (j <= chunk_i)
                    # Cumsum up to j for each element: sum_{t<=j} A[b,h,chunk_i,t]
                    # We need to apply mask: for j > chunk_i, set contribution to 0. But our mask above only covers j<=chunk_i.
                    # For j > chunk_i, cumsum should be 0. Implement by zeroing elements where mask is False.
                    cumsum = torch.cumsum(vec, dim=0)
                    cumsum = torch.where(mask, cumsum, torch.zeros_like(cumsum))
                    L[b, h, chunk_i, torch.arange(CHUNK_SIZE), :] = torch.exp(cumsum)  # L[i>=j -> exp(cumsum), else 0]

        # 2) Compute G on host using torch: G[i,j] = sum_s C[i,s] * B[j,s], shape: [B, num_chunks, CHUNK_SIZE, CHUNK_SIZE, NUM_HEADS]
        # B needs to be expanded from N_GROUPS to NUM_HEADS as in original (repeat_interleave). However, original repeats across chunks, not heads; it repeats from N_GROUPS to NUM_HEADS. We need to match that: original does B = B.repeat_interleave(NUM_HEADS // N_GROUPS, dim=3), same for C.
        repeat_factor = NUM_HEADS // N_GROUPS
        B_exp = B.repeat_interleave(repeat_factor, dim=3)  # [B, num_chunks, CHUNK_SIZE, NUM_HEADS, state_size]
        C_exp = C.repeat_interleave(repeat_factor, dim=3)  # [B, num_chunks, CHUNK_SIZE, NUM_HEADS, state_size]

        # G: [B, num_chunks, CHUNK_SIZE, CHUNK_SIZE, NUM_HEADS]
        # G[i,j] = sum over state_size (B_sd) of C_exp[i,k,s] * B_exp[j,k,s]
        G = torch.einsum('bksw,bksw->bkkw', C_exp, B_exp)

        # 3) M = G * L
        M = G * L  # broadcasting: L has trailing dims (CHUNK_SIZE, CHUNK_SIZE, NUM_HEADS); M has (CHUNK_SIZE, CHUNK_SIZE, NUM_HEADS); einsum produces [B, num_chunks, CHUNK_SIZE, CHUNK_SIZE, NUM_HEADS]

        # 4) Compute Y_diag using Triton: Y_diag[b, chunk_i, k, head, h] = sum_j (M[b, chunk_i, k, j, head] * hidden[b, chunk_i, j, head, h])
        # Allocate output tensor (float32 for numerical stability)
        out = torch.empty((B_bs, B_cs, CHUNK_SIZE, NUM_HEADS, HEAD_DIM), dtype=torch.float32, device=hidden_states.device)

        # Grid: (B, num_chunks, ceil(CHUNK_SIZE/BLOCK_K), ceil(NUM_HEADS/BLOCK_H))
        BLOCK_K = 64  # tile size over CHUNK_SIZE
        BLOCK_H = 64  # tile size over NUM_HEADS
        grid = (B_bs, B_cs, triton.cdiv(CHUNK_SIZE, BLOCK_K), triton.cdiv(NUM_HEADS, BLOCK_H))

        # Launch Triton kernel
        contract_m_hidden_kernel[grid](
            M, hidden_states, out,
            M.stride(0), M.stride(1), M.stride(2), M.stride(3), M.stride(4),
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2), hidden_states.stride(3), hidden_states.stride(4),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3), out.stride(4),
            B_bs, B_cs, CHUNK_SIZE, NUM_HEADS, HEAD_DIM,
            BLOCK_K=BLOCK_K, HEADS=NUM_HEADS, HD=HEAD_DIM,
            num_warps=4,
        )

        # Cast to bfloat16 to match original's output dtype
        out_bf16 = out.to(torch.bfloat16)
        return out_bf16


def run(*args):
    return ModelNew()(*args)
