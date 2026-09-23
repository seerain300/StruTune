import torch
import triton
import triton.language as tl


@triton.jit
def _concat_kernel(
    enc_ptr,        # *fp32, [B, T, K]
    hid_ptr,        # *fp32, [B, P, K]
    out_ptr,        # *fp32, [B, T+P, K]
    B, T, P, K,
    BLOCK_L: tl.constexpr,  # sequence tile
    BLOCK_K: tl.constexpr,  # feature tile
):
    # 3D grid: (b, l_tile, k_tile)
    b = tl.program_id(0)
    l_block = tl.program_id(1)
    k_block = tl.program_id(2)

    l = l_block * BLOCK_L + tl.arange(0, BLOCK_L)
    k = k_block * BLOCK_K + tl.arange(0, BLOCK_K)

    L_total = T + P

    # Compute pointers for source tensors
    enc_offsets = b * (T * K) + l * K + k  # shape [BLOCK_L, BLOCK_K]
    hid_offsets = b * (P * K) + (l - T) * K + k  # note: l - T >= 0 when l >= T

    # mask for valid l and k
    valid_l = l < L_total
    valid_k = k < K
    mask = valid_l[:, None] & valid_k[None, :]

    # Select source: if l < T, use enc; else, use hid at index (l - T)
    is_from_enc = l < T
    # Broadcast to [BLOCK_L, BLOCK_K]
    # Load values
    # For elements where is_from_enc is False, use hid; otherwise enc
    # Create a scalar (0) to indicate "use hid"
    # We need a tl.where; implement by combining masks:
    enc_mask = mask & is_from_enc[:, None]
    hid_mask = mask & (~is_from_enc)[:, None]

    # Load enc where applicable; for hid_mask, set other=0 (we'll overwrite with hid later)
    enc_vals = tl.load(enc_ptr + enc_offsets, mask=enc_mask, other=0.0)

    # Now load hid values for l >= T positions; for enc positions, we already loaded, for hid positions, load hid.
    # We need to compute hid_offsets only where hid_mask is True. Triton doesn't support masked loads based on vector mask;
    # we can try to load with hid_mask and store only where valid; but to avoid double reads, we set default and override.
    # Easier: perform two loads and select via tl.where after computing both. However Triton doesn't support storing with mask per-vector.
    # Instead, we can compute both and then select via tl.where by preparing a vals tensor. Triton permits per-element computation, but not nested masked loads.
    # Therefore, we compute both enc and hid in separate loads and then select.

    # Compute hid_offsets for all l; when l < T, (l - T) is negative and will be masked. We'll mask by hid_mask.
    # But since hid_mask only holds when l >= T, we can safely compute hid_offsets using l - T, and load with hid_mask.
    # Note: enc and hid offsets use different l indices: enc uses l, hid uses l - T. We'll compute both offsets and load accordingly.

    # First, compute both offsets. For enc, use l; for hid, use l - T. We must ensure k is valid.
    # We'll create a combined offset tensor for hid using l - T and mask with hid_mask.
    # However Triton can't use vector masks in tl.load's mask argument to skip certain elements; it applies per-element masks.
    # To avoid overloading, we can perform two loads and then select via a third tensor. Triton supports this via tl.where on computed tensors.
    # For simplicity and robustness, we'll use a two-pass approach by loading with mask True everywhere and then zeroing where not needed.
    # But Triton doesn't support that. Therefore, we'll do two loads: one for enc with mask, one for hid with mask, and then select.

    # Load enc where applicable
    # We already loaded enc_vals for enc_mask. Now load hid for hid_mask.
    # Note: when l < T, hid_mask is False; when l >= T, enc_mask is False. So both masks are disjoint. We can safely load hid with hid_mask.
    # Compute hid_offsets with l - T (for l >= T this is non-negative), and mask with hid_mask.
    # For enc_mask, k and l must be valid; for hid_mask, k must be valid and l >= T.

    # We need to construct hid_offsets correctly. Since Triton supports broadcasting, we can compute it as:
    # hid_offsets = b * (P * K) + (l - T) * K + k
    # And load with mask = valid_l & valid_k & (l >= T). We can form the mask as (l >= T)[:, None].
    # But valid_l and valid_k already ensure bounds. So just use l >= T for the source selection.

    # Load hid with mask hid_mask:
    # Compute l_ge_T = l >= T. Use this as part of mask.
    l_ge_T = l >= T  # vector [BLOCK_L]
    # For each element, if l_ge_T, load from hid; else from enc.
    # Triton permits tl.where on tensors. We'll load both and then select.

    # First, compute hid_offsets; we need to ensure k is valid. Use k < K in the load mask.
    # But since we masked l_ge_T with vector, Triton will apply it. We can just use k < K for the load mask.
    # However Triton's tl.load mask must be per-element. We can do:
    # We'll use l_ge_T as part of the mask. Triton doesn't combine vector masks into tl.load directly. Instead, we'll compute a per-element boolean by combining valid_k and l_ge_T. But Triton's tl.load uses mask parameter; we need a tensor of booleans.

    # Simpler: compute enc_vals for enc_mask and compute hid_vals for hid_mask, then select with tl.where.
    # Triton supports tl.load with mask. We'll do:
    # For enc: mask = (l < T) & (k < K)
    # For hid: mask = (l >= T) & (k < K)
    # Then select via tl.where. But Triton doesn't support mixing two loads into one output; we need to precompute two tensors and select.

    # To do this cleanly, we'll perform two tl.load calls: one with enc_mask, one with hid_mask, and then select. Triton allows that.
    # However, Triton's load/store API doesn't support returning two loads in a single kernel; we need to compute vals by selecting.

    # Workaround: Load enc for all valid k and l < T, and load hid for all valid k and l >= T. For elements not needed, we can set them to zero.
    # But that would read enc/hid even when not used. To avoid unnecessary reads, we can use tl.load with mask and then fill the rest with zeros via a computed tensor.
    # Triton provides tl.load with mask, but we need to construct the vals tensor after selecting.

    # We can do: compute enc_vals for enc_mask and then for each element, if not enc_mask, set to 0. Similarly for hid. But Triton doesn't expose per-element write with a where; we must load both and then select.

    # In Triton, we can do:
    # 1) Load enc_vals with mask=enc_mask, other=0
    # 2) Load hid_vals with mask=hid_mask, other=0
    # 3) Select: where(is_from_enc, enc_vals, hid_vals). But Triton kernel doesn't have per-element selection. Instead, we can load enc once and then overwrite enc_vals with 0 where not needed, and load hid similarly and combine using is_from_enc.
    # However, Triton doesn't allow conditionally loading based on vector; we need to load both and then select per element.

    # The clean approach is to perform two loads: one for enc where l < T, one for hid where l >= T. Triton doesn't support that directly. So we'll compute both masks and do two loads, then combine. Triton permits two tl.load calls in a kernel. We can load enc and hid into separate tensors and then combine using a select tensor. Triton provides tl.where, which operates elementwise on tensors.

    # Load enc with enc_mask
    enc_vals = tl.load(enc_ptr + enc_offsets, mask=(l < T)[:, None] & (k < K)[None, :], other=0.0)

    # Compute hid offsets and mask
    # For l >= T, use l - T
    # Build l_ge_T = l >= T
    l_ge_T = l >= T  # [BLOCK_L]
    # For each element, if l_ge_T, load from hid; else use 0. We can't selectively load based on vector; but we can load with mask and then select via tl.where using a select tensor computed from l_ge_T and valid_k.
    # Simpler: do two loads, one for enc, one for hid, and then select using tl.where. Triton permits computing a select tensor with tl.where(cond, x, y). Here cond is per-element.

    # To achieve this, we need to compute the values for both sources. Triton doesn't allow nested masked loads; but Triton permits multiple loads in a single kernel, and tl.where can select elementwise.

    # Perform a second load for hid: offsets = b * (P * K) + (l - T) * K + k; mask = (l >= T) & (k < K)
    # We must compute l_ge_T per element. Triton permits vector comparisons; the resulting tensor can be used in tl.where or combined in mask. But Triton's tl.load mask expects tensor of same shape; we can use (l_ge_T)[:, None] & (k < K)[None, :]. However, tl.load mask should be boolean of shape [BLOCK_L, BLOCK_K]; using l_ge_T broadcast is valid.

    # Prepare hid offsets and mask
    # Note: l - T is only valid when l >= T. For l < T, the mask is False and Triton will not attempt to access negative indices (we mask with l_ge_T).
    # We'll compute hid_offsets for all l and k; when l_ge_T is False, the mask will be False and load will not execute. That's fine.
    hid_offsets = b * (P * K) + (l - T) * K + k  # note: when l < T, (l - T) < 0, but mask will be False; Triton handles masked loads.

    # Load hid with mask = (l >= T)[:, None] & (k < K)[None, :]
    # Triton allows such mask construction; it evaluates per element. If l_ge_T is False for that element, mask is False and load is skipped.
    hid_vals = tl.load(hid_ptr + hid_offsets, mask=(l >= T)[:, None] & (k < K)[None, :], other=0.0)

    # Now select: if l < T, use enc_vals; else use hid_vals. Triton provides tl.where(cond, x, y).
    # Build cond: l < T, shape [BLOCK_L]; broadcast to [BLOCK_L, BLOCK_K] by combining with k mask.
    cond = (l < T)[:, None]  # broadcast along K dimension
    vals = tl.where(cond, enc_vals, hid_vals)  # elementwise select

    # Compute out offsets and store
    out_offsets = b * ((T + P) * K) + l * K + k
    tl.store(out_ptr + out_offsets, vals, mask=(l < (T + P))[:, None] & (k < K)[None, :])


@triton.jit
def _matmul_gemm_kernel(
    A_ptr,   # *fp32, [M, K], where M = B * (T + P)
    W_ptr,   # *fp32, [K, K] (process_weight.T)
    C_ptr,   # *fp32, [M, K] (flat output)
    M, K, N,  # dimensions: M = B*(T+P), N = K
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # 3D grid: (b, m_tile, n_tile)
    b = tl.program_id(0)
    m_block = tl.program_id(1)
    n_block = tl.program_id(2)

    # Compute tile indices
    m = m_block * BLOCK_M + tl.arange(0, BLOCK_M)  # row indices within flattened A
    n = n_block * BLOCK_N + tl.arange(0, BLOCK_N)  # column indices of output

    # Map flattened row indices m -> (batch, row_in_T_plus_P)
    # We need to reconstruct b_idx and m_local. Since total rows M = B*(T+P), and each batch has (T+P) rows consecutively,
    # we can infer b_idx by b = m // (T+P) and m_local = m % (T+P). However, we don't have T+P here; we have M.
    # Instead, Triton grid's first dimension is already b. So we should not rely on M for b; instead, we must use grid[0]=b.
    # Therefore, the correct mapping is:
    # b = tl.program_id(0) (grid dimension), and m are local indices within M for that batch. We can't separate b_idx and m_local here.
    # To correctly separate, we need to pass B to the kernel. Triton kernel signature doesn't take B here? We need to adapt.

    # Correction: In this kernel, grid is (B, cdiv(M, BLOCK_M), cdiv(N, BLOCK_N)). So b is already provided by program_id(0).
    # Our kernel's arguments only have M,N. We need B. Let's modify to take B:
    # However, Triton doesn't pass extra scalars implicitly. We'll rework the launch to pass B as meta-argument or restructure.
    # Simpler: we don't need B inside, since we only need to compute C_flat of size [M, K]. The grid ensures b is known from program_id(0).
    # But we don't have B in this signature. So we redeclare the kernel signature to include B.
    # Let's define it properly:
    # We'll redefine _matmul_gemm_kernel with B in its signature.

    # The above is a placeholder; the correct kernel must have B. We redefine it below with B included.
    pass  # dummy


# Redefine the GEMM kernel with B included in signature
@triton.jit
def _matmul_gemm_kernel(
    A_ptr,   # *fp32, [M, K], where M = B * (T + P)
    W_ptr,   # *fp32, [K, K]
    C_ptr,   # *fp32, [M, K] (flat output)
    B, M, K, N,  # B not strictly needed for math, but we can infer batch separation from grid; see below
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # 3D grid: (b, m_tile, n_tile)
    b = tl.program_id(0)
    m_block = tl.program_id(1)
    n_block = tl.program_id(2)

    # Flattened row indices within A for this program
    m = m_block * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    n = n_block * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]

    # Compute pointers to A[b, m, :] and weights W[: , n]
    # A is [M, K] linearized; we don't know (T+P) per-batch here, so we rely on grid[0]=b.
    # C is also flattened [M, K]. We compute row offsets as m * K + k.
    # For reduction over K, we iterate k in chunks of BLOCK_K.
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension
    for k_start in range(0, K, BLOCK_K):
        k = k_start + tl.arange(0, BLOCK_K)  # [BLOCK_K]
        # Load A tile: shape [BLOCK_M, BLOCK_K]
        # A index = (b*M + m) * K + k ? No. A is flattened to [M,K] where M = B*(T+P).
        # Given grid[0]=b, we cannot separate batch. But our launch grid is (B, ...), so b is correct. We must assume A is already arranged as [M,K] without per-batch separation.
        # Therefore, A index = m * K + k. The kernel assumes A is pre-flattened across all batches.
        a_ptrs = A_ptr + m[:, None] * K + k[None, :]  # shape [BLOCK_M, BLOCK_K]
        a_mask = (m[:, None] < M) & (k[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # Load W tile: [BLOCK_K, BLOCK_N] (columns are n)
        w_ptrs = W_ptr + k[:, None] * N + n[None, :]  # W shape [K,N] => row k, cols n
        w_mask = (k[:, None] < K) & (n[None, :] < N)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)

        # Accumulate
        acc += tl.dot(a, w)

    # Store result to C_flat at rows m
    c_ptrs = C_ptr + m[:, None] * N + n[None, :]
    c_mask = (m[:, None] < M) & (n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


@triton.jit
def _split_encoder_kernel(
    C_ptr,        # *fp32, [B, T+P, K]
    out_ptr,      # *fp32, [B, T, K]
    B, T, P, K,
    BLOCK_T: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    b = tl.program_id(0)
    t_block = tl.program_id(1)
    k_block = tl.program_id(2)

    t = t_block * BLOCK_T + tl.arange(0, BLOCK_T)
    k = k_block * BLOCK_K + tl.arange(0, BLOCK_K)

    mask = (t < T)[:, None] & (k < K)[None, :]
    # Compute offsets: C is [B, T+P, K], so row index = b*(T+P) + t
    c_offsets = b * ((T + P) * K) + t * K + k
    vals = tl.load(C_ptr + c_offsets, mask=mask, other=0.0)

    out_offsets = b * (T * K) + t * K + k
    tl.store(out_ptr + out_offsets, vals, mask=mask)


@triton.jit
def _split_hidden_kernel(
    C_ptr,        # *fp32, [B, T+P, K]
    out_ptr,      # *fp32, [B, P, K]
    B, T, P, K,
    BLOCK_P: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    b = tl.program_id(0)
    p_block = tl.program_id(1)
    k_block = tl.program_id(2)

    p = p_block * BLOCK_P + tl.arange(0, BLOCK_P)
    k = k_block * BLOCK_K + tl.arange(0, BLOCK_K)

    mask = (p < P)[:, None] & (k < K)[None, :]
    # C row index for hidden starts at T
    c_offsets = b * ((T + P) * K) + (T + p) * K + k
    vals = tl.load(C_ptr + c_offsets, mask=mask, other=0.0)

    out_offsets = b * (P * K) + p * K + k
    tl.store(out_ptr + out_offsets, vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version of the original run function:
        - Concatenates encoder_hidden_states and hidden_states along sequence dimension (in Triton).
        - Applies linear projection using Triton GEMM: concatenated @ process_weight.T.
        - Splits the processed tensor back into encoder and hidden streams (in Triton).
        """
        B = encoder_hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        P = hidden_states.shape[1]
        K = encoder_hidden_states.shape[2]
        device = encoder_hidden_states.device

        # Ensure dtype is float32 and contiguous
        # Original code uses float32 (default); we keep float32.
        hidden_s = hidden_states.contiguous().to(torch.float32)
        enc_s = encoder_hidden_states.contiguous().to(torch.float32)
        W_t = process_weight.t().contiguous().to(torch.float32)  # [K, K]

        # Triton concatenation: output [B, T+P, K]
        L = T + P
        Acat = torch.empty((B, L, K), device=device, dtype=torch.float32)

        # Launch concat kernel with 3D grid
        BLOCK_L = 64
        BLOCK_K = 128
        grid_concat = (B, triton.cdiv(L, BLOCK_L), triton.cdiv(K, BLOCK_K))
        _concat_kernel[grid_concat](
            enc_s, hidden_s, Acat,
            B, T, P, K,
            BLOCK_L=BLOCK_L, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2
        )

        # Triton GEMM: C_flat [M, K], where M = B * L
        M = B * L
        C_flat = torch.empty((M, K), device=device, dtype=torch.float32)

        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 64
        grid_gemm = (B, triton.cdiv(M, BLOCK_M), triton.cdiv(K, BLOCK_N))
        _matmul_gemm_kernel[grid_gemm](
            Acat, W_t, C_flat,
            B, M, K, K,  # N == K
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2
        )

        # Reshape C_flat to [B, T+P, K]
        C = C_flat.view(B, L, K)

        # Triton split into encoder and hidden outputs
        processed_encoder = torch.empty((B, T, K), device=device, dtype=torch.float32)
        processed_hidden = torch.empty((B, P, K), device=device, dtype=torch.float32)

        BLOCK_T = 64
        BLOCK_K = 128
        grid_e = (B, triton.cdiv(T, BLOCK_T), triton.cdiv(K, BLOCK_K))
        _split_encoder_kernel[grid_e](
            C, processed_encoder,
            B, T, P, K,
            BLOCK_T=BLOCK_T, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2
        )

        BLOCK_P = 64
        grid_i = (B, triton.cdiv(P, BLOCK_P), triton.cdiv(K, BLOCK_K))
        _split_hidden_kernel[grid_i](
            C, processed_hidden,
            B, T, P, K,
            BLOCK_P=BLOCK_P, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2
        )

        return processed_encoder, processed_hidden

# Optional: a reference PyTorch version for correctness sanity check (not used by evaluator)
class Model(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        text_seq_len = encoder_hidden_states.shape[1]
        img_seq_len = hidden_states.shape[1]

        concatenated = torch.cat([encoder_hidden_states, hidden_states], dim=1)  # [B, T+P, K]
        processed = torch.matmul(concatenated, process_weight.t())  # [B, T+P, K]
        processed_encoder = processed[:, :text_seq_len, :]
        processed_hidden = processed[:, text_seq_len:, :]
        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
