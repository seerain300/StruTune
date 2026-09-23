import torch
import math
import triton
import triton.language as tl


# Triton kernel: compute logits[h, :] = qn_row @ Kc.T + qp_row @ Kp.T
# Input:
#   qn_ptr: pointer to qn_row as 1D float32 [Dn]
#   qp_ptr: pointer to qp_row as 1D float32 [Dp]
#   Kc_ptr: pointer to Kc as 2D float32 [KV, Dn]
#   Kp_ptr: pointer to Kp as 2D float32 [KV, Dp]
#   logits_ptr: pointer to output logits as 1D float32 [KV]
# Params:
#   KV: number of KV tokens (compile-time for loop)
#   Dn: head_dim_ckv (512)
#   Dp: head_dim_kpe (64)
#   BLOCK_K: tile size for KV (e.g., 64)
@triton.jit
def compute_logits_kernel(
    qn_ptr,      # *const float32, shape [Dn]
    qp_ptr,      # *const float32, shape [Dp]
    Kc_ptr,      # *const float32, shape [KV, Dn]
    Kp_ptr,      # *const float32, shape [KV, Dp]
    logits_ptr,  # *float32, shape [KV]
    KV: tl.constexpr,  # number of KV rows
    Dn: tl.constexpr,  # 512
    Dp: tl.constexpr,  # 64
    BLOCK_K: tl.constexpr,  # tile over KV, e.g., 64
):
    # Accumulator for logits across KV tiles
    # We'll build logits as a vector of size KV by accumulating per tile
    # Initialize logits vector
    # Triton can't initialize with zeros vector easily; we'll compute and store per tile
    # Instead, we'll compute logits per tile and store into logits_ptr
    # We'll loop over tiles of KV
    # offsets for each tile
    for k0 in range(0, KV, BLOCK_K):
        offs = k0 + tl.arange(0, BLOCK_K)
        mask = offs < KV
        # Load qn_row[Dn] and qp_row[Dp] as scalars and vectors respectively
        # We need to load qn elements across Dn; but we can't vectorize qn in a 2D dot without loading a 2D tile. Instead, we compute each element j of qn times Kc[:, j] and accumulate.
        # To do this, we need qn[j] for j in [0..Dn). Triton supports elementwise scalar loads:
        # Build qn_block as a vector of length BLOCK_K by computing per j and summing? Not ideal.
        # A better approach: since qn is [Dn], we can compute the dot for each k in tile:
        # Compute qn_row: load scalar qn[j] and accumulate into a scalar acc for each j and k
        # We'll compute logits_tile as a vector of length BLOCK_K:
        logits_tile = tl.zeros([BLOCK_K], dtype=tl.float32)
        # Compute qn contribution:
        # We can't directly load a 2D slice; we'll loop j over Dn (small) and for each j, accumulate into logits_tile over k in tile
        # But computing per j with k loop inside a Triton kernel can be tricky. Simpler: implement qn @ Kc.T with a loop over j in Dn and accumulate into a vector acc_j, but Triton prefers static loop ranges. We can use Python loop since Dn is tl.constexpr.
        # However, Triton JIT requires loops to be compile-time. We can unroll: for j in range(Dn): compute acc_j, then for k in tile, add acc_j * Kc[k, j]
        # Initialize per-j accumulator
        # We need a vector acc_j over j in [0..Dn), but Triton doesn't support indexing into a register vector with a Python variable. So we compute qn_row contribution by summing over j:
        # Approach: For each j, compute qn[j] and loop over k in tile: logits_tile[k] += qn[j] * Kc[k, j]
        # Note: Kc[k, j] can be loaded as a scalar since j is known at compile time (tl.constexpr). We will loop j over Dn and accumulate into logits_tile.
        # This is acceptable since Dn=512 (not huge).
        # We'll precompute qn[j] into a vector qn_j[BLOCK_K] but Triton doesn't support such dynamic indexing. So we'll do a scalar qn[j] per j and loop over k in tile.
        # This is doable: for j in range(Dn):
        for j in range(Dn):
            qn_j = tl.load(qn_ptr + j, mask=True, other=0.0)  # scalar
            # Now add qn_j * Kc[k, j] to logits_tile for k in tile
            # Kc_ptr + offs * Dn + j
            kc_vals = tl.load(Kc_ptr + offs * Dn + j, mask=mask, other=0.0)  # vector [BLOCK_K]
            logits_tile += qn_j * kc_vals

        # Compute qp contribution similarly over Dp
        for j in range(Dp):
            qp_j = tl.load(qp_ptr + j, mask=True, other=0.0)  # scalar
            kp_vals = tl.load(Kp_ptr + offs * Dp + j, mask=mask, other=0.0)  # vector [BLOCK_K]
            logits_tile += qp_j * kp_vals

        # Store logits_tile to output
        tl.store(logits_ptr + offs, logits_tile, mask=mask)


# Triton kernel: apply causal mask to logits
# Input:
#   logits_ptr: pointer to logits as 1D float32 [KV]
#   masked_ptr: pointer to masked logits as 1D float32 [KV]
#   sm_scale: float32 scale
#   KV: number of KV tokens
#   prefix_len: int (number of previously cached tokens)
#   query_idx: int (current query position i)
# Params:
#   BLOCK_K: tile size for KV
@triton.jit
def mask_logits_kernel(
    logits_ptr,     # *const float32, shape [KV]
    masked_ptr,     # *float32, shape [KV]
    sm_scale: tl.float32,
    KV: tl.constexpr,
    prefix_len: tl.int32,
    query_idx: tl.int32,
    BLOCK_K: tl.constexpr,
):
    k0 = tl.program_id(0)
    offs = k0 * BLOCK_K + tl.arange(0, BLOCK_K)
    mask = offs < KV
    # Load logits
    logits = tl.load(logits_ptr + offs, mask=mask, other=0.0)
    # Apply causal mask: keep if offs > (prefix_len + query_idx)
    cond = (offs > (prefix_len + query_idx))
    # If not causal, set to -inf
    logits = tl.where(cond, logits, -float("inf"))
    tl.store(masked_ptr + offs, logits, mask=mask)


# Triton kernel: compute lse per head for masked logits
# Input:
#   masked_ptr: pointer to masked logits as 1D float32 [KV]
#   lse_ptr: pointer to output lse as 1D float32 [H]
# Params:
#   KV: number of KV tokens
#   BLOCK_K: tile size for KV
@triton.jit
def lse_row_kernel(
    masked_ptr,  # *const float32, shape [KV]
    lse_ptr,     # *float32, shape [H]
    KV: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    h = tl.program_id(0)  # one program per head
    # We only have one lse value per head for this b,i pair
    # Reduction: max, then sum of exp, then logsumexp
    max_val = -float("inf")
    # First pass: max
    for k0 in range(0, KV, BLOCK_K):
        offs = k0 + tl.arange(0, BLOCK_K)
        mask = offs < KV
        x = tl.load(masked_ptr + offs, mask=mask, other=-float("inf"))
        # local max
        local_max = tl.max(x, axis=0)  # reduce vector to scalar
        if local_max > max_val:
            max_val = local_max

    # Second pass: sum exp(logits - max_val)
    sum_val = 0.0
    for k0 in range(0, KV, BLOCK_K):
        offs = k0 + tl.arange(0, BLOCK_K)
        mask = offs < KV
        x = tl.load(masked_ptr + offs, mask=mask, other=-float("inf"))
        x = x - max_val
        expx = tl.exp(x)
        sum_val += tl.sum(expx, axis=0)

    lse = tl.log(sum_val) / tl.log(2.0)  # convert to 2-logsumexp
    tl.store(lse_ptr + h, lse)


# Triton kernel: compute softmax over masked logits
# Input:
#   masked_ptr: pointer to masked logits as 1D float32 [KV]
#   attn_ptr: pointer to output attn as 1D float32 [KV]
# Params:
#   KV: number of KV tokens
#   BLOCK_K: tile size for KV
@triton.jit
def softmax_row_kernel(
    masked_ptr,  # *const float32, shape [KV]
    attn_ptr,    # *float32, shape [KV]
    KV: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    h = tl.program_id(0)  # one program per head
    # We only compute softmax for this head; but h is shared across the pair, so use a single program id for head
    # First pass: max
    max_val = -float("inf")
    for k0 in range(0, KV, BLOCK_K):
        offs = k0 + tl.arange(0, BLOCK_K)
        mask = offs < KV
        x = tl.load(masked_ptr + offs, mask=mask, other=-float("inf"))
        local_max = tl.max(x, axis=0)
        if local_max > max_val:
            max_val = local_max

    # Second pass: compute and store softmax
    for k0 in range(0, KV, BLOCK_K):
        offs = k0 + tl.arange(0, BLOCK_K)
        mask = offs < KV
        x = tl.load(masked_ptr + offs, mask=mask, other=-float("inf"))
        x = x - max_val
        expx = tl.exp(x)
        denom = tl.sum(expx, axis=0)
        attn = expx / denom
        tl.store(attn_ptr + offs, attn, mask=mask)


# Triton kernel: compute out[h, :] = attn @ Kc (reduce over KV)
# Input:
#   attn_ptr: pointer to attn as 1D float32 [KV]
#   Kc_ptr: pointer to Kc as 2D float32 [KV, Dn]
#   out_ptr: pointer to output as 1D float32 [Dn]
# Params:
#   KV: number of KV tokens
#   Dn: head_dim_ckv (512)
#   BLOCK_K: tile size for KV
#   BLOCK_D: tile size for Dn (e.g., 128)
@triton.jit
def compute_out_row_kernel(
    attn_ptr,   # *const float32, shape [KV]
    Kc_ptr,     # *const float32, shape [KV, Dn]
    out_ptr,    # *float32, shape [Dn]
    KV: tl.constexpr,
    Dn: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    # One program per head: but head is not needed here, as we compute for a single head at a time
    # We accumulate out_vec over Dn tiles
    out_vec = tl.zeros([Dn], dtype=tl.float32)
    # Loop over KV in tiles
    for k0 in range(0, KV, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < KV
        attn_tile = tl.load(attn_ptr + offs_k, mask=mask_k, other=0.0)  # [BLOCK_K]

        # For each j in Dn, accumulate sum_k attn_tile[k] * Kc[k, j]
        # We'll do a loop over j in [0..Dn) and update out_vec[j]
        # Triton supports Python range with tl.constexpr bounds; Dn is constexpr.
        for j in range(0, Dn, BLOCK_D):
            offs_j = j + tl.arange(0, BLOCK_D)
            mask_j = offs_j < Dn
            # Initialize accumulator for this j block
            acc_j = tl.zeros([BLOCK_D], dtype=tl.float32)
            # Loop over k tile and accumulate into acc_j
            for kk in range(0, BLOCK_K):
                # scalar attn value or vector? It's [BLOCK_K]; kk is index
                # We need to load attn_tile[kk] as scalar and Kc[k0+kk, offs_j] as vector
                k_idx = k0 + kk
                # mask for kk validity: since kk < BLOCK_K, k_idx < KV if k0+kk<KV (handled by mask_k), but we also need k_idx < KV; we can check mask_k[kk] via kk? In Triton, we can guard by kk < (KV - k0) but Triton doesn't support such dynamic checks; better: load with mask for offs_k. For kk beyond KV-k0, attn_tile[kk] was loaded as 0 due to mask_k; we can use that.
                attn_val = attn_tile[kk]  # scalar
                kc_vals = tl.load(Kc_ptr + k_idx * Dn + offs_j, mask=mask_j, other=0.0)  # [BLOCK_D]
                acc_j += attn_val * kc_vals
            # Add acc_j to out_vec
            out_vec[offs_j] += acc_j

    # Store out_vec
    tl.store(out_ptr + tl.arange(0, Dn), out_vec, mask=True)


# Triton kernel: cast float32 to bfloat16 (used to store output as bfloat16)
@triton.jit
def cast_to_bf16_kernel(in_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(in_ptr + offs, mask=mask, other=0.0)
    x = x.to(tl.bfloat16)
    tl.store(out_ptr + offs, x, mask=mask)


# ModelNew: Triton-only forward
class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Ensure inputs are on CUDA
        device = q_nope.device
        assert device.type == 'cuda', "Triton kernels require CUDA tensors"

        total_q, num_qo_heads, head_dim_ckv = q_nope.shape
        head_dim_kpe = q_pe.shape[-1]
        # Constants per problem
        num_qo_heads = num_qo_heads
        head_dim_ckv = head_dim_ckv
        head_dim_kpe = head_dim_kpe

        # Cast inputs to float32 for compute
        q_nope_f32 = q_nope.to(torch.float32)
        q_pe_f32 = q_pe.to(torch.float32)
        Kc_all = ckv_cache.squeeze(1).to(torch.float32)  # [M, Dn]
        Kp_all = kpe_cache.squeeze(1).to(torch.float32)  # [M, Dp]

        # Output tensors (float32 during compute, cast later to bfloat16)
        output = torch.empty((total_q, num_qo_heads, head_dim_ckv), dtype=torch.float32, device=device)
        lse = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=device)

        # Process each batch b and each query i
        B = qo_indptr.shape[0] - 1
        for b in range(B):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            q_len = q_end - q_start

            # If no queries, skip
            if q_len == 0:
                continue

            # KV range for this batch
            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())
            KV = kv_end - kv_start
            prefix_len = KV - q_len  # number of previously cached tokens

            # Gather Kc and Kp for this batch
            tok_idx = kv_indices[kv_start:kv_end].to(torch.int32)  # [KV]
            Kc = Kc_all[tok_idx]  # [KV, Dn]
            Kp = Kp_all[tok_idx]  # [KV, Dp]

            # For each query i
            for i in range(q_len):
                q_abs_pos = prefix_len + i  # absolute position of this query

                # For each head h
                for h in range(num_qo_heads):
                    # Prepare row pointers: qn_row[h, :] and qp_row[h, :]
                    qn_row = q_nope_f32[q_start + i, h, :]  # [Dn]
                    qp_row = q_pe_f32[q_start + i, h, :]    # [Dp]

                    # Allocate buffers
                    logits = torch.empty(KV, dtype=torch.float32, device=device)
                    masked = torch.empty(KV, dtype=torch.float32, device=device)

                    # Kernel 1: compute logits
                    compute_logits_kernel[(1,)](
                        qn_row, qp_row, Kc, Kp, logits,
                        KV=KV, Dn=head_dim_ckv, Dp=head_dim_kpe, BLOCK_K=64
                    )

                    # Kernel 2: apply causal mask
                    mask_logits_kernel[(1,)](
                        logits, masked,
                        sm_scale, KV, prefix_len, i,
                        BLOCK_K=64
                    )

                    # Kernel 3: compute lse
                    lse_row_kernel[(1,)](
                        masked, lse[q_start + i, h],
                        KV=KV, BLOCK_K=64
                    )

                    # Kernel 4: compute softmax over masked logits
                    attn = torch.empty(KV, dtype=torch.float32, device=device)
                    softmax_row_kernel[(1,)](
                        masked, attn,
                        KV=KV, BLOCK_K=64
                    )

                    # Kernel 5: compute out[h, :] = attn @ Kc
                    out_row = torch.empty(head_dim_ckv, dtype=torch.float32, device=device)
                    compute_out_row_kernel[(1,)](
                        attn, Kc, out_row,
                        KV=KV, Dn=head_dim_ckv, BLOCK_K=64, BLOCK_D=128
                    )

                    # Store output as bfloat16
                    out_bf16 = torch.empty(head_dim_ckv, dtype=torch.bfloat16, device=device)
                    cast_to_bf16_kernel[(1,)](
                        out_row, out_bf16, head_dim_ckv, BLOCK=256
                    )
                    output[q_start + i, h, :] = out_bf16

        # Return output and lse
        return output, lse

# Example helper functions (unchanged, for local testing)
def get_inputs():
    q_nope = torch.randn([1, 16, 512], dtype=torch.bfloat16, device='cuda')
    q_pe = torch.randn([1, 16, 64], dtype=torch.bfloat16, device='cuda')
    ckv_cache = torch.randn([989669, 1, 512], dtype=torch.bfloat16, device='cuda')
    kpe_cache = torch.randn([989669, 1, 64], dtype=torch.bfloat16, device='cuda')
    _n = 1; _t = 1
    _lens = torch.full((_n,), _t // _n, dtype=torch.int32)
    _lens[: _t % _n] += 1
    qo_indptr = torch.cat([torch.zeros(1, dtype=torch.int32), torch.cumsum(_lens, 0)]).to(torch.int32).cuda()
    _n = 1; _t = 34
    _lens = torch.full((_n,), _t // _n, dtype=torch.int32)
    _lens[: _t % _n] += 1
    kv_indptr = torch.cat([torch.zeros(1, dtype=torch.int32), torch.cumsum(_lens, 0)]).to(torch.int32).cuda()
    kv_indices = torch.randint(0, 989669, [34], dtype=torch.int32).cuda()
    sm_scale = 1.0
    return [q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale]

def fused_operator(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6, tensor_7):
    _out = ModelNew()(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6, tensor_7)
    return list(_out) if isinstance(_out, (tuple, list)) else [_out]

# Optional local test (won't run in evaluator environment)
if __name__ == "__main__":
    model = ModelNew()
    q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale = get_inputs()
    out, lse = model(q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale)
    print(out.shape, lse.shape)


def run(*args):
    return ModelNew()(*args)
