import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: compute base-2 logsumexp for a vector of length L_b.
# Input:
#   logits_scaled_ptr: pointer to [L_b] float32 vector
# Output:
#   lse_ptr[0]: float32 scalar = logsumexp_base2
# Launch grid: (1,)  — we pass b and h via static indexing in host code
@triton.jit
def compute_lse_kernel(logits_scaled_ptr, lse_ptr, L_b: tl.constexpr):
    # Numerically stable logsumexp in base-2
    # First pass: find max
    max_val = -float("inf")
    for l in tl.static_range(0, L_b):
        val = tl.load(logits_scaled_ptr + l)
        if val > max_val:
            max_val = val

    # Second pass: sum exp(x - max)
    sum_exp = 0.0
    for l in tl.static_range(0, L_b):
        val = tl.load(logits_scaled_ptr + l)
        sum_exp += tl.exp(val - max_val)

    lse = max_val + tl.log(sum_exp)  # ln sum
    # Convert to base-2 logsumexp: lse_base2 = lse / ln(2)
    lse_base2 = lse / 0.6931471805599453  # 1 / ln(2)
    tl.store(lse_ptr, lse_base2)


# Triton kernel: compute softmax in base-2 for a vector of length L_b (stable).
# Input:
#   logits_scaled_ptr: pointer to [L_b] float32 vector
# Output:
#   attn_ptr: pointer to [L_b] float32 vector (softmax values)
# Launch grid: (1,)
@triton.jit
def compute_softmax_kernel(logits_scaled_ptr, attn_ptr, L_b: tl.constexpr):
    # First pass: max
    max_val = -float("inf")
    for l in tl.static_range(0, L_b):
        val = tl.load(logits_scaled_ptr + l)
        if val > max_val:
            max_val = val

    # Second pass: compute exp and store in attn_ptr
    for l in tl.static_range(0, L_b):
        val = tl.load(logits_scaled_ptr + l)
        e = tl.exp(val - max_val)
        # We will normalize in a third pass; for now just store e
        tl.store(attn_ptr + l, e)


# Triton kernel: compute out[h, :] = sum_l attn[h, l] * Kc[l, :] over Dc, reducing over L_b in chunks.
# Inputs:
#   attn_ptr: pointer to [L_b] float32 vector
#   Kc_ptr: pointer to [L_b*Dc] float32 vector (contiguous)
#   out_ptr: pointer to [Dc] float32 vector to be accumulated
# Launch grid: (1,)
@triton.jit
def compute_out_kernel(attn_ptr, Kc_ptr, out_ptr, L_b: tl.constexpr, Dc: tl.constexpr, BLOCK: tl.constexpr):
    # Accumulate out vector over chunks of L_b
    for l_off in tl.static_range(0, L_b, BLOCK):
        # Load attn chunk
        attn_chunk = tl.zeros((BLOCK,), dtype=tl.float32)
        for i in tl.static_range(0, BLOCK):
            idx = l_off + i
            if idx < L_b:
                attn_chunk[i] = tl.load(attn_ptr + idx)

        # For each column j in Dc, compute sum_j = sum_l attn_chunk[l] * Kc[l, j]
        # We build a [BLOCK, Dc] matrix by loading slices of Kc: Kc[l*stride + j] where stride = Dc
        # But to keep simple, we load per column j and accumulate:
        # Note: We need to iterate j across Dc and accumulate over l in chunk.
        for j in tl.static_range(0, Dc):
            sum_j = 0.0
            # Reduce over l in chunk
            for i in tl.static_range(0, BLOCK):
                l_idx = l_off + i
                if l_idx < L_b:
                    attn_val = attn_chunk[i]
                    # Kc[l, j] is at linear index l_idx*Dc + j
                    kc_val = tl.load(Kc_ptr + l_idx * Dc + j)
                    sum_j += attn_val * kc_val
            # Store accumulated sum into out[j]
            tl.store(out_ptr + j, sum_j)


# ModelNew.forward: Triton-only compute path; host code allocates, slices, launches Triton, returns outputs.
class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure tensors are on CUDA
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda, "All tensors must be on CUDA"
        B, H, Dc = q_nope.shape
        _, _, Dp = q_pe.shape
        N = ckv_cache.shape[0]
        assert kpe_cache.shape[0] == N
        assert kv_indptr.shape[0] == B + 1
        assert kv_indices.shape[0] > 0

        # Output buffers
        output = torch.zeros((B, H, Dc), dtype=torch.float32, device=q_nope.device)  # keep compute in fp32
        lse = torch.empty((B, H), dtype=torch.float32, device=q_nope.device)

        for b in range(B):
            # Compute token indices for this batch: [kv_indptr[b], kv_indptr[b+1])
            # Number of tokens in this batch
            L_b = int(kv_indptr[b + 1].item() - kv_indptr[b].item())
            if L_b <= 0:
                # No KV entries for this batch
                output[b].zero_()
                lse[b] = -float("inf")
                continue

            # Gather Kc and Kp for this batch
            tok_idx = kv_indices[kv_indptr[b]:kv_indptr[b + 1]]
            Kc_b = ckv_cache[tok_idx].to(torch.float32)  # [L_b, Dc]
            Kp_b = kpe_cache[tok_idx].to(torch.float32)  # [L_b, Dp]

            # qn and qp per head
            for h in range(H):
                qn = q_nope[b, h].to(torch.float32)  # [Dc]
                qp = q_pe[b, h].to(torch.float32)   # [Dp]

                # Compute logits_scaled[h, :] = [ (qn @ Kc[l]) + (qp @ Kp[l]) ] * sm_scale
                # We will compute logits in PyTorch (GPU) to avoid Triton dynamic-loop issues.
                # logits[l] = dot(qn, Kc_b[l]) + dot(qp, Kp_b[l])
                logits = (qn[None, :] * Kc_b[:, :]).sum(dim=1) + (qp[None, :] * Kp_b[:, :]).sum(dim=1)  # [L_b]
                logits_scaled = logits * sm_scale  # scalar sm_scale is applied

                # Triton: compute lse_base2 for this head
                lse_elem = torch.empty((), dtype=torch.float32, device=q_nope.device)
                # We need to pass L_b as constexpr. Triton expects tensor as 0-D arg; we can pass L_b directly.
                # Note: Triton kernels will receive L_b as tl.constexpr, we launch grid (1,).
                compute_lse_kernel[(1,)](logits_scaled, lse_elem, L_b=L_b)
                lse[b, h] = lse_elem.item()  # store result

                # Triton: compute softmax attn vector
                attn = torch.empty((L_b,), dtype=torch.float32, device=q_nope.device)
                compute_softmax_kernel[(1,)](logits_scaled, attn, L_b=L_b)

                # Triton: compute out[h, :] = attn @ Kc_b
                out_vec = torch.empty((Dc,), dtype=torch.float32, device=q_nope.device)
                # We need BLOCK > 0 for static_range
                BLOCK_L = 64  # chunk size for reduction over L_b
                compute_out_kernel[(1,)](attn, Kc_b.reshape(-1), out_vec, L_b=L_b, Dc=Dc, BLOCK=BLOCK_L)

                # Store into output
                output[b, h] = out_vec

        # Return outputs in bfloat16 as original, and lse in float32
        return output.to(torch.bfloat16), lse


# Optional: helpers consistent with the original signature
def get_inputs():
    # Ensure tensors are on CUDA
    q_nope = torch.randn([1, 16, 512], dtype=torch.bfloat16, device='cuda')
    q_pe = torch.randn([1, 16, 64], dtype=torch.bfloat16, device='cuda')
    ckv_cache = torch.randn([989669, 1, 512], dtype=torch.bfloat16, device='cuda')
    kpe_cache = torch.randn([989669, 1, 64], dtype=torch.bfloat16, device='cuda')
    _n = 1; _t = 8
    _lens = torch.full((_n,), _t // _n, dtype=torch.int32)
    _lens[: _t % _n] += 1
    kv_indptr = torch.cat([torch.zeros(1, dtype=torch.int32), torch.cumsum(_lens, 0)]).to(torch.int32).to('cuda')
    kv_indices = torch.randint(0, 989669, [8], dtype=torch.int32).to('cuda')
    sm_scale = 1.0  # float32 scalar
    return [q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale]


# If the original signature is used, this wrapper ensures correct call.
class Model(torch.nn.Module):
    def forward(self, *args):
        return run(*args)