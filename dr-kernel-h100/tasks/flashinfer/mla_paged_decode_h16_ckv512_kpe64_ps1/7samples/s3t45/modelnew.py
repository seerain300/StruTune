import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: compute logits_scaled vector for one (b, h) pair.
# Inputs:
#   qn_ptr: pointer to q_nope[b, h] -> shape [Dc]
#   Kc_ptr: pointer to gathered Kc -> shape [L, Dc]
#   Kp_ptr: pointer to gathered Kp -> shape [L, Dp]
#   qp_ptr: pointer to q_pe[b, h] -> shape [Dp]
#   out_ptr: pointer to output logits_scaled vector [L]
# Launch grid: (B, H)
@triton.jit
def compute_logits_kernel_bh(qn_ptr, qp_ptr, Kc_ptr, Kp_ptr, out_ptr,
                             L: tl.constexpr, Dc: tl.constexpr, Dp: tl.constexpr,
                             BLOCK_L: tl.constexpr):
    b = tl.program_id(0)
    h = tl.program_id(1)

    # Load qn[h, :] and qp[h, :] (float32)
    qn = tl.load(qn_ptr)  # [Dc]
    qp = tl.load(qp_ptr)  # [Dp]

    # Accumulate logits for each token l in chunks
    logits = tl.zeros((L,), dtype=tl.float32)
    for l_off in tl.static_range(0, L, BLOCK_L):
        l_idx = l_off + tl.arange(0, BLOCK_L)  # [BLOCK_L]
        mask = l_idx < L

        # Load Kc rows for this chunk
        Kc_chunk = tl.load(Kc_ptr + l_idx[:, None] * Dc + tl.arange(0, Dc), mask=mask[:, None], other=0.0)  # [BLOCK_L, Dc]
        # Load Kp rows for this chunk
        Kp_chunk = tl.load(Kp_ptr + l_idx[:, None] * Dp + tl.arange(0, Dp), mask=mask[:, None], other=0.0)  # [BLOCK_L, Dp]

        # Compute qn @ Kc_chunk.T and qp @ Kp_chunk.T
        sum_qn = tl.zeros((BLOCK_L,), dtype=tl.float32)
        for d in tl.static_range(0, Dc):
            sum_qn += qn[d] * Kc_chunk[:, d]

        sum_qp = tl.zeros((BLOCK_L,), dtype=tl.float32)
        for d in tl.static_range(0, Dp):
            sum_qp += qp[d] * Kp_chunk[:, d]

        logits[l_off:l_off + BLOCK_L] = sum_qn + sum_qp

    # Store logits
    tl.store(out_ptr, logits)


# Triton kernel: scale logits vector (in-place), using scale (we'll compute scaling in host code and pass a pre-scaled buffer)
@triton.jit
def scale_logits_kernel(inp_ptr, out_ptr, L: tl.constexpr):
    # This kernel is a no-op in this design: host pre-scales. We keep it here for future use if needed.
    pass


# Triton kernel: compute base-2 logsumexp for a vector of length L (stored at inp_ptr) -> writes scalar to out_ptr
@triton.jit
def compute_lse_base2_kernel(inp_ptr, out_ptr, L: tl.constexpr):
    # Reduce over L: compute max, then sum of exp(x - max), then lse = max + log(sumexp) - log(2)
    max_val = tl.full((), -1e30, tl.float32)
    for l in tl.static_range(0, L):
        x = tl.load(inp_ptr + l)
        max_val = tl.maximum(max_val, x)

    sumexp = tl.full((), 0.0, tl.float32)
    for l in tl.static_range(0, L):
        x = tl.load(inp_ptr + l)
        sumexp += tl.exp(x - max_val)

    lse_val = max_val + tl.log(sumexp) - math.log(2.0)
    tl.store(out_ptr, lse_val)


# Triton kernel: compute softmax over a vector of length L (stored at inp_ptr) -> writes result to out_ptr
@triton.jit
def softmax_kernel(inp_ptr, out_ptr, L: tl.constexpr):
    max_val = tl.full((), -1e30, tl.float32)
    for l in tl.static_range(0, L):
        x = tl.load(inp_ptr + l)
        max_val = tl.maximum(max_val, x)

    sumexp = tl.full((), 0.0, tl.float32)
    for l in tl.static_range(0, L):
        x = tl.load(inp_ptr + l)
        sumexp += tl.exp(x - max_val)

    for l in tl.static_range(0, L):
        x = tl.load(inp_ptr + l)
        y = tl.exp(x - max_val) / sumexp
        tl.store(out_ptr + l, y)


# Triton kernel: compute out vector for one (b, h): out[h, :] = attn[h, :] @ Kc_b
# Kc_b is [L, Dc], attn is [L] (flat), out is [Dc]
@triton.jit
def compute_out_kernel(attn_ptr, Kc_ptr, out_ptr, Dc: tl.constexpr, L: tl.constexpr, BLOCK_L: tl.constexpr):
    # Accumulate out vector of length Dc
    out_vec = tl.zeros((Dc,), dtype=tl.float32)
    for l_off in tl.static_range(0, L, BLOCK_L):
        l_idx = l_off + tl.arange(0, BLOCK_L)
        mask = l_idx < L
        attn_chunk = tl.load(attn_ptr + l_idx, mask=mask, other=0.0)  # [BLOCK_L]
        Kc_chunk = tl.load(Kc_ptr + l_idx[:, None] * Dc + tl.arange(0, Dc), mask=mask[:, None], other=0.0)  # [BLOCK_L, Dc]
        # out_vec += sum(attn_chunk[:, None] * Kc_chunk, axis=0)
        for d in tl.static_range(0, Dc):
            out_vec[d] += tl.sum(attn_chunk * Kc_chunk[:, d], axis=0)
    tl.store(out_ptr, out_vec)


# Entry point ModelNew
class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure tensors are on CUDA
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda, "Tensors must be on CUDA for Triton."
        B = q_nope.shape[0]
        H = q_nope.shape[1]
        Dc = q_nope.shape[2]
        Dp = q_pe.shape[2]
        N = ckv_cache.shape[0]
        # Compute tok_idx per batch
        # Note: Assumes 1D caches (as in original). We gather by indices.
        output = torch.empty((B, H, Dc), dtype=torch.float32, device=q_nope.device)  # keep compute in fp32, cast later
        lse = torch.empty((B, H), dtype=torch.float32, device=q_nope.device)

        for b in range(B):
            # Derive token indices for this batch
            # Handle empty case
            if kv_indptr[b + 1].item() <= kv_indptr[b].item():
                # No KV for this batch element
                lse[b] = -float('inf')
                # output[b] will be computed below per head (they are zero)
                continue

            tok_idx = kv_indices[kv_indptr[b].item():kv_indptr[b + 1].item()].to(torch.int32)
            L_b = tok_idx.numel()

            # Gather Kc and Kp for this batch
            Kc_b = ckv_cache[tok_idx, 0].to(torch.float32)  # [L_b, Dc]
            Kp_b = kpe_cache[tok_idx, 0].to(torch.float32)  # [L_b, Dp]

            # Prepare per-head buffers
            for h in range(H):
                # 1) Compute logits for this head
                # Pointers for qn and qp
                qn_ptr = q_nope[b, h].to(torch.float32)  # [Dc]
                qp_ptr = q_pe[b, h].to(torch.float32)    # [Dp]
                # Allocate logits buffer [L_b]
                logits = torch.empty(L_b, dtype=torch.float32, device=q_nope.device)
                # Launch Triton kernel to compute logits for (b, h)
                BLOCK_L = 1  # vector length is L_b; using a single-chunk loop keeps code simple
                grid = (b, h)
                compute_logits_kernel_bh[grid](
                    qn_ptr, qp_ptr, Kc_b, Kp_b, logits,
                    L_b, Dc, Dp,
                    BLOCK_L
                )

                # 2) Compute logits_scaled = logits * sm_scale (host-side scaling to avoid Triton kwargs)
                logits_scaled = logits * sm_scale

                # 3) Compute base-2 logsumexp for this head
                lse_buf = torch.empty(1, dtype=torch.float32, device=q_nope.device)
                compute_lse_base2_kernel[(1,)](logits_scaled, lse_buf, L_b)
                lse[b, h] = lse_buf[0]

                # 4) Compute softmax over logits_scaled -> attn
                attn = torch.empty(L_b, dtype=torch.float32, device=q_nope.device)
                softmax_kernel[(1,)](logits_scaled, attn, L_b)

                # 5) Compute output vector for this head: out[h, :] = attn @ Kc_b
                out_vec = torch.empty(Dc, dtype=torch.float32, device=q_nope.device)
                # Choose BLOCK_L for reduction over L_b; set to L_b if small, else 128
                BLOCK_L_red = 128 if L_b > 128 else L_b
                compute_out_kernel[(1,)](attn, Kc_b, out_vec, Dc, L_b, BLOCK_L_red)
                output[b, h, :] = out_vec

        # Cast output to bfloat16 to match original
        output_bf16 = output.to(torch.bfloat16)
        return output_bf16, lse


# Helper functions to match the original signature
def get_inputs():
    # Place tensors on CUDA for Triton
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


# Optional wrapper to match original call
def fused_operator(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6):
    _out = ModelNew()(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6)
    return list(_out) if isinstance(_out, (tuple, list)) else [_out]