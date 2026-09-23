import math
import torch
import triton
import triton.language as tl


# Kernel to compute LSE (logsumexp) over j in [0, max_kv_idx) for a single (b, q_idx, h).
@triton.jit
def _lse_row_kernel(
    kv_indices_ptr,         # *int32
    q_vec_ptr,              # *float32, length head_dim
    k_cache_flat_ptr,       # *float32, shape [num_pages*num_kv_heads, head_dim], contiguous
    lse_out_ptr,            # *float32, scalar (1-element tensor) for this (b, q_idx, h)
    num_kv_tokens: tl.int32,
    kv_start: tl.int32,
    q_idx: tl.int32,        # not used directly
    num_q_tokens: tl.int32, # not used directly
    delta: tl.int32,
    max_kv_idx: tl.int32,
    head_dim: tl.constexpr, # e.g., 128
    num_kv_heads: tl.int32, # e.g., 8
    gqa_ratio: tl.int32,    # e.g., 4
    sm_scale: tl.float32,   # scaling for logits
):
    # This kernel assumes a single head h; h is fixed from host launch (we pass no h here).
    # Running max and sum for LSE
    m = -float('inf')
    sumexp = 0.0  # float32

    # First pass: compute LSE
    j = 0
    while j < max_kv_idx:
        k_id = tl.load(kv_indices_ptr + kv_start + j)  # int32
        # GQA mapping: qo head h maps to kv head h // gqa_ratio
        # Here h is implicit (host fixes it); we don't have h in kernel args.
        # We will rely on host passing the correct gqa_ratio and kv_heads=8, h derived externally if needed.
        # The original Python loop sets kv_head = h // gqa_ratio per head h, so we need h to compute kv_head.
        # Since we are in a single (b, q_idx, h) launch, we can pass h as a constexpr or as an argument.
        # However, Triton kernel signature doesn't allow us to receive h; so we structure host loop to call
        # this kernel per head h with fixed h. We avoid passing h here, instead rely on host loop's q_vec_ptr
        # being q_f32[global_q_idx, h], and compute kv_head using host-side logic before kernel launch.
        # To simplify, we assume host uses this kernel with a fixed h known to host; the Triton side doesn't need h.
        # The earlier snippet needed to access q_f32[global_q_idx, h]—we pass that vector via q_vec_ptr.
        # For kv_head, we need h; Triton doesn't get h here. Therefore, we restructure as follows:
        # We actually need h to compute kv_head. The simplest robust approach is to have host loop set h,
        # and we'll call this kernel only after storing h in a way the kernel can read? Not possible.
        # Hence, we redesign: host will launch this kernel once per (b, q_idx, h) by providing h via a constexpr,
        # but Triton doesn't allow passing h here. The fix is to have the host loop maintain h as a local int
        # and pass it to kernel by embedding into launch; Triton allows passing scalars via args. So we add h.
        # We'll re-declare the kernel to accept h. Let's redefine it properly with h as an argument.
    # Note: The above comments show the constraints. We will fix the kernel signature to include h.
    pass  # placeholder; see the corrected version below


# Kernel to compute output vector for a single (b, q_idx, h) using the precomputed LSE (lso_row_kernel).
@triton.jit
def _output_row_kernel(
    kv_indices_ptr,         # *int32
    q_vec_ptr,              # *float32, length head_dim
    k_cache_flat_ptr,       # *float32, shape [num_pages*num_kv_heads, head_dim], contiguous
    v_cache_flat_ptr,       # *float32, shape [num_pages*num_kv_heads, head_dim], contiguous
    out_ptr,                # *float32, output vector [head_dim] for this (b, q_idx, h)
    lse_val,                # scalar float32 (same for all j for this q_idx,h)
    num_kv_tokens: tl.int32,
    kv_start: tl.int32,
    q_idx: tl.int32,        # not used directly
    num_q_tokens: tl.int32, # not used directly
    delta: tl.int32,
    max_kv_idx: tl.int32,
    head_dim: tl.constexpr, # e.g., 128
    num_kv_heads: tl.int32, # e.g., 8
    gqa_ratio: tl.int32,    # e.g., 4
):
    # Single head h is implicit (host ensures correct launch). We recompute everything.
    # We will re-declare this kernel with h argument too, mirroring _lse_row_kernel.
    pass  # placeholder; see the corrected version below


# Redefine kernels correctly with h argument to avoid confusion.
@triton.jit
def _lse_row_kernel_h(
    kv_indices_ptr,         # *int32
    q_vec_ptr,              # *float32, length head_dim
    k_cache_flat_ptr,       # *float32, [num_pages*num_kv_heads, head_dim], contiguous
    lse_out_ptr,            # *float32, scalar output
    h: tl.int32,            # qo head index
    num_kv_tokens: tl.int32,
    kv_start: tl.int32,
    q_idx: tl.int32,        # not used directly
    num_q_tokens: tl.int32, # not used directly
    delta: tl.int32,
    max_kv_idx: tl.int32,
    head_dim: tl.constexpr, # e.g., 128
    num_kv_heads: tl.int32, # e.g., 8
    sm_scale: tl.float32,   # scaling for logits
):
    # Running max and sum for LSE
    m = -float('inf')
    sumexp = 0.0  # float32

    # First pass: compute LSE over j in [0, max_kv_idx)
    j = 0
    while j < max_kv_idx:
        k_id = tl.load(kv_indices_ptr + kv_start + j)  # int32
        kv_head = h // 4  # GQA mapping: 32 qo heads -> 8 kv heads
        row_id = k_id * (num_kv_heads * head_dim) + kv_head * head_dim
        k_row = tl.load(k_cache_flat_ptr + row_id + tl.arange(0, head_dim))  # [head_dim] float32
        q_vec = tl.load(q_vec_ptr + tl.arange(0, head_dim))                 # [head_dim] float32
        dot = tl.sum(q_vec * k_row)
        scaled = dot * sm_scale
        m_new = tl.maximum(m, scaled)
        sumexp = sumexp * tl.exp(m - m_new) + tl.exp(scaled - m_new)
        m = m_new
        j += 1

    # LSE: logsumexp(scaled) / ln(2)
    ln2 = 0.6931471805599453  # math.log(2.0)
    lse_val = (m + tl.log(sumexp)) / ln2
    tl.store(lse_out_ptr, lse_val)


@triton.jit
def _output_row_kernel_h(
    kv_indices_ptr,         # *int32
    q_vec_ptr,              # *float32, length head_dim
    k_cache_flat_ptr,       # *float32, [num_pages*num_kv_heads, head_dim], contiguous
    v_cache_flat_ptr,       # *float32, [num_pages*num_kv_heads, head_dim], contiguous
    out_ptr,                # *float32, output vector [head_dim]
    lse_val,                # scalar float32
    h: tl.int32,            # qo head index
    num_kv_tokens: tl.int32,
    kv_start: tl.int32,
    q_idx: tl.int32,        # not used directly
    num_q_tokens: tl.int32, # not used directly
    delta: tl.int32,
    max_kv_idx: tl.int32,
    head_dim: tl.constexpr, # e.g., 128
    num_kv_heads: tl.int32, # e.g., 8
    sm_scale: tl.float32,   # scaling for logits (not used here, but kept for signature symmetry)
):
    # Initialize output vector to zeros
    d = 0
    while d < head_dim:
        tl.store(out_ptr + d, 0.0)
        d += 1

    j = 0
    while j < max_kv_idx:
        k_id = tl.load(kv_indices_ptr + kv_start + j)
        kv_head = h // 4
        row_id = k_id * (num_kv_heads * head_dim) + kv_head * head_dim
        k_row = tl.load(k_cache_flat_ptr + row_id + tl.arange(0, head_dim))  # [head_dim]
        q_vec = tl.load(q_vec_ptr + tl.arange(0, head_dim))                 # [head_dim]
        dot = tl.sum(q_vec * k_row)
        scaled = dot * sm_scale
        attn = tl.exp(scaled - lse_val)  # softmax normalized per j
        v_row = tl.load(v_cache_flat_ptr + row_id + tl.arange(0, head_dim))  # [head_dim]
        # out += attn * v_row
        d = 0
        while d < head_dim:
            out_ptr[d] += attn * v_row[d]
            d += 1
        j += 1


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Ensure CUDA and dtype
        device = q.device
        if device.type != "cuda":
            raise RuntimeError("ModelNew requires CUDA tensors. Move inputs to CUDA.")
        q_f32 = q.to(torch.float32).contiguous()

        # Flatten k_cache and v_cache since 'page_size' is 1 in the provided setup.
        # k_cache: [num_pages, 1, num_kv_heads, head_dim]
        num_pages, _, num_kv_heads, head_dim = k_cache.shape
        assert head_dim == 128, "head_dim must be 128"
        assert num_kv_heads == 8, "num_kv_heads must be 8"
        k_flat = k_cache.squeeze(1).to(torch.float32).contiguous()  # [num_pages, num_kv_heads, head_dim]
        k_flat = k_flat.reshape(num_pages * num_kv_heads, head_dim)  # [num_pages*8, 128]
        v_flat = v_cache.squeeze(1).to(torch.float32).contiguous()
        v_flat = v_flat.reshape(num_pages * num_kv_heads, head_dim)

        len_indptr = qo_indptr.shape[0]
        total_q = int(qo_indptr[-1].item())
        # Allocate outputs
        output = torch.empty((total_q, 32, head_dim), dtype=torch.float32, device=device)
        lse = torch.full((total_q, 32), -float("inf"), dtype=torch.float32, device=device)

        gqa_ratio = 4  # 32 qo heads -> 8 kv heads, each qo head maps to 8*4=32 kv heads
        sm_scale = float(sm_scale)

        # Iterate over batches defined by qo_indptr
        for b in range(len_indptr - 1):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            if q_start >= q_end:
                continue
            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())
            if kv_start >= kv_end:
                continue

            num_q_tokens = q_end - q_start
            num_kv_tokens = kv_end - kv_start
            delta = num_kv_tokens - num_q_tokens
            max_kv_idx = min(q_start + 1 + delta, num_kv_tokens)

            # Loop over each query token and each qo head
            for q_idx in range(num_q_tokens):
                global_q_idx = q_start + q_idx
                for h in range(32):
                    # Prepare q vector for this head
                    q_vec = q_f32[global_q_idx, h]  # [head_dim], contiguous

                    # 1) Compute lse_val for this (b, q_idx, h)
                    lse_buf = torch.empty(1, dtype=torch.float32, device=device)
                    _lse


def run(*args):
    return ModelNew()(*args)
