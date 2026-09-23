import torch
import math

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


@triton.jit
def _block_attention_kernel(
    q_ptr,        # *float32, shape [M, G, D], M = q_end - q_start (runtime int passed via M,N logic)
    k_ptr,        # *float32, shape [N, GH, D], N = kv_end - kv_start
    v_ptr,        # *float32, shape [N, GH, D]
    out_ptr,      # *bfloat16, shape [M, G, D]
    lse_ptr,      # *float32, shape [M, G]
    # indices (pointers to 2-element int32 arrays)
    qo_indptr_ptr,  # *int32, length 2: [q_start, q_end]
    kv_indptr_ptr,  # *int32, length 2: [kv_start, kv_end]
    # sizes (compile-time constants for kernel tiling)
    G: tl.constexpr,     # num_qo_heads (e.g., 32)
    GH: tl.constexpr,    # num_kv_heads * gqa_ratio (e.g., 8 * 4 = 32)
    D: tl.constexpr,     # head_dim (e.g., 128)
    SM_SCALE: tl.constexpr,  # scaling factor (e.g., 1.0 / sqrt(128))
    BLOCK_D: tl.constexpr,    # tile for head dim (128)
    BLOCK_N: tl.constexpr,    # tile for KV length (e.g., 128)
):
    # Load q range [q_start, q_end) from device tensors
    q_start = tl.load(qo_indptr_ptr + 0).to(tl.int32)
    q_end = tl.load(qo_indptr_ptr + 1).to(tl.int32)
    if q_start >= q_end:
        return

    # Load kv range [kv_start, kv_end) from device tensors
    kv_start = tl.load(kv_indptr_ptr + 0).to(tl.int32)
    kv_end = tl.load(kv_indptr_ptr + 1).to(tl.int32)
    if kv_start >= kv_end:
        return

    # Compute delta (extra K/V tokens relative to Q tokens in this block)
    delta = kv_end - kv_start - (q_end - q_start)

    # Number of queries in this block
    M = q_end - q_start

    # Process each query position q_idx in the block
    for q_idx in range(0, M):
        # Accumulator for LSE per head [G]
        lse_row = tl.full((G,), -float('inf'), tl.float32)
        # Output accumulator for this q_idx: [G, D]
        out_row = tl.zeros((G, D), dtype=tl.float32)

        # Construct Q vector for this q_idx: q_vec has shape [G, D]
        q_vec = tl.zeros((G, D), dtype=tl.float32)
        for g in range(0, G):
            qg_ptr = q_ptr + q_idx * G * D + g * D
            d_offsets = tl.arange(0, BLOCK_D)
            mask_d = d_offsets < D
            q_vec[g, :] = tl.load(qg_ptr + d_offsets, mask=mask_d, other=0.0)

        # Compute attention over K/V (GH == G in this setup)
        # Logits [G, N] in float32
        logits = tl.zeros((G, D * 2), dtype=tl.float32)  # placeholder; we'll fill N columns
        # We need N as a runtime value; we'll recompute logits with N derived from q_end - q_start and kv ranges.
        # Simpler: recompute logits per q_idx using N from q_end - q_start logic. However, Triton kernel can't know N here.
        # Instead, we recompute logits in two places: initialize and fill; but simpler is to keep logits as [G, N] by defining N later.
        # To handle N, we'll set a large upper bound and mask; but better: recompute logits dynamically per q_idx using N.

        # Fix: redefine logits as [G, N]; but Triton doesn't allow runtime N in types. We'll set a large tensor and slice.
        # Easier approach: allocate logits as zeros with tl.zeros((G, N), ...) using N known in host. Since Triton kernel can't know N, we avoid this.

        # Instead, we'll compute logits as [G, N] inside kernel by using a for-loop over n in 0..N-1. But Triton doesn't support arbitrary loops over runtime N cleanly without tiling.
        # So we'll implement a loop over n0 in 0..N with BLOCK_N and use tl.arange for vectorized loads.

        # Compute Q @ K^T in tiles
        for n0 in range(0, D * 2, BLOCK_N):  # placeholder; replace D*2 with actual N once available
            pass  # not used; actual N will be passed via host-side launch and constraints

        # The above comment indicates a design issue: Triton kernel needs N known at launch time. We'll handle this by passing M, N, and relying on host to ensure N is available.
        # To comply, we'll restructure: the host passes N and M to the kernel. We'll declare N as a tl.constexpr parameter and use it.

        # Note: Triton requires compile-time constants for loops; thus we keep N as tl.constexpr passed from host.

        # Reconstruct: we need N declared as constexpr; but we can't pass N directly since Triton can't bind runtime N. So we'll implement an alternative: compute N inside kernel from qo_indptr and kv_indptr.
        # However, that defeats purpose. Better: simplify by computing N via host and passing it to the kernel via constexpr. Since Triton cannot bind runtime N, we will remove this complexity.

        # We'll implement a simpler, correct version: compute one q_idx per kernel launch and handle the full block by host-side loop, which is what we do next.

        # Simpler approach: perform per-(q_idx) computation by launching kernel once per block with M=N of that block. Triton kernel will process all q_idx in that block using for q_idx in range(M).

        # Compute Q @ K^T for this q_idx, using GH repeats for K/V, apply causal mask, compute LSE, softmax, and output.

        # GH == G (since gqa_ratio=4 and num_kv_heads=8, num_qo_heads=32 => GH=32), so we can iterate gh in 0..G-1.

        # Let's actually implement the full attention computation:

        # Initialize logits [G, N] as zeros (we'll fill them in the loop)
        logits = tl.zeros((G, N), dtype=tl.float32)

        # Compute logits = Q[q_idx, :] @ K_chunk^T for all K tiles
        for n0 in range(0, N, BLOCK_N):
            n_offsets = n0 + tl.arange(0, BLOCK_N)
            mask_n = n_offsets < N

            for gh in range(0, GH):
                # Load K chunk [BLOCK_N, D] for this gh
                k_chunk = tl.zeros((BLOCK_N, D), dtype=tl.float32)
                v_chunk = tl.zeros((BLOCK_N, D), dtype=tl.float32)

                k_base = k_ptr + (kv_start + n_offsets) * GH * D + gh * D
                v_base = v_ptr + (kv_start + n_offsets) * GH * D + gh * D

                # Load rows
                for i in range(0, BLOCK_N):
                    row_valid = (i + n0) < N
                    k_row_ptr = k_base[i, :]  # base pointer for this row
                    v_row_ptr = v_base[i, :]  # base pointer for this row
                    d_offsets = tl.arange(0, BLOCK_D)
                    mask_d = d_offsets < D
                    k_row = tl.load(k_row_ptr + d_offsets, mask=mask_d & row_valid, other=0.0)
                    v_row = tl.load(v_row_ptr + d_offsets, mask=mask_d & row_valid, other=0.0)
                    k_chunk[i, :] = k_row
                    v_chunk[i, :] = v_row

                # Compute scores = Q @ K_chunk^T, shape [G, BLOCK_N]
                for g2 in range(0, G):
                    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
                    for d0 in range(0, D, BLOCK_D):
                        d_offsets = d0 + tl.arange(0, BLOCK_D)
                        mask_d = d_offsets < D
                        q_sub = q_vec[g2, d_offsets]             # [BLOCK_D]
                        k_sub = k_chunk[:, d_offsets]            # [BLOCK_N, BLOCK_D]
                        prod = q_sub[None, :] * k_sub            # [BLOCK_N, BLOCK_D]
                        acc += tl.sum(prod, axis=1)              # [BLOCK_N]
                    logits[g2, n0:n0+BLOCK_N] = acc

        # Scale logits by SM_SCALE
        logits = logits * SM_SCALE

        # Apply causal mask: for this q_idx, kv position j < q_idx + 1 + delta
        q_add = q_idx + 1 + delta
        for n0 in range(0, N, BLOCK_N):
            n_offsets = n0 + tl.arange(0, BLOCK_N)
            mask_n = n_offsets < N
            for g2 in range(0, G):
                mask_causal = n_offsets < q_add
                logits[g2, n0:n0+BLOCK_N] = tl.where(mask_causal & mask_n, logits[g2, n0:n0+BLOCK_N], -float('inf'))

        # Compute row-wise LSE (base 2): logsumexp
        for g2 in range(0, G):
            max_score = tl.max(logits[g2, :])
            sum_exp = tl.sum(tl.exp(logits[g2, :] - max_score))
            lse_val = max_score + tl.log(sum_exp)  # natural log
            lse_val = lse_val / math.log(2.0)      # convert to base-2
            lse_row[g2] = lse_val

        # Compute softmax over N for each g, then output = softmax * V
        for g2 in range(0, G):
            max_score = tl.max(logits[g2, :])
            sum_exp = tl.sum(tl.exp(logits[g2, :] - max_score))
            soft = tl.exp(logits[g2, :] - max_score) / sum_exp  # [N]
            # out[g2, d] += sum over n of soft[n] * V[n, g2, d]
            for n0 in range(0, N, BLOCK_N):
                n_offsets = n0 + tl.arange(0, BLOCK_N)
                mask_n = n_offsets < N
                v_chunk = tl.zeros((BLOCK_N, D), dtype=tl.float32)
                v_base = v_ptr + (kv_start + n_offsets) * GH * D + g2 * D
                for i in range(0, BLOCK_N):
                    row_valid = (i + n0) < N
                    v_row_ptr = v_base[i, :]
                    d_offsets = tl.arange(0, BLOCK_D)
                    mask_d = d_offsets < D
                    v_row = tl.load(v_row_ptr + d_offsets, mask=mask_d & row_valid, other=0.0)
                    v_chunk[i, :] = v_row
                for d0 in range(0, D, BLOCK_D):
                    d_offsets = d0 + tl.arange(0, BLOCK_D)
                    mask_d = d_offsets < D
                    out_row[g2, d_offsets] += tl.sum(soft[n0:n0+BLOCK_N] * v_chunk[:, d_offsets], axis=0)

        # Store output for this q_idx and all heads
        out_base = out_ptr + q_idx * G * D
        for g2 in range(0, G):
            d_offsets = tl.arange(0, BLOCK_D)
            mask_d = d_offsets < D
            tl.store(out_base + g2 * D + d_offsets, out_row[g2, d_offsets], mask=mask_d)
        # Store LSE for this q_idx and all heads
        lse_base = lse_ptr + q_idx * G
        for g2 in range(0, G):
            tl.store(lse_base + g2, lse_row[g2])

        # We processed one q_idx; loop handles all q_idx in M.


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, qo_indptr, kv_indptr, sm_scale):
        # Validate types and shapes
        assert q.dtype == torch.bfloat16 and k.dtype == torch.bfloat16 and v.dtype == torch.bfloat16
        assert q.shape[1] == 32 and k.shape[1] == 8 and q.shape[2] == 128, "Expected q: [*, 32, 128], k: [*, 8, 128]"
        num_qo_heads = 32
        num_kv_heads = 8
        head_dim = 128
        gqa_ratio = num_qo_heads // num_kv_heads  # 4
        GH = num_kv_heads * gqa_ratio  # 32

        device = q.device
        M_total, G, D = q.shape
        N_total, GHk, Dk = k.shape
        assert D == 128 and GHk == GH and Dk == 128, "Invalid shapes"
        assert qo_indptr.shape[0] == len_indptr and kv_indptr.shape[0] == len_indptr, "Invalid indptr lengths"
        # We will loop over blocks b in [0, len_indptr-1], using qo_indptr[b:b+1] and kv_indptr[b:b+1]

        # Allocate outputs (computed by Triton)
        output = torch.empty((M_total, G, D), dtype=torch.bfloat16, device=device)
        lse = torch.full((M_total, G), -float("inf"), dtype=torch.float32, device=device)

        # Convert to float32 for computation; output is bfloat16, lse is float32
        q_f32 = q.to(torch.float32)
        k_f32 = k.to(torch.float32)
        v_f32 = v.to(torch.float32)

        len_indptr = qo_indptr.shape[0]
        for b in range(len_indptr - 1):
            # Slice q, k, v for this block using qo_indptr[b:b+1] and kv_indptr[b:b+1]
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())
            M = q_end - q_start
            N = kv_end - kv_start

            # Create device tensors for per-block indices
            qo_indptr_b = torch.tensor([q_start, q_end], dtype=torch.int32, device=device)
            kv_indptr_b = torch.tensor([kv_start, kv_end], dtype=torch.int32, device=device)

            # Slice tensors for this block
            q_block = q_f32[q_start:q_end]  # [M, G, D]
            k_block = k_f32[kv_start:kv_end]  # [N, GH, D]
            v_block = v_f32[kv_start:kv_end]  # [N, GH, D]

            # Launch Triton kernel once per block
            _block_attention_kernel[(1,)](
                q_block, k_block, v_block,
                output, lse,
                qo_indptr_b, kv_indptr_b,
                G=G, GH=GH, D=D, SM_SCALE=sm_scale,
                BLOCK_D=128, BLOCK_N=128,
            )

        return output, lse


def run(*args):
    return ModelNew()(*args)
