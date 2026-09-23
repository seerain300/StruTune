import math
import torch
import triton
import triton.language as tl


@triton.jit
def qk_matmul_kernel(
    Q_ptr, K_ptr, L_ptr,
    Nq: tl.int32, Nk: tl.int32,
    Hq: tl.constexpr, D: tl.constexpr,
    sm_scale: tl.float32,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr
):
    """
    Compute L[nq, head, nk] = sum_{d=0..D-1} Q[nq, head, d] * K[nk, head, d]
    L is stored as float32. We scale by sm_scale.
    Grid: (ceil_div(Nq, BLOCK_M), ceil_div(Nk, BLOCK_N), Hq)
    """
    pid0 = tl.program_id(0)  # along Nq
    pid1 = tl.program_id(1)  # along Nk
    pid2 = tl.program_id(2)  # along heads

    # Offsets
    q_offsets = pid0 * BLOCK_M + tl.arange(0, BLOCK_M)
    kv_offsets = pid1 * BLOCK_N + tl.arange(0, BLOCK_N)
    # Masks for bounds
    mask_q = q_offsets < Nq
    mask_kv = kv_offsets < Nk

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over D dimension in chunks
    # We use a simple python for to create a loop; Triton will unroll if BLOCK_D is constexpr and D is constexpr.
    for d in range(0, D, BLOCK_D):
        d_offsets = d + tl.arange(0, BLOCK_D)
        # Load Q tile [BLOCK_M, BLOCK_D] for all heads at once? We will load per head scalar for simplicity:
        # But since Hq is constexpr (32), we can broadcast head. We'll compute Q for one head per program.
        # Note: Our grid is (Nq, Nk, Hq), so we can set head = pid2.
        # We compute Q[nq, head, d] and K[nk, head, d] for each tile and accumulate.
        # However, Triton allows us to index by q_offsets and d_offsets, and loop Hq inside.
        # Better approach: since we have Hq as constexpr, we can compute one head at a time using pid2.
        # We'll implement a small loop over Hq=32 (since we hardcode).
        # But the grid already maps heads via pid2, so we compute for head=pid2.
        # We need a 2D load for Q and K: Q is [Nq, Hq, D], K is [Nk, Hq, D].
        # We can do a manual load for each head using pid2.
        # In Triton, we can index with tuple; but simpler is to compute per head directly in the grid.

        # Compute Q tile: Q_ptr has layout [Nq, Hq, D], strides can be inferred via torch strides.
        # Triton pointers are raw, so we pass strides. PyTorch tensors give strides; Triton uses element strides.
        # We can assume contiguous: stride_q_n = Hq*D, stride_q_h = D, stride_q_d = 1.
        # K similarly contiguous.

        # For simplicity, we'll pass strides via arguments by creating views with .contiguous().
        # But we didn't pass strides; Triton requires us to compute addresses. We can pass strides via PyTorch tensor.stride().
        # Since we don't have that, we'll do pointer arithmetic: for Q, address = Q_ptr + n*q_stride + h*k_stride + d*1.
        # Triton will infer element type. Better: we'll pass Q and K as contiguous and compute by simple offsets.

        # We'll implement the load for one head pid2. Note: Triton requires us to specify shapes and compute pointers.
        # Let's do it explicitly:
        # Build Q tile: for each q in q_offsets, load Q[q, pid2, d:d+BLOCK_D] -> [BLOCK_M, BLOCK_D]
        # Build K tile: for each kv in kv_offsets, load K[kv, pid2, d:d+BLOCK_D] -> [BLOCK_N, BLOCK_D]
        # Then acc += Q_tile @ K_tile.T

        # To do this, we can use a while loop over heads (Hq=32) and load Q and K accordingly.
        # However, Triton supports for loops if we define them properly. Since Hq is constexpr, we can loop.
        # We'll implement by loading vectors for Q and K per head.

        # We'll compute acc += sum over heads: for h in range(Hq):
        # But we need to accumulate per head? No: we want dot product over K for each q and each nk, summed over heads (due to GQA).
        # Our current Q has different heads for q. We need to compute Q[nq, h, d] * K[nk, h, d] for all h, then sum over h.
        # Better: compute L[nq, h, nk] = sum_d Q[nq, h, d] * K[nk, h, d], then sum over h after computing all h.
        # To keep it vectorized, we will compute for all heads in this program by iterating h.

        # Initialize per-head accumulators: since Triton does not support tl.zeros((BLOCK_M, BLOCK_N, Hq)), we compute one head at a time.
        # We'll instead compute acc for one head and then loop over heads, but we need to store per-head results. Simpler: compute acc and at the end multiply by sm_scale and store to L.

        # We'll implement the load for each head h and accumulate into acc. Note: Triton can handle Python for with constexpr Hq.
        for h in range(Hq):
            # Load Q tile for head h: [BLOCK_M, BLOCK_D]
            # Address: Q_ptr + q_offsets[:, None] * (Hq*D) + h*D + d_offsets[None, :] * 1
            # We need to ensure contiguous layout; if we passed Q contiguous, this is fine.
            # We'll assume Q is contiguous in the last dim and in (Nq, Hq, D) layout. For simplicity, we'll pass Q as [Nq, Hq, D] contiguous.
            q_ptrs = Q_ptr + q_offsets[:, None] * (Hq * D) + h * D + d_offsets[None, :]
            k_ptrs = K_ptr + kv_offsets[None, :] * (Hq * D) + h * D + d_offsets[:, None]
            # Load with masks
            q_vals = tl.load(q_ptrs, mask=mask_q[:, None], other=0.0)  # [BLOCK_M, BLOCK_D]
            k_vals = tl.load(k_ptrs, mask=mask_kv[None, :], other=0.0)  # [BLOCK_N, BLOCK_D]
            # Accumulate acc += Q_tile @ K_tile.T
            # Ensure shapes: q_vals [BLOCK_M, BLOCK_D], k_vals [BLOCK_N, BLOCK_D], we need [BLOCK_M, BLOCK_N].
            # Compute outer product sum over D: acc += sum_d q_vals[:, d] * k_vals[:, d] for each block
            # We can do: acc += tl.sum(q_vals[:, :, None] * k_vals[None, :, :], axis=2)
            # But Triton supports elementwise multiply and reduction. We can do a simple loop over dd in BLOCK_D and add outer products:
            # Construct outer products
            for dd in range(BLOCK_D):
                d_idx = d + dd
                # Mask for d_idx valid
                if d_idx >= D:
                    break
                q_vec = q_vals[:, dd]            # [BLOCK_M]
                k_vec = k_vals[:, dd]            # [BLOCK_N]
                # Outer product: [BLOCK_M, BLOCK_N]
                acc += q_vec[:, None] * k_vec[None, :]

    # After loop over D and Hq, we have acc for one head. We need to sum over Hq. Since we accumulated per head, we need to loop again and add.
    # Instead, we can initialize acc as zeros and compute for all heads at once by keeping a 3D acc[Hq, BLOCK_M, BLOCK_N] and sum at the end.
    # Triton does not support tl.zeros((BLOCK_M, BLOCK_N, Hq)), but we can loop and store into L separately. Simpler approach: compute L per head and store, then do reduction on host? Not possible because we store L for all heads.

    # Conclusion: We need to compute per-head L and then do a reduction on host. But since we want Triton-heavy, we can compute only for pid2 and rely on host loop? Our grid is 3D over Hq. Let's compute for all heads by using a temporary L[HEADS, BLOCK_M, BLOCK_N] but Triton doesn't support that directly.

    # Therefore, we simplify: we compute L for one head per program (fixed pid2), and we rely on the grid's 3rd dimension to cover Hq. We'll write acc into L[nq, pid2, nk] and then do a reduction across Hq on host. But the host needs the full [Nq, Hq, Nk] tensor for softmax, so this approach is suboptimal.

    # Better approach: we compute per-head L[nq, h, nk] in a separate kernel or handle it inside the grid. Since Triton requires static shapes, we compute for each head and write to L with three-dimensional grid, but Triton kernels typically operate on 1D or 3D with specific indexing. It's cleaner to compute per-head and then sum across Hq in Python.

    # To implement, we'll do: For each program (pid0, pid1, pid2), compute acc as above for head=pid2, scale by sm_scale, and store to L[nq, pid2, nk] using masks. Then on host, we sum across Hq to get L for all heads.

    # However, Triton does not support writing to L with a 3D address that includes Hq dynamically without some indexing. We can still write per head using pid2 and rely on host to sum across heads later. Given the constraints, we will compute per head and sum on host.

    # Now write result for this head:
    # We need to store L[nq, pid2, nk]. We compute addresses accordingly. We assume L is contiguous [Nq, Hq, Nk].
    # L_ptr layout: contiguous, so strides (Nq*Hq*Nk, Hq*Nk, Nk). But Triton can infer from pointer; we just pass L_ptr.
    # We'll compute L addresses:
    # For each q in q_offsets and kv in kv_offsets, store acc[q, kv] into L[q, pid2, kv].
    # We can do this by broadcasting q_offsets and kv_offsets and computing linear index.

    # Create per-head L: We can store acc directly into L with masks; Triton supports storing into 3D by fixing Hq via pid2.

    # For storing, we need to map 2D indices to linear index. We can compute:
    # L index = q_offsets * (Hq * Nk) + pid2 * Nk + kv_offsets
    l_ptrs = L_ptr + q_offsets[:, None] * (Hq * Nk) + pid2 * Nk + kv_offsets[None, :]
    # Store acc with masks
    # Combine masks
    mask = mask_q[:, None] & mask_kv[None, :]
    tl.store(l_ptrs, acc, mask=mask)


@triton.jit
def apply_causal_mask_kernel(
    L_ptr, MaskedL_ptr,
    Nq: tl.int32, Nk: tl.int32,
    Hq: tl.constexpr,
    q_positions_ptr, kv_positions_ptr,
    delta: tl.int32,  # Nk - Nq
    sm_scale: tl.float32,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_H: tl.constexpr
):
    """
    Apply causal mask: for each (q, h, kv), if kv >= q + 1 + delta -> set to -inf.
    Inputs:
      - L_ptr: [Nq, Hq, Nk] float32 logits
      - MaskedL_ptr: same shape
      - q_positions_ptr: [Nq] int32
      - kv_positions_ptr: [Nk] int32
    Grid: (ceil_div(Nq, BLOCK_M), ceil_div(Nk, BLOCK_N), ceil_div(Hq, BLOCK_H))
    """
    pid0 = tl.program_id(0)  # q tiles
    pid1 = tl.program_id(1)  # kv tiles
    pid2 = tl.program_id(2)  # head tiles

    q_offsets = pid0 * BLOCK_M + tl.arange(0, BLOCK_M)
    kv_offsets = pid1 * BLOCK_N + tl.arange(0, BLOCK_N)
    h_offsets = pid2 * BLOCK_H + tl.arange(0, BLOCK_H)

    mask_q = q_offsets < Nq
    mask_kv = kv_offsets < Nk
    mask_h = h_offsets < Hq

    # Load positions
    q_positions = tl.load(q_positions_ptr + q_offsets, mask=mask_q, other=0)
    kv_positions = tl.load(kv_positions_ptr + kv_offsets, mask=mask_kv, other=0)

    # Compute causal condition: kv < (q + 1 + delta)
    cond = kv_positions[None, :] < (q_positions[:, None] + 1 + delta)

    # Build pointers for L and MaskedL. L is [Nq, Hq, Nk] contiguous. Address = q*(Hq*Nk) + h*Nk + kv
    # We'll process a tile of shape [BLOCK_M, BLOCK_N, BLOCK_H]
    # We need a 3D load/store. Triton supports elementwise operations on 3D broadcast.
    # We'll compute per q and kv, then broadcast across h.

    # Initialize masked values
    # We need to load L tile; but we cannot load masked values directly with tl.load. Instead, we'll compute addresses and read L, then decide.
    # Since we don't have q_positions and kv_positions tensors passed into kernel, we'll assume q_positions and kv_positions are the indices themselves.
    # We'll instead pass q_positions_ptr and kv_positions_ptr as int32 tensors of length Nq and Nk.

    # Compute base addresses for L and MaskedL. We need h dimension.
    # L_ptr address for (q,h,kv): q*(Hq*Nk) + h*Nk + kv
    # We'll create 3D indices: q, h, kv. Use broadcasting to form tile.
    # For each h in h_offsets, we can compute L address and store.
    # To store 3D tile, we'll loop over h (since BLOCK_H is constexpr), which Triton supports.

    for h_idx in range(BLOCK_H):
        h = h_offsets[h_idx]
        if h >= Hq:
            break
        # L pointers for this h
        l_ptrs = L_ptr + q_offsets[:, None] * (Hq * Nk) + h * Nk + kv_offsets[None, :]
        # Load L values
        l_vals = tl.load(l_ptrs, mask=(mask_q[:, None] & mask_kv[None, :]), other=0.0)
        # Compute -inf constant
        neg_inf = -float('inf')
        # Apply mask: if cond is False, set to -inf; else keep l_vals
        masked_vals = tl.where(cond, l_vals, neg_inf)
        # Store to MaskedL
        masked_ptrs = MaskedL_ptr + q_offsets[:, None] * (Hq * Nk) + h * Nk + kv_offsets[None, :]
        tl.store(masked_ptrs, masked_vals, mask=(mask_q[:, None] & mask_kv[None, :]))


# ModelNew: Triton-optimized forward
class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Constants as per original assertions
        self.num_qo_heads = 32
        self.num_kv_heads = 8
        self.head_dim = 128
        self.gqa_ratio = self.num_qo_heads // self.num_kv_heads  # 4
        self.sm_scale = 1.0 / math.sqrt(self.head_dim)  # 1/sqrt(128)

        # Triton tuning parameters
        self.BLOCK_M = 32
        self.BLOCK_N = 32
        self.BLOCK_D = 64  # D=128, so two iterations
        # For causal mask kernel
        self.BMASK_M = 32
        self.BMASK_N = 32
        self.BMASK_H = 8

    def forward(self, q, k, v, qo_indptr, kv_indptr, sm_scale):
        # Ensure CUDA tensors
        assert q.is_cuda and k.is_cuda and v.is_cuda and qo_indptr.is_cuda and kv_indptr.is_cuda, "All tensors must be on CUDA for Triton."
        device = q.device

        total_q = int(qo_indptr[-1].item())
        total_kv = int(kv_indptr[-1].item())
        len_indptr = qo_indptr.shape[0]
        assert len_indptr >= 1, "len_indptr must be at least 1"

        # Output buffers
        output = torch.empty((total_q, self.num_qo_heads, self.head_dim), dtype=torch.bfloat16, device=device)
        lse = torch.full((total_q, self.num_qo_heads), -float("inf"), dtype=torch.float32, device=device)

        # Convert to float32 for compute (matches original behavior)
        q_f32 = q.to(torch.float32).contiguous()
        k_f32 = k.to(torch.float32).contiguous()
        v_f32 = v.to(torch.float32).contiguous()

        # Precompute qo positions and kv positions (for causal mask)
        q_positions = torch.arange(total_q, device=device, dtype=torch.int32)
        kv_positions = torch.arange(total_kv, device=device, dtype=torch.int32)

        # We use sm_scale provided; if not provided, use default
        sm_scale = float(sm_scale) if sm_scale is not None else self.sm_scale

        # Process each batch segment
        for b in range(len_indptr - 1):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())

            if q_start >= q_end or kv_start >= kv_end:
                continue

            # Extract segments
            q_batch = q_f32[q_start:q_end]                 # [Nq, 32, 128]
            k_batch = k_f32[kv_start:kv_end]              # [Nk, 8, 128]
            v_batch = v_f32[kv_start:kv_end]              # [Nk, 8, 128]

            Nq = q_batch.shape[0]
            Nk = k_batch.shape[0]
            Hq = self.num_qo_heads  # 32

            # GQA expansion
            k_expanded = k_batch.repeat_interleave(self.gqa_ratio, dim=1)  # [Nk, 32, 128]
            v_expanded = v_batch.repeat_interleave(self.gqa_ratio, dim=1)  # [Nk, 32, 128]

            # Allocate logits buffer L_tmp [Nq, 32, Nk] float32
            L_tmp = torch.empty((Nq, Hq, Nk), dtype=torch.float32, device=device)

            # Launch Triton matmul kernel: Q=q_batch, K=k_expanded, L=L_tmp
            grid = (triton.cdiv(Nq, self.BLOCK_M), triton.cdiv(Nk, self.BLOCK_N), Hq)
            qk_matmul_kernel[grid](
                q_batch, k_expanded, L_tmp,
                Nq, Nk,
                Hq=Hq, D=self.head_dim,
                sm_scale=sm_scale,
                BLOCK_M=self.BLOCK_M, BLOCK_N=self.BLOCK_N, BLOCK_D=self.BLOCK_D,
            )

            # Apply causal mask in Triton (scale by sm_scale before mask if needed, mask uses original L_tmp)
            # We apply mask using L_tmp and write to MaskedL_tmp
            MaskedL_tmp = torch.empty_like(L_tmp)
            grid_mask = (triton.cdiv(Nq, self.BMASK_M), triton.cdiv(Nk, self.BMASK_N), triton.cdiv(Hq, self.BMASK_H))
            apply_causal_mask_kernel[grid_mask](
                L_tmp, MaskedL_tmp,
                Nq, Nk,
                Hq,
                q_positions, kv_positions,
                delta=Nk - Nq,
                sm_scale=sm_scale,
                BLOCK_M=self.BMASK_M, BLOCK_N=self.BMASK_N, BLOCK_H=self.BMASK_H,
            )

            # Softmax along KV tokens: dim=-1 (over Nk)
            # Note: MaskedL_tmp already has -inf where invalid; softmax will zero out those entries.
            attn = torch.softmax(MaskedL_tmp / sm_scale, dim=-1)  # [Nq, 32, Nk], float32

            # Final output: attn @ v_expanded
            # v_expanded: [Nk, 32, 128], attn: [Nq, 32, Nk]
            # We need qhk @ khd -> qhd, which is torch.einsum('qhk,khd->qhd').
            # Compute using PyTorch for simplicity.
            # attn: [Nq, 32, Nk], v_expanded: [Nk,


def run(*args):
    return ModelNew()(*args)
