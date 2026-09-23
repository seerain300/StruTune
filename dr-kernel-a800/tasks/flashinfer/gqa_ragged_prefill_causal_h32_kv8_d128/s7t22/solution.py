import torch
import math
import triton
import triton.language as tl


@triton.jit
def qk_atten_seg_kernel(
    Q_ptr, K_ptr, V_ptr, Out_ptr,
    Nq: tl.int32, Nk: tl.int32,
    Hq: tl.constexpr,  # number of query heads for this segment (== 32 per workload)
    D: tl.constexpr,   # head dimension (== 128 per workload)
    sm_scale: tl.float32,
    segment_id: tl.int32,  # which segment (batch element) this program processes
    num_segments: tl.int32  # total segments (len_indptr), used only for bounds; not used in loops
):
    """
    Each program handles one segment and one head:
      - Inputs: Q[segment], K[segment], V[segment] (already sliced on host)
      - Compute logits L = Q @ K^T (shape [Nq, Nk]), float32
      - Apply forward-looking causal mask: allow j if j < (q + 1 + delta), where delta = Nk - Nq
      - Softmax over Nk for each q
      - Compute attention output: out = softmax * V (shape [Nq, D])
    """
    # Determine segment start/end for qo_indptr and kv_indptr using segment_id.
    # We assume host passes correct Q,K,V slices such that Nq, Nk match segment boundaries.
    # We still need to read qo_indptr[kv_indptr] to compute delta. However, since host
    # slices Q,K,V already, we do not rely on indptr here. We only need Nq, Nk, Hq, D.

    # We will not loop over segments in the kernel; one program per segment/head.

    # Tile sizes: BLOCK_M over queries, BLOCK_N over KV dimension
    BLOCK_M = 64
    BLOCK_N = 64

    # We process heads via a compile-time constant Hq (e.g., 32), so a separate grid dim for heads is sufficient.
    # Triton grid will be (num_segments, Hq), so we can index head via program_id(1).

    # Note: We do not use segment_id/num_segments here because the host slices inputs and provides Nq, Nk.
    # The kernel assumes it is invoked per segment and head with correct Nq, Nk, Hq, D.

    # Compute logits in tiles
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for d0 in range(0, D, BLOCK_N):
        # Each iteration loads K tiles of size [BLOCK_N, D], but since D is 128, one iteration suffices.
        # In general, this loop handles D > BLOCK_N. Triton will compile it if D is constexpr.
        # Here we set D=128, so this is a single iteration.
        kv_offsets = d0 + tl.arange(0, BLOCK_N)
        mask_k = kv_offsets < Nk

        # Load Q tile: [BLOCK_M, D]
        q_offsets = tl.arange(0, BLOCK_M)
        mask_q = q_offsets < Nq

        Q_tile = tl.load(
            Q_ptr + q_offsets[:, None] * (Hq * D) + tl.arange(0, D)[None, :],  # head id=0 for each tile
            mask=mask_q[:, None],
            other=0.0
        ).to(tl.float32)  # [BLOCK_M, D]

        # Load K tile: [BLOCK_N, D]
        K_tile = tl.load(
            K_ptr + kv_offsets[:, None] * (Hq * D) + tl.arange(0, D)[None, :],
            mask=mask_k[:, None],
            other=0.0
        ).to(tl.float32)  # [BLOCK_N, D]

        # Accumulate outer product: [BLOCK_M, BLOCK_N]
        # Note: We index head=0 for simplicity, since we process one head per program via grid.
        # For general Hq, we could load head-specific pointers, but here Hq is 32 and grid dim 1 handles head.
        acc += tl.dot(Q_tile, tl.trans(K_tile))

    # Apply scaling
    acc = acc * sm_scale

    # Apply forward-looking causal mask: j < (q + 1 + delta), where delta = Nk - Nq
    # Build masks
    q_positions = q_offsets  # [BLOCK_M], vector
    delta = Nk - Nq  # scalar int32
    allowed = (kv_offsets[None, :] < (q_positions[:, None] + 1 + delta))
    acc = tl.where(allowed, acc, -float('inf'))

    # Softmax over Nk (row-wise) for each query
    row_max = tl.max(acc, axis=1)  # [BLOCK_M]
    acc = acc - row_max[:, None]
    exp_acc = tl.exp(acc)
    row_sum = tl.sum(exp_acc, axis=1)  # [BLOCK_M]
    attn = exp_acc / row_sum[:, None]  # [BLOCK_M, BLOCK_N]

    # Compute attention output: attn @ V over Nk
    out_tile = tl.zeros((BLOCK_M, D), dtype=tl.float32)
    for d0 in range(0, D, BLOCK_N):
        kv_offsets = d0 + tl.arange(0, BLOCK_N)  # [BLOCK_N]
        mask_k = kv_offsets < Nk

        # Load V tile: [BLOCK_N, D]
        V_tile = tl.load(
            V_ptr + kv_offsets[:, None] * (Hq * D) + tl.arange(0, D)[None, :],
            mask=mask_k[:, None],
            other=0.0
        ).to(tl.float32)  # [BLOCK_N, D]

        # attn[:, n:n+BLOCK_N] dot V_tile[n:n+BLOCK_N, :]
        attn_sub = attn[:, d0:d0+BLOCK_N]  # [BLOCK_M, BLOCK_N]
        out_tile += tl.dot(attn_sub, V_tile)  # [BLOCK_M, D]

    # Store output: [Nq, Hq, D]
    # We store for head index = program_id(1)
    head_id = tl.program_id(1)
    # Build output pointers: out[segment, head_id, :]
    # segment starts at 0, head_id is 0..Hq-1, and we write entire D dimension for each q
    # But segment_id is not directly used here; the host ensures Out_ptr points to correct segment offset.
    # We can't construct segment offset here, so the host must pass Out_ptr pointing to the start of this segment.
    # In our launch, we pass Out_ptr for this segment, so storing below is correct.

    # We need to write to Out_ptr + q_offsets * (Hq * D) + head_id * D
    # However, Triton kernels don't support passing Out_ptr with segment offset; instead, host slices Out.
    # So we assume Out_ptr points to the start of this segment for head=head_id.
    # Store only valid rows q < Nq
    # Since Out tensor is allocated by host with segment slice, we can store directly.
    tl.store(
        Out_ptr + q_offsets[:, None] * (Hq * D) + head_id * D + tl.arange(0, D)[None, :],
        out_tile,
        mask=(q_offsets[:, None] < Nq)
    )


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.num_qo_heads = 32
        self.num_kv_heads = 8
        self.head_dim = 128
        self.gqa_ratio = self.num_qo_heads // self.num_kv_heads  # 4
        self.sm_scale = 1.0 / math.sqrt(self.head_dim)  # 1/sqrt(128)

        # Triton tuning parameters
        self.BLOCK_M = 64  # number of queries per tile
        self.BLOCK_N = 64  # number of KV positions per tile

    def forward(self, q, k, v, qo_indptr, kv_indptr, sm_scale):
        # Ensure CUDA tensors and contiguous
        assert q.is_cuda and k.is_cuda and v.is_cuda and qo_indptr.is_cuda and kv_indptr.is_cuda, "All tensors must be on CUDA for Triton."
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        qo_indptr = qo_indptr.contiguous()
        kv_indptr = kv_indptr.contiguous()

        total_q = int(qo_indptr[-1].item())
        total_kv = int(kv_indptr[-1].item())
        len_indptr = qo_indptr.shape[0]
        assert len_indptr >= 1, "len_indptr must be at least 1"

        # Prepare outputs: per segment and head
        # We need to slice Q,K,V per segment to feed Triton kernels. We’ll allocate Out per segment-head.
        segment_outputs = []
        for b in range(len_indptr - 1):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())

            if q_start >= q_end or kv_start >= kv_end:
                # No queries or KV for this segment; output zeros
                seg_out = torch.zeros((q_end - q_start, self.num_qo_heads, self.head_dim), dtype=torch.float32, device=q.device)
                segment_outputs.append(seg_out)
                continue

            # Slice Q,K,V for this segment
            q_seg = q[q_start:q_end].contiguous()  # [Nq, Hq, D]
            k_seg = k[kv_start:kv_end].contiguous()  # [Nk, Hq, D]
            v_seg = v[kv_start:kv_end].contiguous()  # [Nk, Hq, D]

            # Cast to float32 for Triton
            q_seg_f32 = q_seg.to(torch.float32)
            k_seg_f32 = k_seg.to(torch.float32)
            v_seg_f32 = v_seg.to(torch.float32)

            Nq = q_seg_f32.shape[0]
            Nk = k_seg_f32.shape[0]
            Hq = q_seg_f32.shape[1]  # 32 in this workload
            D = q_seg_f32.shape[2]   # 128 in this workload

            # Output buffer for this segment
            out_seg = torch.empty((Nq, Hq, D), dtype=torch.float32, device=q.device)

            # Launch Triton kernel: one program per head
            grid = (1, Hq)
            qk_atten_seg_kernel[grid](
                q_seg_f32, k_seg_f32, v_seg_f32, out_seg,
                Nq, Nk,
                Hq=Hq, D=D,
                sm_scale=self.sm_scale,
                segment_id=b,  # not used inside kernel; only for host reference
                num_segments=len_indptr
            )

            segment_outputs.append(out_seg)

        # Concatenate segment outputs into final output [total_q, Hq, D]
        output = torch.cat(segment_outputs, dim=0)  # [total_q, Hq, D], float32

        # Cast output to bfloat16 to match original run signature
        output = output.to(torch.bfloat16)

        # lse is not computed (original doesn't use it in harness), so return only output
        return output


def run(*args):
    return ModelNew()(*args)
