import torch
import math
import triton
import triton.language as tl


@triton.jit
def _attention_segment_kernel(
    q_ptr, k_ptr, v_ptr,
    out_ptr, lse_ptr,
    Q, K,
    sm_scale, ln2,
    H: tl.constexpr,  # num_qo_heads = 32
    KD: tl.constexpr,  # head_dim = 128
):
    # Each program instance processes a single segment: compute all i,h pairs.
    # We will iterate i from 0 to Q-1, h from 0 to H-1, j from 0 to K-1.
    # Compute LSE and output for each (i, h).
    # For output shape [Q, H, KD], linearized offset for (i, h, d) is i*H*KD + h*KD + d.

    # Constants and setup
    # For simplicity, assume head_dim KD=128. We'll use for-loops in range(KD).
    # Note: Triton supports range loops with compile-time upper bounds; KD is constexpr.
    # Precompute base offsets for q/h and k/h access within heads.

    # We need to build q[i, h, :] and k[j, h, :] vectors for each (i, h, j).
    # Triton pointer arithmetic for arbitrary strides is possible, but here tensors are contiguous:
    # q is [Q, H, KD], so element offset is i*H*KD + h*KD + d.
    # k is [K, H, KD], so element offset is j*H*KD + h*KD + d.
    # v is [K, H, KD], so element offset is j*H*KD + h*KD + d.

    # We will compute lse for each (i, h) and then compute output.

    # First, compute lse per (i, h) in chunks across j to be robust.
    # However, Triton prefers loops with constexpr bounds; since KD=128 is constexpr, we can do full vector reduction.
    # But to avoid requiring a 2D temporary, we'll compute lse using the standard trick:
    # lse_base = max over j of logits; sum = sum exp(logits - lse_base); lse_base2 = lse_base + log(sum) / ln2.
    # We can compute this with nested loops.

    # However, computing lse requires scanning all j; Triton supports dynamic loops for runtime Q, K.
    # We'll do the scan using Python for-loops over j, accumulating max and sum.
    # Then we'll compute output by re-scanning j again.

    # Initialize per-(i, h) lse and denom (as scalar per (i, h)). Use a 2D array in output buffer? No, we store per i,h.
    # lse_ptr is [Q, H] float32, out_ptr is [Q, H, KD] bfloat16. We will cast float32 to bfloat16 for output.

    # We will not store intermediates; recompute is fine.

    # Loop over i
    for i in range(0, Q):
        # Initialize max and sum for lse
        max_val = -1.0e20
        sum_exp = 0.0

        # Compute max over j of logits[i, :, j]
        # Then compute sum_exp = sum_j exp(logits - max_val)
        for j in range(0, K):
            # delta for causal: if j >= (i + 1 + delta), logits = -inf
            delta_seg = K - Q  # segment delta
            # Compute q[i, :, :] vector
            q_vec = tl.zeros((H,), dtype=tl.float32)
            # Loop over head dimension KD to form q_vec for this i
            # We need q[i, h, d] for all h. To form q_vec, we can build per h, but we need all heads.
            # Instead, we can form q vector as a 2D matrix [H, KD] but Triton tensors here are scalar loops.
            # We'll compute q scalar at a time:
            # But Triton doesn't support direct indexing into q_ptr with i and h; instead, we form q[i,h,d] by pointer math:
            # For each h, we form a scalar qdh for d=0..KD-1 and accumulate into a vector? Not directly.
            # A practical approach: compute q[i, h, d] as scalar loads and update q_vec[h] = q[i, h, d] accumulates? Not.
            # Instead, we compute q as a vector in a more structured way by allocating temporary, but Triton doesn't allow tensor variable to hold q[i, :, :].
            # Therefore, we need to rethink: compute q[i, h, :] vector fully before computing all j for this i.

            # To handle this, we will compute q[i, h, :] into a vector and store it in a temporary tensor-like structure via scalar accumulation is not possible.
            # Triton supports per-dimension vectorized operations, but forming a 2D vector here is non-trivial. A simpler approach is to restructure the kernel to vectorize across heads and keys, but Triton's programming model prefers static unrolling, which we avoid.

            # Since Triton doesn't allow building arbitrary 2D vectors easily, we will implement the computation per (i, h, j) using scalar loads and math, which is acceptable for correctness and allowed by Triton (loops with runtime bounds).
            # We'll compute lse and output in nested loops. This avoids any unsupported constructs.

            # Compute q_vec[h] for this i: q[i, h, :]
            # Initialize q_vec
            q_vec = tl.zeros((H,), dtype=tl.float32)
            for d in range(0, KD):
                q_h_d = tl.load(q_ptr + i * H * KD + h * KD + d, mask=d < KD, other=0.0)
                q_vec += q_h_d  # placeholder; actual accumulation is done per h scalar
            # Note: The above is a placeholder. Triton requires explicit scalar loads for each h to build q_vec. However, Triton does not support indexing into a vector q_ptr for multiple h at once; we must load per h in a nested way.
            # Therefore, we will compute the attention for each (i, h) by reloading q[i, h, :] each time. This is fine for correctness.

            # We can instead compute q[i, h, :] by direct per-h loads. Since H=32 is constexpr, we can unroll. However, Triton prefers static_range with constexpr bounds. We'll use Python for-loop with H.

            # For each h, compute logits per j, update max and sum
            lse_ih = -1.0e20
            sum_ih = 0.0

            for h in range(0, H):
                # Build q[i, h, :] as a vector over d
                q_vec_h = tl.zeros((KD,), dtype=tl.float32)
                for d in range(0, KD):
                    qdh = tl.load(q_ptr + i * H * KD + h * KD + d)
                    q_vec_h[d] = qdh

                # Now compute logits for this (i, h) against all j
                # Note: We need k_expanded[j, h, :] and v_expanded[j, h, :]. Given GQA, k_expanded[j, h, :] = k_batch[j, h // gqa_ratio, :].
                # But Triton does not support integer division on tl.int, and gqa_ratio=4 is constexpr. Simpler is to pre-expand k and v; however, we don't have them in this kernel signature. So we'll compute using k_ptr with original 8 heads and multiply by gqa_ratio mapping.
                # To simplify, we will map k and v to 32 heads by repeating. Triton can't index into k_ptr with h // gqa_ratio; so we will compute k_expanded implicitly by looping over kv_heads and mapping. But that would require another set of loops and isn't feasible here.
                # Therefore, we will stick to the original PyTorch behavior: repeat k and v by 4 before launching this kernel. In practice, we provide expanded tensors to the kernel.

            # The above "q_vec_h" approach won't work because Triton requires explicit pointer loads. We need to compute q[i, h, :] vector directly. Triton supports loading a vector but not assigning to a local vector like q_vec_h. The clean approach is to perform per-(i, h) computation with scalar loads for q[i, h, d], k[j, h, d], v[j, h, d], which Triton supports.

            # Revised approach: compute lse and output by nested loops. We'll define helper functions for q_row, k_row, v_row that return vectors over KD for given i, j, h. Triton allows passing vectors to kernel, but constructing vectors dynamically inside kernel is limited. Simpler: use scalar loads in nested loops.
            # Compute lse per (i, h)
            # We'll recompute q[i, h, :], k[j, h, :], v[j, h, :] scalars in nested loops.
            # Initialize max for this (i, h)
            lse_ih = -1.0e20
            sum_ih = 0.0
            for j in range(0, K):
                delta = K - Q
                if j >= (i + 1 + delta):
                    continue
                # Compute dot product q[i, h, :] dot k[j, h, :]
                dot = 0.0
                for d in range(0, KD):
                    qdh = tl.load(q_ptr + i * H * KD + h * KD + d)
                    kvh_d = tl.load(k_ptr + j * H * KD + h * KD + d)
                    dot += qdh * kvh_d
                logits_ij = dot * sm_scale
                # Update max and sum for lse
                if logits_ij > lse_ih:
                    lse_ih = logits_ij
                sum_ih += tl.exp(logits_ij - lse_ih)  # we will later divide by ln2
            # lse_base2 for this (i, h)
            lse_ih = lse_ih + tl.log(sum_ih) / ln2
            # Store lse
            tl.store(lse_ptr + i * H + h, lse_ih)

            # Now compute output for this (i, h)
            denom_ih = 0.0
            out_row_ptr = out_ptr + i * H * KD + h * KD
            for j in range(0, K):
                delta = K - Q
                if j >= (i + 1 + delta):
                    continue
                dot = 0.0
                for d in range(0, KD):
                    qdh = tl.load(q_ptr + i * H * KD + h * KD + d)
                    kvh_d = tl.load(k_ptr + j * H * KD + h * KD + d)
                    dot += qdh * kvh_d
                logits_ij = dot * sm_scale
                numerator_j = tl.exp(logits_ij - lse_ih)  # since lse is base-2? No, we computed lse with log(sum)/ln2, but the original code uses lse = logsumexp(logits) / ln2. We need to match:
                # The original lse = logsumexp(logits) / ln2, and final output uses exp(logits - lse). We computed lse_ih = lse_base + log(sum)/ln2, which is consistent with output scaling.
                denom_ih += numerator_j
                # Load v[j, h, :]
                vj_h = tl.zeros((KD,), dtype=tl.float32)
                for d in range(0, KD):
                    vj_h[d] = tl.load(v_ptr + j * H * KD + h * KD + d)
                out_row += numerator_j * vj_h
            # Scale output by 1/(ln2 * denom_ih) to match original scaling
            out_row = out_row / (ln2 * denom_ih)
            for d in range(0, KD):
                tl.store(out_row_ptr + d, tl.cast(out_row[d], tl.bfloat16))


# Example host function that launches Triton kernel (forward of ModelNew)
class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # No parameters; everything is computed in kernel
        self.sm_scale = 1.0 / math.sqrt(128)
        self.ln2 = math.log(2.0)

    def forward(self, q, k, v, qo_indptr, kv_indptr, sm_scale=None):
        # Ensure device and contiguity; Triton only supports CUDA. Cast to float32 for compute.
        device = q.device
        assert device.type == "cuda", "Triton kernels require CUDA tensors"
        # Cast to float32 for compute stability
        q = q.contiguous().to(torch.float32)
        k = k.contiguous().to(torch.float32)
        v = v.contiguous().to(torch.float32)

        total_q, num_qo_heads, head_dim = q.shape
        total_kv, num_kv_heads, _ = k.shape
        assert num_qo_heads == 32
        assert num_kv_heads == 8
        assert head_dim == 128
        len_indptr = qo_indptr.shape[0]
        assert total_q == int(qo_indptr[-1].item())
        assert total_kv == int(kv_indptr[-1].item())

        # We need to expand k and v along heads to 32 before launching kernel.
        # gqa_ratio = 32 // 8 = 4
        gqa_ratio = 4
        k_expanded = k.repeat_interleave(gqa_ratio, dim=1).contiguous()
        v_expanded = v.repeat_interleave(gqa_ratio, dim=1).contiguous()

        output = torch.empty((total_q, num_qo_heads, head_dim), dtype=torch.bfloat16, device=device)
        lse = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=device)

        for b in range(len_indptr - 1):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())

            if q_start >= q_end or kv_start >= kv_end:
                continue

            q_batch = q[q_start:q_end]  # [Q, 32, 128]
            k_batch = k_expanded[kv_start:kv_end]  # [K, 32, 128]
            v_batch = v_expanded[kv_start:kv_end]  # [K, 32, 128]

            Q = q_batch.shape[0]
            K = k_batch.shape[0]

            out_batch = torch.empty((Q, 32, 128), dtype=torch.bfloat16, device=device)
            lse_batch = torch.empty((Q, 32), dtype=torch.float32, device=device)

            # Launch one Triton program per segment
            _attention_segment_kernel[(1,)](
                q_batch, k_batch, v_batch,
                out_batch, lse_batch,
                Q, K,
                (self.sm_scale if sm_scale is None else sm_scale), self.ln2,
                H=32, KD=128,
                num_warps=4,
            )

            output[q_start:q_end] = out_batch
            lse[q_start:q_end] = lse_batch

        return output, lse


# Optional: helper functions matching the original signatures
def get_inputs():
    q = torch.randn([1, 32, 128], dtype=torch.bfloat16, device='cuda')
    k = torch.randn([1, 8, 128], dtype=torch.bfloat16, device='cuda')
    v = torch.randn([1, 8, 128], dtype=torch.bfloat16, device='cuda')
    # Construct indptrs
    _n = 1; _t = 1
    _lens = torch.full((_n,), _t // _n, dtype=torch.int32)
    _lens[: _t % _n] += 1
    qo_indptr = torch.cat([torch.zeros(1, dtype=torch.int32), torch.cumsum(_lens, 0)]).to(torch.int32).cuda()
    _n = 1; _t = 1
    _lens = torch.full((_n,), _t // _n, dtype=torch.int32)
    _lens[: _t % _n] += 1
    kv_indptr = torch.cat([torch.zeros(1, dtype=torch.int32), torch.cumsum(_lens, 0)]).to(torch.int32).cuda()
    sm_scale = 1.0 / math.sqrt(128)
    return [q, k, v, qo_indptr, kv_indptr, sm_scale]


def fused_operator(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5):
    # Accept original signature and delegate to ModelNew.forward
    model = ModelNew()
    return model(*[t if t is not None else tensor_5 for t in (tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5)])


def run(*args):
    return ModelNew()(*args)
