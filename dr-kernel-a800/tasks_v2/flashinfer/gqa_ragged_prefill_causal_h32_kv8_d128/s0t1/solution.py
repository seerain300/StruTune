import math
import torch
import triton
import triton.language as tl


@triton.jit
def attention_forward(
    q_ptr, k_ptr, v_ptr, out_ptr, lse_ptr,
    Nq: tl.constexpr, Nk: tl.constexpr,
    q_start, kv_start, q_end, kv_end, sm_scale,
    BLOCK_D: tl.constexpr
):
    """
    Triton kernel computing attention for one segment defined by qo_indptr[b:b+1] and kv_indptr[b:b+1].
    - q_ptr: *float32, [Nq, 32, 128]
    - k_ptr: *float32, [Nk, 8, 128]
    - v_ptr: *float32, [Nk, 8, 128]
    - out_ptr: *float32, [Nq, 32, 128] (will be filled)
    - lse_ptr: *float32, [Nq, 32] (will be filled, base-2 logsumexp)
    - Nq: constexpr (number of queries in this segment)
    - Nk: constexpr (number of keys/values in this segment)
    - q_start, kv_start: int32 (segment starts)
    - q_end, kv_end: int32 (segment ends)
    - sm_scale: float32 scalar
    """
    # Process each query index in the segment; loop bound is constexpr (Nq)
    for q_index in tl.static_range(0, Nq):
        # Base output pointer for this query row
        out_row_base = out_ptr + (q_start + q_index) * 32 * 128

        # For each of the 32 query heads, compute attention over 32 expanded kv heads
        for h in tl.static_range(0, 32):
            # Initialize logits vector for this head (length 32, one per expanded kv head j)
            logits_vec = tl.full((32,), -float('inf'), dtype=tl.float32)

            # Compute dot products across kv heads j=0..31, map to original kv head orig_h = j % 8
            for j in tl.static_range(0, 32):
                orig_h = j % 8  # k/v have 8 heads, expanded to 32

                # q_row base for this (q_index, h)
                q_row_base = q_ptr + (q_start + q_index) * 32 * 128 + h * 128

                # Load q vector for head h (128 elements)
                q_off_d = q_row_base + tl.arange(0, BLOCK_D)
                q_vec = tl.load(q_off_d, mask=tl.arange(0, BLOCK_D) < BLOCK_D, other=0.0).to(tl.float32)

                # Load k vector for token j and head orig_h (128 elements)
                k_row_base = k_ptr + (kv_start + j) * 128 + orig_h * 128
                k_off_d = k_row_base + tl.arange(0, BLOCK_D)
                k_vec = tl.load(k_off_d, mask=tl.arange(0, BLOCK_D) < BLOCK_D, other=0.0).to(tl.float32)

                # Dot product over 128 dims
                acc = tl.sum(q_vec * k_vec, axis=0)

                # Scale by sm_scale
                acc = acc * sm_scale

                # Apply forward-causal mask: allow j if j < (q_index + 1 + (Nk - Nq))
                mask_j = (j < (q_index + 1 + (Nk - Nq)))
                acc = tl.where(mask_j, acc, -float('inf'))

                # Store in logits_vec[j]
                logits_vec[j] = acc

            # Compute base-2 logsumexp for this (q_index, h)
            max_logits = tl.max(logits_vec, axis=0)
            sum_exp = 0.0
            for j in tl.static_range(0, 32):
                sum_exp += tl.exp(logits_vec[j] - max_logits)
            lse_base = tl.log(sum_exp) + max_logits  # natural logsumexp
            lse_base2 = lse_base / 1.4426950408889634  # 1/ln(2)

            # Store lse to lse_ptr[q_index, h]
            lse_off = lse_ptr + q_start * 32 + q_index * 32 + h
            tl.store(lse_off, lse_base2)

            # Softmax across j for this head h
            sum_softmax = 0.0
            for j in tl.static_range(0, 32):
                e = tl.exp(logits_vec[j] - lse_base2)
                sum_softmax += e

            # Accumulate output: out[q_index, h, :] += sum_j softmax[j] * v_expanded[:, j, :]
            out_h_base = out_row_base + h * 128
            out_vec = tl.zeros((BLOCK_D,), dtype=tl.float32)
            for j in tl.static_range(0, 32):
                e = tl.exp(logits_vec[j] - lse_base2)  # softmax[j]
                # For v_expanded[:, j, :], since v has 8 heads and we expand by 4, each j contributes
                # to orig_h = j % 8, and we repeat across 4 expanded slots. We only need the 128-dim vector.
                v_row_base = v_ptr + (kv_start + j) * 128 + orig_h * 128
                v_off_d = v_row_base + tl.arange(0, BLOCK_D)
                v_vec = tl.load(v_off_d, mask=tl.arange(0, BLOCK_D) < BLOCK_D, other=0.0).to(tl.float32)
                out_vec += e * v_vec

            # Store out[q_index, h, :]
            out_off_d = out_h_base + tl.arange(0, BLOCK_D)
            tl.store(out_off_d, out_vec, mask=tl.arange(0, BLOCK_D) < BLOCK_D)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, qo_indptr, kv_indptr, sm_scale):
        """
        Entry point. Computes the same as the original run function but using Triton kernels.
        Inputs:
          q: [total_q, 32, 128], bfloat16
          k: [total_kv, 8, 128], bfloat16
          v: [total_kv, 8, 128], bfloat16
          qo_indptr: [len_indptr], int32, cumulative sums
          kv_indptr: [len_indptr], int32, cumulative sums
          sm_scale: float32
        Returns:
          output: [total_q, 32, 128], bfloat16
          lse: [total_q, 32], float32
        """
        assert q.is_cuda and k.is_cuda and v.is_cuda, "Triton kernels require CUDA tensors"
        assert q.shape[1] == 32 and k.shape[1] == 8 and v.shape[1] == 8 and q.shape[2] == 128 and k.shape[2] == 128 and v.shape[2] == 128
        total_q, _, _ = q.shape
        total_kv, _, _ = k.shape
        len_indptr = qo_indptr.shape[0]
        assert qo_indptr[-1].item() == total_q and kv_indptr[-1].item() == total_kv

        # Prepare outputs
        out = torch.empty((total_q, 32, 128), dtype=torch.float32, device=q.device)
        lse = torch.empty((total_q, 32), dtype=torch.float32, device=q.device)

        # Iterate over segments b
        for b in range(len_indptr - 1):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())

            if q_start >= q_end or kv_start >= kv_end:
                # No work for this segment
                continue

            # Ensure inputs are contiguous and in float32 for computation
            q_batch = q[q_start:q_end].contiguous().to(torch.float32)
            k_batch = k[kv_start:kv_end].contiguous().to(torch.float32)
            v_batch = v[kv_start:kv_end].contiguous().to(torch.float32)

            Nq = q_batch.shape[0]
            Nk = k_batch.shape[0]

            # Launch Triton kernel for this segment; one program instance
            attention_forward[(1,)](
                q_batch, k_batch, v_batch, out, lse,
                Nq=Nq, Nk=Nk,
                q_start=q_start, kv_start=kv_start, q_end=q_end, kv_end=kv_end, sm_scale=sm_scale,
                BLOCK_D=128,
                num_warps=4, num_stages=2
            )

        # Cast output to bfloat16 as in the original
        output = out.to(torch.bfloat16)
        return output, lse

# Example helper (optional)
def get_inputs():
    device = 'cuda'
    q = torch.randn([1, 32, 128], dtype=torch.bfloat16, device=device)
    k = torch.randn([1, 8, 128], dtype=torch.bfloat16, device=device)
    v = torch.randn([1, 8, 128], dtype=torch.bfloat16, device=device)
    _n = 1; _t = 1
    _lens = torch.full((_n,), _t // _n, dtype=torch.int32)
    _lens[: _t % _n] += 1
    qo_indptr = torch.cat([torch.zeros(1, dtype=torch.int32), torch.cumsum(_lens, 0)]).to(torch.int32).to(device)
    _n = 1; _t = 1
    _lens = torch.full((_n,), _t // _n, dtype=torch.int32)
    _lens[: _t % _n] += 1
    kv_indptr = torch.cat([torch.zeros(1, dtype=torch.int32), torch.cumsum(_lens, 0)]).to(torch.int32).to(device)
    sm_scale = 1.0 / math.sqrt(128)  # float32 scalar
    return [q, k, v, qo_indptr, kv_indptr, sm_scale]

def fused_operator(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5):
    _out = ModelNew().forward(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5)
    return list(_out) if isinstance(_out, (tuple, list)) else [_out]


def run(*args):
    return ModelNew()(*args)
