import math
import torch
import triton
import triton.language as tl


@triton.jit
def attention_gqa_two_segments_two_i_kernel(
    q_ptr, k_exp_ptr, v_exp_ptr, out_ptr, lse_ptr,
    qo_indptr_ptr, kv_indptr_ptr,
    total_q, total_kv, sm_scale,
    NUM_SEGMENTS: tl.constexpr,  # must be 2
):
    """
    Triton kernel: process two segments (b=0 and b=1). For each segment:
      - handle two query positions i=0 and i=1, and all 32 query heads h in 0..31.
      - Compute logits per (i, h) across 8 kv heads (j=0..7), apply forward-causal mask.
      - Compute base-2 lse, softmax, and accumulate output[i, h, :] = sum_j softmax[j] * v_exp[:, j, :].
    Grid: one program per (segment, i, h) i.e., 2 * 2 * 32 = 128 programs.
    """
    # Decode program id: axis=0 is flattened over segment and i, axis=1 over h
    # We use a single axis to keep Triton simple; map pid -> (seg, i, h).
    pid = tl.program_id(axis=0)
    num_segments_i = NUM_SEGMENTS * 2
    seg = pid // (2 * 32)
    i = (pid % (2 * 32)) // 32
    h = pid % 32

    # Segment 0
    if seg == 0:
        qo0 = tl.load(qo_indptr_ptr + 0)  # qo_indptr[0]
        qo1 = tl.load(qo_indptr_ptr + 1)  # qo_indptr[1]
        kv00 = tl.load(kv_indptr_ptr + 0)  # kv_indptr[0]
        kv01 = tl.load(kv_indptr_ptr + 1)  # kv_indptr[1]
        Nq0 = qo1 - qo0
        Nk0 = kv01 - kv00

        if (Nq0 > 0) and (i < Nq0):
            delta0 = Nk0 - Nq0

            # Load q[i, h, :]
            q_base0 = (qo0 + i) * 32 * 128 + h * 128
            q_vec = tl.load(q_ptr + q_base0)  # [128] float32

            # Compute logits across 8 kv heads (j=0..7), apply causal mask
            logits0 = tl.zeros((8,), dtype=tl.float32)
            for j in range(8):
                orig_h = h % 8
                acc = 0.0
                t = 0
                while t < Nk0:
                    k_row_ptr = (kv00 + t) * 8 * 128 + orig_h * 128
                    k_vec = tl.load(k_exp_ptr + k_row_ptr)  # [128]
                    acc += tl.sum(q_vec * k_vec)
                    t += 1
                acc *= sm_scale
                allow = j < (i + 1 + delta0)
                logits0[j] = tl.where(allow, acc, -1e20)

            # Base-2 logsumexp
            m = tl.max(logits0, axis=0)
            sum_exp = tl.sum(tl.exp(logits0 - m), axis=0)
            lse_val0 = m + tl.log(sum_exp) / 0.6931471805599453  # ln(2)

            # Softmax across 8 positions
            soft0 = tl.exp(logits0 - lse_val0)  # [8] float32

            # Accumulate output[i, h, :] = sum_j soft0[j] * v_exp[:, j, :]
            out_base0 = (qo0 + i) * 32 * 128 + h * 128
            for j in range(8):
                orig_h = h % 8
                t = 0
                while t < Nk0:
                    v_row_ptr = (kv00 + t) * 32 * 128 + orig_h * 128
                    v_vec = tl.load(v_exp_ptr + v_row_ptr)  # [128]
                    current = tl.load(out_ptr + out_base0)
                    current += v_vec * soft0[j]
                    tl.store(out_ptr + out_base0, current)
                    t += 1

            # Store lse[i, h] for segment 0
            lse_index0 = (qo0 + i) * 32 + h
            tl.store(lse_ptr + lse_index0, lse_val0)

    # Segment 1
    if seg == 1:
        qo1 = tl.load(qo_indptr_ptr + 1)  # qo_indptr[1]
        qo2 = tl.load(qo_indptr_ptr + 2)  # qo_indptr[2]
        kv10 = tl.load(kv_indptr_ptr + 1)  # kv_indptr[1]
        kv11 = tl.load(kv_indptr_ptr + 2)  # kv_indptr[2]
        Nq1 = qo2 - qo1
        Nk1 = kv11 - kv10

        if (Nq1 > 0) and (i < Nq1):
            delta1 = Nk1 - Nq1

            # Load q[i, h, :]
            q_base1 = (qo1 + i) * 32 * 128 + h * 128
            q_vec = tl.load(q_ptr + q_base1)  # [128] float32

            # Compute logits across 8 kv heads (j=0..7), apply causal mask
            logits1 = tl.zeros((8,), dtype=tl.float32)
            for j in range(8):
                orig_h = h % 8
                acc = 0.0
                t = 0
                while t < Nk1:
                    k_row_ptr = (kv10 + t) * 8 * 128 + orig_h * 128
                    k_vec = tl.load(k_exp_ptr + k_row_ptr)  # [128]
                    acc += tl.sum(q_vec * k_vec)
                    t += 1
                acc *= sm_scale
                allow = j < (i + 1 + delta1)
                logits1[j] = tl.where(allow, acc, -1e20)

            # Base-2 logsumexp
            m = tl.max(logits1, axis=0)
            sum_exp = tl.sum(tl.exp(logits1 - m), axis=0)
            lse_val1 = m + tl.log(sum_exp) / 0.6931471805599453  # ln(2)

            # Softmax across 8 positions
            soft1 = tl.exp(logits1 - lse_val1)  # [8] float32

            # Accumulate output[i, h, :] = sum_j soft1[j] * v_exp[:, j, :]
            out_base1 = (qo1 + i) * 32 * 128 + h * 128
            for j in range(8):
                orig_h = h % 8
                t = 0
                while t < Nk1:
                    v_row_ptr = (kv10 + t) * 32 * 128 + orig_h * 128
                    v_vec = tl.load(v_exp_ptr + v_row_ptr)  # [128]
                    current = tl.load(out_ptr + out_base1)
                    current += v_vec * soft1[j]
                    tl.store(out_ptr + out_base1, current)
                    t += 1

            # Store lse[i, h] for segment 1
            lse_index1 = (qo1 + i) * 32 + h
            tl.store(lse_ptr + lse_index1, lse_val1)


class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, qo_indptr, kv_indptr, sm_scale):
        """
        q: [total_q, 32, 128], bfloat16
        k: [total_kv, 8, 128], bfloat16
        v: [total_kv, 8, 128], bfloat16
        qo_indptr: [len_indptr] int32, typically [a0, a1] for two segments
        kv_indptr: [len_indptr] int32, typically [b0, b1] for two segments
        sm_scale: float32 (e.g., 1/sqrt(128))
        Returns:
        - output: [total_q, 32, 128] float32 (to be cast to bfloat16)
        - lse: [total_q, 32] float32 (base-2 logsumexp)
        """
        device = q.device
        total_q, num_qo_heads, head_dim = q.shape
        total_kv, num_kv_heads, _ = k.shape
        assert num_qo_heads == 32 and num_kv_heads == 8 and head_dim == 128

        # Expand k and v to 32 heads (GQA ratio = 4)
        k_expanded = k.repeat_interleave(4, dim=1).contiguous().to(torch.float32)
        v_expanded = v.repeat_interleave(4, dim=1).contiguous().to(torch.float32)

        # Cast q to float32
        q_f32 = q.to(torch.float32).contiguous()

        # Output and lse initialization
        output = torch.zeros((total_q, num_qo_heads, head_dim), dtype=torch.float32, device=device)
        lse = torch.full((total_q, num_qo_heads), -float("inf"), dtype=torch.float32, device=device)

        # Launch Triton kernel: grid = (NUM_SEGMENTS * 2 * 32,)
        grid = (2 * 2 * 32,)  # segment=2, i in {0,1}, h in {0..31}
        attention_gqa_two_segments_two_i_kernel[grid](
            q_f32, k_expanded, v_expanded, output, lse,
            qo_indptr, kv_indptr,
            total_q, total_kv, sm_scale,
            NUM_SEGMENTS=2,
        )

        # Cast output to bfloat16 as in original
        output_bf16 = output.to(torch.bfloat16)
        return output_bf16, lse

# Helper functions
def get_inputs():
    q = torch.randn([1, 32, 128], dtype=torch.bfloat16)
    k = torch.randn([1, 8, 128], dtype=torch.bfloat16)
    v = torch.randn([1, 8, 128], dtype=torch.bfloat16)
    _n = 1; _t = 1
    _lens = torch.full((_n,), _t // _n, dtype=torch.int32)
    _lens[: _t % _n] += 1
    qo_indptr = torch.cat([torch.zeros(1, dtype=torch.int32), torch.cumsum(_lens, 0)]).to(torch.int32)
    _n = 1; _t = 1
    _lens = torch.full((_n,), _t // _n, dtype=torch.int32)
    _lens[: _t % _n] += 1
    kv_indptr = torch.cat([torch.zeros(1, dtype=torch.int32), torch.cumsum(_lens, 0)]).to(torch.int32)
    sm_scale = 1.0 / math.sqrt(128)
    return [q, k, v, qo_indptr, kv_indptr, sm_scale]

def fused_operator(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5):
    _out = ModelNew().forward(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5)
    return list(_out) if isinstance(_out, (tuple, list)) else [_out]


def run(*args):
    return ModelNew()(*args)
