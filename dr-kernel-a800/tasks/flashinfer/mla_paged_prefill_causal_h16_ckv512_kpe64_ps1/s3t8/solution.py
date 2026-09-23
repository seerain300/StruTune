import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernels

@triton.jit
def compute_logits_kernel(
    qn_ptr, Kc_ptr, qp_ptr, Kp_ptr, logits_ptr,
    H, L, K, Kp, sm_scale,
    qn_stride_h, qn_stride_k,
    Kc_stride_l, Kc_stride_k,
    qp_stride_h, qp_stride_kp,
    Kp_stride_l, Kp_stride_kp,
    logits_stride_h, logits_stride_l,
):
    h = tl.program_id(0)
    # Each program computes a single row (head) of logits: logits[h, :]
    # We'll fill logits[h, l] for l in [0..L-1] by accumulating dot products.
    # We assume H is passed as meta-parameter H, but Triton scalar loop is fine.
    for l in range(0, L):
        acc = tl.zeros((), dtype=tl.float32)
        # Accumulate over K dimension: qn[h, k] * Kc[l, k]
        k = 0
        while k < K:
            qn_val = tl.load(qn_ptr + h * qn_stride_h + k * qn_stride_k)  # [H, K]
            Kc_val = tl.load(Kc_ptr + l * Kc_stride_l + k * Kc_stride_k)  # [L, K]
            acc += qn_val * Kc_val
            k += 1
        # Accumulate over Kp dimension: qp[h, k'] * Kp[l, k']
        kp_acc = tl.zeros((), dtype=tl.float32)
        kp = 0
        while kp < Kp:
            qp_val = tl.load(qp_ptr + h * qp_stride_h + kp * qp_stride_kp)  # [H, Kp]
            Kp_val = tl.load(Kp_ptr + l * Kp_stride_l + kp * Kp_stride_kp)  # [L, Kp]
            kp_acc += qp_val * Kp_val
            kp += 1
        # Sum and scale
        logits_val = (acc + kp_acc) * sm_scale
        tl.store(logits_ptr + h * logits_stride_h + l * logits_stride_l, logits_val)


@triton.jit
def compute_lse_row_kernel(
    logits_ptr, lse_ptr,
    L, inv_ln2,
    logits_stride_h, logits_stride_l,
):
    # One program computes lse for a single row (head). Triton uses program_id(0) as head index.
    h = tl.program_id(0)
    # Compute row_max and sum_exp for row h
    row_max = tl.full((), -float("inf"), tl.float32)
    for l in range(0, L):
        val = tl.load(logits_ptr + h * logits_stride_h + l * logits_stride_l)  # scalar
        # If val is -inf, it doesn't affect max. Triton handles scalar update.
        row_max = tl.maximum(row_max, val)
    # Compute sum of exp(val - row_max)
    sum_exp = tl.zeros((), dtype=tl.float32)
    for l in range(0, L):
        val = tl.load(logits_ptr + h * logits_stride_h + l * logits_stride_l)
        sum_exp += tl.exp(val - row_max)
    lse_val = row_max + tl.log(sum_exp) * inv_ln2
    tl.store(lse_ptr + h, lse_val)


@triton.jit
def softmax_row_kernel(
    logits_ptr, lse_ptr, attn_ptr,
    L, prefix_len, i, inv_ln2,
    logits_stride_h, logits_stride_l,
    attn_stride_h, attn_stride_l,
):
    h = tl.program_id(0)
    lse_val = tl.load(lse_ptr + h)
    for l in range(0, L):
        val = tl.load(logits_ptr + h * logits_stride_h + l * logits_stride_l)
        # Compute effective j: here j = l (since we process l in increasing order).
        # Apply causal mask: if j > prefix_len + i, set to -inf (effectively zero after exp).
        j = l
        if j > (prefix_len + i):
            # soft = 0
            soft = tl.zeros((), dtype=tl.float32)
        else:
            soft = tl.exp(val - lse_val)
        tl.store(attn_ptr + h * attn_stride_h + l * attn_stride_l, soft)


@triton.jit
def gemv_out_kernel(
    attn_ptr, Kc_ptr, out_ptr,
    H, L, K,
    attn_stride_h, attn_stride_l,
    Kc_stride_l, Kc_stride_k,
    out_stride_h, out_stride_k,
):
    h = tl.program_id(0)  # one program per head
    for k in range(0, K):
        acc = tl.zeros((), dtype=tl.float32)
        for l in range(0, L):
            attn_val = tl.load(attn_ptr + h * attn_stride_h + l * attn_stride_l)
            Kc_val = tl.load(Kc_ptr + l * Kc_stride_l + k * Kc_stride_k)
            acc += attn_val * Kc_val
        tl.store(out_ptr + h * out_stride_h + k * out_stride_k, acc)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Ensure Triton/CUDA
        if not TRITON_AVAILABLE or not torch.cuda.is_available():
            raise RuntimeError("Triton or CUDA not available for Triton version.")

        device = q_nope.device
        q_nope = q_nope.to(device).contiguous()
        q_pe = q_pe.to(device).contiguous()
        ckv_cache = ckv_cache.to(device).contiguous()
        kpe_cache = kpe_cache.to(device).contiguous()
        qo_indptr = qo_indptr.to(device).contiguous()
        kv_indptr = kv_indptr.to(device).contiguous()
        kv_indices = kv_indices.to(device).contiguous()

        # Prepare squeezed caches
        Kc_all = ckv_cache.squeeze(1)  # [P, K]
        Kp_all = kpe_cache.squeeze(1)  # [P, Kp]

        total_q = q_nope.shape[0]
        H = q_nope.shape[1]
        K = q_nope.shape[2]
        Kp = q_pe.shape[2]

        # Output buffers
        output = torch.empty((total_q, H, K), dtype=torch.bfloat16, device=device)
        lse = torch.full((total_q, H), -float("inf"), dtype=torch.float32, device=device)

        # Loop over batch elements from qo_indptr
        # Note: len(qo_indptr) can be batch_size + 1; we iterate until last element
        for b in range(qo_indptr.shape[0] - 1):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            q_len = q_end - q_start

            if q_len == 0:
                continue

            # Token indices for this batch
            tok_start = int(kv_indptr[b].item())
            tok_end = int(kv_indptr[b + 1].item())
            L = tok_end - tok_start
            if L == 0:
                continue

            # Gather cached tokens
            Kc_curr = Kc_all[tok_start:tok_end].contiguous()  # [L, K], float32
            Kp_curr = Kp_all[tok_start:tok_end].contiguous()  # [L, Kp], float32

            # Loop over queries i in this batch element
            for i in range(q_len):
                q_start_i = q_start + i

                # Prepare qn and qp (convert to fp32 for Triton)
                qn = q_nope[q_start_i].to(torch.float32).contiguous()  # [H, K]
                qp = q_pe[q_start_i].to(torch.float32).contiguous()    # [H, Kp]

                # Allocate per-head buffers
                logits_scaled = torch.empty((H, L), dtype=torch.float32, device=device)

                # Kernel 1: compute logits per head
                grid = (H,)
                compute_logits_kernel[grid](
                    qn, Kc_curr, qp, Kp_curr, logits_scaled,
                    H, L, K, Kp, sm_scale,
                    qn.stride(0), qn.stride(1),
                    Kc_curr.stride(0), Kc_curr.stride(1),
                    qp.stride(0), qp.stride(1),
                    Kp_curr.stride(0), Kp_curr.stride(1),
                    logits_scaled.stride(0), logits_scaled.stride(1),
                )

                # Kernel 2: compute lse per head (robust with max and sum)
                lse_vec = torch.empty((H,), dtype=torch.float32, device=device)
                inv_ln2 = 1.0 / math.log(2.0)
                compute_lse_row_kernel[grid](
                    logits_scaled, lse_vec,
                    L, inv_ln2,
                    logits_scaled.stride(0), logits_scaled.stride(1),
                )

                # Kernel 3: softmax with causal mask per head
                attn = torch.empty((H, L), dtype=torch.float32, device=device)
                prefix_len = L - q_len  # number of tokens before this query in this batch
                softmax_row_kernel[grid](
                    logits_scaled, lse_vec, attn,
                    L, prefix_len, i, inv_ln2,
                    logits_scaled.stride(0), logits_scaled.stride(1),
                    attn.stride(0), attn.stride(1),
                )

                # Kernel 4: GEMV out[h, k] = attn[h, :] @ Kc_curr[l, k]
                out_h = torch.empty((H, K), dtype=torch.float32, device=device)
                gemv_out_kernel[grid](
                    attn, Kc_curr, out_h,
                    H, L, K,
                    attn.stride(0), attn.stride(1),
                    Kc_curr.stride(0), Kc_curr.stride(1),
                    out_h.stride(0), out_h.stride(1),
                )

                # Store outputs
                output[q_start_i] = out_h.to(torch.bfloat16)
                lse[q_start_i] = lse_vec  # lse is per head, store as [H]

        return output, lse

# Helper to get inputs (same as original)
def get_inputs():
    q_nope = torch.randn([1, 16, 512], dtype=torch.bfloat16)
    q_pe = torch.randn([1, 16, 64], dtype=torch.bfloat16)
    ckv_cache = torch.randn([989669, 1, 512], dtype=torch.bfloat16)
    kpe_cache = torch.randn([989669, 1, 64], dtype=torch.bfloat16)
    _n = 1; _t = 1
    _lens = torch.full((_n,), _t // _n, dtype=torch.int32)
    _lens[: _t % _n] += 1
    qo_indptr = torch.cat([torch.zeros(1, dtype=torch.int32), torch.cumsum(_lens, 0)]).to(torch.int32)
    _n = 1; _t = 34
    _lens = torch.full((_n,), _t // _n, dtype=torch.int32)
    _lens[: _t % _n] += 1
    kv_indptr = torch.cat([torch.zeros(1, dtype=torch.int32), torch.cumsum(_lens, 0)]).to(torch.int32)
    kv_indices = torch.randint(0, 989669, [34], dtype=torch.int32)
    sm_scale = 1.0  # float32 scalar
    return [q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale]


# Optional: provide the fused call, as in the original
class Model(torch.nn.Module):
    def forward(self, *args):
        return ModelNew().forward(*args)

# If you want to run the provided inputs:
# model = Model().cuda()
# q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale = get_inputs()
# q_nope = q_nope.cuda(), q_pe = q_pe.cuda(), ckv_cache = ckv_cache.cuda(), kpe_cache = kpe_cache.cuda()
# qo_indptr = qo_indptr.cuda(), kv_indptr = kv_indptr.cuda(), kv_indices = kv_indices.cuda()
# output, lse = model(q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale)
# print(output.shape, lse.shape)


def run(*args):
    return ModelNew()(*args)
