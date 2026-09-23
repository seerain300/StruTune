import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: compute logits vector for one head h:
# logits[h, j] = sum_k qn[h, k] * Kc[j, k] + sum_k' qp[h, k'] * Kp[j, k'], for j in [0..L-1]
@triton.jit
def compute_logits_kernel(
    qn_ptr, Kc_ptr, qp_ptr, Kp_ptr,
    logits_ptr,
    H: tl.constexpr, L: tl.constexpr, K: tl.constexpr, Kp: tl.constexpr,
    sm_scale,  # float32 scalar
    stride_qn_h, stride_qn_k,
    stride_Kc_j, stride_Kc_k,
    stride_qp_h, stride_qp_kp,
    stride_Kp_j, stride_Kp_kp,
    stride_log_h, stride_log_j
):
    h = tl.program_id(0)  # one program per head
    # Accumulate logits for each position j
    j = 0
    while j < L:
        acc = tl.zeros((), dtype=tl.float32)
        # First part: qn @ Kc.T
        k = 0
        while k < K:
            q_val = tl.load(qn_ptr + h * stride_qn_h + k * stride_qn_k)
            kc_val = tl.load(Kc_ptr + j * stride_Kc_j + k * stride_Kc_k)
            acc += q_val * kc_val
            k += 1
        # Second part: qp @ Kp.T
        kp = 0
        while kp < Kp:
            q_val = tl.load(qp_ptr + h * stride_qp_h + kp * stride_qp_kp)
            kp_val = tl.load(Kp_ptr + j * stride_Kp_j + kp * stride_Kp_kp)
            acc += q_val * kp_val
            kp += 1
        acc = acc * sm_scale
        tl.store(logits_ptr + h * stride_log_h + j * stride_log_j, acc)
        j += 1


# Triton kernel: compute logsumexp over a row vector (length L), output to lse_ptr[h]
# We assume logits_ptr has row of length L for this h, already scaled by sm_scale.
@triton.jit
def compute_lse_row_kernel(
    logits_ptr,
    lse_ptr,
    L: tl.constexpr,
    inv_ln2: tl.constexpr,  # 1/ln(2)
    stride_log_h, stride_log_j
):
    h = tl.program_id(0)
    # Compute row_max over the entire row
    row_max = -1e30  # initialize
    j = 0
    while j < L:
        val = tl.load(logits_ptr + h * stride_log_h + j * stride_log_j)
        row_max = tl.maximum(row_max, val)
        j += 1

    # Compute sum_exp = sum(exp(val - row_max))
    sum_exp = 0.0
    j = 0
    while j < L:
        val = tl.load(logits_ptr + h * stride_log_h + j * stride_log_j)
        sum_exp += tl.exp(val - row_max)
        j += 1

    lse = row_max + tl.log(sum_exp) * inv_ln2
    tl.store(lse_ptr + h, lse)


# Triton kernel: softmax on a row vector with causal masking j > (prefix_len + i)
# logits_scaled_ptr contains sm_scaled logits (previously scaled and masked); we compute attn
@triton.jit
def softmax_row_kernel(
    logits_scaled_ptr, attn_ptr,
    lse_ptr,  # pointer to lse for this head
    L: tl.constexpr, prefix_len: tl.constexpr, i: tl.constexpr,
    inv_ln2: tl.constexpr,
    stride_log_h, stride_log_j,
    stride_attn_h, stride_attn_j
):
    h = tl.program_id(0)
    # Load lse for this head
    lse = tl.load(lse_ptr + h)
    # Initialize attn vector
    j = 0
    while j < L:
        val = tl.load(logits_scaled_ptr + h * stride_log_h + j * stride_log_j)
        # Apply causal mask: if j > prefix_len + i, set to -inf; else keep val
        if (j > (prefix_len + i)):
            soft = 0.0  # since exp(-inf) = 0, we can set 0 here directly
        else:
            soft = tl.exp(val - lse)
        tl.store(attn_ptr + h * stride_attn_h + j * stride_attn_j, soft)
        j += 1


# Triton kernel: GEMV out[h, K] = attn[h, :] @ Kc_curr[0..L-1, :] -> output is fp32
@triton.jit
def gemv_out_kernel(
    attn_ptr, Kc_ptr, out_ptr,
    L: tl.constexpr, K: tl.constexpr,
    stride_attn_h, stride_attn_j,
    stride_Kc_j, stride_Kc_k,
    stride_out_h, stride_out_k
):
    h = tl.program_id(0)
    # Compute out[h, k] = sum_j attn[h, j] * Kc[j, k]
    k = 0
    while k < K:
        acc = tl.zeros((), dtype=tl.float32)
        j = 0
        while j < L:
            a = tl.load(attn_ptr + h * stride_attn_h + j * stride_attn_j)
            kc = tl.load(Kc_ptr + j * stride_Kc_j + k * stride_Kc_k)
            acc += a * kc
            j += 1
        tl.store(out_ptr + h * stride_out_h + k * stride_out_k, acc)
        k += 1


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Triton-only: ensure CUDA and Triton
        if not TRITON_AVAILABLE or not torch.cuda.is_available():
            raise RuntimeError("Triton or CUDA not available for Triton version.")

        device = q_nope.device
        H = 16  # num_qo_heads
        K = 512  # head_dim_ckv
        Kp = 64  # head_dim_kpe

        # Prepare caches: squeeze dim=1 to remove the batch of 1
        Kc_all = ckv_cache.squeeze(1).to(torch.float32).contiguous()  # [num_pages, K]
        Kp_all = kpe_cache.squeeze(1).to(torch.float32).contiguous()  # [num_pages, Kp]

        # Output tensors
        total_q = int(qo_indptr[-1].item())
        output = torch.empty((total_q, H, K), dtype=torch.bfloat16, device=device)
        lse = torch.full((total_q, H), -float("inf"), dtype=torch.float32, device=device)

        # Loop over batch elements
        b = 0
        while b < (len(qo_indptr) - 1):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            q_len = q_end - q_start

            if q_len == 0:
                b += 1
                continue

            # Compute token indices and gather Kc, Kp for this batch
            tok_start = int(kv_indptr[b].item())
            tok_end = int(kv_indptr[b + 1].item())
            L = tok_end - tok_start
            if L == 0:
                b += 1
                continue

            # Slice caches for this batch
            Kc_curr = Kc_all[tok_start:tok_end].contiguous()  # [L, K]
            Kp_curr = Kp_all[tok_start:tok_end].contiguous()  # [L, Kp]

            # Loop over queries i in this batch element
            for i in range(q_len):
                q_start_i = q_start + i

                # Prepare qn and qp for this query (convert to fp32 for Triton)
                qn = q_nope[q_start_i].to(torch.float32).contiguous()  # [H, K]
                qp = q_pe[q_start_i].to(torch.float32).contiguous()    # [H, Kp]

                # Allocate per-head buffers
                logits = torch.empty((H, L), dtype=torch.float32, device=device)  # logits per head
                # Launch compute_logits_kernel: one program per head
                grid = (H,)
                compute_logits_kernel[grid](
                    qn, Kc_curr, qp, Kp_curr, logits,
                    H, L, K, Kp, sm_scale,
                    qn.stride(0), qn.stride(1),
                    Kc_curr.stride(0), Kc_curr.stride(1),
                    qp.stride(0), qp.stride(1),
                    Kp_curr.stride(0), Kp_curr.stride(1),
                    logits.stride(0), logits.stride(1),
                )

                # Compute lse per head
                inv_ln2 = 1.0 / math.log(2.0)
                lse_vec = torch.empty((H,), dtype=torch.float32, device=device)
                compute_lse_row_kernel[grid](
                    logits, lse_vec,
                    L, inv_ln2,
                    logits.stride(0), logits.stride(1),
                )

                # Softmax with causal mask per head
                attn = torch.empty((H, L), dtype=torch.float32, device=device)
                prefix_len = L - q_len  # tokens before this query in this batch
                softmax_row_kernel[grid](
                    logits, attn, lse_vec,
                    L, prefix_len, i, inv_ln2,
                    logits.stride(0), logits.stride(1),
                    attn.stride(0), attn.stride(1),
                )

                # GEMV: out[h, K] = attn[h, :] @ Kc_curr.T (fp32)
                out_h = torch.empty((H, K), dtype=torch.float32, device=device)
                gemv_out_kernel[grid](
                    attn, Kc_curr, out_h,
                    L, K,
                    attn.stride(0), attn.stride(1),
                    Kc_curr.stride(0), Kc_curr.stride(1),
                    out_h.stride(0), out_h.stride(1),
                )

                # Store output as bfloat16
                output[q_start_i] = out_h.to(torch.bfloat16)
                # Store lse
                lse[q_start_i] = lse_vec

            b += 1

        return output, lse


# Optional helpers (same as original)
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


# Fused operator matching the original signature
def fused_operator(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6, tensor_7):
    _out = ModelNew().forward(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6, tensor_7)
    return list(_out) if isinstance(_out, (tuple, list)) else [_out]


def run(*args):
    return ModelNew()(*args)
