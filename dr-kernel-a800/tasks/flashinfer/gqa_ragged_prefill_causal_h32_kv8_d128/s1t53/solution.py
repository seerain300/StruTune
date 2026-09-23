import torch
import math

# Triton imports
import triton
import triton.language as tl


# Kernel 1: compute logits = Q @ K^T, shape [Q, H, K], where K = num_kv_tokens
# We assume BLOCK_Q=1, BLOCK_K=64, BLOCK_D=128; no runtime loops.
@triton.jit
def compute_logits_kernel(
    Q, KEXP, LOGITS,
    num_q_tokens, num_kv_tokens, head_dim,
    Q_stride_q, Q_stride_h, Q_stride_d,
    KEXP_stride_k, KEXP_stride_h, KEXP_stride_d,
    LOGITS_stride_q, LOGITS_stride_h, LOGITS_stride_k,
    BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_D: tl.constexpr
):
    # Grid: (ceil(num_q_tokens, BLOCK_Q), heads)
    pid_q = tl.program_id(0)
    h = tl.program_id(1)

    # Each program handles one query position q
    q = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)  # [1]
    q_mask = q < num_q_tokens

    # Accumulator for logits[q, k] over d
    acc = tl.zeros((BLOCK_Q, BLOCK_K), dtype=tl.float32)

    # Loop over k in tiles of BLOCK_K
    for k0 in range(0, 128, BLOCK_K):
        k_idx = k0 + tl.arange(0, BLOCK_K)  # [BLOCK_K]
        k_mask = k_idx < num_kv_tokens

        # Compute Q_vals: [BLOCK_K] = sum_d Q[q, h, d] * KEXP[k, h, d]
        # We need to reduce over d in tiles of BLOCK_D
        for d0 in range(0, 128, BLOCK_D):
            d_idx = d0 + tl.arange(0, BLOCK_D)  # [BLOCK_D]
            d_valid = d_idx < head_dim

            # Load Q[q, h, d] as [BLOCK_D]
            Q_ptrs = Q + q * Q_stride_q + h * Q_stride_h + d_idx * Q_stride_d
            Q_vals_d = tl.load(Q_ptrs, mask=d_valid, other=0.0)  # [BLOCK_D], float32

            # Load KEXP[k, h, d] as [BLOCK_K, BLOCK_D]
            KEXP_ptrs = KEXP + k_idx[:, None] * KEXP_stride_k + h * KEXP_stride_h + d_idx[None, :] * KEXP_stride_d
            K_vals = tl.load(KEXP_ptrs, mask=(k_mask[:, None] & d_valid[None, :]), other=0.0)  # [BLOCK_K, BLOCK_D]

            # Accumulate: acc += sum over d of Q_vals_d * K_vals along d
            # Q_vals_d is [BLOCK_D]; broadcast with K_vals [BLOCK_K, BLOCK_D]
            acc += tl.sum(Q_vals_d[None, :] * K_vals, axis=1)  # sum over D -> shape [BLOCK_K]

        # Store acc to LOGITS[q, h, k]
        LOGITS_ptrs = LOGITS + q * LOGITS_stride_q + h * LOGITS_stride_h + k_idx * LOGITS_stride_k
        tl.store(LOGITS_ptrs, acc, mask=k_mask)  # [BLOCK_K]


# Kernel 2: compute LSE with causal mask
@triton.jit
def _lse_masked_kernel(
    LOGITS, LSE,
    num_q_tokens, num_kv_tokens, head_dim,
    LOGITS_stride_q, LOGITS_stride_h, LOGITS_stride_k,
    LSE_stride_q, LSE_stride_h,
    BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr
):
    # Grid: (ceil(num_q_tokens, BLOCK_Q), heads)
    pid_q = tl.program_id(0)
    h = tl.program_id(1)

    q = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)  # [1]
    q_mask = q < num_q_tokens

    # Compute max over K for numerical stability
    max_vals = tl.full((BLOCK_Q,), -float("inf"), dtype=tl.float32)
    for k0 in range(0, 128, BLOCK_K):
        k_idx = k0 + tl.arange(0, BLOCK_K)  # [BLOCK_K]
        k_mask = k_idx < num_kv_tokens

        LOGITS_ptrs = LOGITS + q * LOGITS_stride_q + h * LOGITS_stride_h + k_idx * LOGITS_stride_k
        vals = tl.load(LOGITS_ptrs, mask=(q_mask & k_mask), other=-float("inf"))  # [BLOCK_K]
        max_vals = tl.maximum(max_vals, vals)

    # Compute sum exp over K with causal mask
    sum_exp = tl.zeros((BLOCK_Q,), dtype=tl.float32)
    for k0 in range(0, 128, BLOCK_K):
        k_idx = k0 + tl.arange(0, BLOCK_K)
        k_mask = k_idx < num_kv_tokens
        q_vec = q  # [1]
        allowed = k_idx < (q_vec + 1)  # [BLOCK_K]
        LOGITS_ptrs = LOGITS + q_vec * LOGITS_stride_q + h * LOGITS_stride_h + k_idx * LOGITS_stride_k
        vals = tl.load(LOGITS_ptrs, mask=(k_mask & allowed), other=-float("inf"))  # [BLOCK_K]
        vals = vals - max_vals  # broadcast to [1, BLOCK_K]
        exp_vals = tl.exp(vals)
        sum_exp += tl.sum(exp_vals, axis=0)  # reduce over K

    lse_vals = max_vals + tl.log(sum_exp)  # [1]
    LSE_ptrs = LSE + q * LSE_stride_q + h * LSE_stride_h
    tl.store(LSE_ptrs, lse_vals, mask=q_mask)  # store per (q,h)


# Kernel 3: softmax over K with causal mask, then output = softmax @ V_exp
@triton.jit
def _softmax_output_kernel(
    LOGITS, VEXP, LSE, OUT,
    num_q_tokens, num_kv_tokens, head_dim,
    LOGITS_stride_q, LOGITS_stride_h, LOGITS_stride_k,
    VEXP_stride_k, VEXP_stride_h, VEXP_stride_d,
    OUT_stride_q, OUT_stride_h, OUT_stride_d,
    BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_D: tl.constexpr
):
    # Grid: (ceil(num_q_tokens, BLOCK_Q), heads)
    pid_q = tl.program_id(0)
    h = tl.program_id(1)

    q = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)  # [1]
    q_mask = q < num_q_tokens

    # Load lse for this (q,h): [1]
    LSE_ptrs = LSE + q * LSE_stride_q + h * LSE_stride_h
    lse_vals = tl.load(LSE_ptrs, mask=q_mask, other=-float("inf"))  # [1]

    # For each d tile, compute output[q, h, d] = sum_k softmax(LOGITS) * VEXP[k, h, d]
    for d0 in range(0, 128, BLOCK_D):
        d_idx = d0 + tl.arange(0, BLOCK_D)
        d_valid = d_idx < head_dim

        OUT_ptrs = OUT + q * OUT_stride_q + h * OUT_stride_h + d_idx * OUT_stride_d
        out_row = tl.zeros((BLOCK_Q, BLOCK_D), dtype=tl.float32)

        for k0 in range(0, 128, BLOCK_K):
            k_idx = k0 + tl.arange(0, BLOCK_K)
            k_mask = k_idx < num_kv_tokens

            q_vec = q  # [1]
            allowed = k_idx < (q_vec + 1)  # [BLOCK_K]

            LOGITS_ptrs = LOGITS + q_vec * LOGITS_stride_q + h * LOGITS_stride_h + k_idx * LOGITS_stride_k
            vals = tl.load(LOGITS_ptrs, mask=(k_mask & allowed), other=-float("inf"))  # [BLOCK_K]
            vals = vals - lse_vals  # [1, BLOCK_K] after broadcasting
            exp_vals = tl.exp(vals)                  # [BLOCK_K]
            sum_exp = tl.sum(exp_vals, axis=0)      # [1]
            probs = exp_vals / sum_exp              # [BLOCK_K]

            VEXP_ptrs = VEXP + k_idx * VEXP_stride_k + h * VEXP_stride_h + d_idx * VEXP_stride_d
            V_vals = tl.load(VEXP_ptrs, mask=(k_mask & d_valid), other=0.0)  # [BLOCK_K, BLOCK_D]

            out_row += probs[:, None] * V_vals  # [BLOCK_D]

        tl.store(OUT_ptrs, out_row, mask=(q_mask[:, None] & d_valid[None, :]))


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Fixed tiling parameters (compile-time constants for Triton)
        self.BLOCK_Q = 1
        self.BLOCK_K = 64
        self.BLOCK_D = 128

    def forward(self, q, k, v, qo_indptr, kv_indptr, sm_scale):
        # Shapes and assertions
        total_q, num_qo_heads, head_dim = q.shape
        total_kv, num_kv_heads, _ = k.shape
        assert num_qo_heads == 32
        assert num_kv_heads == 8
        assert head_dim == 128

        # Parse segments
        len_indptr = qo_indptr.numel()
        assert total_q == int(qo_indptr[-1].item())
        assert total_kv == int(kv_indptr[-1].item())

        device = q.device
        output = torch.empty((total_q, num_qo_heads, head_dim), dtype=torch.bfloat16, device=device)
        lse = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=device)

        # GQA: repeat K/V by 4
        k_exp = k.repeat_interleave(4, dim=1)
        v_exp = v.repeat_interleave(4, dim=1)

        # For each batch segment
        for b in range(len_indptr - 1):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())

            if q_start >= q_end or kv_start >= kv_end:
                continue

            num_q_tokens = q_end - q_start
            num_kv_tokens = kv_end - kv_start

            # Segment tensors
            q_batch = q[q_start:q_end]
            k_batch = k_exp[kv_start:kv_end]
            v_batch = v_exp[kv_start:kv_end]

            # Ensure float32 for Triton
            q_batch = q_batch.to(torch.float32)
            k_batch = k_batch.to(torch.float32)
            v_batch = v_batch.to(torch.float32)

            # Output segment and logits buffer [Q, H=32, K] with K=num_kv_tokens
            out_seg = torch.empty((num_q_tokens, num_qo_heads, head_dim), dtype=torch.bfloat16, device=device)
            logits_seg = torch.empty((num_q_tokens, num_qo_heads, num_kv_tokens), dtype=torch.float32, device=device)

            # Launch Triton kernels
            # 1) Compute logits
            _ = compute_logits_kernel[(triton.cdiv(num_q_tokens, self.BLOCK_Q), 32)](
                q_batch, k_batch, logits_seg,
                num_q_tokens, num_kv_tokens, head_dim,
                q_batch.stride(0), q_batch.stride(1), q_batch.stride(2),
                k_batch.stride(0), k_batch.stride(1), k_batch.stride(2),
                logits_seg.stride(0), logits_seg.stride(1), logits_seg.stride(2),
                BLOCK_Q=self.BLOCK_Q, BLOCK_K=self.BLOCK_K, BLOCK_D=self.BLOCK_D
            )

            # 2) LSE with causal mask
            _lse_masked_kernel[(triton.cdiv(num_q_tokens, self.BLOCK_Q), 32)](
                logits_seg, lse,
                num_q_tokens, num_kv_tokens, head_dim,
                logits_seg.stride(0), logits_seg.stride(1), logits_seg.stride(2),
                lse.stride(0), lse.stride(1),
                BLOCK_Q=self.BLOCK_Q, BLOCK_K=self.BLOCK_K
            )

            # 3) Softmax + output
            _softmax_output_kernel[(triton.cdiv(num_q_tokens, self.BLOCK_Q), 32)](
                logits_seg, v_batch, lse, out_seg,
                num_q_tokens, num_kv_tokens, head_dim,
                logits_seg.stride(0), logits_seg.stride(1), logits_seg.stride(2),
                v_batch.stride(0), v_batch.stride(1), v_batch.stride(2),
                out_seg.stride(0), out_seg.stride(1), out_seg.stride(2),
                BLOCK_Q=self.BLOCK_Q, BLOCK_K=self.BLOCK_K, BLOCK_D=self.BLOCK_D
            )

            # Store segment to global output
            output[q_start:q_end] = out_seg

        return output, lse


# Example helpers (optional, not used in evaluation):
def get_inputs():
    q = torch.randn([1, 32, 128], dtype=torch.bfloat16, device='cuda')
    k = torch.randn([1, 8, 128], dtype=torch.bfloat16, device='cuda')
    v = torch.randn([1, 8, 128], dtype=torch.bfloat16, device='cuda')
    _n = 1; _t = 1
    _lens = torch.full((_n,), _t // _n, dtype=torch.int32, device='cuda')
    _lens[: _t % _n] += 1
    qo_indptr = torch.cat([torch.zeros(1, dtype=torch.int32, device='cuda'), torch.cumsum(_lens, 0)]).to(torch.int32)
    _n = 1; _t = 1
    _lens = torch.full((_n,), _t // _n, dtype=torch.int32, device='cuda')
    _lens[: _t % _n] += 1
    kv_indptr = torch.cat([torch.zeros(1, dtype=torch.int32, device='cuda'), torch.cumsum(_lens, 0)]).to(torch.int32)
    sm_scale = 1.0 / math.sqrt(128)
    return [q, k, v, qo_indptr, kv_indptr, sm_scale]


def fused_operator(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5):
    _out = ModelNew()(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5)
    return list(_out) if isinstance(_out, (tuple, list)) else [_out]


def run(*args):
    return ModelNew()(*args)
