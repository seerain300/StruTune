import math
import torch
import triton
import triton.language as tl


# Triton kernel: compute logits = q_vec @ k_rows.T
# q_vec_ptr: [HEAD_DIM] float32 (contiguous)
# k_rows_ptr: [NUM_KV, HEAD_DIM] float32, row-major contiguous
# logits_out_ptr: [NUM_KV] float32
@triton.jit
def _dot_logits_kernel(q_vec_ptr, k_rows_ptr, logits_out_ptr,
                        HEAD_DIM: tl.constexpr, NUM_KV: tl.constexpr):
    for i in tl.static_range(NUM_KV):
        acc = tl.zeros((), dtype=tl.float32)
        for j in tl.static_range(HEAD_DIM):
            qj = tl.load(q_vec_ptr + j)
            ki_j = tl.load(k_rows_ptr + i * HEAD_DIM + j)
            acc += qj * ki_j
        tl.store(logits_out_ptr + i, acc)


# Triton kernel: logsumexp over a 1D vector of length NUM_REAL
# inp_ptr: [NUM_REAL] float32
# out_ptr: [1] float32
@triton.jit
def _logsumexp_dynamic_kernel(inp_ptr, out_ptr, NUM_REAL: tl.constexpr):
    m = tl.load(inp_ptr + 0)
    for j in tl.static_range(1, NUM_REAL):
        vj = tl.load(inp_ptr + j)
        m = tl.maximum(m, vj)

    sum_exp = tl.zeros((), dtype=tl.float32)
    for j in tl.static_range(NUM_REAL):
        vj = tl.load(inp_ptr + j)
        sum_exp += tl.exp(vj - m)

    lse_val = tl.log(sum_exp) + m  # logsumexp without 1/ln(2) scaling
    tl.store(out_ptr, lse_val)


# Triton kernel: softmax over a 1D vector of length NUM_REAL
# inp_ptr: [NUM_REAL] float32
# out_ptr: [NUM_REAL] float32
@triton.jit
def _softmax_dynamic_kernel(inp_ptr, out_ptr, NUM_REAL: tl.constexpr):
    m = tl.load(inp_ptr + 0)
    for j in tl.static_range(1, NUM_REAL):
        vj = tl.load(inp_ptr + j)
        m = tl.maximum(m, vj)

    sum_exp = tl.zeros((), dtype=tl.float32)
    for j in tl.static_range(NUM_REAL):
        vj = tl.load(inp_ptr + j)
        sum_exp += tl.exp(vj - m)

    for j in tl.static_range(NUM_REAL):
        vj = tl.load(inp_ptr + j)
        out_j = tl.exp(vj - m) / sum_exp
        tl.store(out_ptr + j, out_j)


# Triton kernel: matvec out = v_rows @ attn, where v_rows is [HEAD_DIM, NUM_KV] row-major, attn is [NUM_KV]
# v_rows_ptr: [HEAD_DIM * NUM_KV] float32, row-major: row 0 at [0:HEAD_DIM), row 1 at [HEAD_DIM:2*HEAD_DIM), ...
# attn_ptr: [NUM_KV] float32
# out_ptr: [HEAD_DIM] float32
@triton.jit
def _matvec_kernel(v_rows_ptr, attn_ptr, out_ptr,
                   HEAD_DIM: tl.constexpr, NUM_KV: tl.constexpr):
    for i in tl.static_range(HEAD_DIM):
        acc = tl.zeros((), dtype=tl.float32)
        base = i * NUM_KV
        for j in tl.static_range(NUM_KV):
            acc += tl.load(v_rows_ptr + base + j) * tl.load(attn_ptr + j)
        tl.store(out_ptr + i, acc)


@torch.no_grad()
def run(q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
    total_q, num_qo_heads, head_dim = q.shape
    num_pages, _, num_kv_heads, _ = k_cache.shape  # singleton squeezed
    device = q.device
    # Cast inputs to float32 for compute
    q_f32 = q.to(torch.float32)
    k_cache_flat = k_cache.squeeze(1).to(torch.float32)  # [num_pages, 8, 128]
    v_cache_flat = v_cache.squeeze(1).to(torch.float32)  # [num_pages, 8, 128]

    # Output buffers
    output = torch.empty((total_q, num_qo_heads, head_dim), dtype=torch.bfloat16, device=device)
    lse = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=device)

    gqa_ratio = num_qo_heads // num_kv_heads

    len_indptr = qo_indptr.numel()
    assert total_q == int(qo_indptr[-1].item()), "total_q must equal the last element of qo_indptr"
    num_qo_heads = 32
    num_kv_heads = 8
    head_dim = 128

    # Iterate segments and queries
    for b in range(len_indptr - 1):
        q_start = int(qo_indptr[b].item())
        q_end = int(qo_indptr[b + 1].item())
        kv_start = int(kv_indptr[b].item())
        kv_end = int(kv_indptr[b + 1].item())

        num_q_tokens = q_end - q_start
        num_segments_b = kv_end - kv_start
        if num_q_tokens <= 0 or num_segments_b <= 0:
            continue

        # Gather kv_ids for this segment
        kv_ids = kv_indices[kv_start:kv_end].to(torch.int32).contiguous()  # [num_segments_b]

        for q_idx in range(num_q_tokens):
            global_q_idx = q_start + q_idx

            # Causal-like bound
            delta = num_segments_b - num_q_tokens
            max_kv_idx = min(q_idx + 1 + delta, num_segments_b)
            if max_kv_idx <= 0:
                # No valid KV rows; skip compute, but set outputs
                output[global_q_idx] = 0.0
                lse[global_q_idx] = 0.0
                continue

            # Query vector for this head h
            for h in range(num_qo_heads):
                kv_head = h // gqa_ratio  # map to 8 KV heads

                # q_vec: [head_dim] float32
                q_vec = q_f32[global_q_idx, h]  # already contiguous 1D in memory

                # Prepare k_rows and v_rows: [max_kv_idx, head_dim]
                # Slice k_cache_flat and v_cache_flat by kv_ids[:max_kv_idx], head kv_head
                # Build flat k_rows and v_rows
                k_rows_flat = k_cache_flat[kv_ids[:max_kv_idx], kv_head].contiguous()  # [max_kv_idx, 128]
                v_rows_flat = v_cache_flat[kv_ids[:max_kv_idx], kv_head].contiguous()  # [max_kv_idx, 128]

                # Compute logits_scaled = q_vec @ k_rows.T
                logits = torch.empty((max_kv_idx,), dtype=torch.float32, device=device)
                _dot_logits_kernel[(1,)](
                    q_vec, k_rows_flat, logits,
                    HEAD_DIM=head_dim, NUM_KV=max_kv_idx
                )
                # Scale
                logits_scaled = logits * sm_scale

                # lse = logsumexp(logits_scaled)
                lse_scalar = torch.empty((1,), dtype=torch.float32, device=device)
                _logsumexp_dynamic_kernel[(1,)](
                    logits_scaled, lse_scalar, NUM_REAL=max_kv_idx
                )
                lse_val = lse_scalar[0]
                lse[global_q_idx, h] = lse_val

                # attn = softmax(logits_scaled)
                attn = torch.empty((max_kv_idx,), dtype=torch.float32, device=device)
                _softmax_dynamic_kernel[(1,)](
                    logits_scaled, attn, NUM_REAL=max_kv_idx
                )

                # out_vec = attn @ v_rows
                # Reshape v_rows_flat to row-major [head_dim * max_kv_idx] view as 2D rows, but we have flat row-major:
                # For i in 0..head_dim-1, row i is [i*max_kv_idx + 0 : i*max_kv_idx + max_kv_idx)
                v_rows_ptr = v_rows_flat.view(-1)  # [head_dim * max_kv_idx]
                out_vec = torch.empty((head_dim,), dtype=torch.float32, device=device)
                _matvec_kernel[(1,)](
                    v_rows_ptr, attn, out_vec,
                    HEAD_DIM=head_dim, NUM_KV=max_kv_idx
                )

                # Store output as bfloat16
                output[global_q_idx, h] = out_vec.to(torch.bfloat16)

    return output, lse


def get_inputs():
    # Use CUDA tensors for evaluation
    q = torch.randn([1, 32, 128], dtype=torch.bfloat16, device='cuda')
    k_cache = torch.randn([51, 1, 8, 128], dtype=torch.bfloat16, device='cuda')
    v_cache = torch.randn([51, 1, 8, 128], dtype=torch.bfloat16, device='cuda')
    _n = 1; _t = 1
    _lens = torch.full((_n,), _t // _n, dtype=torch.int32, device='cuda')
    _lens[: _t % _n] += 1
    qo_indptr = torch.cat([torch.zeros(1, dtype=torch.int32, device='cuda'), torch.cumsum(_lens, 0)]).to(torch.int32)
    _n = 1; _t = 34
    _lens = torch.full((_n,), _t // _n, dtype=torch.int32, device='cuda')
    _lens[: _t % _n] += 1
    kv_indptr = torch.cat([torch.zeros(1, dtype=torch.int32, device='cuda'), torch.cumsum(_lens, 0)]).to(torch.int32)
    kv_indices = torch.randint(0, 51, [34], dtype=torch.int32, device='cuda')
    sm_scale = 1.0 / math.sqrt(128)  # float32 scalar
    return [q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale]


def fused_operator(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6):
    output, lse = run(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6)
    return [output, lse]


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Run Triton-optimized attention
        output, lse = run(q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale)
        return output, lse


def run(*args):
    return ModelNew()(*args)
