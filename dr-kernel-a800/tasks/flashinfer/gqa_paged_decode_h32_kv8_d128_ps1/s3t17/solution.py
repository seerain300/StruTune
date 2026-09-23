import math
import torch
import triton
import triton.language as tl


@triton.jit
def lse_and_max_kernel(
    q_ptr,                  # *bfloat16, [B, H, D], contiguous
    token_ids_ptr,          # *int32,    [B, T_MAX], contiguous
    k_ptr_prepacked,        # *bfloat16, [B, T_MAX, D], contiguous
    l_max_out_ptr,          # *float32,  [B, H]
    sm_scale,               # float32 scalar
    B: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    T_MAX: tl.constexpr,
    gqa_ratio: tl.constexpr,
):
    # grid: (B, H)
    b = tl.program_id(0)
    h = tl.program_id(1)

    # Load q[b, h, :] as vector
    q_base = q_ptr + b * (H * D) + h * D
    q_vec = tl.load(q_base).to(tl.float32)  # [D]

    # GQA mapping: kv head index for this query head
    kvh = h // gqa_ratio

    # Compute l_max and sum of exp(scaled - l_max)
    l_max = -float("inf")
    lse_sum = 0.0

    for t in range(T_MAX):
        tok_id = tl.load(token_ids_ptr + b * T_MAX + t).to(tl.int32)
        # read k row: k_ptr_prepacked[b, t, :]
        k_row_ptr = k_ptr_prepacked + b * (T_MAX * D) + t * D
        k_vec = tl.load(k_row_ptr).to(tl.float32)  # [D]
        logits = tl.dot(q_vec, k_vec)
        scaled = logits * sm_scale
        l_max = tl.maximum(l_max, scaled)
        lse_sum += tl.exp(scaled - l_max)

    inv_log2 = 1.0 / math.log(2.0)
    lse_bh = l_max + tl.log(lse_sum) * inv_log2
    tl.store(l_max_out_ptr + b * H + h, lse_bh)


@triton.jit
def output_kernel(
    q_ptr,                  # *bfloat16, [B, H, D], contiguous
    token_ids_ptr,          # *int32,    [B, T_MAX], contiguous
    lse_out_ptr,            # *float32,  [B, H]
    v_ptr_prepacked,        # *bfloat16, [B, T_MAX, D], contiguous
    output_ptr,             # *bfloat16, [B, H, D], contiguous
    sm_scale,               # float32 scalar
    B: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    T_MAX: tl.constexpr,
    gqa_ratio: tl.constexpr,
):
    # grid: (B, H)
    b = tl.program_id(0)
    h = tl.program_id(1)

    q_base = q_ptr + b * (H * D) + h * D
    q_vec = tl.load(q_base).to(tl.float32)  # [D]

    # load lse for this (b, h)
    lse_bh = tl.load(lse_out_ptr + b * H + h)

    # accumulate output vector
    acc = tl.zeros((D,), dtype=tl.float32)

    for t in range(T_MAX):
        tok_id = tl.load(token_ids_ptr + b * T_MAX + t).to(tl.int32)
        # read v row: v_ptr_prepacked[b, t, :]
        v_row_ptr = v_ptr_prepacked + b * (T_MAX * D) + t * D
        v_vec = tl.load(v_row_ptr).to(tl.float32)  # [D]

        # compute logits_scaled for this token and its contribution
        logits = tl.dot(q_vec, tl.load(v_row_ptr).to(tl.float32))  # compute q·v
        scaled = logits * sm_scale

        # attention probability
        attn = tl.exp(scaled - lse_bh)

        # accumulate
        acc += attn * v_vec

    # store output as bfloat16
    out_base = output_ptr + b * (H * D) + h * D
    tl.store(out_base, acc.to(tl.bfloat16))


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure tensors on GPU and contiguous
        device = q.device
        q = q.contiguous().to(torch.bfloat16)
        k_cache = k_cache.contiguous().to(torch.bfloat16)
        v_cache = v_cache.contiguous().to(torch.bfloat16)
        kv_indptr = kv_indptr.to(torch.int32)
        kv_indices = kv_indices.to(torch.int32)

        B = q.shape[0]
        H = q.shape[1]
        D = q.shape[2]
        N = k_cache.shape[2]  # num kv heads (8)
        gqa_ratio = H // N    # 4

        # Compute num tokens per batch from kv_indptr
        num_tokens_per_b = (kv_indptr[1:] - kv_indptr[:-1]).to(torch.int32).tolist()
        num_tokens_max = max(num_tokens_per_b) if num_tokens_per_b else 0
        # Set T_MAX as the max number of tokens
        T_MAX = num_tokens_max if num_tokens_max > 0 else 1

        # Pack token_ids_all: shape [B, T_MAX]
        token_ids_all = torch.empty((B, T_MAX), dtype=torch.int32, device=device)
        for i, (start, end) in enumerate(zip(kv_indptr[:-1], kv_indptr[1:])):
            tokens = kv_indices[start:end]
            token_ids_all[i, :tokens.shape[0]].copy_(tokens)
            if tokens.shape[0] < T_MAX:
                token_ids_all[i, tokens.shape[0]:] = -1  # mask invalid

        # Prepack k_ptr_prepacked: [B, T_MAX, D]
        # Gather k_cache[token_ids_all[b, t], kvh, :] and pack into k_ptr_prepacked[b, t, :]
        # Note: we need N to index kvh. Since kvh is per query head, we broadcast kvh across tokens.
        k_ptr_prepacked = torch.empty((B, T_MAX, D), dtype=torch.bfloat16, device=device)
        for i in range(B):
            for t in range(T_MAX):
                if token_ids_all[i, t] >= 0:
                    pid = int(token_ids_all[i, t].item())
                    kvh = (i * gqa_ratio)  # query head to kv head mapping
                    # Load row from k_cache: k_cache[pid, 0, kvh, :]
                    # k_cache shape: [P, 1, N, D] so index as k_cache[pid, 0, kvh, :]
                    k_row = k_cache[pid, 0, kvh, :].contiguous()
                    k_ptr_prepacked[i, t, :] = k_row
                else:
                    # dummy row of zeros
                    k_ptr_prepacked[i, t, :] = torch.zeros((D,), dtype=torch.bfloat16, device=device)

        # Prepack v_ptr_prepacked: [B, T_MAX, D], same mapping
        v_ptr_prepacked = torch.empty((B, T_MAX, D), dtype=torch.bfloat16, device=device)
        for i in range(B):
            for t in range(T_MAX):
                if token_ids_all[i, t] >= 0:
                    pid = int(token_ids_all[i, t].item())
                    kvh = (i * gqa_ratio)
                    v_row = v_cache[pid, 0, kvh, :].contiguous()
                    v_ptr_prepacked[i, t, :] = v_row
                else:
                    v_ptr_prepacked[i, t, :] = torch.zeros((D,), dtype=torch.bfloat16, device=device)

        # Allocate outputs
        lse_out = torch.empty((B, H), dtype=torch.float32, device=device)
        output = torch.empty((B, H, D), dtype=torch.bfloat16, device=device)

        # Launch Triton kernels
        grid = (B, H)
        lse_and_max_kernel[grid](
            q_ptr=q.to(torch.float32),
            token_ids_ptr=token_ids_all,
            k_ptr_prepacked=k_ptr_prepacked,
            l_max_out_ptr=lse_out,  # we store logsumexp base-2 in this output
            sm_scale=float(sm_scale),
            B=B, H=H, D=D, T_MAX=T_MAX, gqa_ratio=gqa_ratio,
            num_warps=4, num_stages=2,
        )

        output_kernel[grid](
            q_ptr=q.to(torch.float32),
            token_ids_ptr=token_ids_all,
            lse_out_ptr=lse_out,
            v_ptr_prepacked=v_ptr_prepacked,
            output_ptr=output,
            sm_scale=float(sm_scale),
            B=B, H=H, D=D, T_MAX=T_MAX, gqa_ratio=gqa_ratio,
            num_warps=4, num_stages=2,
        )

        return output, lse_out

# Helper functions for testing
def get_inputs():
    q = torch.randn([1, 32, 128], dtype=torch.bfloat16, device='cuda')
    k_cache = torch.randn([11, 1, 8, 128], dtype=torch.bfloat16, device='cuda')
    v_cache = torch.randn([11, 1, 8, 128], dtype=torch.bfloat16, device='cuda')
    _n = 1; _t = 10
    _lens = torch.full((_n,), _t // _n, dtype=torch.int32, device='cuda')
    _lens[: _t % _n] += 1
    kv_indptr = torch.cat([torch.zeros(1, dtype=torch.int32, device='cuda'), torch.cumsum(_lens, 0)], 0).to(torch.int32)
    kv_indices = torch.randint(0, 11, [10], dtype=torch.int32, device='cuda')
    sm_scale = 1.0 / math.sqrt(128)  # float32 scalar
    return [q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale]


def fused_operator(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5):
    _out = ModelNew().forward(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5)
    return list(_out) if isinstance(_out, (tuple, list)) else [_out]


def run(*args):
    return ModelNew()(*args)
