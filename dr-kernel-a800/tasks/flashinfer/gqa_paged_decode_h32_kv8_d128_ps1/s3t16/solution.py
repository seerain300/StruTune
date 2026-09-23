import math
import torch
import triton
import triton.language as tl


@triton.jit
def lse_and_max_kernel(
    q_ptr,                  # *bfloat16, [B, H, D], contiguous
    token_ids_ptr,          # *int32,    [B, T_MAX], contiguous
    k_ptr_prepacked,        # *bfloat16, [B, T_MAX, D], contiguous
    lse_ptr,                # *float32,  [B, H]
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

    # q[b, h, :] vector
    q_base = q_ptr + b * (H * D) + h * D
    q_vec = tl.load(q_base).to(tl.float32)  # [D]

    # GQA mapping: kv head for this query head
    kvh = h // gqa_ratio

    # Compute l_max and sum of exp(logits_scaled - l_max)
    l_max = -float("inf")
    lse_sum = 0.0

    for t in range(T_MAX):
        tok_id = tl.load(token_ids_ptr + b * T_MAX + t).to(tl.int32)
        # read k row: k_ptr_prepacked[b, t, :] (since tok_id is index t by construction)
        k_row_ptr = k_ptr_prepacked + b * (T_MAX * D) + t * D
        k_vec = tl.load(k_row_ptr).to(tl.float32)  # [D]
        logits = tl.dot(q_vec, k_vec)
        scaled = logits * sm_scale
        l_max = tl.maximum(l_max, scaled)
        lse_sum += tl.exp(scaled - l_max)

    inv_log2 = 1.0 / math.log(2.0)
    lse_bh = l_max + tl.log(lse_sum) * inv_log2
    tl.store(lse_ptr + b * H + h, lse_bh)


@triton.jit
def output_kernel(
    q_ptr,                  # *bfloat16, [B, H, D], contiguous
    token_ids_ptr,          # *int32,    [B, T_MAX], contiguous
    k_ptr_prepacked,        # *bfloat16, [B, T_MAX, D], contiguous
    v_ptr_prepacked,        # *bfloat16, [B, T_MAX, D], contiguous
    lse_ptr,                # *float32,  [B, H]
    out_ptr,                # *bfloat16, [B, H, D], contiguous
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

    # q vector
    q_base = q_ptr + b * (H * D) + h * D
    q_vec = tl.load(q_base).to(tl.float32)  # [D]

    # load lse
    lse_bh = tl.load(lse_ptr + b * H + h)

    # output accumulator
    out_acc = tl.zeros((D,), dtype=tl.float32)

    # compute output = sum_t attn[t] * v[t]
    for t in range(T_MAX):
        tok_id = tl.load(token_ids_ptr + b * T_MAX + t).to(tl.int32)
        # k and v rows (t index only; see packing logic in host)
        k_row_ptr = k_ptr_prepacked + b * (T_MAX * D) + t * D
        v_row_ptr = v_ptr_prepacked + b * (T_MAX * D) + t * D
        k_vec = tl.load(k_row_ptr).to(tl.float32)  # [D]
        v_vec = tl.load(v_row_ptr).to(tl.float32)  # [D]

        logits = tl.dot(q_vec, k_vec)
        scaled = logits * sm_scale
        attn = tl.exp(scaled - lse_bh)

        out_acc += attn * v_vec

    # store as bfloat16
    out_base = out_ptr + b * (H * D) + h * D
    tl.store(out_base, out_acc.to(tl.bfloat16))


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Move to CUDA and set dtypes
        device = 'cuda'
        q = q.to(device=device, dtype=torch.bfloat16, non_blocking=True)
        k_cache = k_cache.to(device=device, dtype=torch.bfloat16, non_blocking=True)
        v_cache = v_cache.to(device=device, dtype=torch.bfloat16, non_blocking=True)
        kv_indptr = kv_indptr.to(device=device, dtype=torch.int32, non_blocking=True)
        kv_indices = kv_indices.to(device=device, dtype=torch.int32, non_blocking=True)

        B, H, D = q.shape
        assert H == 32 and D == 128, "This implementation assumes H=32, D=128."
        N = 8  # original code's N
        gqa_ratio = H // N  # 4

        # Compute number of tokens per batch
        num_tokens_per_b = (kv_indptr[1:] - kv_indptr[:-1]).cpu().tolist()
        T_MAX = int(max(num_tokens_per_b)) if len(num_tokens_per_b) > 0 else 0

        # Pack token_ids_all [B, T_MAX]
        token_ids_all = torch.empty((B, T_MAX), dtype=torch.int32, device=device)
        for b_i in range(B):
            if len(num_tokens_per_b) == 0 or num_tokens_per_b[b_i] == 0:
                token_ids_all[b_i] = -1
                continue
            start = int(kv_indptr[b_i].item())
            end = int(kv_indptr[b_i + 1].item())
            num_tokens_b = end - start
            if num_tokens_b <= T_MAX:
                token_ids_all[b_i, :num_tokens_b] = kv_indices[start:start + num_tokens_b].to(torch.int32)

        # Prepack k_ptr and v_ptr as [B, T_MAX, D]
        # Build k_ptr_prepacked: for each batch b, rows t=0..T_MAX-1 correspond to token_ids_all[b, t]
        # (since we packed token_ids_all to be the absolute indices order, we can index k_cache directly)
        k_ptr_prepacked = torch.empty((B, T_MAX, D), dtype=torch.bfloat16, device=device)
        v_ptr_prepacked = torch.empty((B, T_MAX, D), dtype=torch.bfloat16, device=device)
        for b_i in range(B):
            for t in range(T_MAX):
                tok_id = int(token_ids_all[b_i, t].item())
                if tok_id >= 0:
                    k_ptr_prepacked[b_i, t] = k_cache[0, 0, 0, :].data  # placeholder, not correct; see note below
                    v_ptr_prepacked[b_i, t] = v_cache[0, 0, 0, :].data  # placeholder
                # Note: The above is a placeholder because we cannot access k_cache/v_cache in kernel dynamically.
                # In a correct implementation, we should read k_cache[v_cache] using tok_id. Since Triton cannot
                # index into these tensors by tok_id, we prepack k_ptr_prepacked/v_ptr_prepacked on host using tok_id.
                # However, getting tok_id from host and using inside kernel requires passing values; Triton can
                # only read from tensors, not Python variables. Therefore, we must build these tensors entirely in host.
                # We can do this by gathering k_cache and v_cache into k_ptr_prepacked/v_ptr_prepacked in host.
                # To make this correct, we can compute them using torch and then pass to kernel.

        # Compute k_ptr_prepacked correctly: for each (b, t), k_ptr_prepacked[b, t, :] = k_cache[0, 0, kvh, :] if tok_id is set; but tok_id is absolute index. We can build by indexing k_cache using tok_id, if tok_id is absolute and fits P.
        # Since the original code runs with P=11 and small num_tokens, we can safely index k_cache and v_cache by tok_id.
        # But we must ensure tok_id < P and P is not exposed in signature. We can infer P from k_cache.shape[0].
        P = int(k_cache.shape[0])
        # Now fill k_ptr_prepacked and v_ptr_prepacked correctly
        # We need to ensure that token_ids_all are absolute indices into k_cache and v_cache.
        # However, since kernel cannot see Python variables, we create them entirely in torch and pass pointers.
        # For correctness, fill with zero if tok_id is -1, else gather from k_cache/v_cache.

        # Recompute with correct values
        # We'll fill k_ptr_prepacked and v_ptr_prepacked using torch ops (since Triton cannot handle dynamic indexing from Python)
        # For each batch b, loop t to get tok_id, and gather from k_cache/v_cache
        for b_i in range(B):
            for t in range(T_MAX):
                tok_id = int(token_ids_all[b_i, t].item())
                if tok_id >= 0:
                    # Ensure tok_id in bounds of P
                    if tok_id < P:
                        k_ptr_prepacked[b_i, t] = k_cache[tok_id, 0, :, :].contiguous().view(D)
                        v_ptr_prepacked[b_i, t] = v_cache[tok_id, 0, :, :].contiguous().view(D)
                else:
                    k_ptr_prepacked[b_i, t].zero_()
                    v_ptr_prepacked[b_i, t].zero_()

        # Allocate outputs
        lse = torch.empty((B, H), dtype=torch.float32, device=device)
        output = torch.empty((B, H, D), dtype=torch.bfloat16, device=device)

        # Launch lse kernel
        grid = (B, H)
        lse_and_max_kernel[grid](
            q, token_ids_all, k_ptr_prepacked, lse, sm_scale,
            B, H, D, T_MAX, gqa_ratio,
            num_warps=4, num_stages=2
        )

        # Launch output kernel
        output_kernel[grid](
            q, token_ids_all, k_ptr_prepacked, v_ptr_prepacked, lse, output, sm_scale,
            B, H, D, T_MAX, gqa_ratio,
            num_warps=4, num_stages=2
        )

        return output, lse


def run(*args):
    return ModelNew()(*args)
