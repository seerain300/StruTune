import math
import torch
import triton
import triton.language as tl


@triton.jit
def softmax_and_attention_single_bh(
    q_ptr,                 # *fp32, pointer to q[b, h] vector, length = HEAD_DIM
    K_ptr,                 # *fp32, pointer to K tokens for this batch, shape [NUM_TOKENS, HEAD_DIM], contiguous
    V_ptr,                 # *fp32, pointer to V tokens for this batch, shape [NUM_TOKENS, HEAD_DIM], contiguous
    OUT_ptr,               # *fp32, output vector for this head [HEAD_DIM]
    LSE_ptr,               # *fp32, single scalar lse for this (b, h), length 1
    NUM_TOKENS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    SM_SCALE: tl.float32,      # 1.0 / sqrt(HEAD_DIM) (here 1/sqrt(128))
    LOG2_INVERSE: tl.float32,  # 1.0 / ln(2.0) ≈ 1.4426950408889634
):
    # One program instance per (b, h) where h = program_id(0)
    h = tl.program_id(0)

    # First pass: compute lse = logsumexp(logits_scaled) / ln(2)
    running_max = -float("inf")
    running_sum = 1.0  # initialize to 1 to avoid exp(inf - inf) in the first update

    for t in range(0, NUM_TOKENS):
        # Load q vector for this head: [HEAD_DIM]
        q_vec = tl.load(q_ptr + h * HEAD_DIM + tl.arange(0, HEAD_DIM))
        # Load k vector for token t: [HEAD_DIM]
        k_vec = tl.load(K_ptr + t * HEAD_DIM + tl.arange(0, HEAD_DIM))
        # Compute logits_t = q_vec · k_vec
        logits_t = tl.sum(q_vec * k_vec, axis=0)  # scalar
        scaled = logits_t * SM_SCALE
        # update running max
        running_max = tl.maximum(running_max, scaled)
        # running_sum accumulates exp(scaled - running_max) in a numerically stable way
        running_sum = running_sum * tl.exp(running_max - running_max) + tl.exp(scaled - running_max)

    # Final lse: logsumexp(scaled) = log(running_sum) + running_max
    lse = tl.log(running_sum) + running_max
    # Divide by ln(2) to match original: lse_scaled = lse / ln(2) == lse * (1/ln(2))
    lse_scaled = lse * LOG2_INVERSE
    # Store lse as a scalar at LSE_ptr[0]
    tl.store(LSE_ptr, lse_scaled)

    # Second pass: compute output vector
    out_vec = tl.zeros((HEAD_DIM,), dtype=tl.float32)
    for t in range(0, NUM_TOKENS):
        q_vec = tl.load(q_ptr + h * HEAD_DIM + tl.arange(0, HEAD_DIM))
        v_vec = tl.load(V_ptr + t * HEAD_DIM + tl.arange(0, HEAD_DIM))
        k_vec = tl.load(K_ptr + t * HEAD_DIM + tl.arange(0, HEAD_DIM))
        logits_t = tl.sum(q_vec * k_vec, axis=0)
        scaled = logits_t * SM_SCALE
        attn = tl.exp(scaled - lse_scaled)  # softmax over tokens
        out_vec += attn * v_vec

    # Store output vector for head h
    tl.store(OUT_ptr + h * HEAD_DIM + tl.arange(0, HEAD_DIM), out_vec)


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        """
        q: [B, 32, 128], bfloat16
        k_cache: [num_pages, 1, 8, 128], bfloat16
        v_cache: [num_pages, 1, 8, 128], bfloat16
        kv_indptr: [len_indptr], int32, with len_indptr == B + 1
        kv_indices: [num_kv_indices], int32
        sm_scale: float32 scalar (e.g., 1/sqrt(128))
        Returns:
        output: [B, 32, 128], bfloat16
        lse: [B, 32], float32
        """
        assert q.is_cuda, "Input q must be on CUDA device"
        assert k_cache.is_cuda and v_cache.is_cuda, "k_cache and v_cache must be on CUDA device"
        assert kv_indptr.is_cuda and kv_indices.is_cuda, "kv_indptr and kv_indices must be on CUDA device"

        B, num_qo_heads, head_dim = q.shape
        # GQA parameters
        gqa_ratio = num_qo_heads // 8  # 32 // 8 = 4
        num_kv_heads = 8
        assert head_dim == 128

        # Compute per-batch number of tokens
        num_tokens = []  # length B
        for b in range(B):
            start = int(kv_indptr[b].item())
            end = int(kv_indptr[b + 1].item())
            num_tokens.append(end - start)

        # Build K_t and V_t per batch: gather k_cache and v_cache rows indicated by kv_indices
        K_t_list = []
        V_t_list = []
        for b in range(B):
            start = int(kv_indptr[b].item())
            end = int(kv_indptr[b + 1].item())
            token_indices = kv_indices[start:end].to(torch.long).cuda()  # [num_tokens[b]]
            # Slice k_cache and v_cache along num_kv_heads dim = 8
            K_t = k_cache.index_select(0, token_indices)  # [num_tokens[b], 1, 8, 128]
            V_t = v_cache.index_select(0, token_indices)  # [num_tokens[b], 1, 8, 128]
            K_t_list.append(K_t)
            V_t_list.append(V_t)

        # Prepare output and lse tensors
        output = torch.empty((B, num_qo_heads, head_dim), dtype=torch.float32, device=q.device)
        lse = torch.empty((B, num_qo_heads), dtype=torch.float32, device=q.device)

        # Launch Triton kernels: one per (b, h)
        for b in range(B):
            for h in range(num_qo_heads):
                # GQA mapping
                kv_head = h // gqa_ratio  # 0..7
                # Cast q vector to fp32 for compute
                q_vec = q[b, h].to(torch.float32).contiguous()  # [128]
                # Slice per head: [num_tokens[b], 128]
                K_t = K_t_list[b]  # [num_tokens[b], 1, 8, 128]
                V_t = V_t_list[b]  # [num_tokens[b], 1, 8, 128]
                K_t_h = K_t[:, 0, kv_head, :]  # [num_tokens[b], 128], originally bfloat16, but we’ll cast to fp32 in kernel args
                V_t_h = V_t[:, 0, kv_head, :]  # [num_tokens[b], 128]

                # Ensure contiguous; Triton expects contiguous memory for these loads
                K_t_h = K_t_h.to(torch.float32).contiguous()
                V_t_h = V_t_h.to(torch.float32).contiguous()

                num_tokens_b = num_tokens[b]
                out_vec = torch.empty((head_dim,), dtype=torch.float32, device=q.device)
                lse_scalar = torch.empty((1,), dtype=torch.float32, device=q.device)

                softmax_and_attention_single_bh[(1,)](
                    q_vec, K_t_h, V_t_h, out_vec, lse_scalar,
                    NUM_TOKENS=num_tokens_b,
                    HEAD_DIM=head_dim,
                    SM_SCALE=sm_scale,
                    LOG2_INVERSE=1.0 / math.log(2.0),
                )

                output[b, h] = out_vec
                lse[b, h] = lse_scalar[0]

        # Cast output to bfloat16 to match original
        output = output.to(torch.bfloat16)
        return output, lse

# Example get_inputs remains the same as provided
def get_inputs():
    q = torch.randn([1, 32, 128], dtype=torch.bfloat16, device='cuda')
    k_cache = torch.randn([11, 1, 8, 128], dtype=torch.bfloat16, device='cuda')
    v_cache = torch.randn([11, 1, 8, 128], dtype=torch.bfloat16, device='cuda')
    _n = 1; _t = 10
    _lens = torch.full((_n,), _t // _n, dtype=torch.int32)
    _lens[: _t % _n] += 1
    kv_indptr = torch.cat([torch.zeros(1, dtype=torch.int32), torch.cumsum(_lens, 0)]).to(torch.int32).cuda()
    kv_indices = torch.randint(0, 11, [10], dtype=torch.int32).cuda()
    sm_scale = 1.0 / math.sqrt(128)  # float32 scalar
    return [q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale]

# Optional local test
if __name__ == "__main__":
    model = ModelNew().cuda()
    q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale = get_inputs()
    out, lse = model(q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale)
    print("Output shape:", out.shape, out.dtype)
    print("lse shape:", lse.shape, lse.dtype)


def run(*args):
    return ModelNew()(*args)
