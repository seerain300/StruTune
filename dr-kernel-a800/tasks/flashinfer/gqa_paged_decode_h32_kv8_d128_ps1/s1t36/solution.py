import torch
import math

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:
    @triton.jit
    def _gqa_attention_kernel(
        q_ptr,           # *f32, shape [B, num_qo_heads, HEAD_DIM], contiguous
        k_ptr, v_ptr,    # *bf16 or *f16, shape [num_pages, 1, num_kv_heads, HEAD_DIM], contiguous
        kv_indptr_ptr,   # *i32, shape [B+1], contiguous
        out_ptr,         # *f32, shape [B, num_qo_heads, HEAD_DIM] (内核中存储 float32)
        lse_ptr,         # *f32, shape [B, num_qo_heads]
        B,               # int32
        num_qo_heads: tl.constexpr,
        num_kv_heads: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        sm_scale: tl.constexpr,
        gqa_ratio: tl.constexpr,
        ln2: tl.constexpr,  # natural log of 2 (由主机计算并传递)
    ):
        # 一个程序处理一个 (b, h)
        pid = tl.program_id(0)
        b = pid // num_qo_heads
        h = pid % num_qo_heads
        if b >= B or h >= num_qo_heads:
            return

        # 该批的 token 范围
        kv_start = tl.load(kv_indptr_ptr + b)        # i32
        kv_end = tl.load(kv_indptr_ptr + b + 1)     # i32
        num_tokens = kv_end - kv_start              # i32

        # 加载 q[b, h] 作为 float32 向量
        q_offset = b * num_qo_heads * HEAD_DIM + h * HEAD_DIM
        q_vec = tl.load(q_ptr + q_offset)  # [HEAD_DIM] f32

        # 初始化输出向量和 lse
        out_vec = tl.zeros((HEAD_DIM,), dtype=tl.float32)
        lse_val = -float('inf')  # f32

        # 遍历 tokens
        for t in range(0, num_tokens):
            idx = kv_start + t  # token index

            # GQA 映射：kv_head = h // gqa_ratio
            kv_head = h // gqa_ratio  # int32

            # 加载 k_t 和 v_t 作为向量
            base_k = idx * (num_kv_heads * HEAD_DIM) + kv_head * HEAD_DIM
            base_v = idx * (num_kv_heads * HEAD_DIM) + kv_head * HEAD_DIM
            k_t = tl.zeros((HEAD_DIM,), dtype=tl.float32)
            v_t = tl.zeros((HEAD_DIM,), dtype=tl.float32)
            for i in range(HEAD_DIM):
                k_elem = tl.load(k_ptr + base_k + i).to(tl.float32)
                v_elem = tl.load(v_ptr + base_v + i).to(tl.float32)
                k_t[i] = k_elem
                v_t[i] = v_elem

            # 计算点积 q_t · k_t
            logits = 0.0
            for i in range(HEAD_DIM):
                logits += q_vec[i] * k_t[i]

            # 缩放并更新 LSE 稳定
            scaled = logits * sm_scale
            if lse_val == -float('inf'):
                lse_new = scaled
            else:
                diff = scaled - lse_val
                lse_new = tl.maximum(lse_val, scaled) + tl.log(1.0 + tl.exp(-tl.abs(diff)))

            attn = tl.exp(scaled - lse_new)  # 该 token 的注意力权重

            # 累加输出向量
            out_vec += attn * v_t

            # 更新 lse
            lse_val = lse_new

        # 存储结果：output [B, num_qo_heads, HEAD_DIM] 的线性基址为 b*H + h
        out_base = b * (num_qo_heads * HEAD_DIM) + h * HEAD_DIM
        for i in range(HEAD_DIM):
            tl.store(out_ptr + out_base + i, out_vec[i])  # float32
        # 存储 lse / ln(2) 到 lse_ptr[b, h]
        lse_ptr_bh = b * num_qo_heads + h
        tl.store(lse_ptr + lse_ptr_bh, lse_val / ln2)


class ModelNew(torch.nn.Module):
    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # 确保 Triton 和 CUDA 可用
        assert TRITON_AVAILABLE, "Triton is required but not available"
        assert q.is_cuda and k_cache.is_cuda and v_cache.is_cuda and kv_indptr.is_cuda, "All tensors must be on CUDA"

        B, num_qo_heads, HEAD_DIM = q.shape
        num_kv_heads, _, _, _ = k_cache.shape
        gqa_ratio = num_qo_heads // num_kv_heads
        ln2 = math.log(2.0)

        # 准备输入：确保连续
        q_f32 = q.to(torch.float32).contiguous()         # 计算 q 时使用 f32
        k_cache = k_cache.contiguous()
        v_cache = v_cache.contiguous()
        kv_indptr = kv_indptr.contiguous()

        # 分配输出（内核中存储 float32，主机端转换为 bfloat16）
        output = torch.empty((B, num_qo_heads, HEAD_DIM), dtype=torch.float32, device=q.device)
        lse = torch.empty((B, num_qo_heads), dtype=torch.float32, device=q.device)

        # 启动 Triton 内核：每个 (b, h) 一个程序
        grid = (B * num_qo_heads,)
        _gqa_attention_kernel[grid](
            q_f32, k_cache, v_cache, kv_indptr, output, lse,
            B,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            HEAD_DIM=HEAD_DIM,
            sm_scale=float(sm_scale),
            gqa_ratio=gqa_ratio,
            ln2=ln2,
            num_warps=4,  # 可以根据设备调整
            num_stages=2,
        )

        # 转换 output 为 bfloat16 以符合原始返回类型
        output_bf16 = output.to(torch.bfloat16)
        return output_bf16, lse


def run(*args):
    return ModelNew()(*args)
