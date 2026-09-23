import torch
import triton
import triton.language as tl


@triton.jit
def triton_normal_fill(out_ptr, n_elements: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    start = pid * BLOCK
    offsets = start + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    # Use Triton's rand to produce normal-like values. We implement a simple standardization:
    # u ~ Uniform[0,1), v ~ Uniform[0,1): z = sqrt(-2*log(u)) * sin(2*pi*v) ~ N(0,1).
    u = tl.rand(offsets)
    v = tl.rand(offsets)
    s = tl.sqrt(-2.0 * tl.log(u))
    theta = 2.0 * 3.141592653589793 * v
    z = s * tl.sin(theta)
    tl.store(out_ptr + offsets, z, mask=mask)


@triton.jit
def triton_sigmoid(x_ptr, y_ptr, n_elements: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    start = pid * BLOCK
    offsets = start + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = 1.0 / (1.0 + tl.exp(-x))
    tl.store(y_ptr + offsets, y, mask=mask)


@triton.jit
def triton_silu(x_ptr, y_ptr, n_elements: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    start = pid * BLOCK
    offsets = start + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    sig = 1.0 / (1.0 + tl.exp(-x))
    y = x * sig
    tl.store(y_ptr + offsets, y, mask=mask)


@triton.jit
def triton_topk_row(scores_ptr, indices_ptr, values_ptr,
                    n_rows: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                    stride_sb, stride_sn,
                    stride_ib, stride_in,
                    stride_vb, stride_vk):
    # Each program handles one row of scores. Select K largest values and indices.
    b = tl.program_id(0)
    for t in range(K):
        best_val = -1.0e20
        best_idx = 0
        for i in range(N):
            score = tl.load(scores_ptr + b * stride_sb + i * stride_sn)
            if score > best_val:
                best_val = score
                best_idx = i
        tl.store(values_ptr + b * stride_vb + t * stride_vk, best_val)
        tl.store(indices_ptr + b * stride_ib + t * stride_in, best_idx.to(tl.int32))
        # Remove selected from consideration by setting to -inf
        tl.store(scores_ptr + b * stride_sb + best_idx * stride_sn, -1.0e20)


@triton.jit
def triton_row_sum(x_ptr, out_ptr, n_elements: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    start = pid * BLOCK
    offsets = start + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    acc = tl.sum(x, axis=0)
    tl.store(out_ptr + pid, acc)


@triton.jit
def triton_fill_constant(out_ptr, n_elements: tl.constexpr, value: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    start = pid * BLOCK
    offsets = start + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    tl.store(out_ptr + offsets, value, mask=mask)


# GEMV per row: out[b, m] = dot(hidden[b, :], weight[m, :])
# hidden: [B, H], weight: [M, H], out: [B, M]
@triton.jit
def triton_gemv_row(hidden_ptr, weight_ptr, out_ptr,
                    B: tl.constexpr, H: tl.constexpr, M: tl.constexpr,
                    stride_bh, stride_bk,
                    stride_wh, stride_wk,
                    stride_ob, stride_om,
                    BLOCK_K: tl.constexpr):
    b = tl.program_id(0)  # one program per batch row
    acc = tl.zeros((M,), dtype=tl.float32)
    for k0 in range(0, H, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < H
        x = tl.load(hidden_ptr + b * stride_bh + offs_k * stride_bk, mask=mask_k, other=0.0)
        for m in range(0, M):
            w = tl.load(weight_ptr + m * stride_wh + offs_k * stride_wk, mask=mask_k, other=0.0)
            acc[m] += tl.sum(x * w, axis=0)
    offs_m = tl.arange(0, M)
    tl.store(out_ptr + b * stride_ob + offs_m * stride_om, acc, mask=offs_m < M)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, axes_and_scalars: dict, device: torch.device) -> dict:
        # Extract axes
        batch_seq_len = axes_and_scalars["batch_seq_len"]
        hidden_size = 4096
        n_routed_experts = 128
        num_experts_per_tok = 8
        routed_scaling_factor = 1.0
        B = batch_seq_len

        # 1) grad_output: torch.randn(B, hidden_size, bfloat16) -> Triton fill
        grad_output_flat = torch.empty(B * hidden_size, dtype=torch.bfloat16, device=device)
        triton_normal_fill[(B * hidden_size,)](grad_output_flat, n_elements=B * hidden_size, BLOCK=1024)
        grad_output = grad_output_flat.view(B, hidden_size)

        # 2) hidden_states: torch.randn(B, hidden_size, bfloat16) -> Triton fill
        hidden_states_flat = torch.empty(B * hidden_size, dtype=torch.bfloat16, device=device)
        triton_normal_fill[(B * hidden_size,)](hidden_states_flat, n_elements=B * hidden_size, BLOCK=1024)
        hidden_states = hidden_states_flat.view(B, hidden_size)

        # 3) router_weight: torch.randn(n_routed_experts, hidden_size, bfloat16) * 0.02 -> Triton fill
        E, H = n_routed_experts, hidden_size
        router_weight_flat = torch.empty(E * H, dtype=torch.bfloat16, device=device)
        triton_normal_fill[(E * H,)](router_weight_flat, n_elements=E * H, BLOCK=1024)
        router_weight = (router_weight_flat.view(E, H)).to(torch.bfloat16) * 0.02

        # 4) Compute logits = hidden_states @ router_weight.T -> [B, E], float32 via Triton GEMV
        logits = torch.empty((B, E), dtype=torch.float32, device=device)
        triton_gemv_row[(B,)](
            hidden_states, router_weight, logits,
            B=B, H=H, M=E,
            stride_bh=hidden_states.stride(0), stride_bk=hidden_states.stride(1),
            stride_wh=router_weight.stride(0), stride_wk=router_weight.stride(1),
            stride_ob=logits.stride(0), stride_om=logits.stride(1),
            BLOCK_K=1024,
        )

        # 5) scores = sigmoid(logits) -> Triton elementwise
        scores = torch.empty_like(logits, dtype=torch.float32, device=device)
        triton_sigmoid[(logits.numel(),)](logits, scores, n_elements=logits.numel(), BLOCK=1024)

        # 6) topk_indices and topk_values: Triton top-k per row
        topk_indices = torch.empty((B, num_experts_per_tok), dtype=torch.int32, device=device)
        topk_values = torch.empty((B, num_experts_per_tok), dtype=torch.float32, device=device)
        triton_topk_row[(B,)](
            scores, topk_indices, topk_values,
            n_rows=B, N=E, K=num_experts_per_tok,
            stride_sb=scores.stride(0), stride_sn=scores.stride(1),
            stride_ib=topk_indices.stride(0), stride_in=topk_indices.stride(1),
            stride_vb=topk_values.stride(0), stride_vk=topk_values.stride(1),
        )

        # 7) Normalize topk weights using denom = sum(topk_values, dim=-1) + 1e-20, then scale
        denom = torch.empty(B, dtype=torch.float32, device=device)
        tr


def run(*args):
    return ModelNew()(*args)
