import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _linear_proj_kernel(
    hidden_ptr,         # *ptr to hidden_states [N, K]
    weight_ptr,         # *ptr to weight [E, K]
    out_ptr,            # *ptr to output logits [N, E]
    N, E, K,            # sizes
    stride_hn, stride_hk,
    stride_we, stride_wk,
    stride_on, stride_oe,
    TILE_E: tl.constexpr,  # tile size over E (e.g., 32)
    TILE_K: tl.constexpr,  # tile size over K (e.g., 64)
):
    # Grid: (pid_n over tokens, pid_te over tiles of experts)
    pid_n = tl.program_id(0)   # token row index
    pid_te = tl.program_id(1)  # tile id over E

    # Compute E tile
    e_start = pid_te * TILE_E
    e_offsets = e_start + tl.arange(0, TILE_E)
    e_mask = e_offsets < E

    # Accumulator for this tile of E
    acc = tl.zeros((TILE_E,), dtype=tl.float32)

    # Loop over K in chunks
    for k0 in range(0, K, TILE_K):
        k_offsets = k0 + tl.arange(0, TILE_K)
        k_mask = k_offsets < K

        # Load hidden row slice: [TILE_K]
        h = tl.load(hidden_ptr + pid_n * stride_hn + k_offsets * stride_hk, mask=k_mask, other=0.0)

        # Load weight tile: [TILE_E, TILE_K]
        w = tl.load(
            weight_ptr + e_offsets[:, None] * stride_we + k_offsets[None, :] * stride_wk,
            mask=e_mask[:, None] & k_mask[None, :],
            other=0.0,
        )

        # Accumulate dot product for this chunk
        for j in range(TILE_E):
            acc[j] += tl.sum(w[j, :] * h, axis=0)

    # Store results
    tl.store(out_ptr + pid_n * stride_on + e_offsets * stride_oe, acc, mask=e_mask)


@triton.jit
def _sigmoid_add_bias_kernel(
    logits_ptr,          # *ptr to [N, E] float32
    bias_ptr,            # *ptr to [E] float32
    out_ptr,             # *ptr to [N, E] float32
    N, E,
    stride_ln, stride_le,
    stride_bn,
    stride_on, stride_oe,
):
    # 1D launch: one program per element
    pid = tl.program_id(0)
    cols = tl.arange(0, E)
    row = pid // E
    col = pid % E
    if row >= N:
        return
    l = tl.load(logits_ptr + row * stride_ln + col * stride_le)
    b = tl.load(bias_ptr + col * stride_bn)
    s = 1.0 / (1.0 + tl.exp(-l))
    out = s + b
    tl.store(out_ptr + row * stride_on + col * stride_oe, out)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor, expert_bias: torch.Tensor, routed_scaling_factor: float):
        """
        Triton-optimized implementation. Computes logits = hidden @ weight.T using a Triton GEMM,
        then applies sigmoid and expert bias using a Triton elementwise kernel. The rest of routing
        logic (group selection, masking, final top-8, normalization) is done in PyTorch to ensure correctness.
        Returns:
          - topk_idx: [num_tokens, 8] LongTensor of expert indices chosen
          - topk_weight: [num_tokens, 8] float32 normalized weights scaled
        """
        # Ensure inputs are 2D and correct shapes; assume num_experts = hidden_dim = 256
        assert hidden_states.dim() == 2, "hidden_states must be [num_tokens, hidden_dim]"
        assert weight.dim() == 2, "weight must be [num_experts, hidden_dim]"
        assert expert_bias.dim() == 1, "expert_bias must be [num_experts]"

        device = hidden_states.device
        num_tokens, hidden_dim = hidden_states.shape
        num_experts = weight.shape[0]
        assert num_experts == 256 and hidden_dim == 256, "This Triton implementation assumes num_experts=hidden_dim=256"

        # Ensure dtype float32 and contiguous
        hidden_f32 = hidden_states.contiguous().to(torch.float32)
        weight_f32 = weight.contiguous().to(torch.float32)
        expert_bias_f32 = expert_bias.contiguous().to(torch.float32)

        # 1) Compute logits using Triton GEMM: [N, K] @ [E, K]^T -> [N, E]
        logits = torch.empty((num_tokens, num_experts), dtype=torch.float32, device=device)
        stride_hn, stride_hk = hidden_f32.stride()
        stride_we, stride_wk = weight_f32.stride()
        stride_on, stride_oe = logits.stride()

        TILE_E = 32
        TILE_K = 64
        grid_gemm = (num_tokens, triton.cdiv(num_experts, TILE_E))  # tuple of ints
        _linear_proj_kernel[grid_gemm](
            hidden_f32, weight_f32, logits,
            num_tokens, num_experts, hidden_dim,
            stride_hn, stride_hk,
            stride_we, stride_wk,
            stride_on, stride_oe,
            TILE_E=TILE_E, TILE_K=TILE_K,
            num_warps=4, num_stages=2,
        )

        # 2) Apply sigmoid and add expert bias using Triton elementwise kernel
        scores = torch.empty_like(logits)
        stride_ln, stride_le = logits.stride()
        stride_bn = expert_bias_f32.stride(0)
        stride_on_sb, stride_oe_sb = scores.stride()
        grid_elem = (num_tokens * num_experts,)  # tuple of ints
        _sigmoid_add_bias_kernel[grid_elem](
            logits, expert_bias_f32, scores,
            num_tokens, num_experts,
            stride_ln, stride_le,
            stride_bn,
            stride_on_sb, stride_oe_sb,
            num_warps=1, num_stages=1,
        )

        # 3) Group aggregation: sum of top-2 per group for each token
        n_group = 8
        experts_per_group = num_experts // n_group  # 32
        group_scores = torch.zeros((num_tokens, n_group), dtype=torch.float32, device=device)
        for t in range(num_tokens):
            for g in range(n_group):
                start = g * experts_per_group
                group_vals = scores[t, start : start + experts_per_group]
                top2 = torch.topk(group_vals, k=2, largest=True, sorted=False).values
                group_scores[t, g] = top2.sum()

        # 4) Select top-4 groups per token
        _, group_idx = torch.topk(group_scores, k=4, dim=-1, sorted=False)  # [num_tokens, 4]

        # 5) Build group mask: set 1 for selected groups, 0 otherwise
        group_mask = torch.zeros((num_tokens, n_group), dtype=torch.float32, device=device)
        for t in range(num_tokens):
            for k in range(4):
                g = int(group_idx[t, k].item())
                group_mask[t, g] = 1.0

        # 6) Expand group mask to expert level [num_tokens,


def run(*args):
    return ModelNew()(*args)
