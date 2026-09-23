import torch
import torch.nn as nn
import torch.nn.functional as F

import triton
import triton.language as tl

# 1) Triton kernel for dense linear: output = input @ weight.T + bias
#    Input:  hidden [B, S, 128] (we'll launch per (b,h,i)), weight [H, 128], bias [H]
#    Output: out [S] (per (b,h,i) row), but since we launch per (b,h,i), we can store directly
@triton.jit
def triton_linear_row(b_ptr, w_ptr, bias_ptr, out_ptr, S, K, stride_b, stride_w, stride_out, BLOCK: tl.constexpr):
    # program ids: we'll launch grid (B, H, S), each program handles one (b, h, i)
    b_id = tl.program_id(0)
    h_id = tl.program_id(1)
    i_id = tl.program_id(2)

    # base pointers
    # hidden: b_ptr is [B, S, K], we need row i_id
    base_hidden = b_ptr + b_id * stride_b + i_id * K
    # weight: w_ptr is [H, K], we need all H rows
    # out: out_ptr is [B, H, S] flattened as (b, h, i) -> b*H*S + h*S + i
    out_index = b_id * H * S + h_id * S + i_id
    out_row_ptr = out_ptr + out_index

    # Accumulator for this output row
    acc = tl.zeros([K], dtype=tl.float32)

    # Loop over K dimension in blocks
    for k_start in range(0, K, BLOCK):
        k_offsets = k_start + tl.arange(0, BLOCK)
        # Load hidden row slice: base_hidden + k_offsets
        hidden_vec = tl.load(base_hidden + k_offsets, mask=k_offsets < K, other=0.0).to(tl.float32)
        # Load weight rows: w_ptr[h_id, k_offsets]
        weight_vec = tl.load(w_ptr + h_id * K + k_offsets, mask=k_offsets < K, other=0.0).to(tl.float32)
        # Accumulate dot product for this block
        acc += tl.sum(hidden_vec[:, None] * weight_vec[None, :], axis=0)

    # Add bias
    bias_val = tl.load(bias_ptr + h_id).to(tl.float32)
    acc += bias_val

    # Store output (cast to original dtype if needed). out_ptr dtype should match hidden's dtype.
    tl.store(out_row_ptr, acc)

# 2) Triton kernel for attention per (b, h, i) row: computes output[b,h,i]
#    Inputs:
#      Q: [B, H, S, K] -> we'll index as Q[b,h,i,:]
#      K: [B, H, S, K] -> K[b, h, j, :]
#      V: [B, H, S, K] -> V[b, h, j, :]
#      causal_mask: [S, S], float32, -inf where j <= i, 0 elsewhere
#    Output: out_vec [S] for this row (we'll store into Out[B, H, S])
@triton.jit
def triton_attention_row(Q_ptr, K_ptr, V_ptr, causal_mask_ptr, Out_ptr,
                          S, K, stride_q, stride_k, stride_v, stride_mask, stride_out,
                          scaling: tl.float32,
                          BLOCK_K: tl.constexpr, BLOCK_J: tl.constexpr):
    b_id = tl.program_id(0)
    h_id = tl.program_id(1)
    i_id = tl.program_id(2)

    # We will compute attention for position i_id (query row) against all j positions.
    # First, initialize attn_scores as -inf and then fill with valid scores.
    # Note: Triton doesn't support tensor-of-tensor initialization easily; we'll compute in blocks.

    # 2.1) Compute and softmax over sequence dimension (j) in BLOCK_J chunks
    # We'll keep a running max and sum for numerical stability
    # Initialize m (max) and l (sum) vectors
    m = tl.full([BLOCK_J], -1.0e30, dtype=tl.float32)
    l = tl.zeros([BLOCK_J], dtype=tl.float32)

    # 2.1.1) Loop over j to compute max and sum in tiles
    for j_start in range(0, S, BLOCK_J):
        j_offsets = j_start + tl.arange(0, BLOCK_J)
        # Load Q row i: Q[b,h,i,:]
        q_row = tl.load(Q_ptr + b_id * stride_q + h_id * K + i_id * K, mask=True, other=0.0).to(tl.float32)  # [K]
        # Compute dot products with K[j,:] in tiles
        for jj in range(0, BLOCK_J):
            j_idx = j_start + jj
            # mask for jj < S
            if j_idx < S:
                k_row = tl.load(K_ptr + b_id * stride_k + h_id * K + j_idx * K, mask=True, other=0.0).to(tl.float32)  # [K]
                score = tl.sum(q_row * k_row, axis=0) * scaling
                # Apply causal mask: mask[i_idx, j_idx]
                mask_val = tl.load(causal_mask_ptr + i_id * stride_mask + j_idx, mask=True, other=-1.0e30)
                score = score + mask_val
                # Update running max and sum for this position
                m = tl.maximum(m, score)
                l += tl.exp(score - m)

    # 2.1.2) Second loop to compute normalized probabilities and accumulate output
    for j_start in range(0, S, BLOCK_J):
        j_offsets = j_start + tl.arange(0, BLOCK_J)
        q_row = tl.load(Q_ptr + b_id * stride_q + h_id * K + i_id * K, mask=True, other=0.0).to(tl.float32)  # [K]
        out_vec = tl.zeros([S], dtype=tl.float32)
        for jj in range(0, BLOCK_J):
            j_idx = j_start + jj
            if j_idx < S:
                k_row = tl.load(K_ptr + b_id * stride_k + h_id * K + j_idx * K, mask=True, other=0.0).to(tl.float32)  # [K]
                score = tl.sum(q_row * k_row, axis=0) * scaling
                mask_val = tl.load(causal_mask_ptr + i_id * stride_mask + j_idx, mask=True, other=-1.0e30)
                score = score + mask_val
                # Softmax-normalized
                prob = tl.exp(score - m) / l
                v_row = tl.load(V_ptr + b_id * stride_v + h_id * K + j_idx * K, mask=True, other=0.0).to(tl.float32)  # [K]
                out_vec[j_idx] = tl.sum(q_row * v_row, axis=0) * prob

        # Store out_vec to Out[b,h,i]
        out_index = b_id * H * S + h_id * S + i_id
        tl.store(Out_ptr + out_index, out_vec)

# Note: The above kernel assumes K=128, S up to reasonable values. We launch grid (B, H, S).
# H is num_attention_heads=96. We'll set num_warps=4 for this light kernel.

class ModelNew(nn.Module):
    def __init__(self, num_attention_heads=96, num_key_value_heads=8, head_dim=128):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.scaling = 1.0 / (head_dim ** 0.5)

    def forward(self, hidden_states, q_proj_weight, q_proj_bias,
                k_proj_weight, k_proj_bias,
                v_proj_weight, v_proj_bias,
                o_proj_weight,
                q_norm_weight, k_norm_weight,
                cos, sin,
                rms_norm_eps):
        # hidden_states: [B, S, 128]
        assert hidden_states.is_cuda, "ModelNew requires CUDA tensors."
        B, S, K = hidden_states.shape
        H = self.num_attention_heads

        # 1) Triton compute Q, K, V: shapes [B, S, H]
        # We'll allocate outputs with same dtype as hidden_states
        dtype = hidden_states.dtype
        device = hidden_states.device

        # For Q
        Q = torch.empty((B, S, H), dtype=dtype, device=device)
        # Launch triton_linear_row for Q: grid (B, H, S)
        grid_q = (B, H, S)
        triton_linear_row[grid_q](
            hidden_states, q_proj_weight, q_proj_bias, Q,
            S, K, hidden_states.stride(0), q_proj_weight.stride(0), Q.stride(0),
            BLOCK=128,
            num_warps=4
        )

        # For K
        Kt = torch.empty((B, S, H), dtype=dtype, device=device)
        grid_k = (B, H, S)
        triton_linear_row[grid_k](
            hidden_states, k_proj_weight, k_proj_bias, Kt,
            S, K, hidden_states.stride(0), k_proj_weight.stride(0), Kt.stride(0),
            BLOCK=128,
            num_warps=4
        )

        # For V
        Vt = torch.empty((B, S, H), dtype=dtype, device=device)
        grid_v = (B, H, S)
        triton_linear_row[grid_v](
            hidden_states, v_proj_weight, v_proj_bias, Vt,
            S, K, hidden_states.stride(0), v_proj_weight.stride(0), Vt.stride(0),
            BLOCK=128,
            num_warps=4
        )

        # 2) Reshape and transpose to [B, H, S]
        # (PyTorch reshapes; these are metadata ops, not heavy compute)
        Q = Q.transpose(1, 2).contiguous()  # [B, H, S]
        Kt = Kt.transpose(1, 2).contiguous()  # [B, H, S]
        Vt = Vt.transpose(1, 2).contiguous()  # [B, H, S]

        # 3) RMSNorm per head (PyTorch implementation). Use q_norm_weight and k_norm_weight on query and key respectively.
        #    For Q: input is Q [B, H, S], weight is q_norm_weight [H]
        #    For K: input is Kt [B, H, S], weight is k_norm_weight [H]
        def rms_norm(x, weight):
            # x: [B, H, S], weight: [H]
            x_fp32 = x.float()
            var = x_fp32.pow(2).mean(dim=-1, keepdim=True)
            inv_rms = torch.rsqrt(var + rms_norm_eps)
            return (weight.float().unsqueeze(-1) * x_fp32 * inv_rms).to(x.dtype)

        Q = rms_norm(Q, q_norm_weight)
        Kt = rms_norm(Kt, k_norm_weight)

        # 4) Rotary Position Embedding (RoPE): split 128 into two halves
        #    For Q and K
        # We assume cos, sin are [S, 128]. We'll expand to [1, 1, S, 128] and apply elementwise.
        # Note: PyTorch elementwise is fine here. The original code applies rotation using cos/sin.
        # We split head_dim into halves: 64 each
        half = self.head_dim // 2
        q1 = Q[..., :half]
        q2 = Q[..., half:]
        k1 = Kt[..., :half]
        k2 = Kt[..., half:]
        # Rotate: q1, -q2; k1, -k2
        Q_rot = torch.cat([-q2, q1], dim=-1)
        K_rot = torch.cat([-k2, k1], dim=-1)
        # Apply rotation: Q = Q*cos + Q_rot*sin; same for K
        # cos and sin are [S, 128], expand to [1, 1, S, 128]
        cos_exp = cos.unsqueeze(0).unsqueeze(0).unsqueeze(-1)  # [1, 1, S, 128]
        sin_exp = sin.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
        Q = Q * cos_exp + Q_rot * sin_exp
        K = K_rot * sin_exp + Kt * cos_exp  # Note: original code applies rotation to K too

        # 5) Grouped-Query Attention: expand KV heads from 8 to 96
        #    Original code expands to [B, 96, S, 128]. We'll do it as view (no copy).
        num_key_value_groups = self.num_key_value_heads * (H // self.num_key_value_heads)  # 8 * 12 = 96
        # K and V shapes are [B, H, S, 128], we expand along H dimension to 96:
        K = K.unsqueeze(2)  # [B, H, 1, S, 128]
        V = Vt.unsqueeze(2)
        # Each of the H groups repeats S*128 elements; since original code expands to 96, we use expand(3) with size (B, num_key_value_groups, S, 128)
        # But the original uses 12 groups per 8 heads: num_key_value_groups = 12. We need 96.
        # The original uses num_key_value_groups = H // num_key_value_heads * num_key_value_heads ? Not directly given.
        # Given in the original code num_key_value_groups = 12, but here H=96, num_key_value_heads=8, so 96 // 8 = 12.
        # We'll use the same logic: groups = H // num_key_value_heads. So groups = 12.
        groups = H // self.num_key_value_heads  # 96 // 8 = 12
        K = K.expand(B, H, groups, S, K.shape[-1]).reshape(B, H, S, K.shape[-1])  # [B, H, S, 128]
        V = V.expand(B, H, groups, S, V.shape[-1]).reshape(B, H, S, V.shape[-1])

        # We need [B, 96, S, 128]. But we expanded to [B, H, S, 128]. Since H=96, it is already correct.
        # However, original code expanded from 8 to 96, implying it used groups=12. We should ensure output uses 96 heads.
        # To match, we keep K and V as [B, 96, S, 128] already.

        # 6) Compute attention per (b,h,i) row using Triton kernel:
        #    We will allocate Out [B, H, S] in float32 for stability, then cast at the end.
        Out = torch.empty((B, H, S), dtype=torch.float32, device=device)

        # causal mask: [S, S], -inf for i >= j, 0 otherwise
        causal_mask = torch.triu(
            torch.full((S, S), float('-inf'), device=device, dtype=torch.float32),
            diagonal=1
        )

        # Launch attention kernel: grid (B, H, S)
        grid_attn = (B, H, S)
        triton_attention_row[grid_attn](
            Q, K, V, causal_mask,
            Out,
            S, self.head_dim,
            Q.stride(0), K.stride(0), V.stride(0), causal_mask.stride(0), Out.stride(0),
            scaling=self.scaling,
            BLOCK_K=128, BLOCK_J=128,
            num_warps=4
        )

        # 7) Reshape and final linear projection (o_proj_weight): output [B, S, H*head_dim]
        #    We need to reshape Out [B, H, S] -> [B, S, H*128]
        Out = Out.transpose(1, 2).contiguous()  # [B, S, H]
        out = Out.reshape(B, S, H * self.head_dim)

        # 8) Output projection: F.linear(out, o_proj_weight, None)
        #    o_proj_weight: [H, 128*H] -> [H, 128*H] because num_attention_heads * head_dim = 96 * 128 = 12288
        #    out: [B, S, 12288]
        # We can implement this with PyTorch's F.linear to keep it simple. This is a dense op, not a heavy Triton candidate here.
        # Ensure o_proj_weight is on device
        o = F.linear(out, o_proj_weight, None)

        return o


def run(*args):
    return ModelNew()(*args)
