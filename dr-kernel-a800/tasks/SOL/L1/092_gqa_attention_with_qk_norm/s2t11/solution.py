import torch
import torch.nn as nn
import torch.nn.functional as F

@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    q_proj_weight: torch.Tensor,
    q_proj_bias: torch.Tensor,
    k_proj_weight: torch.Tensor,
    k_proj_bias: torch.Tensor,
    v_proj_weight: torch.Tensor,
    v_proj_bias: torch.Tensor,
    o_proj_weight: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rms_norm_eps: float,
):
    batch_size, seq_length, _ = hidden_states.shape
    num_attention_heads = 96
    num_key_value_heads = 8
    head_dim = 128
    num_key_value_groups = 12
    scaling = head_dim ** -0.5
    
    # Q/K/V projections
    query_states = F.linear(hidden_states, q_proj_weight, q_proj_bias)
    key_states = F.linear(hidden_states, k_proj_weight, k_proj_bias)
    value_states = F.linear(hidden_states, v_proj_weight, v_proj_bias)
    
    # Reshape to separate heads
    query_states = query_states.view(batch_size, seq_length, num_attention_heads, head_dim)
    key_states = key_states.view(batch_size, seq_length, num_key_value_heads, head_dim)
    value_states = value_states.view(batch_size, seq_length, num_key_value_heads, head_dim)
    
    # Apply QK RMSNorm
    def rms_norm(x, weight):
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + rms_norm_eps)
        return (weight * x).to(input_dtype)
    
    query_states = rms_norm(query_states, q_norm_weight)
    key_states = rms_norm(key_states, k_norm_weight)
    
    # Transpose to [batch, num_heads, seq_len, head_dim]
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)
    
    # Apply RoPE
    cos_expanded = cos.unsqueeze(1)  # [batch, 1, seq_len, head_dim]
    sin_expanded = sin.unsqueeze(1)
    
    # Rotate half for Q
    q1, q2 = query_states[..., :64], query_states[..., 64:]
    q_rot_half = torch.cat((-q2, q1), dim=-1)
    query_states = (query_states * cos_expanded) + (q_rot_half * sin_expanded)
    
    # Rotate half for K
    k1, k2 = key_states[..., :64], key_states[..., 64:]
    k_rot_half = torch.cat((-k2, k1), dim=-1)
    key_states = (key_states * cos_expanded) + (k_rot_half * sin_expanded)
    
    # Repeat KV heads for GQA: [batch, 8, seq_len, 128] -> [batch, 96, seq_len, 128]
    key_states = key_states[:, :, None, :, :].expand(
        batch_size, num_key_value_heads, num_key_value_groups, seq_length, head_dim
    ).reshape(batch_size, num_attention_heads, seq_length, head_dim)
    value_states = value_states[:, :, None, :, :].expand(
        batch_size, num_key_value_heads, num_key_value_groups, seq_length, head_dim
    ).reshape(batch_size, num_attention_heads, seq_length, head_dim)
    
    # Compute attention scores
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * scaling
    
    # Apply causal mask
    causal_mask = torch.triu(
        torch.full((seq_length, seq_length), float('-inf'), device=hidden_states.device, dtype=attn_weights.dtype),
        diagonal=1
    )
    attn_weights = attn_weights + causal_mask
    
    # Softmax
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    
    # Compute attention output
    attn_output = torch.matmul(attn_weights, value_states)
    
    # Transpose and reshape
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(batch_size, seq_length, num_attention_heads * head_dim)
    
    # Output projection (no bias)
    output = F.linear(attn_output, o_proj_weight, None)
    
    return output

class ModelNew(nn.Module):
    def forward(self, hidden_states, q_proj_weight, q_proj_bias, k_proj_weight, k_proj_bias, v_proj_weight, v_proj_bias, o_proj_weight, q_norm_weight, k_norm_weight, cos, sin, rms_norm_eps):
        # Your Triton implementation goes here. Call real Triton kernels and return output.
        # Do not use torch.nn.functional.linear or torch.matmul in forward (leave only tensor metadata ops like view/reshape).
        pass


def run(*args):
    return ModelNew()(*args)
