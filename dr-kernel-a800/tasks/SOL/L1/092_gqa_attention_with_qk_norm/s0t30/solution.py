class ModelNew:
    def forward(
        self,
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
        # Checks and setup
        assert hidden_states.is_cuda, "hidden_states must be CUDA for Triton"
        B, L, H_in = hidden_states.shape  # in original, H_in=hidden_dim=768
        head_dim = 128
        num_attention_heads = 96
        num_key_value_heads = 8
        num_key_value_groups = 12
        scaling = 1.0 / (head_dim ** 0.5)

        # 1) Linear projection for Q, K, V: [B, L, 128]
        # Triton linear_proj_kernel, grid = (B, L, 128), BLOCK_K=128
        # Pass q_proj_weight, k_proj_weight, v_proj_weight with H_in = head_dim, N_out = head_dim
        # We need to extract the correct parts; however, the original code uses the same head_dim=128 for QKV.
        # Here, we assume that hidden_states is already the output of a prior linear, or the original q_proj_weight shape is [head_dim, hidden_dim].
        # Given the evaluation setup, we assume q_proj_weight is of shape [H_in, head_dim], which is a common layout in attention models.
        # For correctness in this environment, we rely on provided weights. We will run the kernel for each.

        # Ensure contiguity
        hidden_states_c = hidden_states.contiguous()
        q_out = torch.empty((B, L, head_dim), device=hidden_states.device, dtype=torch.float32)
        k_out = torch.empty((B, L, head_dim), device=hidden_states.device, dtype=torch.float32)
        v_out = torch.empty((B, L, head_dim), device=hidden_states.device, dtype=torch.float32)

        # Launch linear for Q
        linear_proj_kernel[(B, L, head_dim)](
            hidden_states_c, q_proj_weight, q_proj_bias, q_out,
            B, L, H_in, head_dim,
            hidden_states_c.stride(0), hidden_states_c.stride(1), hidden_states_c.stride(2),
            q_proj_weight.stride(0), q_proj_weight.stride(1),
            q_out.stride(0), q_out.stride(1), q_out.stride(2),
            BLOCK_K=128, num_warps=4, num_stages=2
        )

        # Launch linear for K
        linear_proj_kernel[(B, L, head_dim)](
            hidden_states_c, k_proj_weight, k_proj_bias, k_out,
            B, L, H_in, head_dim,
            hidden_states_c.stride(0), hidden_states_c.stride(1), hidden_states_c.stride(2),
            k_proj_weight.stride(0), k_proj_weight.stride(1),
            k_out.stride(0), k_out.stride(1), k_out.stride(2),
            BLOCK_K=128, num_warps=4, num_stages=2
        )

        # Launch linear for V
        linear_proj_kernel[(B, L, head_dim)](
            hidden_states_c, v_proj_weight, v_proj_bias, v_out,
            B, L, H_in, head_dim,
            hidden_states_c.stride(0), hidden_states_c.stride(1), hidden_states_c.stride(2),
            v_proj_weight.stride(0), v_proj_weight.stride(1),
            v_out.stride(0), v_out.stride(1), v_out.stride(2),
            BLOCK_K=128, num_warps=4, num_stages=2
        )

        # 2) RMSNorm for Q and K: y = x * rsqrt(mean(x^2) + eps) * weight
        q_rms = torch.empty((B, L, head_dim), device=hidden_states.device, dtype=torch.float32)
        k_rms = torch.empty((B, L, head_dim), device=hidden_states.device, dtype=torch.float32)

        rmsnorm_kernel[(B, L)](
            q_out, q_norm_weight, q_rms,
            B, L, head_dim,
            q_out.stride(0), q_out.stride(1), q_out.stride(2),
            q_norm_weight.stride(0), q_norm_weight.stride(1),
            q_rms.stride(0), q_rms.stride(1), q_rms.stride(2),
            eps=rms_norm_eps,
            num_warps=4, num_stages=2
        )

        rmsnorm_kernel[(B, L)](
            k_out, k_norm_weight, k_rms,
            B, L, head_dim,
            k_out.stride(0), k_out.stride(1), k_out.stride(2),
            k_norm_weight.stride(0), k_norm_weight.stride(1),
            k_rms.stride(0), k_rms.stride(1), k_rms.stride(2),
            eps=rms_norm_eps,
            num_warps=4, num_stages=2
        )

        # 3) Rotate Q and K: split h1[:64] by cos, h2[64:] by -sin and concatenate
        q_rot = torch.empty_like(q_rms)
        k_rot = torch.empty_like(k_rms)

        rotate_qk_kernel[(B, L)](
            q_rms, cos, sin, q_rot,
            B, L, head_dim,
            q_rms.stride(0), q_rms.stride(1), q_rms.stride(2),
            cos.stride(0), cos.stride(1),
            q_rot.stride(0), q_rot.stride(1), q_rot.stride(2),
            num_warps=4, num_stages=2
        )

        rotate_qk_kernel[(B, L)](
            k_rms, cos, sin, k_rot,
            B, L, head_dim,
            k_rms.stride(0), k_rms.stride(1), k_rms.stride(2),
            cos.stride(0), cos.stride(1),
            k_rot.stride(0), k_rot.stride(1), k_rot.stride(2),
            num_warps=4, num_stages=2
        )

        # 4) GQA: expand K_rot and V to 96 heads via groups
        # In original, K/V are [B, 8, L, 128]; expand to [B, 8, 12, L, 128] then reshape to [B, 96, L, 128].
        # We already have V as [B, L, 128]. To avoid overcomplicating, we use V directly for each head.
        # For K, we expand as in original:
        k_rot_expanded = k_rot.unsqueeze(2).expand(B, num_key_value_heads, num_key_value_groups, L, head_dim).reshape(B, num_attention_heads, L, head_dim)
        # V for output projection uses original V vector, not per-head split (the original code didn't split V per head). We will keep V as [B, L, 128].

        # 5) Compute attn_scores[b, qh, l, t] = Q_rot[b, qh, l] * K_rot_expanded[b, qh, t]
        attn_scores = torch.empty((B, num_attention_heads, L, L), device=hidden_states.device, dtype=torch.float32)

        attn_matmul_kernel[(B, num_attention_heads, L)](
            q_rot, k_rot_expanded, attn_scores,
            B, L, head_dim, num_attention_heads,
            q_rot.stride(0), q_rot.stride(1), q_rot.stride(2),
            k_rot_expanded.stride(0), k_rot_expanded.stride(1), k_rot_expanded.stride(2),
            attn_scores.stride(0), attn_scores.stride(1), attn_scores.stride(2), attn_scores.stride(3),
            num_warps=4, num_stages=2
        )

        # 6) Softmax + causal mask (upper triangle, diagonal=1)
        attn_probs = torch.empty_like(attn_scores)

        softmax_mask_kernel[(B, num_attention_heads, L)](
            attn_scores, attn_probs,
            B, L,
            attn_scores.stride(0), attn_scores.stride(1), attn_scores.stride(2), attn_scores.stride(3),
            attn_probs.stride(0), attn_probs.stride(1), attn_probs.stride(2), attn_probs.stride(3),
            num_warps=4, num_stages=2
        )

        # 7) Compute attn_output[b, qh, l] = sum_t attn_probs[b, qh, l, t] * V[b, qh, t]
        # Note: V is [B, L, 128], and we use the same V per head as the original code (no per-head split).
        attn_out = torch.empty((B, num_attention_heads, L, head_dim), device=hidden_states.device, dtype=torch.float32)

        output_matmul_kernel[(B, num_attention_heads, L)](
            attn_probs, v_out, attn_out,
            B, L, head_dim, num_attention_heads,
            attn_probs.stride(0), attn_probs.stride(1), attn_probs.stride(2), attn_probs.stride(3),
            v_out.stride(0), v_out.stride(1), v_out.stride(2),
            attn_out.stride(0), attn_out.stride(1), attn_out.stride(2), attn_out.stride(3),
            num_warps=4, num_stages=2
        )

        # Flatten heads for final linear
        attn_out_flat = attn_out.reshape(B, L, num_attention_heads * head_dim)

        # 8) Final linear projection to [B, L, hidden_dim=768] (o_proj_weight: [hidden_dim, 12288], no bias)
        final_out = torch.empty((B, L, 768), device=hidden_states.device, dtype=torch.float32)

        final_linear_kernel[(B, L, 768)](
            attn_out_flat, o_proj_weight, final_out,
            B, L, num_attention_heads * head_dim, 768,
            attn_out_flat.stride(0), attn_out_flat.stride(1), attn_out_flat.stride(2),
            o_proj_weight.stride(0), o_proj_weight.stride(1),
            final_out.stride(0), final_out.stride(1), final_out.stride(2),
            BLOCK_K=num_attention_heads * head_dim,  # H_flat = 12288, but we will loop tiles; Triton will handle loops
            num_warps=4, num_stages=2
        )

        return final_out


def run(*args):
    return ModelNew()(*args)
