import math
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def conv1d_short_groups_kernel(u_padded_ptr, w_ptr, out_ptr,
                                N, D, L_in, OUT_L, K, pad_left,
                                BLOCK_D: tl.constexpr):
    # grid = (N, D)
    n = tl.program_id(0)
    d = tl.program_id(1)
    for j in range(0, OUT_L):
        acc = 0.0
        for k in range(0, K):
            idx = j + pad_left - k
            valid = (idx >= 0) & (idx < L_in)
            u_val = tl.load(u_padded_ptr + n * (D * L_in) + d * L_in + idx, mask=valid, other=0.0)
            w_val = tl.load(w_ptr + d * K + k)
            acc += u_val * w_val
        tl.store(out_ptr + n * (D * OUT_L) + d * OUT_L + j, acc)


class ModelNew(nn.Module):
    def forward(self, *args):
        # args correspond to: hidden_states, norm1_weight, norm1_bias, norm2_weight, norm2_bias,
        # in_proj_weight, in_proj_bias, short_conv_weight, short_conv_bias, filter_linear1_weight,
        # filter_linear1_bias, sin_freq, filter_linear2_weight, filter_linear2_bias, filter_linear3_weight,
        # filter_linear3_bias, filter_linear_final_weight, filter_bias, exp_mod_deltas, out_proj_weight,
        # out_proj_bias, mlp_fc1_weight, mlp_fc1_bias, mlp_fc2_weight, mlp_fc2_bias, layer_norm_eps, exp_mod_shift

        # Short 1D convolution with groups=D and K=3, padding=2
        hidden_states = args[0]
        N, L, D = hidden_states.shape
        L_in = L + 4
        pad_left = 2
        OUT_L = L
        # Build padded u: [N, D, L_in]
        u_padded = torch.empty((N, D, L_in), dtype=torch.float32, device=hidden_states.device)
        u_padded[:, :, 2:L + 2] = hidden_states.to(torch.float32)
        # short_conv_weight: [D, 1, 3]
        short_conv_weight = args[7].to(torch.float32)  # shape [D, 1, 3]
        # Output tensor [N, D, OUT_L]
        out_conv = torch.empty((N, D, OUT_L), dtype=torch.float32, device=hidden_states.device)
        # Launch Triton conv kernel: grid = (N, D)
        conv1d_short_groups_kernel[(N, D)](
            u_padded, short_conv_weight, out_conv, N, D, L_in, OUT_L, 3, pad_left, BLOCK_D=256
        )
        return out_conv


def run(*args):
    return ModelNew()(*args)
