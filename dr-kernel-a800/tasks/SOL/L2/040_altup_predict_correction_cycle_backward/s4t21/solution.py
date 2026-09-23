import torch
import triton
import triton.language as tl


# 1) RMSNorm forward: per-token rstd = rsqrt(mean(x^2) + eps)
@triton.jit
def rms_norm_forward(x_ptr, rstd_ptr, H: tl.int32, eps: tl.float32, BLOCK: tl.constexpr):
    token_id = tl.program_id(0)  # ranges over B*S*I
    sumsq = 0.0
    for off in range(0, H, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < H
        x = tl.load(x_ptr + token_id * H + idx, mask=mask, other=0.0).to(tl.float32)
        sumsq += tl.sum(x * x, axis=0)
    mean = sumsq / H
    rstd = tl.rsqrt(mean + eps)
    tl.store(rstd_ptr + token_id, rstd)


# 2) tanh(linear) no bias: y[k] = tanh(dot(scaled, weight_row[k, :]))
@triton.jit
def tanh_linear_no_bias(scaled_ptr, weight_ptr, y_ptr, H: tl.int32, BLOCK: tl.constexpr):
    k = tl.program_id(0)  # output index, ranges over I*I
    acc = 0.0
    for off in range(0, H, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < H
        s = tl.load(scaled_ptr + idx, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(weight_ptr + k * H + idx, mask=mask, other=0.0).to(tl.float32)
        acc += tl.sum(s * w, axis=0)
    out = tl.tanh(acc)
    tl.store(y_ptr + k, out)


# 3) Per-token predictions matmul: out[b, s, i, j] = sum_h h_permuted[b, s, i, h] * all_coefs[j, h]
@triton.jit
def per_token_predictions_matmul(h_ptr, w_ptr, out_ptr, B: tl.int32, S: tl.int32,
                                 I: tl.int32, H: tl.int32, BLOCK: tl.constexpr):
    # program ids: j along dim 1, i along dim 2, token_id along dim 0
    j = tl.program_id(1)
    i = tl.program_id(2)
    token_id = tl.program_id(0)  # ranges over B*S
    b = token_id // S
    s = token_id % S

    acc = 0.0
    for off in range(0, H, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < H
        # h[b, s, i, h] => index = (b*S + s)*I*H + i*H + h
        h_idx = (b * S + s) * I * H + i * H + idx
        h_vec = tl.load(h_ptr + h_idx, mask=mask, other=0.0).to(tl.float32)
        w_vec = tl.load(w_ptr + j * H + idx, mask=mask, other=0.0).to(tl.float32)
        acc += tl.sum(h_vec * w_vec, axis=0)

    out_index = token_id * (I * I) + i * I + j
    tl.store(out_ptr + out_index, acc)


class ModelNew(torch.nn.Module):
    def forward(
        grad_corrected: torch.Tensor,
        hidden_states: torch.Tensor,
        activated: torch.Tensor,
        prediction_coef_weight: torch.Tensor,
        correction_coef_weight: torch.Tensor,
        router_weight: torch.Tensor,
        norm_weight: torch.Tensor,
        altup_active_idx: int,
        rms_norm_eps: float,
    ):
        # Move to CUDA if needed (Triton kernels require CUDA tensors)
        device = hidden_states.device
        if device.type != "cuda":
            # Move all inputs to CUDA for Triton execution
            hidden_states = hidden_states.to("cuda")
            activated = activated.to("cuda")
            prediction_coef_weight = prediction_coef_weight.to("cuda")
            correction_coef_weight = correction_coef_weight.to("cuda")
            norm_weight = norm_weight.to("cuda")
            # Ensure grad_corrected is on CUDA (not used for compute, but keep consistency)
            grad_corrected = grad_corrected.to("cuda")

        # Shapes
        B = hidden_states.shape[0]
        S = hidden_states.shape[2]
        I = 3
        H = hidden_states.shape[3]

        # 1) Predict RMSNorm: per-token rstd for all i (i=0 in original, but we launch for all I)
        # hidden_states: [B, S, I, H]; flatten to x_pred of shape [B*S*I, H]
        active_hidden = hidden_states.view(B, S, I, H).contiguous().to(torch.float32)
        x_pred = active_hidden.view(B * S * I, H).cont


def run(*args):
    return ModelNew()(*args)
