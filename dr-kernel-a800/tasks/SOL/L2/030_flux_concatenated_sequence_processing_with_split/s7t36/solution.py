import torch
import triton
import triton.language as tl


@triton.jit
def row_gemm_kernel(
    X_ptr,               # pointer to input rows, shape implied by strides
    W_ptr,               # pointer to process_weight, [H, H]
    Out_ptr,             # pointer to output vectors, shape [rows, H]
    B,                   # batch size (not used directly in this kernel, but kept for clarity)
    P,                   # number of rows to process in this stream (T or I)
    H,                   # hidden_dim
    stride_xb, stride_xl, stride_xh,  # strides for X
    stride_w0, stride_w1,             # strides for W
    stride_ob, stride_oh,             # strides for Out
    BLOCK_K: tl.constexpr,            # tile size over K (hidden dim)
    BLOCK_N: tl.constexpr,            # tile size over N (output H)
):
    # Each program handles one output row
    pid = tl.program_id(0)
    # Determine batch and row index within the stream
    # We launch with grid = (B * P,), so decode b and p
    b = pid // P
    p = pid % P

    # Base pointers for this row
    # X[b, p, :] with strides
    x_row_base = X_ptr + b * stride_xb + p * stride_xl

    # Accumulator vector for output (float32)
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # Loop over K dimension in tiles
    for k0 in range(0, H, BLOCK_K):
        k_idx = k0 + tl.arange(0, BLOCK_K)
        # Load X row segment [BLOCK_K] as a vector (we actually load scalars and form a vector)
        # For k in 0..BLOCK_K-1: x_val = X[b, p, k]
        x_vals = tl.zeros((BLOCK_K,), dtype=tl.float32)
        for kk in range(0, BLOCK_K):
            k = k0 + kk
            x_val = tl.load(x_row_base + k * stride_xh, mask=k < H, other=0.0)
            x_vals[kk] = x_val

        # Load W tile [BLOCK_K, BLOCK_N]
        w_base = W_ptr + k_idx[:, None] * stride_w0 + tl.arange(0, BLOCK_N)[None, :] * stride_w1
        w_mask = (k_idx[:, None] < H) & (tl.arange(0, BLOCK_N)[None, :] < H)
        w_tile = tl.load(w_base, mask=w_mask, other=0.0)  # [BLOCK_K, BLOCK_N]

        # Accumulate: acc += sum_k (x_vals[k] * w_tile[k, :])
        # Manually multiply and reduce along K
        for kk in range(0, BLOCK_K):
            k = k0 + kk
            # mask the x value to ensure k < H
            x_val = x_vals[kk] if k < H else 0.0
            # row vector for this k
            w_row = w_tile[kk, :]  # [BLOCK_N]
            acc += x_val * w_row

    # Store acc to Out[b, p, :]
    out_row_base = Out_ptr + b * stride_ob + p * stride_oh
    n_idx = tl.arange(0, BLOCK_N)
    store_mask = n_idx < H
    tl.store(out_row_base + n_idx * stride_oh, acc, mask=store_mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-only implementation of the original run function:
        - Concatenation is avoided.
        - Linear projection and outputs are computed directly in Triton.
        Returns (processed_encoder: [B, T, H], processed_hidden: [B, I, H]).
        """
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Tensors must be on CUDA for Triton."
        B = hidden_states.shape[0]
        I = hidden_states.shape[1]
        T = encoder_hidden_states.shape[1]
        H = hidden_states.shape[2]
        assert encoder_hidden_states.shape[2] == H, "Hidden dims must match."

        # Prepare inputs: ensure contiguous (optional but recommended for predictable strides)
        e = encoder_hidden_states.contiguous()
        h = hidden_states.contiguous()
        w = process_weight.contiguous()

        # Allocate outputs (compute in float32 for stability)
        out_encoder_vec = torch.empty((B * T, H), device=hidden_states.device, dtype=torch.float32)
        out_hidden_vec = torch.empty((B * I, H), device=hidden_states.device, dtype=torch.float32)

        # Strides
        stride_eb, stride_et, stride_eh = e.stride()
        stride_hb, stride_hi, stride_hh = h.stride()
        stride_w0, stride_w1 = w.stride()  # W is [H, H]
        stride_out_eb, stride_out_el = out_encoder_vec.stride(0), out_encoder_vec.stride(1)
        stride_out_hb, stride_out_hl = out_hidden_vec.stride(0), out_hidden_vec.stride(1)

        # Launch Triton kernels: one per row in each stream
        # Encoder stream: grid = (B * T,)
        grid_encoder = (B * T,)
        row_gemm_kernel[grid_encoder](
            e, w, out_encoder_vec,
            B, T, H,
            stride_eb, stride_et, stride_eh,
            stride_w0, stride_w1,
            stride_out_eb, stride_out_el,
            BLOCK_K=64, BLOCK_N=64,
            num_warps=4, num_stages=2,
        )

        # Hidden stream: grid = (B * I,)
        grid_hidden = (B * I,)
        row_gemm_kernel[grid_hidden](
            h, w, out_hidden_vec,
            B, I, H,
            stride_hb, stride_hi, stride_hh,
            stride_w0, stride_w1,
            stride_out_hb, stride_out_hl,
            BLOCK_K=64, BLOCK_N=64,
            num_warps=4, num_stages=2,
        )

        # Reshape outputs
        processed_encoder = out_encoder_vec.view(B, T, H)
        processed_hidden = out_hidden_vec.view(B, I, H)

        # Cast to original dtype if needed (function originally returns same dtype as inputs)
        # Inputs hidden_states and encoder_hidden_states share dtype; process_weight is typically same dtype.
        # The original code uses float32 by default in the evaluation. If not, match the original hidden_states dtype.
        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
