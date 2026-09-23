import math
import torch
import torch.nn as nn

# Triton is required
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernels: conv (3x3, stride=2, padding=1) with bias + GELU, and linear GEMV
if TRITON_AVAILABLE:
    @triton.jit
    def conv_ci1_stride2_bias_gelu_kernel(
        input_ptr,  # *f16, [N, 1, In, T]
        weight_ptr, # *f16, [Co, 1, 3, 3]
        bias_ptr,   # *f16, [Co]
        output_ptr, # *f16, [N, Co, Out, T_out]
        N: tl.constexpr,
        Co: tl.constexpr,
        In: tl.constexpr,  # 80
        Out: tl.constexpr, # 80 for conv1
        T: tl.constexpr,   # input time
        T_out: tl.constexpr,  # (T - 3)//2 + 1
        BLOCK_T: tl.constexpr, # tile size along T_out
    ):
        n = tl.program_id(0)
        co = tl.program_id(1)
        o = tl.program_id(2)
        t_block = tl.program_id(3)
        t_offsets = t_block * BLOCK_T + tl.arange(0, BLOCK_T)
        mask_t = t_offsets < T_out

        # Accumulator for this output channel
        acc = tl.zeros([BLOCK_T], dtype=tl.float32)

        # Ci=1, so skip ci loop
        # For each 3x3 kernel element
        for kh in range(3):
            for kw in range(3):
                t_in = 2 * t_offsets + kh  # stride 2, padding=1 => output t = (t_in - 1)/1 when kh=0; but since stride=2, t = (t_in - 1)//1? We need to derive input time index properly.
                # Correction: output time o corresponds to input time index t_in = 2*o - kh + 1
                t_in = 2 * o - kh + 1  # only valid when 0 <= t_in < T
                valid_t = (t_in >= 0) & (t_in < T)
                # Input channel index is 0 (Ci=1), so input pointer: input_ptr + n*In*T + 0*In*T + o*In + t_in
                # Simplify indexing: since Ci=1, input shape [N, 1, In, T] => base = n*In*T
                base_input = n * In * T
                inp_index = base_input + o * In + t_in
                # Load input scalar; mask by valid_t & mask_t
                x = tl.load(input_ptr + inp_index, mask=valid_t & mask_t, other=0.0)
                x = x.to(tl.float32)

                # Load weight scalar (Co, 1, 3, 3)
                base_w = co * 1 * 3 * 3
                w_index = base_w + kh * 3 + kw
                w = tl.load(weight_ptr + w_index).to(tl.float32)

                acc += x * w

        # Add bias
        b = tl.load(bias_ptr + co).to(tl.float32)
        acc += b

        # GELU (tanh approximation): 0.5*x*(1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))
        c0 = 0.7978845608028654  # sqrt(2/pi)
        x3 = acc * acc * acc
        gelu_in = acc + 0.044715 * x3
        gelu_val = 0.5 * acc * (1.0 + tl.tanh(c0 * gelu_in))

        # Store to output
        base_out = n * Co * Out * T_out + co * Out * T_out + o * T_out
        tl.store(output_ptr + base_out + t_offsets, gelu_val, mask=mask_t)


    @triton.jit
    def conv_generic_stride2_bias_gelu_kernel(
        input_ptr,  # *f16, [N, Ci, In, T]
        weight_ptr, # *f16, [Co, Ci, 3, 3]
        bias_ptr,   # *f16, [Co]
        output_ptr, # *f16, [N, Co, Out, T_out]
        N: tl.constexpr,
        Ci: tl.constexpr,
        Co: tl.constexpr,
        In: tl.constexpr,
        Out: tl.constexpr,
        T: tl.constexpr,
        T_out: tl.constexpr,
        BLOCK_T: tl.constexpr,
    ):
        n = tl.program_id(0)
        co = tl.program_id(1)
        o = tl.program_id(2)
        t_block = tl.program_id(3)
        t_offsets = t_block * BLOCK_T + tl.arange(0, BLOCK_T)
        mask_t = t_offsets < T_out

        acc = tl.zeros([BLOCK_T], dtype=tl.float32)

        # For each input channel
        for ci in range(Ci):
            # For each 3x3 kernel element
            for kh in range(3):
                for kw in range(3):
                    t_in = 2 * o - kh + 1  # output o corresponds to input t_in = 2*o - kh + 1
                    valid_t = (t_in >= 0) & (t_in < T)
                    base_input = n * Ci * In * T + ci * In * T + o * In + t_in
                    x = tl.load(input_ptr + base_input, mask=valid_t & mask_t, other=0.0).to(tl.float32)

                    base_w = co * Ci * 3 * 3
                    w_index = base_w + ci * 3 * 3 + kh * 3 + kw
                    w = tl.load(weight_ptr + w_index).to(tl.float32)

                    acc += x * w

        # Add bias
        b = tl.load(bias_ptr + co).to(tl.float32)
        acc += b

        # GELU
        c0 = 0.7978845608028654
        x3 = acc * acc * acc
        gelu_in = acc + 0.044715 * x3
        gelu_val = 0.5 * acc * (1.0 + tl.tanh(c0 * gelu_in))

        base_out = n * Co * Out * T_out + co * Out * T_out + o * T_out
        tl.store(output_ptr + base_out + t_offsets, gelu_val, mask=mask_t)


    @triton.jit
    def linear_bmm_kernel(
        x_ptr,       # *f16, [N, T_out, M=3840]
        w_ptr,       # *f16, [M, K=1024]  (host transposed view of original [K, M])
        y_ptr,       # *f16, [N, T_out, K]
        N: tl.constexpr,
        T_out: tl.constexpr,
        M: tl.constexpr,  # 3840
        K: tl.constexpr,  # 1024
        BLOCK_K: tl.constexpr,
        BLOCK_T: tl.constexpr,
    ):
        # Grid: (N, T_out, K) with internal tiling over K
        n = tl.program_id(0)
        t = tl.program_id(1)
        k_block = tl.program_id(2)
        k_offsets = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < K

        acc = tl.zeros([BLOCK_K], dtype=tl.float32)

        # Iterate over M in tiles of BLOCK_T to reduce register pressure and allow masking
        # Here M=3840 is fixed; we implement a loop over j in tiles.
        # We need to iterate j from 0 to M-1.
        for j_start in range(0, M, BLOCK_T):
            j_offsets = j_start + tl.arange(0, BLOCK_T)
            mask_j = j_offsets < M

            # Load x[n, t, j_offsets] vector (length BLOCK_T), then dot with w[j_offsets, k_offsets] (BLOCK_T x BLOCK_K)
            # x index: n*(T_out*M) + t*M + j_offsets
            x_idx = n * (T_out * M) + t * M + j_offsets
            x_vec = tl.load(x_ptr + x_idx, mask=mask_j, other=0.0).to(tl.float32)  # [BLOCK_T]

            # Load w[j_offsets, k_offsets] as a 2D tile: [BLOCK_T, BLOCK_K]
            w_idx = j_offsets[:, None] * K + k_offsets[None, :]  # [BLOCK_T, BLOCK_K]
            w_tile = tl.load(w_ptr + w_idx, mask=mask_j[:, None] & mask_k[None, :], other=0.0).to(tl.float32)  # [BLOCK_T, BLOCK_K]

            # Dot: [BLOCK_T] * [BLOCK_T, BLOCK_K] -> [BLOCK_K]
            acc += tl.sum(x_vec[:, None] * w_tile, axis=0)

        # Store results
        y_base = n * (T_out * K) + t * K + k_offsets
        tl.store(y_ptr + y_base, acc, mask=mask_k)


    @triton.jit
    def scale_embed_kernel(
        y_ptr,       # *f16, [N, T_out, K]
        scale,       # float32
        N: tl.constexpr,
        T_out: tl.constexpr,
        K: tl.constexpr,
        BLOCK_T: tl.constexpr,
    ):
        n = tl.program_id(0)
        t_block = tl.program_id(1)
        t_offsets = t_block * BLOCK_T + tl.arange(0, BLOCK_T)
        mask_t = t_offsets < T_out
        for k in range(0, K):
            y_base = n * (T_out * K) + t_offsets * K + k
            y_val = tl.load(y_ptr + y_base, mask=mask_t, other=0.0).to(tl.float32)
            y_val *= scale
            tl.store(y_ptr + y_base, y_val, mask=mask_t)


    @triton.jit
    def add_pos_emb_kernel(
        y_ptr,          # *f16, [N, T_out, K]
        pos_ptr,        # *f16, [T_out, K]
        N: tl.constexpr,
        T_out: tl.constexpr,
        K: tl.constexpr,
        BLOCK_T: tl.constexpr,
    ):
        n = tl.program_id(0)
        t_block = tl.program_id(1)
        t_offsets = t_block * BLOCK_T + tl.arange(0, BLOCK_T)
        mask_t = t_offsets < T_out
        for k in range(0, K):
            y_base = n * (T_out * K) + t_offsets * K + k
            y_val = tl.load(y_ptr + y_base, mask=mask_t, other=0.0).to(tl.float32)
            pos_val = tl.load(pos_ptr + t_offsets * K + k, mask=mask_t, other=0.0).to(tl.float32)
            y_val += pos_val
            tl.store(y_ptr + y_base, y_val, mask=mask_t)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_scale = math.sqrt(1024.0)

    def forward(self, input_features, conv2d1_weight, conv2d1_bias,
                conv2d2_weight, conv2d2_bias,
                conv2d3_weight, conv2d3_bias,
                conv_out_weight, positional_embedding):
        """
        input_features: [N, 1, 80, T] bfloat16
        conv* weights: [Co, Ci, 3, 3] bfloat16
        conv_out_weight: [K=1024, M=3840] bfloat16
        positional_embedding: [max_source_positions, d_model=1024] bfloat16
        """
        assert TRITON_AVAILABLE, "Triton is not available"
        device = input_features.device
        dtype = input_features.dtype

        N = input_features.shape[0]
        In = 80
        T = input_features.shape[3]
        # Ensure contiguity and dtype
        input_ci1 = input_features.contiguous().to(torch.bfloat16)

        # Conv1: Ci=1 -> Co=384
        Co1 = 384
        Out1 = In  # output length is In (80) after padding and stride 2
        T_out1 = (T - 3) // 2 + 1  # matches original: (T-3)//2+1

        x1 = torch.empty((N, Co1, Out1, T_out1), device=device, dtype=torch.bfloat16)

        grid1 = (N, Co1, Out1, triton.cdiv(T_out1, 64))
        conv_ci1_stride2_bias_gelu_kernel[grid1](
            input_ci1, conv2d1_weight.contiguous(), conv2d1_bias.contiguous(),
            x1,
            N, Co1, In, Out1, T, T_out1, 64
        )

        # Conv2: Ci=384 -> Co=384
        Co2 = 384
        In2 = Out1  # 80
        Out2 = In2 // 2  # 40
        T_out2 = (T_out1 - 3) // 2 + 1

        x2 = torch.empty((N, Co2, Out2, T_out2), device=device, dtype=torch.bfloat16)

        grid2 = (N, Co2, Out2, triton.cdiv(T_out2, 64))
        conv_generic_stride2_bias_gelu_kernel[grid2](
            x1, conv2d2_weight.contiguous(), conv2d2_bias.contiguous(),
            x2,
            N, 384, Co2, In2, Out2, T_out1, T_out2, 64
        )

        # Conv3: Ci=384 -> Co=384
        Co3 = 384
        In3 = Out2  # 40
        Out3 = In3 // 2  # 20
        T_out3 = (T_out2 - 3) // 2 + 1  # matches provided workloads

        x3 = torch.empty((N, Co3, Out3, T_out3), device=device, dtype=torch.bfloat16)

        grid3 = (N, Co3, Out3, triton.cdiv(T_out3, 64))
        conv_generic_stride2_bias_gelu_kernel[grid3](
            x2, conv2d3_weight.contiguous(), conv2d3_bias.contiguous(),
            x3,
            N, 384, Co3, In3, Out3, T_out2, T_out3, 64
        )

        # Reshape: [N, T_out3, C*F] = [N, T_out3, 384*20]
        y_shape = (N, T_out3, Co3 * Out3)
        y = x3.permute(0, 3, 1, 2).contiguous().view(y_shape).to(torch.bfloat16)

        # Linear projection: [N, T_out3, K=1024] = y @ conv_out_weight.T
        # conv_out_weight is [K=1024, M=3840]; transpose to [M, K]
        w_t = conv_out_weight.transpose(0, 1).contiguous()  # [M=3840, K=1024]
        y_linear = torch.empty((N, T_out3, 1024), device=device, dtype=torch.bfloat16)

        grid_linear = (N, T_out3, triton.cdiv(1024, 128))
        linear_bmm_kernel[grid_linear](
            y, w_t, y_linear,
            N, T_out3, 3840, 1024, 128, 64
        )

        # Scale
        y_scaled = y_linear.clone()
        grid_scale = (N, triton.cdiv(T_out3, 128))
        scale_embed_kernel[grid_scale](
            y_scaled, float(self.embed_scale),
            N, T_out3, 1024, 128
        )

        # Add positional embedding: [1, T_out3, 1024]
        pos_emb = positional_embedding[:T_out3, :].unsqueeze(0).contiguous().to(torch.bfloat16)  # shape [1, T_out3, 1024]
        y_final = y_scaled.clone()
        grid_add = (N, triton.cdiv(T_out3, 128))
        add_pos_emb_kernel[grid_add](
            y_final, pos_emb[0],  # pass the [T_out3, 1024] slice
            N, T_out3, 1024, 128
        )

        return y_final


def run(*args):
    return ModelNew()(*args)
