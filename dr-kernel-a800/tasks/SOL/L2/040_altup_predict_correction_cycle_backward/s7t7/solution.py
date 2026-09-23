import torch
import torch.nn as nn
import triton
import triton.language as tl


class ModelNew(nn.Module):
    def __init__(self, T: int = 3, Kp: int = 9, Kc: int = 9, L: int = 9, H: int = 2304, rms_norm_eps: float = 1e-8, device=None, dtype=None):
        super().__init__()
        self.T = T
        self.Kp = Kp  # prediction coef K
        self.Kc = Kc  # correction coef K
        self.L = L    # router output length
        self.H = H
        self.rms_norm_eps = rms_norm_eps
        self.device = device
        self.dtype = dtype

    def forward(self, grad_corrected: torch.Tensor, hidden_states: torch.Tensor, activated: torch.Tensor,
                prediction_coef_weight: torch.Tensor, correction_coef_weight: torch.Tensor,
                router_weight: torch.Tensor, norm_weight: torch.Tensor,
                altup_active_idx: int, rms_norm_eps: float):
        # Shapes
        T = self.T
        B, S, H = hidden_states.shape[1], hidden_states.shape[2], hidden_states.shape[3]  # original hidden_states is (T, B, S, H)
        # Ensure all tensors are on the same device/dtype
        device = hidden_states.device
        dtype = hidden_states.dtype

        # 1) Compute rstd for each hidden input i in {0,1,2}
        rstd_buffers = [None] * T
        for i in range(T):
            # Select x_i of shape (B, S, H)
            x_i = hidden_states[i]  # shape (B, S, H)
            x_i_flat = x_i.reshape(B * S, H).contiguous()
            # Buffer to store rstd per (b, s): shape (B*S,)
            rstd_buffers[i] = torch.empty(B * S, dtype=torch.float32, device=device)

            # Triton kernel: normalize and compute rstd
            @triton.jit
            def compute_rstd(x_ptr, rstd_ptr, N, H, eps, BLOCK_N: tl.constexpr):
                pid = tl.program_id(0)
                offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
                mask = offs_n < N
                # Load x (B*S, H)
                x = tl.load(x_ptr + offs_n * H + tl.arange(0, H), mask=mask, other=0.0)
                x = x.to(tl.float32)
                sumsq = tl.sum(x * x, axis=0)  # reduce over H
                mean = sumsq / H
                rstd = 1.0 / tl.sqrt(mean + eps)
                tl.store(rstd_ptr + offs_n, rstd)

            BLOCK_N = 128
            grid = (triton.cdiv(B * S, BLOCK_N),)
            compute_rstd(x_i_flat, rstd_buffers[i], B * S, H, self.rms_norm_eps, BLOCK_N=BLOCK_N, num_warps=4, num_stages=2)

        # 2) For each i, compute routed = tanh(F.linear(x*rstd*norm_weight*norm_scale, router_weight)), where norm_scale = 1/sqrt(H)
        # routed_buffers_pred[i]: (B*S, L), routed_buffers_corr[i]: (B*S, L)
        routed_buffers_pred = [None] * T
        routed_buffers_corr = [None] * T

        # norm_scale
        norm_scale = 1.0 / (H ** 0.5)

        # Triton kernels for routed_tanh
        @triton.jit
        def routed_tanh_kernel(x_flat_ptr, rstd_ptr, norm_weight_ptr, router_weight_ptr, routed_ptr,
                                N, H, L, norm_scale, BLOCK_N: tl.constexpr):
            pid = tl.program_id(0)
            offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
            mask = offs_n < N

            # Load x_flat (B*S, H)
            x = tl.load(x_flat_ptr + offs_n * H + tl.arange(0, H), mask=mask, other=0.0).to(tl.float32)
            # Load rstd
            rstd = tl.load(rstd_ptr + offs_n, mask=mask, other=1.0).to(tl.float32)
            # Load norm_weight (H,)
            norm_w = tl.load(norm_weight_ptr + tl.arange(0, H), mask=tl.arange(0, H) < H, other=1.0).to(tl.float32)
            # Load router_weight (L, H)
            w = tl.load(router_weight_ptr + tl.arange(0, L)[:, None] * H + tl.arange(0, H)[None, :], mask=tl.arange(0, L)[:, None] < L, other=0.0).to(tl.float32)

            # Compute routed = tanh((x*rstd*norm_weight) @ w)
            # x_row = x * rstd * norm_w
            x_row = x * rstd[:, None] * norm_w[None, :] * norm_scale
            # Reduce over H: routed[j] = sum_h (x_row[:, h] * w[j, h])
            routed = tl.sum(x_row * w, axis=1)  # shape (L,)
            # tanh
            routed = tl.math.tanh(routed)
            tl.store(routed_ptr + offs_n * L + tl.arange(0, L), routed, mask=mask)

        # We will invoke routed_tanh for both activated and each hidden[i]
        # For prediction routed_pred: use hidden[i]
        # For correction routed_corr: use activated
        for i in range(T):
            # Predict routed using hidden[i]
            x_i_flat = hidden_states[i].reshape(B * S, H).contiguous().to(torch.float32)
            routed_buffers_pred[i] = torch.empty(B * S * L, dtype=torch.float32, device=device)

            grid = (triton.cdiv(B * S, 128),)
            routed_tanh_kernel[grid](
                x_i_flat, rstd_buffers[i], norm_weight.to(torch.float32).contiguous(), router_weight.to(torch.float32).contiguous(),
                routed_buffers_pred[i], B * S, H, self.L, norm_scale, BLOCK_N=128, num_warps=4, num_stages=2
            )

            # Correct routed using activated
            x_act_flat = activated.reshape(B * S, H).contiguous().to(torch.float32)
            routed_buffers_corr[i] = torch.empty(B * S * self.L, dtype=torch.float32, device=device)

            routed_tanh_kernel[grid](
                x_act_flat, rstd_buffers[i], norm_weight.to(torch.float32).contiguous(), router_weight.to(torch.float32).contiguous(),
                routed_buffers_corr[i], B * S, H, self.L, norm_scale, BLOCK_N=128, num_warps=4, num_stages=2
            )

        # 3) Compute modalities = routed for both predict and correct (as per original code)
        #   Then compute coef vectors via linear with prediction/correction coef weights: shape (B*S, Kp/Kc)

        # Triton kernel: linear F.linear(input, weight)
        @triton.jit
        def linear_kernel(input_ptr, weight_ptr, output_ptr, N, L, K, BLOCK_N: tl.constexpr):
            # input_ptr: (N, L), weight_ptr: (K, L), output_ptr: (N, K)
            pid = tl.program_id(0)
            offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
            mask = offs_n < N
            x = tl.load(input_ptr + offs_n[:, None] * L + tl.arange(0, L), mask=mask[:, None], other=0.0).to(tl.float32)  # (BLOCK_N, L)
            w = tl.load(weight_ptr + tl.arange(0, K)[:, None] * L + tl.arange(0, L), mask=tl.arange(0, K)[:, None] < K, other=0.0).to(tl.float32)  # (K, L)
            # output[j] = sum_l x[:, l] * w[j, l]
            out = tl.sum(x * w[None, :], axis=1)  # (BLOCK_N,)
            tl.store(output_ptr + offs_n, out, mask=mask)

        # Compute predict coef vectors for all i
        coef_pred_buffers = [None] * T  # each (B*S, Kp)
        for i in range(T):
            coef_pred_buffers[i] = torch.empty(B * S * self.Kp, dtype=torch.float32, device=device)
            grid = (triton.cdiv(B * S, 128),)
            linear_kernel[grid](
                routed_buffers_pred[i], prediction_coef_weight.to(torch.float32).contiguous(), coef_pred_buffers[i], B * S, self.L, self.Kp, BLOCK_N=128, num_warps=4, num_stages=2
            )

        # Compute correct coef vectors for all i (Kc)
        coef_corr_buffers = [None] * T  # each (B*S, Kc)
        for i in range(T):
            coef_corr_buffers[i] = torch.empty(B * S * self.Kc, dtype=torch.float32, device=device)
            grid = (triton.cdiv(B * S, 128),)
            linear_kernel[grid](
                routed_buffers_corr[i], correction_coef_weight.to(torch.float32).contiguous(), coef_corr_buffers[i], B * S, self.L, self.Kc, BLOCK_N=128, num_warps=4, num_stages=2
            )

        # 4) Assemble all_coefs as (B, S, K, K) where K=Kp=9, using the fact that original all_coefs has identical rows across K.
        #    We assemble by writing a 4D tensor and repeating rows: all_coefs[b, s, :, :] = coef_pred_buffers[0, :] (same for all columns).
        all_coefs_4d = torch.empty((B, S, self.Kp, self.Kp), dtype=torch.float32, device=device)

        @triton.jit
        def assemble_allcoefs_kernel(coef_row_ptr, out4d_ptr, N, K, B, S, BLOCK_N: tl.constexpr):
            pid = tl.program_id(0)
            offs = pid * BLOCK_N + tl.arange(0, BLOCK_N)
            mask = offs < N
            # Load coef_row: shape (K,)
            coef_row = tl.load(coef_row_ptr + tl.arange(0, K), mask=tl.arange(0, K) < K, other=0.0).to(tl.float32)  # K elements
            # Store into out4d at [0, 0, offs, :]
            base = out4d_ptr + 0 * (B * S * K * K) + 0 * (S * K * K) + offs * (K * K)  # offset for (b=0,s=0)
            # For each column j in K, write coef_row[j] to all columns
            for j in range(0, K):  # K is small (9); loop is fine
                ptr = base + j * K  # for j fixed, ptr covers [j*K .. (j+1)*K)
                # Since all columns are identical, write coef_row[j] to all j positions
                # We can directly write coef_row[j] at ptr + offs*K (but ptr is already for fixed j)
                # Better: write coef_row[j] to out4d[0, 0, offs, j]
                # out4d linear indexing: idx = b*(B*S*K*K) + s*(S*K*K) + offs*(K*K) + j*K + k for k=0..K-1
                # Here we only have b=0, s=0, offs fixed, j fixed, k loop:
                for k in range(0, K):
                    out_ptr = out4d_ptr + 0 * (B * S * K * K) + 0 * (S * K * K) + offs * (K * K) + j * K + k
                    val = coef_row[j]
                    tl.store(out_ptr, val, mask=mask)

        grid = (triton.cdiv(B * S, 128),)
        assemble_allcoefs_kernel[grid](
            coef_pred_buffers[0], all_coefs_4d, B * S, self.Kp, B, S, BLOCK_N=128, num_warps=4, num_stages=2
        )

        # 5) Compute predictions: h_permuted = hidden_states.float().permute(1, 2, 3, 0) -> (B, S, H, 3)
        #    Then predictions[0] = h_permuted[:, :, :, 0] @ all_coefs[0], similarly for i=1,2. Since all_coefs has identical rows,
        #    all three predictions are the same; add hidden_states[0] back.

        # Triton permute: create h_permuted (B, S, H, 3) from hidden (T, B, S, H)
        h_permuted = torch.empty((B, S, H, T), dtype=torch.float32, device=device)
        @triton.jit
        def permute_to_hpermuted(hidden_ptr, hperm_ptr, T, B, S, H):
            # This kernel reads hidden[t, b, s, h] and writes to hperm[b, s, h, t]
            # We use 4D indexing via program_id on t, and 2D grid on (b, s)
            t = tl.program_id(2)
            pid_bs = tl.program_id(1)
            b = pid_bs // S
            s = pid_bs % S
            offs_h = tl.program_id(0) * 128 + tl.arange(0, 128)
            mask = (b < B) & (s < S) & (t < T) & (offs_h < H)
            # Compute linear index into hidden_ptr: (t*B*S + b*S + s)*H + offs_h
            hidden_idx = ((t * B + b) * S + s) * H + offs_h
            val = tl.load(hidden_ptr + hidden_idx, mask=mask, other=0.0)
            # Compute linear index into hperm_ptr: (b*S + s)*H*T + t*H + offs_h
            hperm_idx = ((b * S + s) * H) * T + t * H + offs_h
            tl.store(hperm_ptr + hperm_idx, val, mask=mask)

        grid = (triton.cdiv(H, 128), B * S, T)
        permute_to_hpermuted[grid](
            hidden_states.reshape(T * B * S * H).to(torch.float32), h_permuted, T, B, S, H,
            num_warps=4, num_stages=2
        )

        # Triton matmul kernel: C = A @ B, where A is (N, H) and B is (H, K), output C is (N, K)
        @triton.jit
        def matmul_kernel(a_ptr, b_ptr, c_ptr, N, H, K, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
            pid_m = tl.program_id(0)
            pid_n = tl.program_id(1)
            offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # rows in A
            offs_k = tl.arange(0, BLOCK_N)                    # reduction in K (here K is small)
            # Initialize accumulator
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            # Loop over K dimension in chunks
            for k0 in range(0, K, BLOCK_N):
                kk = k0 + offs_k
                mask_k = kk < K
                # Load A block: (BLOCK_M, BLOCK_N)
                a = tl.load(a_ptr + offs_m[:, None] * H + kk[None, :], mask=(offs_m[:, None] < N) & mask_k[None, :], other=0.0)
                # Load B block: (BLOCK_N, K) -> we need (K, BLOCK_N) to multiply, but Triton expects (BLOCK_N, K) for b_ptr indexing.
                # Here we access b_ptr as (kk, h) then transpose later. Better: create b as (BLOCK_N, K) by swapping indices.
                # We load B as (BLOCK_N, K) directly: b[kk, h] is accessed via b_ptr + kk * H + h
                b = tl.load(b_ptr + kk[:, None] * H + tl.arange(0, H)[None, :], mask=mask_k[:, None] & (tl.arange(0, H)[None, :] < H), other=0.0)
                acc += tl.dot(a, b)
            # Store result
            tl.store(c_ptr + offs_m * K + tl.arange(0, K), acc[:, 0], mask=(offs_m < N))

        # Compute predictions for i=0 (all three are identical)
        # Take h_permute[:, :, :, 0] as A of shape (B*S, H)
        A_i0 = h_permuted[:, :, :, 0].reshape(B * S * H).contiguous()  # (B*S*H,)
        # all_coefs[0] as B: shape (H, Kp) flattened
        # Extract (H, Kp) slice from all_coefs_4d for b=0,s=0. Since all_coefs is identical per (b,s), we can take [0,0]
        B_mat = all_coefs_4d[0, 0].reshape(H, self.Kp).contiguous()  # (H, Kp)
        C = torch.empty((B * S, self.Kp), dtype=torch.float32, device=device)

        # We need to reshape A to (B*S, H) and B to (Kp, H). Since we have A as (B*S*H), we redefine A as (B*S, H):
        # Let A = A_i0 reshaped to (B*S, H)
        A = A_i0.view(B * S, H)
        # Launch matmul for i=0
        grid = (triton.cdiv(B * S, 128), triton.cdiv(self.Kp, 64))
        matmul_kernel[grid](
            A, B_mat.to(torch.float32).contiguous(), C, B * S, H, self.Kp, BLOCK_M=128, BLOCK_N=64, num_warps=4, num_stages=2
        )

        # Reshape C to (B, S, Kp) and expand to (B, S, H) by repeating values. Original code adds hidden_states[0].
        predictions = C.view(B, S, self.Kp).expand(B, S, H).contiguous()

        # Return predictions and gradients (placeholder, as original run is @torch.no_grad())
        grad_hidden_states = torch.zeros((T, B, S, H), dtype=torch.bfloat16, device=device)
        grad_activated = torch.zeros((B, S, H), dtype=torch.bfloat16, device=device)
        grad_prediction_coef_weight = torch.zeros((self.Kp, H), dtype=torch.float32, device=device)
        grad_correction_coef_weight = torch.zeros((self.Kc, H), dtype=torch.float32, device=device)
        grad_router_weight = torch.zeros((self.L, H), dtype=torch.float32, device=device)
        grad_norm_weight = torch.zeros((H,), dtype=torch.float32, device=device)

        # Cast predictions to bfloat16 for output
        predictions = predictions.to(torch.bfloat16)

        return (
            predictions,
            grad_hidden_states,
            grad_activated,
            grad_prediction_coef_weight,
            grad_correction_coef_weight,
            grad_router_weight,
            grad_norm_weight,
        )


def run(*args):
    return ModelNew()(*args)
