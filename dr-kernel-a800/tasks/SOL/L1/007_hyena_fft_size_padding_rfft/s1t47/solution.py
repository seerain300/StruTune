class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        # x: (batch, channels, seqlen)
        B, C, L = x.shape
        N = 2 * L
        M = L + 1

        # Ensure x is contiguous and on CUDA for Triton
        x = x.contiguous()
        if not x.is_cuda:
            x = x.cuda()

        # Allocate outputs; flatten to contiguous for kernel
        out_real = torch.empty((B, C, M), dtype=torch.float32, device=x.device).view(-1)
        out_imag = torch.empty((B, C, M), dtype=torch.float32, device=x.device).view(-1)

        # Launch Triton kernel: one program per (b, c) slice
        grid = (B, C)
        _real_rfft_compute_kernel[grid](
            x, out_real, out_imag,
            B, C, L, N, M,
        )

        # Reshape back to (B, C, M)
        out_real = out_real.view(B, C, M)
        out_imag = out_imag.view(B, C, M)

        return out_real, out_imag


def run(*args):
    return ModelNew()(*args)
