class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Preserve original signature: run(inputs, target_sparsity)
        if len(args) == 2:
            inputs, target_sparsity = args
            # Ensure inputs are float32 and contiguous
            x = inputs.to(torch.float32).contiguous()
            B, S, F = x.shape
            stride_b, stride_s, stride_f = x.stride()

            # Output buffer in float32
            out = torch.empty((B, S, F), dtype=torch.float32, device=x.device)

            # Launch Triton run
            run_triton(x, float(target_sparsity), out, B, S, F, stride_b, stride_s, stride_f)

            # Return in bfloat16 to match original behavior
            return out.to(torch.bfloat16)
        elif len(args) == 1:
            # Default sparsity if only one argument is provided
            inputs = args[0]
            x = inputs.to(torch.float32).contiguous()
            # We need target_sparsity; default to 0.01
            target_sparsity = 0.01
            B, S, F = x.shape
            stride_b, stride_s, stride_f = x.stride()
            out = torch.empty((B, S, F), dtype=torch.float32, device=x.device)
            run_triton(x, float(target_sparsity), out, B, S, F, stride_b, stride_s, stride_f)
            return out.to(torch.bfloat16)
        else:
            # If more args, assume second is target_sparsity
            if len(args) > 1 and isinstance(args[1], (float, int)):
                inputs = args[0]
                target_sparsity = float(args[1])
                x = inputs.to(torch.float32).contiguous()
                B, S, F = x.shape
                stride_b, stride_s, stride_f = x.stride()
                out = torch.empty((B, S, F), dtype=torch.float32, device=x.device)
                run_triton(x, target_sparsity, out, B, S, F, stride_b, stride_s, stride_f)
                return out.to(torch.bfloat16)
            # Fallback
            inputs = args[0]
            x = inputs.to(torch.float32).contiguous()
            B, S, F = x.shape
            stride_b, stride_s, stride_f = x.stride()
            out = torch.empty((B, S, F), dtype=torch.float32, device=x.device)
            run_triton(x, 0.01, out, B, S, F, stride_b, stride_s, stride_f)
            return out.to(torch.bfloat16)


def run(*args):
    return ModelNew()(*args)
