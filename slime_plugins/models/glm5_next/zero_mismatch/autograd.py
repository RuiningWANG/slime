"""Forward serving values while retaining the native backward source."""
import torch

class ServingValue(torch.autograd.Function):
    @staticmethod
    def forward(ctx, native, serving):
        return serving

    @staticmethod
    def backward(ctx, gradient):
        return gradient, None
