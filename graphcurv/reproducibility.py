import os
import torch


def configure(device='cpu'):
    # deterministic kernels, no tf32
    if str(device).startswith('cuda'):
        if os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8':
            raise RuntimeError('set CUBLAS_WORKSPACE_CONFIG=:4096:8 before starting python')
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
