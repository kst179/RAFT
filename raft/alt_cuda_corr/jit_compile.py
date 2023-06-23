from torch.utils.cpp_extension import load
from pathlib import Path

def jit_compile():
    path = Path(__file__).parent
    return load(
        name="alt_cuda_corr", 
        sources=[
            path / "correlation.cpp", 
            path / "correlation_kernel.cu"
        ],
        extra_cuda_cflags=["-O3"],
    )