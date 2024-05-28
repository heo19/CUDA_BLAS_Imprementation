from cuda import cuda, nvrtc
import numpy as np

# Error Cheking Function
def ASSERT_DRV(err):
    if isinstance(err, cuda.CUresult):
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError("Cuda Error: {}".format(err))
    elif isinstance(err, nvrtc.nvrtcResult):
        if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            raise RuntimeError("Nvrtc Error: {}".format(err))
    else:
        raise RuntimeError("Unknown error type: {}".format(err))


negation = """\
extern "C" __global__
void neg(float *a, float *b, size_t n)
{
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n) {
    b[tid] = -a[tid];
  }
}
"""

addition = """\
extern "C" __global__
void add(float *a, float *b, float *c, size_t n)
{
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n) {
    c[tid] = a[tid] + b[tid];
  }
}
"""

subtraction = """\
extern "C" __global__
void sub(float *a, float *b, float *c, size_t n)
{
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n) {
    c[tid] = a[tid] - b[tid];
  }
} 
"""

multiplication = """\
extern "C" __global__
void mul(float *a, float *b, float *c, size_t n)
{
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n) {
    c[tid] = a[tid] * b[tid];
  }
}    
"""

