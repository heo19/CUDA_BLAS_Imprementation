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


add = """\
extern "C" __global__
void add(float *a, float *b, float *c, size_t n)
{
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n) {
    c[tid] = a[tid] + b[tid];
  }
}
"""

sub = """\
extern "C" __global__
void sub(float *a, float *b, float *c, size_t n)
{
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n) {
    c[tid] = a[tid] - b[tid];
  }
} 
"""

mul = """\
extern "C" __global__
void mul(float *a, float *b, float *c, size_t n)
{
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n) {
    c[tid] = a[tid] * b[tid];
  }
}    
"""


def execute_kernel(kernel_code, kernel_name):
    NUM_THREADS = 512  # Threads per block
    NUM_BLOCKS = 32768  # Blocks per grid

    temp = np.array([2.0], dtype=np.float32)
    n = np.array(NUM_THREADS * NUM_BLOCKS, dtype=np.uint32)
    bufferSize = n * (temp).itemsize

    a = np.random.rand(n).astype(dtype=np.float32)
    b = np.random.rand(n).astype(dtype=np.float32)
    out = np.zeros(n).astype(dtype=np.float32)

    print("Kernel to Execute: ", kernel_code)
    print("Vector A: ", a)
    print("Vector B: ", b)
    print("Vector Out: ", out)
    print("Size of Vector: ", n)
    
    #Create program
    err, prog = nvrtc.nvrtcCreateProgram(str.encode(kernel_code), str.encode(f"{kernel_name}.cu"), 0, [], [])
    
    #compile program
    opts = [b"--fmad=false", b"--gpu-architecture=compute_75"]
    err, = nvrtc.nvrtcCompileProgram(prog, 2, opts)
    
    #Get PTX
    err, ptxSize = nvrtc.nvrtcGetPTXSize(prog)
    ptx = b" " * ptxSize
    err, = nvrtc.nvrtcGetPTX(prog, ptx)
    
    #Initialize CUDA Driver API
    err, = cuda.cuInit(0)
    
    # Retrieve handle for device 0
    err, cuDevice = cuda.cuDeviceGet(0)

    # Create context
    err, context = cuda.cuCtxCreate(0, cuDevice)

    # Load PTX as module data and retrieve function
    ptx = np.char.array(ptx)
    # Note: Incompatible --gpu-architecture would be detected here
    err, module = cuda.cuModuleLoadData(ptx.ctypes.data)
    ASSERT_DRV(err)
    err, kernel = cuda.cuModuleGetFunction(module, str.encode(f"{kernel_name}"))
    ASSERT_DRV(err)
    
    err, dAclass = cuda.cuMemAlloc(bufferSize)
    err, dBclass = cuda.cuMemAlloc(bufferSize)
    err, dOutclass = cuda.cuMemAlloc(bufferSize)

    err, stream = cuda.cuStreamCreate(0)

    err, = cuda.cuMemcpyHtoDAsync(
        dAclass, a.ctypes.data, bufferSize, stream
    )
    err, = cuda.cuMemcpyHtoDAsync(
        dBclass, b.ctypes.data, bufferSize, stream
    )
    
    # The following code example is not intuitive 
    # Subject to change in a future release
    dA = np.array([int(dAclass)], dtype=np.uint64)
    dB = np.array([int(dBclass)], dtype=np.uint64)
    dOut = np.array([int(dOutclass)], dtype=np.uint64)

    args = [dA, dB, dOut, n]
    args = np.array([arg.ctypes.data for arg in args], dtype=np.uint64)

    err, = cuda.cuLaunchKernel(
        kernel,
        NUM_BLOCKS,  # grid x dim
        1,  # grid y dim
        1,  # grid z dim
        NUM_THREADS,  # block x dim
        1,  # block y dim
        1,  # block z dim
        0,  # dynamic shared memory
        stream,  # stream
        args.ctypes.data,  # kernel arguments
        0,  # extra (ignore)
    )

    err, = cuda.cuMemcpyDtoHAsync(
    out.ctypes.data, dOutclass, bufferSize, stream
    )
    err, = cuda.cuStreamSynchronize(stream)

    # Assert values are same after running kernel
    print("Vector Out After Kernel Execution: ", out)
    
    err, = cuda.cuStreamDestroy(stream)
    err, = cuda.cuMemFree(dAclass)
    err, = cuda.cuMemFree(dBclass)
    err, = cuda.cuMemFree(dOutclass)
    err, = cuda.cuModuleUnload(module)
    err, = cuda.cuCtxDestroy(context)

execute_kernel(add, "add")
execute_kernel(sub, "sub")
execute_kernel(mul, "mul")