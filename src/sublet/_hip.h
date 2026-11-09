// HIP adapter for the shared allocator algorithm. Compile as C with ROCm's public headers.
#include <hip/hip_runtime_api.h>

typedef uintptr_t CUdeviceptr;
typedef hipMemGenericAllocationHandle_t CUmemGenericAllocationHandle;
typedef int CUcontext;
typedef hipStream_t CUstream;
typedef hipMemAllocationProp CUmemAllocationProp;
typedef hipMemAccessDesc CUmemAccessDesc;

// HIP device selection avoids the deprecated context API and retained-context leaks.
static hipError_t cuDevicePrimaryCtxRetain(CUcontext* ctx, int device) {
  int supported = 0;
  hipError_t error = hipDeviceGetAttribute(&supported, hipDeviceAttributeVirtualMemoryManagementSupported, device);
  if (error != hipSuccess) return error;
  if (!supported) return hipErrorNotSupported;
  hipMemAllocationProp prop = {.type = hipMemAllocationTypePinned,
                              .location = {.type = hipMemLocationTypeDevice, .id = device}};
  size_t granularity = 0;
  error = hipMemGetAllocationGranularity(&granularity, &prop, hipMemAllocationGranularityMinimum);
  if (error != hipSuccess) return error;
  if (!granularity || (2ull * 1024 * 1024) % granularity) return hipErrorNotSupported;
  *ctx = device;
  return hipSuccess;
}
#define cuCtxGetCurrent hipGetDevice
#define cuCtxSetCurrent hipSetDevice
#define cuCtxSynchronize hipDeviceSynchronize
#define cuMemCreate hipMemCreate
#define cuMemRelease hipMemRelease
#define cuMemHostAlloc hipHostMalloc
#define cuMemFreeHost hipHostFree
#define cuStreamCreate hipStreamCreateWithFlags
#define cuStreamDestroy hipStreamDestroy
#define cuStreamSynchronize hipStreamSynchronize

static hipError_t cuMemAddressReserve(CUdeviceptr* ptr, size_t size, size_t alignment,
                                      CUdeviceptr address, unsigned long long flags) {
  void* reserved = NULL;
  hipError_t error = hipMemAddressReserve(&reserved, size, alignment, (void*)address, flags);
  *ptr = (CUdeviceptr)reserved;
  return error;
}
#define cuMemAddressFree(ptr, size) hipMemAddressFree((void*)(ptr), size)
#define cuMemMap(ptr, size, offset, handle, flags) hipMemMap((void*)(ptr), size, offset, handle, flags)
#define cuMemUnmap(ptr, size) hipMemUnmap((void*)(ptr), size)
static hipError_t cuMemSetAccess(CUdeviceptr ptr, size_t size, const CUmemAccessDesc* access, size_t count) {
  hipError_t error = hipMemSetAccess((void*)ptr, size, access, count);
  if (error != hipSuccess) return error;
  int large_bar = 0;
  error = hipDeviceGetAttribute(&large_bar, hipDeviceAttributeIsLargeBar, access[0].location.id);
  if (error != hipSuccess || !large_bar) return error;
  // PyTorch's ROCm scalar fast path directly dereferences caching-allocator memory
  // on large-BAR devices. VMM addresses need explicit CPU access for that path.
  hipMemAccessDesc host = {.location = {.type = hipMemLocationTypeHost, .id = 0},
                           .flags = hipMemAccessFlagsProtReadWrite};
  return hipMemSetAccess((void*)ptr, size, &host, 1);
}
#define cuMemcpyDtoHAsync(dst, src, size, stream) hipMemcpyDtoHAsync(dst, (hipDeviceptr_t)(src), size, stream)
#define cuMemcpyHtoDAsync(dst, src, size, stream) hipMemcpyHtoDAsync((hipDeviceptr_t)(dst), src, size, stream)

int sublet_init(void) { return hipInit(0) == hipSuccess ? 0 : -1; }

static hipError_t cuMemAlloc(CUdeviceptr* ptr, size_t size) {
  void* allocation = NULL;
  hipError_t error = hipMalloc(&allocation, size);
  *ptr = (CUdeviceptr)allocation;
  return error;
}
#define cuMemFree(ptr) hipFree((void*)(ptr))
// Direct DtoD copies between VMM aliases corrupted optimizer state on the tested
// HIP 7.15 runtime. Move surviving allocations through at most 64 MiB of pinned
// host memory. The caller already synchronizes this move-out path.
static hipError_t cuMemcpyDtoDAsync(CUdeviceptr dst, CUdeviceptr src, size_t size, CUstream stream) {
  const size_t limit = 64ull * 1024 * 1024;
  void* staging = NULL;
  hipError_t error = hipHostMalloc(&staging, size < limit ? size : limit, 0);
  if (error != hipSuccess) return error;
  while (size && error == hipSuccess) {
    size_t chunk = size < limit ? size : limit;
    error = hipMemcpyDtoHAsync(staging, (hipDeviceptr_t)src, chunk, stream);
    if (error == hipSuccess) error = hipStreamSynchronize(stream);
    if (error == hipSuccess) error = hipMemcpyHtoDAsync((hipDeviceptr_t)dst, staging, chunk, stream);
    if (error == hipSuccess) error = hipStreamSynchronize(stream);
    dst += chunk, src += chunk, size -= chunk;
  }
  // Complete any submitted transfer before releasing its host buffer on failure.
  hipStreamSynchronize(stream);
  hipHostFree(staging);
  return error;
}
