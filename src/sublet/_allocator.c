// sublet's allocator: the memory of a lendable tensor, and how a loan of it serves PyTorch's allocations.
//
// A lendable tensor (a region) is npages physical pages of 2 MiB, mapped twice: at the tensor's addresses (base), and
// in an arena of the same size. sublet_hook points the cudaMalloc and cudaFree that PyTorch's caching allocator calls
// (in libc10_cuda.so) at sublet_cuda_malloc and sublet_cuda_free, which call the real ones outside loans. During a loan
// of a region, a new segment on its device gets the cheapest memory that fits:
//
//   1. a stretch of the arena over pages that hold nothing to keep: already mapped, it costs nothing;
//   2. new device memory, if the GPU has enough free;
//   3. the stretch of the arena with the fewest kept pages, which are copied to pinned CPU memory first;
//   4. when no stretch is long enough, pages from anywhere in the region, mapped at new addresses (gathered).
//
// Before 3 and 4, the request is declined once, so that PyTorch frees the memory it caches and asks again.
//
// A kept page is copied once per loan; after that it costs nothing either. When the loan ends, PyTorch gives back its
// free segments, the segments still holding allocations are moved out of the region (copied to new device memory,
// mapped at the same addresses), and the kept pages are copied back.
//
// The CUDA driver and runtime are opened at runtime (PyTorch has loaded them), so building needs only a C compiler.

#define _GNU_SOURCE
#include <dlfcn.h>
#include <elf.h>
#include <link.h>
#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <unistd.h>

#ifdef SUBLET_HIP
#include "_hip.h"
#else
typedef unsigned long long CUdeviceptr;
typedef unsigned long long CUmemGenericAllocationHandle;
typedef int CUresult;
typedef void* CUstream;
typedef void* CUcontext;
typedef struct {
  int type;  // CU_MEM_LOCATION_TYPE_DEVICE
  int id;
} CUmemLocation;
typedef struct {
  int type;                  // CU_MEM_ALLOCATION_TYPE_PINNED
  int requestedHandleTypes;  // CU_MEM_HANDLE_TYPE_NONE
  CUmemLocation location;
  void* win32HandleMetaData;
  unsigned char compressionType, gpuDirectRDMACapable;
  unsigned short usage;
  unsigned char reserved[4];
} CUmemAllocationProp;
typedef struct {
  CUmemLocation location;
  int flags;  // CU_MEM_ACCESS_FLAGS_PROT_READWRITE
} CUmemAccessDesc;

#define DRIVER_FUNCTIONS(X)                                                                                         \
  X(cuDevicePrimaryCtxRetain, "cuDevicePrimaryCtxRetain", CUcontext*, int)                                         \
  X(cuCtxGetCurrent, "cuCtxGetCurrent", CUcontext*)                                                                \
  X(cuCtxSetCurrent, "cuCtxSetCurrent", CUcontext)                                                                 \
  X(cuCtxSynchronize, "cuCtxSynchronize")                                                                          \
  X(cuMemAlloc, "cuMemAlloc_v2", CUdeviceptr*, size_t)                                                             \
  X(cuMemFree, "cuMemFree_v2", CUdeviceptr)                                                                        \
  X(cuMemAddressReserve, "cuMemAddressReserve", CUdeviceptr*, size_t, size_t, CUdeviceptr, unsigned long long)     \
  X(cuMemAddressFree, "cuMemAddressFree", CUdeviceptr, size_t)                                                     \
  X(cuMemCreate, "cuMemCreate", CUmemGenericAllocationHandle*, size_t, const CUmemAllocationProp*, unsigned long long) \
  X(cuMemRelease, "cuMemRelease", CUmemGenericAllocationHandle)                                                    \
  X(cuMemMap, "cuMemMap", CUdeviceptr, size_t, size_t, CUmemGenericAllocationHandle, unsigned long long)           \
  X(cuMemUnmap, "cuMemUnmap", CUdeviceptr, size_t)                                                                 \
  X(cuMemSetAccess, "cuMemSetAccess", CUdeviceptr, size_t, const CUmemAccessDesc*, size_t)                         \
  X(cuMemHostAlloc, "cuMemHostAlloc", void**, size_t, unsigned int)                                                \
  X(cuMemFreeHost, "cuMemFreeHost", void*)                                                                         \
  X(cuMemcpyDtoHAsync, "cuMemcpyDtoHAsync_v2", void*, CUdeviceptr, size_t, CUstream)                               \
  X(cuMemcpyHtoDAsync, "cuMemcpyHtoDAsync_v2", CUdeviceptr, const void*, size_t, CUstream)                         \
  X(cuMemcpyDtoDAsync, "cuMemcpyDtoDAsync_v2", CUdeviceptr, CUdeviceptr, size_t, CUstream)                         \
  X(cuStreamCreate, "cuStreamCreate", CUstream*, unsigned int)                                                     \
  X(cuStreamDestroy, "cuStreamDestroy_v2", CUstream)                                                               \
  X(cuStreamSynchronize, "cuStreamSynchronize", CUstream)

#define DECLARE(name, symbol, ...) static CUresult (*name)(__VA_ARGS__);
DRIVER_FUNCTIONS(DECLARE)

// Returns 0 on success, -1 if libcuda.so.1 or one of its symbols is missing.
int sublet_init(void) {
  void* h = dlopen("libcuda.so.1", RTLD_NOW);
  if (!h) return -1;
#define LOAD(name, symbol, ...) \
  if (!(name = (CUresult(*)(__VA_ARGS__))dlsym(h, symbol))) return -1;
  DRIVER_FUNCTIONS(LOAD)
  return 0;
}

#endif  // SUBLET_HIP

#ifdef SUBLET_HIP
#define ALLOCATOR_LIBRARY "libc10_hip.so"
#define RUNTIME_LIBRARY "libamdhip64.so"
#define RUNTIME_SYMBOL(name) "hip" name
#else
#define ALLOCATOR_LIBRARY "libc10_cuda.so"
#define RUNTIME_LIBRARY "libcudart.so"
#define RUNTIME_SYMBOL(name) "cuda" name
#endif

#define PAGE (2ull * 1024 * 1024)
#define CHUNK 32  // CPU slots per chunk of pinned memory (64 MiB)
#define MAX_DEVICES 64
#define CUDA_SUCCESS 0
#define CUDA_ERROR_MEMORY_ALLOCATION 2  // cudaErrorMemoryAllocation, which PyTorch's caching allocator recovers from

typedef struct Region {
  int device;
  CUcontext ctx;  // the device's primary context, which PyTorch uses
  CUdeviceptr base, arena;
  int64_t npages, nmapped;  // pages mapped at both addresses so far: all of them, once made
  CUmemGenericAllocationHandle* pages;
  CUmemGenericAllocationHandle scratch;  // one page, mapped over allocations that could not be moved out
  uint8_t* keep;    // page holds data that comes back when the loan ends
  uint8_t* copied;  // kept page copied to the CPU during this loan
  uint8_t* busy;    // page backs a segment
  uint8_t* hole;    // the arena has a segment's own memory there, not the page
  uint8_t* lent;    // page lent during this loan
  uint8_t* want;    // scratch: pages to copy
  int64_t* slot;    // CPU slot holding the copy of each copied page
  char** slot_chunks;  // pinned when first needed, and reused from one loan to the next
  int64_t nslot_chunks;
  int64_t* free_slots;
  int64_t nfree_slots;
  CUstream copy_stream;
  int64_t extra, extra_start, extra_peak;  // bytes of new device memory in segments; at the start of the loan; peak
  int64_t moved;                           // bytes moved out of the region when the loan ended
  int64_t gathered;                        // bytes of segments gathered during the loan
  int64_t nsegments;                       // segments made from r and not freed yet
  int dropped;                             // the tensor is gone: destroy r once its last segment is freed
} Region;

enum { STRETCH, GATHERED, NEW, MOVED, BLANK };

typedef struct Segment {
  CUdeviceptr ptr;
  int64_t npages;
  int kind;
  int in_arena;   // ptr is a stretch of the arena (a STRETCH, or a segment moved or blanked from one)
  int64_t* page;  // region page behind each page, for STRETCH and GATHERED
  CUmemGenericAllocationHandle handle;  // its own memory, for MOVED
  Region* r;
  struct Segment* next;
} Segment;

static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
static Segment* segments;            // all live segments made during loans
static Region* lender[MAX_DEVICES];  // region lent on each device, during a loan

static CUmemAllocationProp device_memory(int device) {
  CUmemAllocationProp prop = {.type = 1, .location = {.type = 1, .id = device}};
  return prop;
}

static int allow_access(CUdeviceptr ptr, size_t size, int device) {
  CUmemAccessDesc access = {.location = {.type = 1, .id = device}, .flags = 3};
  return cuMemSetAccess(ptr, size, &access, 1);
}

// Makes r's device current on the calling thread, which may be using another one; returns the context to restore.
static CUcontext enter(Region* r) {
  CUcontext previous = 0;
  cuCtxGetCurrent(&previous);
  cuCtxSetCurrent(r->ctx);
  return previous;
}

// Maps n region pages, `page[i]` or `first + i` if page is NULL, at ptr.
static int map_pages(Region* r, CUdeviceptr ptr, const int64_t* page, int64_t first, int64_t n) {
  for (int64_t i = 0; i < n; i++)
    if (cuMemMap(ptr + i * PAGE, PAGE, 0, r->pages[page ? page[i] : first + i], 0) != 0) {
      if (i) cuMemUnmap(ptr, i * PAGE);
      return -1;
    }
  if (allow_access(ptr, n * PAGE, r->device) != 0) {
    cuMemUnmap(ptr, n * PAGE);
    return -1;
  }
  return 0;
}

// ---------------------------------------------------------------------------------------------------------------------
// Pointing PyTorch's cudaMalloc and cudaFree at sublet

typedef int (*cuda_malloc_t)(void**, size_t);
typedef int (*cuda_free_t)(void*);
typedef int (*cuda_int_t)(int*);
static cuda_malloc_t real_malloc;
static cuda_free_t real_free;
static cuda_int_t real_get_device, real_exchange_capture_mode;
static int32_t* log_level;  // c10's FLAGS_caffe2_log_level: PyTorch 2.13+ logs a warning for each failed cudaMalloc
static int hooked;

int sublet_cuda_malloc(void** ptr, size_t size);
int sublet_cuda_free(void* ptr);

#if defined(__x86_64__)
#define JUMP_SLOT R_X86_64_JUMP_SLOT
#define GLOB_DAT R_X86_64_GLOB_DAT
#elif defined(__aarch64__)
#define JUMP_SLOT R_AARCH64_JUMP_SLOT
#define GLOB_DAT R_AARCH64_GLOB_DAT
#endif

// Called for each loaded library: in libc10_cuda.so, rewrites the slots through which it calls cudaMalloc and cudaFree
// (its global offset table) to sublet's, and sets *(int*)found to 1 once both are rewritten.
static int patch_library(struct dl_phdr_info* info, size_t size, void* found) {
  (void)size;
  if (!info->dlpi_name || !strstr(info->dlpi_name, ALLOCATOR_LIBRARY)) return 0;
  ElfW(Dyn)* dyn = NULL;
  for (int i = 0; i < info->dlpi_phnum; i++)
    if (info->dlpi_phdr[i].p_type == PT_DYNAMIC) dyn = (ElfW(Dyn)*)(info->dlpi_addr + info->dlpi_phdr[i].p_vaddr);
  if (!dyn) return 1;
  ElfW(Sym)* symtab = NULL;
  const char* strtab = NULL;
  ElfW(Rela)* tables[2] = {NULL, NULL};
  size_t sizes[2] = {0, 0}, needed[64];
  int nneeded = 0;
  for (ElfW(Dyn)* d = dyn; d->d_tag != DT_NULL; d++) {
    // The loader has usually made these addresses absolute already
    uintptr_t p = d->d_un.d_ptr < info->dlpi_addr ? d->d_un.d_ptr + info->dlpi_addr : d->d_un.d_ptr;
    if (d->d_tag == DT_SYMTAB) symtab = (ElfW(Sym)*)p;
    if (d->d_tag == DT_STRTAB) strtab = (const char*)p;
    if (d->d_tag == DT_JMPREL) tables[0] = (ElfW(Rela)*)p;
    if (d->d_tag == DT_PLTRELSZ) sizes[0] = d->d_un.d_val;
    if (d->d_tag == DT_RELA) tables[1] = (ElfW(Rela)*)p;
    if (d->d_tag == DT_RELASZ) sizes[1] = d->d_un.d_val;
    if (d->d_tag == DT_NEEDED && nneeded < 64) needed[nneeded++] = d->d_un.d_val;
  }
  if (!symtab || !strtab) return 1;
  // The real functions, from the CUDA runtime it links to
  void* cudart = NULL;
  for (int i = 0; i < nneeded && !cudart; i++)
    if (!strncmp(strtab + needed[i], RUNTIME_LIBRARY, strlen(RUNTIME_LIBRARY))) cudart = dlopen(strtab + needed[i], RTLD_NOW | RTLD_NOLOAD);
  if (!cudart) return 1;
  real_malloc = (cuda_malloc_t)dlsym(cudart, RUNTIME_SYMBOL("Malloc"));
  real_free = (cuda_free_t)dlsym(cudart, RUNTIME_SYMBOL("Free"));
  real_get_device = (cuda_int_t)dlsym(cudart, RUNTIME_SYMBOL("GetDevice"));
  real_exchange_capture_mode = (cuda_int_t)dlsym(cudart, RUNTIME_SYMBOL("ThreadExchangeStreamCaptureMode"));
  if (!real_malloc || !real_free || !real_get_device || !real_exchange_capture_mode) return 1;
  for (int i = 0; i < nneeded && !log_level; i++)
    if (!strcmp(strtab + needed[i], "libc10.so")) {
      void* c10 = dlopen(strtab + needed[i], RTLD_NOW | RTLD_NOLOAD);
      if (c10) log_level = (int32_t*)dlsym(c10, "FLAGS_caffe2_log_level");
    }
  long pagesize = sysconf(_SC_PAGESIZE);
  int rewritten = 0;
  for (int t = 0; t < 2; t++)
    for (size_t k = 0; tables[t] && k < sizes[t] / sizeof(ElfW(Rela)); k++) {
      ElfW(Rela)* rel = &tables[t][k];
      if (ELF64_R_TYPE(rel->r_info) != JUMP_SLOT && ELF64_R_TYPE(rel->r_info) != GLOB_DAT) continue;
      const char* name = strtab + symtab[ELF64_R_SYM(rel->r_info)].st_name;
      void* hook = !strcmp(name, RUNTIME_SYMBOL("Malloc")) ? (void*)sublet_cuda_malloc
                   : !strcmp(name, RUNTIME_SYMBOL("Free")) ? (void*)sublet_cuda_free
                                               : NULL;
      if (!hook) continue;
      void** slot = (void**)(info->dlpi_addr + rel->r_offset);
      if (mprotect((void*)((uintptr_t)slot & ~(uintptr_t)(pagesize - 1)), pagesize, PROT_READ | PROT_WRITE) != 0) return 1;
      __atomic_store_n(slot, hook, __ATOMIC_SEQ_CST);
      rewritten |= hook == (void*)sublet_cuda_malloc ? 1 : 2;
    }
  *(int*)found = rewritten == 3;
  return 1;
}

// Returns 0 once PyTorch's caching allocator gets its memory from sublet, -1 if it cannot be done.
int sublet_hook(void) {
  pthread_mutex_lock(&mu);
  if (!hooked) dl_iterate_phdr(patch_library, &hooked);
  pthread_mutex_unlock(&mu);
  return hooked ? 0 : -1;
}

// ---------------------------------------------------------------------------------------------------------------------
// Regions

static void destroy(Region* r) {
  CUcontext previous = enter(r);
  cuCtxSynchronize();  // as cudaFree does: kernels may still use the memory
  if (r->nmapped) cuMemUnmap(r->base, r->nmapped * PAGE), cuMemUnmap(r->arena, r->nmapped * PAGE);
  for (int64_t p = 0; p < r->npages && r->pages[p]; p++) cuMemRelease(r->pages[p]);
  if (r->scratch) cuMemRelease(r->scratch);
  if (r->base) cuMemAddressFree(r->base, r->npages * PAGE);
  if (r->arena) cuMemAddressFree(r->arena, r->npages * PAGE);
  for (int64_t c = 0; c < r->nslot_chunks; c++) cuMemFreeHost(r->slot_chunks[c]);
  if (r->copy_stream) cuStreamDestroy(r->copy_stream);
  cuCtxSetCurrent(previous);
  free(r->pages), free(r->keep), free(r->copied), free(r->busy), free(r->hole), free(r->lent), free(r->want),
      free(r->slot), free(r->slot_chunks), free(r->free_slots), free(r);
}

Region* sublet_create(int device, size_t nbytes) {
  CUcontext ctx;
  if (cuDevicePrimaryCtxRetain(&ctx, device) != 0) return NULL;
  Region* r = calloc(1, sizeof(Region));
  r->device = device;
  r->ctx = ctx;
  CUcontext previous = enter(r);
  r->npages = nbytes / PAGE;
  r->pages = calloc(r->npages, sizeof(CUmemGenericAllocationHandle));
  r->keep = calloc(r->npages, 1), r->copied = calloc(r->npages, 1), r->busy = calloc(r->npages, 1);
  r->hole = calloc(r->npages, 1), r->lent = calloc(r->npages, 1), r->want = calloc(r->npages, 1);
  r->slot = malloc(r->npages * sizeof(int64_t));
  for (int64_t p = 0; p < r->npages; p++) r->slot[p] = -1;
  CUmemAllocationProp prop = device_memory(device);
  int failed = cuStreamCreate(&r->copy_stream, 1 /* CU_STREAM_NON_BLOCKING */) != 0;
  if (failed) r->copy_stream = NULL;
  if (!failed && cuMemAddressReserve(&r->base, nbytes, 0, 0, 0) != 0) r->base = 0, failed = 1;
  if (!failed && cuMemAddressReserve(&r->arena, nbytes, 0, 0, 0) != 0) r->arena = 0, failed = 1;
  if (!failed && cuMemCreate(&r->scratch, PAGE, &prop, 0) != 0) r->scratch = 0, failed = 1;
  for (int64_t p = 0; p < r->npages && !failed; p++) {
    if (cuMemCreate(&r->pages[p], PAGE, &prop, 0) != 0) {
      r->pages[p] = 0, failed = 1;
    } else if (cuMemMap(r->base + p * PAGE, PAGE, 0, r->pages[p], 0) != 0) {
      failed = 1;
    } else if (cuMemMap(r->arena + p * PAGE, PAGE, 0, r->pages[p], 0) != 0) {
      cuMemUnmap(r->base + p * PAGE, PAGE), failed = 1;
    } else {
      r->nmapped = p + 1;
    }
  }
  if (!failed) failed = allow_access(r->base, nbytes, device) != 0 || allow_access(r->arena, nbytes, device) != 0;
  cuCtxSetCurrent(previous);
  if (failed) destroy(r), r = NULL;
  return r;
}

// The tensor is gone. If segments made from r are still alive, destroy it when the last one is freed.
void sublet_destroy(Region* r) {
  pthread_mutex_lock(&mu);
  if (r->nsegments)
    r->dropped = 1;
  else
    destroy(r);
  pthread_mutex_unlock(&mu);
}

CUdeviceptr sublet_base(Region* r) { return r->base; }

// For inspection: one byte per page, set if the page was lent during the current loan (or the last one, after it).
void sublet_lent(Region* r, uint8_t* out) {
  pthread_mutex_lock(&mu);
  memcpy(out, r->lent, r->npages);
  pthread_mutex_unlock(&mu);
}

// For inspection: 1 if the memory at ptr is pages of r, lent to a segment.
int sublet_backs(Region* r, CUdeviceptr ptr) {
  pthread_mutex_lock(&mu);
  int backs = 0;
  for (Segment* s = segments; s && !backs; s = s->next)
    backs = s->r == r && (s->kind == STRETCH || s->kind == GATHERED) && ptr >= s->ptr && ptr < s->ptr + s->npages * PAGE;
  pthread_mutex_unlock(&mu);
  return backs;
}

// ---------------------------------------------------------------------------------------------------------------------
// Copies of kept pages

static int needs_copy(Region* r, int64_t p) { return r->keep[p] && !r->copied[p]; }

static char* slot_memory(Region* r, int64_t slot) { return r->slot_chunks[slot / CHUNK] + (slot % CHUNK) * PAGE; }

static int descending(const void* a, const void* b) {
  int64_t x = *(const int64_t*)a, y = *(const int64_t*)b;
  return (x < y) - (x > y);
}

// Copies the pages in r->want to the CPU (to_host) or back, on stream, and clears r->want. Copying out gives each page
// a CPU slot, pinning CHUNK more slots when none is free; copying back frees them, for the next copies once the caller
// has waited for this one. Returns the number of pages, or -1, with nothing copied, if pinned memory runs out.
static int64_t copy_wanted(Region* r, int to_host, CUstream stream) {
  if (to_host) {
    // Popped from the end, the lowest slot first: consecutive pages get consecutive slots, copied in one go
    qsort(r->free_slots, r->nfree_slots, sizeof(int64_t), descending);
    for (int64_t p = 0; p < r->npages; p++) {
      if (!r->want[p]) continue;
      if (!r->nfree_slots) {
        char* chunk;
        if (cuMemHostAlloc((void**)&chunk, CHUNK * PAGE, 0) != 0) {
          for (int64_t q = 0; q < p; q++)
            if (r->want[q]) r->free_slots[r->nfree_slots++] = r->slot[q], r->slot[q] = -1;
          memset(r->want, 0, r->npages);
          return -1;
        }
        r->slot_chunks = realloc(r->slot_chunks, (r->nslot_chunks + 1) * sizeof(char*));
        r->slot_chunks[r->nslot_chunks] = chunk;
        r->free_slots = realloc(r->free_slots, (r->nslot_chunks + 1) * CHUNK * sizeof(int64_t));
        for (int64_t i = CHUNK - 1; i >= 0; i--) r->free_slots[r->nfree_slots++] = r->nslot_chunks * CHUNK + i;
        r->nslot_chunks++;
      }
      r->slot[p] = r->free_slots[--r->nfree_slots];
    }
  }
  int64_t n = 0;
  for (int64_t p = 0; p < r->npages;) {
    if (!r->want[p]) {
      p++;
      continue;
    }
    int64_t q = p + 1;  // a run of pages in consecutive slots of one chunk
    while (q < r->npages && r->want[q] && r->slot[q] == r->slot[q - 1] + 1 && r->slot[q] / CHUNK == r->slot[p] / CHUNK) q++;
    if (to_host)
      cuMemcpyDtoHAsync(slot_memory(r, r->slot[p]), r->base + p * PAGE, (q - p) * PAGE, stream);
    else
      cuMemcpyHtoDAsync(r->base + p * PAGE, slot_memory(r, r->slot[p]), (q - p) * PAGE, stream);
    n += q - p;
    p = q;
  }
  for (int64_t p = 0; p < r->npages; p++) {
    if (r->want[p] && !to_host) r->free_slots[r->nfree_slots++] = r->slot[p], r->slot[p] = -1;
    r->want[p] = 0;
  }
  return n;
}

// ---------------------------------------------------------------------------------------------------------------------
// Segments

// First page of the stretch for n pages that costs nothing (free pages, the arena mapping them), or -1: in the
// shortest such run that fits, so that long runs stay whole for large segments.
static int64_t free_stretch(Region* r, int64_t n) {
  int64_t best = -1, best_length = INT64_MAX;
  for (int64_t p = 0; p < r->npages;) {
    int64_t q = p;
    while (q < r->npages && !r->busy[q] && !r->hole[q] && !needs_copy(r, q)) q++;
    if (q - p >= n && q - p < best_length) best = p, best_length = q - p;
    p = q + 1;
  }
  return best;
}

// First page of the stretch of n pages with the fewest pages to copy, or -1 if no stretch is long enough.
static int64_t cheapest_stretch(Region* r, int64_t n) {
  int64_t best = -1, best_cost = INT64_MAX;
  for (int64_t p = 0; p < r->npages;) {
    int64_t q = p;
    while (q < r->npages && !r->busy[q] && !r->hole[q]) q++;
    if (q - p >= n) {  // slide a window over [p, q)
      int64_t cost = 0;
      for (int64_t i = p; i < p + n; i++) cost += needs_copy(r, i);
      for (int64_t s = p;; s++) {
        if (cost < best_cost) best = s, best_cost = cost;
        if (s + n == q) break;
        cost += needs_copy(r, s + n) - needs_copy(r, s);
      }
    }
    p = q + 1;
  }
  return best;
}

// Copies the kept pages behind s that are not copied yet to the CPU, waits for the copies, and marks its pages busy.
static int take_pages(Region* r, Segment* s) {
  for (int64_t i = 0; i < s->npages; i++) r->want[s->page[i]] = needs_copy(r, s->page[i]);
  if (copy_wanted(r, 1, r->copy_stream) < 0) return -1;
  cuStreamSynchronize(r->copy_stream);  // the copies are done before the borrower gets the memory
  for (int64_t i = 0; i < s->npages; i++) {
    int64_t p = s->page[i];
    if (r->keep[p]) r->copied[p] = 1;
    r->busy[p] = r->lent[p] = 1;
  }
  return 0;
}

static int use_stretch(Region* r, Segment* s, int64_t first) {
  for (int64_t i = 0; i < s->npages; i++) s->page[i] = first + i;
  if (take_pages(r, s) != 0) return -1;
  s->kind = STRETCH, s->in_arena = 1, s->ptr = r->arena + first * PAGE;
  return 0;
}

// Pages from anywhere in the region, free ones first, mapped at new addresses.
static int gather(Region* r, Segment* s) {
  int64_t k = 0;
  for (int with_copies = 0; with_copies <= 1; with_copies++)
    for (int64_t p = 0; p < r->npages && k < s->npages; p++)
      if (!r->busy[p] && needs_copy(r, p) == with_copies) s->page[k++] = p;
  if (k < s->npages || cuMemAddressReserve(&s->ptr, s->npages * PAGE, 0, 0, 0) != 0) return -1;
  if (take_pages(r, s) != 0) {
    cuMemAddressFree(s->ptr, s->npages * PAGE);
    return -1;
  }
  if (map_pages(r, s->ptr, s->page, 0, s->npages) != 0) {
    for (int64_t i = 0; i < s->npages; i++) r->busy[s->page[i]] = 0;
    cuMemAddressFree(s->ptr, s->npages * PAGE);
    return -1;
  }
  s->kind = GATHERED;
  r->gathered += s->npages * PAGE;
  return 0;
}

// A segment for size bytes during a loan of r, or NULL if neither r nor the GPU has the memory, or if only kept pages
// are left and may_copy is 0 (then *would_copy is set).
static Segment* new_segment(Region* r, size_t size, int may_copy, int* would_copy) {
  Segment* s = calloc(1, sizeof(Segment));
  s->r = r;
  s->npages = (size + PAGE - 1) / PAGE;
  int fits = s->npages <= r->npages;  // in the region
  s->page = fits ? malloc(s->npages * sizeof(int64_t)) : NULL;
  int64_t first = fits ? free_stretch(r, s->npages) : -1;
  int ok = first >= 0 && use_stretch(r, s, first) == 0;
  if (!ok && cuMemAlloc(&s->ptr, s->npages * PAGE) == 0) {
    ok = 1, s->kind = NEW;
    r->extra += s->npages * PAGE;
    if (r->extra > r->extra_peak) r->extra_peak = r->extra;
  }
  *would_copy = !ok && fits && !may_copy;
  if (!ok && fits && may_copy && (first = cheapest_stretch(r, s->npages)) >= 0) ok = use_stretch(r, s, first) == 0;
  if (!ok && fits && may_copy) ok = gather(r, s) == 0;
  if (!ok) {
    free(s->page), free(s);
    return NULL;
  }
  s->next = segments, segments = s;
  r->nsegments++;
  return s;
}

// The thread allocates for a CUDA graph capture: PyTorch switches it to relaxed capture mode for the cudaMalloc, and
// does not free its cached memory when that fails.
static int capturing(void) {
  int mode = 0;  // cudaStreamCaptureModeGlobal
  real_exchange_capture_mode(&mode);
  int current = mode;
  real_exchange_capture_mode(&mode);
  return current == 2;  // cudaStreamCaptureModeRelaxed
}

static int declined[MAX_DEVICES];  // the last request on each device was declined rather than copy kept pages
static int32_t saved_log_level = -1;  // c10's log level while PyTorch reports a declined request
#define DECLINE_MIN (32ull << 20)  // smaller requests copy right away: a few kept pages cost less than a decline

// PyTorch logs a warning when a cudaMalloc fails, before it frees its cached memory and asks again. A declined request
// is not a failure: errors only are logged until the next request, or the end of the loan.
static void quiet_log(void) {
  if (log_level && saved_log_level < 0) saved_log_level = *log_level, *log_level = 2;  // ERROR
}

static void restore_log(void) {
  if (saved_log_level >= 0) *log_level = saved_log_level, saved_log_level = -1;
}

int sublet_cuda_malloc(void** ptr, size_t size) {
  int device = 0;
  real_get_device(&device);
  pthread_mutex_lock(&mu);
  restore_log();
  Region* r = device >= 0 && device < MAX_DEVICES ? lender[device] : NULL;
  Segment* s = NULL;
  if (r) {
    // Before copying kept pages, a large request is declined once: PyTorch then frees the memory it caches (what the
    // borrower freed, which it may hold on another stream), and asks again, finding free pages where it would have
    // copied
    int would_copy = 0;
    CUcontext previous = enter(r);
    s = new_segment(r, size, declined[device] || size < DECLINE_MIN || capturing(), &would_copy);
    cuCtxSetCurrent(previous);
    declined[device] = would_copy;
    if (would_copy) quiet_log();
  }
  pthread_mutex_unlock(&mu);
  if (!r) return real_malloc(ptr, size);
  if (!s) return CUDA_ERROR_MEMORY_ALLOCATION;  // PyTorch frees its cached segments and asks again
  *ptr = (void*)s->ptr;
  return CUDA_SUCCESS;
}

int sublet_cuda_free(void* ptr) {
  pthread_mutex_lock(&mu);
  Segment** link = &segments;
  while (*link && (*link)->ptr != (CUdeviceptr)ptr) link = &(*link)->next;
  Segment* s = *link;
  if (!s) {
    pthread_mutex_unlock(&mu);
    return real_free(ptr);
  }
  *link = s->next;
  Region* r = s->r;
  CUcontext previous = enter(r);
  cuCtxSynchronize();  // as cudaFree does: kernels may still use the memory
  size_t bytes = s->npages * PAGE;
  if (s->kind == STRETCH || s->kind == GATHERED) {
    for (int64_t i = 0; i < s->npages; i++) r->busy[s->page[i]] = 0;
    if (s->kind == GATHERED) cuMemUnmap(s->ptr, bytes), cuMemAddressFree(s->ptr, bytes);
  } else if (s->kind == NEW) {
    cuMemFree(s->ptr);
    r->extra -= bytes;
  } else {  // MOVED or BLANK
    cuMemUnmap(s->ptr, bytes);
    if (s->kind == MOVED) cuMemRelease(s->handle);
    if (s->in_arena) {  // the arena maps the region's pages there again
      map_pages(r, s->ptr, NULL, s->page[0], s->npages);
      memset(r->hole + s->page[0], 0, s->npages);
    } else {
      cuMemAddressFree(s->ptr, bytes);
    }
  }
  cuCtxSetCurrent(previous);
  if (--r->nsegments == 0 && r->dropped) destroy(r);
  free(s->page), free(s);
  pthread_mutex_unlock(&mu);
  return CUDA_SUCCESS;
}

// ---------------------------------------------------------------------------------------------------------------------
// Loans

// Starts a loan of r; keep has one byte per page. Nothing is copied until a segment needs it.
void sublet_begin(Region* r, const uint8_t* keep) {
  pthread_mutex_lock(&mu);
  memcpy(r->keep, keep, r->npages);
  memset(r->copied, 0, r->npages);
  memset(r->lent, 0, r->npages);
  r->extra_start = r->extra_peak = r->extra;
  r->moved = r->gathered = 0;
  lender[r->device] = r;
  declined[r->device] = 0;
  pthread_mutex_unlock(&mu);
}

// New segments no longer come from r.
void sublet_stop(Region* r) {
  pthread_mutex_lock(&mu);
  restore_log();
  if (lender[r->device] == r) lender[r->device] = NULL;
  pthread_mutex_unlock(&mu);
}

// Moves the segments still holding region pages of r out of it: copies each to new device memory, mapped at the same
// addresses. Writes the address of those that could not be moved, for lack of GPU memory, to stuck (up to cap) and
// returns their number. Needs a synchronized device.
int64_t sublet_move_out(Region* r, uint64_t* stuck, int64_t cap) {
  pthread_mutex_lock(&mu);
  CUcontext previous = enter(r);
  int64_t nstuck = 0;
  for (Segment* s = segments; s; s = s->next) {
    if (s->r != r || (s->kind != STRETCH && s->kind != GATHERED)) continue;
    size_t size = s->npages * PAGE;
    CUmemAllocationProp prop = device_memory(r->device);
    CUdeviceptr staging = 0;
    int ok = cuMemCreate(&s->handle, size, &prop, 0) == 0;
    if (ok && cuMemAddressReserve(&staging, size, 0, 0, 0) != 0) cuMemRelease(s->handle), ok = 0;
    if (ok && cuMemMap(staging, size, 0, s->handle, 0) != 0) cuMemAddressFree(staging, size), cuMemRelease(s->handle), ok = 0;
    if (ok && (allow_access(staging, size, r->device) != 0 ||
               cuMemcpyDtoDAsync(staging, s->ptr, size, r->copy_stream) != 0 ||
               cuStreamSynchronize(r->copy_stream) != 0)) {
      cuMemUnmap(staging, size), cuMemAddressFree(staging, size), cuMemRelease(s->handle);
      ok = 0;
    }
    if (!ok) {
      if (nstuck < cap) stuck[nstuck] = s->ptr;
      nstuck++;
      continue;
    }
    cuMemUnmap(staging, size), cuMemAddressFree(staging, size);
    cuMemUnmap(s->ptr, size);
    cuMemMap(s->ptr, size, 0, s->handle, 0);
    allow_access(s->ptr, size, r->device);
    for (int64_t i = 0; i < s->npages; i++) r->busy[s->page[i]] = 0;
    if (s->in_arena) memset(r->hole + s->page[0], 1, s->npages);
    s->kind = MOVED;
    r->moved += size;
  }
  cuCtxSetCurrent(previous);
  pthread_mutex_unlock(&mu);
  return nstuck;
}

// Maps the scratch page over every page of the segments still holding region pages of r: their allocations keep
// their addresses but no longer hold data, so that the kept pages behind them can be copied back. For those that
// sublet_move_out could not move.
void sublet_blank_out(Region* r) {
  pthread_mutex_lock(&mu);
  CUcontext previous = enter(r);
  for (Segment* s = segments; s; s = s->next) {
    if (s->r != r || (s->kind != STRETCH && s->kind != GATHERED)) continue;
    cuMemUnmap(s->ptr, s->npages * PAGE);
    for (int64_t i = 0; i < s->npages; i++) cuMemMap(s->ptr + i * PAGE, PAGE, 0, r->scratch, 0);
    allow_access(s->ptr, s->npages * PAGE, r->device);
    for (int64_t i = 0; i < s->npages; i++) r->busy[s->page[i]] = 0;
    if (s->in_arena) memset(r->hole + s->page[0], 1, s->npages);
    s->kind = BLANK;
  }
  cuCtxSetCurrent(previous);
  pthread_mutex_unlock(&mu);
}

// Ends the loan of r: copies back, on stream, the kept pages it copied, and waits for them. out gets the bytes copied,
// the peak new device memory taken during the loan, the bytes moved out of the region at its end, and the bytes of the
// segments gathered during it.
void sublet_end(Region* r, void* stream, int64_t* out) {
  pthread_mutex_lock(&mu);
  CUcontext previous = enter(r);
  if (lender[r->device] == r) lender[r->device] = NULL;
  for (int64_t p = 0; p < r->npages; p++) r->want[p] = r->copied[p];
  int64_t copied = copy_wanted(r, 0, (CUstream)stream);
  cuStreamSynchronize((CUstream)stream);
  memset(r->copied, 0, r->npages);
  out[0] = copied * PAGE;
  out[1] = r->extra_peak - r->extra_start;
  out[2] = r->moved;
  out[3] = r->gathered;
  cuCtxSetCurrent(previous);
  pthread_mutex_unlock(&mu);
}
