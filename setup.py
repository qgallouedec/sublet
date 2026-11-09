"""Builds sublet's C allocator (src/sublet/_allocator.c) as a plain shared library loaded with ctypes.

CUDA needs only a C compiler. The optional HIP backend also needs ROCm headers and libamdhip64.
"""

import os
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class BuildAllocator(build_ext):
    def build_extension(self, ext):
        try:
            super().build_extension(ext)
        except Exception as e:
            raise RuntimeError(
                "sublet could not compile its C allocator (src/sublet/_allocator.c). It needs a C compiler such as "
                "gcc or clang on PATH. HIP builds additionally need ROCm headers and libamdhip64 under ROCM_PATH "
                "(default /opt/rocm); use SUBLET_BUILD_HIP=0 for CUDA only. Original error: " + str(e)
            ) from e


extensions = [Extension("sublet._allocator", ["src/sublet/_allocator.c"], libraries=["dl", "pthread"])]
rocm = Path(os.environ.get("ROCM_PATH", "/opt/rocm"))
build_hip = os.environ.get("SUBLET_BUILD_HIP", "auto")
if build_hip not in {"0", "1", "auto"}:
    raise RuntimeError("SUBLET_BUILD_HIP must be 0, 1, or auto")
if build_hip == "1" or (build_hip == "auto" and (rocm / "include/hip/hip_runtime_api.h").exists()):
    extensions.append(Extension(
        "sublet._allocator_hip", ["src/sublet/_allocator_hip.c"],
        depends=["src/sublet/_hip.h", "src/sublet/_allocator.c"],
        define_macros=[("SUBLET_HIP", "1"), ("__HIP_PLATFORM_AMD__", "1")],
        include_dirs=[str(rocm / "include")],
        library_dirs=[str(rocm / "lib")], runtime_library_dirs=[str(rocm / "lib")],
        libraries=["amdhip64", "pthread"],
    ))

setup(
    ext_modules=extensions,
    cmdclass={"build_ext": BuildAllocator},
)
