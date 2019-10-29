import torch
import os.path as osp
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

compute_arch = 'compute_61'
prefix = 'ext_modules/src/nn/cpp'
prefix_cuda = 'ext_modules/src/nn/cuda'
include_dir = 'ext_modules/include'


def extension(name, source_basename):
    '''Create a build extension. Use CUDA if available, otherwise C++ only'''

    if torch.cuda.is_available():
        return CUDAExtension(name=name,
                             sources=[
                                 osp.join(prefix, source_basename + '.cpp'),
                                 osp.join(prefix_cuda, source_basename + '.cu'),
                             ],
                             include_dirs=[include_dir],
                             extra_compile_args={
                                 'cxx': ['-fopenmp', '-O3'],
                                 'nvcc': ['--gpu-architecture=' + compute_arch]
                             })
    else:
        return CppExtension(name=name,
                            sources=[
                                osp.join(prefix, source_basename + '.cpp'),
                            ],
                            include_dirs=[include_dir],
                            define_macros=[('__NO_CUDA__', None)],
                            extra_compile_args={
                                'cxx': ['-fopenmp', '-O3'],
                                'nvcc': []
                            })


setup(
    name='Profile Patch Convolution',
    version='0.0',
    author='Marc',
    description='Profiling patch convolutions',
    ext_package='_patchconv_ext',
    ext_modules=[

        # ------------------------------------------------
        # Patch CNN operations
        # ------------------------------------------------
        extension('_patch_convolution', 'patch_convolution_layer'),
        extension('_transposed_patch_convolution',
                  'transposed_patch_convolution_layer'),
    ],
    packages=['patch_convolution'],
    package_dir={'patch_convolution': 'layers'},
    cmdclass={'build_ext': BuildExtension},
)
