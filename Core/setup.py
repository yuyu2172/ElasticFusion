from future.utils import iteritems
import  os
from os.path import join as pjoin
from setuptools import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import subprocess
import numpy


def find_in_path(name, path):
    "Find a file in a search path"
    #adapted fom http://code.activestate.com/recipes/52224-find-a-file-given-a-search-path/
    for dir in path.split(os.pathsep):
        binpath = pjoin(dir, name)
        if os.path.exists(binpath):
            return os.path.abspath(binpath)
    return None


def locate_cuda():
    """Locate the CUDA environment on the system

    Returns a dict with keys 'home', 'nvcc', 'include', and 'lib64'
    and values giving the absolute path to each directory.

    Starts by looking for the CUDAHOME env variable. If not found, everything
    is based on finding 'nvcc' in the PATH.
    """

    # first check if the CUDAHOME env variable is in use
    if 'CUDAHOME' in os.environ:
        home = os.environ['CUDAHOME']
        nvcc = pjoin(home, 'bin', 'nvcc')
    else:
        # otherwise, search the PATH for NVCC
        nvcc = find_in_path('nvcc', os.environ['PATH'])
        if nvcc is None:
            raise EnvironmentError('The nvcc binary could not be '
                'located in your $PATH. Either add it to your path, or set $CUDAHOME')
        home = os.path.dirname(os.path.dirname(nvcc))

    cudaconfig = {'home':home, 'nvcc':nvcc,
                  'include': pjoin(home, 'include'),
                  'lib64': pjoin(home, 'lib64')}
    for k, v in iteritems(cudaconfig):
        if not os.path.exists(v):
            raise EnvironmentError('The CUDA %s path could not be located in %s' % (k, v))

    return cudaconfig


def customize_compiler_for_nvcc(self):
    """inject deep into distutils to customize how the dispatch
    to gcc/nvcc works.

    If you subclass UnixCCompiler, it's not trivial to get your subclass
    injected in, and still have the right customizations (i.e.
    distutils.sysconfig.customize_compiler) run on it. So instead of going
    the OO route, I have this. Note, it's kindof like a wierd functional
    subclassing going on."""

    # tell the compiler it can processes .cu
    self.src_extensions.append('.cu')

    # save references to the default compiler_so and _comple methods
    default_compiler_so = self.compiler_so
    super = self._compile

    # now redefine the _compile method. This gets executed for each
    # object but distutils doesn't have the ability to change compilers
    # based on source extension: we add it.
    def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
        if os.path.splitext(src)[1] == '.cu':
            # use the cuda for .cu files
            self.set_executable('compiler_so', CUDA['nvcc'])
            # use only a subset of the extra_postargs, which are 1-1 translated
            # from the extra_compile_args in the Extension class
            postargs = extra_postargs['nvcc']
        else:
            postargs = extra_postargs['gcc']

        super(obj, src, ext, cc_args, postargs, pp_opts)
        # reset the default compiler_so, which we might have changed for cuda
        self.compiler_so = default_compiler_so

    # inject our redefined _compile method into the class
    self._compile = _compile


# run the customize_compiler
class custom_build_ext(build_ext):
    def build_extensions(self):
        customize_compiler_for_nvcc(self.compiler)
        build_ext.build_extensions(self)



CUDA = locate_cuda()

# Obtain the numpy include directory.  This logic works across numpy versions.
try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()


sources = list()
sources += [os.path.join('src', name) for name
            in os.listdir('src') if name[-4:] == '.cpp']
sources += [os.path.join('src/Utils', name) for name
            in os.listdir('src/Utils') if name[-4:] == '.cpp']
sources += [os.path.join('src/Shaders', name) for name
            in os.listdir('src/Shaders') if name[-4:] == '.cpp']
sources += [os.path.join('src/Cuda', name) for name
            in os.listdir('src/Cuda') if name[-3:] == '.cu']
sources += [os.path.join('src/Cuda/containers', name) for name
            in os.listdir('src/Cuda/containers') if name[-4:] == '.cpp']

library_dirs = [CUDA['lib64'],
                '/home/leus/projects/ElasticFusion/Pangolin/build/src',
                '/usr/lib/x86_64-linux-gnu']

ext_data = {
    'elafu.rgbd_odometry': {
        'pyxfile': 'cython/rgbd_odometry.pyx'},
    'elafu.gpu_texture': {
        'pyxfile': 'cython/gpu_texture.pyx'}
}

exts = list()
for key, val in ext_data.items():
    exts.append(Extension(key,
                    sources=sources + [val['pyxfile']],
                    library_dirs=library_dirs,
                    libraries=['cudart', 'pangolin', 'cholmod'],
                    language='c++',
                    runtime_library_dirs=library_dirs,
                    # this syntax is specific to this build system
                    # we're only going to use certain compiler args with nvcc and not with gcc
                    # the implementation of this trick is in customize_compiler() below
                    extra_compile_args={
                        'gcc': ['-std=c++11', '-fPIC'],
                        'nvcc': ['-arch=sm_61', '--ptxas-options=-v', '-c',
                                '--compiler-options', "'-fPIC'", '-std=c++11']},
                    include_dirs = [
                        '../Pangolin/include',
                        '/usr/include/suitesparse/',
                        numpy_include,
                        CUDA['include'],
                        'src',
                        '/home/leus/miniconda2/envs/general/include/eigen3'],
                    ))

setup(name='elafu',
      author='Yusuke Niitani',
      version='0.1',

      ext_modules = exts,

      # inject our custom trigger
      cmdclass={'build_ext': custom_build_ext},

      # since the package has c code, the egg cannot be zipped
      zip_safe=False)
