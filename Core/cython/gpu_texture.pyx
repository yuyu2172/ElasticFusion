from libcpp cimport bool as c_bool


ctypedef unsigned int GLenum

cdef extern from "../src/GPUTexture.h":
    cdef cppclass C_GPUTexture "GPUTexture":
        C_GPUTexture(int, int, GLenum, GLenum, GLenum, c_bool, c_bool)


cdef class GPUTexture:
    cdef C_GPUTexture* c_gpu_texture

    def __cinit__(self, int width, int height,
                  unsigned int internal_format, unsigned int format_,
                  unsigned int dtype, c_bool draw, c_bool cuda):
        self.c_gpu_texture = new C_GPUTexture(
            width, height, internal_format, format_, dtype, draw, cuda)
