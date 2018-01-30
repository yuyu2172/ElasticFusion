from libcpp cimport bool


ctypedef unsigned int GLenum

# cdef extern from "src/GPUTexture.h":
#     cdef cppclass C_GPUTexture "GPUTexture":
#         C_GPUTexture(int, int, GLenum, GLenum, GLenum, bool, cuda)


cdef class GPUTexture:
    cdef C_GPUTexture* c_gpu_texture
