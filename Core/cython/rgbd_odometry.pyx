cimport numpy as np
import numpy as np


cdef extern from "../src/Utils/RGBDOdometry.h":
    cdef cppclass C_RGBDOdometry "RGBDOdometry":
        C_RGBDOdometry(int, int, float, float, float, float,
                     float, float)


cdef class RGBDOdometry:
    cdef C_RGBDOdometry* c_rgbd_odom

    def __cinit__(self, int width, int height,
                  float cx, float cy, float fx, float fy,
                  float dist_thresh=0.1, float angle_thresh=np.sin(20 * np.pi / 180.)):
        self.c_rgbd_odom = new C_RGBDOdometry(
            width, height, cx, cy, fx, fy, dist_thresh, angle_thresh)



