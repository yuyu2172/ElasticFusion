#pragma once

#include <stdio.h>
#include <stdlib.h>

#include <signal.h>
#include <chrono>
#include <thread>

#include <Utils/Parse.h>

#include "LogReader.h"
#include "CameraInterface.h"

#include "librealsense2/rs.hpp"

#include <unsupported/Eigen/CXX11/Tensor>


using ImageTensor = Eigen::Tensor<uint8_t, 3, Eigen::RowMajor>;
// using DepthTensor = Eigen::Tensor<uint16_t, 3, Eigen::RowMajor>;
using DepthTensor = Eigen::Tensor<uint16_t, 3, Eigen::RowMajor>;

class RealSenseLogReader : public LogReader
{
    public: 
        RealSenseLogReader();

        void getNext();

        int getNumFrames()
        {
            return 10000000;
        }

        bool hasMore()
        {
            return true;
        }

        bool rewound()
        {
            return false;
        }

        void rewind()
        {

        }

        void getBack()
        {

        }

        void fastForward(int frame)
        {

        }

        const std::string getFile() { }
        
        void setAuto(bool value) { }

        CameraInterface * cam;
        rs2::pipeline pipe;


        std::pair<ImageTensor, int64_t> getRgbBuffers(int i)
        {
            return rgbBuffers[i];
        }

        std::pair<std::pair<ImageTensor, DepthTensor>, int64_t> getFrameBuffers(int i)
        {
            return frameBuffers_[i];
        }


    private:
        int64_t lastFrameTime;
		    int lastGot;

        static const int numBuffers = 10;
        std::pair<ImageTensor,int64_t> rgbBuffers[numBuffers];

        std::pair<std::pair<ImageTensor, DepthTensor>,int64_t> frameBuffers_[numBuffers];
};
