#include <vector>
#include <iostream>
#include <string>
#include <thread>
#include <chrono>

#include <vis3d/window.h>
#include <vis3d/view/view.h>
#include <Eigen/Core>

#include <IO3D.h>

#include "../Tools/RealSenseLogReader.h"
#include "../Tools/RawLogReader.h"


#include <cnpy.h>


int main(int argc, char** argv)
{
    Resolution::getInstance(640, 480);
    auto log_reader = RealSenseLogReader();

    log_reader.getNext();
    auto io3d_rgb_image = log_reader.getFrameBuffers(0).first.first;
    auto io3d_depth_image = log_reader.getFrameBuffers(0).first.second;

    io3d::WriteDepthImage("depth.png", io3d_depth_image);

    auto vec_rgb_data = std::vector<uint8_t>(io3d_rgb_image.data(), io3d_rgb_image.data() + io3d_rgb_image.size());
    auto rgb_image = Image(
            io3d_rgb_image.dimension(1),
            io3d_rgb_image.dimension(0),
            io3d_rgb_image.dimension(2),
            1,
            vec_rgb_data);

    auto vec_depth_data = std::vector<uint8_t>();
    vec_depth_data.resize(640 * 480 * 2);
    std::memcpy(vec_depth_data.data(), io3d_depth_image.data(), 640 * 480 * 2);
    auto depth_image = Image(
            io3d_depth_image.dimension(1),
            io3d_depth_image.dimension(0),
            io3d_depth_image.dimension(2),
            2,
            vec_depth_data);

    auto window = Window();
    auto& view_1 = window.add_view(0, 0, 0.5, 0.5);
    view_1.draw_image(rgb_image);
    auto& view_2 = window.add_view(0.5, 0.5, 0.5, 0.5);
    view_2.draw_image(depth_image);
    window.show();

    return 0;
}
