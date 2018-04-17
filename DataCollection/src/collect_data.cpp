// License: Apache 2.0. See LICENSE file in root directory.
// Copyright(c) 2017 Intel Corporation. All Rights Reserved.

#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API

#include <chrono>
// #include "example.hpp"          // Include short list of convenience functions for rendering

#include <vis3d/window.h>
#include <vis3d/view/view.h>

// Capture Example demonstrates how to
// capture depth and color video streams and render them to the screen
int main() try
{
    int width = 640;
    int height = 480;

    auto window = Window();
    auto& view_1 = window.add_view(0, 0, 0.5, 0.5);
    auto& renderer_1 = view_1.draw_image(Image(width, height, 3, 1, std::vector<uint8_t>(width * height * 3)));
    window.start_window_thread();

    rs2::colorizer color_map;
    rs2::pipeline pipe;
    rs2::config config;

    config.enable_stream(RS2_STREAM_COLOR, width, height, RS2_FORMAT_RGB8);
    config.enable_stream(RS2_STREAM_DEPTH, width, height, RS2_FORMAT_Z16);
    // Start streaming with default recommended configuration
    pipe.start(config);


    while(true) // Application still alive?
    {
        rs2::frameset data = pipe.wait_for_frames(); // Wait for next set of frames from the camera

        // rs2::frame depth = color_map(data.get_depth_frame()); // Find and colorize the depth data
        auto color_frame = data.get_color_frame();            // Find the color data

        int size = color_frame.get_width() * color_frame.get_height() * 3;


        auto color_src = (const uint8_t*)color_frame.get_data();
        auto vec_color_data = std::vector<uint8_t>(color_src, color_src + size);

        auto start = std::chrono::system_clock::now();
        auto rgb_image = Image(
                color_frame.get_width(),
                color_frame.get_height(),
                3,
                1,
                vec_color_data);
        auto end = std::chrono::system_clock::now();
        renderer_1.set_data(std::make_shared<Image>(rgb_image));

        std::cout << "During creation of image elapsed " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << "\n";
    }

    return EXIT_SUCCESS;
}
catch (const rs2::error & e)
{
    std::cerr << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    " << e.what() << std::endl;
    return EXIT_FAILURE;
}
catch (const std::exception& e)
{
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
}
