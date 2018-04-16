#include "RealSenseLogReader.h"


RealSenseLogReader::RealSenseLogReader()
    : LogReader(".", true)
{
    int width = Resolution::getInstance().width();
    int height = Resolution::getInstance().height();

    for(int i = 0; i < numBuffers; i++)
    {
        auto new_image = ImageTensor(height, width, 3);
        rgbBuffers[i] = std::pair<ImageTensor, int64_t>(new_image, 0);
    }

    for (int i=0; i < numBuffers; i++)
    {
        auto new_color = ImageTensor(height, width, 3);
        auto new_depth = DepthTensor(height, width, 1);
        frameBuffers_[i].first = std::make_pair(new_color, new_depth);
        frameBuffers_[i].second = 0;
    }
    // TODO:  Set up the format correctly
    rs2::config config;

    config.enable_stream(RS2_STREAM_COLOR, width, height, RS2_FORMAT_RGB8);
    config.enable_stream(RS2_STREAM_DEPTH, width, height, RS2_FORMAT_Z16);
    pipe.start(config);
    // pipe.start();
}


void RealSenseLogReader::getNext()
{
    auto frames = pipe.wait_for_frames();

    // Color
    auto color_frame = frames.get_color_frame();
    auto rgb_dst = frameBuffers_[0].first.first.data(); 
    auto size = frameBuffers_[0].first.first.size();
    auto rgb_src  = (const uint8_t*)color_frame.get_data();
    std::copy(rgb_src, rgb_src + size, rgb_dst);

    // Depth
    auto depth_frame = frames.get_depth_frame();
    auto depth_dst = frameBuffers_[0].first.second.data(); 
    int width = Resolution::getInstance().width();
    int height = Resolution::getInstance().height();
    std::memcpy(depth_dst, depth_frame.get_data(), height * width * 2);


    timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
    rgb = rgb_dst;
    depth = depth_dst;
    frameBuffers_[0].second = timestamp;
    // Check
    assert(color_frame.get_bytes_per_pixel() == 3);
    assert(Resolution::getInstance().width() == color_frame.get_width());
    assert(Resolution::getInstance().height() == color_frame.get_height());
    assert(depth_frame.get_bytes_per_pixel() == 2);
    assert(Resolution::getInstance().width() == depth_frame.get_width());
    assert(Resolution::getInstance().height() == depth_frame.get_height());
}
