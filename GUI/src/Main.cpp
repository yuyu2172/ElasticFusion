/*
 * This file is part of ElasticFusion.
 *
 * Copyright (C) 2015 Imperial College London
 * 
 * The use of the code within this file and all code within files that 
 * make up the software that is ElasticFusion is permitted for 
 * non-commercial purposes only.  The full terms and conditions that 
 * apply to the code within this file are detailed within the LICENSE.txt 
 * file and at <http://www.imperial.ac.uk/dyson-robotics-lab/downloads/elastic-fusion/elastic-fusion-license/> 
 * unless explicitly stated.  By downloading this file you agree to 
 * comply with these terms.
 *
 * If you wish to use any of this code for commercial purposes then 
 * please email researchcontracts.engineering@imperial.ac.uk.
 *
 */

#include "MainController.h"
#include "DebugMainController.h"


int main(int argc, char * argv[])
{
    bool debug = true;
    if (debug) {
        Resolution::getInstance(640, 480);

        std::string calibrationFile;
        Parse::get().arg(argc, argv, "-cal", calibrationFile);

        if(calibrationFile.length())
        {
            throw std::runtime_error("aaaa \n");
            // loadCalibration(calibrationFile);
        }
        else
        {
            Intrinsics::getInstance(528, 528, 320, 240);
        }


        bool good;
        std::string empty;
        std::string logFile;
        LogReader * logReader;
        bool iclnuim = Parse::get().arg(argc, argv, "-icl", empty) > -1;
        Parse::get().arg(argc, argv, "-l", logFile);

        if(logFile.length())
        {
            logReader = new RawLogReader(logFile, Parse::get().arg(argc, argv, "-f", empty) > -1);
        }
        else
        {
            bool flipColors = Parse::get().arg(argc, argv, "-f", empty) > -1;
            logReader = new LiveLogReader(logFile, flipColors, LiveLogReader::CameraType::OpenNI2);

            good = ((LiveLogReader *)logReader)->cam->ok();

#ifdef WITH_REALSENSE
            if(!good)
            {
            delete logReader;
            logReader = new LiveLogReader(logFile, flipColors, LiveLogReader::CameraType::RealSense);

            good = ((LiveLogReader *)logReader)->cam->ok();
            }
#endif
        }

        // Ground truth skip
        //
        //
        float confidence = 10.0f;
        float depth = 3.0f;
        float icp = 10.0f;
        float icpErrThresh = 5e-05;
        float covThresh = 1e-05;
        float photoThresh = 115;
        float fernThresh = 0.3095f;

        int timeDelta = 200;
        int icpCountThresh = 40000;
        int start = 1;
        int end;
        bool so3;
        so3 = !(Parse::get().arg(argc, argv, "-nso", empty) > -1);
        end = std::numeric_limits<unsigned short>::max(); //Funny bound, since we predict times in this format really!

        Parse::get().arg(argc, argv, "-c", confidence);
        Parse::get().arg(argc, argv, "-d", depth);
        Parse::get().arg(argc, argv, "-i", icp);
        Parse::get().arg(argc, argv, "-ie", icpErrThresh);
        Parse::get().arg(argc, argv, "-cv", covThresh);
        Parse::get().arg(argc, argv, "-pt", photoThresh);
        Parse::get().arg(argc, argv, "-ft", fernThresh);
        Parse::get().arg(argc, argv, "-t", timeDelta);
        Parse::get().arg(argc, argv, "-ic", icpCountThresh);
        Parse::get().arg(argc, argv, "-s", start);
        Parse::get().arg(argc, argv, "-e", end);

        logReader->flipColors = Parse::get().arg(argc, argv, "-f", empty) > -1;

        bool openLoop = Parse::get().arg(argc, argv, "-o", empty) > -1;
        bool reloc = Parse::get().arg(argc, argv, "-rl", empty) > -1;
        bool frameskip = Parse::get().arg(argc, argv, "-fs", empty) > -1;
        bool quiet = Parse::get().arg(argc, argv, "-q", empty) > -1;
        bool fastOdom = Parse::get().arg(argc, argv, "-fo", empty) > -1;
        bool rewind = Parse::get().arg(argc, argv, "-r", empty) > -1;
        bool frameToFrameRGB = Parse::get().arg(argc, argv, "-ftf", empty) > -1;


        bool liveCap = logFile.length() == 0;
        bool showcaseMode = Parse::get().arg(argc, argv, "-sc", empty) > -1;
        GUI* gui = new GUI(liveCap, showcaseMode);

        gui->flipColors->Ref().Set(logReader->flipColors);
        gui->rgbOnly->Ref().Set(false);
        gui->pyramid->Ref().Set(true);
        gui->fastOdom->Ref().Set(fastOdom);
        gui->confidenceThreshold->Ref().Set(confidence);
        gui->depthCutoff->Ref().Set(depth);
        gui->icpWeight->Ref().Set(icp);
        gui->so3->Ref().Set(so3);
        gui->frameToFrameRGB->Ref().Set(frameToFrameRGB);

        Resize* resizeStream = new Resize(Resolution::getInstance().width(),
                                Resolution::getInstance().height(),
                                Resolution::getInstance().width() / 2,
                                Resolution::getInstance().height() / 2);

        auto eFusion = new ElasticFusion(openLoop ? std::numeric_limits<int>::max() / 2 : timeDelta,
                                        icpCountThresh,
                                        icpErrThresh,
                                        covThresh,
                                        !openLoop,
                                        iclnuim,
                                        reloc,
                                        photoThresh,
                                        confidence,
                                        depth,
                                        icp,
                                        fastOdom,
                                        fernThresh,
                                        so3,
                                        frameToFrameRGB,
                                        logReader->getFile()
                                        );

        auto debug_main_controller = DebugMainController(eFusion, gui, logReader, quiet, start, end, frameskip, iclnuim);
        // Run!
        debug_main_controller.run();
    }

    else {
        MainController mainController(argc, argv);

        mainController.launch();
    }

    return 0;
}
