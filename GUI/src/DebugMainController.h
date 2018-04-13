#include <ElasticFusion.h>
#include <Utils/Parse.h>

#include "Tools/GUI.h"
#include "Tools/GroundTruthOdometry.h"
#include "Tools/RawLogReader.h"
#include "Tools/LiveLogReader.h"

#include <iostream>

#ifndef DEBUGMAINCONTROLLER_H_
#define DEBUGMAINCONTROLLER_H_


class DebugMainController
{
    public:
        DebugMainController(ElasticFusion* elafu, GUI* gui,
                LogReader* logReader,
                bool quiet, int start, int end, bool frameskip,
                bool iclnuim
                )
            : eFusion(elafu),
              gui(gui),
              logReader_(logReader),
              quiet(quiet),
              start(start),
              end(end),
              framesToSkip(0),
              frameskip(frameskip),
              iclnuim(iclnuim)
        {
            icpErrThresh = elafu->getIcpErrThresh();
            icpCountThresh = elafu->getIcpCountThresh();
        }

        virtual ~DebugMainController();

        void run();

    private:
        bool resetButton;
        bool good;


        ElasticFusion* eFusion;
        GUI* gui;
        LogReader* logReader_;

        bool quiet;
        int start;
        int end;
        int framesToSkip;
        bool frameskip;
        bool iclnuim;
        float icpErrThresh;
        int icpCountThresh;
};


#endif /* DEBUGMAINCONTROLLER_H_ */
