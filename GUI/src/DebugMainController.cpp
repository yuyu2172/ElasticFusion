#include "DebugMainController.h"



DebugMainController::~DebugMainController()
{
    if(eFusion)
    {
        delete eFusion;
    }

    if(gui)
    {
        delete gui;
    }

    if(logReader)
    {
        delete logReader;
    }
}


void DebugMainController::run()
{
    while(!pangolin::ShouldQuit() && !((!logReader->hasMore()) && quiet) && !(eFusion->getTick() == end && quiet))
    {
        if(!gui->pause->Get() || pangolin::Pushed(*gui->step))
        {
            if((logReader->hasMore() || rewind) && eFusion->getTick() < end)
            {
                TICK("LogRead");
                if(rewind)
                {
                    if(!logReader->hasMore())
                        logReader->getBack();
                    else
                        logReader->getNext();

                    if(logReader->rewound())
                        logReader->currentFrame = 0;
                }
                else {
                    logReader->getNext();
                }
                TOCK("LogRead");

                if(eFusion->getTick() < start) {
                    eFusion->setTick(start);
                    logReader->fastForward(start);
                }

                float weightMultiplier = framesToSkip + 1;

                if(framesToSkip > 0) {
                    eFusion->setTick(eFusion->getTick() + framesToSkip);
                    logReader->fastForward(logReader->currentFrame + framesToSkip);
                    framesToSkip = 0;
                }

                Eigen::Matrix4f * currentPose = 0;

                // if(groundTruthOdometry)
                // {
                //     currentPose = new Eigen::Matrix4f;
                //     currentPose->setIdentity();
                //     *currentPose = groundTruthOdometry->getTransformation(logReader->timestamp);
                // }

                eFusion->processFrame(logReader->rgb, logReader->depth, logReader->timestamp, currentPose, weightMultiplier);

                if(currentPose)
                    delete currentPose;

                if(frameskip && Stopwatch::getInstance().getTimings().at("Run") > 1000.f / 30.f)
                    framesToSkip = int(Stopwatch::getInstance().getTimings().at("Run") / (1000.f / 30.f));
            }
        }
        else {
            eFusion->predict();
        }

        TICK("GUI");

        if(gui->followPose->Get()) {
            pangolin::OpenGlMatrix mv;

            Eigen::Matrix4f currPose = eFusion->getCurrPose();
            Eigen::Matrix3f currRot = currPose.topLeftCorner(3, 3);

            Eigen::Quaternionf currQuat(currRot);
            Eigen::Vector3f forwardVector(0, 0, 1);
            Eigen::Vector3f upVector(0, iclnuim ? 1 : -1, 0);

            Eigen::Vector3f forward = (currQuat * forwardVector).normalized();
            Eigen::Vector3f up = (currQuat * upVector).normalized();

            Eigen::Vector3f eye(currPose(0, 3), currPose(1, 3), currPose(2, 3));

            eye -= forward;

            Eigen::Vector3f at = eye + forward;

            Eigen::Vector3f z = (eye - at).normalized();  // Forward
            Eigen::Vector3f x = up.cross(z).normalized(); // Right
            Eigen::Vector3f y = z.cross(x);

            Eigen::Matrix4d m;
            m << x(0),  x(1),  x(2),  -(x.dot(eye)),
                 y(0),  y(1),  y(2),  -(y.dot(eye)),
                 z(0),  z(1),  z(2),  -(z.dot(eye)),
                    0,     0,     0,              1;

            memcpy(&mv.m[0], m.data(), sizeof(Eigen::Matrix4d));

            gui->s_cam.SetModelViewMatrix(mv);
        }

        gui->preCall();

        std::stringstream stri;
        stri << eFusion->getModelToModel().lastICPCount;
        gui->trackInliers->Ref().Set(stri.str());

        std::stringstream stre;
        stre << (std::isnan(eFusion->getModelToModel().lastICPError) ? 0 : eFusion->getModelToModel().lastICPError);
        gui->trackRes->Ref().Set(stre.str());

        if(!gui->pause->Get()) {
            gui->resLog.Log((std::isnan(eFusion->getModelToModel().lastICPError) ? std::numeric_limits<float>::max() : eFusion->getModelToModel().lastICPError), icpErrThresh);
            gui->inLog.Log(eFusion->getModelToModel().lastICPCount, icpCountThresh);
        }

        Eigen::Matrix4f pose = eFusion->getCurrPose();

        if(gui->drawRawCloud->Get() || gui->drawFilteredCloud->Get()) {
            eFusion->computeFeedbackBuffers();
        }

        if(gui->drawRawCloud->Get()) {
            eFusion->getFeedbackBuffers().at(FeedbackBuffer::RAW)->render(
                    gui->s_cam.GetProjectionModelViewMatrix(),
                    pose,
                    gui->drawNormals->Get(),
                    gui->drawColors->Get());
        }

        if(gui->drawFilteredCloud->Get()) {
            eFusion->getFeedbackBuffers().at(FeedbackBuffer::FILTERED)->render(
                    gui->s_cam.GetProjectionModelViewMatrix(),
                    pose,
                    gui->drawNormals->Get(),
                    gui->drawColors->Get());
        }

        if(gui->drawGlobalModel->Get()) {
            glFinish();
            TICK("Global");

            if(gui->drawFxaa->Get()) {
                gui->drawFXAA(gui->s_cam.GetProjectionModelViewMatrix(),
                              gui->s_cam.GetModelViewMatrix(),
                              eFusion->getGlobalModel().model(),
                              eFusion->getConfidenceThreshold(),
                              eFusion->getTick(),
                              eFusion->getTimeDelta(),
                              iclnuim);
            }
            else {
                eFusion->getGlobalModel().renderPointCloud(gui->s_cam.GetProjectionModelViewMatrix(),
                                                           eFusion->getConfidenceThreshold(),
                                                           gui->drawUnstable->Get(),
                                                           gui->drawNormals->Get(),
                                                           gui->drawColors->Get(),
                                                           gui->drawPoints->Get(),
                                                           gui->drawWindow->Get(),
                                                           gui->drawTimes->Get(),
                                                           eFusion->getTick(),
                                                           eFusion->getTimeDelta());
            }
            glFinish();
            TOCK("Global");
        }

        // Draw Frustrum
        if(eFusion->getLost())
            glColor3f(1, 1, 0);
        else
            glColor3f(1, 0, 1);
        gui->drawFrustum(pose);
        glColor3f(1, 1, 1);

        if(gui->drawFerns->Get())
        {
            glColor3f(0, 0, 0);
            for(size_t i = 0; i < eFusion->getFerns().frames.size(); i++)
            {
                if((int)i == eFusion->getFerns().lastClosest)
                    continue;

                gui->drawFrustum(eFusion->getFerns().frames.at(i)->pose);
            }
            glColor3f(1, 1, 1);
        }

        if(gui->drawDefGraph->Get())
        {
            const std::vector<GraphNode*> & graph = eFusion->getLocalDeformation().getGraph();

            for(size_t i = 0; i < graph.size(); i++)
            {
                pangolin::glDrawCross(graph.at(i)->position(0),
                                      graph.at(i)->position(1),
                                      graph.at(i)->position(2),
                                      0.1);

                for(size_t j = 0; j < graph.at(i)->neighbours.size(); j++)
                {
                    pangolin::glDrawLine(graph.at(i)->position(0),
                                         graph.at(i)->position(1),
                                         graph.at(i)->position(2),
                                         graph.at(graph.at(i)->neighbours.at(j))->position(0),
                                         graph.at(graph.at(i)->neighbours.at(j))->position(1),
                                         graph.at(graph.at(i)->neighbours.at(j))->position(2));
                }
            }
        }

        if(eFusion->getFerns().lastClosest != -1) {
            glColor3f(1, 0, 0);
            gui->drawFrustum(eFusion->getFerns().frames.at(eFusion->getFerns().lastClosest)->pose);
            glColor3f(1, 1, 1);
        }

        const std::vector<PoseMatch> & poseMatches = eFusion->getPoseMatches();

        // int maxDiff = 0;
        // for(size_t i = 0; i < poseMatches.size(); i++)
        // {
        //     if(poseMatches.at(i).secondId - poseMatches.at(i).firstId > maxDiff)
        //     {
        //         maxDiff = poseMatches.at(i).secondId - poseMatches.at(i).firstId;
        //     }
        // }

        for(size_t i = 0; i < poseMatches.size(); i++)
        {
            if(gui->drawDeforms->Get())
            {
                if(poseMatches.at(i).fern)
                {
                    glColor3f(1, 0, 0);
                }
                else
                {
                    glColor3f(0, 1, 0);
                }
                for(size_t j = 0; j < poseMatches.at(i).constraints.size(); j++)
                {
                    pangolin::glDrawLine(
                            poseMatches.at(i).constraints.at(j).sourcePoint(0),
                            poseMatches.at(i).constraints.at(j).sourcePoint(1),
                            poseMatches.at(i).constraints.at(j).sourcePoint(2),
                            poseMatches.at(i).constraints.at(j).targetPoint(0),
                            poseMatches.at(i).constraints.at(j).targetPoint(1),
                            poseMatches.at(i).constraints.at(j).targetPoint(2));
                }
            }
        }
        glColor3f(1, 1, 1);

        eFusion->normaliseDepth(0.3f, gui->depthCutoff->Get());

        // for(std::map<std::string, GPUTexture*>::const_iterator it = eFusion->getTextures().begin();
        //         it != eFusion->getTextures().end();
        //         ++it)
        // {
        //     if(it->second->draw)
        //     {
        //         gui->displayImg(it->first, it->second);
        //     }
        // }
        for (auto const& tup : eFusion->getTextures()) {
            if (tup.second->draw) {
                gui->displayImg(tup.first, tup.second);
            }
        }

        // Model Image Related
        auto& index_map = eFusion->getIndexMap();
        index_map.renderDepth(gui->depthCutoff->Get());
        gui->displayImg("ModelImg", index_map.imageTex());
        gui->displayImg("Model", index_map.drawTex());

        // Post Call
        gui->postCall();

        /*
        / Update Parameters
        */
        std::stringstream strs;
        strs << eFusion->getGlobalModel().lastCount();
        gui->totalPoints->operator=(strs.str());

        std::stringstream strs2;
        strs2 << eFusion->getLocalDeformation().getGraph().size();
        gui->totalNodes->operator=(strs2.str());

        std::stringstream strs3;
        strs3 << eFusion->getFerns().frames.size();
        gui->totalFerns->operator=(strs3.str());

        std::stringstream strs4;
        strs4 << eFusion->getDeforms();
        gui->totalDefs->operator=(strs4.str());

        std::stringstream strs5;
        strs5 << eFusion->getTick() << "/" << logReader->getNumFrames();
        gui->logProgress->operator=(strs5.str());

        std::stringstream strs6;
        strs6 << eFusion->getFernDeforms();
        gui->totalFernDefs->operator=(strs6.str());

        logReader->flipColors = gui->flipColors->Get();
        eFusion->setRgbOnly(gui->rgbOnly->Get());
        eFusion->setPyramid(gui->pyramid->Get());
        eFusion->setFastOdom(gui->fastOdom->Get());
        eFusion->setConfidenceThreshold(gui->confidenceThreshold->Get());
        eFusion->setDepthCutoff(gui->depthCutoff->Get());
        eFusion->setIcpWeight(gui->icpWeight->Get());
        eFusion->setSo3(gui->so3->Get());
        eFusion->setFrameToFrameRGB(gui->frameToFrameRGB->Get());

        if(gui->autoSettings)
        {
            std::cout << "autoSettings \n";
            static bool last = gui->autoSettings->Get();

            if(gui->autoSettings->Get() != last)
            {
                last = gui->autoSettings->Get();
                static_cast<LiveLogReader*>(logReader)->setAuto(last);
            }
        }

        Stopwatch::getInstance().sendAll();

        // Reset
        resetButton = pangolin::Pushed(*gui->reset);
        if(resetButton) {
            break;
        }
        // Save
        if(pangolin::Pushed(*gui->save)) {
            eFusion->savePly();
        }

        TOCK("GUI");
    }
}
