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

#include "RGBDOdometry.h"

#include"cnpy.h"
#include <cuda_profiler_api.h>

template<typename T>
void save(DeviceArray2D<T> dev_arr, std::string filename) {

    cudaSafeCall(cudaDeviceSynchronize());
    T* host_arr;
    int H = dev_arr.rows();
    int W = dev_arr.cols();
    host_arr = new T[H * W * sizeof(T)];
    // std::cout << "size " << sizeof(T) << "\n";
    // std::cout << nextImage[0].cols() << " " << nextImage[0].elem_size << "\n";
    dev_arr.download(host_arr, W * sizeof(T));
    cnpy::npy_save(filename, &host_arr[0], {H, W}, "w");
}


RGBDOdometry::RGBDOdometry(int width,
                           int height,
                           float cx, float cy, float fx, float fy,
                           float distThresh,
                           float angleThresh)
: lastICPError(0),
  lastICPCount(width * height),
  lastRGBError(0),
  lastRGBCount(width * height),
  lastSO3Error(0),
  lastSO3Count(width * height),
  lastA(Eigen::Matrix<double, 6, 6, Eigen::RowMajor>::Zero()),
  lastb(Eigen::Matrix<double, 6, 1>::Zero()),
  sobelSize(3),
  sobelScale(1.0 / pow(2.0, sobelSize)),
  maxDepthDeltaRGB(0.07),
  maxDepthRGB(6.0),
  distThres_(distThresh),
  angleThres_(angleThresh),
  width(width),
  height(height),
  cx(cx), cy(cy), fx(fx), fy(fy)
{
    sumDataSE3.create(MAX_THREADS);
    outDataSE3.create(1);
    sumResidualRGB.create(MAX_THREADS);

    sumDataSO3.create(MAX_THREADS);
    outDataSO3.create(1);

    for(int i = 0; i < NUM_PYRS; i++)
    {
        int2 nextDim = {height >> i, width >> i};
        pyrDims.push_back(nextDim);
    }

    for(int i = 0; i < NUM_PYRS; i++)
    {
        lastDepth[i].create(pyrDims.at(i).x, pyrDims.at(i).y);
        lastImage[i].create(pyrDims.at(i).x, pyrDims.at(i).y);

        nextDepth[i].create(pyrDims.at(i).x, pyrDims.at(i).y);
        nextImage[i].create(pyrDims.at(i).x, pyrDims.at(i).y);

        lastNextImage[i].create(pyrDims.at(i).x, pyrDims.at(i).y);

        nextdIdx[i].create(pyrDims.at(i).x, pyrDims.at(i).y);
        nextdIdy[i].create(pyrDims.at(i).x, pyrDims.at(i).y);

        pointClouds[i].create(pyrDims.at(i).x, pyrDims.at(i).y);

        corresImg[i].create(pyrDims.at(i).x, pyrDims.at(i).y);
    }

    intr.cx = cx;
    intr.cy = cy;
    intr.fx = fx;
    intr.fy = fy;

    iterations.resize(NUM_PYRS);

    depth_tmp.resize(NUM_PYRS);

    vmaps_g_prev_.resize(NUM_PYRS);
    nmaps_g_prev_.resize(NUM_PYRS);

    vmaps_curr_.resize(NUM_PYRS);
    nmaps_curr_.resize(NUM_PYRS);

    for (int i = 0; i < NUM_PYRS; ++i)
    {
        int pyr_rows = height >> i;
        int pyr_cols = width >> i;

        depth_tmp[i].create (pyr_rows, pyr_cols);

        vmaps_g_prev_[i].create (pyr_rows*3, pyr_cols);
        nmaps_g_prev_[i].create (pyr_rows*3, pyr_cols);

        vmaps_curr_[i].create (pyr_rows*3, pyr_cols);
        nmaps_curr_[i].create (pyr_rows*3, pyr_cols);
    }

    vmaps_tmp.create(height * 4 * width);
    nmaps_tmp.create(height * 4 * width);

    minimumGradientMagnitudes.resize(NUM_PYRS);
    minimumGradientMagnitudes[0] = 5;
    minimumGradientMagnitudes[1] = 3;
    minimumGradientMagnitudes[2] = 1;
}

RGBDOdometry::~RGBDOdometry()
{

}

void RGBDOdometry::initICP(GPUTexture * filteredDepth, const float depthCutoff)
{
    cudaArray * textPtr;

    cudaGraphicsMapResources(1, &filteredDepth->cudaRes);

    cudaGraphicsSubResourceGetMappedArray(&textPtr, filteredDepth->cudaRes, 0, 0);

    cudaMemcpy2DFromArray(
            depth_tmp[0].ptr(0), depth_tmp[0].step(), textPtr, 0, 0,
            depth_tmp[0].colsBytes(), depth_tmp[0].rows(),
            cudaMemcpyDeviceToDevice);

    cudaGraphicsUnmapResources(1, &filteredDepth->cudaRes);

    for(int i = 1; i < NUM_PYRS; ++i)
    {
        pyrDown(depth_tmp[i - 1], depth_tmp[i]);
        // save(depth_tmp[i-1], "depth_pyr_down.npy");
        // save(depth_tmp[i], "depth_pyr_down_output.npy");
    }

    for(int i = 0; i < NUM_PYRS; ++i)
    {
        // save(depth_tmp[i], "depth_vmap.npy");
        // save(vmaps_curr_[i], "vmaps_curr_before.npy");
        createVMap(intr(i), depth_tmp[i], vmaps_curr_[i], depthCutoff);
        // save(vmaps_curr_[i], "vmaps_curr_create.npy");
        // save(nmaps_curr_[i], "nmaps_curr_before.npy");
        createNMap(vmaps_curr_[i], nmaps_curr_[i]);
        // save(nmaps_curr_[i], "nmaps_curr_create.npy");
    }

    cudaDeviceSynchronize();
}

void RGBDOdometry::initICP(GPUTexture * predictedVertices, GPUTexture * predictedNormals, const float depthCutoff)
{
    cudaArray * textPtr;

    cudaGraphicsMapResources(1, &predictedVertices->cudaRes);
    cudaGraphicsSubResourceGetMappedArray(&textPtr, predictedVertices->cudaRes, 0, 0);
    cudaMemcpyFromArray(vmaps_tmp.ptr(), textPtr, 0, 0, vmaps_tmp.sizeBytes(), cudaMemcpyDeviceToDevice);
    cudaGraphicsUnmapResources(1, &predictedVertices->cudaRes);

    cudaGraphicsMapResources(1, &predictedNormals->cudaRes);
    cudaGraphicsSubResourceGetMappedArray(&textPtr, predictedNormals->cudaRes, 0, 0);
    cudaMemcpyFromArray(nmaps_tmp.ptr(), textPtr, 0, 0, nmaps_tmp.sizeBytes(), cudaMemcpyDeviceToDevice);
    cudaGraphicsUnmapResources(1, &predictedNormals->cudaRes);

    copyMaps(vmaps_tmp, nmaps_tmp, vmaps_curr_[0], nmaps_curr_[0]);

    for(int i = 1; i < NUM_PYRS; ++i)
    {
        resizeVMap(vmaps_curr_[i - 1], vmaps_curr_[i]);
        resizeNMap(nmaps_curr_[i - 1], nmaps_curr_[i]);
    }

    cudaDeviceSynchronize();
}

void RGBDOdometry::initICPModel(GPUTexture * predictedVertices,
                                GPUTexture * predictedNormals,
                                const float depthCutoff,
                                const Eigen::Matrix4f & modelPose)
{
    cudaArray * textPtr;

    cudaGraphicsMapResources(1, &predictedVertices->cudaRes);
    cudaGraphicsSubResourceGetMappedArray(&textPtr, predictedVertices->cudaRes, 0, 0);
    cudaMemcpyFromArray(vmaps_tmp.ptr(), textPtr, 0, 0, vmaps_tmp.sizeBytes(), cudaMemcpyDeviceToDevice);
    cudaGraphicsUnmapResources(1, &predictedVertices->cudaRes);

    cudaGraphicsMapResources(1, &predictedNormals->cudaRes);
    cudaGraphicsSubResourceGetMappedArray(&textPtr, predictedNormals->cudaRes, 0, 0);
    cudaMemcpyFromArray(nmaps_tmp.ptr(), textPtr, 0, 0, nmaps_tmp.sizeBytes(), cudaMemcpyDeviceToDevice);
    cudaGraphicsUnmapResources(1, &predictedNormals->cudaRes);

    copyMaps(vmaps_tmp, nmaps_tmp, vmaps_g_prev_[0], nmaps_g_prev_[0]);

    for(int i = 1; i < NUM_PYRS; ++i)
    {
        //save(vmaps_g_prev_[i-1], "vmaps_initICP.npy");
        //save(vmaps_g_prev_[i], "vmaps_initICP_output_before.npy");
        resizeVMap(vmaps_g_prev_[i - 1], vmaps_g_prev_[i]);
        // save(vmaps_g_prev_[i], "vmasp_initICP_output.npy");
        resizeNMap(nmaps_g_prev_[i - 1], nmaps_g_prev_[i]);
    }

    Eigen::Matrix<float, 3, 3, Eigen::RowMajor> Rcam = modelPose.topLeftCorner(3, 3);
    Eigen::Vector3f tcam = modelPose.topRightCorner(3, 1);

    mat33 device_Rcam = Rcam;
    float3 device_tcam = *reinterpret_cast<float3*>(tcam.data());

    
    cnpy::npy_save("Rcam_transform_maps.npy", Rcam.cast<float>().eval().data(), {3, 3}, "w", "C");
    cnpy::npy_save("tcam_transform_maps.npy", tcam.cast<float>().eval().data(), {3}, "w", "C");
    for(int i = 0; i < NUM_PYRS; ++i)
    {
        // save(vmaps_g_prev_[i], "vmaps_g_prev_transform_maps.npy");
        // save(nmaps_g_prev_[i], "nmaps_g_prev_transform_maps.npy");
        tranformMaps(vmaps_g_prev_[i], nmaps_g_prev_[i], device_Rcam, device_tcam, vmaps_g_prev_[i], nmaps_g_prev_[i]);
        // save(vmaps_g_prev_[i], "vmaps_g_prev_transform_maps_output.npy");
        // save(nmaps_g_prev_[i], "nmaps_g_prev_transform_maps_output.npy");
    }

    cudaDeviceSynchronize();
}

void RGBDOdometry::populateRGBDData(GPUTexture * rgb,
                                    DeviceArray2D<float> * destDepths,
                                    DeviceArray2D<unsigned char> * destImages)
{
    verticesToDepth(vmaps_tmp, destDepths[0], maxDepthRGB);

    for(int i = 0; i + 1 < NUM_PYRS; i++)
    {
        save(destDepths[i], "pyrDownFGauss_input.npy");
        pyrDownGaussF(destDepths[i], destDepths[i + 1]);
        save(destDepths[i + 1], "pyrDownFGauss_output.npy");
    }

    cudaArray * textPtr;

    cudaGraphicsMapResources(1, &rgb->cudaRes);

    cudaGraphicsSubResourceGetMappedArray(&textPtr, rgb->cudaRes, 0, 0);

    imageBGRToIntensity(textPtr, destImages[0]);

    cudaGraphicsUnmapResources(1, &rgb->cudaRes);

    for(int i = 0; i + 1 < NUM_PYRS; i++)
    {
        // save(destImages[i], "pyrDownUcharGauss_input.npy");
        pyrDownUcharGauss(destImages[i], destImages[i + 1]);
        // save(destImages[i+1], "pyrDownUcharGauss_output.npy");
    }

    cudaDeviceSynchronize();
}

void RGBDOdometry::initRGBModel(GPUTexture * rgb)
{
    //NOTE: This depends on vmaps_tmp containing the corresponding depth from initICPModel
    populateRGBDData(rgb, &lastDepth[0], &lastImage[0]);
}

void RGBDOdometry::initRGB(GPUTexture * rgb)
{
    //NOTE: This depends on vmaps_tmp containing the corresponding depth from initICP
    populateRGBDData(rgb, &nextDepth[0], &nextImage[0]);
}

void RGBDOdometry::initFirstRGB(GPUTexture * rgb)
{
    cudaArray * textPtr;

    cudaGraphicsMapResources(1, &rgb->cudaRes);

    cudaGraphicsSubResourceGetMappedArray(&textPtr, rgb->cudaRes, 0, 0);

    imageBGRToIntensity(textPtr, lastNextImage[0]);

    cudaGraphicsUnmapResources(1, &rgb->cudaRes);

    for(int i = 0; i + 1 < NUM_PYRS; i++)
    {
        pyrDownUcharGauss(lastNextImage[i], lastNextImage[i + 1]);
    }
}



void RGBDOdometry::getIncrementalTransformation(Eigen::Vector3f & trans,
                                                Eigen::Matrix<float, 3, 3, Eigen::RowMajor> & rot,
                                                const bool & rgbOnly,
                                                const float & icpWeight,
                                                const bool & pyramid,
                                                const bool & fastOdom,
                                                const bool & so3)
{
    bool icp = !rgbOnly && icpWeight > 0;
    bool rgb = rgbOnly || icpWeight < 100;

    Eigen::Matrix<float, 3, 3, Eigen::RowMajor> Rprev = rot;
    Eigen::Vector3f tprev = trans;

    Eigen::Matrix<float, 3, 3, Eigen::RowMajor> Rcurr = Rprev;
    Eigen::Vector3f tcurr = tprev;

    if(rgb)
    {
        for(int i = 0; i < NUM_PYRS; i++)
        {
            computeDerivativeImages(nextImage[i], nextdIdx[i], nextdIdy[i]);
        }
    }

    Eigen::Matrix<double, 3, 3, Eigen::RowMajor> resultR = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>::Identity();

    if(so3)
    {
        int pyramidLevel = 2;

        Eigen::Matrix<float, 3, 3, Eigen::RowMajor> R_lr = Eigen::Matrix<float, 3, 3, Eigen::RowMajor>::Identity();

        Eigen::Matrix<double, 3, 3, Eigen::RowMajor> K = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>::Zero();

        K(0, 0) = intr(pyramidLevel).fx;
        K(1, 1) = intr(pyramidLevel).fy;
        K(0, 2) = intr(pyramidLevel).cx;
        K(1, 2) = intr(pyramidLevel).cy;
        K(2, 2) = 1;

        float lastError = std::numeric_limits<float>::max() / 2;
        float lastCount = std::numeric_limits<float>::max() / 2;

        Eigen::Matrix<double, 3, 3, Eigen::RowMajor> lastResultR = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>::Identity();

        save(lastNextImage[pyramidLevel], "lastNextImage.npy");
        save(nextImage[pyramidLevel], "nextImage.npy");
        cnpy::npy_save("k.npy", K.cast<float>().eval().data(), {3, 3}, "w", "C");
        for(int i = 0; i < 10; i++)
        {
            Eigen::Matrix<float, 3, 3, Eigen::RowMajor> jtj;
            Eigen::Matrix<float, 3, 1> jtr;

            Eigen::Matrix<double, 3, 3, Eigen::RowMajor> homography = K * resultR * K.inverse();

            mat33 imageBasis;
            memcpy(&imageBasis.data[0], homography.cast<float>().eval().data(), sizeof(mat33));

            Eigen::Matrix<double, 3, 3, Eigen::RowMajor> K_inv = K.inverse();
            mat33 kinv;
            memcpy(&kinv.data[0], K_inv.cast<float>().eval().data(), sizeof(mat33));

            Eigen::Matrix<double, 3, 3, Eigen::RowMajor> K_R_lr = K * resultR;
            mat33 krlr;
            memcpy(&krlr.data[0], K_R_lr.cast<float>().eval().data(), sizeof(mat33));
            cnpy::npy_save("imageBasis.npy", homography.cast<float>().eval().data(), {3, 3}, "w", "C");
            cnpy::npy_save("kinv.npy", K_inv.cast<float>().eval().data(), {3, 3}, "w", "C");
            cnpy::npy_save("krlr.npy", K_R_lr.cast<float>().eval().data(), {3, 3}, "w", "C");

            float residual[2];

            TICK("so3Step");
            so3Step(lastNextImage[pyramidLevel],
                    nextImage[pyramidLevel],
                    imageBasis,
                    kinv,
                    krlr,
                    sumDataSO3,
                    outDataSO3,
                    jtj.data(),
                    jtr.data(),
                    &residual[0],
                    GPUConfig::getInstance().so3StepThreads,
                    GPUConfig::getInstance().so3StepBlocks);
            TOCK("so3Step");
            cnpy::npy_save("jtj.npy", jtj.cast<float>().eval().data(), {3, 3}, "w", "C");
            cnpy::npy_save("jtr.npy", jtr.cast<float>().eval().data(), {3, 1}, "w", "C");
            // throw std::invalid_argument("a");

            lastSO3Error = sqrt(residual[0]) / residual[1];
            lastSO3Count = residual[1];

            //Converged
            if(lastSO3Error < lastError && lastCount == lastSO3Count)
            {
                std::cout<< "converged \n";
                break;
            }
            else if(lastSO3Error > lastError + 0.001) //Diverging
            {
                std::cout<< "diverging \n";
                lastSO3Error = lastError;
                lastSO3Count = lastCount;
                resultR = lastResultR;
                break;
            }

            lastError = lastSO3Error;
            lastCount = lastSO3Count;
            lastResultR = resultR;

            Eigen::Vector3f delta = jtj.ldlt().solve(jtr);
            std::cout <<delta(0) << " " << delta(1) << " " << delta(2) << " delta \n";

            Eigen::Matrix<double, 3, 3, Eigen::RowMajor> rotUpdate = OdometryProvider::rodrigues(delta.cast<double>());
            std::cout<< rotUpdate<< "\n\n";

            R_lr = rotUpdate.cast<float>() * R_lr;

            for(int x = 0; x < 3; x++)
            {
                for(int y = 0; y < 3; y++)
                {
                    resultR(x, y) = R_lr(x, y);
                }
            }
        }
    }
    cnpy::npy_save("resultR.npy", resultR.cast<float>().eval().data(), {3, 3}, "w", "C");

    iterations[0] = fastOdom ? 3 : 10;
    iterations[1] = pyramid ? 5 : 0;
    iterations[2] = pyramid ? 4 : 0;

    Eigen::Matrix<float, 3, 3, Eigen::RowMajor> Rprev_inv = Rprev.inverse();
    std::cout << " Rprev_inv " << Rprev_inv << "\n";
    mat33 device_Rprev_inv = Rprev_inv;
    float3 device_tprev = *reinterpret_cast<float3*>(tprev.data());

    Eigen::Matrix<double, 4, 4, Eigen::RowMajor> resultRt = Eigen::Matrix<double, 4, 4, Eigen::RowMajor>::Identity();

    if(so3)
    {
        for(int x = 0; x < 3; x++)
        {
            for(int y = 0; y < 3; y++)
            {
                resultRt(x, y) = resultR(x, y);
            }
        }
    }

    for(int i = NUM_PYRS - 1; i >= 0; i--)
    {
        if(rgb)
        {
            save(lastDepth[i], "lastDepth_2.npy");
            projectToPointCloud(lastDepth[i], pointClouds[i], intr, i);
        }

        Eigen::Matrix<double, 3, 3, Eigen::RowMajor> K = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>::Zero();

        K(0, 0) = intr(i).fx;
        K(1, 1) = intr(i).fy;
        K(0, 2) = intr(i).cx;
        K(1, 2) = intr(i).cy;
        K(2, 2) = 1;
        cnpy::npy_save("k_2.npy", K.cast<float>().eval().data(), {3, 3}, "w", "C");

        lastRGBError = std::numeric_limits<float>::max();

        cnpy::npy_save("resultRt.npy", resultRt.cast<float>().eval().data(), {4, 4}, "w", "C");
        for(int j = 0; j < iterations[i]; j++)
        {
            Eigen::Matrix<double, 4, 4, Eigen::RowMajor> Rt = resultRt.inverse();

            Eigen::Matrix<double, 3, 3, Eigen::RowMajor> R = Rt.topLeftCorner(3, 3);

            Eigen::Matrix<double, 3, 3, Eigen::RowMajor> KRK_inv = K * R * K.inverse();
            mat33 krkInv;
            memcpy(&krkInv.data[0], KRK_inv.cast<float>().eval().data(), sizeof(mat33));

            Eigen::Vector3d Kt = Rt.topRightCorner(3, 1);
            Kt = K * Kt;
            float3 kt = {(float)Kt(0), (float)Kt(1), (float)Kt(2)};

            int sigma = 0;
            int rgbSize = 0;

            if(rgb)
            {
                save(lastDepth[i], "lastDepth_3.npy");
                save(nextDepth[i], "nextDepth_3.npy");
                save(nextdIdx[i], "nextdIdx_3.npy");
                save(nextdIdy[i], "nextdIdy_3.npy");
                save(lastImage[i], "lastImage_3.npy");
                save(nextImage[i], "nextImage_3.npy");
                TICK("computeRgbResidual");
                std::cout << "minimumGradient " << minimumGradientMagnitudes[i] << " sobelScale " << sobelScale << "\n";
                std::cout << pow(minimumGradientMagnitudes[i], 2.0) / pow(sobelScale, 2.0) << "\n";
                std::cout << "maxDepthRGB " << maxDepthDeltaRGB << "\n";
                computeRgbResidual(pow(minimumGradientMagnitudes[i], 2.0) / pow(sobelScale, 2.0),
                                   nextdIdx[i],
                                   nextdIdy[i],
                                   lastDepth[i],
                                   nextDepth[i],
                                   lastImage[i],
                                   nextImage[i],
                                   corresImg[i],
                                   sumResidualRGB,
                                   maxDepthDeltaRGB,
                                   kt,
                                   krkInv,
                                   sigma,
                                   rgbSize,
                                   GPUConfig::getInstance().rgbResThreads,
                                   GPUConfig::getInstance().rgbResBlocks);
                TOCK("computeRgbResidual");
            }
            // save(corresImg[i], "corresImage_3.npy");

            float sigmaVal = std::sqrt((float)sigma / rgbSize == 0 ? 1 : rgbSize);
            float rgbError = std::sqrt(sigma) / (rgbSize == 0 ? 1 : rgbSize);

            if(rgbOnly && rgbError > lastRGBError)
            {
                break;
            }

            lastRGBError = rgbError;
            lastRGBCount = rgbSize;

            if(rgbOnly)
            {
                sigmaVal = -1; //Signals the internal optimisation to weight evenly
            }

            Eigen::Matrix<float, 6, 6, Eigen::RowMajor> A_icp;
            Eigen::Matrix<float, 6, 1> b_icp;

            mat33 device_Rcurr = Rcurr;
            float3 device_tcurr = *reinterpret_cast<float3*>(tcurr.data());

            DeviceArray2D<float>& vmap_curr = vmaps_curr_[i];
            DeviceArray2D<float>& nmap_curr = nmaps_curr_[i];

            DeviceArray2D<float>& vmap_g_prev = vmaps_g_prev_[i];
            DeviceArray2D<float>& nmap_g_prev = nmaps_g_prev_[i];

            float residual[2];

            if(icp)
            {
                // Pose calculated in the previous iteration (NOT the intermediate values)
                // cnpy::npy_save("icp_Rcurr.npy", Rcurr.cast<float>().eval().data(), {3, 3}, "w", "C");
                // cnpy::npy_save("tcurr.npy", tcurr.cast<float>().eval().data(), {3}, "w", "C");
                // cnpy::npy_save("Rprev_inv.npy", Rprev_inv.cast<float>().eval().data(), {3, 3}, "w", "C");
                // cnpy::npy_save("tprev.npy", tprev.cast<float>().eval().data(), {3}, "w", "C");

                // save(vmap_curr, "vmap_curr.npy");
                // save(nmap_curr, "nmap_curr.npy");
                // save(vmap_g_prev, "vmap_g_prev.npy");
                // save(nmap_g_prev, "nmap_g_prev.npy");
                TICK("icpStep");
                icpStep(device_Rcurr,
                        device_tcurr,
                        vmap_curr,
                        nmap_curr,
                        device_Rprev_inv,
                        device_tprev,
                        intr(i),
                        vmap_g_prev,
                        nmap_g_prev,
                        distThres_,
                        angleThres_,
                        sumDataSE3,
                        outDataSE3,
                        A_icp.data(),
                        b_icp.data(),
                        &residual[0],
                        GPUConfig::getInstance().icpStepThreads,
                        GPUConfig::getInstance().icpStepBlocks);
                TOCK("icpStep");
                // cnpy::npy_save("A_icp.npy", A_icp.cast<float>().eval().data(), {6, 6}, "w", "C");
                // cnpy::npy_save("B_icp.npy", b_icp.cast<float>().eval().data(), {6, 1}, "w", "C");
            }

            lastICPError = sqrt(residual[0]) / residual[1];
            lastICPCount = residual[1];

            Eigen::Matrix<float, 6, 6, Eigen::RowMajor> A_rgbd;
            Eigen::Matrix<float, 6, 1> b_rgbd;

            if(rgb)
            {
                TICK("rgbStep");
                save(nextdIdx[i], "nextdIdx_rgbdstep.npy");
                save(nextdIdy[i], "nextdIdy_rgbdstep.npy");
                std::cout << " sobelScale " << sobelScale << " \n";
                std::cout << " sigmaVal " << sigmaVal << " \n";
                rgbStep(corresImg[i],
                        sigmaVal,
                        pointClouds[i],
                        intr(i).fx,
                        intr(i).fy,
                        nextdIdx[i],
                        nextdIdy[i],
                        sobelScale,
                        sumDataSE3,
                        outDataSE3,
                        A_rgbd.data(),
                        b_rgbd.data(),
                        GPUConfig::getInstance().rgbStepThreads,
                        GPUConfig::getInstance().rgbStepBlocks);
                TOCK("rgbStep");
                cnpy::npy_save("A_rgbd.npy", A_rgbd.cast<float>().eval().data(), {6, 6}, "w", "C");
                cnpy::npy_save("B_rgbd.npy", b_rgbd.cast<float>().eval().data(), {6}, "w", "C");
                cudaProfilerStop();
                throw std::invalid_argument("a");
            }

            Eigen::Matrix<double, 6, 1> result;
            Eigen::Matrix<double, 6, 6, Eigen::RowMajor> dA_rgbd = A_rgbd.cast<double>();
            Eigen::Matrix<double, 6, 6, Eigen::RowMajor> dA_icp = A_icp.cast<double>();
            Eigen::Matrix<double, 6, 1> db_rgbd = b_rgbd.cast<double>();
            Eigen::Matrix<double, 6, 1> db_icp = b_icp.cast<double>();

            if(icp && rgb)
            {
                double w = icpWeight;
                lastA = dA_rgbd + w * w * dA_icp;
                lastb = db_rgbd + w * db_icp;
                result = lastA.ldlt().solve(lastb);
            }
            else if(icp)
            {
                lastA = dA_icp;
                lastb = db_icp;
                result = lastA.ldlt().solve(lastb);
            }
            else if(rgb)
            {
                lastA = dA_rgbd;
                lastb = db_rgbd;
                result = lastA.ldlt().solve(lastb);
            }
            else
            {
                assert(false && "Control shouldn't reach here");
            }

            Eigen::Isometry3f rgbOdom;

            OdometryProvider::computeUpdateSE3(resultRt, result, rgbOdom);

            Eigen::Isometry3f currentT;
            currentT.setIdentity();
            currentT.rotate(Rprev);
            currentT.translation() = tprev;

            currentT = currentT * rgbOdom.inverse();

            tcurr = currentT.translation();
            Rcurr = currentT.rotation();
        }
    }

    if(rgb && (tcurr - tprev).norm() > 0.3)
    {
        Rcurr = Rprev;
        tcurr = tprev;
    }

    if(so3)
    {
        for(int i = 0; i < NUM_PYRS; i++)
        {
            std::swap(lastNextImage[i], nextImage[i]);
        }
    }

    trans = tcurr;
    rot = Rcurr;
}

Eigen::MatrixXd RGBDOdometry::getCovariance()
{
    return lastA.cast<double>().lu().inverse();
}
