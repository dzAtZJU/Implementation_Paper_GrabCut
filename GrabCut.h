#pragma once
#include <cv.h>
#include <vector>
#include <iostream>
#include "GaussDistribution.h"
#include "GMM.h"
#include "maxflow-v3/graph.h"
#include "ImageAccessor.h"
#include "MyUtilityOpenCV.h"

using namespace cv;
enum
{
    GC_WITH_RECT  = 0,
    GC_WITH_MASK  = 1,
    GC_CUT        = 2
};

enum AreaMask
{
    MASK_B = 0,
    MASK_F = 1,
    MASK_PB = 2,
    MASK_PF= 3
};

using ImageGraph = Graph<double,double,double>;
using MaskType = char;

class GrabCut2D
{
public://Grab Cut Interface
    void GrabCut( cv::InputArray _img, cv::InputOutputArray _mask, cv::Rect rect,
                  cv::InputOutputArray _bgdModel,cv::InputOutputArray _fgdModel,
                  int iterCount, int mode );
    ~GrabCut2D(void){}

private://Grab CutAlgorithm Core
    void initializeGmm();
        Mat getBgSamples(const Rect& rect);
        Mat getFgSamples(const Rect &rect);

    void reAssignPixelsToGmmComponents();
        void reAssignPixelToGmmComponentForArea(Point2i pixel, AreaMask area);
    
    void learnGMMParams();

    void minCut();
    void constructGraph();
        double edgeWeight_Pixel_Source(int pixelIndex);
        double edgeWeight_Pixel_Terminal(int pixelIndex);
            double lambda();
            double computeK();
            double D(AreaMask alpha, int k_GMMComp, Vec3b z_color);
            double B(Point2i p, Point2i neighborQ);
                double gama();
                double Beta();
            double R(Point2i pixel, AreaMask mask);
                int gmmComponentIfPixelIsInArea(Point2i pixel, AreaMask area);
    double edgeWeight_p_neighborQ(int p, int neighborQ);

private: //Data Structure
    Mat image;
        ImageAccessor imageAccessor;
    Rect rectTu;
    void copy_Image_and_Rect(Mat _img, Rect _rect);
    Mat mask;
        Mat TuMaskMat() { return mask(rectTu); }
        void receiveMask(const Mat &_mask);
        void returnMask(cv::InputOutputArray _mask);
        void updateMaskFromMinCut();
        AreaMask markOfPixel(Point2i pixel);
    vector<int> k;
    GMM gmmFG;
    GMM gmmBG;
    void extraDataStructureForConvinientUse_Vectors_PixelsInAreaByGaussComps(vector<vector<Point2i>> &v, AreaMask area);
    double K = 0;
    double beta = 0;
    ImageGraph* graph = nullptr;
};

