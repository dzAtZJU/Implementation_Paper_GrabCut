#pragma once
#include <cv.h>
#include <vector>
#include <iostream>
#include "GaussDistribution.h"
#include "GMM.h"
#include "maxflow-v3/graph.h"
#include "ImageAccessor.h"

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
using PixelType = cv::Vec3b;
using MaskType = char;

class GrabCut2D
{
public:// User
    void GrabCut( cv::InputArray _img, cv::InputOutputArray _mask, cv::Rect rect,
                  cv::InputOutputArray _bgdModel,cv::InputOutputArray _fgdModel,
                  int iterCount, int mode );
    ~GrabCut2D(void);
    void test();

private:// Implementer
    void initializeMaskAlphaGMM(Mat &_mask, const Rect &rect, const Mat &_img, Mat &fgModel);
    void learnGMMParams(Mat& model, const Mat& image);
    void assignGMMComponentsToFGPixels(const Mat& fgModel, const Mat& mask, const Mat& image);
    void constructGraph();

private:// Helper
    void testAssignGMMComponentsToFGPixels();
    void generateFGPixelsVector(vector<Point2i>& fgPixels, const Mat& mask);
    void assignGMMComponentsToFGPixel(Point2i pixel, const Mat &fgModel, const Mat& image);

private:// Paper: BOYKOV, Y., AND JOLLY, M.-P. Interactive graph cuts for optimal boundary and region segmentation of objects in n-d images. In ICCV, 2001, 105–112.
    int nPixels();
    AreaMask markOfPixel(Point2i pixel);
    int k_GMMCompOfPixel(int i);
    double edgeWeight_Pixel_Source(int pixelIndex, int k_GMMComp);
    double K();
    double R(Point2i pixel, AreaMask mask, int k_GMMComp);
    double lambda();
    double edgeWeight_Pixel_Terminal(int pixelIndex, int k_GMMComp);
    vector<int> neighborsOfPixel(int pixelIndex);
    double edgeWeight_p_neighborQ(int p, int neighborQ);
    double B(Point2i p, Point2i neighborQ);

private:// Paper
    double D(AreaMask alpha, int k_GMMComp, Vec3b z_color);
    double Beta();
    Mat getBgSamples(const Mat& _img, const Rect& rect);
    Mat getFgSamples(const Mat& _img, const Rect& rect);
    GMM gmmFG;
    GMM gmmBG;
    vector<vector<Point2i>> pixelsInFgGaussComps = vector<vector<Point2i>>{5,vector<Point2i>()};
    vector<vector<Point2i>> pixelsInBgGaussComps= vector<vector<Point2i>>{5,vector<Point2i>()};
    ImageGraph  graph;

private: //Data
    Mat& image;//!not set; Must synchronized; Owner is outside!
    Mat& mask;//!not set; Must synchronized; Owner is outside!
private: //Data Accessor
    ImageAccessor imageAccessor;//!not set; Must synchronized;!
};

