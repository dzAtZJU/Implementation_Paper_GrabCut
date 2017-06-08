#pragma once
#include <cv.h>
#include <vector>
#include <iostream>
#include "GaussDistribution.h"
#include "GMM.h"

using namespace cv;
enum
{
    GC_WITH_RECT  = 0,
    GC_WITH_MASK  = 1,
    GC_CUT        = 2
};
enum
{
    MASK_B = 0,
    MASK_F = 1,
    MASK_PB = 2,
    MASK_PF= 3
};
class GrabCut2D
{
public:
    void GrabCut( cv::InputArray _img, cv::InputOutputArray _mask, cv::Rect rect,
                  cv::InputOutputArray _bgdModel,cv::InputOutputArray _fgdModel,
                  int iterCount, int mode );
    ~GrabCut2D(void);
    void test();
private:
    void initializeMaskAlphaGMM(Mat &_mask, const Rect &rect, const Mat &_img, Mat &fgModel);
    void assignGMMComponentsToFGPixels(const Mat& fgModel, const Mat& mask, const Mat& image);
    void generateFGPixelsVector(vector<Point2i>& fgPixels, const Mat& mask);
    void assignGMMComponentsToFGPixel(Point2i pixel, const Mat &fgModel, const Mat& image);
    void testAssignGMMComponentsToFGPixels();
    void learnGMMParams(Mat& model);

private:
    Mat getBgSamples(const Mat& _img, const Rect& rect);
    Mat getFgSamples(const Mat& _img, const Rect& rect);
    GMM gaussFG;
    GMM gaussBG;
    vector<vector<Point2i>> pixelsInFgGaussComps = vector<vector<Point2i>>{5,vector<Point2i>()};
    vector<vector<Point2i>> pixelsInBgGaussComps= vector<vector<Point2i>>{5,vector<Point2i>()};
};

