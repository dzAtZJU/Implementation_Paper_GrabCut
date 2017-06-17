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
public:// User
    void GrabCut( cv::InputArray _img, cv::InputOutputArray _mask, cv::Rect rect,
                  cv::InputOutputArray _bgdModel,cv::InputOutputArray _fgdModel,
                  int iterCount, int mode );
    ~GrabCut2D(void);

private:// Constructure
    void setInternalDataStructure(Mat _img, Mat _mask, Rect _rect);

private:// Implementer
    void initializeMaskGmm(Mat &fgModel, Mat &bgModel);
    void addFgPixelsToFgGmmComponents();
    void addBgPixelsToFgGmmComponents();
    void learnGMMParams();
    void constructGraph();

private:// Data Factory
    void reAssignPixelToGmmComponent(Point2i pixel, AreaMask area);
    void addBgPixelToBgGmmComponent(Point2i pixel);
    void addFgPixelToFgGmmComponent(Point2i pixel);

    void generateBGPixelsVector(vector<Point2i>& bgPixels);
    void generateFGPixelsVector(vector<Point2i>& fgPixels);
    void addPixelsInAnAreaToVector(vector<Point2i> &pixels, AreaMask area);

private:// Paper: BOYKOV, Y., AND JOLLY, M.-P. Interactive graph cuts for optimal boundary and region segmentation of objects in n-d images. In ICCV, 2001, 105–112.
    int nPixels();
    AreaMask markOfPixel(Point2i pixel);
    AreaMask markOfEle(int row, int col);
    int k_GMMCompOfPixel(int i);
    double edgeWeight_Pixel_Source(int pixelIndex);
    double computeK();
    double R(Point2i pixel, AreaMask mask);
    double lambda();
    double edgeWeight_Pixel_Terminal(int pixelIndex);
    vector<int> neighborsOfPixel(int pixelIndex);
    double edgeWeight_p_neighborQ(int p, int neighborQ);
    double B(Point2i p, Point2i neighborQ);
    int comp_of_pixel(int pixelIndex);
    int gmmComponentForPixel(Point2i pixel, AreaMask area);
private:// Paper
    double D(AreaMask alpha, int k_GMMComp, Vec3b z_color);
    double Beta();
    Mat getBgSamples(const Rect& rect);
    Mat getFgSamples(const Rect &rect);
    GMM gmmFG;
    GMM gmmBG;
    void generatePixelsInAreaByGaussComps(vector<vector<Point2i>>& v, AreaMask area);
    //vector<vector<Point2i>> pixelsInFgGaussComps = vector<vector<Point2i>>{5,vector<Point2i>()};
    //vector<vector<Point2i>> pixelsInBgGaussComps= vector<vector<Point2i>>{5,vector<Point2i>()};
    ImageGraph* graph = nullptr;
    void minCut();
    void updateMask();
    void reassignPixelsInTuToGmmComponents();
    double K = 0;
    void reAssignPixelsToGmmComponents();
private: //Data
    Mat image;//!not set; Must synchronized; Owner is outside!
    Mat mask;//!not set; Must synchronized; Owner is outside!
    Rect rectTu;
    vector<int> k;
private: //Data Accessor
    ImageAccessor imageAccessor;//!not set; Must synchronized;!
    Mat TuMaskMat() { return mask(rectTu); }
    void reassignPixelsInBGToComponents();
    int nPixelsInRect();
    void returnExternalDataStructure(cv::InputOutputArray _mask);
private: //test
    void test();
    void testInitializeMaskGmm();
    void testReAssignPixelsInBGToComponents();
    void testReAssignPixelsInTuToGmmComponents();
    void testMinCut();
};

