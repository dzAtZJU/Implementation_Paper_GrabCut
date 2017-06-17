//
// Created by Zhou Wei Ran on 13/06/2017.
//

#ifndef GRABCUT_IMAGEACCESSOR_H
#define GRABCUT_IMAGEACCESSOR_H

#include <vector>
#include <cv.h>
#include "MyUtility.h"
using namespace cv;
using namespace std;

/*
class MatAccessor {
public:
    MatAccessor();
    MatAccessor(Mat& mat);
    /// @attention Conform to OpenCV Image Coordinate
    int nPixels();
    int rows();
    int cols();
    vector<int> neighborsInRect_Of(int pixelIndex);
    int indexOfPixel(Point2i p);
    Point2i coordOfPixel(int pixelIndex);
private:
    Mat* mat = nullptr;
};
 */
using PixelType = cv::Vec3b;

class ImageAccessor {
public:
    ImageAccessor();
    ImageAccessor(Mat& image);
    /// @attention Conform to OpenCV Image Coordinate
    int nPixels();
    int rows();
    int cols();
    vector<int> neighborsInRect_Of(int pixelIndex);
    Vec3b pixelValue_At(Point2i pixelPosition);
    double distanceInColorSpace_Square(Point2i p, Point2i q);
    int indexOfPixel(Point2i p);
    Point2i coordOfPixel(int pixelIndex);
private:
    Mat* image = nullptr;
};


#endif //GRABCUT_IMAGEACCESSOR_H
