//
// Created by Zhou Wei Ran on 13/06/2017.
//

#ifndef GRABCUT_MYUTILITYOPENCV_H
#define GRABCUT_MYUTILITYOPENCV_H

#include "cv.h"
#include "vector"
using namespace cv;

Point2i point(int row, int col);
std::vector<cv::Point2i> neighbors(Point2i pixel, int rows, int cols);
int indexOfPixel(Point2i pixel, int rows, int cols);
int indexofPixel(Point2i pixel, Point2i origin, int rows, int cols);
void printMat(const Mat& mat);

#endif //GRABCUT_MYUTILITYOPENCV_H
