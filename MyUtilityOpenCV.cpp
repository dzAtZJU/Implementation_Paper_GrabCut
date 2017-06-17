//
// Created by Zhou Wei Ran on 13/06/2017.
//

#include "MyUtilityOpenCV.h"

cv::Point2i point(int row, int col) {
    return cv::Point2i(col, row);
}

std::vector<Point2i> neighbors(Point2i pixel, int rows, int cols) {
    std::vector<Point2i> ns;
    for (int i = -1; i < 1; ++i) {
        for (int j = -1; j < 1; ++j) {
            auto neighbor = Point2i(pixel.x + i, pixel.y + i);
            auto row = neighbor.y, col = neighbor.x;
            if((i!=0) || (j!=0)) {
                if ((row >= 0) && (row < rows) && (col >= 0) && (col < cols)) {
                    ns.push_back(neighbor);
                }
            }
        }
    }
    return ns;
}

int indexOfPixel(Point2i pixel, int rows, int cols) {
    return pixel.y*cols + pixel.x;
}

void printMat(const Mat &mat) {
    for (int i = 0; i < mat.rows; ++i) {
        for (int j = 0; j < mat.cols; ++j) {
            std::cout<<int(mat.at<char>(i, j)) << " ";
        }
        std::cout<<std::endl;
    }
}

int indexofPixel(Point2i pixel, Point2i origin, int rows, int cols) {
    auto cood = pixel + origin;
    return indexOfPixel(cood, rows, cols);
}
