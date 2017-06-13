//
// Created by Zhou Wei Ran on 13/06/2017.
//

#include "ImageAccessor.h"

ImageAccessor::ImageAccessor(Mat& image) : image(&image) {}

Vec3b ImageAccessor::pixelValue_At(Point2i pixelPosition) {
    return image->at<PixelType>(pixelPosition);
}

ImageAccessor::ImageAccessor() {}

double ImageAccessor::distanceInColorSpace_Square(Point2i p, Point2i q) {
    auto color_p = pixelValue_At(p);
    auto color_q = pixelValue_At(q);
    auto diff = color_p - color_q;
    auto d2 = diff.dot(diff);
    return d2;
}

int ImageAccessor::rows() {
    return image->rows;
}

int ImageAccessor::cols() {
    return image->cols;
}

int ImageAccessor::nPixels() {
    return rows()*cols();
}

vector<int> ImageAccessor::neighbors_Of(int pixelIndex) {
    cout<<"#ImageAccessor::neighbors_Of# !Not Implement"<<endl;
    return vector<int, allocator<int>>();
}

int ImageAccessor::indexOfPixel(Point2i p) {
    return p.y*cols() + p.x;
}

Point2i ImageAccessor::coordOfPixel(int pixelIndex) {
    auto y = pixelIndex % cols();
    auto x = pixelIndex - y * cols();
    return Point2i(x, y);
}


