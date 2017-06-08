//
// Created by zz on 07/06/2017.
//

#ifndef GRABCUT_MYUTILITY_H
#define GRABCUT_MYUTILITY_H

#include <cv.h>
#include <utility>

class MyUtility {
public:
    static std::pair<int, int> rowColOfUpperLeftFromRect(cv::Rect rect) {
        return std::pair<int, int>(rect.y, rect.x);
    }
    template <class T>
    static void printVector(std::vector<T>& vec){
        std::cout<<"[";
        int i=0;
        for(auto ele: vec) {
            std::cout<<ele<<", ";
            i++;
        }
        std::cout<<"]";
    }
    template <class T>
    static void printArray(T vec[], int size){
        std::cout<<"[";
        int i=0;
        for(int i=0; i<size; i++) {
            std::cout<<vec[i]<<", ";
        }
        std::cout<<"]";
    }
    static const int AMOUNT_GMM_COMPNENTS = 5;
    static void testMat(const cv::Mat& mat) {
        int n = 0;
        for (int i = 0; i < mat.rows; ++i) {
            for (int j = 0; j < mat.cols; ++j) {
                if(mat.at<char>(i,j) == 1) {
                    n++;
                }
            }
        }
        std::cout<<"#MyUtility::testMat# "<<mat.rows<<" "<<mat.cols<<" "<<n;
    }

};


#endif //GRABCUT_MYUTILITY_H
