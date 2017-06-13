//
// Created by Zhou Wei Ran on 13/06/2017.
//

#ifndef GRABCUT_MYUTILITYOPENCV_H
#define GRABCUT_MYUTILITYOPENCV_H

#include "cv.h"

namespace myUtilityOpenCV {
    cv::Point2i point(int row, int col) {
        return cv::Point2i(col, row);
    }
}



#endif //GRABCUT_MYUTILITYOPENCV_H
