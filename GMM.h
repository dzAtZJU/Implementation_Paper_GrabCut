//
// Created by zz on 08/06/2017.
//

#ifndef GRABCUT_GMM_H
#define GRABCUT_GMM_H

#include <cv.h>
#include <ml.h>
#include <vector>
#include "MyUtility.h"
using namespace cv;
using namespace std;

class GMM {
public:
    /// @param labels CV_32SC1
    bool estimateParas(InputArray samples, OutputArray logLikelihoods=noArray(), OutputArray labels=noArray()) {
        cout<<"#GaussDistribution::estimateParas# Positive Definite Covariance; labels type CV_32SC1"<<endl;
        auto r = false;
        try {
            r = em.train(samples, noArray(), labels);
            return r;
        }
        catch (Exception& e) {
            cout<<e.what()<<endl;
            return false;
        }
    }

    /// @attention Only use after estimateParas
    /// @param fgModel 13*n
    void constructFGModel(Mat& fgModel) {
        fgModel.create(13, NClusters, CV_64F);
        auto weights = em.get<Mat>("weights"); assert(weights.cols==NClusters);
        auto means = em.get<Mat>("means"); assert((means.cols==3) && (means.rows==5));
        auto covs = em.get<vector<Mat>>("covs"); assert(covs.size()==5);
        weights.copyTo(fgModel(Range(0,1),Range::all()));
        Mat mean3by5;
        mean3by5 = means.t();
        mean3by5.copyTo(fgModel(Range(1,4),Range::all()));
        for(int i=0; i<NClusters; i++) {
            auto cov = covs[i];
            cov.reshape(1,9).copyTo(fgModel(Range(4,13),Range(i,i+1)));
        }
    }

private:
    static constexpr int NClusters = 5;
    /// @weights 1*5
    /// @mean 5*3
    /// @cov vector 5, element 3*3
    EM em = EM(NClusters, EM::COV_MAT_GENERIC, TermCriteria(CV_TERMCRIT_ITER, 1, 0.1));
    vector<Matx33d> covs;
    vector<Matx31d> means;
    vector<double> pi;

public:
    //Test
    void testEstimateParas() {
        auto weights = em.get<Mat>("weights");
        auto mean = em.get<Mat>("means");
        auto cov = em.get<vector<Mat>>("covs");
        ;
    }
};


#endif //GRABCUT_GMM_H
