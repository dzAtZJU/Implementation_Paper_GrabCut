//
// Created by zz on 06/06/2017.
//

#ifndef GRABCUT_GAUSSDISTRIBUTION_H
#define GRABCUT_GAUSSDISTRIBUTION_H

#include <cv.h>
#include <ml.h>
#include <vector>
#include "MyUtility.h"
using namespace cv;
using namespace std;

class GaussDistribution {
public:
    double _pi = 0;
    Matx33d _cov;
    Matx31d _mu;

    GaussDistribution(){};

    /// @param model 13*1 pi mean cov
    GaussDistribution(const Mat& gmmComponent) {
        assert((gmmComponent.rows==13) && (gmmComponent.cols==1));
        _pi = gmmComponent.at<double>(0,0);
        _mu = gmmComponent(Range(1,4), Range::all()).clone();
        _cov = gmmComponent(Range(4,13), Range::all()).clone().reshape(1, 3);
    }
    /// @param fgModel 13*n
    void constructFGModel(Mat& fgModel) {
        fgModel.create(13, NClusters, CV_64F);
        auto weights = em.get<Mat>("weights");
        auto means = em.get<Mat>("means");
        auto covs = em.get<vector<Mat>>("covs");
        weights.copyTo(fgModel(Range(0,1),Range::all()));
        Mat mean3by5;
        mean3by5 = means.t();
        mean3by5.copyTo(fgModel(Range(1,4),Range::all()));
        for(int i=0; i<NClusters; i++) {
            auto cov = covs[i];
            cov.reshape(1,9).copyTo(fgModel(Range(4,13),Range(i,i+1)));
        }
    }

    double minusLogProbDensConstDeled(Matx31d z) {
        assert((_cov.rows==3) && (_cov.cols==3) && (_mu.rows==3) && (_mu.cols==1));
        auto result = minusLogProbDensConstDeled(z, _mu, _cov);
        result += -log(_pi);
        return result;
    }

    static double minusLogProbDensConstDeled(Matx31d z, Matx31d mu, Matx33d cov) {

        auto covDet = determinant(cov);
        auto covInv  = Mat(cov.inv());

        auto zMuDiff = Mat(z) - Mat(mu);
        auto zMuDiffTrans = zMuDiff.t();

        Mat temp;
        temp = zMuDiffTrans * covInv * zMuDiff;
        auto product = temp.at<double>(0,0);

        auto result = log(covDet) + product;
        result = result/2;
        return result;
    }

    /// @param labels CV_32SC1
    /*
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
     */

    void testMinusLogProbDensConstDeled() {
        Matx33d cov = {1,1,1, 1,2,1, 1,1,2};
        Matx31d mu = {0,0,0}, z = {1,1,1};
        auto val = minusLogProbDensConstDeled(z, mu, cov);
        val += 1.5 * log(2*M_PI);
        auto p = exp(-val);
        cout<<"Test: GaussDistribution::Prob({0,0,0},{1,1,1, 1,2,1, 1,1,2}) = " <<p<<endl;
    }

    void testEstimateParas() {
        auto weights = em.get<Mat>("weights");
        auto mean = em.get<Mat>("means");
        auto cov = em.get<vector<Mat>>("covs");
        ;
    }

private:
    static constexpr int NClusters = 5;
    /// @weights 1*5
    /// @mean 5*3
    /// @cov 3*3
    EM em = EM(NClusters, EM::COV_MAT_GENERIC, TermCriteria(CV_TERMCRIT_ITER, 1, 0.1));
};

#endif //GRABCUT_GAUSSDISTRIBUTION_H
