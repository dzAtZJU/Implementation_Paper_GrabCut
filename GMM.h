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
    void estimateParas(vector<vector<Point2i>>& samplesVec, const Mat& valueMat) {
        int nAllSamples = 0;
        for(auto samples:samplesVec) {
            nAllSamples += samples.size();
        }
        for(int i=0; i<samplesVec.size(); i++) {
            estimateParasForComponent(i, samplesVec[i], valueMat, nAllSamples);
        }
        testEstimateParasSamplesVec();
    }

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
    void constructFGModelFromEM(Mat &fgModel) {
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

    void constructFGModel(Mat &fgModel) {
        /*
        Mat weights;
        weights = Mat(pis).t();
        weights.copyTo(fgModel(Range(0,1),Range::all()));
        */
        cout<<"#GMM::constructFGModel# not implemented yet"<<endl;
    }

private:
    void estimateParasForComponent(int i, vector<Point2i>& samples, const Mat& valueMat, int nAllSamples) {
        //mean
        Vec3d mean = {0, 0, 0};
        for(auto sample:samples) {
            auto color = valueMat.at<Vec3b>(sample);
            mean += color;
        }
        mean = mean/double(samples.size());
        auto itMean = means.begin() + i;
        means.insert(itMean, Matx31d(mean));

        //cov
        auto cov = Mat(3,3,CV_64FC1);
        cov = Scalar(0);
        for(auto sample:samples) {
            auto color = valueMat.at<Vec3b>(sample);
            Mat colorMat(3,1,CV_64FC1);
            for(auto v:color.val){
                colorMat.push_back(v);
            }
            cov +=  colorMat*colorMat.t();
        }
        cov /= samples.size();
        auto itCovs = covs.begin() + i;
        covs.insert(itCovs, Matx33d(cov));

        //pi
        auto itPi = pis.begin() + i;
        auto pi = samples.size()/nAllSamples;
        pis.insert(itPi, pi);
    }

private:
    static constexpr int NClusters = 5;
    /// @weights 1*5
    /// @mean 5*3
    /// @cov vector 5, element 3*3
    EM em = EM(NClusters, EM::COV_MAT_GENERIC, TermCriteria(CV_TERMCRIT_ITER, 1, 0.1));
    vector<Matx33d> covs = vector<Matx33d>(3, Matx33d());
    vector<Matx31d> means = vector<Matx31d>(3, Matx31d());
    vector<double> pis;

public:
    //Test
    void testEstimateParas() {
        auto weights = em.get<Mat>("weights");
        auto mean = em.get<Mat>("means");
        auto cov = em.get<vector<Mat>>("covs");
        ;
    }

    void testEstimateParasSamplesVec() {
        cout<<"#GMM::testEstimateParasSamplesVec#"<<endl;
        for(int i=0; i<NClusters; i++) {
            cout<<"Component "<<i<<endl;
            cout << "pi:" << pis[i] << endl;
            double *mu = means[i].val;
            MyUtility::printArray(mu, 3);
            double *cov = covs[i].val;
            MyUtility::printArray(cov, 9);
        }
    }
};


#endif //GRABCUT_GMM_H
