#include <highgui.h>
#include "GrabCut.h"
#include "maxflow-v3/graph.h"

using namespace std;
using namespace cv;

//2.Init Mask: 用矩形框初始化Mask的Label值（确定背景：0， 确定前景：1，可能背景：2，可能前景：3）,矩形框以外设置为确定背景，矩形框以内设置为可能前景;
//3.Init GMM: 定义并初始化GMM(其他模型完成分割也可得到基本分数，GMM完成会加分）
//4.Sample Points:前背景颜色采样并进行聚类（建议用kmeans，其他聚类方法也可)
//5.Learn GMM(根据聚类的样本更新每个GMM组件中的均值、协方差等参数）
//4.Construct Graph（计算t-weight(数据项）和n-weight（平滑项））
//7.Estimate Segmentation(调用maxFlow库进行分割)
//8.Save Result输入结果（将结果mask输出，将mask中前景区域对应的彩色图像保存和显示在交互界面中）
/// @attention
void GrabCut2D::GrabCut( cv::InputArray _img, cv::InputOutputArray _mask, cv::Rect rect, cv::InputOutputArray _bgdModel,cv::InputOutputArray _fgModel, int iterCount, int mode )
{
    switch(mode) {
        case GC_WITH_RECT: {
            copy_Image_and_Rect(_img.getMat(), rect);
            initializeGmm();
        }
        case GC_CUT: {
            receiveMask(_mask.getMat()); // update background&foreground pixels from user
            reAssignPixelsToGmmComponents(); // update k
            learnGMMParams(); // update theta
            minCut(); //update alpha
            break;
        }
        case GC_WITH_MASK: assert(false); break;
        default: assert(false);
    }
    returnMask(_mask);
}

// Data exchange with GrabCut invoker
void GrabCut2D::copy_Image_and_Rect(Mat _img, Rect _rect) {
    image = _img;
    rectTu = _rect;
    k = vector<int>(image.rows*image.cols, -1);
    imageAccessor = ImageAccessor(image);
}
void GrabCut2D::receiveMask(const Mat &_mask) {
    mask = _mask;
}
void GrabCut2D::returnMask(cv::InputOutputArray _mask) {
    mask.copyTo(_mask);
}

//Algorithm Core

//Background and foreground GMMs initialised
void GrabCut2D::initializeGmm() {
    // Initialize FG GMM
    auto fgSamples = getFgSamples(rectTu);
    Mat fgLabels(rectTu.height*rectTu.width, 1, CV_32SC1);
    gmmFG.estimateParas(fgSamples, noArray(), fgLabels);

    // Initialize BG GMM
    auto bgSamples = getBgSamples(rectTu);
    Mat bgLabels(image.cols*image.rows - rectTu.height*rectTu.width, 1, CV_32SC1);
    gmmBG.estimateParas(bgSamples, noArray(), bgLabels);
}
Mat GrabCut2D::getBgSamples(const Rect& rect) {
    auto rowColOfUpperLeft = MyUtility::rowColOfUpperLeftFromRect(rect);
    auto beginRow = rowColOfUpperLeft.first;
    auto beginCol = rowColOfUpperLeft.second;
    auto overRow = beginRow + rect.height;
    auto overCol = beginCol + rect.width;
    auto bgUp = image(Range(0, beginRow), Range::all()).clone();
    auto bgBottom = image(Range(overRow, image.rows), Range::all()).clone();
    auto bgLeft = image(Range::all(), Range(0, beginCol)).clone();
    auto bgRight = image(Range::all(), Range(overCol, image.cols)).clone();
    bgUp = bgUp.reshape(1,bgUp.rows*bgUp.cols);
    bgBottom = bgBottom.reshape(1,bgBottom.rows*bgBottom.cols);
    bgLeft = bgLeft.reshape(1,bgLeft.rows*bgLeft.cols);
    bgRight = bgRight.reshape(1,bgRight.rows*bgRight.cols);

    bgUp.push_back(bgBottom);bgUp.push_back(bgLeft);bgUp.push_back(bgRight);
    return bgUp;
}
Mat GrabCut2D::getFgSamples(const Rect &rect) {// Syn with Mask
    auto fg = image(rect).clone();
    fg = fg.reshape(1, rect.height * rect.width);
    return fg;
}

//Assign GMM components to pixels
void GrabCut2D::reAssignPixelsToGmmComponents() {
    for (int i = 0; i < imageAccessor.nPixels(); ++i) {
        auto cood = imageAccessor.coordOfPixel(i);
        auto area = AreaMask(mask.at<MaskType>(cood));
        reAssignPixelToGmmComponentForArea(cood, area);
    }
}
void GrabCut2D::reAssignPixelToGmmComponentForArea(Point2i pixel, AreaMask area) {
    auto intensity = imageAccessor.pixelValue_At(pixel);
    GMM gmm;
    if ((area==MASK_F) || (area==MASK_PF)) {
        gmm = gmmFG;
    }
    else {
        gmm = gmmBG;
    }

    auto Ds = map<int, double>();
    for (int i = 0; i < gmm.nComps(); ++i) {
        auto D = gmm.minusLogProbDensConstDeled_at_Comp_Sample(i, intensity);
        Ds.insert(pair<int, double>(i, D));
    }
    auto iterMaxD = max_element(Ds.begin(), Ds.end());

    if(iterMaxD!=Ds.end()) {
        auto component = (*iterMaxD).first;
        k[imageAccessor.indexOfPixel(pixel)] = component;
    }
}

//Learn GMM parameters
void GrabCut2D::learnGMMParams() {
    cout<<"#GrabCut2D::learnGMMParams# Begin"<<endl;
    auto pixelsInFgByGaussComps = vector<vector<Point2i>>{5,vector<Point2i>()};
    extraDataStructureForConvinientUse_Vectors_PixelsInAreaByGaussComps(pixelsInFgByGaussComps, MASK_F);
    extraDataStructureForConvinientUse_Vectors_PixelsInAreaByGaussComps(pixelsInFgByGaussComps, MASK_PF);
    gmmFG.estimateParas(pixelsInFgByGaussComps, image);
    //gmmFG.constructFGModel(model);
    //5 comps, each has pi, mean, cov
    auto pixelsInBgByGaussComps = vector<vector<Point2i>>{5,vector<Point2i>()};
    extraDataStructureForConvinientUse_Vectors_PixelsInAreaByGaussComps(pixelsInFgByGaussComps, MASK_B);
    extraDataStructureForConvinientUse_Vectors_PixelsInAreaByGaussComps(pixelsInFgByGaussComps, MASK_PB);
    gmmBG.estimateParas(pixelsInFgByGaussComps, image);
    cout<<"#GrabCut2D::learnGMMParams# End"<<endl;
}
void GrabCut2D::extraDataStructureForConvinientUse_Vectors_PixelsInAreaByGaussComps(vector<vector<Point2i>> &v,
                                                                                    AreaMask area) {
    assert(v.size() == 5);
    for (int i = 0; i < k.size(); ++i) {
        auto pixelArea = mask.at<MaskType>(imageAccessor.coordOfPixel(i));
        auto pixelK = k[i];
        if(pixelArea == area) {
            v[pixelK].push_back(imageAccessor.coordOfPixel(i));
        }
    }
}

//min cut
void GrabCut2D::minCut() {
    constructGraph();
    graph->maxflow();
    updateMaskFromMinCut();
}
void GrabCut2D::constructGraph() {
    K = computeK();
    beta = Beta();
    graph = new ImageGraph(2500, 20000);
    for (int i = 0; i < imageAccessor.nPixels(); ++i) {
        graph->add_node();
        graph->add_tweights( i, edgeWeight_Pixel_Source(i), edgeWeight_Pixel_Terminal(i) );
    }

    for (int i = 0; i < imageAccessor.nPixels(); ++i) {
        auto ns = neighbors(imageAccessor.coordOfPixel(i), imageAccessor.rows(), imageAccessor.cols());
        for(auto neighbor:ns) {
            auto nIndex = imageAccessor.indexOfPixel(neighbor);
            if (nIndex > i) {
                auto edgeWeight = edgeWeight_p_neighborQ(i, nIndex);
                graph->add_edge(i, nIndex, edgeWeight, edgeWeight);
            }
        }
    }
}
double GrabCut2D::edgeWeight_Pixel_Source(int pixelIndex) {
    auto area = markOfPixel(imageAccessor.coordOfPixel(pixelIndex));

    double weight;
    switch(area) {
        case MASK_B: weight = 0; break;
        case MASK_F: weight = K; break;
        default: {
            auto cood = imageAccessor.coordOfPixel(pixelIndex);
            weight = lambda()* R(cood, MASK_B);
        }
    }
    return weight;
}
double GrabCut2D::edgeWeight_Pixel_Terminal(int pixelIndex) {
    auto area = markOfPixel(imageAccessor.coordOfPixel(pixelIndex));

    double weight;
    switch(area) {
        case MASK_B: weight = K; break;
        case MASK_F: weight = 0; break;
        default: {
            auto cood = imageAccessor.coordOfPixel(pixelIndex);
            weight = lambda()* R(cood, MASK_F);
        }
    }

    return weight;
}
AreaMask GrabCut2D::markOfPixel(Point2i pixel) {
    auto _theMask = mask.at<MaskType>(pixel);
    auto theMask = AreaMask(_theMask);
    return theMask;
}
double GrabCut2D::computeK() {
    double maxZigma = DBL_MIN;

    for (int i = 0; i < imageAccessor.nPixels(); ++i) {
        double zigma = 0;

        auto cood_i = imageAccessor.coordOfPixel(i);
        auto neighbors = imageAccessor.neighborsInRect_Of(i);
        for(auto n:neighbors) {
            auto cood_n = imageAccessor.coordOfPixel(n);
            zigma += B(cood_i, cood_n);
        }

        if(zigma>maxZigma) {
            maxZigma = zigma;
        }
    }

    return maxZigma + 1;
}
double GrabCut2D::lambda() {
    return 0.1;
}
double GrabCut2D::R(Point2i pixel, AreaMask mask) {
    auto color = imageAccessor.pixelValue_At(pixel);
    int k_gmmComp = gmmComponentIfPixelIsInArea(pixel, mask);
    return D(mask, k_gmmComp, color);
}
int GrabCut2D::gmmComponentIfPixelIsInArea(Point2i pixel, AreaMask area) {
    auto intensity = imageAccessor.pixelValue_At(pixel);
    auto gmm = area==MASK_F? gmmFG: gmmBG;

    auto Ds = map<int, double>();
    for (int i = 0; i < gmm.nComps(); ++i) {
        auto D = gmm.minusLogProbDensConstDeled_at_Comp_Sample(i, intensity);
        Ds.insert(pair<int, double>(i, D));
    }
    auto iterMaxD = max_element(Ds.begin(), Ds.end());

    return  (*iterMaxD).first;
}
double GrabCut2D::D(AreaMask alpha, int k_GMMComp, Vec3b z_color) {
    double d;
    switch(alpha){
        case MASK_F:  d = gmmFG.minusLogProbDensConstDeled_at_Comp_Sample(k_GMMComp, Matx31d(z_color));break;
        case MASK_B: d = gmmBG.minusLogProbDensConstDeled_at_Comp_Sample(k_GMMComp, Matx31d(z_color));break;
        default: assert(false);
    }
    return d;
}
double GrabCut2D::edgeWeight_p_neighborQ(int p, int neighborQ) {
    auto pCoord = imageAccessor.coordOfPixel(p);
    auto qCoord = imageAccessor.coordOfPixel(neighborQ);
    return B(pCoord, qCoord);
}
double GrabCut2D::B(Point2i p, Point2i neighborQ) {
    auto colorDistance = imageAccessor.distanceInColorSpace_Square(p, neighborQ);
    auto result = gama()*exp(-beta*colorDistance);
    return result;
}
double GrabCut2D::gama() {
    return 50.0;
}
double GrabCut2D::Beta() {
    double expetation = 0;
    int number = 0;
    for (int i = 0; i < imageAccessor.rows(); ++i) {
        for (int j = 0; j < imageAccessor.cols(); ++j) {
            auto pixel = point(i, j);
            auto ns = neighbors(pixel, imageAccessor.rows(), imageAccessor.cols());
            for(auto n:ns) {
                auto dist = imageAccessor.distanceInColorSpace_Square(pixel, n);
                expetation = (expetation*(number++) + dist)/number;
            }
        }
    }

    return 1/(2*expetation);
}
void GrabCut2D::updateMaskFromMinCut() {
    auto m = TuMaskMat();
    for (int i = 0; i < m.rows; ++i) {
        for (int j = 0; j < m.cols; ++j) {
            auto index = indexofPixel(point(i, j), Point2i(rectTu.x, rectTu.y), imageAccessor.rows(),
                                      imageAccessor.cols());
            auto cood = imageAccessor.coordOfPixel(index);
            auto originalMask = int(mask.at<MaskType>(cood));
            if( (originalMask!=MASK_B) && (originalMask!=MASK_F) ) {
                if (graph->what_segment(index) == ImageGraph::SOURCE) {
                    mask.at<MaskType>(cood) = MASK_PF;
                } else {
                    mask.at<MaskType>(cood) = MASK_PB;
                }
            }
            else {
                cout<<"false";
            }
        }
    }
}

