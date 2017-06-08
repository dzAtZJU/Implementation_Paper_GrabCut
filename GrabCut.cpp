#include <highgui.h>
#include "GrabCut.h"
using namespace std;
using namespace cv;

void GrabCut2D::GrabCut( cv::InputArray _img, cv::InputOutputArray _mask, cv::Rect rect, cv::InputOutputArray _bgdModel,cv::InputOutputArray _fgModel, int iterCount, int mode )
{
    std::cout<<"GrabCut2D::GrabCut"<<std::endl;

    auto maskMat = _mask.getMat();
    auto imgMat = _img.getMat();
    auto fgModelMat = _fgModel.getMat();
    initializeMaskAlphaGMM(maskMat, rect, imgMat, fgModelMat);

    cout<<"maskMat is the modified one?"<<endl;
    assignGMMComponentsToFGPixels(fgModelMat, maskMat, imgMat);
    cout<<"fgModelMat is the modified one?"<<endl;
    learnGMMParams(fgModelMat);
//一.参数解释：
	//输入：
	 //cv::InputArray _img,     :输入的color图像(类型-cv:Mat)
     //cv::Rect rect            :在图像上画的矩形框（类型-cv:Rect) 
  	//int iterCount :           :每次分割的迭代次数（类型-int)


	//中间变量
	//cv::InputOutputArray _bgdModel ：   背景模型（推荐GMM)（类型-13*n（组件个数）个double类型的自定义数据结构，可以为cv:Mat，或者Vector/List/数组等）
	//cv::InputOutputArray _fgdModel :    前景模型（推荐GMM) （类型-13*n（组件个数）个double类型的自定义数据结构，可以为cv:Mat，或者Vector/List/数组等）


	//输出:
	//cv::InputOutputArray _mask  : 输出的分割结果 (类型： cv::Mat)

//二. 伪代码流程：
	//1.Load Input Image: 加载输入颜色图像;
	//2.Init Mask: 用矩形框初始化Mask的Label值（确定背景：0， 确定前景：1，可能背景：2，可能前景：3）,矩形框以外设置为确定背景，矩形框以内设置为可能前景;
	//3.Init GMM: 定义并初始化GMM(其他模型完成分割也可得到基本分数，GMM完成会加分）
	//4.Sample Points:前背景颜色采样并进行聚类（建议用kmeans，其他聚类方法也可)
	//5.Learn GMM(根据聚类的样本更新每个GMM组件中的均值、协方差等参数）
	//4.Construct Graph（计算t-weight(数据项）和n-weight（平滑项））
	//7.Estimate Segmentation(调用maxFlow库进行分割)
	//8.Save Result输入结果（将结果mask输出，将mask中前景区域对应的彩色图像保存和显示在交互界面中）
	
}

void GrabCut2D::initializeMaskAlphaGMM(Mat &_mask, const Rect &rect, const Mat &_img, Mat &fgModel) {
    // InitializeMask
    cout<<"#GrabCut2D::initializeMaskAlphaGMM# Mask Zeroed At First"<<endl;
    _mask = MASK_B;
    _mask(rect) = MASK_F;

    // Initialize FG GMM
    auto fgSamples = getFgSamples(_img, rect);
    Mat fgLabels(rect.height*rect.width, 1, CV_32SC1);
    gaussFG.estimateParas(fgSamples, noArray(), fgLabels);
    gaussFG.testEstimateParas();
    gaussFG.constructFGModel(fgModel);

    // Initialize BG GMM
    auto bgSamples = getBgSamples(_img, rect);
    Mat bgLabels(_img.cols*_img.rows - rect.height*rect.width, 1, CV_32SC1);
    gaussBG.estimateParas(bgSamples, noArray(), bgLabels);
    gaussBG.testEstimateParas();
}

void GrabCut2D::test() {
    string filename = "/Users/tgbus/Desktop/bird.png";
    Mat image = imread( filename, 1 );
    cout<<"Image depth not known"<<endl;
    if(image.empty() )
    {
        cout << "\n , couldn't read image filename " << filename << endl;
        return;
    }
    Mat mask;
    mask.create(image.size(), CV_8UC1);
    mask.setTo( MASK_B );
    Rect rect;
    rect.x = 10;
    rect.y = 10;
    rect.width = 100;
    rect.height = 100;
    auto maskFG = mask(rect);
    maskFG.setTo(MASK_F);
    Mat fgm, bgm;
    GrabCut(image, mask, rect, fgm, bgm, 100, GC_WITH_MASK);
}

Mat GrabCut2D::getBgSamples(const Mat &_img, const Rect &rect) {
    auto rowColOfUpperLeft = MyUtility::rowColOfUpperLeftFromRect(rect);
    auto beginRow = rowColOfUpperLeft.first;
    auto beginCol = rowColOfUpperLeft.second;
    auto overRow = beginRow + rect.height;
    auto overCol = beginCol + rect.width;
    auto bgUp = _img(Range(0, beginRow), Range::all()).clone();
    auto bgBottom = _img(Range(overRow, _img.rows), Range::all()).clone();
    auto bgLeft = _img(Range::all(), Range(0, beginCol)).clone();
    auto bgRight = _img(Range::all(), Range(overCol, _img.cols)).clone();
    bgUp = bgUp.reshape(1,bgUp.rows*bgUp.cols);
    bgBottom = bgBottom.reshape(1,bgBottom.rows*bgBottom.cols);
    bgLeft = bgLeft.reshape(1,bgLeft.rows*bgLeft.cols);
    bgRight = bgRight.reshape(1,bgRight.rows*bgRight.cols);

    bgUp.push_back(bgBottom);bgUp.push_back(bgLeft);bgUp.push_back(bgRight);
    return bgUp;
}

Mat GrabCut2D::getFgSamples(const Mat &_img, const Rect &rect) {
    auto fg = _img(rect).clone();
    fg = fg.reshape(1, rect.height * rect.width);
    return fg;
}

void GrabCut2D::assignGMMComponentsToFGPixels(const Mat &fgModel, const Mat &mask, const Mat &image) {
    MyUtility::testMat(mask);
    vector<Point2i> fgPixels;
    auto& fgPixelsRef = fgPixels;
    cout<<"#GrabCut2D::assignGMMComponentsToFGPixels# fgPixelsRef not sure"<<endl;
    generateFGPixelsVector(fgPixelsRef, mask);
    for(auto pixel:fgPixelsRef) {
        assignGMMComponentsToFGPixel(pixel, fgModel, image);
    }
    testAssignGMMComponentsToFGPixels();
}

void GrabCut2D::assignGMMComponentsToFGPixel(Point2i pixel, const Mat &fgModel, const Mat &image) {
    cout<<"#GrabCut2D::assignGMMComponentsToFGPixel# image depth? channel?"<<endl;
    auto Ds = map<int, double>();
    auto amountGMMComponents = fgModel.cols;
    for (int i = 0; i < amountGMMComponents; ++i) {
        auto gmmComponent = fgModel(Range::all(), Range(i, i+1));
        auto gauss = GaussDistribution(gmmComponent);
        auto intensity = image.at<Vec3b>(pixel);
        auto D = gauss.minusLogProbDensConstDeled(intensity);
        Ds.insert(pair<int, double>(i, D));
    }
    auto iterMaxD = max_element(Ds.begin(), Ds.end());
    if(iterMaxD!=Ds.end()) {
        auto component = (*iterMaxD).first;
        pixelsInFgGaussComps[component].push_back(pixel);
    }
}

void GrabCut2D::generateFGPixelsVector(vector<Point2i>& fgPixels, const Mat& mask) {
    int n = 0;
    int fn = 0;
    for (int i = 0; i < mask.rows; ++i) {
        for (int j = 0; j < mask.cols; ++j) {
            n++;
            auto theMask = mask.at<char>(i, j);
            if((theMask==MASK_F) || (theMask==MASK_PF)) {
                fgPixels.push_back(Point2i(j, i));
                fn++;
            }
        }
    }
    cout<<"#GrabCut2D::generateFGPixelsVector# "<<n<<" "<<fn<<endl;
}

void GrabCut2D::testAssignGMMComponentsToFGPixels() {
    for(auto comp:pixelsInFgGaussComps) {
        cout<<"#fg gauss comp# "<<comp.size()<<" "<<"pixels"<<endl;
    }
}

void GrabCut2D::learnGMMParams(Mat &model) {
    pixelsInFgGaussComps;
    gaussFG;
    //5 comps, each has pi, mean, cov
}

GrabCut2D::~GrabCut2D(void)
{
}



