#include <highgui.h>
#include "GrabCut.h"
#include "maxflow-v3/graph.h"
#include "MyUtilityOpenCV.h"

using namespace std;
using namespace cv;
using namespace myUtilityOpenCV;

//User
/// @attention
void GrabCut2D::GrabCut( cv::InputArray _img, cv::InputOutputArray _mask, cv::Rect rect, cv::InputOutputArray _bgdModel,cv::InputOutputArray _fgModel, int iterCount, int mode )
{
    std::cout<<"GrabCut2D::GrabCut"<<std::endl;
    image = _img.getMat();
    mask = _mask.getMat();
    auto fgModelMat = _fgModel.getMat();

    switch(mode) {
        case GC_WITH_RECT: initializeMaskGmm(rect, fgModelMat); cout<<"maskMat is the modified one?"<<endl;
        case GC_CUT: break;
        case GC_WITH_MASK: break;
        default: assert(false);
    }

    addFgPixelsToFgGmmComponents(); cout<<"fgModelMat is the modified one?"<<endl;
    learnGMMParams(fgModelMat);
    minCut();
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

//Implementer

void GrabCut2D::initializeMaskGmm(const Rect &rect, Mat &fgModel) {
    // InitializeMask
    cout<<"#GrabCut2D::initializeMaskGmm# Mask Zeroed At First"<<endl;
    mask = MASK_B;
    mask(rect) = MASK_F;

    // Initialize FG GMM
    auto fgSamples = getFgSamples(rect);
    Mat fgLabels(rect.height*rect.width, 1, CV_32SC1);
    gmmFG.estimateParas(fgSamples, noArray(), fgLabels);
    gmmFG.testEstimateParas();
    gmmFG.constructFGModelFromEM(fgModel);

    /*
    // Initialize BG GMM
    auto bgSamples = getBgSamples(, rect);
    Mat bgLabels(image.cols*image.rows - rect.height*rect.width, 1, CV_32SC1);
    gmmBG.estimateParas(bgSamples, noArray(), bgLabels);
    gmmBG.testEstimateParas();
     */
}

void GrabCut2D::addFgPixelsToFgGmmComponents() {
    vector<Point2i> fgPixels;
    auto& fgPixelsRef = fgPixels; cout<<"#GrabCut2D::addFgPixelsToFgGmmComponents# fgPixelsRef not sure"<<endl;
    generateFGPixelsVector(fgPixelsRef);

    for(auto pixel:fgPixelsRef) {
        addFgPixelToFgGmmComponent(pixel);
    }
    testAssignGMMComponentsToFGPixels();
}

void GrabCut2D::addFgPixelToFgGmmComponent(Point2i pixel) {
    //cout<<"#GrabCut2D::addFgPixelToFgGmmComponent# image depth? channel?"<<endl;
    auto intensity = imageAccessor.pixelValue_At(pixel);

    auto Ds = map<int, double>();
    for (int i = 0; i < gmmFG.nComps(); ++i) {
        auto D = gmmFG.minusLogProbDensConstDeled_at_Comp_Sample(i, intensity);
        Ds.insert(pair<int, double>(i, D));
    }
    auto iterMaxD = max_element(Ds.begin(), Ds.end());

    if(iterMaxD!=Ds.end()) {
        auto component = (*iterMaxD).first;
        pixelsInFgGaussComps[component].push_back(pixel);
    }
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

// Image Data
Mat GrabCut2D::getFgSamples(const Rect &rect) {// Syn with Mask
    auto fg = image(rect).clone();
    fg = fg.reshape(1, rect.height * rect.width);
    return fg;
}

void GrabCut2D::generateFGPixelsVector(vector<Point2i>& fgPixels) {
    int n = 0;
    int fn = 0;
    for (int i = 0; i < mask.rows; ++i) {
        for (int j = 0; j < mask.cols; ++j) {
            n++;
            auto theMask = markOfEle(i, j);
            if((theMask==MASK_F) || (theMask==MASK_PF)) {
                fgPixels.push_back(point(i, j));
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
    cout<<"GrabCut2D::learnGMMParams"<<endl;
    gmmFG.estimateParas(pixelsInFgGaussComps, image);
    gmmFG.constructFGModel(model);
    //5 comps, each has pi, mean, cov
}

GrabCut2D::~GrabCut2D(void)
{
}

void GrabCut2D::constructGraph() {
    graph = ImageGraph(/*estimated # of nodes*/ 400, /*estimated # of edges*/ 3200); cout<<"#GrabCut2D::constructGraph()# !Implement estimated nodes and edges"<<endl;

    for (int i = 0; i < imageAccessor.nPixels(); ++i) {
        graph.add_node();
        auto comp = comp_of_pixel(i);
        graph.add_tweights( i, edgeWeight_Pixel_Source(i, comp), edgeWeight_Pixel_Terminal(i, comp) );
    }

    for (int i = 0; i < imageAccessor.nPixels(); ++i) {
        auto neighbors = imageAccessor.neighbors_Of(i);
        for(auto neighbor:neighbors) {
            if (neighbor > i) {
                graph.add_edge(i, neighbor, edgeWeight_p_neighborQ(i, neighbor), edgeWeight_p_neighborQ(neighbor, i));
            }
        }
    }
}

int GrabCut2D::nPixels() {
    cout<<"#GrabCut2D::nPixels# !Not Implement"<<endl;
    return 0;
}

AreaMask GrabCut2D::markOfPixel(Point2i pixel) {
    auto _theMask = mask.at<MaskType>(pixel);
    auto theMask = AreaMask(_theMask);
    return theMask;
}

double GrabCut2D::edgeWeight_Pixel_Source(int pixelIndex, int k_GMMComp) {
    auto area = markOfPixel(imageAccessor.coordOfPixel(pixelIndex));

    double weight;
    switch(area) {
        case MASK_B: weight = 0; break;
        case MASK_F: weight = K(); break;
        default: {
            auto cood = imageAccessor.coordOfPixel(pixelIndex);
            weight = lambda()*R(cood, MASK_B, k_GMMComp);
        }
    }

    return weight;
}

double GrabCut2D::edgeWeight_Pixel_Terminal(int pixelIndex, int k_GMMComp) {
    auto area = markOfPixel(imageAccessor.coordOfPixel(pixelIndex));

    double weight;
    switch(area) {
        case MASK_B: weight = K(); break;
        case MASK_F: weight = 0; break;
        default: {
            auto cood = imageAccessor.coordOfPixel(pixelIndex);
            weight = lambda()*R(cood, MASK_F, k_GMMComp);
        }
    }

    return weight;
}

double GrabCut2D::K() {
    double maxZigma = DBL_MIN;

    for (int i = 0; i < imageAccessor.nPixels(); ++i) {
        double zigma = 0;

        auto cood_i = imageAccessor.coordOfPixel(i);
        auto neighbors = imageAccessor.neighbors_Of(i);
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

double GrabCut2D::R(Point2i pixel, AreaMask mask, int k_GMMComp) {
    auto alpha = markOfPixel(pixel);
    auto color = imageAccessor.pixelValue_At(pixel);
    return D(alpha, k_GMMComp, color);
}

double GrabCut2D::lambda() {
    cout<<"#GrabCut2D::lambda# !Not Implement"<<endl;
    return 0;
}

vector<int> GrabCut2D::neighborsOfPixel(int pixelIndex) {
    cout<<"#GrabCut2D::neighborsOfPixel# !Not Implement"<<endl;
    return vector<int, allocator<int>>();
}

double GrabCut2D::edgeWeight_p_neighborQ(int p, int neighborQ) {
    auto pCoord = imageAccessor.coordOfPixel(p);
    auto qCoord = imageAccessor.coordOfPixel(neighborQ);
    return B(pCoord, qCoord);
}

double GrabCut2D::B(Point2i p, Point2i neighborQ) {

    auto colorDistance = imageAccessor.distanceInColorSpace_Square(p, neighborQ);
    auto result = exp(-Beta()*colorDistance);
    return result;
}

double GrabCut2D::Beta() {
    cout<<"#GrabCut2D::Beta# !Not Implement"<<endl;
    return 0;
}

double GrabCut2D::D(AreaMask alpha, int k_GMMComp, Vec3b z_color) {
    double d;
    switch(alpha){
        case MASK_F:  d = gmmFG.minusLogProbDensConstDeled_at_Comp_Sample(k_GMMComp, Matx31d(z_color));
        default: assert(false);
    }
    return d;
}

int GrabCut2D::k_GMMCompOfPixel(int i) {
    cout<<"#GrabCut2D::k_GMMCompOfPixel# !Not Implement"<<endl;
    return 0;
}

AreaMask GrabCut2D::markOfEle(int row, int col) {
    return markOfPixel(point(col, row));
}

int GrabCut2D::comp_of_pixel(int pixelIndex) {
    int k_comp = 0;
    for(auto comp:pixelsInFgGaussComps) {
        for(auto pixel: comp) {
            auto index = imageAccessor.indexOfPixel(pixel);
            if(index == pixelIndex) {
                return k_comp;
            }
        }
        k_comp++;
    }
    assert(false);
    return -1;
}

void GrabCut2D::minCut() {
    constructGraph();
    graph.maxflow();
    updateMask();
}

void GrabCut2D::updateMask() {
    for (int i = 0; i < imageAccessor.nPixels(); ++i) {
        if(graph.what_segment(i) == ImageGraph::SOURCE) {
            mask.at<MaskType>(imageAccessor.coordOfPixel(i)) = MASK_F;//!Change to Mask Accessor!
        }
        else {
            mask.at<MaskType>(imageAccessor.coordOfPixel(i)) = MASK_B;//!Change to Mask Accessor!
        }
    }
}


