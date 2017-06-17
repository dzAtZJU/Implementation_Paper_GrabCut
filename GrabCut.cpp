#include <highgui.h>
#include "GrabCut.h"
#include "maxflow-v3/graph.h"

using namespace std;
using namespace cv;

//Alg
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
    std::cout<<"GrabCut2D::GrabCut"<<std::endl;
    setInternalDataStructure(_img.getMat(), _mask.getMat(), rect);
    auto fgModelMat = _fgModel.getMat();
    auto bgModelMat = _bgdModel.getMat();
    switch(mode) {
        case GC_WITH_RECT:
            initializeMaskGmm(fgModelMat, bgModelMat); cout << "maskMat is the modified one?" << endl;
        case GC_CUT: {
            reAssignPixelsToGmmComponents(); cout<<"fgModelMat is the modified one?"<<endl;
            learnGMMParams();
            minCut();
            break;
        }
        case GC_WITH_MASK: assert(false); break;
        default: assert(false);
    }
    returnExternalDataStructure(_mask);
    ;
}

// Construct
void GrabCut2D::setInternalDataStructure(Mat _img, Mat _mask, Rect _rect) {
    image = _img;
    mask = _mask;
    rectTu = _rect;
    k = vector<int>(image.rows*image.cols, -1);
    imageAccessor = ImageAccessor(image);
}
void GrabCut2D::returnExternalDataStructure(cv::InputOutputArray _mask) {
    mask.copyTo(_mask);
}
//Implementer

void GrabCut2D::initializeMaskGmm(Mat &fgModel, Mat &bgModel) {
    // InitializeMask
    cout<<"#GrabCut2D::initializeMaskGmm# Mask Zeroed At First"<<endl;
    mask = MASK_B;
    mask(rectTu) = MASK_PF;

    // Initialize FG GMM
    auto fgSamples = getFgSamples(rectTu);
    Mat fgLabels(rectTu.height*rectTu.width, 1, CV_32SC1);
    gmmFG.estimateParas(fgSamples, noArray(), fgLabels);
    gmmFG.constructModelFromEM(fgModel);


    // Initialize BG GMM
    auto bgSamples = getBgSamples(rectTu);
    Mat bgLabels(image.cols*image.rows - rectTu.height*rectTu.width, 1, CV_32SC1);
    gmmBG.estimateParas(bgSamples, noArray(), bgLabels);
    gmmBG.constructModelFromEM(bgModel);

    //Initialize k
    reassignPixelsInBGToComponents();

    //test
    testInitializeMaskGmm();
    cout<<"#GrabCut2D::initializeMaskGmm# End"<<endl;
}

void GrabCut2D::reassignPixelsInBGToComponents() {
    for (int row = 0; row < image.rows; ++row) {
        for (int col = 0; col < image.cols; ++col) {
            auto area = AreaMask(mask.at<MaskType>(row, col));
            if(area == MASK_B) {
                reAssignPixelToGmmComponent(point(row, col), MASK_B);
            }
        }
    }
}

void GrabCut2D::reAssignPixelsToGmmComponents() {
    for (int i = 0; i < imageAccessor.nPixels(); ++i) {
        auto cood = imageAccessor.coordOfPixel(i);
        auto area = AreaMask(mask.at<MaskType>(cood));
        reAssignPixelToGmmComponent(cood, area);
    }
}
void GrabCut2D::reassignPixelsInTuToGmmComponents() {
    cout<<"# GrabCut2D::reassignPixelsInTuToGmmComponents# Begin"<<endl;
    auto tu = TuMaskMat();
    for (int row = 0; row < tu.rows; ++row) {
        for (int col = 0; col < tu.cols; ++col) {
            auto area = AreaMask(tu.at<MaskType>(row, col));
            auto coodInImage = point(row + rectTu.y, col + rectTu.x);
            reAssignPixelToGmmComponent(coodInImage, area);
        }
    }
    testReAssignPixelsInTuToGmmComponents();
    cout<<"# GrabCut2D::reassignPixelsInTuToGmmComponents# End"<<endl;
}

int GrabCut2D::gmmComponentForPixel(Point2i pixel, AreaMask area) {
    auto intensity = imageAccessor.pixelValue_At(pixel);
    auto gmm = area==MASK_F? gmmFG: gmmBG;

    //auto v = area==MASK_F? pixelsInFgGaussComps: pixelsInBgGaussComps;

    auto Ds = map<int, double>();
    for (int i = 0; i < gmm.nComps(); ++i) {
        auto D = gmm.minusLogProbDensConstDeled_at_Comp_Sample(i, intensity);
        Ds.insert(pair<int, double>(i, D));
    }
    auto iterMaxD = max_element(Ds.begin(), Ds.end());

    return  (*iterMaxD).first;
}

void GrabCut2D::reAssignPixelToGmmComponent(Point2i pixel, AreaMask area) {
    //cout<<"#GrabCut2D::addFgPixelToFgGmmComponent# image depth? channel?"<<endl;
    auto intensity = imageAccessor.pixelValue_At(pixel);
    GMM gmm;
    if ((area==MASK_F) || (area==MASK_PF)) {
        gmm = gmmFG;
    }
    else {
        gmm = gmmBG;
    }
    //auto v = area==MASK_F? pixelsInFgGaussComps: pixelsInBgGaussComps;

    auto Ds = map<int, double>();
    for (int i = 0; i < gmm.nComps(); ++i) {
        auto D = gmm.minusLogProbDensConstDeled_at_Comp_Sample(i, intensity);
        Ds.insert(pair<int, double>(i, D));
    }
    auto iterMaxD = max_element(Ds.begin(), Ds.end());

    if(iterMaxD!=Ds.end()) {
        auto component = (*iterMaxD).first;
        k[imageAccessor.indexOfPixel(pixel)] = component;
        //v[component].push_back(pixel);
    }
}

void GrabCut2D::generatePixelsInAreaByGaussComps(vector<vector<Point2i>> &v, AreaMask area) {
    assert(v.size() == 5);
    for (int i = 0; i < k.size(); ++i) {
        auto pixelArea = mask.at<MaskType>(imageAccessor.coordOfPixel(i));
        auto pixelK = k[i];
        if(pixelArea == area) {
            v[pixelK].push_back(imageAccessor.coordOfPixel(i));
        }
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

// Image Data
Mat GrabCut2D::getFgSamples(const Rect &rect) {// Syn with Mask
    auto fg = image(rect).clone();
    fg = fg.reshape(1, rect.height * rect.width);
    return fg;
}

// Data Factory
void GrabCut2D::generateBGPixelsVector(vector<Point2i>& bgPixels) {
    addPixelsInAnAreaToVector(bgPixels, MASK_B);
    addPixelsInAnAreaToVector(bgPixels, MASK_PB);
}
void GrabCut2D::generateFGPixelsVector(vector<Point2i>& fgPixels) {
    addPixelsInAnAreaToVector(fgPixels, MASK_F);
    addPixelsInAnAreaToVector(fgPixels, MASK_PF);
}
void GrabCut2D::addPixelsInAnAreaToVector(vector<Point2i> &pixels, AreaMask area) {
    int n = 0;
    int fn = 0;
    for (int i = 0; i < mask.rows; ++i) {
        for (int j = 0; j < mask.cols; ++j) {
            n++;
            auto theMask = markOfEle(i, j);
            if(theMask == area) {
                pixels.push_back(point(i, j));
                fn++;
            }
        }
    }
    cout<<"#GrabCut2D::generateFGPixelsVector# "<<n<<" "<<fn<<endl;
}
// Data Factory
void GrabCut2D::addBgPixelToBgGmmComponent(Point2i pixel) {
    reAssignPixelToGmmComponent(pixel, MASK_B);
}
void GrabCut2D::addFgPixelToFgGmmComponent(Point2i pixel) {
    reAssignPixelToGmmComponent(pixel, MASK_F);
}

void GrabCut2D::learnGMMParams() {
    cout<<"#GrabCut2D::learnGMMParams# Begin"<<endl;
    auto pixelsInFgByGaussComps = vector<vector<Point2i>>{5,vector<Point2i>()};
    generatePixelsInAreaByGaussComps(pixelsInFgByGaussComps, MASK_F);
    generatePixelsInAreaByGaussComps(pixelsInFgByGaussComps, MASK_PF);
    gmmFG.estimateParas(pixelsInFgByGaussComps, image);
    //gmmFG.constructFGModel(model);
    //5 comps, each has pi, mean, cov
    auto pixelsInBgByGaussComps = vector<vector<Point2i>>{5,vector<Point2i>()};
    generatePixelsInAreaByGaussComps(pixelsInFgByGaussComps, MASK_B);
    generatePixelsInAreaByGaussComps(pixelsInFgByGaussComps, MASK_PB);
    gmmBG.estimateParas(pixelsInFgByGaussComps, image);
    cout<<"#GrabCut2D::learnGMMParams# End"<<endl;
}

GrabCut2D::~GrabCut2D(void)
{
}

void GrabCut2D::constructGraph() {
    K = computeK();
    graph = new ImageGraph(/*estimated # of nodes*/ 2500, /*estimated # of edges*/ 20000); cout<<"#GrabCut2D::constructGraph()# !Implement estimated nodes and edges"<<endl;
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
    /*
    int nodeID = 0;
    for (int i = 0; i < rectTu.height; ++i) {
        for (int j = 0; j < rectTu.width; ++j) {
            auto cood = point(i + rectTu.y, j + rectTu.x);
            auto index = imageAccessor.indexOfPixel(cood);
            graph->add_node();
            graph->add_tweights(nodeID++, edgeWeight_Pixel_Source(index), edgeWeight_Pixel_Terminal(index));
        }
    }

    for (int i = 0; i < rectTu.height; ++i) {
        for (int j = 0; j < rectTu.width; ++j) {
            auto pix = point(i, j);
            auto ns = neighbors(pix, rectTu.height, rectTu.width);
            for(auto neighbor:ns) {
                auto index = indexOfPixel(pix, rectTu.height, rectTu.width);
                auto neighborIndex = indexOfPixel(neighbor, rectTu.height, rectTu.width);
                if (neighborIndex > index) {
                    auto pixInImage = Point2i(pix.x + rectTu.x, pix.y + rectTu.y);
                    auto neighborInImage = Point2i(neighbor.x + rectTu.x, neighbor.y + rectTu.y);
                    auto pixInImageIndex = imageAccessor.indexOfPixel(pixInImage);
                    auto neighborInImageIndex = imageAccessor.indexOfPixel(neighborInImage);
                    graph->add_edge(index, neighborIndex, edgeWeight_p_neighborQ(pixInImageIndex, neighborInImageIndex), edgeWeight_p_neighborQ(neighborInImageIndex, pixInImageIndex));
                }
            }
        }
    }
     */
    cout<<"#GrabCut2D::constructGraph# End"<<endl;
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

double GrabCut2D::R(Point2i pixel, AreaMask mask) {
    auto color = imageAccessor.pixelValue_At(pixel);
    int k_gmmComp = gmmComponentForPixel(pixel, mask);
    return D(mask, k_gmmComp, color);
}

double GrabCut2D::lambda() {
    return 50;
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
    cout<<"#GrabCut2D::Beta# !Not Fully Implement"<<endl;
    return 0.1;
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

int GrabCut2D::k_GMMCompOfPixel(int i) {
    return k[i];
}

AreaMask GrabCut2D::markOfEle(int row, int col) {
    return markOfPixel(point(col, row));
}

int GrabCut2D::comp_of_pixel(int pixelIndex) {
    return k[pixelIndex];
}

void GrabCut2D::minCut() {
    cout<<"#GrabCut2D::minCut# Begin"<<endl;
    constructGraph();
    graph->maxflow();
    updateMask();
    testMinCut();
    cout<<"#GrabCut2D::minCut# End"<<endl;
}

void GrabCut2D::updateMask() {
    auto m = TuMaskMat();
    for (int i = 0; i < m.rows; ++i) {
        for (int j = 0; j < m.cols; ++j) {
            auto index = indexofPixel(point(i, j), Point2i(rectTu.x, rectTu.y), imageAccessor.rows(), imageAccessor.cols());
            if(graph->what_segment(index) == ImageGraph::SOURCE) {
                mask.at<MaskType>(imageAccessor.coordOfPixel(index)) = MASK_PF;//!Change to Mask Accessor!
            }
            else {
                mask.at<MaskType>(imageAccessor.coordOfPixel(index)) = MASK_PB;//!Change to Mask Accessor!
            }
        }
    }
}

void GrabCut2D::addBgPixelsToFgGmmComponents() {
    vector<Point2i> bgPixels;
    auto& bgPixelsRef = bgPixels; cout<<"#GrabCut2D::addBgPixelsToBgGmmComponents# bgPixelsRef not sure"<<endl;
    generateBGPixelsVector(bgPixelsRef);

    for(auto pixel:bgPixelsRef) {
        addBgPixelToBgGmmComponent(pixel);
    }
}
void GrabCut2D::addFgPixelsToFgGmmComponents() {
    vector<Point2i> fgPixels;
    auto& fgPixelsRef = fgPixels; cout<<"#GrabCut2D::addFgPixelsToFgGmmComponents# fgPixelsRef not sure"<<endl;
    generateFGPixelsVector(fgPixelsRef);

    for(auto pixel:fgPixelsRef) {
        addFgPixelToFgGmmComponent(pixel);
    }
}
// test
void GrabCut2D::testInitializeMaskGmm() {
    cout<<"#GrabCut2D::testInitializeMaskGmm()#"<<endl;
    cout<<"FG"<<endl;
    gmmFG.testEstimatedParas();
    cout<<"BG"<<endl;
    gmmBG.testEstimatedParas();
    testReAssignPixelsInBGToComponents();
    cout<<"#GrabCut2D::testInitializeMaskGmm()# End"<<endl;
}
void GrabCut2D::testReAssignPixelsInBGToComponents() {
    cout<<"#GrabCut2D::testReAssignPixelsInBGToComponents()#"<<endl;
    MyUtility::printVector(k);
    cout<<"#GrabCut2D::testReAssignPixelsInBGToComponents()# End"<<endl;
}

void GrabCut2D::testReAssignPixelsInTuToGmmComponents() {
    cout<<"#GrabCut2D::testReassignPixelsInTuToGmmComponents()#"<<endl;
    MyUtility::printVector(k);
    cout<<"#GrabCut2D::testReassignPixelsInTuToGmmComponents()# End"<<endl;
}

void GrabCut2D::testMinCut() {
    printMat(mask);
    //imwrite("/Users/zhouweiran/Desktop/mask.png", maskPic);
}

//
int GrabCut2D::nPixelsInRect() {
    return rectTu.width * rectTu.height;
}

