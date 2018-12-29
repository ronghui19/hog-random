#include <fstream>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/ml/ml.hpp>
#include <string>
#include <time.h>
#include <stdlib.h>

using namespace std;
using namespace cv;

#define PosSamNO 1000    //正样本个数
#define NegSamNO 1000    //负样本个数

int main()
{
    std::string ImgName;

    //正样本图片的文件列表
    std::ifstream finPos("positive_samples.txt");
    //负样本图片的文件列表
    std::ifstream finNeg("negative_samples.txt");

    //所有训练样本的特征向量组成的矩阵，行数等于所有样本的个数，列数等于HOG描述子维数
    cv::Mat sampleFeatureMat;
    //训练样本的类别向量，行数等于所有样本的个数，列数等于1；1表示有目标，-1表示无目标
    cv::Mat sampleLabelMat;

    //创建HOG
    //检测窗口(64,64),块尺寸(8,8),块步长(8,8),cell尺寸(8,8),直方图bin个数9
    cv::HOGDescriptor detector(cv::Size(64, 64), cv::Size(8, 8), cv::Size(4, 4), cv::Size(4, 4), 9);
    int DescriptorDim;

    //依次读取正样本图片，生成HOG描述子
    for (int num = 0; num < PosSamNO && getline(finPos, ImgName); num++)
    {
        std::cout << "Processing：" << ImgName << std::endl;
        cv::Mat image = cv::imread(ImgName);
        std::vector<float> descriptors; //HOG描述子的维数，由图片大小、检测窗口大小、块大小、细胞单元中直方图bin个数决定

        detector.compute(image, descriptors, cv::Size(8, 8));

        if ( 0 == num )
        {
            //HOG描述子的维数
            DescriptorDim = descriptors.size();

            sampleFeatureMat = cv::Mat::zeros(PosSamNO + NegSamNO, DescriptorDim, CV_32FC1);
            sampleLabelMat = cv::Mat::zeros(PosSamNO + NegSamNO, 1, CV_32SC1);
        }

        for (int i = 0; i < DescriptorDim; i++)
        {
            //将计算好的HOG描述子复制到样本特征矩阵sampleFeatureMat
            sampleFeatureMat.at<float>(num, i) = descriptors[i];
        }
        //正样本类别为1，有目标
        sampleLabelMat.at<float>(num, 0) = 1;

    }

    for (int num = 0; num < NegSamNO && getline(finNeg, ImgName); num++ )
    {

        cv::Mat src = cv::imread(ImgName);
        std::vector<float> descriptors; //HOG描述子的维数，由图片大小、检测窗口大小、块大小、细胞单元中直方图bin个数决定

        DescriptorDim = descriptors.size();
        detector.compute(src, descriptors, cv::Size(8, 8));

        for (int i = 0; i < DescriptorDim; i++)
        {
            //将计算好的HOG描述子复制到样本特征矩阵sampleFeatureMat
            sampleFeatureMat.at<float>(num + PosSamNO, i) = descriptors[i];
        }
        //负样本类别为0，wu目标
        sampleLabelMat.at<float>(num + PosSamNO, 0) = 0;

    }

    sampleFeatureMat.convertTo(sampleFeatureMat, CV_32F); //uchar型转换为cv_32f
    int samplesNum = sampleFeatureMat.rows;

//    std::cout << "samplesNum :" <<samplesNum << std::endl;

    int trainNum = 1400;
    cv::Mat trainData, trainLabels;

    trainData = sampleFeatureMat(cv::Range(0, trainNum), cv::Range::all());   //前800个样本为训练数据
    trainLabels = sampleLabelMat(cv::Range(0, trainNum), cv::Range::all());

    int nMinSampleCount = 10;
    int nMaxCategories = 10;

    //训练数据
    cv::Ptr<cv::ml::TrainData> cvTrainData = cv::ml::TrainData::create(trainData, cv::ml::SampleTypes::ROW_SAMPLE, trainLabels);
    // 创建分类器
    cv::Ptr<cv::ml::RTrees> m_RTreesModel = cv::ml::RTrees::create();

    //节点最小样本数量
    m_RTreesModel->setMinSampleCount(nMinSampleCount); //optional
    //是否建立替代分裂点
    m_RTreesModel->setUseSurrogates(false); //optional
    //最大聚类簇数
    m_RTreesModel->setMaxCategories(nMaxCategories); //optional
    //终止标准
    m_RTreesModel->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER + (0.01f > 0 ? cv::TermCriteria::EPS : 0), 100, 0.01f)); //optional
    //训练模型
    m_RTreesModel->train(cvTrainData);

    m_RTreesModel->save("aaa.xml");

    //预测分类
    double train_hr = 0, test_hr = 0;
    // compute prediction error on train and test data
    for (int i = 0; i < samplesNum; i++)
    {
        cv::Mat sample = sampleFeatureMat.row(i);
        float r = m_RTreesModel->predict(sample);   //对所有行进行预测
                                            //预测结果与原结果相比，相等为1，不等为0
        r = std::abs(r - sampleLabelMat.at<int>(i)) <= FLT_EPSILON ? 1.f : 0.f;

        if (i < trainNum)
            train_hr += r;  //累积正确数
        else
            test_hr += r;
    }
    std::cout << "train_hr :" << train_hr << std::endl << "test_hr :" << test_hr << std::endl;
    test_hr /= samplesNum - trainNum;
    train_hr = trainNum > 0 ? train_hr / trainNum : 1.;

    printf("accuracy: train = %.1f%%, test = %.1f%%\n",
        train_hr*100., test_hr*100.);

    return 0;
}

