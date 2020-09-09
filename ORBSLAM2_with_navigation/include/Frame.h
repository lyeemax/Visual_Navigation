
#ifndef FRAME_H
#define FRAME_H

#include<vector>

#include "MapPoint.h"
#include "Thirdparty/DBoW2/DBoW2/BowVector.h"
#include "Thirdparty/DBoW2/DBoW2/FeatureVector.h"
#include "ORBVocabulary.h"
#include "KeyFrame.h"
#include "ORBextractor.h"
#include "orbtypes.h"

#include <opencv2/opencv.hpp>

namespace ORB_SLAM2
{
#define FRAME_GRID_ROWS 48

#define FRAME_GRID_COLS 64

class MapPoint;
class KeyFrame;

class Frame
{
public:

    Frame();

    Frame(const Frame &frame);

    Frame(const cv::Mat &imGray, const cv::Mat &imDepth, const double &timeStamp, ORBextractor* extractor,ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth,cv::Mat imRGB);

    void ExtractORB(int flag, const cv::Mat &im);

    void ComputeBoW();

    // Set the camera pose.
    // 用Tcw更新mTcw
    /**
     * @brief 用 Tcw 更新 mTcw 以及类中存储的一系列位姿
     * 
     * @param[in] Tcw 从世界坐标系到当前帧相机位姿的变换矩阵
     */
    void SetPose(cv::Mat Tcw);

    // Computes rotation, translation and camera center matrices from the camera pose.
    /**
     * @brief 根据相机位姿,计算相机的旋转,平移和相机中心等矩阵.
     * @details 其实就是根据Tcw计算mRcw、mtcw和mRwc、mOw.
     */
    void UpdatePoseMatrices();

    // Returns the camera center.
    /**
     * @brief 返回位于当前帧位姿时,相机的中心
     * 
     * @return cv::Mat 相机中心在世界坐标系下的3D点坐标
     */
    inline cv::Mat GetCameraCenter()
	{
        return mOw.clone();
    }

    // Returns inverse of rotation
    //NOTICE 默认的mRwc存储的是当前帧时，相机从当前的坐标系变换到世界坐标系所进行的旋转，而我们常谈的旋转则说的是从世界坐标系到当前相机坐标系的旋转
    inline cv::Mat GetRotationInverse()
	{
		//所以直接返回其实就是我们常谈的旋转的逆了
        return mRwc.clone();
    }

    // Check if a MapPoint is in the frustum of the camera
    // and fill variables of the MapPoint to be used by the tracking
    bool isInFrustum(MapPoint* pMP, float viewingCosLimit);

    bool PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY);


    vector<size_t> GetFeaturesInArea(const float &x, const float  &y, const float  &r, const int minLevel=-1, const int maxLevel=-1) const;


    void ComputeStereoFromRGBD(const cv::Mat &imDepth);

    cv::Mat UnprojectStereo(const int &i);
    cv::Mat GetPoseInverse(){
        cv::Mat Twc=mTcw.inv();
        return Twc.clone();
    }

public:

    // Vocabulary used for relocalization.
    ///用于重定位的ORB特征字典
    ORBVocabulary* mpORBvocabulary;

    ORBextractor* mpORBextractorLeft;

    // Frame timestamp.
    double mTimeStamp;

    cv::Mat mK;
	//NOTICE 注意这里的相机内参数其实都是类的静态成员变量；此外相机的内参数矩阵和矫正参数矩阵却是普通的成员变量，
	//NOTE 这样是否有些浪费内存空间？

    
    static float fx;        ///<x轴方向焦距
    static float fy;        ///<y轴方向焦距
    static float cx;        ///<x轴方向光心偏移
    static float cy;        ///<y轴方向光心偏移
    static float invfx;     ///<x轴方向焦距的逆
    static float invfy;     ///<x轴方向焦距的逆

	//TODO 目测是opencv提供的图像去畸变参数矩阵的，但是其具体组成未知
    ///去畸变参数
    cv::Mat mDistCoef;

    // Stereo baseline multiplied by fx.
    ///baseline x fx
    float mbf;

    // Stereo baseline in meters.
    ///相机的基线长度,单位为米
    float mb;

    /** @} */

    // Threshold close/far points. Close points are inserted from 1 view.
    // Far points are inserted as in the monocular case from 2 views.
	//TODO 这里它所说的话还不是很理解。尤其是后面的一句。
    //而且,这个阈值不应该是在哪个帧中都一样吗?
    ///判断远点和近点的深度阈值
    float mThDepth;

    // Number of KeyPoints.
    int N; 

    /**
     * @name 关于特征点
     * @{ 
     */

    // Vector of keypoints (original for visualization) and undistorted (actually used by the system).
    // In the stereo case, mvKeysUn is redundant as images must be rectified.
    // In the RGB-D case, RGB images can be distorted.
    // mvKeys:原始左图像提取出的特征点（未校正）
    // mvKeysRight:原始右图像提取出的特征点（未校正）
    // mvKeysUn:校正mvKeys后的特征点，对于双目摄像头，一般得到的图像都是校正好的，再校正一次有点多余
    
    ///原始左图像提取出的特征点（未校正）
    std::vector<cv::KeyPoint> mvKeys;
    ///原始右图像提取出的特征点（未校正）
    std::vector<cv::KeyPoint> mvKeysRight;
	///校正mvKeys后的特征点
    std::vector<cv::KeyPoint> mvKeysUn;

    

    ///@note 之所以对于双目摄像头只保存左图像矫正后的特征点,是因为对于双目摄像头,一般得到的图像都是矫正好的,这里再矫正一次有些多余.\n
    ///校正操作是在帧的构造函数中进行的。
    
    // Corresponding stereo coordinate and depth for each keypoint.
    // "Monocular" keypoints have a negative value.
    // 对于双目，mvuRight存储了左目像素点在右目中的对应点的横坐标 （因为纵坐标是一样的）
    // mvDepth对应的深度
    // 单目摄像头，这两个容器中存的都是-1

    ///@note 对于单目摄像头，这两个容器中存的都是-1
    ///对于双目相机,存储左目像素点在右目中的对应点的横坐标 （因为纵坐标是一样的）

    std::vector<float> mvuRight;	//m-member v-vector u-指代横坐标,因为最后这个坐标是通过各种拟合方法逼近出来的，所以使用float存储
    ///对应的深度
    std::vector<float> mvDepth;
    
    // Bag of Words Vector structures.
    ///和词袋模型有关的向量
    DBoW2::BowVector mBowVec;
    ///和词袋模型中特征有关的向量
    DBoW2::FeatureVector mFeatVec;
    ///@todo 这两个向量目前的具体含义还不是很清楚

    // ORB descriptor, each row associated to a keypoint.
    /// 左目摄像头和右目摄像头特征点对应的描述子
    cv::Mat mDescriptors, mDescriptorsRight;

    // MapPoints associated to keypoints, NULL pointer if no association.
    /// 每个特征点对应的MapPoint.如果特征点没有对应的地图点,那么将存储一个空指针
    std::vector<MapPoint*> mvpMapPoints;

    // Flag to identify outlier associations.
    // 观测不到Map中的3D点
    /// 属于外点的特征点标记,在 Optimizer::PoseOptimization 使用了
    std::vector<bool> mvbOutlier;

    // Keypoints are assigned to cells in a grid to reduce matching complexity when projecting MapPoints.
	//原来通过对图像分区域还能够降低重投影地图点时候的匹配复杂度啊。。。。。
    ///@note 注意到上面也是类的静态成员变量， 有一个专用的标志mbInitialComputations用来在帧的构造函数中标记这些静态成员变量是否需要被赋值
    /// 坐标乘以mfGridElementWidthInv和mfGridElementHeightInv就可以确定在哪个格子
    static float mfGridElementWidthInv;
    /// 坐标乘以mfGridElementWidthInv和mfGridElementHeightInv就可以确定在哪个格子
    static float mfGridElementHeightInv;
    

    // 每个格子分配的特征点数，将图像分成格子，保证提取的特征点比较均匀
    // FRAME_GRID_ROWS 48
    // FRAME_GRID_COLS 64
	///这个向量中存储的是每个图像网格内特征点的id（左图）
    std::vector<std::size_t> mGrid[FRAME_GRID_COLS][FRAME_GRID_ROWS];

    /** @} */

    // Camera pose.
    cv::Mat mTcw; ///< 相机姿态 世界坐标系到相机坐标坐标系的变换矩阵,是我们常规理解中的相机位姿

    // Current and Next Frame id.
    // 类的静态成员变量，这些变量则是在整个系统开始执行的时候被初始化的——它在全局区被初始化
    static long unsigned int nNextId; ///< Next Frame id.
    long unsigned int mnId; ///< Current Frame id.

    // Reference Keyframe.
    KeyFrame* mpReferenceKF;//<指针，指向参考关键帧

    /**
     * @name 图像金字塔信息
     * @{
     */
    // Scale pyramid info.
    int mnScaleLevels;                  ///<图像金字塔的层数
    float mfScaleFactor;                ///<图像金字塔的尺度因子
    float mfLogScaleFactor;             ///<图像金字塔的尺度因子的对数值，用于仿照特征点尺度预测地图点的尺度
                                  
    vector<float> mvScaleFactors;		///<图像金字塔每一层的缩放因子
    vector<float> mvInvScaleFactors;	///<以及上面的这个变量的倒数
    vector<float> mvLevelSigma2;		///@todo 目前在frame.c中没有用到，无法下定论
    vector<float> mvInvLevelSigma2;		///<上面变量的倒数

    /** @} */

    // Undistorted Image Bounds (computed once).
    /**
     * @name 用于确定画格子时的边界 
     * @note（未校正图像的边界，只需要计算一次，因为是类的静态成员变量）
     * @{
     */
    static float mnMinX;
    static float mnMaxX;
    static float mnMinY;
    static float mnMaxY;

    /** @} */

    /**
     * @brief 一个标志，标记是否已经进行了这些初始化计算
     * @note 由于第一帧以及SLAM系统进行重新校正后的第一帧会有一些特殊的初始化处理操作，所以这里设置了这个变量. \n
     * 如果这个标志被置位，说明再下一帧的帧构造函数中要进行这个“特殊的初始化操作”，如果没有被置位则不用。
    */ 
    static bool mbInitialComputations;


private:

    // Undistort keypoints given OpenCV distortion parameters.
    // Only for the RGB-D case. Stereo must be already rectified!
    // (called in the constructor).
	/**
     * @brief 用内参对特征点去畸变，结果报存在mvKeysUn中
     * 
     */
    void UndistortKeyPoints();

    /**
     * @brief 计算去畸变图像的边界
     * 
     * @param[in] imLeft            需要计算边界的图像
     */
    void ComputeImageBounds(const cv::Mat &imLeft);

    // Assign keypoints to the grid for speed up feature matching (called in the constructor).
    /**
     * @brief 将提取到的特征点分配到图像网格中 \n
     * @details 该函数由构造函数调用
     * 
     */
    void AssignFeaturesToGrid();

    /**
     * @name 和相机位姿有关的变量
     * @{
     */
    // Rotation, translation and camera center
    cv::Mat mRcw; ///< Rotation from world to camera
    cv::Mat mtcw; ///< Translation from world to camera
    cv::Mat mRwc; ///< Rotation from camera to world
    cv::Mat mOw;  ///< mtwc,Translation from camera to world

    /** @} */


};

}// namespace ORB_SLAM

#endif // FRAME_H
