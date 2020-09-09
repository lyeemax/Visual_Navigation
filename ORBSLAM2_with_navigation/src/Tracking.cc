
//HACK 更新一下.h文件中的文档注释,主要是函数;以及当前文件中每个步骤的doxygen文档信息


#include "Tracking.h"

#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>

#include"ORBmatcher.h"
#include"FrameDrawer.h"
#include"Converter.h"
#include"Map.h"

#include"Optimizer.h"
#include"PnPsolver.h"

#include<iostream>
#include<cmath>
#include<mutex>


using namespace std;

namespace ORB_SLAM2
{

///构造函数
Tracking::Tracking(
    System *pSys,                       //系统实例
    ORBVocabulary* pVoc,                //BOW字典
    FrameDrawer *pFrameDrawer,          //帧绘制器
    Map *pMap,                          //地图句柄
    KeyFrameDatabase* pKFDB,            //关键帧产生的词袋数据库
    const string &strSettingPath,       //配置文件路径
    const int sensor,
    float viewangle):                  //传感器类型
        mState(NO_IMAGES_YET),                              //当前系统还没有准备好
        mSensor(sensor),                                
        mbOnlyTracking(false),                              //处于SLAM模式
        mbVO(false),                                        //当处于纯跟踪模式的时候，这个变量表示了当前跟踪状态的好坏
        mpORBVocabulary(pVoc),          
        mpKeyFrameDB(pKFDB),
        mpSystem(pSys), 
        mpViewer(NULL),                                     //注意可视化的查看器是可选的，因为ORB-SLAM2最后是被编译成为一个库，所以对方人拿过来用的时候也应该有权力说我不要可视化界面（何况可视化界面也要占用不少的CPU资源）
        mpFrameDrawer(pFrameDrawer),
        mpMap(pMap), 
        mnLastRelocFrameId(0),                               //恢复为0,没有进行这个过程的时候的默认值
        mviewAngle(viewangle)
{
    // Load camera parameters from settings file
    // Step 1 从配置文件中加载相机参数
    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];

    //     |fx  0   cx|
    // K = |0   fy  cy|
    //     |0   0   1 |
    //构造相机内参矩阵
    cv::Mat K = cv::Mat::eye(3,3,CV_32F);
    K.at<float>(0,0) = fx;
    K.at<float>(1,1) = fy;
    K.at<float>(0,2) = cx;
    K.at<float>(1,2) = cy;
    K.copyTo(mK);

    // 图像矫正系数
    // [k1 k2 p1 p2 k3]
    cv::Mat DistCoef(4,1,CV_32F);
    DistCoef.at<float>(0) = fSettings["Camera.k1"];
    DistCoef.at<float>(1) = fSettings["Camera.k2"];
    DistCoef.at<float>(2) = fSettings["Camera.p1"];
    DistCoef.at<float>(3) = fSettings["Camera.p2"];
    const float k3 = fSettings["Camera.k3"];
    //有些相机的畸变系数中会没有k3项
    if(k3!=0)
    {
        DistCoef.resize(5);
        DistCoef.at<float>(4) = k3;
    }
    DistCoef.copyTo(mDistCoef);

    // 双目摄像头baseline * fx 50
    mbf = fSettings["Camera.bf"];

    float fps = fSettings["Camera.fps"];
    if(fps==0)
        fps=30;

    // Max/Min Frames to insert keyframes and to check relocalisation
    mMinFrames = 0;
    mMaxFrames = fps;

    // 1:RGB 0:BGR
    int nRGB = fSettings["Camera.RGB"];
    mbRGB = nRGB;

    int nFeatures = fSettings["ORBextractor.nFeatures"];
    float fScaleFactor = fSettings["ORBextractor.scaleFactor"];
    int nLevels = fSettings["ORBextractor.nLevels"];
    int fIniThFAST = fSettings["ORBextractor.iniThFAST"];
    int fMinThFAST = fSettings["ORBextractor.minThFAST"];

    mpORBextractorLeft = new ORBextractor(
        nFeatures,
        fScaleFactor,
        nLevels,
        fIniThFAST,
        fMinThFAST);

    if(sensor==System::STEREO || sensor==System::RGBD)
    {
        // 判断一个3D点远/近的阈值 mbf * 35 / fx
        //ThDepth其实就是表示基线长度的多少倍
        mThDepth = fSettings["ThDepth"];
        //mThDepth=5.0;
        cout << endl << "Depth Threshold (Close/Far Points): " << mThDepth << endl;
    }

    if(sensor==System::RGBD)
    {
        // 深度相机disparity转化为depth时的因子
        mDepthMapFactor = fSettings["DepthMapFactor"];
        if(fabs(mDepthMapFactor)<1e-5)
            mDepthMapFactor=1;
        else
            mDepthMapFactor = 1.0f/mDepthMapFactor;
    }

}

//设置局部建图器
void Tracking::SetLocalMapper(LocalMapping *pLocalMapper)
{
    mpLocalMapper=pLocalMapper;
}

//设置回环检测器
void Tracking::SetLoopClosing(LoopClosing *pLoopClosing)
{
    mpLoopClosing=pLoopClosing;
}

//设置可视化查看器
void Tracking::SetViewer(Viewer *pViewer)
{
    mpViewer=pViewer;
}

// 输入左目RGB或RGBA图像和深度图
// 1、将图像转为mImGray和imDepth并初始化mCurrentFrame
// 2、进行tracking过程
// 输出世界坐标系到该帧相机坐标系的变换矩阵
cv::Mat Tracking::GrabImageRGBD(
    const cv::Mat &imRGB,           //彩色图像
    const cv::Mat &imD,             //深度图像
    const double &timestamp)        //时间戳
{
    mImGray = imRGB;
    mImRGB=imRGB.clone();
    mImDepth=imD.clone();

    // step 1：将RGB或RGBA图像转为灰度图像
    if(mImGray.channels()==3)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGB2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGR2GRAY);
    }
    else if(mImGray.channels()==4)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGBA2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGRA2GRAY);
    }

    // 步骤3：构造Frame
    mCurrentFrame = Frame(
        mImGray,                //灰度图像
        mImDepth,                //深度图像
        timestamp,              //时间戳
        mpORBextractorLeft,     //ORB特征提取器
        mpORBVocabulary,        //词典
        mK,                     //相机内参矩阵
        mDistCoef,              //相机的去畸变参数
        mbf,                    //相机基线*相机焦距
        mThDepth,mImRGB);              //内外点区分深度阈值

    // 步骤4：跟踪
    Track();
    mlHistroyFrame.push_back(mCurrentFrame);
    if(mlHistroyFrame.size()>(mMaxFrames/2)){
        mlHistroyFrame.pop_front();
    }
    //返回当前帧的位姿
    return mCurrentFrame.mTcw.clone();
}
/*
 * @brief Main tracking function. It is independent of the input sensor.
 *
 * Tracking 线程
 */

bool Tracking::TrackLocalMapOnce(){
    cout<<"start track by local frame"<<endl;
    if(mlHistroyFrame.empty()){
        return false;
    }
    vector<MapPoint*> LocalMapPoints;
    LocalMapPoints.clear();
    cout<<"histroy frame is "<<mlHistroyFrame.size()<<endl;
    for (list<Frame>::const_iterator itKF = mlHistroyFrame.begin(), itEndKF = mlHistroyFrame.end();itKF != itEndKF; itKF++) {
        Frame pF = *itKF;
        const vector<MapPoint *> vpMPs = pF.mvpMapPoints;
        for (vector<MapPoint *>::const_iterator itMP = vpMPs.begin(), itEndMP = vpMPs.end(); itMP != itEndMP; itMP++) {
            MapPoint *pMP = *itMP;
            if(!pMP)
                continue;
            //这个地方还是需要有IMU来辅助，不然成功率几乎为零
            if(!mLastFrame.isInFrustum(pMP,0.4))
                continue;
            // 表示它已经是当前帧的局部地图点了，可以防止重复添加局部地图点
            if(pMP->mnTrackReferenceForFrame == mCurrentFrame.mnId)
                continue;

            if(!pMP->isBad()) {
                LocalMapPoints.push_back(pMP);
                pMP->mnTrackReferenceForFrame = mCurrentFrame.mnId;
            }
        }
    }

    cout<<"histroy frame has points of : "<<LocalMapPoints.size()<<endl;
    ORBmatcher matcher(0.8);
    int th = 5;
    // 对视野范围内的MapPoints通过投影进行特征点匹配
    int n = matcher.SearchByProjection(mCurrentFrame, LocalMapPoints, th);
    int nmatchesMap = 0;
    if(n>20){
        mCurrentFrame.mvpMapPoints=LocalMapPoints;

        mCurrentFrame.SetPose(mLastFrame.mTcw);

        // step 4:通过优化3D-2D的重投影误差来获得位姿
        Optimizer::PoseOptimization(&mCurrentFrame);

        for(int i =0; i<mCurrentFrame.N; i++)
        {
            if(mCurrentFrame.mvpMapPoints[i])
            {
                //如果对应到的某个特征点是外点
                if(mCurrentFrame.mvbOutlier[i])
                {
                    //清除它在当前帧中存在过的痕迹
                    MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];

                    mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                    mCurrentFrame.mvbOutlier[i]=false;
                    pMP->mbTrackInView = false;
                    pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                }
                else if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                    nmatchesMap++;
            }
        }
    }

    if(nmatchesMap > 20) {
        cout<<"tracked by local frame+++++++++++++: "<<nmatchesMap<<endl;
        return true;
    }
    else {
        cout<<"tracked failed ------------------"<<endl;
        return false;
    }

}
void Tracking::Track()
{
    // track包含两部分：估计运动、跟踪局部地图
    
    // mState为tracking的状态，包括 SYSTME_NOT_READY, NO_IMAGE_YET, NOT_INITIALIZED, OK, LOST
    // 如果图像复位过、或者第一次运行，则为NO_IMAGE_YET状态
    if(mState==NO_IMAGES_YET)

    {
        mState = NOT_INITIALIZED;
    }

    // mLastProcessedState 存储了Tracking最新的状态，用于FrameDrawer中的绘制
    mLastProcessedState=mState;

    unique_lock<mutex> lock(mpMap->mMutexMapUpdate);

    // Step 1：初始化
    if(mState==NOT_INITIALIZED)
    {
        if(mSensor==System::STEREO || mSensor==System::RGBD)
            StereoInitialization();

        mpFrameDrawer->Update(this);

        //这个状态量在上面的初始化函数中被更新
        if(mState!=OK)
            return;
        else{
            mpLocalMapper->InsertPointCloud(mpReferenceKF,mImDepth,mImRGB);
        }
    }
    // Step 2：跟踪
    else
    {
        // System is initialized. Track Frame.
        bool bOK;

        if(!mbOnlyTracking)
        {
            //SLAM模式
            if(mState==OK)
            {
                // 更新Fuse函数和SearchAndFuse函数替换的MapPoints
                //由于追踪线程需要使用上一帧的信息,而局部建图线程则可能会对原有的地图点进行替换.在这里进行检查
                CheckReplacedInLastFrame();

                // 运动模型是空的或刚完成重定位
                if(mVelocity.empty() || mCurrentFrame.mnId<mnLastRelocFrameId+2)
                {
                    // 将上一帧的位姿作为当前帧的初始位姿
                    // 通过BoW的方式在参考帧中找当前帧特征点的匹配点
                    // 优化每个特征点都对应3D点重投影误差即可得到位姿
                    bOK = TrackReferenceKeyFrame();
                }
                else
                {
                    // 根据恒速模型设定当前帧的初始位姿
                    // 通过投影的方式在参考帧中找当前帧特征点的匹配点
                    // 优化每个特征点所对应3D点的投影误差即可得到位姿
                    bOK = TrackWithMotionModel();
                    if(!bOK)
                        bOK = TrackReferenceKeyFrame();

//                    if(!bOK){
//                        TrackLocalMapOnce();
//                    }
                }
            }
            else
            {
                //如果正常的初始化不成功,那么就只能重定位了
                // BOW搜索，PnP求解位姿
                bOK = Relocalization();
            }
        }

        // 将最新的关键帧作为reference frame
        //mpReferenceKF在哪里更新的 ??
        //在跟踪局部地图时更新的，选取mpReferenceKF的准则是选取和当前帧有最多匹配地图点的关键帧
        mCurrentFrame.mpReferenceKF = mpReferenceKF;

        // If we have an initial estimation of the camera pose and matching. Track the local map.
        // step 2.2：在帧间匹配得到初始的姿态后，现在对local map进行跟踪得到更多的匹配，并优化当前位姿
        if(!mbOnlyTracking)
        {
            if(bOK)
                bOK = TrackLocalMap();
        }

        if(bOK)
            mState = OK;
        else
            mState=LOST;

        // Update drawer 中的帧副本的信息
        mpFrameDrawer->Update(this);

        // If tracking were good, check if we insert a keyframe
        if(bOK)
        {
            // Update motion model
            if(!mLastFrame.mTcw.empty())
            {
                // step 2.3：更新恒速运动模型 TrackWithMotionModel 中的mVelocity
                cv::Mat LastTwc = cv::Mat::eye(4,4,CV_32F);
                mLastFrame.GetRotationInverse().copyTo(LastTwc.rowRange(0,3).colRange(0,3));
                mLastFrame.GetCameraCenter().copyTo(LastTwc.rowRange(0,3).col(3));
                mVelocity = mCurrentFrame.mTcw*LastTwc; // 其实就是 Tcl
            }
            else
                mVelocity = cv::Mat();


            // Clean VO matches
            // step 2.4：清除UpdateLastFrame中为当前帧临时添加的MapPoints
            for(int i=0; i<mCurrentFrame.N; i++)
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
                if(pMP)
                    // 排除 UpdateLastFrame 函数中为了跟踪增加的MapPoints
                    if(pMP->Observations()<1)
                    {
                        mCurrentFrame.mvbOutlier[i] = false;
                        mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                    }
            }

            // Delete temporal MapPoints
            // step 2.5：清除临时的MapPoints，这些MapPoints在 TrackWithMotionModel 的 UpdateLastFrame 函数里生成（仅双目和rgbd）
            // 步骤2.4中只是在当前帧中将这些MapPoints剔除，这里从MapPoints数据库中删除
            for(list<MapPoint*>::iterator lit = mlpTemporalPoints.begin(), lend =  mlpTemporalPoints.end(); lit!=lend; lit++)
            {
                MapPoint* pMP = *lit;
                delete pMP;
            }
            //不能够直接执行这个是因为其中存储的都是指针,之前的操作都是为了避免内存泄露
            mlpTemporalPoints.clear();

            // Check if we need to insert a new keyframe
            // step 2.6：检测并插入关键帧，对于双目会产生新的MapPoints
            // NOTICE 在关键帧的时候生成地图点
            if(NeedNewKeyFrame()){
                CreateNewKeyFrame();
                if(NeedNewCloud()){
                    mpLocalMapper->InsertPointCloud(mpReferenceKF,mImDepth,mImRGB);
                    //cout<<"cloud map ++"<<endl;
                }
            }

            // We allow points with high innovation (considererd outliers by the Huber Function)
            // pass to the new keyframe, so that bundle adjustment will finally decide
            // if they are outliers or not. We don't want next frame to estimate its position
            // with those points so we discard them in the frame.
            // step 2.7 删除那些在bundle adjustment中检测为outlier的3D map点
            for(int i=0; i<mCurrentFrame.N;i++)
            {
                //这里第一个条件还要执行判断是因为, 前面的操作中可能删除了其中的地图点
                if(mCurrentFrame.mvpMapPoints[i] && mCurrentFrame.mvbOutlier[i])
                    mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
            }
        }

        // Reset if the camera get lost soon after initialization
        // 跟踪失败，并且relocation也没有搞定，只能重新Reset
        if(mState==LOST)
        {
            //如果地图中的关键帧信息过少的话甚至直接重新进行初始化了
            if(mpMap->KeyFramesInMap()<=5)
            {
                cout << "Track lost soon after initialisation, reseting..." << endl;
                mpSystem->Reset();
                return;
            }
        }

        //确保已经设置了参考关键帧
        if(!mCurrentFrame.mpReferenceKF)
            mCurrentFrame.mpReferenceKF = mpReferenceKF;

        // 保存上一帧的数据,当前帧变上一帧
        mLastFrame = Frame(mCurrentFrame);
    }

    // Store frame pose information to retrieve the complete camera trajectory afterwards.
    // step 3：记录位姿信息，用于轨迹复现
    if(!mCurrentFrame.mTcw.empty())
    {
        // 计算相对姿态T_currentFrame_referenceKeyFrame
        //这里的关键帧存储的位姿,表示的也是从参考关键帧的相机坐标系到世界坐标系的变换
        cv::Mat Tcr = mCurrentFrame.mTcw*mCurrentFrame.mpReferenceKF->GetPoseInverse();
        //保存各种状态
        mlRelativeFramePoses.push_back(Tcr);
        mlpReferences.push_back(mpReferenceKF);
        mlFrameTimes.push_back(mCurrentFrame.mTimeStamp);
        mlbLost.push_back(mState==LOST);
    }
    else
    {
        // This can happen if tracking is lost
        // 如果跟踪失败，则相对位姿使用上一次值
        mlRelativeFramePoses.push_back(mlRelativeFramePoses.back());
        mlpReferences.push_back(mlpReferences.back());
        mlFrameTimes.push_back(mlFrameTimes.back());
        mlbLost.push_back(mState==LOST);
    }

}// Tracking 


void Tracking::StereoInitialization()
{
    if(mCurrentFrame.N>500)
    {
        // Set Frame pose to the origin
        mCurrentFrame.SetPose(cv::Mat::eye(4,4,CV_32F));

        // Create KeyFrame
        // step 2：将当前帧构造为初始关键帧
        // mCurrentFrame的数据类型为Frame
        // KeyFrame包含Frame、地图3D点、以及BoW
        // KeyFrame里有一个mpMap，Tracking里有一个mpMap，而KeyFrame里的mpMap都指向Tracking里的这个mpMap
        // KeyFrame里有一个mpKeyFrameDB，Tracking里有一个mpKeyFrameDB，而KeyFrame里的mpMap都指向Tracking里的这个mpKeyFrameDB
        // 提问: 为什么要指向Tracking中的相应的变量呢? -- 因为Tracking是主线程，是它创建和加载的这些模块
        KeyFrame* pKFini = new KeyFrame(mCurrentFrame,mpMap,mpKeyFrameDB);

        mpMap->AddKeyFrame(pKFini);

        // Create MapPoints and asscoiate to KeyFrame
        for(int i=0; i<mCurrentFrame.N;i++)
        {
            float z = mCurrentFrame.mvDepth[i];
            if(z>0)
            {
                cv::Mat x3D = mCurrentFrame.UnprojectStereo(i);
                MapPoint* pNewMP = new MapPoint(x3D,pKFini,mpMap);

                pNewMP->AddObservation(pKFini,i);

                pNewMP->ComputeDistinctiveDescriptors();

                pNewMP->UpdateNormalAndDepth();

                mpMap->AddMapPoint(pNewMP);

                pKFini->AddMapPoint(pNewMP,i);

                mCurrentFrame.mvpMapPoints[i]=pNewMP;
            }
        }

        cout << "New map created with " << mpMap->MapPointsInMap() << " points" << endl;

        // step 4：在局部地图中添加该初始关键帧
        mpLocalMapper->InsertKeyFrame(pKFini);

        //当前帧变上一帧
        mLastFrame = Frame(mCurrentFrame);
        mnLastKeyFrameId=mCurrentFrame.mnId;
        mpLastKeyFrame = pKFini;


        mvpLocalKeyFrames.push_back(pKFini);
        mvpLocalMapPoints=mpMap->GetAllMapPoints();
        mpReferenceKF = pKFini;
        mCurrentFrame.mpReferenceKF = pKFini;

        // 把当前（最新的）局部MapPoints作为ReferenceMapPoints
        // ReferenceMapPoints是DrawMapPoints函数画图的时候用的
        mpMap->SetReferenceMapPoints(mvpLocalMapPoints);
        mpMap->mvpKeyFrameOrigins.push_back(pKFini);

        //追踪成功
        mState=OK;
    }
}

/*
 * @brief 检查上一帧中的MapPoints是否被替换
 * 
 * Local Mapping线程可能会将关键帧中某些MapPoints进行替换，由于tracking中需要用到mLastFrame，这里检查并更新上一帧中被替换的MapPoints
 * @see LocalMapping::SearchInNeighbors()
 */
void Tracking::CheckReplacedInLastFrame()
{
    for(int i =0; i<mLastFrame.N; i++)
    {
        MapPoint* pMP = mLastFrame.mvpMapPoints[i];
        if(pMP)
        {
            MapPoint* pRep = pMP->GetReplaced();
            if(pRep)
            {
                mLastFrame.mvpMapPoints[i] = pRep;
            }
        }
    }
}

/*
 * @brief 对参考关键帧的MapPoints进行跟踪
 * 
 * 1. 计算当前帧的词包，将当前帧的特征点分到特定层的nodes上
 * 2. 对属于同一node的描述子进行匹配
 * 3. 根据匹配对估计当前帧的姿态
 * 4. 根据姿态剔除误匹配
 * @return 如果匹配数大于10，返回true
 */
bool Tracking::TrackReferenceKeyFrame()
{
    // Compute Bag of Words vector
    // step 1：将当前帧的描述子转化为BoW向量
    mCurrentFrame.ComputeBoW();

    // We perform first an ORB matching with the reference keyframe
    // If enough matches are found we setup a PnP solver
    ORBmatcher matcher(0.7,true);
    vector<MapPoint*> vpMapPointMatches;

    // step 2：通过特征点的BoW加快当前帧与参考帧之间的特征点匹配
    int nmatches = matcher.SearchByBoW(
        mpReferenceKF,          //参考关键帧
        mCurrentFrame,          //当前帧
        vpMapPointMatches);     //存储匹配关系

        //cout<<"TrackReferenceKeyFrame get matches: "<<nmatches<<endl;
    if(nmatches<15)
        return false;

    mCurrentFrame.mvpMapPoints = vpMapPointMatches;
    mCurrentFrame.SetPose(mLastFrame.mTcw); // 用上一次的Tcw设置初值，在PoseOptimization可以收敛快一些

    // step 4:通过优化3D-2D的重投影误差来获得位姿
    Optimizer::PoseOptimization(&mCurrentFrame);

    // Discard outliers
    int nmatchesMap = 0;
    for(int i =0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            //如果对应到的某个特征点是外点
            if(mCurrentFrame.mvbOutlier[i])
            {
                //清除它在当前帧中存在过的痕迹
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];

                mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                mCurrentFrame.mvbOutlier[i]=false;
                pMP->mbTrackInView = false;
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                nmatches--;
            }
            else if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                nmatchesMap++;
        }
    }

    return nmatchesMap>=10;
}

//用于恒速模型跟踪，把上一帧的所有距离较近的点拿出来反投影成为临时地图点，用于当前帧的跟踪
void Tracking::UpdateLastFrame()
{
    // Update pose according to reference keyframe
    // step 1：更新最近一帧的位姿
    KeyFrame* pRef = mLastFrame.mpReferenceKF;  //上一帧的参考KF
    // ref_keyframe 到 lastframe的位姿
    cv::Mat Tlr = mlRelativeFramePoses.back();
    mLastFrame.SetPose(Tlr*pRef->GetPose()); // Tlr*Trw = Tlw l:last r:reference w:world

    // 如果上一帧为关键帧，则退出
    if(mnLastKeyFrameId==mLastFrame.mnId )
        return;

    // step 2：对于双目或rgbd摄像头，为上一帧临时生成新的MapPoints
    // 注意这些MapPoints不加入到Map中，在tracking的最后会删除
    // 跟踪过程中需要将将上一帧的MapPoints投影到当前帧可以缩小匹配范围，加快当前帧与上一帧进行特征点匹配

    vector<pair<float,int> > vDepthIdx;
    vDepthIdx.reserve(mLastFrame.N);

    for(int i=0; i<mLastFrame.N;i++)
    {
        float z = mLastFrame.mvDepth[i];
        if(z>0)

        {
            vDepthIdx.push_back(make_pair(z,i));
        }
    }

    if(vDepthIdx.empty())
        return;

    // step 2.2：按照深度从小到大排序
    sort(vDepthIdx.begin(),vDepthIdx.end());

    // We insert all close points (depth<mThDepth)
    // If less than 100 close points, we insert the 100 closest ones.
    // step 2.3：将距离比较近的点包装成MapPoints
    int nPoints = 0;
    for(size_t j=0; j<vDepthIdx.size();j++)
    {
        int i = vDepthIdx[j].second;

        bool bCreateNew = false;

        //如果这个点对应在上一帧中的地图点没有,或者创建后就没有被观测到,那么就生成一个临时的地图点
        MapPoint* pMP = mLastFrame.mvpMapPoints[i];
        if(!pMP)
            bCreateNew = true;
        else if(pMP->Observations()<1)      //? 从地图点被创建后就没有观测到,意味这是在上一帧中新添加的地图点吗
        {
            bCreateNew = true;
        }
        if(bCreateNew)
        {
            cv::Mat x3D = mLastFrame.UnprojectStereo(i);
            MapPoint* pNewMP = new MapPoint(
                x3D,
                mpMap,
                &mLastFrame,
                i);

            //? 上一帧在处理结束的时候,没有进行添加的操作吗?
            //有，但是人家说了地图点为空！
            mLastFrame.mvpMapPoints[i]=pNewMP;

            // 标记为临时添加的MapPoint，之后在CreateNewKeyFrame之前会全部删除
            mlpTemporalPoints.push_back(pNewMP);
            nPoints++;
        }
        else//如果不需要创建新的 临时地图点
        {
            nPoints++;
        }

        //当当前的点的深度已经超过了远点的阈值,并且已经这样处理了超过100个点的时候,说明就足够了
        if(vDepthIdx[j].first>mThDepth && nPoints>100)
            break;
    }
}

/**
 * @brief 根据匀速度模型对上一帧的MapPoints进行跟踪
 * 
 * 1. 非单目情况，需要对上一帧产生一些新的MapPoints（临时） (因为传感器的原因，单目情况下仅仅凭借一帧没法生成可靠的地图点)
 * 2. 将上一帧的MapPoints投影到当前帧的图像平面上，在投影的位置进行区域匹配  NOTICE 加快了匹配的速度
 * 3. 根据匹配对估计当前帧的姿态
 * 4. 根据姿态剔除误匹配
 * @return 如果匹配数大于10，返回true
 * @see V-B Initial Pose Estimation From Previous Frame
 */
bool Tracking::TrackWithMotionModel()
{
    // 最小距离 < 0.9*次小距离 匹配成功
    ORBmatcher matcher(0.9,true);

    // Update last frame pose according to its reference keyframe
    // Create "visual odometry" points

    UpdateLastFrame();

    mCurrentFrame.SetPose(mVelocity*mLastFrame.mTcw);
    //清空当前帧的地图点
    fill(mCurrentFrame.mvpMapPoints.begin(),mCurrentFrame.mvpMapPoints.end(),static_cast<MapPoint*>(NULL));

    // Project points seen in previous frame
    //这个阈值和下面的的跟踪有关,表示了匹配过程中的搜索半径
    int th=7;

    // step 2：根据匀速度模型进行对上一帧的MapPoints进行跟踪, 根据上一帧特征点对应的3D点投影的位置缩小特征点匹配范围
    //我觉的这个才是使用恒速模型的根本目的
    int nmatches = matcher.SearchByProjection(mCurrentFrame,mLastFrame,th,mSensor==System::MONOCULAR);


    // If few matches, uses a wider window search
    // 如果跟踪的点少，则扩大搜索半径再来一次
    if(nmatches<20)
    {
        fill(mCurrentFrame.mvpMapPoints.begin(),mCurrentFrame.mvpMapPoints.end(),static_cast<MapPoint*>(NULL));
        nmatches = matcher.SearchByProjection(mCurrentFrame,mLastFrame,2*th,mSensor==System::MONOCULAR); // 2*th
    }

    //cout<<"TrackWithMotionModel get matches: "<<nmatches<<endl;
    //如果就算是这样还是不能够获得足够的跟踪点,那么就认为运动跟踪失败了.
    if(nmatches<20)
        return false;

    Optimizer::PoseOptimization(&mCurrentFrame);

    // Discard outliers
    // step 4：优化位姿后剔除outlier的mvpMapPoints,这个和前面相似
    int nmatchesMap = 0;
    for(int i =0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            if(mCurrentFrame.mvbOutlier[i])
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];

                mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                mCurrentFrame.mvbOutlier[i]=false;
                pMP->mbTrackInView = false;
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                nmatches--;
            }
            else if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                //累加成功匹配到的地图点数目
                nmatchesMap++;
        }
    }
    return nmatchesMap>=10;
}

/**
 * @brief 对Local Map的MapPoints进行跟踪
 * Step 1：更新局部关键帧 mvpLocalKeyFrames 和局部地图点 mvpLocalMapPoints
 * Step 2：在局部地图中查找与当前帧匹配的MapPoints, 其实也就是对局部地图点进行跟踪
 * Step 3：更新局部所有MapPoints后对位姿再次优化
 * Step 4：更新当前帧的MapPoints被观测程度，并统计跟踪局部地图的效果
 * Step 5：根据跟踪匹配数目及回环情况决定是否跟踪成功
 * @return true         跟踪成功
 * @return false        跟踪失败
 */
bool Tracking::TrackLocalMap()
{
    // We have an estimation of the camera pose and some map points tracked in the frame.
    // We retrieve the local map and try to find matches to points in the local map.

    // Update Local KeyFrames and Local Points
    // Step 1：更新局部关键帧 mvpLocalKeyFrames 和局部地图点 mvpLocalMapPoints
    UpdateLocalMap();

    // Step 2：在局部地图中查找与当前帧匹配的MapPoints, 为Frame中的特征点增加对应的MapPoint,其实也就是对局部地图点进行跟踪
    SearchLocalPoints();

    Optimizer::PoseOptimization(&mCurrentFrame);
    mnMatchesInliers = 0;

    // Update MapPoints Statistics
    // Step 4：更新当前帧的MapPoints被观测程度，并统计跟踪局部地图的效果
    for(int i=0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            if(!mCurrentFrame.mvbOutlier[i])
            {
                mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
                if(!mbOnlyTracking)
                {
                    if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                        mnMatchesInliers++;
                }
                else
                    mnMatchesInliers++;
            }

        }
    }

    // Decide if the tracking was succesful
    // More restrictive if there was a relocalization recently
    // Step 5：根据跟踪匹配数目及回环情况决定是否跟踪成功
    //如果最近刚刚发生了重定位,那么至少跟踪上了50个点我们才认为是跟踪上了
    if(mCurrentFrame.mnId<mnLastRelocFrameId+mMaxFrames && mnMatchesInliers<50)
        return false;

    if(mnMatchesInliers<30)
        return false;
    else
        return true;
}

/*
 * @brief 判断当前帧是否为关键帧
 * @return true if needed
 */

bool Tracking::NeedNewCloud(){
    cv::Mat TWCl=mLastFrame.GetPoseInverse();
    cv::Mat TWCc=mCurrentFrame.GetPoseInverse();
    cv::Mat TClCc=TWCl.inv()*TWCc;
    cv::Mat rot;
    cv::Mat RClCc=TClCc.colRange(0,3).rowRange(0,3);
    cv::Mat tClCc=TClCc.col(3).rowRange(0,3);
    cv::Rodrigues(RClCc,rot);
    float rotangle=180.0*sqrt(rot.dot(rot))/M_PI;
    if(rotangle>1.5){
        return true;
    }
    float md=mpReferenceKF->mbf;
    float d=tClCc.dot(tClCc);
    if(d>md){
        return true;
    }
    return false;
}
bool Tracking::NeedNewKeyFrame()
{
    // ?step 1：如果用户在界面上选择重定位，那么将不插入关键帧
    // 由于插入关键帧过程中会生成MapPoint，因此用户选择重定位后地图上的点云和关键帧都不会再增加
    if(mbOnlyTracking)
        return false;

    // If Local Mapping is freezed by a Loop Closure do not insert keyframes
    // 如果局部地图被闭环检测使用，则不插入关键帧
    if(mpLocalMapper->isStopped() || mpLocalMapper->stopRequested())
        return false;

    const int nKFs = mpMap->KeyFramesInMap();

    // Do not insert keyframes if not enough frames have passed from last relocalisation
    // step 2：判断是否距离上一次插入关键帧的时间太短
    // mCurrentFrame.mnId是当前帧的ID
    // mnLastRelocFrameId是最近一次重定位帧的ID
    // mMaxFrames等于图像输入的帧率
    // 如果关键帧比较少，则考虑插入关键帧
    // 或距离上一次重定位超过1s，则考虑插入关键帧
    if( mCurrentFrame.mnId < mnLastRelocFrameId + mMaxFrames &&     //距离上一次重定位不超过1s
        nKFs>mMaxFrames)                                            //地图中的关键帧已经足���
        return false;

    // Tracked MapPoints in the reference keyframe
    // step 3：得到参考关键帧跟踪到的MapPoints数量
    // NOTICE 在 UpdateLocalKeyFrames 函数中会将与当前关键帧共视程度最高的关键帧设定为当前帧的参考关键帧 -- 一般的参考关键帧的选择原则
    //地图点的最小观测次数
    int nMinObs = 3;
    if(nKFs<=2)
        nMinObs=2;
    //获取地图点的数目, which 参考帧观测的数目大于等于 nMinObs
    int nRefMatches = mpReferenceKF->TrackedMapPoints(nMinObs);

    // Local Mapping accept keyframes?
    // step 4：查询局部地图管理器是否繁忙,也就是当前能否接受新的关键帧
    bool bLocalMappingIdle = mpLocalMapper->AcceptKeyFrames();
    // "total matches = matches to map + visual odometry matches"
    // Visual odometry matches will become MapPoints if we insert a keyframe.
    // This ratio measures how many MapPoints we could create if we insert a keyframe.
    // step 5：对于双目或RGBD摄像头，统计 总的可以添加的MapPoints数量 和 跟踪到地图中的MapPoints数量
    int nMap = 0;       //现有地图中,可以被关键帧观测到的地图点数目
    int nTotal= 0;      //当前帧中可以添加到地图中的地图点数量
    if(mSensor!=System::MONOCULAR)// 双目或rgbd
    {
        for(int i =0; i<mCurrentFrame.N; i++)
        {
            //如果是近点,并且这个特征点的深度合法,就可以被添加到地图中
            if(mCurrentFrame.mvDepth[i]>0 && mCurrentFrame.mvDepth[i]<mThDepth/4.0)
            {
                nTotal++;// 总的可以添加mappoints数
                if(mCurrentFrame.mvpMapPoints[i])
                    if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                        nMap++;// 被关键帧观测到的mappoints数，即观测到地图中的MapPoints数量
            }
        }
    }
    else
    {
        // There are no visual odometry matches in the monocular case
        //? 提问:究竟什么才是 visual odometry matches ?
        nMap=1;
        nTotal=1;
    }

    //计算这个比例,当前帧中观测到的地图点数目和当前帧中总共的地图点数目之比.这个值越接近1越好,越接近0说明跟踪上的地图点太少,tracking is weak
    const float ratioMap = (float)nMap/(float)(std::max(1,nTotal));

    // step 6：决策是否需要插入关键帧
    // Thresholds
    // 设定inlier阈值，和之前帧特征点匹配的inlier比例
    float thRefRatio = 0.75f;
    if(nKFs<2)
        thRefRatio = 0.4f;// 关键帧只有一帧，那么插入关键帧的阈值设置很低 //? 这句话不应该放在下面这句话的后面吗?
    if(mSensor==System::MONOCULAR)
        thRefRatio = 0.9f;//单目情况下插入关键帧的阈值很高

    // MapPoints中和地图关联的比例阈值
    float thMapRatio = 0.35f;
    if(mnMatchesInliers>300)
        thMapRatio = 0.20f;

    // Condition 1a: More than "MaxFrames" have passed from last keyframe insertion
    // 很长时间没有插入关键帧
    const bool c1a = mCurrentFrame.mnId>=mnLastKeyFrameId+mMaxFrames;
    // Condition 1b: More than "MinFrames" have passed and Local Mapping is idle
    // localMapper处于空闲状态,才有生成关键帧的基本条件
    const bool c1b = (mCurrentFrame.mnId>=mnLastKeyFrameId+mMinFrames && bLocalMappingIdle);
    // Condition 1c: tracking is weak
    // 跟踪要跪的节奏，0.25和0.3是一个比较低的阈值
    const bool c1c =  mSensor!=System::MONOCULAR &&             //只有在双目的时候才成立
                      (mnMatchesInliers<nRefMatches*0.25 ||       //和地图点匹配的数目非常少
                       ratioMap<0.3f) ;                          //地图点跟踪成功的比例非常小,要挂了
    // Condition 2: Few tracked points compared to reference keyframe. Lots of visual odometry compared to map matches.
    // 阈值比c1c要高，与之前参考帧（最近的一个关键帧）重复度不是太高
    const bool c2 = ((mnMatchesInliers<nRefMatches*thRefRatio ||    // 总的来说,还是参考关键帧观测到的地图点的数目太少,少于给定的阈值
                      ratioMap<thMapRatio) &&                     // 追踪到的地图点的数目比例太少,少于阈值
                     mnMatchesInliers>15);                           //匹配到的内点太少

    if((c1a||c1b||c1c)&&c2)
    {
        // If the mapping accepts keyframes, insert keyframe.
        // Otherwise send a signal to interrupt BA
        if(bLocalMappingIdle)
        {
            //可以插入关键帧
            return true;
        }
        else
        {
            mpLocalMapper->InterruptBA();
            if(mSensor!=System::MONOCULAR)
            {
                // 队列里不能阻塞太多关键帧
                // tracking插入关键帧不是直接插入，而且先插入到mlNewKeyFrames中，
                // 然后localmapper再逐个pop出来插入到mspKeyFrames
                if(mpLocalMapper->KeyframesInQueue()<3)
                    //队列中的关键帧数目不是很多,可以插入
                    return true;
                else
                    //队列中缓冲的关键帧数目太多,暂时不能插入
                    return false;
            }
            else
                //对于单目情况,就直接无法插入关键帧了
                //? 为什么这里对单目情况的处理不一样?
                return false;
        }
    }
    else
        //不满足上面的条件,自然不能插入关键帧
        return false;
}

/**
 * @brief 创建新的关键帧
 *
 * 对于非单目的情况，同时创建新的MapPoints
 */
void Tracking::CreateNewKeyFrame()
{
    //如果不能保持局部建图器开启的状态,就无法顺利插入关键帧
    if(!mpLocalMapper->SetNotStop(true))
        return;

    // step 1：将当前帧构造成关键帧
    KeyFrame* pKF = new KeyFrame(mCurrentFrame,mpMap,mpKeyFrameDB);

    // step 2：将当前关键帧设置为当前帧的参考关键帧
    // 在UpdateLocalKeyFrames函数中会将与当前关键帧共视程度最高的关键帧设定为当前帧的参考关键帧
    mpReferenceKF = pKF;
    mCurrentFrame.mpReferenceKF = pKF;

    // 这段代码和 UpdateLastFrame 中的那一部分代码功能相同
    // step 3：对于双目或rgbd摄像头，为当前帧生成新的MapPoints
    if(mSensor!=System::MONOCULAR)
    {
        // 根据Tcw计算mRcw、mtcw和mRwc、mOw
        mCurrentFrame.UpdatePoseMatrices();

        // We sort points by the measured depth by the stereo/RGBD sensor.
        // We create all those MapPoints whose depth < mThDepth.
        // If there are less than 100 close points we create the 100 closest.
        // step 3.1：得到当前帧深度小于阈值的特征点
        // 创建新的MapPoint, depth < mThDepth
        //第一个元素是深度,第二个元素是对应的特征点的id
        vector<pair<float,int> > vDepthIdx;
        vDepthIdx.reserve(mCurrentFrame.N);
        for(int i=0; i<mCurrentFrame.N; i++)
        {
            float z = mCurrentFrame.mvDepth[i];
            if(z>0)
            {
                vDepthIdx.push_back(make_pair(z,i));
            }
        }

        if(!vDepthIdx.empty())
        {
            // step 3.2：按照深度从小到大排序
            sort(vDepthIdx.begin(),vDepthIdx.end());

            // step 3.3：将距离比较近的点包装成MapPoints
            //处理的近点的个数
            int nPoints = 0;
            for(size_t j=0; j<vDepthIdx.size();j++)
            {
                int i = vDepthIdx[j].second;

                bool bCreateNew = false;

                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
                //如果当前帧中无这个地图点
                if(!pMP)
                    bCreateNew = true;
                else if(pMP->Observations()<1)
                {
                    //或者是刚刚创立
                    bCreateNew = true;
                    mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
                }

                //如果需要新建地图点.这里是实打实的在全局地图中新建地图点
                if(bCreateNew)
                {
                    cv::Mat x3D = mCurrentFrame.UnprojectStereo(i);
                    MapPoint* pNewMP = new MapPoint(x3D,pKF,mpMap);
                    // 这些添加属性的操作是每次创建MapPoint后都要做的
                    pNewMP->AddObservation(pKF,i);
                    pKF->AddMapPoint(pNewMP,i);
                    pNewMP->ComputeDistinctiveDescriptors();
                    pNewMP->UpdateNormalAndDepth();
                    mpMap->AddMapPoint(pNewMP);

                    mCurrentFrame.mvpMapPoints[i]=pNewMP;
                    nPoints++;
                }
                else
                {
                    nPoints++;
                }

                // 这里决定了双目和rgbd摄像头时地图点云的稠密程度
                // 但是仅仅为了让地图稠密直接改这些不太好，
                // 因为这些MapPoints会参与之后整个slam过程
                //当当前处理的点大于深度阈值或者已经处理的点超过阈值的时候,就不再进行了
                if(vDepthIdx[j].first>mThDepth && nPoints>100)
                    break;
            }
        }
    }

    //执行插入关键帧的操作,其实也是在列表中等待
    mpLocalMapper->InsertKeyFrame(pKF);

    //然后现在允许局部建图器停止了
    mpLocalMapper->SetNotStop(false);

    //当前帧成为新的关键帧
    mnLastKeyFrameId = mCurrentFrame.mnId;
    mpLastKeyFrame = pKF;
}


 //把局部地图上的点投影到当前帧以建立匹配关系，也就是跟踪局部地图
void Tracking::SearchLocalPoints()
{
    // Do not search map points already matched
    // Step 1：遍历当前帧的mvpMapPoints，标记这些MapPoints不参与之后的搜索
    // 因为当前的mvpMapPoints一定在当前帧的视野中
    for(vector<MapPoint*>::iterator vit=mCurrentFrame.mvpMapPoints.begin(), vend=mCurrentFrame.mvpMapPoints.end(); vit!=vend; vit++)
    {
        MapPoint* pMP = *vit;
        if(pMP)
        {
            if(pMP->isBad())
            {
                *vit = static_cast<MapPoint*>(NULL);
            }
            else
            {
                pMP->IncreaseVisible();
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                // 标记该点将来不被投影，因为已经匹配过(指的是使用恒速运动模型进行投影) ???????????????
                pMP->mbTrackInView = false;
            }
        }
    }
    //准备进行投影匹配的点的数目
    int nToMatch=0;

    // Project points in frame and check its visibility
    // Step 2：将所有 局部MapPoints 投影到当前帧，判断是否在视野范围内
    for(vector<MapPoint*>::iterator vit=mvpLocalMapPoints.begin(), vend=mvpLocalMapPoints.end(); vit!=vend; vit++)
    {
        MapPoint* pMP = *vit;
        if(pMP->mnLastFrameSeen == mCurrentFrame.mnId)
            continue;
        if(pMP->isBad())
            continue;
        
        // Project (this fills MapPoint variables for matching)
        if(mCurrentFrame.isInFrustum(pMP,0.5))
        {
            pMP->IncreaseVisible();
            nToMatch++;
        }
    }

    if(nToMatch>0)
    {
        ORBmatcher matcher(0.8);
        int th = 1;
        if(mSensor==System::RGBD)
            th=3;

        // If the camera has been relocalised recently, perform a coarser search
        // 如果不久前进行过重定位，那么进行一个更加宽泛的搜索，阈值需要增大
        if(mCurrentFrame.mnId<mnLastRelocFrameId+2)
            th=5;

        // 对视野范围内的MapPoints通过投影进行特征点匹配
        matcher.SearchByProjection(mCurrentFrame,mvpLocalMapPoints,th);
    }
}

/**
 * @brief 更新局部地图 LocalMap
 *
 * 局部地图包括：共视关键帧、临近关键帧及其子父关键帧，由这些关键帧观测到的MapPoints
 */
void Tracking::UpdateLocalMap()
{
    // This is for visualization
    // 设置参考地图点用于绘图显示局部地图点（红色）
    mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

    // Update
    // 更新局部关键帧和局部MapPoints
    UpdateLocalKeyFrames();
    UpdateLocalPoints();
}

/**
 * @brief 更新局部地图点（来自局部关键帧）
 * 
 */
void Tracking::UpdateLocalPoints()
{
    // Step 1：清空局部MapPoints
    mvpLocalMapPoints.clear();

    // Step 2：遍历局部关键帧 mvpLocalKeyFrames
    for(vector<KeyFrame*>::const_iterator itKF=mvpLocalKeyFrames.begin(), itEndKF=mvpLocalKeyFrames.end(); itKF!=itEndKF; itKF++)
    {
        KeyFrame* pKF = *itKF;
        const vector<MapPoint*> vpMPs = pKF->GetMapPointMatches();

        // step 2：将局部关键帧的MapPoints添加到mvpLocalMapPoints
        for(vector<MapPoint*>::const_iterator itMP=vpMPs.begin(), itEndMP=vpMPs.end(); itMP!=itEndMP; itMP++)
        {
            MapPoint* pMP = *itMP;
            if(!pMP)
                continue;
            // 表示它已经是当前帧的局部地图点了，可以防止重复添加局部地图点
            if(pMP->mnTrackReferenceForFrame==mCurrentFrame.mnId)
                continue;
            if(!pMP->isBad())
            {
                mvpLocalMapPoints.push_back(pMP);
                pMP->mnTrackReferenceForFrame=mCurrentFrame.mnId;
            }
        }
    }
}

/**
 * @brief 更新局部关键帧
 * 方法是遍历当前帧的MapPoints，将观测到这些MapPoints的关键帧和相邻的关键帧及其父子关键帧，作为mvpLocalKeyFrames
 * Step 1：遍历当前帧的MapPoints，记录所有能观测到当前帧MapPoints的关键帧 
 * Step 2：更新局部关键帧（mvpLocalKeyFrames），添加局部关键帧有三个策略
 * Step 2.1 策略1：能观测到当前帧MapPoints的关键帧作为局部关键帧 （将邻居拉拢入伙）
 * Step 2.2 策略2：遍历策略1得到的局部关键帧里共视程度很高的关键帧，将他们的家人和邻居作为局部关键帧
 * Step 3：更新当前帧的参考关键帧，与自己共视程度最高的关键帧作为参考关键帧
 */
void Tracking:: UpdateLocalKeyFrames()
{
    // Each map point vote for the keyframes in which it has been observed
    // Step 1：遍历当前帧的MapPoints，记录所有能观测到当前帧MapPoints的关键帧以及观测次数
    map<KeyFrame*,int> keyframeCounter;
    for(int i=0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
            if(!pMP->isBad())
            {
                // 观测到该MapPoint的KF和该MapPoint在KF中的索引
                const map<KeyFrame*,size_t> observations = pMP->GetObservations();
                //这里由于一个地图点可以被多个关键帧观测到,因此对于每一次观测,都获得观测到这个地图点的关键帧,并且对关键帧进行投票
                for(map<KeyFrame*,size_t>::const_iterator it=observations.begin(), itend=observations.end(); it!=itend; it++)
                    keyframeCounter[it->first]++;
            }
            else
            {
                mCurrentFrame.mvpMapPoints[i]=NULL;
            }
        }
    }

    if(keyframeCounter.empty())
        return;

    int max=0;
    KeyFrame* pKFmax= static_cast<KeyFrame*>(NULL);

    // Step 2：更新局部关键帧（mvpLocalKeyFrames），添加局部关键帧有三个策略
    // 先清空局部关键帧
    mvpLocalKeyFrames.clear();
    //? 提问:为什么要乘3呢? 
    mvpLocalKeyFrames.reserve(3*keyframeCounter.size());

    // All keyframes that observe a map point are included in the local map. Also check which keyframe shares most points
    // V-D K1: shares the map points with current frame
    // Step 2.1 策略1：能观测到当前帧MapPoints的关键帧作为局部关键帧 （将邻居拉拢入伙）
    for(map<KeyFrame*,int>::const_iterator it=keyframeCounter.begin(), itEnd=keyframeCounter.end(); it!=itEnd; it++)
    {
        KeyFrame* pKF = it->first;
        if(pKF->isBad())
            continue;

        if(it->second>max)
        {
            max=it->second;
            pKFmax=pKF;
        }

        mvpLocalKeyFrames.push_back(it->first);
        
        // 用该关键帧的成员变量mnTrackReferenceForFrame 记录当前帧的id
        // 表示它已经是当前帧的局部关键帧了，可以防止重复添加局部关键帧
        pKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
    }


    // Include also some not-already-included keyframes that are neighbors to already-included keyframes
    // V-D K2: neighbors to K1 in the covisibility graph
    // Step 2.2 策略2：遍历策略1得到的局部关键帧里共视程度很高的关键帧，将他们的家人和邻居作为局部关键帧
    for(vector<KeyFrame*>::const_iterator itKF=mvpLocalKeyFrames.begin(), itEndKF=mvpLocalKeyFrames.end(); itKF!=itEndKF; itKF++)
    {
        // Limit the number of keyframes
        if(mvpLocalKeyFrames.size()>80)
            break;

        KeyFrame* pKF = *itKF;

        // 策略2.1:最佳共视的10帧; 如果共视帧不足10帧,那么就返回所有具有共视关系的关键帧（将邻居的邻居拉拢入伙）
        const vector<KeyFrame*> vNeighs = pKF->GetBestCovisibilityKeyFrames(10);
        for(vector<KeyFrame*>::const_iterator itNeighKF=vNeighs.begin(), itEndNeighKF=vNeighs.end(); itNeighKF!=itEndNeighKF; itNeighKF++)
        {
            KeyFrame* pNeighKF = *itNeighKF;
            if(!pNeighKF->isBad())
            {
                // mnTrackReferenceForFrame防止重复添加局部关键帧
                if(pNeighKF->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
                {
                    mvpLocalKeyFrames.push_back(pNeighKF);
                    pNeighKF->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                    break;
                }
            }
        }

        // 策略2.2:将自己的子关键帧作为局部关键帧（将邻居的子孙们拉拢入伙）
        const set<KeyFrame*> spChilds = pKF->GetChilds();
        for(set<KeyFrame*>::const_iterator sit=spChilds.begin(), send=spChilds.end(); sit!=send; sit++)
        {
            KeyFrame* pChildKF = *sit;
            if(!pChildKF->isBad())
            {
                if(pChildKF->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
                {
                    mvpLocalKeyFrames.push_back(pChildKF);
                    pChildKF->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                    break;
                }
            }
        }

        // 策略2.3:自己的父关键帧（将邻居的父母们拉拢入伙）
        KeyFrame* pParent = pKF->GetParent();
        if(pParent)
        {
            // mnTrackReferenceForFrame防止重复添加局部关键帧
            if(pParent->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
            {
                mvpLocalKeyFrames.push_back(pParent);
                pParent->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                break;
            }
        }

    }

    // V-D Kref： shares the most map points with current frame
    // Step 3：更新当前帧的参考关键帧，与自己共视程度最高的关键帧作为参考关键帧
    if(pKFmax)
    {
        mpReferenceKF = pKFmax;
        mCurrentFrame.mpReferenceKF = mpReferenceKF;
    }
}

/**
 * @details 重定位过程
 * 
 * Step 1：计算当前帧特征点的Bow映射
 * 
 * Step 2：找到与当前帧相似的候选关键帧
 * 
 * Step 3：通过BoW进行匹配
 * 
 * Step 4：通过EPnP算法估计姿态
 * 
 * Step 5：通过PoseOptimization对姿态进行优化求解
 * 
 * Step 6：如果内点较少，则通过投影的方式对之前未匹配的点进行匹配，再进行优化求解
 */
bool Tracking::Relocalization()
{
    // Compute Bag of Words Vector
    // Step 1： 计算当前帧特征点的Bow映射
    mCurrentFrame.ComputeBoW();

    // Relocalization is performed when tracking is lost
    // Track Lost: Query KeyFrame Database for keyframe candidates for relocalisation
    // Step 2：找到与当前帧相似的候选关键帧
    vector<KeyFrame*> vpCandidateKFs = mpKeyFrameDB->DetectRelocalizationCandidates(&mCurrentFrame);
    
    //如果没有候选关键帧，则退出
    if(vpCandidateKFs.empty())
        return false;

    const int nKFs = vpCandidateKFs.size();

    // We perform first an ORB matching with each candidate
    // If enough matches are found we setup a PnP solver
    ORBmatcher matcher(0.75,true);
    //每个关键帧的解算器
    vector<PnPsolver*> vpPnPsolvers;
    vpPnPsolvers.resize(nKFs);

    //每个关键帧和当前帧中特征点的匹配关系
    vector<vector<MapPoint*> > vvpMapPointMatches;
    vvpMapPointMatches.resize(nKFs);
    
    //放弃某个关键帧的标记
    vector<bool> vbDiscarded;
    vbDiscarded.resize(nKFs);

    //有效的候选关键帧数目
    int nCandidates=0;

    //遍历所有的候选关键帧
    for(int i=0; i<nKFs; i++)
    {
        KeyFrame* pKF = vpCandidateKFs[i];
        if(pKF->isBad())
            //? 前面的查找候选关键帧的时候,为什么不会检查一下这个呢? 为什么非要返回bad的关键帧呢? 关键帧为bad意味着什么呢? 
            //? 此外这个变量的初始值也不一定全部都是false吧
            vbDiscarded[i] = true;
        else
        {
            // Step 3：通过BoW进行匹配
            int nmatches = matcher.SearchByBoW(pKF,mCurrentFrame,vvpMapPointMatches[i]);
            //如果和当前帧的匹配数小于15,那么只能放弃这个关键帧
            if(nmatches<15)
            {
                vbDiscarded[i] = true;
                continue;
            }
            else
            {
                // 初始化PnPsolver
                PnPsolver* pSolver = new PnPsolver(mCurrentFrame,vvpMapPointMatches[i]);
                pSolver->SetRansacParameters(
                    0.99,   //用于计算RANSAC迭代次数理论值的概率
                    10,     //最小内点数, NOTICE 但是要注意在程序中实际上是min(给定最小内点数,最小集,内点数理论值),不一定使用这个
                    300,    //最大迭代次数
                    4,      //最小集(求解这个问题在一次采样中所需要采样的最少的点的个数,对于Sim3是3,EPnP是4),参与到最小内点数的确定过程中
                    0.5,    //这个是表示(最小内点数/样本总数);实际上的RANSAC正常退出的时候所需要的最小内点数其实是根据这个量来计算得到的
                    5.991); // 目测是自由度为2的卡方检验的阈值,作为内外点判定时的距离的baseline(程序中还会根据特征点所在的图层对这个阈值进行缩放的)
                vpPnPsolvers[i] = pSolver;
                nCandidates++;
            }
        }
    }//遍历所有的候选关键帧,确定出满足进一步要求的候选关键帧并且为其创建pnp优化器

    // Alternatively perform some iterations of P4P RANSAC
    // Until we found a camera pose supported by enough inliers
    //? 这里的 P4P RANSAC 是啥意思啊   @lishuwei0424:我认为是Epnp，每次迭代需要4个点
    //是否已经找到相匹配的关键帧的标志
    bool bMatch = false;
    ORBmatcher matcher2(0.9,true);

    //通过一系列骚操作,直到找到能够进行重定位的匹配上的关键帧
    while(nCandidates>0 && !bMatch)
    {
        //遍历当前所有的候选关键帧
        for(int i=0; i<nKFs; i++)
        {
            //如果刚才已经放弃了,那么这里也放弃了
            if(vbDiscarded[i])
                continue;
    
            // Perform 5 Ransac Iterations
            //内点标记
            vector<bool> vbInliers;     
            
            //内点数
            int nInliers;
            
            // 表示RANSAC已经没有更多的迭代次数可用 -- 也就是说数据不够好，RANSAC也已经尽力了。。。
            bool bNoMore;

            // Step 4：通过EPnP算法估计姿态
            PnPsolver* pSolver = vpPnPsolvers[i];
            cv::Mat Tcw = pSolver->iterate(5,bNoMore,vbInliers,nInliers);

            // If Ransac reachs max. iterations discard keyframe
            // 如果这里的迭代已经尽力了。。。
            if(bNoMore)
            {
                vbDiscarded[i]=true;
                nCandidates--;
            }

            // If a Camera Pose is computed, optimize
            if(!Tcw.empty())
            {
                Tcw.copyTo(mCurrentFrame.mTcw);
                
                //成功被再次找到的地图点的集合,其实就是经过RANSAC之后的内点
                set<MapPoint*> sFound;

                const int np = vbInliers.size();
                //遍历所有内点
                for(int j=0; j<np; j++)
                {
                    if(vbInliers[j])
                    {
                        mCurrentFrame.mvpMapPoints[j]=vvpMapPointMatches[i][j];
                        sFound.insert(vvpMapPointMatches[i][j]);
                    }
                    else
                        mCurrentFrame.mvpMapPoints[j]=NULL;
                }

                // Step 5：通过PoseOptimization对姿态进行优化求解
                //只优化位姿,不优化地图点的坐标;返回的是内点的数量
                int nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                // ? 如果优化之后的内点数目不多,注意这里是直接跳过了本次循环,但是却没有放弃当前的这个关键帧
                if(nGood<10)
                    continue;

                //删除外点对应的地图点
                for(int io =0; io<mCurrentFrame.N; io++)
                    if(mCurrentFrame.mvbOutlier[io])
                        mCurrentFrame.mvpMapPoints[io]=static_cast<MapPoint*>(NULL);

                // If few inliers, search by projection in a coarse window and optimize again
                // Step 6：如果内点较少，则通过投影的方式对之前未匹配的点进行匹配，再进行优化求解
                // 前面的匹配关系是用词袋匹配过程得到的
                if(nGood<50)
                {
                    int nadditional =matcher2.SearchByProjection(
                        mCurrentFrame,          //当前帧
                        vpCandidateKFs[i],      //关键帧
                        sFound,                 //已经找到的地图点集合
                        10,                     //窗口阈值
                        100);                   //ORB描述子距离

                    //如果通过投影过程获得了比较多的特征点
                    if(nadditional+nGood>=50)
                    {
                        //根据投影匹配的结果，采用3D-2D pnp非线性优化求解
                        nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                        // If many inliers but still not enough, search by projection again in a narrower window
                        // the camera has been already optimized with many points
                        //如果这样依赖内点数还是比较少的话,就使用更小的窗口搜索投影点;由于相机位姿已经使用了更多的点进行了优化,所以可以认为使用更小的窗口搜索能够取得意料之内的效果
                        //我觉得可以理解为 垂死挣扎 2333
                        if(nGood>30 && nGood<50)
                        {
                            //重新进行搜索
                            sFound.clear();
                            for(int ip =0; ip<mCurrentFrame.N; ip++)
                                if(mCurrentFrame.mvpMapPoints[ip])
                                    sFound.insert(mCurrentFrame.mvpMapPoints[ip]);
                            nadditional =matcher2.SearchByProjection(
                                mCurrentFrame,          //当前帧
                                vpCandidateKFs[i],      //候选的关键帧
                                sFound,                 //已经找到的地图点
                                3,                      //新的窗口阈值
                                64);                    //ORB距离? 

                            // Final optimization
                            if(nGood+nadditional>=50)
                            {
                                nGood = Optimizer::PoseOptimization(&mCurrentFrame);
                                //更新地图点
                                for(int io =0; io<mCurrentFrame.N; io++)
                                    if(mCurrentFrame.mvbOutlier[io])
                                        mCurrentFrame.mvpMapPoints[io]=NULL;
                            }
                            //如果还是不能够满足就放弃了
                        }//如果地图点还是比较少的话
                    }//如果通过投影过程获得了比较多的特征点
                }//如果内点较少,那么尝试通过投影关系再次进行匹配

                // If the pose is supported by enough inliers stop ransacs and continue
                //如果对于当前的关键帧已经有足够的内点(50个)了,那么就认为当前的这个关键帧已经和当前帧匹配上了
                if(nGood>=50)
                {
                    bMatch = true;
                    break;
                }
            }//遍历所有的候选关键帧
            // ? 大哥，这里PnPSolver 可不能够保证一定能够得到相机位姿啊？怎么办？
        }//一直运行,知道已经没有足够的关键帧,或者是已经有成功匹配上的关键帧
    }

    //折腾了这么久还是没有匹配上
    if(!bMatch)
    {
        return false;
    }
    else
    {
        //如果匹配上了,说明当前帧重定位成功了(也就是在上面的优化过程中,当前帧已经拿到了属于自己的位姿).因此记录当前帧的id
        mnLastRelocFrameId = mCurrentFrame.mnId;
        return true;
    }
}//重定位

//整个追踪线程执行复位操作
void Tracking::Reset()
{
    //基本上是挨个请求各个线程终止

    cout << "System Reseting" << endl;

    // Reset Local Mapping
    cout << "Reseting Local Mapper...";
    mpLocalMapper->RequestReset();
    cout << " done" << endl;

    // Reset Loop Closing
    cout << "Reseting Loop Closing...";
    mpLoopClosing->RequestReset();
    cout << " done" << endl;

    // Clear BoW Database
    cout << "Reseting Database...";
    mpKeyFrameDB->clear();
    cout << " done" << endl;

    // Clear Map (this erase MapPoints and KeyFrames)
    mpMap->clear();

    //然后复位各种变量
    KeyFrame::nNextId = 0;
    Frame::nNextId = 0;
    mState = NO_IMAGES_YET;

    mlRelativeFramePoses.clear();
    mlpReferences.clear();
    mlFrameTimes.clear();
    mlbLost.clear();
}

//? 目测是根据配置文件中的参数重新改变已经设置在系统中的参数,但是当前文件中没有找到对它的调用
void Tracking::ChangeCalibration(const string &strSettingPath)
{
    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];

    cv::Mat K = cv::Mat::eye(3,3,CV_32F);
    K.at<float>(0,0) = fx;
    K.at<float>(1,1) = fy;
    K.at<float>(0,2) = cx;
    K.at<float>(1,2) = cy;
    K.copyTo(mK);

    cv::Mat DistCoef(4,1,CV_32F);
    DistCoef.at<float>(0) = fSettings["Camera.k1"];
    DistCoef.at<float>(1) = fSettings["Camera.k2"];
    DistCoef.at<float>(2) = fSettings["Camera.p1"];
    DistCoef.at<float>(3) = fSettings["Camera.p2"];
    const float k3 = fSettings["Camera.k3"];
    if(k3!=0)
    {
        DistCoef.resize(5);
        DistCoef.at<float>(4) = k3;
    }
    DistCoef.copyTo(mDistCoef);

    mbf = fSettings["Camera.bf"];

    //做标记,表示在初始化帧的时候将会是第一个帧,要对它进行一些特殊的初始化操作
    Frame::mbInitialComputations = true;
}

void Tracking::InformOnlyTracking(const bool &flag)
{
    mbOnlyTracking = flag;
}

} //namespace ORB_SLAM
