

#ifndef MAP_H
#define MAP_H

#include "MapPoint.h"
#include "KeyFrame.h"
#include "Converter.h"
#include <set>

#include <mutex>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>

namespace ORB_SLAM2
{

class MapPoint;
class KeyFrame;

/**
 * @brief 地图
 * 
 */
class Map
{
public:
    /** @brief 构造函数 */
    Map();

    /**
     * @brief 向地图中添加关键帧
     * 
     * @param[in] pKF 关键帧
     */
    void AddKeyFrame(KeyFrame* pKF);
    /**
     * @brief 向地图中添加地图点
     * 
     * @param[in] pMP 地图点
     */
    void AddMapPoint(MapPoint* pMP);



    void  UpdateCloudMap();

    /**
     * @brief 从地图中擦除地图点
     * 
     * @param[in] pMP 地图点
     */
    void EraseMapPoint(MapPoint* pMP);
    /**
     * @brief 从地图中删除关键帧
     * @detials 实际上这个函数中目前仅仅是删除了在std::set中保存的地图点的指针,并且删除后
     * 之前的地图点所占用的内存其实并没有得到释放
     * @param[in] pKF 关键帧
     */
    void EraseKeyFrame(KeyFrame* pKF);
    /**
     * @brief 设置参考地图点
     * @detials 一般是指,设置当前帧中的参考地图点; 这些点将用于DrawMapPoints函数画图
     * 
     * @param[in] vpMPs 地图点们
     */
    void SetReferenceMapPoints(const std::vector<MapPoint*> &vpMPs);
    /**
     * @brief 这个函数好像没有被用到过
     * 
     */
    //REVIEW
    void InformNewBigChange();
    /**
     * @brief 获取最大改变;但是这个函数最终好像并没有被使用到
     * 
     * @return int 
     */
    int GetLastBigChangeIdx();

    /**
     * @brief 获取地图中的所有关键帧
     * 
     * @return std::vector<KeyFrame*> 获得的关键帧序列
     */
    std::vector<KeyFrame*> GetAllKeyFrames();
    /**
     * @brief 获取地图中的所有地图点
     * 
     * @return std::vector<MapPoint*> 获得的地图点序列
     */
    std::vector<MapPoint*> GetAllMapPoints();
    /**
     * @brief 获取地图中的所有参考地图点
     * 
     * @return std::vector<MapPoint*> 获得的参考地图点序列
     */
    std::vector<MapPoint*> GetReferenceMapPoints();

    /**
     * @brief 获得当前地图中的地图点个数
     * 
     * @return long unsigned int 个数
     */
    long unsigned int MapPointsInMap();
    /**
     * @brief 获取当前地图中的关键帧个数
     * 
     * @return long unsigned 关键帧个数
     */
    long unsigned  KeyFramesInMap();

    /**
     * @brief 获取关键帧的最大id
     * 
     * @return long unsigned int  id
     */
    long unsigned int GetMaxKFid();

    /** @brief 清空地图 */
    void clear();

    // 虽然是一个向量，但是实际上值保存了最初始的关键帧
    vector<KeyFrame*> mvpKeyFrameOrigins;

    ///当更新地图时的互斥量.回环检测中和局部BA后更新全局地图的时候会用到这个
    std::mutex mMutexMapUpdate;

    // This avoid that two points are created simultaneously in separate threads (id conflict)
    ///为了避免地图点id冲突设计的互斥量
    //? 如果插入的地图点id不同,但是由于误差等原因,产生的地图点非常相近,这个时候怎么处理?
    std::mutex mMutexPointCreation;

protected:
    
    std::set<MapPoint*> mspMapPoints; ///< MapPoints
    std::set<KeyFrame*> mspKeyFrames; ///< Keyframs

    ///参考地图点
    std::vector<MapPoint*> mvpReferenceMapPoints;

    ///当前地图中具有最大ID的关键帧
    long unsigned int mnMaxKFid;

    //NOTICE 这个在泡泡机器人注释的版本中也被去除了
    // 并且貌似在程序中并没有被使用过
    // Index related to a big change in the map (loop closure, global BA)
    int mnBigChangeIdx;

    ///类的成员函数在对类成员变量进行操作的时候,防止冲突的互斥量
    std::mutex mMutexMap;

    ////点云地图
public:
    PointTCloudPtr mPointCloudMap;
    std::mutex mMutexPcMap;


};

} //namespace ORB_SLAM

#endif // MAP_H