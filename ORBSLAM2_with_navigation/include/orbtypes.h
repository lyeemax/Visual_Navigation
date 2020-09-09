//
// Created by unicorn on 2020/7/28.
//

#ifndef ORBTYPES_H
#define ORBTYPES_H
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
namespace ORB_SLAM2
{
typedef pcl::PointXYZRGB PointT;

typedef pcl::PointCloud<PointT> PointTCloud;

typedef PointTCloud::Ptr PointTCloudPtr;
}
#endif //ORBTYPES_H
