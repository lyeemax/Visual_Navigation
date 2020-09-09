
#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>

#include<ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include<opencv2/core/core.hpp>
#include <pcl/io/pcd_io.h>
#include "sensor_msgs/point_cloud_conversion.h"
#include "sensor_msgs/PointCloud2.h"
#include"../include/System.h"
#include <pcl_conversions/pcl_conversions.h>
#include <tf/transform_broadcaster.h>
#include "orbtypes.h"
#include <image_transport/image_transport.h>

using namespace std;

class ImageGrabber
{
public:
    ImageGrabber(ORB_SLAM2::System* pSLAM,ros::NodeHandle &nh):mpSLAM(pSLAM),node(nh),it(nh){
		pclPub=nh.advertise<sensor_msgs::PointCloud2>("/pointclouds",1);
		T=cv::Mat::eye(3,3,CV_32F);
		imgPub=it.advertise("tracking_image",1);
    }

    void GrabRGBD(const sensor_msgs::ImageConstPtr& msgRGB,const sensor_msgs::ImageConstPtr& msgD);

    ORB_SLAM2::System* mpSLAM;

    ros::NodeHandle node;
    ros::Publisher pclPub;
    tf::TransformBroadcaster odom_broadcaster;
    mutex poseMutex;
    cv::Mat T;
    mutex mapMutex;
    ORB_SLAM2::PointTCloudPtr map;
    mutex imgMutex;
    cv::Mat trackingImg;

    image_transport::ImageTransport it;
    image_transport::Publisher imgPub;

};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "RGBD");
    ros::start();

    string strVoc="/home/unicorn/catkin_ws/src/ORBSLAM2_with_navigation/Vocabulary/ORBvoc.txt";
    string config="/home/unicorn/catkin_ws/src/ORBSLAM2_with_navigation/astra.yaml";

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM2::System SLAM(strVoc,config,ORB_SLAM2::System::RGBD,true);
	ros::NodeHandle nh;
    ImageGrabber igb(&SLAM,nh);



    message_filters::Subscriber<sensor_msgs::Image> rgb_sub(nh, "/camera/rgb/image_raw", 1);
    message_filters::Subscriber<sensor_msgs::Image> depth_sub(nh, "/camera/depth/image_rect_raw", 1);
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> sync_pol;
    message_filters::Synchronizer<sync_pol> sync(sync_pol(10), rgb_sub,depth_sub);
    sync.registerCallback(boost::bind(&ImageGrabber::GrabRGBD,&igb,_1,_2));

    ros::Rate loop_rate(5);
    while(ros::ok()){

        {
            unique_lock<mutex> lock(igb.mapMutex);
            if(!(igb.map==nullptr)){
//            pcl::VoxelGrid<ORB_SLAM2::PointT> sor;
//            sor.setInputCloud(boost::make_shared<ORB_SLAM2::PointTCloud>(igb.map));
//            sor.setLeafSize(0.04, 0.04, 0.04);
//            sor.filter(igb.map);
                if(!igb.map->empty()){
                    sensor_msgs::PointCloud2 clouldMap;
                    pcl::toROSMsg(*igb.map, clouldMap);
                    clouldMap.header.stamp = ros::Time().now();
                    clouldMap.header.frame_id="map";
                    igb.pclPub.publish(clouldMap);
                }

            }
        }

        {
            unique_lock<mutex> lock(igb.imgMutex);
            if(!igb.trackingImg.empty()){
                sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", igb.trackingImg).toImageMsg();
                msg->header.frame_id="camera_link";
                msg->header.stamp=ros::Time().now();
                igb.imgPub.publish(msg);
            }
        }
        {
            unique_lock<mutex> lock(igb.poseMutex);
            if(!igb.T.empty()){
                cv::Mat Rcw=igb.T.rowRange(0,3).colRange(0,3);
                auto q=ORB_SLAM2::Converter::toQuaternion(Rcw.inv());
                geometry_msgs::Quaternion odom_quat;
                odom_quat.x=q[0];
                odom_quat.y=q[1];
                odom_quat.z=q[2];
                odom_quat.w=q[3];
                cv::Mat t=igb.T.col(2).rowRange(0,3);
                t=-Rcw*t;
                //first, we'll publish the transform over tf
                geometry_msgs::TransformStamped odom_trans;
                odom_trans.header.stamp = ros::Time::now();
                odom_trans.header.frame_id = "map";
                odom_trans.child_frame_id = "camera_link";

                odom_trans.transform.translation.x = t.at<float>(0,3);
                odom_trans.transform.translation.y = t.at<float>(1,3);
                odom_trans.transform.translation.z = t.at<float>(2,3);
                odom_trans.transform.rotation = odom_quat;

                //send the transform
                igb.odom_broadcaster.sendTransform(odom_trans);
            }
        }
        loop_rate.sleep();
        ros::spinOnce();
    }


    // Stop all threads
    SLAM.Shutdown();

    // Save camera trajectory
    //SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");

    ros::shutdown();

    return 0;
}

void ImageGrabber::GrabRGBD(const sensor_msgs::ImageConstPtr& msgRGB,const sensor_msgs::ImageConstPtr& msgD)
{
    // Copy the ros image message to cv::Mat.
    cv_bridge::CvImageConstPtr cv_ptrRGB;
    try
    {
        cv_ptrRGB = cv_bridge::toCvShare(msgRGB);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    cv_bridge::CvImageConstPtr cv_ptrD;
    try
    {
        cv_ptrD = cv_bridge::toCvShare(msgD);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }


    {
        unique_lock<mutex> lock(poseMutex);
        T=mpSLAM->TrackRGBD(cv_ptrRGB->image,cv_ptrD->image,cv_ptrRGB->header.stamp.toSec());
    }
    {
        unique_lock<mutex> lock(mapMutex);
        map=mpSLAM->GetPointCloudMap();
    }
    {
        unique_lock<mutex> lock(imgMutex);
        trackingImg=mpSLAM->mTrackingImg;
    }


}


