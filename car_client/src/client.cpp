#include "ros/ros.h"
#include"ros_arduino_msgs/Raw_imu.h"
#include"ros_arduino_msgs/Wheel_velocity.h"
#include <sensor_msgs/Imu.h>
#include "std_msgs/String.h"
#include "../include/StatelessOrientation.h"
#include <nav_msgs/Odometry.h>
#include <tf/transform_broadcaster.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include "../include/imu_filter.h"

using namespace std;

typedef ros_arduino_msgs::Wheel_velocity  VelMsg;
typedef ros_arduino_msgs::Raw_imu    ImuMsg;
typedef message_filters::sync_policies::ApproximateTime<ImuMsg, VelMsg> SyncPolicy;
typedef message_filters::Synchronizer<SyncPolicy> Synchronizer;
typedef message_filters::Subscriber<ImuMsg> ImuSubscriber;
typedef message_filters::Subscriber<VelMsg> VelSubscriber;

class CarClient{
private:
    ros::NodeHandlePtr nh;
    sensor_msgs::Imu imu;
    geometry_msgs::Pose pose;
    double gbias[3] , abias[3] ,gnoise[3],anoise[3], linear_vx_scale,linear_vy_scale , angular_v_scale;
    ros::Publisher imu_publisher_;
    ros::Publisher odom_pub;
    ros::Time last_time;
    boost::mutex mutex_;
    bool initialized_;
    ImuFilter filter_;
    ros::Timer check_topics_timer_;
    tf2_ros::TransformBroadcaster tf_broadcaster_;

    boost::shared_ptr<ImuSubscriber> imu_subscriber_;
    boost::shared_ptr<VelSubscriber> vel_subscriber_;
    boost::shared_ptr<Synchronizer> sync_;
public:

    void publishTransform(sensor_msgs::Imu imu_msg_raw)
    {
        auto qua=filter_.getOrientation();
        qua.normalize();
        geometry_msgs::TransformStamped transform;
        transform.header.stamp = imu_msg_raw.header.stamp;

        transform.header.frame_id="odom_link";
        transform.child_frame_id="imu_link";
        transform.transform.rotation.w=qua.w();
        transform.transform.rotation.x=-qua.x();
        transform.transform.rotation.y=-qua.y();
        transform.transform.rotation.z=-qua.z();
        tf_broadcaster_.sendTransform(transform);

    }
    void fusionCallback(const ImuMsg::ConstPtr &imu_msg_raw, const VelMsg::ConstPtr & vel_msg) {
        boost::mutex::scoped_lock lock(mutex_);
        geometry_msgs::Vector3 ang_vel = imu_msg_raw->raw_angular_velocity;
        geometry_msgs::Vector3 lin_acc = imu_msg_raw->raw_linear_acceleration;
        geometry_msgs::Vector3 mag_fld = imu_msg_raw->raw_magnetic_field;
        ros::Time time = imu_msg_raw->header.stamp;
        geometry_msgs::Vector3 mag_compensated;
        mag_compensated.x = mag_fld.x;// - mag_bias_.x;
        mag_compensated.y = mag_fld.y;// - mag_bias_.y;
        mag_compensated.z = mag_fld.z;// - mag_bias_.z;
        lin_acc.x-=abias[0]+anoise[0];
        lin_acc.y-=abias[1]+anoise[1];
        lin_acc.z-=abias[2]+anoise[2];
        ang_vel.x-=gbias[0]+gnoise[0];
        ang_vel.y-=gbias[1]+gnoise[1];
        ang_vel.z-=gbias[2]+gnoise[2];
        if (!initialized_ )
        {
            geometry_msgs::Quaternion init_q;
            if (!StatelessOrientation::computeOrientation(lin_acc, mag_compensated, init_q))
            {
                ROS_WARN_THROTTLE(5.0, "The IMU seems to be in free fall or close to magnetic north pole, cannot determine gravity direction!");
                return;
            }
            filter_.setOrientation(Eigen::Quaterniond(init_q.w, init_q.x, init_q.y, init_q.z));//why set bias to zero???????
        }

        if (!initialized_)
        {
            ROS_INFO("First pair of IMU and magnetometer messages received.");
            check_topics_timer_.stop();
            // initialize time
            last_time = time;
            initialized_ = true;
        }
        float dt= (time - last_time).toSec();


        filter_.madgwickAHRSupdate(
                    Eigen::Vector3d(ang_vel.x, ang_vel.y, ang_vel.z),
                    Eigen::Vector3d(lin_acc.x, lin_acc.y, lin_acc.z),
                    Eigen::Vector3d(mag_compensated.x, mag_compensated.y, mag_compensated.z),
                    dt);

        sensor_msgs::Imu imu_msg;
        imu_msg.header.frame_id="imu_link";
        imu_msg.header.stamp=ros::Time::now();
        double q0,q1,q2,q3;
        filter_.getOrientation(q0,q1,q2,q3);
        imu_msg.orientation.x=-q1;
        imu_msg.orientation.y=q2;
        imu_msg.orientation.z=q3;
        imu_msg.orientation.w=q0;
        imu_msg.linear_acceleration=imu_msg_raw->raw_linear_acceleration;
        imu_msg.angular_velocity=imu_msg_raw->raw_angular_velocity;
        imu_publisher_.publish(imu_msg);
        publishTransform(imu_msg);

        nav_msgs::Odometry odom_msg;
        odom_msg.header.frame_id="odom_link";
        odom_msg.header.stamp=ros::Time::now();
        odom_msg.child_frame_id ="base_link";
        odom_msg.twist.twist.angular.z = vel_msg->vel.z*angular_v_scale;
        odom_msg.twist.twist.linear.x = vel_msg->vel.x*linear_vx_scale;
        odom_msg.twist.twist.linear.y = vel_msg->vel.y*linear_vy_scale;
        double roll,pitch,theta;
        tf2::Matrix3x3(tf2::Quaternion(q1, q2, q3, q0)).getRPY(roll,pitch,theta);
        pose.position.x+=(odom_msg.twist.twist.linear.x*cos(theta)-odom_msg.twist.twist.linear.y*sin(theta))*dt;
        pose.position.y+=(odom_msg.twist.twist.linear.x*sin(theta)+odom_msg.twist.twist.linear.y*cos(theta))*dt;
        odom_msg.pose.pose.position=pose.position;
        odom_msg.pose.pose.orientation=tf::createQuaternionMsgFromYaw(theta);
        odom_pub.publish(odom_msg);
        last_time = time;

    };
    CarClient(ros::NodeHandlePtr n){
        memset(gbias , 0.0 , sizeof(gbias));
        memset(abias , 0.0 , sizeof(abias));
        memset(anoise , 0.0 , sizeof(anoise));
        memset(gnoise , 0.0 , sizeof(gnoise));
        linear_vx_scale=linear_vy_scale = 1.0;
        angular_v_scale = 1.0;

        nh=n;
//        nh->getParam("accXBias" , abias[0]);
//        nh->getParam("accYBias" , abias[1]);
//        nh->getParam("accZBias" , abias[2]);
//        nh->getParam("gryoXBias" , gbias[0]);
//        nh->getParam("gryoYBias" , gbias[1]);
//        nh->getParam("gryoZBias" , gbias[2]);
        abias[0]=1.7365845830709956e-03;abias[1]=8.8043384056282281e-04;abias[2]=2.8583760823456174e-03;
        gbias[0]=5.0689899187058845e-05;gbias[1]=5.8887520889936484e-05;gbias[2]=5.3703431283131154e-05;
        //anoise[0]=3.6886655529540242e-02;anoise[1]=2.9455740260214226e-02;anoise[2]=4.0839629253773969e-02;
        //gnoise[0]=5.0689899187058845e-05;gnoise[1]=8.1623063708193932e-04;gnoise[2]=8.1746285316654308e-04;
        nh->getParam("linear_x_velocity_scale" , linear_vx_scale);
        nh->getParam("linear_y_velocity_scale" , linear_vy_scale);
        nh->getParam("angular_velocity_scale" , angular_v_scale);

        last_time=ros::Time::now();
        odom_pub = nh->advertise<nav_msgs::Odometry>("/odom" , 100);
        imu_publisher_=nh->advertise<sensor_msgs::Imu>("imu/data",100);
        imu_subscriber_.reset(new ImuSubscriber(
                *nh, ros::names::resolve("raw_imu"), 10));

        vel_subscriber_.reset(new VelSubscriber(
                    *nh, ros::names::resolve("/wheel_vel"), 10));
        sync_.reset(new Synchronizer(
                    SyncPolicy(10), *imu_subscriber_, *vel_subscriber_));
        sync_->registerCallback(boost::bind(&CarClient::fusionCallback, this, _1, _2));
        }

};

int main(int argc , char **argv) {
    ros::init(argc,argv,"CarClient");
    ros::NodeHandlePtr nh=boost::shared_ptr<ros::NodeHandle>(new ros::NodeHandle);
    CarClient car(nh);
    ros::spin();
}
