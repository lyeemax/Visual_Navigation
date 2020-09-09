
#include "geometry_msgs/Twist.h"
#include "ros/ros.h"
#include "sensor_msgs/Joy.h"
#include "teleop_twist_joy/teleop_twist_joy.h"
#include"std_msgs/Bool.h"

#include <map>
#include <string>


namespace teleop_twist_joy {

/**
 * Internal members of class. This is the pimpl idiom, and allows more flexibility in adding
 * parameters later without breaking ABI compatibility, for robots which link TeleopTwistJoy
 * directly into base nodes.
 */
    struct TeleopTwistJoy::Impl {
        void joyCallback(const sensor_msgs::Joy::ConstPtr &joy);

        void sendCmdVelMsg(const sensor_msgs::Joy::ConstPtr &joy_msg);

        ros::Subscriber joy_sub;
        ros::Publisher cmd_vel_pub;
	ros::Publisher light_pub;
	bool lastState=false;
	bool changeState;
	int times=1;
	
    };

/**
 * Constructs TeleopTwistJoy.
 * \param nh NodeHandle to use for setting up the publisher and subscriber.
 * \param nh_param NodeHandle to use for searching for configuration parameters.
 */
    TeleopTwistJoy::TeleopTwistJoy(ros::NodeHandle *nh, ros::NodeHandle *nh_param) {
        pimpl_ = new Impl;

        pimpl_->cmd_vel_pub = nh->advertise<geometry_msgs::Twist>("cmd_vel", 1, true);
	pimpl_->light_pub=nh->advertise<std_msgs::Bool>("light",1,true);
        pimpl_->joy_sub = nh->subscribe<sensor_msgs::Joy>("joy", 1, &TeleopTwistJoy::Impl::joyCallback, pimpl_);
    }

    void TeleopTwistJoy::Impl::sendCmdVelMsg(const sensor_msgs::Joy::ConstPtr &joy_msg) {
        // Initializes with zeros by default.
        geometry_msgs::Twist cmd_vel_msg;

        cmd_vel_msg.linear.x = 2*joy_msg->axes[1];
        cmd_vel_msg.linear.y = 2*joy_msg->axes[0];
        cmd_vel_msg.linear.z = 0;
        cmd_vel_msg.angular.z = 2*joy_msg->axes[3] * 2;
        cmd_vel_msg.angular.y = 0;
        cmd_vel_msg.angular.x = 0;
        cmd_vel_pub.publish(cmd_vel_msg);

	std_msgs::Bool lightMsg;
	bool state=joy_msg->buttons[1];
	if((state==true&&lastState==false)){
		changeState=true;
		times++;
		std::cout<<"current times: "<<times<<std::endl;
	}else{
		changeState=false;
	}
	if(changeState){
		if(times%2==0){
			lightMsg.data=true;
		}else{
			lightMsg.data=false;
		}
		
		light_pub.publish(lightMsg);
	}
	lastState=state;
	
    }

    void TeleopTwistJoy::Impl::joyCallback(const sensor_msgs::Joy::ConstPtr &joy_msg) {
        sendCmdVelMsg(joy_msg);
    }  // namespace teleop_twist_joy
}
