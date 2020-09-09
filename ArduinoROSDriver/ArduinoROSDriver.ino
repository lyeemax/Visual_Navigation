#include <ros.h>
#include <ros/time.h>
#include <geometry_msgs/Twist.h>
#include<ros_arduino_msgs/Raw_imu.h>
#include<ros_arduino_msgs/Wheel_velocity.h>
#include<std_msgs/Bool.h>


#include"encoder.h"
#include"kinematics.hpp"
#include"motor.h"
#include"pid.hpp"
#include"gy85.hpp"


ros::NodeHandle_<ArduinoHardware, 5, 5, 512, 1024> nh;


float ctrlrate=1.0;
unsigned long lastctrl;
geometry_msgs::Twist twist;
ros_arduino_msgs::Wheel_velocity vel;
ros::Publisher vel_pub("wheel_vel", &vel);
ros_arduino_msgs::Raw_imu imu_msg;
ros::Publisher imu_pub("raw_imu",&imu_msg);


uint32_t previous_imu_time = 0;
unsigned long last_time;
long last_encA;
long last_encB;
long last_encC;
long last_encD;
bool is_first=true;
bool first_enc=true;
bool gyroscope =false;
bool accelerometer =false;
bool magnetometer=false;
bool light=false;

Encoder *encD=new Encoder(18, 31);
Encoder *encC=new Encoder(19, 38);//rev
Encoder *encB=new Encoder(3, 49);
Encoder *encA=new Encoder(2,A1);//rev

float KP=0.3,KI=0.2,KD=0.2;
MPID PIDA(encA,KP,KI,KD,true);
MPID PIDB(encB,KP,KI,KD,false);
MPID PIDC(encC,KP,KI,KD,true);
MPID PIDD(encD,KP,KI,KD,false);


Gy85 *imu=new Gy85();

void cmdVelCb( const geometry_msgs::Twist& twist_msg){
  if(digitalRead(48)==LOW||digitalRead(40)==LOW){
    STOP();
  }else{
    float vx=twist_msg.linear.x;
  float vy=twist_msg.linear.y;
  float w=twist_msg.angular.z;

  float pwma=0,pwmb=0,pwmc=0,pwmd=0;
  InverseKinematic(vx,vy,w,pwma,pwmb,pwmc,pwmd);

  
  PIDA.tic();PIDB.tic();PIDC.tic();PIDD.tic();
  
  MotorA(PIDA.getPWM(pwma));
  MotorB(PIDB.getPWM(pwmb));
  MotorC(PIDC.getPWM(pwmc));  
  MotorD(PIDD.getPWM(pwmd));
  
  PIDA.toc();PIDB.toc();PIDC.toc();PIDD.toc();

  }
  
  lastctrl=millis();
}

void lgCb(const std_msgs::Bool &msg){
  if(msg.data==true){
    light=true;
  }else{
    light=false;
  }
}

ros::Subscriber<geometry_msgs::Twist> sub("cmd_vel", cmdVelCb );
ros::Subscriber<std_msgs::Bool> lightSub("light",lgCb);


void setup()
{
	IO_init();
  PIDA.init();
  PIDB.init();
  PIDC.init();
  PIDD.init();
  imu->init();
  nh.initNode();
  nh.subscribe(sub);
  nh.advertise(vel_pub);
  nh.advertise(imu_pub);
  nh.subscribe(lightSub);
  lastctrl=millis();
  pinMode(48,INPUT);//front crash detect PL1
  pinMode(40,INPUT);//back crash detect PG0
  pinMode(41,OUTPUT);//light PG1
}

void loop(){
  float wA=0;
  float wB=0;
  float wC=0;
  float wD=0;
  
  if(first_enc){
    last_encA=-encA->read();
    last_encB=encB->read();
    last_encC=-encC->read();
    last_encD=encD->read();
    last_time=millis();
    first_enc=false;
  }else{
    wA=getWheelRotatialSpeed(encA,last_encA,last_time,true);
    wB=getWheelRotatialSpeed(encB,last_encB,last_time,false);
    wC=getWheelRotatialSpeed(encC,last_encC,last_time,true);
    wD=getWheelRotatialSpeed(encD,last_encD,last_time,false);
    last_time=millis();
  }
  float vxi=0,vyi=0,omegai=0;
  ForwardKinematic(wA,wB,wC,wD,vxi,vyi,omegai);

  vel.vel.x=vxi;
  vel.vel.y=vyi;
  vel.vel.z=omegai;
  vel.header.stamp=nh.now();
  vel_pub.publish(&vel);

  if ((millis() - previous_imu_time) >= 10){
      if (is_first){
        gyroscope = imu->check_gyroscope();
        accelerometer = imu->check_accelerometer();
        magnetometer = imu->check_magnetometer();
        if (!accelerometer){
          nh.logerror("Accelerometer NOT FOUND!");
        }

        if (!gyroscope){
          nh.logerror("Gyroscope NOT FOUND!");
        }

        if (!magnetometer){
          nh.logerror("Magnetometer NOT FOUND!");
        }
        is_first = false;
      } else{
        imu_msg.header.stamp = nh.now();
        if (accelerometer){
          imu->measure_acceleration();
          imu_msg.raw_linear_acceleration = imu->raw_acceleration;
        }
        if (gyroscope){
          imu->measure_gyroscope();
          imu_msg.raw_angular_velocity = imu->raw_rotation;
        }

        if (magnetometer){
          imu->measure_magnetometer();
          imu_msg.raw_magnetic_field = imu->raw_magnetic_field;
        }
        imu_pub.publish(&imu_msg);
      }
      previous_imu_time = millis();
  }

  if(light){
    digitalWrite(41,HIGH);
  }else{
    digitalWrite(41,LOW);
  }
  
  if((millis()-lastctrl)>=200){
    STOP();
  }
  
  nh.spinOnce();
 
}
