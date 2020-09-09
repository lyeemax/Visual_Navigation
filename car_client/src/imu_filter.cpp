//
// Created by unicorn on 2020/5/30.
//

#include <iostream>
#include "../include/imu_filter.h"

Eigen::Vector4d QuaternionCross(Eigen::Vector4d a,Eigen::Vector4d b){
    double wa=a[0],xa=a[1],ya=a[2],za=a[3];
    double wb=b[0],xb=b[1],yb=b[2],zb=b[3];
    Eigen::Vector4d res;
    res<<wa*wb-xa*xb-ya*yb-za*zb,
        wa*xb+xa*wb+ya*zb-za*yb,
        wa*yb-xa*zb+ya*wb+za*xb,
        wa*zb+xa*yb-ya*xb+za*wb;
    return res;
}
void ImuFilter::EstimateBias(Eigen::Vector4d qdelta, double dt) {
    auto gyro_err=QuaternionCross(Eigen::Vector4d(q_.w(),-q_.x(),-q_.y(),-q_.z()),qdelta);
    gyro_bias_+=Eigen::Vector3d(gyro_err.y()*2.0,gyro_err.x()*2.0,gyro_err.w()*2.0)*zeta_*dt;//EQ48
}
void ImuFilter::MagCompensate(Eigen::Vector3d mag) {
    Eigen::Isometry3d Tem=Eigen::Isometry3d::Identity();
    Tem.rotate(q_);
    Eigen::Vector3d fixm=Tem*mag;
    Eigen::Vector3d Eb(0.0,sqrt(pow(fixm[0],2)+pow(fixm[1],2)),fixm[2]);
    magnetic_=Eb;
    //std::cout<<"my eBt is "<<magnetic_<<std::endl;
}
void ImuFilter::ComputeJaccobianAndError( Eigen::Vector3d acc, Eigen::Vector3d mag) {
    //compute error EQ31
    Eigen::Isometry3d Tg=Eigen::Isometry3d::Identity();
    Tg.rotate(q_.inverse());
    Eigen::Vector3d eg=Tg*gravity_-acc;

    Eigen::Isometry3d Tm=Eigen::Isometry3d::Identity();
    Tm.rotate(q_.inverse());
    Eigen::Vector3d em=Tm*magnetic_-mag;

    errf_<<eg,em;
    //compute jacobian
    double q1=q_.w(),q2=q_.x(),q3=q_.y(),q4=q_.z();
    double by=magnetic_.y(),bz=magnetic_.z();
    Jac_<<-2.0f*q3,    2.0f*q4,     -2.0f*q1,     2.0f*q2,
          2.0f*q2,     2.0f*q1,      2.0f*q4,     2.0f*q3,
          0,           -4.0f*q2,    -4.0f*q3,     0,//Jacobian of Fg
          -2.0f*bz*q3+2.0f*by*q4,              2.0f*bz*q4+2.0f*by*q3,              2.0f*by*q2-2.0f*bz*q1,     2.0f*by*q1+2.0f*bz*q2,
          2.0f*bz*q2,   -4.0f*by*q2+2.0f*bz*q1,    2.0f*bz*q4,     -4.0f*by*q4+2.0f*bz*q3,
          -2.0f*by*q2,               -2.0f*by*q1-4.0f*bz*q2,    2.0f*by*q4-4.0f*bz*q3,      2.0f*by*q3;

}

void ImuFilter::madgwickAHRSupdate(Eigen::Vector3d gyro, Eigen::Vector3d acc, Eigen::Vector3d mag, double dt){

    Eigen::Vector4d vest(0,0,0,0);//vector form of quaternion
    Eigen::Vector4d dq(0,0,0,0);//derviation of quaternion
    acc.normalize();
    mag.normalize();



    //estimate the mag on earth frame ,EQ 45 46
    MagCompensate(mag);//EQ45 46

    //EQ44 vest is normalized delatF
    ComputeJaccobianAndError(acc,mag);
    vest=Jac_.transpose()*errf_;
    vest.normalize();


    //get bias of gyro and remove bias from measurement
    EstimateBias(vest,dt);
    gyro-=gyro_bias_;//EQ49
    
    //EQ12 EQ43
    dq=QuaternionCross(Eigen::Vector4d(q_.w(),q_.x(),q_.y(),q_.z()),0.5*Eigen::Vector4d(0,gyro.x(),gyro.y(),gyro.z()));
    //EQ13

    q_.w()+=(dq[0]-gain_*vest[0])*dt;
    q_.x()+=(dq[1]-gain_*vest[1])*dt;
    q_.y()+=(dq[2]-gain_*vest[2])*dt;
    q_.z()+=(dq[3]-gain_*vest[3])*dt;
    q_.normalize();
}
