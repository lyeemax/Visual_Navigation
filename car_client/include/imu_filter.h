//
// Created by unicorn on 2020/5/30.
//

#ifndef CAR_CLIENT_IMU_FILTER_H
#define CAR_CLIENT_IMU_FILTER_H

#include <Eigen/Core>
#include <Eigen/Geometry>

class ImuFilter{
private:
    double gain_;
    double zeta_;
    Eigen::Quaterniond q_;
    Eigen::Matrix<double,6,4> Jac_;
    Eigen::Matrix<double,6,1> errf_;
    Eigen::Vector3d  gravity_;
    Eigen::Vector3d  magnetic_;
    Eigen::Vector3d gyro_bias_;
public:
    ImuFilter():gain_(0.1),zeta_(0){
        q_.Identity();
        gravity_.setZero();
        gravity_.z()=1.0;
        gyro_bias_.setZero();
    }
    void setAlgorithmGain(double gain)
    {
        gain_ = gain;
    }

    void setDriftBiasGain(double zeta)
    {
        zeta_ = zeta;
    }
    void setOrientation(Eigen::Quaterniond quaterniond){
        q_=quaterniond;
        q_.normalize();
        gyro_bias_.setZero();
    }
    Eigen::Quaterniond getOrientation(){
        return q_;
    }
    void getOrientation(double &q0,double &q1,double &q2,double &q3){
        q_.normalize();
        q0=q_.w();
        q1=q_.x();
        q2=q_.y();
        q3=q_.z();
    }
    void ComputeJaccobianAndError(Eigen::Vector3d acc,
                          Eigen::Vector3d mag);
    void madgwickAHRSupdate(Eigen::Vector3d gyro,
                            Eigen::Vector3d acc,
                            Eigen::Vector3d mag,
                            double dt);
    void MagCompensate(Eigen::Vector3d mag);
    void EstimateBias(Eigen::Vector4d qdelta,double dt);

};

#endif //CAR_CLIENT_IMU_FILTER_H
