//
// Created by unicorn on 2020/5/30.
//

#ifndef CAR_CLIENT_STATELESSORIENTATION_H
#define CAR_CLIENT_STATELESSORIENTATION_H
#include <geometry_msgs/Vector3.h>
#include <geometry_msgs/Quaternion.h>

class StatelessOrientation
{
public:
    static bool computeOrientation(
            geometry_msgs::Vector3 acceleration,
            geometry_msgs::Vector3 magneticField,
            geometry_msgs::Quaternion& orientation);

    static bool computeOrientation(
            geometry_msgs::Vector3 acceleration,
            geometry_msgs::Quaternion& orientation);

};


#endif //CAR_CLIENT_STATELESSORIENTATION_H
