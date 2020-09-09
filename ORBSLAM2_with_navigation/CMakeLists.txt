cmake_minimum_required(VERSION 2.8)
project(ORB_SLAM2)

find_package(OpenCV)
find_package(PCL )
find_package(Eigen3 )
find_package(Pangolin )

find_package(catkin REQUIRED COMPONENTS
    roscpp
    std_msgs
    sensor_msgs
    cv_bridge
    camera_model
        tf
message_filters
        image_transport
    )

catkin_package()

include_directories(
    ${catkin_INCLUDE_DIRS}
    )



include_directories(
${PROJECT_SOURCE_DIR}
${PROJECT_SOURCE_DIR}/include
${EIGEN3_INCLUDE_DIR}
${Pangolin_INCLUDE_DIRS}
        ${PCL_INCLUDE_DIRS}
)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

set(PROJECT_NAME
src/System.cc
src/Tracking.cc
src/LocalMapping.cc
src/LoopClosing.cc
src/ORBextractor.cc
src/ORBmatcher.cc
src/FrameDrawer.cc
src/Converter.cc
src/MapPoint.cc
src/KeyFrame.cc
src/Map.cc
        src/Optimizer.cc
src/PnPsolver.cc
src/Frame.cc
src/KeyFrameDatabase.cc
        src/Sim3Solver.cc
        include/orbtypes.h
src/main.cc include/PointCloudMapper.h)

set(LIBS
${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2/lib/libDBoW2.so
${PROJECT_SOURCE_DIR}/Thirdparty/g2o/lib/libg2o.so
${OpenCV_LIBS}
${EIGEN3_LIBS}
${Pangolin_LIBRARIES}
        ${PCL_LIBRARIES}
${catkin_LIBRARIES}
)

# Build examples

#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/RGB-D)

#add_executable(rgbd_tum
#Examples/RGB-D/rgbd_tum.cc)
#target_link_libraries(rgbd_tum ${PROJECT_NAME})

#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/Stereo)

#add_executable(stereo_kitti
#Examples/Stereo/stereo_kitti.cc)
#target_link_libraries(stereo_kitti ${PROJECT_NAME})

#add_executable(stereo_euroc
#Examples/Stereo/stereo_euroc.cc)
#target_link_libraries(stereo_euroc ${PROJECT_NAME})


#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/Monocular)
#
#add_executable(mono_tum
#Examples/Monocular/mono_tum.cc)
#target_link_libraries(mono_tum ${PROJECT_NAME})
#
#add_executable(mono_kitti
#Examples/Monocular/mono_kitti.cc)
#target_link_libraries(mono_kitti ${PROJECT_NAME})
#
#add_executable(mono_euroc
#Examples/Monocular/mono_euroc.cc)
#target_link_libraries(mono_euroc ${PROJECT_NAME})
add_executable(ORBSLAM2 ${PROJECT_NAME} include/PointCloudMapper.h)
target_link_libraries(ORBSLAM2 ${LIBS} )
