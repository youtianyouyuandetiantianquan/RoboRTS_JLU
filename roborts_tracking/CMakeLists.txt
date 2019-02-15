cmake_minimum_required(VERSION 3.0)
project(roborts_tracking)
set(CMAKE_BUILD_TYPE Release)
find_package(OpenCV REQUIRED)
find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  image_transport
  roscpp
  tf
  roborts_msgs
  sensor_msgs
  )

if(NOT WIN32)
  ADD_DEFINITIONS("-std=c++11")
endif(NOT WIN32)

catkin_package()

add_library(kcf_tracker STATIC
  KCFcpp/src/fhog.cpp
  KCFcpp/src/kcftracker.cpp)
target_link_libraries(kcf_tracker ${OpenCV_LIBRARIES})
target_include_directories(kcf_tracker PRIVATE KCFcpp/src)

add_executable(my_publisher my_publisher.cpp)
target_link_libraries(my_publisher
        kcf_tracker
        ${catkin_LIBRARIES}
        ${OpenCV_LIBS})
target_include_directories(my_publisher
        PUBLIC
        ${catkin_INCLUDE_DIRS})

add_executable(roborts_tracking_test
  track_people.cpp
  tracking_utility.cpp)
target_link_libraries(roborts_tracking_test
  kcf_tracker
  ${OpenCV_LIBS}
  ${catkin_LIBRARIES})
target_include_directories(roborts_tracking_test
  PUBLIC
  ${catkin_INCLUDE_DIRS})



