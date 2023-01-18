cmake_minimum_required(VERSION 3.0.2)
project(image_matching)

add_compile_options(-std=c++17)
# set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  bb_msgs
  tf
  message_generation
)

find_package(OpenCV 4 QUIET)
if (NOT OpenCV_FOUND)
    find_package(OpenCV 3 REQUIRED)
endif()

catkin_python_setup()

catkin_package(
 CATKIN_DEPENDS rospy bb_msgs
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)
