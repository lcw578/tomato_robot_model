cmake_minimum_required(VERSION 3.8)
project(vision_tomato_interfaces)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/InstanceMask.msg"
  "msg/FruitMaturity.msg"
  "msg/PickPoseCandidate.msg"
  "msg/PickPoseArray.msg"
  "srv/SegmentTomato.srv"
  "srv/GeneratePickPoses.srv"
  "srv/GetPickPoses.srv"
  DEPENDENCIES std_msgs sensor_msgs geometry_msgs
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
