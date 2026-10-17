cmake_minimum_required(VERSION 3.8)
project(xc_aubo_arm_control)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(xc_interfaces REQUIRED) 
find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)  # 添加std_msgs依赖
find_package(urdf REQUIRED)

find_package(Bullet REQUIRED)



add_executable(xc_aubo_arm_control src/xc_aubo_control.cpp src/interfacetest.cpp )
add_executable(motion_service src/motion_service.cpp src/auboi5_path_check.cpp)
add_executable(fake_picking src/fake_picking.cpp)
add_executable(continuous_motion_client src/continuous_motion_client.cpp)



# 查找并链接共享库（假设库文件在 /opt/mylib/lib/libawesome.so）
find_library(AUBO_SDK_LIB
    NAMES aubo_sdk    # 尝试可能的库名称
    PATHS "${CMAKE_CURRENT_SOURCE_DIR}/lib"     # 指定搜索路径
    NO_DEFAULT_PATH            # 禁止搜索系统路径
)

if(NOT AUBO_SDK_LIB)
    message(FATAL_ERROR "auborobotcontroller library not found!")
endif()

message(STATUS "Found library at: ${AUBO_SDK_LIB}")




# 查找并链接共享库（假设库文件在 /opt/mylib/lib/libawesome.so）
find_library(AUBO_SDK_D_LIB
    NAMES aubo_sdk_d   # 尝试可能的库名称
    PATHS "${CMAKE_CURRENT_SOURCE_DIR}/lib"     # 指定搜索路径
    NO_DEFAULT_PATH            # 禁止搜索系统路径
)

if(NOT AUBO_SDK_D_LIB)
    message(FATAL_ERROR "aubo_sdk_d library not found!")
endif()

message(STATUS "Found library at: ${AUBO_SDK_D_LIB}")

# 设置头文件路径（假设头文件在 /opt/mylib/include）
target_include_directories(xc_aubo_arm_control PUBLIC 
    "${CMAKE_CURRENT_SOURCE_DIR}/include")


target_include_directories(motion_service PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${BULLET_INCLUDE_DIRS}")
target_link_libraries(xc_aubo_arm_control
  ${AUBO_SDK_D_LIB}
  ${AUBO_SDK_LIB}
)

target_link_libraries(motion_service
  ${AUBO_SDK_D_LIB}
  ${AUBO_SDK_LIB}
  ${BULLET_LIBRARIES}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

# 添加ROS2依赖
ament_target_dependencies(xc_aubo_arm_control
    rclcpp
    sensor_msgs
    xc_interfaces
    tf2
    tf2_geometry_msgs
    std_msgs  # 添加std_msgs依赖
)

ament_target_dependencies(motion_service
    rclcpp
    sensor_msgs
    xc_interfaces
    tf2
    tf2_geometry_msgs
    urdf
)


ament_target_dependencies(fake_picking
    rclcpp
    sensor_msgs
    xc_interfaces
    tf2
    tf2_geometry_msgs
)

ament_target_dependencies(continuous_motion_client
    rclcpp
    sensor_msgs
    xc_interfaces
    tf2
    tf2_geometry_msgs
)




install(TARGETS xc_aubo_arm_control motion_service fake_picking continuous_motion_client
DESTINATION lib/${PROJECT_NAME}
)

# install(TARGETS motion_service
# DESTINATION lib/${PROJECT_NAME}
# )

# install(TARGETS fake_picking
# DESTINATION lib/${PROJECT_NAME}
# )

# install(TARGETS continuous_motion_client
# DESTINATION lib/${PROJECT_NAME}
# )




# 在install(TARGETS ...)之后添加库文件安装指令
# 修改库文件安装路径为直接安装到lib目录
install(DIRECTORY 
    lib/
    DESTINATION lib
    FILES_MATCHING PATTERN "*.so*"
)

# 优化rpath配置
set_target_properties(xc_aubo_arm_control PROPERTIES
    INSTALL_RPATH "$ORIGIN/../lib"
    BUILD_WITH_INSTALL_RPATH TRUE
)

set_target_properties(motion_service PROPERTIES
    INSTALL_RPATH "$ORIGIN/../lib"
    BUILD_WITH_INSTALL_RPATH TRUE
)




# # 在target_link_libraries之后添加库路径解析
# link_directories(
#     ${CMAKE_CURRENT_SOURCE_DIR}/lib/xc_aubo_arm_control/log4cplus
#     ${CMAKE_CURRENT_SOURCE_DIR}/lib/xc_aubo_arm_control/robotSDK
# )




ament_package()