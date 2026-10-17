cmake_minimum_required(VERSION 3.8)
project(tomato_robot_ui)
#Qt生成文件的，copy到这里
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(Boost 1.71.0 REQUIRED COMPONENTS thread)
#寻找Qt文件位置，Qt生成文件，copy到这里
find_package(Qt5 COMPONENTS Widgets Core REQUIRED)  # 合并查找并添加Core模块
# find_package(Qt5 COMPONENTS QProcess REQUIRED)
# find_package(Qt5 COMPONENTS QDebug REQUIRED)
find_package(xc_interfaces REQUIRED) 


#opencv
find_package(OpenCV REQUIRED)
# 在find_package部分添加
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)

# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)
#指定Qt各类文件位置
file(GLOB PRO_SOURCE_DIR RELATIVE ${CMAKE_SOURCE_DIR} FOLLOW_SYMLINKS src/*.cpp)
file(GLOB PRO_UI_DIR RELATIVE ${CMAKE_SOURCE_DIR} FOLLOW_SYMLINKS ui/*.ui)
file(GLOB PRO_INCLUDE_DIR RELATIVE ${CMAKE_SOURCE_DIR} FOLLOW_SYMLINKS include/*.h *.hpp)
file(GLOB PRO_RESOURCE_DIR RELATIVE ${CMAKE_SOURCE_DIR} FOLLOW_SYMLINKS resource/*.qrc)
#生成的UI和资源头文件文位置
qt5_wrap_ui(QT_UI_HPP ${PRO_UI_DIR})
qt5_wrap_cpp(QT_MOC_HPP ${PRO_INCLUDE_DIR})
qt5_add_resources(QT_RESOURCE_HPP ${PRO_RESOURCE_DIR})

include_directories(
	include/
	${CMAKE_CURRENT_BINARY_DIR}
  ${OpenCV_INCLUDE_DIRS} #opencv头文件位置
)
#指定各类文件位置
add_executable(${PROJECT_NAME}
    ${PRO_SOURCE_DIR}
    ${PRO_UI_DIR}
    ${PRO_INCLUDE_DIR}
    ${QT_MOC_HPP}
    ${QT_RESOURCE_HPP}  
  )
#Qt生成文件的，copy到这里
# 修改链接库部分
target_link_libraries(${PROJECT_NAME} 
    Qt5::Widgets 
    Qt5::Core  # 替换原来的Qt5::QProcess
    ${OpenCV_LIBS}
)

# target_link_libraries(${PROJECT_NAME} Qt5::QProcess)


#ros2相关的库需要使用ament_target_dependencies方式添加依赖
ament_target_dependencies(${PROJECT_NAME} 
rclcpp Boost sensor_msgs cv_bridge image_transport xc_interfaces)

install(TARGETS ${PROJECT_NAME}
        DESTINATION lib/${PROJECT_NAME}
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

ament_package()
