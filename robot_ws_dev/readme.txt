qt-ros项目生成可执行文件方法：
	1./home/xc/code/test/ros2/qt_ros2_ws/src/tomato_robot_ui/目录下新建build目录，然后进入build目录，执行cmake ..   再执行make目录即可，可执行文件在新建的build目录中。通过qt打开cmakelist.txt文件，可自动识别新建的build目录
	2.进入/home/xc/code/test/ros2/qt_ros2_ws/目录，执行colcon build，生成的可知性文件在/home/xc/code/test/ros2/qt_ros2_ws/install/tomato_robot_ui/lib/tomato_robot_ui/目录中，可通过ros2 run tomato_robot_ui tomato_robot_ui  执行
	3.删除步骤1新建的build目录，通过qt打开cmakelist.txt文件，可自动生成可执行文件


修改功能包名称
	1.修改cmakelist文件中project(tomato_robot_ui)
	2.使用vscode进行全部修改


如果colcon build编译卡顿,那请执行colcon build --executor sequential命令
编译指定功能包：colcon build --packages-select my_package

ui界面增加ros相关包：
	1.cmakelist文件中增加以下内容 例rclcpp
	find_package(rclcpp REQUIRED)
	find package(Boost 1.71.0 REQUIRED COMPONENTS thread)
	target_link_libraries(${PROJECT_NAME} PRIVATE Qt5::Widgets rclcpp)
	2.将target_link_libraries(${PROJECT_NAME} PRIVATE Qt5::Widgets)中的PRIVATE删除



ui界面增加安装的第三方包，直接一键三联 （通过apt install安装的库一般可以这样找到，以opencv为例）
	find_package(OpenCV REQUIRED) #opencv
	include_directories(
		include/
		${CMAKE_CURRENT_BINARY_DIR}
		${OpenCV_INCLUDE_DIRS} #opencv头文件位置
	)
	target_link_libraries(${PROJECT_NAME} 
		Qt5::Widgets 
		${OpenCV_LIBS} #opencv库文件位置
	)

	其中 ${CMAKE_CURRENT_SOURCE_DIR}表示当前cmakefiles文件所在的位置



添加第三方.so库文件
	file(GLOB ORBBEC_LIBS "lib/x86_64/*") #.so库文件的位置

	include_directories(include) #头文件的位置
	include_directories(utility)

	add_executable(image_publisher src/orbbec.cpp src/image_publisher.cpp) #构建c++
	ament_target_dependencies(image_publisher rclcpp sensor_msgs cv_bridge OpenCV) #ros2相关的软件包

	target_link_libraries(image_publisher
		${ORBBEC_LIBS}
	) #链接库和即将生成的可执行文件image_publisher

	install(TARGETS
	image_publisher
	DESTINATION lib/${PROJECT_NAME})  #将可执行文件安装到/install/lib/${PROJECT_NAME}文件夹中

	install(DIRECTORY 
	lib/x86_64/
	DESTINATION lib/)   #将库文件安装到/install/lib文件夹中






	自动安装软件包
rosdepc install --from-path src --ignore-src -r -y

	多核编译

colcon build --parallel-workers 8

指定相机序列号
ros2 launch realsense2_camera rs_launch.py serial_no:=_819612070123
右相机：serial_no:=f1150496   左相机：serial_no:=f1422449


切换相机：
查看src/xc_robot_main_control/launch/realsense_hot_change_launch.py
终端执行 kill -9 $(cat /tmp/realsense_camera.pid) 即可关闭相机


串口永久权限
whoami

sudo usermod -aG dialout username


到达share目录：
from ament_index_python.packages import get_package_share_directory
import 

package_share_dir = get_package_share_directory('hand_eye_calibration_python')
self.get_logger().info(f'包share目录: {package_share_dir}')

model_path = os.path.join(package_share_dir, 'weights', 'tomato-seg2025063018-yolov8-xucan.pt')

在setup.py文件中把想要使用的文件夹挪到share文件夹中

('share/' + package_name + '/weights', ['weights/best.pt']),



ros2 run <package_name> <node_name> 2>&1 | tee output.log