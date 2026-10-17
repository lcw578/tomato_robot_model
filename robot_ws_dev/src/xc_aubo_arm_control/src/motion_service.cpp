#include "rclcpp/rclcpp.hpp"
#include "xc_interfaces/srv/move_to_pose.hpp"
#include "xc_interfaces/srv/move_joints.hpp"
#include "aubo_sdk/AuboRobotMetaType.h"
#include "aubo_sdk/serviceinterface.h"
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include "xc_interfaces/msg/arm_status.hpp"
#include "xc_interfaces/msg/midpoint_reached.hpp"
#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/quaternion.hpp>
#include <chrono>
#include <mutex>
#include <std_msgs/msg/int32.hpp>
#include <std_msgs/msg/u_int8.hpp>
#include <std_msgs/msg/bool.hpp>  // 添加Bool消息头文件
#include <cmath> // 添加数学库头文件
#include "xc_aubo_arm_control/auboi5_path_check.h"


#define SERVER_HOST "192.168.1.5"//工控机IP
// #define SERVER_HOST "192.168.102.128"//仿真机IP
// #define SERVER_HOST "192.168.119.129"//home仿真机IP


#define SERVER_PORT 8899
using namespace std::chrono_literals;


class MotionService : public rclcpp::Node
{
public:
MotionService() : Node("aubo_motion_service")
    {
        // 初始化初始位置，即机械臂到果箱的位置
        initial_position_[0]=10.32*M_PI/180;
        initial_position_[1]=15.37*M_PI/180; 
        initial_position_[2]=134.05*M_PI/180;
        initial_position_[3]=117.05*M_PI/180;
        initial_position_[4]=86.63*M_PI/180;
        initial_position_[5]=0.0*M_PI/180;

        // 初始化采摘时的起始位置（比果箱位置X略小）
        pick_start_position_[0]=12.74*M_PI/180;
        pick_start_position_[1]=34.05*M_PI/180; 
        pick_start_position_[2]=137.08*M_PI/180;
        pick_start_position_[3]=106.70*M_PI/180;
        pick_start_position_[4]=86.70*M_PI/180;
        pick_start_position_[5]=0.0*M_PI/180;

        //初始化机械臂中间点的内缩位置（与中间点的关节1位置相同）
        midpoint_contract_position_[0]=0*M_PI/180;
        midpoint_contract_position_[1]=34.05*M_PI/180; 
        midpoint_contract_position_[2]=137.08*M_PI/180;
        midpoint_contract_position_[3]=106.70*M_PI/180;
        midpoint_contract_position_[4]=86.70*M_PI/180;
        midpoint_contract_position_[5]=0.0*M_PI/180;

        last_midpoint_contract_position_[0]=midpoint_contract_position_[0];
        last_midpoint_contract_position_[1]=midpoint_contract_position_[1];
        last_midpoint_contract_position_[2]=midpoint_contract_position_[2];
        last_midpoint_contract_position_[3]=midpoint_contract_position_[3];
        last_midpoint_contract_position_[4]=midpoint_contract_position_[4];
        last_midpoint_contract_position_[5]=midpoint_contract_position_[5];

        midpoint_min_y=0.25;

        last_x1_status_=0.0;
        
        // 初始化控制是否发送末端传感器信息
        send_end_effector_sensor_info_=true;

        // 初始化工具偏移（默认无偏移，identity四元数 w=1,x=y=z=0）
        tool_offset_pos_[0]=0.0;
        tool_offset_pos_[1]=0.0;
        tool_offset_pos_[2]=0.0;
        tool_offset_q_.setRPY(0.0, 0.0, 0.0);
        has_tool_offset_=false;

        //设置速度参数
        arm_joints_speed_=120;//单位是 度/s
        arm_joints_low_speed_=5;//单位是 度/s

        line_speed_=0.5;//单位是m/s

        // 创建回调组
        MutuallyExclusive_callback_group_ = this->create_callback_group(
            rclcpp::CallbackGroupType::MutuallyExclusive);
        reentrant_callback_group_ = this->create_callback_group(
        rclcpp::CallbackGroupType::Reentrant);
        // 初始化去采摘服务连接
        if (robot_service_.robotServiceLogin(SERVER_HOST, SERVER_PORT, "aubo", "123456") != aubo_robot_namespace::InterfaceCallSuccCode)
        {
            throw std::runtime_error("服务连接失败");
        }
        robot_service_.robotServiceInitGlobalMoveProfile();

        // 初始化返回服务连接
        if (robot_return_service_.robotServiceLogin(SERVER_HOST, SERVER_PORT, "aubo", "123456") != aubo_robot_namespace::InterfaceCallSuccCode)
        {
            throw std::runtime_error("服务连接失败");
        }
        robot_return_service_.robotServiceInitGlobalMoveProfile();
        

        // 初始化话题连接
        if (robot_pub_.robotServiceLogin(SERVER_HOST, SERVER_PORT, "aubo", "123456") != aubo_robot_namespace::InterfaceCallSuccCode)
        {
            throw std::runtime_error("话题连接失败");
        }

        robot_pub_.robotServiceInitGlobalMoveProfile();

        // 读取当前已设置的工具运动学参数（工具相对法兰盘TCP的偏移）
        refreshToolOffset();

        publisher_ = create_publisher<xc_interfaces::msg::ArmStatus>("arm_status", 10);
        if (send_end_effector_sensor_info_==true)
        {
            x1_status_publisher_ = create_publisher<std_msgs::msg::UInt8>("gripper/input_x1_status", 10 );
        }

        midpoint_publisher_=create_publisher<xc_interfaces::msg::MidpointReached>("midpoint_reached", 10);


        timer_ = create_wall_timer(
            std::chrono::milliseconds(20),
            [this]()
            { this->publish_status(); },reentrant_callback_group_);

        auto service_qos = rmw_qos_profile_services_default;
        // 去采摘服务
        service_ = create_service<xc_interfaces::srv::MoveToPose>(
            "move_to_pose",
            [this](const std::shared_ptr<xc_interfaces::srv::MoveToPose::Request> request,
                   std::shared_ptr<xc_interfaces::srv::MoveToPose::Response> response)
            {
                handle_request_1(request, response);
            },
            service_qos,
            MutuallyExclusive_callback_group_);

        // 返回服务
        return_service_ = create_service<xc_interfaces::srv::MoveToPose>(
            "return_to_box",
            [this](const std::shared_ptr<xc_interfaces::srv::MoveToPose::Request> request,
                   std::shared_ptr<xc_interfaces::srv::MoveToPose::Response> response)
            {
                handle_return_request_1(request, response);
            },
            service_qos,
            MutuallyExclusive_callback_group_);
        //从目标点返回中间点的服务
        target_return_midpoint_service_ = create_service<xc_interfaces::srv::MoveToPose>(
            "target_return_midpoint",  
            [this](const std::shared_ptr<xc_interfaces::srv::MoveToPose::Request> request,
                    std::shared_ptr<xc_interfaces::srv::MoveToPose::Response> response)
            {
                handle_target_return_midpoint_request_1(request, response);
            },
            service_qos,
            MutuallyExclusive_callback_group_);        
        
        // 关节运动服务
        move_joints_service_ = create_service<xc_interfaces::srv::MoveJoints>(
            "move_joints",
            [this](const std::shared_ptr<xc_interfaces::srv::MoveJoints::Request> request,
                   std::shared_ptr<xc_interfaces::srv::MoveJoints::Response> response) {
                handle_joints_request(request, response);
            });
    }

    ~MotionService()
    {
        arm_checker_.~Collision_Joint_Checker();
        robot_service_.robotServiceLogout();
        robot_return_service_.robotServiceLogout();
        robot_pub_.robotServiceLogout();
    }

private:
    void publish_status()
    {
        auto t_start = std::chrono::high_resolution_clock::now();

        std::lock_guard<std::mutex> lock(pub_mutex_);

        xc_interfaces::msg::ArmStatus msg;
        aubo_robot_namespace::wayPoint_S current_pose;

        // 使用话题专用连接
        robot_pub_.robotServiceGetCurrentWaypointInfo(current_pose);

        // 填充位置信息
        msg.position.x = current_pose.cartPos.position.x;
        msg.position.y = current_pose.cartPos.position.y;
        msg.position.z = current_pose.cartPos.position.z;

        // 填充姿态信息
        msg.orientation.w = current_pose.orientation.w;
        msg.orientation.x = current_pose.orientation.x;
        msg.orientation.y = current_pose.orientation.y;
        msg.orientation.z = current_pose.orientation.z;


        aubo_robot_namespace::Rpy tempRpy;
        robot_pub_.quaternionToRPY(current_pose.orientation,tempRpy);
        msg.rx = tempRpy.rx;
        msg.ry = tempRpy.ry;
        msg.rz = tempRpy.rz;
        // msg.jointpos = current_pose.jointpos;
        // for (int i = 0; i < 6; i++)
        // {
        //     msg.jointpos[i] = current_pose.jointpos[i]*180/M_PI;
        // }
        for (int i = 0; i < 6; i++)
        {
            msg.joints[i] = current_pose.jointpos[i]*180/M_PI;
        }




        // RCLCPP_INFO(this->get_logger(), "方向: w=%.3f, x=%.3f, y=%.3f, z=%.3f",
        //     msg.orientation.w,
        //     msg.orientation.x,
        //     msg.orientation.y,
        //     msg.orientation.z);

        // RCLCPP_INFO(this->get_logger(), "RPY: rz=%.3f, ry=%.3f, rz=%.3f",
        //     tempRpy.rx*180/M_PI,
        //     tempRpy.ry*180/M_PI,
        //     tempRpy.rz*180/M_PI);



        // 获取IO状态
        // double di_status;
        // // robot_pub_.robotServiceGetBoardIOStatus(aubo_robot_namespace::RobotToolDO, di_status);
        // robot_pub_.robotServiceGetToolIoStatus("T_DI/O_01", di_status);
        // msg.di_01 = di_status;
        

        // // robot_pub_.robotServiceGetBoardIOStatus(aubo_robot_namespace::RobotToolDO, di_status);
        // robot_pub_.robotServiceGetToolIoStatus("T_DI/O_03", di_status);
        // msg.di_03 = di_status;



        publisher_->publish(msg);
        // RCLCPP_INFO(this->get_logger(), "----机械臂位置x：%f, y：%f, z：%f", msg.position.x, msg.position.y, msg.position.z );



        if (send_end_effector_sensor_info_==true)
        {
            std_msgs::msg::UInt8 msg2;
            double di_status2;

            
            // robot_pub_.robotServiceGetBoardIOStatus(aubo_robot_namespace::RobotToolDO, di_status);
            robot_pub_.robotServiceGetToolIoStatus("T_DI/O_02", di_status2);
            msg2.data = di_status2;
            if (last_x1_status_!=di_status2)
            {
                last_x1_status_=di_status2;
                RCLCPP_INFO(this->get_logger(), "传感器信号是：%f", di_status2 );                
                auto t_end = std::chrono::high_resolution_clock::now();
                double dt_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
                RCLCPP_INFO(this->get_logger(), "publish_status 耗时: %.3f ms", dt_ms);
            }
            x1_status_publisher_->publish(msg2);
        }

    }
    // 添加辅助函数
    void logMoveResult(int ret, const std::string& success_msg, const std::string& fail_msg) 
    {
        if (ret != aubo_robot_namespace::InterfaceCallSuccCode) 
        {
            RCLCPP_WARN(this->get_logger(), "%s，返回值:%d", fail_msg.c_str(), ret);
        } else {
            RCLCPP_WARN(this->get_logger(), "%s", success_msg.c_str());
        }
    };

    // 工具坐标系相对法兰盘(TCP)的运动学偏移（硬编码值，如需修改请在此处更新）
    void refreshToolOffset()
    {
        // 1. 硬编码偏移值
        tool_offset_pos_[0] = -0.001132;  // X (m)
        tool_offset_pos_[1] = -0.001292;  // Y (m)
        tool_offset_pos_[2] =  0.057288;  // Z (m)
        tool_offset_q_.setRPY(0.0, 0.0, 0.0);  // RX, RY, RZ (rad)
        RCLCPP_INFO(this->get_logger(),
            "工具偏移(硬编码): pos=(%.4f, %.4f, %.4f) m, rpy=(%.4f, %.4f, %.4f) rad",
            tool_offset_pos_[0], tool_offset_pos_[1], tool_offset_pos_[2], 0.0, 0.0, 0.0);

        // 2. 写入SDK
        aubo_robot_namespace::ToolKinematicsParam set_param;
        set_param.toolInEndPosition.x = tool_offset_pos_[0];
        set_param.toolInEndPosition.y = tool_offset_pos_[1];
        set_param.toolInEndPosition.z = tool_offset_pos_[2];
        set_param.toolInEndOrientation.w = tool_offset_q_.w();
        set_param.toolInEndOrientation.x = tool_offset_q_.x();
        set_param.toolInEndOrientation.y = tool_offset_q_.y();
        set_param.toolInEndOrientation.z = tool_offset_q_.z();
        int set_ret = robot_service_.robotServiceSetToolKinematicsParam(set_param);
        if (set_ret != aubo_robot_namespace::InterfaceCallSuccCode)
        {
            RCLCPP_WARN(this->get_logger(),
                "写入SDK工具偏移失败，ret=%d", set_ret);
        }
        else
        {
            RCLCPP_INFO(this->get_logger(), "写入SDK工具偏移成功");
        }

        // 3. 从SDK读回验证
        aubo_robot_namespace::ToolKinematicsParam get_param;
        int get_ret = robot_service_.robotServiceGetToolKinematicsParam(get_param);
        if (get_ret == aubo_robot_namespace::InterfaceCallSuccCode)
        {
            RCLCPP_INFO(this->get_logger(),
                "SDK读回工具偏移: pos=(%.6f, %.6f, %.6f), q=(w=%.6f, x=%.6f, y=%.6f, z=%.6f)",
                get_param.toolInEndPosition.x,
                get_param.toolInEndPosition.y,
                get_param.toolInEndPosition.z,
                get_param.toolInEndOrientation.w,
                get_param.toolInEndOrientation.x,
                get_param.toolInEndOrientation.y,
                get_param.toolInEndOrientation.z);

            // 数值对比
            bool pos_ok = std::fabs(get_param.toolInEndPosition.x - tool_offset_pos_[0]) < 1e-6
                      && std::fabs(get_param.toolInEndPosition.y - tool_offset_pos_[1]) < 1e-6
                      && std::fabs(get_param.toolInEndPosition.z - tool_offset_pos_[2]) < 1e-6;
            bool ori_ok = std::fabs(get_param.toolInEndOrientation.w - tool_offset_q_.w()) < 1e-6
                      && std::fabs(get_param.toolInEndOrientation.x - tool_offset_q_.x()) < 1e-6
                      && std::fabs(get_param.toolInEndOrientation.y - tool_offset_q_.y()) < 1e-6
                      && std::fabs(get_param.toolInEndOrientation.z - tool_offset_q_.z()) < 1e-6;
            if (pos_ok && ori_ok)
            {
                RCLCPP_INFO(this->get_logger(), "工具偏移写入SDK并读回验证一致 ✓");
                has_tool_offset_ = true;
            }
            else
            {
                RCLCPP_WARN(this->get_logger(),
                    "SDK读回值与硬编码不一致！pos_ok=%d, ori_ok=%d", pos_ok, ori_ok);
                has_tool_offset_ = false;
            }
        }
        else
        {
            RCLCPP_WARN(this->get_logger(),
                "SDK读回工具偏移失败，ret=%d，本次不做工具偏移变换", get_ret);
            has_tool_offset_ = false;
        }
    };

    
    


    void handle_request_1(const std::shared_ptr<xc_interfaces::srv::MoveToPose::Request> request,
        std::shared_ptr<xc_interfaces::srv::MoveToPose::Response> response)
    {
        std::lock_guard<std::mutex> lock(service_mutex_);
        RCLCPP_INFO(this->get_logger(), "move_to_pose 收到了请求");

        // 每次请求重新应用工具偏移
        refreshToolOffset();

        // 获取当前关节角度（使用SDK标准接口）
        bool only_go_target=false;
        if (request->mid_to_target==true)
        {
            only_go_target=true;
        }
        
        
        aubo_robot_namespace::wayPoint_S intermediate_point;
        // 示例：设置所有关节的最大速度为30度/秒（转换为弧度）
        aubo_robot_namespace::JointVelcAccParam jointMaxVelc;
        for(int i=0; i<6; i++) {
        jointMaxVelc.jointPara[i] = arm_joints_speed_ * M_PI/180; // 30度/秒 → 0.5236 rad/s
        }
        robot_service_.robotServiceSetGlobalMoveJointMaxVelc(jointMaxVelc);
        robot_service_.robotServiceSetGlobalMoveJointMaxAcc(jointMaxVelc); 
        aubo_robot_namespace::wayPoint_S current_waypoint;
        aubo_robot_namespace::wayPoint_S target_point;
        aubo_robot_namespace::wayPoint_S target_point1;
        aubo_robot_namespace::wayPoint_S target_point2;
        aubo_robot_namespace::wayPoint_S target_point3;
        aubo_robot_namespace::wayPoint_S target_point4;




        // double temp_pick_intermediate_joint_[6]={0.0,0.0,0.0,0.0,0.0,0.0};
    
        //设置直线运动的最大速度
        robot_service_.robotServiceSetGlobalMoveEndMaxLineVelc(line_speed_);
        robot_service_.robotServiceSetGlobalMoveEndMaxLineAcc(line_speed_);

        //机械臂轨迹容器，用于存储机械臂运动轨迹
        std::vector<std::vector<double>> trajectory;
        trajectory.clear();

        std::vector<aubo_robot_namespace::wayPoint_S> temp_waypoints_IKsuccess;
        temp_waypoints_IKsuccess.clear();   

        int ret=-1;
        ret = robot_service_.robotServiceGetCurrentWaypointInfo(current_waypoint);
        // trajectory.push_back(current_waypoint.jointpos);
        trajectory.push_back(std::vector<double>(current_waypoint.jointpos, current_waypoint.jointpos + 6));
        

        
        double mid2target_angle_cal=0.0*M_PI/180;

        if(request->y>0)
        {
            if(request->mid2target_angle<-70*M_PI/180 && request->mid2target_angle>-110*M_PI/180) 
            {
                mid2target_angle_cal=-90.0*M_PI/180;
            }
                // 角度转换到-90°-90°上
            else if (request->mid2target_angle>90*M_PI/180)
            {
                mid2target_angle_cal=request->mid2target_angle-180*M_PI/180;
            }
            else if (request->mid2target_angle<=-110*M_PI/180)
            {
                mid2target_angle_cal=request->mid2target_angle+180*M_PI/180;
            }
            else
            {
                mid2target_angle_cal=request->mid2target_angle;
            }
        }
        else
        {
            if(request->mid2target_angle<110*M_PI/180 && request->mid2target_angle>70*M_PI/180) //朝前生长
            {
                mid2target_angle_cal=90.0*M_PI/180;
            }
            else if (request->mid2target_angle>=110*M_PI/180)
            {
                mid2target_angle_cal=request->mid2target_angle-180*M_PI/180;
            }
            else if (request->mid2target_angle<-90*M_PI/180)
            {
                mid2target_angle_cal=request->mid2target_angle+180*M_PI/180;
            }
            else
            {
                mid2target_angle_cal=request->mid2target_angle;
            }
        }

        RCLCPP_INFO(this->get_logger(), "request->mid2target_angle是：%f ,mid2target_angle_cal是：%f", request->mid2target_angle,mid2target_angle_cal );                


        
    
        // 创建过渡点（保持原有逻辑）

        create_waypoint(request, intermediate_point, true, mid2target_angle_cal);
        // 创建目标点（保持原有逻辑）
        create_waypoint(request, target_point, false, mid2target_angle_cal);

        if (only_go_target==false)
        {
            aubo_robot_namespace::wayPoint_S temp_pick_start_point;
            temp_pick_start_point.jointpos[0]=pick_start_position_[0];
            temp_pick_start_point.jointpos[1]=pick_start_position_[1];
            temp_pick_start_point.jointpos[2]=pick_start_position_[2];
            temp_pick_start_point.jointpos[3]=pick_start_position_[3];
            temp_pick_start_point.jointpos[4]=pick_start_position_[4];
            temp_pick_start_point.jointpos[5]=pick_start_position_[5];
            if (IK_pos2joint(temp_pick_start_point, intermediate_point) == false) 
            {
                response->result = 30001;
                return;
            }
        }
        else
        {
            if (IK_pos2joint(current_waypoint, intermediate_point) == false) 
            {
                response->result = 30001;
                return;
            }
        }
        if(intermediate_point.jointpos[1]>90*M_PI/180 || intermediate_point.jointpos[1]<(-90)*M_PI/180)
            {
                response->result = 30001;
                return;
            }
        if(request->y>0)
        {
            if (intermediate_point.jointpos[0]<10*M_PI/180)
            {
                intermediate_point.jointpos[0]+=M_PI*2;
            }
        }
        else
        {
            if (intermediate_point.jointpos[0]>10*M_PI/180)
            {
                intermediate_point.jointpos[0]-=M_PI*2;
            }
        }
        //设置直线运行过渡点
        target_point1=target_point;
        target_point2=target_point;
        target_point3=target_point;
        target_point4=target_point;
        double delta_distance=0.1/5;
        if (request->y>0)
        {
            if(mid2target_angle_cal==-90.0*M_PI/180) // 朝前生长就从上往下采摘
            {                
                target_point1.cartPos.position.y=target_point.cartPos.position.y-delta_distance*std::cos(60*M_PI/180)*4;
                target_point2.cartPos.position.y=target_point.cartPos.position.y-delta_distance*std::cos(60*M_PI/180)*3;
                target_point3.cartPos.position.y=target_point.cartPos.position.y-delta_distance*std::cos(60*M_PI/180)*2;
                target_point4.cartPos.position.y=target_point.cartPos.position.y-delta_distance*std::cos(60*M_PI/180)*1;

                target_point1.cartPos.position.z=target_point.cartPos.position.z+delta_distance*std::sin(60*M_PI/180)*4;
                target_point2.cartPos.position.z=target_point.cartPos.position.z+delta_distance*std::sin(60*M_PI/180)*3;
                target_point3.cartPos.position.z=target_point.cartPos.position.z+delta_distance*std::sin(60*M_PI/180)*2;
                target_point4.cartPos.position.z=target_point.cartPos.position.z+delta_distance*std::sin(60*M_PI/180)*1;
            }
            else //不是完全朝前生长的
            {               
                target_point1.cartPos.position.x=target_point.cartPos.position.x+delta_distance*std::sin(mid2target_angle_cal)*4;
                target_point2.cartPos.position.x=target_point.cartPos.position.x+delta_distance*std::sin(mid2target_angle_cal)*3;
                target_point3.cartPos.position.x=target_point.cartPos.position.x+delta_distance*std::sin(mid2target_angle_cal)*2;
                target_point4.cartPos.position.x=target_point.cartPos.position.x+delta_distance*std::sin(mid2target_angle_cal)*1;
                
                target_point1.cartPos.position.y=target_point.cartPos.position.y-delta_distance*std::cos(mid2target_angle_cal)*4;
                target_point2.cartPos.position.y=target_point.cartPos.position.y-delta_distance*std::cos(mid2target_angle_cal)*3;
                target_point3.cartPos.position.y=target_point.cartPos.position.y-delta_distance*std::cos(mid2target_angle_cal)*2;
                target_point4.cartPos.position.y=target_point.cartPos.position.y-delta_distance*std::cos(mid2target_angle_cal)*1;
            }
        }
        else
        {
            if (mid2target_angle_cal== 90.0*M_PI/180) //完全朝前生长的部分，要从上往下采摘
            {
                target_point1.cartPos.position.y=target_point.cartPos.position.y+delta_distance*std::cos(60*M_PI/180)*4;
                target_point2.cartPos.position.y=target_point.cartPos.position.y+delta_distance*std::cos(60*M_PI/180)*3;
                target_point3.cartPos.position.y=target_point.cartPos.position.y+delta_distance*std::cos(60*M_PI/180)*2;
                target_point4.cartPos.position.y=target_point.cartPos.position.y+delta_distance*std::cos(60*M_PI/180)*1;

                target_point1.cartPos.position.z=target_point.cartPos.position.z+delta_distance*std::sin(60*M_PI/180)*4;
                target_point2.cartPos.position.z=target_point.cartPos.position.z+delta_distance*std::sin(60*M_PI/180)*3;
                target_point3.cartPos.position.z=target_point.cartPos.position.z+delta_distance*std::sin(60*M_PI/180)*2;
                target_point4.cartPos.position.z=target_point.cartPos.position.z+delta_distance*std::sin(60*M_PI/180)*1;
            }
            else
            {
                target_point1.cartPos.position.x=target_point.cartPos.position.x-delta_distance*std::sin(mid2target_angle_cal)*4;
                target_point2.cartPos.position.x=target_point.cartPos.position.x-delta_distance*std::sin(mid2target_angle_cal)*3;
                target_point3.cartPos.position.x=target_point.cartPos.position.x-delta_distance*std::sin(mid2target_angle_cal)*2;
                target_point4.cartPos.position.x=target_point.cartPos.position.x-delta_distance*std::sin(mid2target_angle_cal)*1;
                
                target_point1.cartPos.position.y=target_point.cartPos.position.y+delta_distance*std::cos(mid2target_angle_cal)*4;
                target_point2.cartPos.position.y=target_point.cartPos.position.y+delta_distance*std::cos(mid2target_angle_cal)*3;
                target_point3.cartPos.position.y=target_point.cartPos.position.y+delta_distance*std::cos(mid2target_angle_cal)*2;
                target_point4.cartPos.position.y=target_point.cartPos.position.y+delta_distance*std::cos(mid2target_angle_cal)*1;
            }
        }
        std::vector<aubo_robot_namespace::wayPoint_S> temp_waypoints={target_point1, target_point2, target_point3, target_point4, target_point};
        
        // bool IK_success=true;
        aubo_robot_namespace::wayPoint_S start_points=intermediate_point;
        for(int i=0; i<5; i++)
        {
            if (IK_pos2joint(start_points, temp_waypoints[i]) == true) 
            {
                temp_waypoints_IKsuccess.push_back(temp_waypoints[i]);
                start_points=temp_waypoints[i];
            }
            else
            {
                if (i==4)
                {
                    response->result = 30007;
                    return;
                }
                continue;
            }                
        }

        RCLCPP_WARN(this->get_logger(), "go_target 新建:直线过渡点数量为: %ld",temp_waypoints_IKsuccess.size());
        //     intermediate_point.jointpos[0],

        // RCLCPP_WARN(this->get_logger(), "过渡点关节坐标为:%.3f,%.3f,%.3f,%.3f,%.3f,%.3f", 
        //     intermediate_point.jointpos[0],
        //     intermediate_point.jointpos[1],
        //     intermediate_point.jointpos[2],
        //     intermediate_point.jointpos[3],
        //     intermediate_point.jointpos[4],
        //     intermediate_point.jointpos[5]);
        // RCLCPP_WARN(this->get_logger(), "目标点关节坐标为:%.3f,%.3f,%.3f,%.3f,%.3f,%.3f", 
        //     target_point.jointpos[0],
        //     target_point.jointpos[1],
        //     target_point.jointpos[2],
        //     target_point.jointpos[3],
        //     target_point.jointpos[4],
        //     target_point.jointpos[5]);
        
        //统一调整直线过渡点1关节角度
        if(request->y>0)
        {
            for (int i=0; i<static_cast<int>(temp_waypoints_IKsuccess.size()); i++)
            {
                if (temp_waypoints_IKsuccess[i].jointpos[0]<10*M_PI/180)
                {
                    temp_waypoints_IKsuccess[i].jointpos[0]+=M_PI*2;
                }
            }
        }
        else
        {
            for (int i=0; i<static_cast<int>(temp_waypoints_IKsuccess.size()); i++)
            {
                if (temp_waypoints_IKsuccess[i].jointpos[0]>10*M_PI/180)
                {
                    temp_waypoints_IKsuccess[i].jointpos[0]-=M_PI*2;
                }
            }
        }

        //关节1运动到指定角度
        double joints5=pick_start_position_[4];
        if (request->x<0)
        {
            joints5 = (request->y > 0) ? -33*M_PI/180 : 33*M_PI/180;
            
        }
        midpoint_contract_position_[0]=intermediate_point.jointpos[0];
        midpoint_contract_position_[4] = joints5;


        if (only_go_target==false)
        {
            // 关节运动到过渡点（使用SDK标准接口）
            //设置提前到位距离模式
            robot_service_.robotServiceSetArrivalAheadDistanceMode(0.02);
            //获取关节速度
            // aubo_robot_namespace::JointVelcAccParam m_get_jointvelc;
            // robot_service_.robotServiceGetGlobalMoveJointMaxVelc(m_get_jointvelc);
            // RCLCPP_WARN(this->get_logger(), "关节速度1:%.3f,%.3f,%.3f,%.3f,%.3f,%.3f", 
            //     m_get_jointvelc.jointPara[0],
            //     m_get_jointvelc.jointPara[1],
            //     m_get_jointvelc.jointPara[2],
            //     m_get_jointvelc.jointPara[3],
            //     m_get_jointvelc.jointPara[4],
            //     m_get_jointvelc.jointPara[5]);             

            

            //获取关节速度
            // aubo_robot_namespace::JointVelcAccParam m_get_jointvelc;
            // robot_service_.robotServiceGetGlobalMoveJointMaxVelc(m_get_jointvelc);
            // RCLCPP_WARN(this->get_logger(), "关节速度2:%.3f,%.3f,%.3f,%.3f,%.3f,%.3f", 
            //     m_get_jointvelc.jointPara[0],
            //     m_get_jointvelc.jointPara[1],
            //     m_get_jointvelc.jointPara[2],
            //     m_get_jointvelc.jointPara[3],
            //     m_get_jointvelc.jointPara[4],
            //     m_get_jointvelc.jointPara[5]);
            trajectory.push_back(std::vector<double>(pick_start_position_, pick_start_position_ + 6));
            trajectory.push_back(std::vector<double>(midpoint_contract_position_, midpoint_contract_position_ + 6));
        }

        trajectory.push_back(std::vector<double>(intermediate_point.jointpos, intermediate_point.jointpos + 6));
        
        // 过渡点到目标点的碰撞检测
        if (request->start2mid!=true)
        {        
            for (int i = 0; i < static_cast<int>(temp_waypoints_IKsuccess.size()); i++)
            {
                trajectory.push_back(std::vector<double>(temp_waypoints_IKsuccess[i].jointpos, temp_waypoints_IKsuccess[i].jointpos + 6));
            }
        }
        bool is_safe = arm_checker_.checkTrajectory(trajectory, false);  // true表示显示详细信息
        RCLCPP_WARN(this->get_logger(), "去往目标点碰撞检测结果：%s", is_safe ? "安全" : "碰撞风险");
        if (request->start2mid!=true)
        {        
            for (int i=static_cast<int>(temp_waypoints_IKsuccess.size())-1; i>=0; i--)
            {
                trajectory.push_back(std::vector<double>(temp_waypoints_IKsuccess[i].jointpos, temp_waypoints_IKsuccess[i].jointpos + 6));
            }
            // 增加返回碰撞检测            
            trajectory.push_back(std::vector<double>(intermediate_point.jointpos, intermediate_point.jointpos + 6));
            trajectory.push_back(std::vector<double>(midpoint_contract_position_, midpoint_contract_position_ + 6));
            trajectory.push_back(std::vector<double>(pick_start_position_, pick_start_position_ + 6));
            trajectory.push_back(std::vector<double>(initial_position_, initial_position_ + 6));
        }
        bool is_safe_go_return = arm_checker_.checkTrajectory(trajectory, true);  // true表示显示详细信息
        RCLCPP_WARN(this->get_logger(), "去往目标点及返回碰撞检测结果：%s", is_safe_go_return ? "安全" : "碰撞风险");

        // 增加返回碰撞检测结束
        if (is_safe_go_return==false && only_go_target==false)
        {
            response->result = 31001;
            return;
        }
        else if (is_safe_go_return==false && only_go_target==true)
        {
            response->result = 31004;//需要特殊处理
            return;
        }
        //保存过渡点到成员变量

        midpoint_point_ = intermediate_point;        
        target_point_ = target_point;
        from_mid2target_waypoints_.clear();
        from_mid2target_waypoints_ = temp_waypoints_IKsuccess;
        last_midpoint_contract_position_[0]=midpoint_contract_position_[0];
        last_midpoint_contract_position_[4]=midpoint_contract_position_[4];

        

        //要开始运动了！！！！！！！！！！！robotServiceSetArrivalAheadDistanceMode
        robot_service_.robotServiceSetArrivalAheadDistanceMode(0.02);
        if (only_go_target==false)
        {
            // 先运动到采摘时的起始位置
            ret = robot_service_.robotServiceJointMove(pick_start_position_, true);
            logMoveResult(ret, "去往采摘点:去往了pick_start_position_成功", "去往采摘点:去往pick_start_position_失败");//打印动作结果

            //再运动到过渡准备点
            ret = robot_service_.robotServiceJointMove(last_midpoint_contract_position_, true);
            logMoveResult(ret, "去往采摘点:去往了last_midpoint_contract_position_成功", "去往采摘点:去往last_midpoint_contract_position_失败");//打印动作结果
        }
        
        ret = robot_service_.robotServiceJointMove(intermediate_point.jointpos, true);        
        logMoveResult(ret, "去往采摘点:去往了intermediate_point成功", "去往采摘点:去往intermediate_point失败");//打印动作结果

        // ret=0;
        if(ret != aubo_robot_namespace::InterfaceCallSuccCode && ret!=10023) {
            // response->result = ret;
            RCLCPP_ERROR(this->get_logger(), "关节运动至过渡点失败，返回值:%d", ret);
            // ret = robot_service_.robotServiceCollisionRecover();
            //变为收缩状态，主程序会让其直接回到初始位置

            ret = robot_service_.robotServiceJointMove(last_midpoint_contract_position_, true);
            logMoveResult(ret, 
                "去往采摘点:运动至过渡点失败,补救措施:去往了last_midpoint_contract_position_成功", 
                "去往采摘点:运动至过渡点失败,补救措施:去往了last_midpoint_contract_position_失败");//打印动作结果
            ret = robot_service_.robotServiceJointMove(pick_start_position_, true);
            logMoveResult(ret, 
                "去往采摘点:运动至过渡点失败,补救措施:去往了pick_start_position_成功", 
                "去往采摘点:运动至过渡点失败,补救措施:去往了pick_start_position_失败");//打印动作结果
            response->result = 30002;
            return;
        }
        else
        {
            //到达了中间点，让手爪反馈系统开始工作
            RCLCPP_INFO(this->get_logger(), "---------------------go goal 到达了中间点-----------------");
            xc_interfaces::msg::MidpointReached midpoint_msg;
            midpoint_msg.reached = true;
            // RCLCPP_INFO(this->get_logger(), "go goal 到达了中间点:%d",midpoint_msg.reached);
            midpoint_publisher_->publish(midpoint_msg);
            std::this_thread::sleep_for(0.5s);
            midpoint_publisher_->publish(midpoint_msg);
            // RCLCPP_INFO(this->get_logger(), "go goal我已经发送了中间点到达消息:");
        }
    
        // RCLCPP_WARN(this->get_logger(), "目标点关节坐标为:%.3f,%.3f,%.3f,%.3f,%.3f,%.3f", 
        // target_point.jointpos[0],
        // target_point.jointpos[1],
        // target_point.jointpos[2],
        // target_point.jointpos[3],
        // target_point.jointpos[4],
        // target_point.jointpos[5]);
        
        

        if(request->start2mid != true)
        {
            
        
            
            
            for(int i=0; i<6; i++) {
                jointMaxVelc.jointPara[i] = arm_joints_low_speed_ * M_PI/180; // 20度/秒 → 0.33 rad/s
            }
            robot_service_.robotServiceSetGlobalMoveJointMaxVelc(jointMaxVelc);
            // robot_service_.robotServiceSetGlobalMoveJointMaxAcc(jointMaxVelc); 

            //设置直线运动的最大速度
            robot_service_.robotServiceSetGlobalMoveEndMaxLineVelc(0.01);
            // robot_service_.robotServiceSetGlobalMoveEndMaxLineAcc(0.3);
            
            //精准到达目标点

            RCLCPP_WARN(this->get_logger(), "go_target 运动 :直线过渡点数量为:%ld",temp_waypoints_IKsuccess.size());
            int num_=static_cast<int>(temp_waypoints_IKsuccess.size());
            for (int i=0; i<static_cast<int>(temp_waypoints_IKsuccess.size()); i++)
            {
                if(i>=num_-2)
                {
                    robot_service_.robotServiceSetNoArrivalAhead(); //快到目标点就取消提前到位
                }
                else
                {
                    robot_service_.robotServiceSetArrivalAheadDistanceMode(0.02);
                }
                
                ret = robot_service_.robotServiceJointMove(temp_waypoints_IKsuccess[i].jointpos, true);
                std::string success_msg = "去往采摘点:temp_waypoints_IKsuccess[" + std::to_string(i+1) + "]成功";
                std::string fail_msg = "去往采摘点:temp_waypoints_IKsuccess[" + std::to_string(i+1) + "]失败";            
                logMoveResult(ret, 
                    success_msg, 
                    fail_msg);//打印动作结果
            }

            // ret=0;
            if(ret != aubo_robot_namespace::InterfaceCallSuccCode) 
            {
                // response->result = ret;
                // RCLCPP_ERROR(this->get_logger(), "从过渡点运动至目标点失败，返回值:%d", ret);             
                RCLCPP_ERROR(this->get_logger(), "从过渡点轴动运动至目标点失败了，返回值:%d", ret);
                
                for (int i=static_cast<int>(temp_waypoints_IKsuccess.size())-1; i>=0; i--)
                {
                    ret = robot_service_.robotServiceJointMove(temp_waypoints_IKsuccess[i].jointpos, true);
                    std::string success_msg = "去往采摘点失败补救措施:返回temp_waypoints_IKsuccess[" + std::to_string(i+1) + "]成功";
                    std::string fail_msg = "去往采摘点失败补救措施:返回temp_waypoints_IKsuccess[" + std::to_string(i+1) + "]失败";  
                    logMoveResult(ret, 
                        success_msg, 
                        fail_msg);//打印动作结果
                }
                
                ret = robot_service_.robotServiceJointMove(intermediate_point.jointpos, true); 
                logMoveResult(ret, 
                    "去往采摘点失败补救措施:返回 intermediate_point.jointpos 成功", 
                    "去往采摘点失败补救措施:返回 intermediate_point.jointpos 失败");//打印动作结果
                
                ret = robot_service_.robotServiceJointMove(last_midpoint_contract_position_, true); 
                logMoveResult(ret, 
                    "去往采摘点失败补救措施:返回 last_midpoint_contract_position_ 成功", 
                    "去往采摘点失败补救措施:返回 last_midpoint_contract_position_ 失败");//打印动作结果
                
                ret = robot_service_.robotServiceJointMove(pick_start_position_, true); 
                logMoveResult(ret, 
                    "去往采摘点失败补救措施:返回 pick_start_position_ 成功", 
                    "去往采摘点失败补救措施:返回 pick_start_position_ 失败");//打印动作结果
                response->result = 30003;
                return;
            }
            RCLCPP_WARN(this->get_logger(), "去往采摘点:我到达了目标点"); 
        }
        else
        {
            RCLCPP_WARN(this->get_logger(), "分为两个阶段执行，我已到达了过渡点，等待精准定位"); 
        }          

        response->result = 0;
    }


    bool IK_pos2joint(const aubo_robot_namespace::wayPoint_S& start_point, aubo_robot_namespace::wayPoint_S& point)
    {
        double original_intermediate_y = point.cartPos.position.y;
        int intermediate_attempts = 0;

        while (true) {
            // 逆解计算（遵循example_5.cpp中的调用方式）
            aubo_robot_namespace::wayPoint_S ik_result;
            // RCLCPP_WARN(this->get_logger(), "第%d次尝试值是%.3f*%.3f*%.3f", 
            // intermediate_attempts,
            // point.cartPos.position.x,
            // point.cartPos.position.y,
            // point.cartPos.position.z);
            int ret = robot_service_.robotServiceRobotIk(
                start_point.jointpos, 
                point.cartPos.position,
                point.orientation,
                ik_result
            );
            if(ret == aubo_robot_namespace::InterfaceCallSuccCode) {
                point = ik_result;
                print_waypoint_info("逆解后的点", point);
                return true;                
            }

            if (intermediate_attempts >= 20) {
                RCLCPP_ERROR(this->get_logger(), "逆解失败，达到最大尝试次数");
                RCLCPP_ERROR(this->get_logger(), "最后的尝试值是%.3f*%.3f*%.3f", 
                    point.cartPos.position.x,
                    point.cartPos.position.y,
                    point.cartPos.position.z);                
                                
                return false;
            }

            // 调整Y值
            if (intermediate_attempts % 2 == 0) {
                point.cartPos.position.y = original_intermediate_y - 0.01*int(intermediate_attempts/2);
            } else {
                point.cartPos.position.y = original_intermediate_y + 0.01*int(intermediate_attempts/2);
            }
            intermediate_attempts++;
        }
    }
    
    
    

    void handle_return_request_1(const std::shared_ptr<xc_interfaces::srv::MoveToPose::Request> request,
                        std::shared_ptr<xc_interfaces::srv::MoveToPose::Response> response)
    {
        //一开始就发送取消中间点到达消息  避免干涉
        xc_interfaces::msg::MidpointReached midpoint_msg;
        midpoint_msg.reached = false;
        RCLCPP_INFO(this->get_logger(), "返回时到达了中间点:%d",midpoint_msg.reached);
        midpoint_publisher_->publish(midpoint_msg);
        RCLCPP_INFO(this->get_logger(), "我发送了取消中间点到达消息:");
        midpoint_publisher_->publish(midpoint_msg);
        RCLCPP_INFO(this->get_logger(), "接收到的数据:%d",request->start2mid);

        std::lock_guard<std::mutex> lock(service_mutex_);
        // 创建过渡点（保持原有逻辑）
        aubo_robot_namespace::wayPoint_S intermediate_point;
        intermediate_point = midpoint_point_;
        std::vector<aubo_robot_namespace::wayPoint_S> temp_line_waypoints;
        temp_line_waypoints=from_mid2target_waypoints_;

        

        // 示例：设置所有关节的最大速度为30度/秒（转换为弧度）
        aubo_robot_namespace::JointVelcAccParam jointMaxVelc;
        for(int i=0; i<6; i++) {
        jointMaxVelc.jointPara[i] = arm_joints_speed_ * M_PI/180; // 30度/秒 → 0.5236 rad/s
        }
        robot_service_.robotServiceSetGlobalMoveJointMaxVelc(jointMaxVelc);
        robot_service_.robotServiceSetGlobalMoveJointMaxAcc(jointMaxVelc);  

        //设置直线运动的最大速度
        robot_service_.robotServiceSetGlobalMoveEndMaxLineVelc(line_speed_);
        //设置直线运动的最大加速度
        robot_service_.robotServiceSetGlobalMoveEndMaxLineAcc(line_speed_);
        

        // 直线运动到过渡点
        int ret=0;

        RCLCPP_WARN(this->get_logger(), "return_box :直线过渡点数量为:%ld",temp_line_waypoints.size());
        
        if(request->mid_to_target==false)
        {
            for (int i = static_cast<int>(temp_line_waypoints.size())-1; i >= 0; i--)
            {
                ret = robot_service_.robotServiceJointMove(temp_line_waypoints[i].jointpos, true);
                std::string success_msg = "返回果箱:去往 temp_line_waypoints[" + std::to_string(i+1) + "] 成功";
                std::string fail_msg = "返回果箱:去往 temp_line_waypoints[" + std::to_string(i+1) + "] 失败";  
                logMoveResult(ret, 
                    success_msg, 
                    fail_msg);//打印动作结果
            }
        }
        ret = robot_service_.robotServiceJointMove(intermediate_point.jointpos, true);
        logMoveResult(ret, 
            "返回果箱:去往 intermediate_point 成功", 
            "返回果箱:去往 intermediate_point 失败");//打印动作结果
        //设置提前到位距离模式
        robot_service_.robotServiceSetArrivalAheadDistanceMode(0.02);

        if(ret != aubo_robot_namespace::InterfaceCallSuccCode) 
        {
            ret = robot_service_.robotServiceJointMove(last_midpoint_contract_position_, true);
            logMoveResult(ret, 
            "返回果箱：返回intermediate_point失败补救:去往 last_midpoint_contract_position_ 成功", 
            "返回果箱:返回intermediate_point失败补救:去往 last_midpoint_contract_position_ 失败");//打印动作结果
            ret = robot_service_.robotServiceJointMove(pick_start_position_, true);
            logMoveResult(ret, 
            "返回果箱:返回intermediate_point失败补救:去往 pick_start_position_ 成功", 
            "返回果箱:返回intermediate_point失败补救:去往 pick_start_position_ 失败");//打印动作结果            
            response->result = 30004;
            return; 
        }
        else
        {
            //到达了中间点，让手爪反馈系统开始工作
            RCLCPP_INFO(this->get_logger(), "---------------------返回时到达了中间点-直线方式----------------");
        }
        // double joints5=pick_start_position_[4];
        // if (request->x<0)
        // {
        //     joints5 = (request->y > 0) ? -33*M_PI/180 : 33*M_PI/180;
        // }
        // double temp_pick_intermediate_joint_[6] = {intermediate_point.jointpos[0], pick_start_position_[1], pick_start_position_[2], pick_start_position_[3], joints5, pick_start_position_[5]};
        
        ret = robot_service_.robotServiceJointMove(last_midpoint_contract_position_, true);
        logMoveResult(ret, 
            "返回果箱:去往 last_midpoint_contract_position_ 成功", 
            "返回果箱:去往 last_midpoint_contract_position_ 失败");//打印动作结果
        
        ret = robot_service_.robotServiceJointMove(pick_start_position_, true); 
        logMoveResult(ret, 
            "返回果箱:去往 pick_start_position_ 成功", 
            "返回果箱:去往 pick_start_position_ 失败");//打印动作结果                

        //取消提前到位
        robot_service_.robotServiceSetNoArrivalAhead();
        //返回果箱位置
        ret = robot_service_.robotServiceJointMove(initial_position_, true);

        if(ret != aubo_robot_namespace::InterfaceCallSuccCode && ret!=10023) {
            RCLCPP_WARN(this->get_logger(), "返回果箱进程：返回原点时发生错误，错误码:%d",ret);
            response->result = 30005;
            return;
           
        }
        else
        {
            RCLCPP_WARN(this->get_logger(), "返回果箱进程:我到达了果箱位置");
            response->result = 0;
        }
        RCLCPP_ERROR(this->get_logger(), "------------------------------------------------------"); 
    }
    
    
    void handle_target_return_midpoint_request_1(const std::shared_ptr<xc_interfaces::srv::MoveToPose::Request> request,
                         std::shared_ptr<xc_interfaces::srv::MoveToPose::Response> response)
    {
        // 创建过渡点（保持原有逻辑）
        //一开始就发送取消中间点到达消息  避免干涉
        xc_interfaces::msg::MidpointReached midpoint_msg;
        midpoint_msg.reached = false;
        RCLCPP_INFO(this->get_logger(), "返回时到达了中间点:%d",midpoint_msg.reached);
        midpoint_publisher_->publish(midpoint_msg);
        RCLCPP_INFO(this->get_logger(), "我发送了取消中间点到达消息:");
        midpoint_publisher_->publish(midpoint_msg);
        std::lock_guard<std::mutex> lock(service_mutex_);
        // 创建过渡点（保持原有逻辑）
        aubo_robot_namespace::wayPoint_S intermediate_point;
        intermediate_point = midpoint_point_;
        std::vector<aubo_robot_namespace::wayPoint_S> temp_line_waypoints;
        temp_line_waypoints=from_mid2target_waypoints_;    

        // 示例：设置所有关节的最大速度为30度/秒（转换为弧度）
        aubo_robot_namespace::JointVelcAccParam jointMaxVelc;
        for(int i=0; i<6; i++) 
        {
            jointMaxVelc.jointPara[i] = arm_joints_speed_ * M_PI/180; // 30度/秒 → 0.5236 rad/s
        }
        robot_service_.robotServiceSetGlobalMoveJointMaxVelc(jointMaxVelc);
        robot_service_.robotServiceSetGlobalMoveJointMaxAcc(jointMaxVelc); 
    

        //设置直线运动的最大速度
        robot_service_.robotServiceSetGlobalMoveEndMaxLineVelc(line_speed_);
        robot_service_.robotServiceSetGlobalMoveEndMaxLineAcc(line_speed_);


        // 直线运动退到过渡点
        int ret=0;
        RCLCPP_WARN(this->get_logger(), "return_midpoints :直线过渡点数量为:%ld",temp_line_waypoints.size());
        int line_point_arrive_count=0;
        for (int i = static_cast<int>(temp_line_waypoints.size())-1 ; i >=0; i--)
        {
            ret=robot_service_.robotServiceJointMove(temp_line_waypoints[i].jointpos, true);
            line_point_arrive_count++;
            std::string success_msg="return_midpoints :返回过渡点:"+std::to_string(line_point_arrive_count)+"成功";
            std::string fail_msg="return_midpoints :返回过渡点:"+std::to_string(line_point_arrive_count)+"失败";
            logMoveResult(ret, 
                success_msg, 
                fail_msg);//打印动作结果
            // RCLCPP_WARN(this->get_logger(), "return_midpoints :已到达过渡点数量:%d",line_point_arrive_count);

        }

        ret=robot_service_.robotServiceJointMove(intermediate_point.jointpos, true);
        if(ret != aubo_robot_namespace::InterfaceCallSuccCode && ret!=10023) 
        {
            RCLCPP_ERROR(this->get_logger(), "尝试使用轴动方式返回过渡点时发生错误，错误码:%d",ret); 
            ret = robot_service_.robotServiceJointMove(last_midpoint_contract_position_, true);
            logMoveResult(ret, 
                "返回过渡点:返回过渡点:last_midpoint_contract_position_ 成功", 
                "返回过渡点:返回过渡点:last_midpoint_contract_position_ 失败");//打印动作结果                
            
            ret = robot_service_.robotServiceJointMove(pick_start_position_, true);   
            logMoveResult(ret, 
                "返回过渡点:返回过渡点:pick_start_position_ 成功", 
                "返回过渡点:返回过渡点:pick_start_position_ 失败");//打印动作结果    
                        
            response->result = 30006;

            return;

        }
        else
        {
            RCLCPP_INFO(this->get_logger(), "return_midpoints：返回过渡点成功"); 
            response->result = 0;
        }
    }

    void create_waypoint(const std::shared_ptr<xc_interfaces::srv::MoveToPose::Request> req,
                         aubo_robot_namespace::wayPoint_S &waypoint, bool is_intermediate , double mid2target_angle_cal=0.0)
    {
        // 设置位置
        waypoint.cartPos.position.x = req->x;
        waypoint.cartPos.position.y = req->y;
        waypoint.cartPos.position.z = req->z;
        if (is_intermediate)
        {
            if(req->y > 0.0)
            {
                if(mid2target_angle_cal == 90.0*M_PI/180)
                {
                    waypoint.cartPos.position.y = req->y-0.1*cos(mid2target_angle_cal);
                    waypoint.cartPos.position.z = req->z+0.1*sin(mid2target_angle_cal);
                }
                else
                {
                    waypoint.cartPos.position.x = req->x+0.1*sin(mid2target_angle_cal);
                    waypoint.cartPos.position.y = req->y-0.1*cos(mid2target_angle_cal);
                }

            }
            else
            {
                if(mid2target_angle_cal == 90.0*M_PI/180)
                {
                    waypoint.cartPos.position.y = req->y+0.1*sin(mid2target_angle_cal);
                    waypoint.cartPos.position.z = req->z+0.1*sin(mid2target_angle_cal);
                }
                else
                {
                    waypoint.cartPos.position.x = req->x-0.1*sin(mid2target_angle_cal);
                    waypoint.cartPos.position.y = req->y+0.1*cos(mid2target_angle_cal);
                }
            }            
            // 处理过渡点Y坐标
            // if ( abs(waypoint.cartPos.position.y) >  midpoint_min_y)
            // {
            //     waypoint.cartPos.position.y = (req->y > 0) ? midpoint_min_y : -midpoint_min_y; // 单位米，假设250mm转换为0.25m
            // }
        }
        else
        {
            waypoint.cartPos.position.y = req->y;
        }

        // 欧拉角转四元数
        tf2::Quaternion q;
        if(mid2target_angle_cal == 90.0*M_PI/180 || mid2target_angle_cal == -90.0*M_PI/180)
        {
            if (req->y >0)
            {
                q.setRPY((90+60)*M_PI/180, 0.0*M_PI/180,180*M_PI/180);
            }
            else
            {
                q.setRPY((90+60)*M_PI/180, 0.0*M_PI/180,0.0*M_PI/180);
            }
            
        }
        else
        {
            q.setRPY(req->rx, req->ry, req->rz+mid2target_angle_cal);
        }

        // 工具坐标系目标位姿 → TCP(法兰盘)位姿
        // T_base_flange = T_base_tool · T_flange_tool⁻¹
        //   q_flange = q_tool · q_off⁻¹
        //   p_flange = p_tool − R_flange · p_off
        if (has_tool_offset_)
        {
            tf2::Vector3 p_tool(waypoint.cartPos.position.x,
                                waypoint.cartPos.position.y,
                                waypoint.cartPos.position.z);
            q = q * tool_offset_q_.inverse();              // R_flange = R_tool · R_off⁻¹
            tf2::Matrix3x3 R_flange(q);
            tf2::Vector3 p_off(tool_offset_pos_[0], tool_offset_pos_[1], tool_offset_pos_[2]);
            tf2::Vector3 p_flange = p_tool - R_flange * p_off;
            waypoint.cartPos.position.x = p_flange.x();
            waypoint.cartPos.position.y = p_flange.y();
            waypoint.cartPos.position.z = p_flange.z();
        }

        waypoint.orientation.w = q.w();
        waypoint.orientation.x = q.x();
        waypoint.orientation.y = q.y();
        waypoint.orientation.z = q.z();
    }

    void print_waypoint_info(const std::string &name, const aubo_robot_namespace::wayPoint_S &wp)
    {
        RCLCPP_INFO(this->get_logger(), "%s位姿信息:", name.c_str());
        RCLCPP_INFO(this->get_logger(), "位置: x=%.3f m, y=%.3f m, z=%.3f m",
                    wp.cartPos.position.x,
                    wp.cartPos.position.y,
                    wp.cartPos.position.z);

        // 将四元数转换为欧拉角
        tf2::Quaternion q(
            wp.orientation.x,
            wp.orientation.y,
            wp.orientation.z,
            wp.orientation.w);
        tf2::Matrix3x3 m(q);
        double roll, pitch, yaw;
        m.getRPY(roll, pitch, yaw);

        // RCLCPP_INFO(this->get_logger(), "四元数: w=%.3f, x=%.3f, y=%.3f, z=%.3f",
        //             wp.orientation.w, wp.orientation.x,
        //             wp.orientation.y, wp.orientation.z);
        // RCLCPP_INFO(this->get_logger(), "欧拉角: roll=%.3f rad, pitch=%.3f rad, yaw=%.3f rad\n",
        //             roll, pitch, yaw);
        RCLCPP_INFO(this->get_logger(), "轴角度: [%.3f * %.3f *%.3f *%.3f *%.3f *%.3f\n",
                    wp.jointpos[0], wp.jointpos[1], wp.jointpos[2], wp.jointpos[3], wp.jointpos[4], wp.jointpos[5]);
    }

    void handle_joints_request(
        const std::shared_ptr<xc_interfaces::srv::MoveJoints::Request> request,
        std::shared_ptr<xc_interfaces::srv::MoveJoints::Response> response)
    {
        std::lock_guard<std::mutex> lock(service_mutex_);

        // 将请求的关节值复制到数组
        //  RCLCPP_INFO(this->get_logger(), "Received move request for joints: [%.2f, %.2f, %.2f, %.2f, %.2f, %.2f]",
        //        request->joints[0], request->joints[1], request->joints[2], 
        //        request->joints[3], request->joints[4], request->joints[5]);
        double joint_angles[6];
        for(int i = 0; i < 6; ++i) {
            joint_angles[i] = request->joints[i];
        }



        // 示例：设置所有关节的最大速度为30度/秒（转换为弧度）
        aubo_robot_namespace::JointVelcAccParam jointMaxVelc;
        for(int i=0; i<6; i++) {
        jointMaxVelc.jointPara[i] = arm_joints_speed_ * M_PI/180; // 30度/秒 → 0.5236 rad/s
        // RCLCPP_INFO(this->get_logger(), "-------->>>>>>>---------: %f",jointMaxVelc.jointPara[i]);
        }
        // std::print("jointMaxVelc", jointMaxVelc.jointPara[0]);
        // RCLCPP_INFO(this->get_logger(), "--------jointMaxVelc.jointPara: %f,%f,%f,%f,%f,%f", 
        //             jointMaxVelc.jointPara[0],
        //             jointMaxVelc.jointPara[1],
        //             jointMaxVelc.jointPara[2],
        //             jointMaxVelc.jointPara[3],
        //             jointMaxVelc.jointPara[4],
        //             jointMaxVelc.jointPara[5]);
        robot_service_.robotServiceSetGlobalMoveJointMaxVelc(jointMaxVelc);
        robot_service_.robotServiceSetGlobalMoveJointMaxAcc(jointMaxVelc);



        int ret = robot_service_.robotServiceJointMove(joint_angles, true);

        if(ret != aubo_robot_namespace::InterfaceCallSuccCode && ret!=10023) {
            RCLCPP_WARN(this->get_logger(), "关节运动失败! 错误码: %d", ret);
            RCLCPP_INFO(this->get_logger(), "关节运动失败! 延时1s后我再尝试一次！！");
            std::this_thread::sleep_for(1s);
            ret = robot_service_.robotServiceJointMove(joint_angles, true);
            if(ret != aubo_robot_namespace::InterfaceCallSuccCode && ret!=10023) {
                RCLCPP_ERROR(this->get_logger(), "再次尝试关节运动失败! 错误码: %d", ret);
                response->result = ret;
                return;                
            }



        }
        // 设置响应结果
        response->result = ret;
        
    //    response->result = 0;
    //    RCLCPP_INFO(this->get_logger(), "----------------");
       
    }


    // 成员变量
    ServiceInterface robot_service_;
    ServiceInterface robot_return_service_;    
    ServiceInterface robot_pub_;
    std::mutex service_mutex_;
    std::mutex pub_mutex_;
    Collision_Joint_Checker arm_checker_;//机械臂路径碰撞与关节限制检查类
    rclcpp::Service<xc_interfaces::srv::MoveToPose>::SharedPtr service_;
    rclcpp::Service<xc_interfaces::srv::MoveToPose>::SharedPtr return_service_;
    rclcpp::Service<xc_interfaces::srv::MoveToPose>::SharedPtr target_return_midpoint_service_ ;    
    rclcpp::Publisher<xc_interfaces::msg::ArmStatus>::SharedPtr publisher_;
    rclcpp::Publisher<xc_interfaces::msg::MidpointReached>::SharedPtr midpoint_publisher_;

    rclcpp::Publisher<std_msgs::msg::UInt8>::SharedPtr x1_status_publisher_;

    rclcpp::Service<xc_interfaces::srv::MoveJoints>::SharedPtr move_joints_service_;
    rclcpp::TimerBase::SharedPtr timer_;
    float arm_joints_speed_;//单位是 度/s
    float arm_joints_low_speed_;//单位是 度/s
    double line_speed_;//单位是m/s
    double initial_position_[6] ;
    double pick_start_position_[6] ;//机械臂内缩时的起始位置（比果箱位置X略小）
    double midpoint_contract_position_[6] ;//相对于中间点的内缩位置（比果箱位置X略小）  实时更新
    double last_midpoint_contract_position_[6] ;//相对于中间点的内缩位置（比果箱位置X略小）  确认无误才更新
    double last_x1_status_;

    float midpoint_min_y;//机械臂中间点的y值
    
    aubo_robot_namespace::wayPoint_S midpoint_point_;
    aubo_robot_namespace::wayPoint_S target_point_;
    std::vector<aubo_robot_namespace::wayPoint_S> from_mid2target_waypoints_;

    // 回调组
    rclcpp::CallbackGroup::SharedPtr MutuallyExclusive_callback_group_;
    rclcpp::CallbackGroup::SharedPtr reentrant_callback_group_;

    //控制是否发送末端传感器信息
    bool send_end_effector_sensor_info_;

    //工具坐标系相对法兰盘(TCP)的运动学偏移（从SDK读取）
    double tool_offset_pos_[3];                              //工具偏移位置 x,y,z
    tf2::Quaternion tool_offset_q_;                          //工具偏移姿态四元数
    bool has_tool_offset_;                                   //是否成功读取到工具偏移
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::executors::MultiThreadedExecutor executor;
    auto node = std::make_shared<MotionService>();
    executor.add_node(node);
    executor.spin();
    rclcpp::shutdown();
    return 0;
}
