#include <rclcpp/rclcpp.hpp>
#include <xc_interfaces/srv/move_to_pose.hpp>
#include <xc_interfaces/srv/move_joints.hpp>
#include <chrono>
#include <thread>
#include <cmath>

using namespace std::chrono_literals;

class ContinuousMotionClient : public rclcpp::Node
{
public:
    ContinuousMotionClient() : Node("continuous_motion_client")
    {
        // 创建服务客户端
        move_to_pose_client_ = this->create_client<xc_interfaces::srv::MoveToPose>("move_to_pose");
        move_joints_client_ = this->create_client<xc_interfaces::srv::MoveJoints>("move_joints");
        
        // 等待服务可用
        while (!move_to_pose_client_->wait_for_service(1s)) {
            if (!rclcpp::ok()) {
                RCLCPP_ERROR(this->get_logger(), "Interrupted while waiting for move_to_pose service. Exiting.");
                return;
            }
            RCLCPP_INFO(this->get_logger(), "move_to_pose service not available, waiting again...");
        }
        
        while (!move_joints_client_->wait_for_service(1s)) {
            if (!rclcpp::ok()) {
                RCLCPP_ERROR(this->get_logger(), "Interrupted while waiting for move_joints service. Exiting.");
                return;
            }
            RCLCPP_INFO(this->get_logger(), "move_joints service not available, waiting again...");
        }
        
        // 初始化执行标志
        is_executing_ = false;
        
        // 启动顺序执行循环，使用定时器而不是独立线程
        // 设置较长的定时器间隔，确保服务调用有足够时间完成
        motion_timer_ = this->create_wall_timer(
            5s, std::bind(&ContinuousMotionClient::motion_sequence_loop, this));
    }

private:
    void motion_sequence_loop()
    {
        // 检查是否已经在执行运动序列
        if (is_executing_) {
            RCLCPP_WARN(this->get_logger(), "Previous motion sequence still executing, skipping this cycle");
            return;
        }
        
        // 设置执行标志
        is_executing_ = true;
        
        // 开始执行运动序列
        execute_motion_sequence_step(0); // 从第一步开始
    }
    
    void execute_motion_sequence_step(int step)
    {
        switch (step) {
            case 0: // 首先调用move_joints服务回到初始位置
                move_to_initial_position_async([this](bool success) {
                    if (!success) {
                        RCLCPP_ERROR(this->get_logger(), "Failed to move to initial position");
                        is_executing_ = false;
                        return;
                    }
                    // 继续下一步
                    execute_motion_sequence_step(1);
                });
                break;
                
            case 1: // 然后调用move_to_pose服务
                move_to_target_pose_async([this](bool success) {
                    if (!success) {
                        RCLCPP_ERROR(this->get_logger(), "Failed to move to target pose");
                        is_executing_ = false;
                        return;
                    }
                    // 继续下一步
                    execute_motion_sequence_step(2);
                });
                break;
                
            case 2: // 最后再次调用move_joints服务回到初始位置
                move_to_initial_position_async([this](bool success) {
                    if (!success) {
                        RCLCPP_ERROR(this->get_logger(), "Failed to move back to initial position");
                        is_executing_ = false;
                        return;
                    }
                    
                    RCLCPP_INFO(this->get_logger(), "Completed one motion sequence successfully");
                    is_executing_ = false;
                });
                break;
        }
    }
    
    void move_to_initial_position_async(std::function<void(bool)> callback)
    {
        // 调用move_joints服务回到初始位置
        auto move_joints_request = std::make_shared<xc_interfaces::srv::MoveJoints::Request>();
        // 设置初始关节角度值 (示例值)
        move_joints_request->joints[0] = 10.32 * M_PI / 180.0;
        move_joints_request->joints[1] = 15.37 * M_PI / 180.0;
        move_joints_request->joints[2] = 134.05 * M_PI / 180.0;
        move_joints_request->joints[3] = 117.05 * M_PI / 180.0;
        move_joints_request->joints[4] = 86.63 * M_PI / 180.0;
        move_joints_request->joints[5] = 0.0 * M_PI / 180.0;
        
        RCLCPP_INFO(this->get_logger(), "Moving to initial joint positions");
        
        // 异步发送请求
        auto future = move_joints_client_->async_send_request(
            move_joints_request,
            [this, callback](rclcpp::Client<xc_interfaces::srv::MoveJoints>::SharedFuture future) {
                // 处理响应
                auto response = future.get();
                if (response->result == 0) {
                    RCLCPP_INFO(this->get_logger(), "Move to initial position succeeded");
                    callback(true);
                } else {
                    RCLCPP_ERROR(this->get_logger(), "Move to initial position failed with result: %d", response->result);
                    callback(false);
                }
            }
        );
    }
    
    void move_to_target_pose_async(std::function<void(bool)> callback)
    {
        // 调用move_to_pose服务
        auto move_to_pose_request = std::make_shared<xc_interfaces::srv::MoveToPose::Request>();
        move_to_pose_request->x = 0.139;
        move_to_pose_request->y = -0.568;
        move_to_pose_request->z = 0.41;
        move_to_pose_request->rx = 79.23* M_PI / 180.0;
        move_to_pose_request->ry = -6.57* M_PI / 180.0;
        move_to_pose_request->rz = 5.58* M_PI / 180.0;
        
        RCLCPP_INFO(this->get_logger(), "Moving to target pose: (%.3f, %.3f, %.3f)", 
                   move_to_pose_request->x, move_to_pose_request->y, move_to_pose_request->z);
        
        // 异步发送请求
        auto future = move_to_pose_client_->async_send_request(
            move_to_pose_request,
            [this, callback](rclcpp::Client<xc_interfaces::srv::MoveToPose>::SharedFuture future) {
                // 处理响应
                auto response = future.get();
                if (response->result == 0) {
                    RCLCPP_INFO(this->get_logger(), "Move to target pose succeeded");
                    callback(true);
                } else {
                    RCLCPP_ERROR(this->get_logger(), "Move to target pose failed with result: %d", response->result);
                    callback(false);
                }
            }
        );
    }
    
    rclcpp::Client<xc_interfaces::srv::MoveToPose>::SharedPtr move_to_pose_client_;
    rclcpp::Client<xc_interfaces::srv::MoveJoints>::SharedPtr move_joints_client_;
    rclcpp::TimerBase::SharedPtr motion_timer_;
    bool is_executing_;  // 添加执行标志
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<ContinuousMotionClient>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}