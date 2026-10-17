// this->declare_parameter("folder_path", "/home/xc/code/test/ros2/robot_ws/src/xc_aubo_arm_control/fake_joints");
// this->declare_parameter("initial_position", std::vector<double>{10.32, 15.37, 134.05, 117.05, 86.63, 0.0});
#include <rclcpp/rclcpp.hpp>
#include <xc_interfaces/srv/move_joints.hpp>
#include <filesystem>
#include <fstream>
#include <vector>
#include <array>
#include <cmath>

using namespace std::chrono_literals;

class JointTrajectoryPlayer : public rclcpp::Node
{
public:
    JointTrajectoryPlayer() : Node("joint_trajectory_player")
    {
        // 初始化服务客户端
        client_ = this->create_client<xc_interfaces::srv::MoveJoints>("move_joints");
        
        // 声明参数
        this->declare_parameter("folder_path", "/home/xc/code/test/ros2/robot_ws/src/xc_aubo_arm_control/fake_joints");//工控机
        // this->declare_parameter("folder_path", "/home/xucan/code/test/robot_ws/src/xc_aubo_arm_control/fake_joints");//徐灿WSL
        this->declare_parameter("initial_position", std::vector<double>{10.32, 15.37, 134.05, 117.05, 86.63, 0.0});
        
        // 获取参数并转换为弧度
        folder_path_ = this->get_parameter("folder_path").as_string();
        auto initial_pos_vec = this->get_parameter("initial_position").as_double_array();
        
        // 将初始位置从角度转换为弧度
        for(size_t i = 0; i < 6; ++i) {
            initial_position_[i] = degrees_to_radians(initial_pos_vec[i]);
        }
        
        // 加载轨迹文件
        load_trajectory_files();
        
        // 直接在主线程中执行轨迹
        execute_trajectories();
    }

private:
    // 角度转弧度函数
    double degrees_to_radians(double degrees) {
        return degrees * M_PI / 180.0;
    }
    
    void load_trajectory_files()
    {
        trajectory_files_.clear();
        
        // 遍历文件夹查找joints*.txt文件
        for(const auto& entry : std::filesystem::directory_iterator(folder_path_)) {
            if(entry.path().filename().string().find("joints") == 0 && 
               entry.path().extension() == ".txt") {
                trajectory_files_.push_back(entry.path().string());
            }
        }
        
        // 按文件名排序
        std::sort(trajectory_files_.begin(), trajectory_files_.end());
        
        if(trajectory_files_.empty()) {
            RCLCPP_ERROR(this->get_logger(), "No trajectory files found in %s", folder_path_.c_str());
        } else {
            RCLCPP_INFO(this->get_logger(), "Loaded %zu trajectory files", trajectory_files_.size());
        }
    }
    
    std::vector<std::array<double, 6>> read_joint_positions(const std::string& file_path)
    {
        std::vector<std::array<double, 6>> positions;
        std::ifstream file(file_path);
        std::string line;
        
        if(!file.is_open()) {
            RCLCPP_ERROR(this->get_logger(), "Failed to open file: %s", file_path.c_str());
            return positions;
        }
        
        while(std::getline(file, line)) {
            std::array<double, 6> joint_pos;
            size_t pos = 0;
            size_t end_pos;
            int i = 0;
            
            // 解析每行的6个关节角度
            while((end_pos = line.find(',', pos)) != std::string::npos && i < 6) {
                joint_pos[i++] = degrees_to_radians(std::stod(line.substr(pos, end_pos - pos)));
                pos = end_pos + 1;
            }
            
            // 处理最后一个值
            if(i < 6 && pos < line.length()) {
                joint_pos[i] = degrees_to_radians(std::stod(line.substr(pos)));
            }
            
            positions.push_back(joint_pos);
        }
        
        file.close();
        return positions;
    }
    
    bool send_joint_position(const std::array<double, 6>& joints)
    {
        // 等待服务可用，最多等待3秒
        if(!client_->wait_for_service(3s)) {
            RCLCPP_ERROR(this->get_logger(), "Service not available after waiting 3 seconds");
            return false;
        }
        
        auto request = std::make_shared<xc_interfaces::srv::MoveJoints::Request>();
        request->joints = joints;
        
        auto future = client_->async_send_request(request);
        
        // 使用更可靠的spin_until_future_complete方式等待响应
        auto start_time = this->now();
        rclcpp::Rate rate(10); // 10Hz检查频率
        
        while(rclcpp::ok() && (this->now() - start_time) < 10s) {
            // 检查future状态
            if(future.wait_for(0s) == std::future_status::ready) {
                auto response = future.get();
                if(response->result == 0) {
                    return true;
                } else {
                    RCLCPP_ERROR(this->get_logger(), "Service returned error code: %d", response->result);
                    return false;
                }
            }
            
            // 保持节点活跃
            rclcpp::spin_some(this->get_node_base_interface());
            rate.sleep();
        }
        
        RCLCPP_ERROR(this->get_logger(), "Service call timed out after 10 seconds");
        return false;
    }
    
    void execute_trajectories()
    {
        while(rclcpp::ok()) {
            if(trajectory_files_.empty()) {
                RCLCPP_WARN(this->get_logger(), "No trajectory files to execute");
                std::this_thread::sleep_for(1s);
                continue;
            }
            
            // 循环执行每个轨迹文件
            for(const auto& file_path : trajectory_files_) {
                RCLCPP_INFO(this->get_logger(), "Executing trajectory file: %s", file_path.c_str());
                
                // 读取关节位置(已转换为弧度)
                auto positions = read_joint_positions(file_path);
                if(positions.empty()) {
                    RCLCPP_ERROR(this->get_logger(), "No valid positions in file: %s", file_path.c_str());
                    continue;
                }
                
                // 从初始位置开始
                if(!send_joint_position(initial_position_)) {
                    RCLCPP_ERROR(this->get_logger(), "Failed to move to initial position");
                    continue;
                }
                
                // 正向执行轨迹
                for(const auto& pos : positions) {
                    if(!send_joint_position(pos)) {
                        RCLCPP_ERROR(this->get_logger(), "Failed to move to joint position, but will continue to next point");
                    }
                }
                
                // 在终点停留2秒
                std::this_thread::sleep_for(2s);
                
                // 反向执行轨迹
                for(auto it = positions.rbegin() + 1; it != positions.rend(); ++it) {
                    if(!send_joint_position(*it)) {
                        RCLCPP_ERROR(this->get_logger(), "Failed to move to joint position, but will continue to next point");
                    }
                }
                
                // 回到初始位置
                if(!send_joint_position(initial_position_)) {
                    RCLCPP_ERROR(this->get_logger(), "Failed to return to initial position");
                }
            }
        }
    }
    
    rclcpp::Client<xc_interfaces::srv::MoveJoints>::SharedPtr client_;
    std::vector<std::string> trajectory_files_;
    std::array<double, 6> initial_position_;
    std::string folder_path_;
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<JointTrajectoryPlayer>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}