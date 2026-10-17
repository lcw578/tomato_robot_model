#include "rclcomm.h"
#include <QImage>
#include <QDebug>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <chrono>
#include <ctime>
#include <cstdlib>



// 修改构造函数初始化方式
//继承自QThread   所以这个代码是在子线程中运行的
rclcomm::rclcomm() 
    // : QThread(nullptr),  // 调整基类初始化顺序，QThread在前
    //   Node("tomato_ui_subscriber", rclcpp::NodeOptions()
    //        .use_global_arguments(true)
    //        .enable_rosout(true))
{
    int argc=0;
    char** argv=NULL;
    
    rclcpp::init(argc,argv);
    node_=rclcpp::Node::make_shared("tomato_ui_subscriber");
    
    // subscription_ = node_->create_subscription<sensor_msgs::msg::Image>(
    //     "yolo_combined_image", 1,
    //     std::bind(&rclcomm::imageCallback, this, std::placeholders::_1));
    detection_result_sub_ = node_->create_subscription<xc_interfaces::msg::Detection2UI>(
        "detection2ui", 1,
        std::bind(&rclcomm::imageCallback_1, this, std::placeholders::_1));
    originImage_sub_ = node_->create_subscription<sensor_msgs::msg::Image>(
        "/camera/color/image_raw", 1,
        std::bind(&rclcomm::origin_imageCallback, this, std::placeholders::_1));
}

// 修改run函数
void rclcomm::run()
{


    rclcpp::WallRate loop_rate(30);
    qDebug()<<"I am run";
    while (rclcpp::ok())
    {
 
        rclcpp::spin_some(node_);//和spin（）比，不会造成阻塞，只接受一次
        loop_rate.sleep();
    }
    rclcpp::shutdown();

}

void rclcomm::imageCallback(const sensor_msgs::msg::Image::SharedPtr msg)
{
    try {
        cv::Mat frame = cv_bridge::toCvCopy(msg, "bgr8")->image;
        
        // 建议添加尺寸校验
        if (frame.cols % 2 != 0) {
            RCLCPP_WARN(node_->get_logger(), "合并图像宽度不是偶数，可能造成分割错误");
        }
        
        // 分割左右图像（当前逻辑正确）
        cv::Mat origin_img = frame(cv::Rect(0, 0, frame.cols/2, frame.rows));
        cv::Mat processed_img = frame(cv::Rect(frame.cols/2, 0, frame.cols/2, frame.rows));

        // 转换图像格式
        QImage origin_qimg(origin_img.data, origin_img.cols, origin_img.rows, 
                         origin_img.step, QImage::Format_BGR888);
        QImage processed_qimg(processed_img.data, processed_img.cols, processed_img.rows,
                           processed_img.step, QImage::Format_BGR888);


        // qDebug() << "Processed image size: " << processed_qimg.width() << "x" << processed_qimg.height();
        // qDebug() << "Processed Mat size: " << processed_img.cols << "x" << processed_img.rows;

        // emit newOriginImage(QPixmap::fromImage(origin_qimg.copy()));
        emit newProcessedImage(QPixmap::fromImage(processed_qimg.copy()));
    } 
    catch (const cv_bridge::Exception& e) {
        RCLCPP_ERROR(node_->get_logger(), "cv_bridge exception: %s", e.what());
    }
}


void rclcomm::imageCallback_1(const xc_interfaces::msg::Detection2UI::SharedPtr msg)
{
    try {
        // 获取彩色图像
        cv::Mat cv_color_image = cv_bridge::toCvCopy(msg->color_image, "bgr8")->image;
        // RCLCPP_INFO(node_->get_logger(), "----------");
        
        // 绘制类名为 tomato 的识别框
        if (msg->class_names.size() > 0 && msg->bboxes.size() > 0) {
            for (size_t i = 0; i < msg->class_names.size(); i++) {                
                // 每个 bounding box 有 4 个值：x1, y1, x2, y2
                size_t bbox_index = i * 4;
                if (bbox_index + 3 < msg->bboxes.size()) {
                    float x1 = msg->bboxes[bbox_index];
                    float y1 = msg->bboxes[bbox_index + 1];
                    float x2 = msg->bboxes[bbox_index + 2];
                    float y2 = msg->bboxes[bbox_index + 3];
                    if (msg->class_names[i] == "tomato") {
                        // 绘制矩形框，颜色为蓝色
                        cv::rectangle(cv_color_image, 
                                    cv::Point(static_cast<int>(x1), static_cast<int>(y1)), 
                                    cv::Point(static_cast<int>(x2), static_cast<int>(y2)), 
                                    cv::Scalar(255, 0, 0), 2);
                    }
                    if (msg->class_names[i] == "stem") {
                        // 绘制矩形框，颜色为黄色
                        cv::rectangle(cv_color_image, 
                                    cv::Point(static_cast<int>(x1), static_cast<int>(y1)), 
                                    cv::Point(static_cast<int>(x2), static_cast<int>(y2)), 
                                    cv::Scalar(0, 255, 255), 2);   
                    }
                     
                    // // 添加类别标签
                    // std::string label = "tomato: " + std::to_string(static_cast<int>(msg->confidences[i] * 100)) + "%";
                    // cv::putText(cv_color_image, label, 
                    //            cv::Point(static_cast<int>(x1), static_cast<int>(y1) - 10), 
                    //            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);
                }
                
            }
        }
        
        // 绘制类名为 stem 的识别 mask
        if (msg->masks.size() > 0) 
        {
            for (const auto& mask_msg : msg->masks) {
                if (mask_msg.class_names == "stem") {
                    // 获取掩码图像
                    cv::Mat mask = cv_bridge::toCvCopy(mask_msg.mask, "mono8")->image;
                    
                    // 确保掩码尺寸与彩色图像匹配
                    if (mask.size() == cv_color_image.size()) {
                        // 创建一个绿色的掩码
                        cv::Mat colored_mask = cv::Mat::zeros(mask.size(), CV_8UC3);
                        colored_mask.setTo(cv::Scalar(255, 0, 0), mask > 0);
                        
                        // 将掩码叠加到彩色图像上
                        cv::addWeighted(cv_color_image, 1.0, colored_mask, 0.5, 0, cv_color_image);
                    }
                }
            }
        }

        if (msg->matched_rectangles.size() > 0)
        {
            size_t num_rectangles = msg->matched_rectangles.size() / 4;
            for (size_t i = 0; i < num_rectangles; i++) {
                // 每个矩形框有 4 个值：x1, y1, x2, y2
                size_t rect_index = i * 4;
                if (rect_index + 3 < msg->matched_rectangles.size()) {
                    float x1 = msg->matched_rectangles[rect_index];
                    float y1 = msg->matched_rectangles[rect_index + 1];
                    float x2 = msg->matched_rectangles[rect_index + 2];
                    float y2 = msg->matched_rectangles[rect_index + 3];
                    
                    // 绘制矩形框，颜色为绿色
                    cv::rectangle(cv_color_image, 
                                 cv::Point(static_cast<int>(x1), static_cast<int>(y1)), 
                                 cv::Point(static_cast<int>(x2), static_cast<int>(y2)), 
                                 cv::Scalar(255, 255, 0), 2);
                    // 添加编号标签
                    std::string label = "id: " + std::to_string(i+1);
                    cv::putText(cv_color_image, label, 
                               cv::Point(static_cast<int>(x1), static_cast<int>(y1) + 10), 
                               cv::FONT_HERSHEY_SIMPLEX, 1.5, cv::Scalar(255, 255, 0), 2);
                }
            }
            if (msg->distances>0.0 && msg->picking_points_x > 0 && msg->picking_points_y > 0)
            {
                // 绘制采摘点
                cv::circle(cv_color_image, 
                    cv::Point(int(msg->picking_points_x), int(msg->picking_points_y)), 
                    5, cv::Scalar(0, 0, 255), -1);
                
                // 1. 根据arm_ry的角度绘制一条经过采摘点的线段
                float line_length = 100.0; // 线段长度
                float angle_rad = msg->arm_ry * M_PI / 180.0; // 将角度转换为弧度
                
                // 计算线段的终点坐标
                int x0 = static_cast<int>(msg->picking_points_x);
                int y0 = static_cast<int>(msg->picking_points_y);
                int x1 = static_cast<int>(x0 + line_length * cos(angle_rad));
                int y1 = static_cast<int>(y0 + line_length * sin(angle_rad)); // 注意y轴方向与常规坐标系相反
                
                // 确保线段不超过图片边界
                int img_width = cv_color_image.cols;
                int img_height = cv_color_image.rows;
                
                // 调整线段终点，确保在图片边界内
                if (x1 < 0 || x1 >= img_width || y1 < 0 || y1 >= img_height) {
                    // 如果终点超出边界，计算线段与边界的交点
                    float t = 1.0;
                    
                    // 检查与左右边界的交点
                    if (x1 < 0) t = std::min(t, static_cast<float>(-x0) / (x1 - x0));
                    if (x1 >= img_width) t = std::min(t, static_cast<float>(img_width - 1 - x0) / (x1 - x0));
                    
                    // 检查与上下边界的交点
                    if (y1 < 0) t = std::min(t, static_cast<float>(-y0) / (y1 - y0));
                    if (y1 >= img_height) t = std::min(t, static_cast<float>(img_height - 1 - y0) / (y1 - y0));
                    
                    // 计算调整后的终点
                    x1 = static_cast<int>(x0 + t * (x1 - x0));
                    y1 = static_cast<int>(y0 + t * (y1 - y0));
                }
                
                // 绘制线段
                cv::line(cv_color_image, cv::Point(x0, y0), cv::Point(x1, y1), cv::Scalar(0, 255, 0), 2);
                
                // 2. 以文字的方式显示mid2target_angle在图片上
                // 使用 stringstream 格式化输出，保留3位小数
                std::stringstream ss;
                ss << std::fixed << std::setprecision(3) << msg->mid2target_angle;
                std::string angle_label = "mid2target_angle: " + ss.str() + " deg";
                cv::putText(cv_color_image, angle_label, 
                            cv::Point(30, 40), 
                            cv::FONT_HERSHEY_SIMPLEX, 1.5, cv::Scalar(0, 0, 255), 2);
                
                // 显示距离信息
                ss.str(""); // 清空stringstream，重新使用
                ss << std::fixed << std::setprecision(3) << msg->distances;
                std::string distance_label = "z: " + ss.str() + " m";
                cv::putText(cv_color_image, distance_label, 
                            cv::Point(30, 100),
                            cv::FONT_HERSHEY_SIMPLEX, 1.5, cv::Scalar(0, 0, 255), 2);
                
                // cv::imwrite("test.png", cv_color_image);
            }
        }
        if (is_save_img_==true && msg->distances>0.0 && msg->picking_points_x > 0 && msg->picking_points_y > 0)
        {
            // 创建保存目录
            std::string save_dir = "~/code/test/temp_xc/";
            // 展开波浪号
            const char* home_dir = getenv("HOME");
            if (home_dir) {
                if (save_dir.substr(0, 2) == "~") {
                    save_dir = home_dir + save_dir.substr(1);
                }
            }
            
            // 创建目录（如果不存在）
            std::filesystem::create_directories(save_dir);
            
            // 生成带时间戳的文件名
            auto now = std::chrono::system_clock::now();
            auto now_c = std::chrono::system_clock::to_time_t(now);
            std::tm* now_tm = std::localtime(&now_c);
            
            // 计算毫秒部分
            auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;
            
            char filename[100];
            std::strftime(filename, sizeof(filename), "tomato_%Y%m%d_%H%M%S", now_tm);
            
            // 拼接毫秒部分
            char full_filename[120];
            std::snprintf(full_filename, sizeof(full_filename), "%s_%03d.png", filename, static_cast<int>(ms.count()));
            
            std::string full_path = std::string("/home/xc/code/test/ros2/robot_ws/temp_img/") + full_filename;
            cv::imwrite(full_path, cv_color_image);
            RCLCPP_INFO(node_->get_logger(), "Image saved to: %s", full_path.c_str());
        }
        // 转换为 QImage
        QImage processed_qimg(cv_color_image.data, cv_color_image.cols, cv_color_image.rows,
                           cv_color_image.step, QImage::Format_BGR888); 

        // 发送信号
        emit newProcessedImage(QPixmap::fromImage(processed_qimg.copy()));
    } 
    catch (const cv_bridge::Exception& e) {
        RCLCPP_ERROR(node_->get_logger(), "cv_bridge exception: %s", e.what());
    }
    catch (const std::exception& e) {
        RCLCPP_ERROR(node_->get_logger(), "Exception: %s", e.what());
    }
}

void rclcomm::origin_imageCallback(const sensor_msgs::msg::Image::SharedPtr msg)
{
    try 
    {
        cv::Mat origin_img = cv_bridge::toCvCopy(msg, "bgr8")->image;

        // 转换图像格式
        QImage origin_qimg(origin_img.data, origin_img.cols, origin_img.rows, 
                         origin_img.step, QImage::Format_BGR888);
        // qDebug() << "Origin image size: " << origin_qimg.width() << "x" << origin_qimg.height();
        // qDebug() << "Origin Mat size: " << origin_img.cols << "x" << origin_img.rows;

        

        emit newOriginImage(QPixmap::fromImage(origin_qimg.copy()));
    }
    catch (const cv_bridge::Exception& e) {
        RCLCPP_ERROR(node_->get_logger(), "origin_imageCallback error: %s", e.what());
    }
}