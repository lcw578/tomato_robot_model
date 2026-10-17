#ifndef RCLCOMM_H
#define RCLCOMM_H

#include <QObject>
#include <QThread>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <QWidget>
#include "xc_interfaces/msg/detection_result.hpp"
#include "xc_interfaces/msg/detection2_ui.hpp"
#include <sstream>
#include <iomanip>



// 修改类继承方式
class rclcomm:public QThread
{
    Q_OBJECT  // 必须包含这个宏
public:
    rclcomm();
    void run() override;

signals:
    void newOriginImage(const QPixmap &img);
    void newProcessedImage(const QPixmap &img);

private:
    void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg);
    void origin_imageCallback(const sensor_msgs::msg::Image::SharedPtr msg);
    void imageCallback_1(const xc_interfaces::msg::Detection2UI::SharedPtr msg);

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr originImage_sub_ ;
    rclcpp::Subscription<xc_interfaces::msg::Detection2UI>::SharedPtr detection_result_sub_;

    bool is_save_img_=true;



    std::shared_ptr<rclcpp::Node> node_;

};
#endif // RCLCOMM_H
