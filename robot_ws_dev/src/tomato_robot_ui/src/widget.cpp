#include "widget.h"
#include "ui_widget.h"
#include <QThread>
#include <QFile>
#include <QTextStream>

// 在Widget构造函数开头添加ROS2初始化
Widget::Widget(QWidget *parent)
    : QWidget(parent), ui(new Ui::Widget)
{
    // 确保主线程初始化ROS2上下文
    

    ui->setupUi(this);
    // csvFile_path="/home/xucan/code/test/robot_ws/install/xc_car_control/share/xc_car_control/navigation_file/navigation_file.csv";//WSL
    csvFile_path="/home/xc/code/test/ros2/robot_ws/install/xc_car_control/share/xc_car_control/navigation_file/navigation_file.csv"; //工控机

    // 加载CSV文件并提取需要作业的行号
    loadAndDisplayPickRows();

    // 创建并启动ROS线程
    // 修改对象创建和信号连接部分

    // 设置label的固定尺寸策略，防止其随内容变化
    QSizePolicy sizePolicy(QSizePolicy::Ignored, QSizePolicy::Ignored);
    ui->origin_imgLabel->setSizePolicy(sizePolicy);
    ui->processed_imgLabel->setSizePolicy(sizePolicy);
    
    // 不使用setScaledContents，而是手动控制图片缩放以保持长宽比
    ui->origin_imgLabel->setScaledContents(false);
    ui->processed_imgLabel->setScaledContents(false);

    commNode = new rclcomm();

    // // 在连接信号槽之前先设置标签的固定尺寸或最小/最大尺寸
    // QSize originLabelSize = ui->origin_imgLabel->size();
    // QSize processedLabelSize = ui->processed_imgLabel->size();
    

    // // 设置标签的尺寸策略，防止其随内容变化
    // ui->origin_imgLabel->setMinimumSize(originLabelSize);
    // ui->origin_imgLabel->setMaximumSize(originLabelSize);
    // ui->processed_imgLabel->setMinimumSize(processedLabelSize);
    // ui->processed_imgLabel->setMaximumSize(processedLabelSize);

    // 连接信号槽
    connect(commNode, &rclcomm::newOriginImage, this, [this](const QPixmap &img) { 
        if(!img.isNull() && ui->origin_imgLabel) {
            // 使用label的当前尺寸进行缩放，保持长宽比
            QPixmap scaledPixmap = img.scaled(
                ui->origin_imgLabel->width(),
                ui->origin_imgLabel->height(),
                Qt::KeepAspectRatio,
                Qt::SmoothTransformation
            );
            ui->origin_imgLabel->setPixmap(scaledPixmap);
            
            // 输出当前label和缩放后图片尺寸用于调试
            // qDebug() << "Setting origin image, label size:" << ui->origin_imgLabel->width() << "x" << ui->origin_imgLabel->height()
            //          << ", pixmap size:" << scaledPixmap.width() << "x" << scaledPixmap.height();
        }
    },Qt::DirectConnection);//Qt::DirectConnection 直连，让处理代码也在子线程中运行

    connect(commNode, &rclcomm::newProcessedImage, this, [this](const QPixmap &img) { 
         if(!img.isNull() && ui->processed_imgLabel) {
            // 使用label的当前尺寸进行缩放，保持长宽比
            QPixmap scaledPixmap = img.scaled(
                ui->processed_imgLabel->width(),
                ui->processed_imgLabel->height(),
                Qt::KeepAspectRatio,
                Qt::SmoothTransformation
            );
            ui->processed_imgLabel->setPixmap(scaledPixmap);
            
            // 输出当前label和缩放后图片尺寸用于调试
            // qDebug() << "Setting processed image, label size:" << ui->processed_imgLabel->width() << "x" << ui->processed_imgLabel->height()
            //          << ", pixmap size:" << scaledPixmap.width() << "x" << scaledPixmap.height();
        }
    },Qt::DirectConnection);//Qt::DirectConnection 直连，让处理代码也在子线程中运行

    connect(ui->add_row_pushButton, &QPushButton::clicked, this, &Widget::onAddRowClicked);
    connect(ui->write_row_pushButton, &QPushButton::clicked, this, &Widget::onWriteRowsClicked);
    connect(ui->delete_row_pushButton, &QPushButton::clicked, this, &Widget::onDeleteRowClicked);

    // connect(rosThread, &QThread::started, rosNode.get(), &rclcomm::run);  // 添加 .get()

    commNode->start();
}

Widget::~Widget()
{
    // 先删除子对象
    if (motionProcess) {
        if (motionProcess->state() == QProcess::Running)
        {
            // 终止整个进程组
            pid_t pgid = ::getpgid(motionProcess->processId());
            if (pgid > 0)
            {
                ::kill(-pgid, SIGTERM); // 发送信号到整个进程组
            }
            motionProcess->waitForFinished(2000);
        }
        delete motionProcess;
        motionProcess = nullptr;  // 确保指针置空
    }

    if (commNode) {
        commNode->quit();
        commNode->wait();
        delete commNode;
        commNode = nullptr;  // 确保指针置空
    }

    // 最后删除UI
    delete ui;
    ui = nullptr;
}

void Widget::on_auto_runButton_clicked()
{
    if (!motionProcess)
    {
        motionProcess = new QProcess(this);
        motionProcess->setProcessChannelMode(QProcess::MergedChannels);
        connect(motionProcess, &QProcess::readyReadStandardOutput, [&]()
                { qDebug() << "[ROS Launch]:" << motionProcess->readAllStandardOutput(); });
    }

    if (motionProcess->state() == QProcess::NotRunning)
    {
        // 修改启动命令，通过bash创建新进程组
        qDebug() << "-------UI 启动自动运行------";
        motionProcess->start("bash", QStringList()
                                         << "-c"
                                         << "exec ros2 launch xc_robot_main_control fake_robot_launch.py");//fake_robot_launch或者robot_launch.py
    }
}

// 修改关闭事件处理
void Widget::closeEvent(QCloseEvent *event)
{
    Q_UNUSED(event);
    // 移除 delete this; 让Qt自动管理对象生命周期
}



void Widget::onAddRowClicked()
{
    // 获取起始和结束行号
    int startRow = ui->start_lineEdit->text().toInt();
    int endRow = ui->end_lineEdit->text().toInt();
    
    // 验证输入有效性
    if (startRow <= 0 || endRow <= 0 || startRow > endRow) {
        qDebug() << "无效的行号范围";
        return;
    }

    // 生成范围字符串
    QString rangeStr;
    if (startRow == endRow) {
        rangeStr = QString::number(startRow);
    } else {
        rangeStr = QString("%1-%2").arg(startRow).arg(endRow);
    }

    // 添加到列表最后
    ui->need_pick_row_listWidget->addItem(rangeStr);
    
    // 清空输入框
    ui->start_lineEdit->clear();
    ui->end_lineEdit->clear();
}



void Widget::onWriteRowsClicked()
{
    QString csvFile = this->csvFile_path;
    QFile file(csvFile);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        qDebug() << "无法打开CSV文件进行写入";
        return;
    }

    QTextStream out(&file);
    
    // 收集所有需要作业的行号
    QSet<int> workRows;
    for (int i = 0; i < ui->need_pick_row_listWidget->count(); ++i) {
        QString itemText = ui->need_pick_row_listWidget->item(i)->text();
        
        // 忽略分割线行
        if (itemText == "-----------") {
            continue;
        }
        
        QStringList ranges = itemText.split(',');
        
        for (const QString& range : ranges) {
            if (range.contains('-')) {
                QStringList bounds = range.split('-');
                if (bounds.size() == 2) {
                    int start = bounds[0].toInt();
                    int end = bounds[1].toInt();
                    for (int row = start; row <= end; ++row) {
                        workRows.insert(row);
                    }
                }
            } else {
                workRows.insert(range.toInt());
            }
        }
    }

    // 如果没有作业行，写入空文件
    if (workRows.isEmpty()) {
        file.close();
        return;
    }

    // 获取最大行号
    int maxRow = *std::max_element(workRows.begin(), workRows.end());

    // 从第1行到最大行号，按顺序写入
    for (int row = 1; row <= maxRow; ++row) {
        if (workRows.contains(row)) {
            // 写入完整作业序列
            out << row << "1,5\n";
            out << row << "1,1\n";
            out << row << "2,3\n";
            out << row << "3,4\n";
            out << row << "2,2\n";
            out << row << "1,6\n";
            out << row << "1,1\n";
        } else {
            // 写入基础记录
            out << row << "1,1\n";
        }
    }

    file.close();
    
    // 重新加载并显示CSV文件中的作业行
    loadAndDisplayPickRows();
}

// 实现loadAndDisplayPickRows函数
void Widget::loadAndDisplayPickRows()
{
    QString csvFile = this->csvFile_path;
    QSet<int> pickRows;
    
    QFile file(csvFile);
    if (!file.exists()) {
        ui->need_pick_row_listWidget->clear();
        ui->need_pick_row_listWidget->addItem("0");
        return;
    }

    if (file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        QTextStream in(&file);
        QStringList lines;
        while (!in.atEnd()) {
            lines.append(in.readLine().trimmed());
        }
        file.close();

        // 检查完整的作业序列模式
        for (int i = 0; i < lines.size() - 6; i++) {
            QStringList seq;
            for (int j = 0; j < 7; j++) {
                QStringList parts = lines[i+j].split(',');
                if (parts.size() >= 2) {
                    seq << parts[1]; // 收集动作序列
                }
            }
            
            // 检查是否是完整作业序列: 5,1,3,4,2,6,1
            if (seq == QStringList{"5","1","3","4","2","6","1"}) {
                QStringList firstParts = lines[i].split(',');
                if (firstParts.size() >= 1) {
                    bool ok;
                    int cardId = firstParts[0].toInt(&ok);
                    if (ok && cardId >= 10) {
                        int row = cardId / 10;
                        pickRows.insert(row);
                    }
                }
            }
        }
    }

    ui->need_pick_row_listWidget->clear();
    if (pickRows.isEmpty()) {
        ui->need_pick_row_listWidget->addItem("0");
    } else {
        QList<int> sortedRows = pickRows.values();
        std::sort(sortedRows.begin(), sortedRows.end());
        
        // 将连续的行号合并为范围表示
        QStringList result;
        int start = sortedRows.first();
        int prev = start;
        
        for (int i = 1; i < sortedRows.size(); ++i) {
            if (sortedRows[i] != prev + 1) {
                if (start == prev) {
                    result << QString::number(start);
                } else {
                    result << QString("%1-%2").arg(start).arg(prev);
                }
                start = sortedRows[i];
            }
            prev = sortedRows[i];
        }
        
        // 添加最后一个范围
        if (start == prev) {
            result << QString::number(start);
        } else {
            result << QString("%1-%2").arg(start).arg(prev);
        }
        
        // 添加到列表
        for (const QString& row : result) {
            ui->need_pick_row_listWidget->addItem(row);
        }
        
        // 在最后添加分隔线
        ui->need_pick_row_listWidget->addItem("-----------");
    }
}

void Widget::onDeleteRowClicked()
{
    // 获取当前选中的行
    QListWidgetItem *selectedItem = ui->need_pick_row_listWidget->currentItem();
    
    if (selectedItem) {
        // 删除选中的行
        int row = ui->need_pick_row_listWidget->row(selectedItem);
        delete ui->need_pick_row_listWidget->takeItem(row);
    }
}
