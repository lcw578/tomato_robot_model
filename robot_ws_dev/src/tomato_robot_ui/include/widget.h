#ifndef WIDGET_H
#define WIDGET_H

#include <QWidget>
#include "rclcomm.h"
#include <QProcess>
#include <QDebug>

QT_BEGIN_NAMESPACE
namespace Ui { class Widget; }
QT_END_NAMESPACE

class Widget : public QWidget
{
    Q_OBJECT

public:
    Widget(QWidget *parent = nullptr);
    ~Widget();
    rclcomm *commNode;

private slots:
    void on_auto_runButton_clicked();

private:
    Ui::Widget *ui;
    QProcess *motionProcess = nullptr;  // 新增进程指针
    void closeEvent(QCloseEvent *event) override;  // 添加关闭事件处理
    void loadAndDisplayPickRows();
    void onAddRowClicked();
    void onWriteRowsClicked();
    void onDeleteRowClicked();
    QString csvFile_path;
    // QImage newOriginImage_;
    // QImage newProcessedImage_;

    
    
};
#endif // WIDGET_H
