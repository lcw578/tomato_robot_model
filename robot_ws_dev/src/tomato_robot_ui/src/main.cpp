#include "widget.h"

#include <QApplication>
// #include <QTextCodec>

int main(int argc, char *argv[])
{
    // // 设置文本编码为UTF-8，解决中文乱码问题
    // QTextCodec::setCodecForLocale(QTextCodec::codecForName("UTF-8"));
    QApplication a(argc, argv);
    Widget w;
    w.show();
    return a.exec();
}
