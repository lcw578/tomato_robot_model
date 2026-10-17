# 上传：
1 git add .
2 git commit -m "黄焕培 视觉"
3 git push

# 拉取最新代码：
git pull


# 强制拉取覆盖
## 获取远程更新
git fetch gitcode master
## 强制重置
git reset --hard gitcode/master

# 撤回git commit
git reset --soft HEAD~1  保留修改
# 撤回git add .
git reset

# 多人协作
git push前先git pull
如果两个人同时修改了不同文件，不会产生冲突

# 拉取某个项目
git clone https://xc325:MoRppP4ew3q7YFAma5iNNcmF@gitcode.com/xc325/yolov8-10-mo.git

# 建立远程连接
git remote add gitcode https://xc325:MoRppP4ew3q7YFAma5iNNcmF@gitcode.com/xc325/yolov8-10-mo.git

# 推送
git push -u gitcode  master
