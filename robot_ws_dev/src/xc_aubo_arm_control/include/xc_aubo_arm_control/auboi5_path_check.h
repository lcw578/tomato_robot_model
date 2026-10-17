#pragma once

#include <iostream>
#include <vector>
#include <string>

#include <btBulletDynamicsCommon.h>
#include <BulletCollision/CollisionShapes/btShapeHull.h>
#include <BulletCollision/Gimpact/btGImpactCollisionAlgorithm.h>
#include <BulletCollision/Gimpact/btGImpactShape.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ============================================================================
// 机械臂路径碰撞与关节限制检查类
// ============================================================================
class Collision_Joint_Checker
{
public:
    // 构造函数
    Collision_Joint_Checker();

    // 析构函数
    ~Collision_Joint_Checker();

    // 主检查接口：检查完整轨迹（碰撞 + 关节限制）
    // 返回：true = 轨迹安全；false = 检测到问题
    bool checkTrajectory(const std::vector<std::vector<double>>& trajectory, bool verbose = true);

    // 单独检查关节限制
    bool checkJointLimitsOnly(const std::vector<std::vector<double>>& trajectory, bool verbose = true);

    // 单独检查碰撞
    bool checkCollisionOnly(const std::vector<std::vector<double>>& trajectory, bool verbose = true);

private:
    // ========================================================================
    // 关节限制检查相关
    // ========================================================================
    struct JointLimit {
        double lower;  // 弧度
        double upper;  // 弧度
        std::string name;
    };

    struct SpecialJoint1Limit {
        double lower_left;   // py > 0时的下限（度）
        double upper_left;   // py > 0时的上限（度）
        double lower_right;  // py < 0时的下限（度）
        double upper_right;  // py < 0时的上限（度）
    };

    std::vector<JointLimit> joint_limits_;
    SpecialJoint1Limit special_joint1_limit_;

    bool checkJointLimits(const std::vector<double>& joint_angles, int waypoint_index = -1);

    // ========================================================================
    // 碰撞检测相关（Bullet物理引擎）
    // ========================================================================
    btDefaultCollisionConfiguration* collisionConfiguration;
    btCollisionDispatcher* dispatcher;
    btBroadphaseInterface* overlappingPairCache;
    btCollisionWorld* collisionWorld;

    std::vector<btCollisionObject*> linkCollisionObjects;
    std::vector<btCollisionShape*> collisionShapes;

    // 障碍物相关
    std::vector<btCollisionObject*> obstacleCollisionObjects;
    std::vector<btCollisionShape*> obstacleShapes;

    void createCollisionShapes();
    void createObstacles();
    void updateRobotPose(const std::vector<double>& joint_angles);
    btTransform computeLinkTransform(int link_index, const std::vector<double>& joints);
    bool checkSelfCollision(const std::vector<double>& joint_angles);
    bool checkObstacleCollision(const std::vector<double>& joint_angles);
    bool checkAllCollisions(const std::vector<double>& joint_angles);

    // 辅助函数
    static btQuaternion quatFromRPY_URDF(double roll, double pitch, double yaw);
    static btCompoundShape* makeCompound(btCollisionShape* child,
        const btVector3& local_xyz,
        const btQuaternion& local_q);
    std::string getJointName(int link_index);
};