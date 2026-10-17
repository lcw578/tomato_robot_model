#include "xc_aubo_arm_control/auboi5_path_check.h"
#include <urdf/model.h>
#include <fstream>
#include <iostream>
#include <string>

// ============================================================================
// 构造函数
// ============================================================================
Collision_Joint_Checker::Collision_Joint_Checker()
    : collisionConfiguration(nullptr)
    , dispatcher(nullptr)
    , overlappingPairCache(nullptr)
    , collisionWorld(nullptr)
{
    // ========================================================================
    // 加载URDF文件
    // ========================================================================
    urdf::Model model;
    std::string urdf_path = "/home/xc/code/test/ros2/robot_ws/src/xc_aubo_arm_control/aubo_description/urdf/aubo_i5_gripper.urdf"; //工控机
    // std::string urdf_path = "/home/xucan/code/test/robot_ws/src/xc_aubo_arm_control/aubo_description/urdf/aubo_i5_gripper.urdf";//office wsl
    // std::string urdf_path = "/home/homevm/code/test/robot_ws/src/xc_aubo_arm_control/aubo_description/urdf/aubo_i5_gripper.urdf";//home VMWare

    
    if (!model.initFile(urdf_path)) {
        std::cerr << "❌ 无法加载URDF文件: " << urdf_path << std::endl;
        exit(1);
    }
    
    std::cout << "✅ URDF文件加载成功: " << urdf_path << std::endl;

    // ========================================================================
    // 从URDF中提取关节限制
    // ========================================================================
    std::vector<std::string> joint_names = {"shoulder_joint", "upperArm_joint", "foreArm_joint", "wrist1_joint", "wrist2_joint", "wrist3_joint"};
    
    for (const auto& joint_name : joint_names) {
        const urdf::JointConstSharedPtr& joint = model.getJoint(joint_name);
        if (joint && joint->type == urdf::Joint::REVOLUTE) {
            double lower = joint->limits->lower;
            double upper = joint->limits->upper;
            joint_limits_.push_back({lower, upper, joint_name});
            std::cout << "   - 关节 " << joint_name << " 限制: " << lower << " 到 " << upper << " 弧度" << std::endl;
        }
    }

    // 设置第一关节的特殊限制（度）
    special_joint1_limit_.lower_left = 10;   // py > 0时 -90.0
    special_joint1_limit_.upper_left = 370;
    special_joint1_limit_.lower_right = -350; // py < 0时
    special_joint1_limit_.upper_right = 0;

    // ========================================================================
    // 初始化Bullet碰撞检测
    // ========================================================================
    collisionConfiguration = new btDefaultCollisionConfiguration();
    dispatcher = new btCollisionDispatcher(collisionConfiguration);
    overlappingPairCache = new btDbvtBroadphase();
    collisionWorld = new btCollisionWorld(dispatcher, overlappingPairCache, collisionConfiguration);

    btGImpactCollisionAlgorithm::registerAlgorithm(dispatcher);

    createCollisionShapes();
    createObstacles();

    std::cout << "✅ Collision_Joint_Checker 初始化完成" << std::endl;
    std::cout << "   - 已添加 2 个固定障碍物" << std::endl;
}

// ============================================================================
// 析构函数
// ============================================================================
Collision_Joint_Checker::~Collision_Joint_Checker()
{
    // 清理碰撞对象
    for (auto obj : linkCollisionObjects) {
        collisionWorld->removeCollisionObject(obj);
        delete obj;
    }

    // 清理障碍物对象
    for (auto obj : obstacleCollisionObjects) {
        collisionWorld->removeCollisionObject(obj);
        delete obj;
    }

    // 清理碰撞形状
    for (auto shape : collisionShapes) {
        delete shape;
    }

    // 清理障碍物形状
    for (auto shape : obstacleShapes) {
        delete shape;
    }

    // 清理Bullet世界
    delete collisionWorld;
    delete overlappingPairCache;
    delete dispatcher;
    delete collisionConfiguration;

    std::cout << "Collision_Joint_Checker 资源已释放" << std::endl;
}

// ============================================================================
// 主检查接口：检查完整轨迹
// ============================================================================
bool Collision_Joint_Checker::checkTrajectory(const std::vector<std::vector<double>>& trajectory, bool verbose)
{
    if (trajectory.empty()) {
        std::cout << "轨迹为空，无法进行检查" << std::endl;
        return false;
    }

    if (verbose) {
        std::cout << "\n========================================" << std::endl;
        std::cout << "开始检查轨迹（共 " << trajectory.size() << " 个路径点）" << std::endl;
        std::cout << "========================================\n" << std::endl;
    }

    bool limit_check_passed = true;
    bool collision_check_passed = true;

    // 1. 关节限制检查
    if (verbose) {
        std::cout << "关节限制检查..." << std::endl;
    }

    std::vector<int> violation_waypoints;
    for (size_t i = 0; i < trajectory.size(); ++i) {
        bool is_valid = checkJointLimits(trajectory[i], verbose ? static_cast<int>(i) : -1);
        if (!is_valid) {
            violation_waypoints.push_back(static_cast<int>(i));
            limit_check_passed = false;
        }
    }

    if (verbose) {
        if (limit_check_passed) {
            std::cout << "   关节限制检查通过\n" << std::endl;
        }
        else {
            std::cout << "  关节限制检查失败，共 " << violation_waypoints.size() << " 个路径点超限" << std::endl;
            std::cout << "  超限路径点: ";
            for (size_t i = 0; i < violation_waypoints.size(); ++i) {
                std::cout << (violation_waypoints[i] + 1);
                if (i < violation_waypoints.size() - 1) std::cout << ", ";
            }
            std::cout << "\n" << std::endl;
        }
    }

    // 2. 碰撞检测（包括自碰撞和障碍物碰撞）
    if (verbose) {
        std::cout << "碰撞检测（自碰撞 + 障碍物碰撞）..." << std::endl;
    }

    std::vector<int> collision_waypoints;
    for (size_t i = 0; i < trajectory.size(); ++i) {
        bool has_collision = checkAllCollisions(trajectory[i]);
        if (has_collision) {
            collision_waypoints.push_back(static_cast<int>(i));
            collision_check_passed = false;
            if (verbose) {
                std::cout << " 路径点 " << (i + 1) << " 发生碰撞" << std::endl;
            }
        }
    }

    if (verbose) {
        if (collision_check_passed) {
            std::cout << "  碰撞检测通过\n" << std::endl;
        }
        else {
            std::cout << "  碰撞检测失败，共 " << collision_waypoints.size() << " 个路径点发生碰撞" << std::endl;
            std::cout << "  碰撞路径点: ";
            for (size_t i = 0; i < collision_waypoints.size(); ++i) {
                std::cout << (collision_waypoints[i] + 1);
                if (i < collision_waypoints.size() - 1) std::cout << ", ";
            }
            std::cout << "\n" << std::endl;
        }
    }

    // 3. 总结
    bool overall_passed = limit_check_passed && collision_check_passed;

    if (verbose) {
        std::cout << "========================================" << std::endl;
        std::cout << "检查结果总结" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "关节限制: " << (limit_check_passed ? " 通过" : "失败") << std::endl;
        std::cout << "碰撞检测: " << (collision_check_passed ? "通过" : "失败") << std::endl;
        std::cout << "----------------------------------------" << std::endl;
        std::cout << "最终结果: " << (overall_passed ? "轨迹安全，可以执行" : " 轨迹不安全，不可执行") << std::endl;
        std::cout << "========================================\n" << std::endl;
    }

    return overall_passed;
}

// ============================================================================
// 单独检查关节限制
// ============================================================================
bool Collision_Joint_Checker::checkJointLimitsOnly(const std::vector<std::vector<double>>& trajectory, bool verbose)
{
    bool all_passed = true;

    for (size_t i = 0; i < trajectory.size(); ++i) {
        bool is_valid = checkJointLimits(trajectory[i], verbose ? static_cast<int>(i) : -1);
        if (!is_valid) {
            all_passed = false;
        }
    }

    return all_passed;
}

// ============================================================================
// 单独检查碰撞
// ============================================================================
bool Collision_Joint_Checker::checkCollisionOnly(const std::vector<std::vector<double>>& trajectory, bool verbose)
{
    bool all_passed = true;

    for (size_t i = 0; i < trajectory.size(); ++i) {
        bool has_collision = checkAllCollisions(trajectory[i]);
        if (has_collision) {
            all_passed = false;
            if (verbose) {
                std::cout << "路径点 " << (i + 1) << " 发生碰撞" << std::endl;
            }
        }
    }

    return all_passed;
}

// ============================================================================
// 检查单个配置的关节限制
// ============================================================================
bool Collision_Joint_Checker::checkJointLimits(const std::vector<double>& joint_angles, int waypoint_index)
{
    if (joint_angles.size() != 6) {
        std::cout << "错误：关节角度数量不正确！" << std::endl;
        return false;
    }

    bool all_valid = true;
    std::string waypoint_str = (waypoint_index >= 0) ?
        ("路径点 " + std::to_string(waypoint_index + 1)) : "当前配置";

    // 检查关节1（索引0）
    double joint1_deg = joint_angles[0] * 180.0 / M_PI;
    double overall_lower = std::min(special_joint1_limit_.lower_left, special_joint1_limit_.lower_right);
    double overall_upper = std::max(special_joint1_limit_.upper_left, special_joint1_limit_.upper_right);

    if (joint1_deg < overall_lower || joint1_deg > overall_upper) {
        if (waypoint_index >= 0) {
            std::cout << "  " << waypoint_str << " - 关节1 超限: " << joint1_deg
                << "° (范围: " << overall_lower << "° 到 " << overall_upper << "°)" << std::endl;
        }
        all_valid = false;
    }

    // 检查关节2-6（索引1-5）
    for (size_t i = 1; i < joint_angles.size(); ++i) {
        if (joint_angles[i] < joint_limits_[i].lower || joint_angles[i] > joint_limits_[i].upper) {
            double angle_deg = joint_angles[i] * 180.0 / M_PI;
            double lower_deg = joint_limits_[i].lower * 180.0 / M_PI;
            double upper_deg = joint_limits_[i].upper * 180.0 / M_PI;

            if (waypoint_index >= 0) {
                std::cout << "   " << waypoint_str << " - 关节" << (i + 1)
                    << " 超限: " << angle_deg << "° (范围: " << lower_deg
                    << "° 到 " << upper_deg << "°)" << std::endl;
            }
            all_valid = false;
        }
    }

    return all_valid;
}

// ============================================================================
// 获取关节名称
// ============================================================================
std::string Collision_Joint_Checker::getJointName(int link_index)
{
    // 关节名称映射
    std::vector<std::string> joint_names = {"base_link", "shoulder_joint", "upperArm_joint", "foreArm_joint", "wrist1_joint", "wrist2_joint", "wrist3_joint"};
    if (link_index >= 0 && link_index < joint_names.size()) {
        return joint_names[link_index];
    }
    return "unknown_joint";
}

// ============================================================================
// 创建碰撞形状（使用圆柱体）
// ============================================================================
void Collision_Joint_Checker::createCollisionShapes()
{
    const double safety_margin = 1.0;

    // Link 0: base_link - 圆柱体（Z轴方向）
    {
        btCollisionShape* child = new btCylinderShapeZ(
            btVector3(0.09 * safety_margin, 0.09 * safety_margin, 0.06 * safety_margin));
        collisionShapes.push_back(child);

        btCompoundShape* comp = makeCompound(child, btVector3(0, 0, 0.06), btQuaternion::getIdentity());
        collisionShapes.push_back(comp);
    }

    // Link 1: shoulder_Link - 圆柱体（Z轴方向）
    {
        btCollisionShape* child = new btCylinderShapeZ(
            btVector3(0.075 * safety_margin, 0.075 * safety_margin, 0.09 * safety_margin));
        collisionShapes.push_back(child);

        btVector3 local(0.0, 0.0053495535, -0.0088368933);
        btCompoundShape* comp = makeCompound(child, local, btQuaternion::getIdentity());
        collisionShapes.push_back(comp);
    }

    // Link 2: upperArm_Link - 圆柱体（X轴方向）
    {
        btCollisionShape* child = new btCylinderShapeX(
            btVector3(0.204 * safety_margin, 0.045 * safety_margin, 0.045 * safety_margin));
        collisionShapes.push_back(child);

        btVector3 local(0.203996647, 0.0, 0.0127641545);
        btCompoundShape* comp = makeCompound(child, local, btQuaternion::getIdentity());
        collisionShapes.push_back(comp);
    }

    // Link 3: foreArm_Link - 圆柱体（X轴方向）
    {
        btCollisionShape* child = new btCylinderShapeX(
            btVector3(0.188 * safety_margin, 0.04 * safety_margin, 0.04 * safety_margin));
        collisionShapes.push_back(child);

        btVector3 local(0.188922116, 0.0, 0.098102674);
        btCompoundShape* comp = makeCompound(child, local, btQuaternion::getIdentity());
        collisionShapes.push_back(comp);
    }

    // Link 4: wrist1_Link - 圆柱体（Z轴方向）
    {
        btCollisionShape* child = new btCylinderShapeZ(
            btVector3(0.038 * safety_margin, 0.038 * safety_margin, 0.045 * safety_margin));
        collisionShapes.push_back(child);

        btVector3 local(0.0, 0.00624812543, -0.00392367464);
        btCompoundShape* comp = makeCompound(child, local, btQuaternion::getIdentity());
        collisionShapes.push_back(comp);
    }

    // Link 5: wrist2_Link - 圆柱体（Z轴方向）
    {
        btCollisionShape* child = new btCylinderShapeZ(
            btVector3(0.038 * safety_margin, 0.038 * safety_margin, 0.045 * safety_margin));
        collisionShapes.push_back(child);

        btVector3 local(-0.0000007542, -0.00624812543, -0.00392367464);
        btCompoundShape* comp = makeCompound(child, local, btQuaternion::getIdentity());
        collisionShapes.push_back(comp);
    }

    // Link 6: wrist3_Link - 圆柱体（Z轴方向）
    {
        btCollisionShape* child = new btCylinderShapeZ(
            btVector3(0.032 * safety_margin, 0.032 * safety_margin, 0.035 * safety_margin));
        collisionShapes.push_back(child);

        btVector3 local(0.0, 0.000175788057, -0.02132944907);
        btCompoundShape* comp = makeCompound(child, local, btQuaternion::getIdentity());
        collisionShapes.push_back(comp);
    }

    // 为每个Link创建碰撞对象
    const int link_count = 7;
    for (int i = 0; i < link_count; ++i) {
        btCollisionShape* linkShape = collisionShapes[i * 2 + 1];

        btCollisionObject* obj = new btCollisionObject();
        obj->setCollisionShape(linkShape);
        obj->setUserIndex(i);
        linkCollisionObjects.push_back(obj);
        collisionWorld->addCollisionObject(obj);
    }
}

// ============================================================================
// 创建障碍物
// ============================================================================
void Collision_Joint_Checker::createObstacles()
{
    // 障碍物1：30cm(y) x 23cm(x) x 14cm(z)
    // 底部中心位置：x = -(27 + 23/2) = -(27 + 11.5) = -38.5cm = -0.385m
    // 由于box的中心在几何中心，z方向偏移为高度的一半
    {
        // 尺寸转换为米并除以2（Bullet的box尺寸是半长）
        btVector3 halfExtents(0.23 / 2.0, 0.30 / 2.0, 0.14 / 2.0);  // x, y, z
        btBoxShape* box1 = new btBoxShape(halfExtents);
        obstacleShapes.push_back(box1);

        btCollisionObject* obstacle1 = new btCollisionObject();
        obstacle1->setCollisionShape(box1);

        // 设置位置：底部中心在 (-0.385, 0, 0)，但box中心在 z = 0.14/2 = 0.07m
        btTransform transform;
        transform.setIdentity();
        transform.setOrigin(btVector3(-0.385, 0, 0.14 / 2.0));
        obstacle1->setWorldTransform(transform);
        obstacle1->setUserIndex(100);  // 使用特殊索引标识障碍物

        obstacleCollisionObjects.push_back(obstacle1);
        collisionWorld->addCollisionObject(obstacle1);

        std::cout << "   - 障碍物1: 位置 x=" << -0.385 << "m, 尺寸 23x30x14cm" << std::endl;
    }

    // 障碍物2：11cm(y) x 7cm(x) x 41cm(z)
    // 底部中心位置：-(27 + 23 + 7/2) = -(27 + 23 + 3.5) = -53.5cm = -0.535m
    {
        btVector3 halfExtents(0.07 / 2.0, 0.11 / 2.0, 0.41 / 2.0);  // x, y, z
        btBoxShape* box2 = new btBoxShape(halfExtents);
        obstacleShapes.push_back(box2);

        btCollisionObject* obstacle2 = new btCollisionObject();
        obstacle2->setCollisionShape(box2);

        // 设置位置：底部中心在 (-0.42, 0, 0)，但box中心在 z = 0.41/2 = 0.205m
        btTransform transform;
        transform.setIdentity();
        transform.setOrigin(btVector3(-0.535, 0, 0.41 / 2.0));
        obstacle2->setWorldTransform(transform);
        obstacle2->setUserIndex(101);  // 使用特殊索引标识障碍物

        obstacleCollisionObjects.push_back(obstacle2);
        collisionWorld->addCollisionObject(obstacle2);

        std::cout << "   - 障碍物2: 位置 x=" << -0.42 << "m, 尺寸 7x11x41cm" << std::endl;
    }
}

// ============================================================================
// 计算连杆变换矩阵
// ============================================================================
btTransform Collision_Joint_Checker::computeLinkTransform(int link_index, const std::vector<double>& joints)
{
    btTransform T;
    T.setIdentity();

    if (link_index == 0) {
        return T;
    }

    // Joint0: world/base -> shoulder_Link
    {
        btTransform joint_origin;
        joint_origin.setIdentity();
        joint_origin.setOrigin(btVector3(0, 0, 0.122));
        joint_origin.setRotation(quatFromRPY_URDF(0, 0, M_PI));

        btTransform joint_rot;
        joint_rot.setIdentity();
        joint_rot.setRotation(btQuaternion(btVector3(0, 0, 1), joints[0]));

        T = T * joint_origin * joint_rot;
        if (link_index == 1) return T;
    }

    // Joint1: shoulder -> upperArm
    {
        btTransform joint_origin;
        joint_origin.setIdentity();
        joint_origin.setOrigin(btVector3(0, 0.1215, 0));
        joint_origin.setRotation(quatFromRPY_URDF(-M_PI / 2, -M_PI / 2, 0));

        btTransform joint_rot;
        joint_rot.setIdentity();
        joint_rot.setRotation(btQuaternion(btVector3(0, 0, 1), joints[1]));

        T = T * joint_origin * joint_rot;
        if (link_index == 2) return T;
    }

    // Joint2: upperArm -> foreArm
    {
        btTransform joint_origin;
        joint_origin.setIdentity();
        joint_origin.setOrigin(btVector3(0.408, 0, 0));
        joint_origin.setRotation(quatFromRPY_URDF(-M_PI, 0, 0));

        btTransform joint_rot;
        joint_rot.setIdentity();
        joint_rot.setRotation(btQuaternion(btVector3(0, 0, 1), joints[2]));

        T = T * joint_origin * joint_rot;
        if (link_index == 3) return T;
    }

    // Joint3: foreArm -> wrist1
    {
        btTransform joint_origin;
        joint_origin.setIdentity();
        joint_origin.setOrigin(btVector3(0.376, 0, 0));
        joint_origin.setRotation(quatFromRPY_URDF(M_PI, 0, M_PI / 2));

        btTransform joint_rot;
        joint_rot.setIdentity();
        joint_rot.setRotation(btQuaternion(btVector3(0, 0, 1), joints[3]));

        T = T * joint_origin * joint_rot;
        if (link_index == 4) return T;
    }

    // Joint4: wrist1 -> wrist2
    {
        btTransform joint_origin;
        joint_origin.setIdentity();
        joint_origin.setOrigin(btVector3(0, 0.1025, 0));
        joint_origin.setRotation(quatFromRPY_URDF(-M_PI / 2, 0, 0));

        btTransform joint_rot;
        joint_rot.setIdentity();
        joint_rot.setRotation(btQuaternion(btVector3(0, 0, 1), joints[4]));

        T = T * joint_origin * joint_rot;
        if (link_index == 5) return T;
    }

    // Joint5: wrist2 -> wrist3
    {
        btTransform joint_origin;
        joint_origin.setIdentity();
        joint_origin.setOrigin(btVector3(0, -0.094, 0));
        joint_origin.setRotation(quatFromRPY_URDF(M_PI / 2, 0, 0));

        btTransform joint_rot;
        joint_rot.setIdentity();
        joint_rot.setRotation(btQuaternion(btVector3(0, 0, 1), joints[5]));

        T = T * joint_origin * joint_rot;
        if (link_index == 6) return T;
    }

    return T;
}

// ============================================================================
// 更新机器人姿态
// ============================================================================
void Collision_Joint_Checker::updateRobotPose(const std::vector<double>& joint_angles)
{
    for (size_t i = 0; i < linkCollisionObjects.size(); ++i) {
        btTransform tf = computeLinkTransform(static_cast<int>(i), joint_angles);
        linkCollisionObjects[i]->setWorldTransform(tf);
    }
}

// ============================================================================
// 检查自碰撞
// ============================================================================
bool Collision_Joint_Checker::checkSelfCollision(const std::vector<double>& joint_angles)
{
    updateRobotPose(joint_angles);

    bool collision_detected = false;

    for (size_t i = 0; i < linkCollisionObjects.size(); ++i) {
        for (size_t j = i + 2; j < linkCollisionObjects.size(); ++j) {
            // 跳过相邻连杆和某些特定组合
            if ((i == 0 && j == 2) || (i == 1 && j == 3)) continue;

            btCollisionObject* objA = linkCollisionObjects[i];
            btCollisionObject* objB = linkCollisionObjects[j];

            btCollisionObjectWrapper wrapA(nullptr, objA->getCollisionShape(), objA, objA->getWorldTransform(), -1, -1);
            btCollisionObjectWrapper wrapB(nullptr, objB->getCollisionShape(), objB, objB->getWorldTransform(), -1, -1);

            btCollisionAlgorithm* algorithm =
                dispatcher->findAlgorithm(&wrapA, &wrapB, nullptr, BT_CLOSEST_POINT_ALGORITHMS);

            btManifoldResult contactPointResult(&wrapA, &wrapB);
            algorithm->processCollision(&wrapA, &wrapB, collisionWorld->getDispatchInfo(), &contactPointResult);

            btPersistentManifold* manifold = contactPointResult.getPersistentManifold();
            if (manifold && manifold->getNumContacts() > 0) {
                collision_detected = true;
                std::cout << "   自碰撞: " << getJointName(i) << " 与 " << getJointName(j) << std::endl;
            }

            dispatcher->freeCollisionAlgorithm(algorithm);

            if (collision_detected) break;
        }
        if (collision_detected) break;
    }

    return collision_detected;
}

// ============================================================================
// 检查与障碍物的碰撞
// ============================================================================
bool Collision_Joint_Checker::checkObstacleCollision(const std::vector<double>& joint_angles)
{
    updateRobotPose(joint_angles);

    bool collision_detected = false;

    // 检查每个机械臂连杆与每个障碍物的碰撞
    for (size_t i = 0; i < linkCollisionObjects.size(); ++i) {
        for (size_t j = 0; j < obstacleCollisionObjects.size(); ++j) {
            btCollisionObject* linkObj = linkCollisionObjects[i];
            btCollisionObject* obstacleObj = obstacleCollisionObjects[j];

            btCollisionObjectWrapper wrapA(nullptr, linkObj->getCollisionShape(),
                linkObj, linkObj->getWorldTransform(), -1, -1);
            btCollisionObjectWrapper wrapB(nullptr, obstacleObj->getCollisionShape(),
                obstacleObj, obstacleObj->getWorldTransform(), -1, -1);

            btCollisionAlgorithm* algorithm =
                dispatcher->findAlgorithm(&wrapA, &wrapB, nullptr, BT_CLOSEST_POINT_ALGORITHMS);

            btManifoldResult contactPointResult(&wrapA, &wrapB);
            algorithm->processCollision(&wrapA, &wrapB, collisionWorld->getDispatchInfo(), &contactPointResult);

            btPersistentManifold* manifold = contactPointResult.getPersistentManifold();
            if (manifold && manifold->getNumContacts() > 0) {
                collision_detected = true;
                std::cout << "   障碍物碰撞: " << getJointName(i) << " 与 障碍物" << (j + 1) << std::endl;
            }

            dispatcher->freeCollisionAlgorithm(algorithm);

            if (collision_detected) break;
        }
        if (collision_detected) break;
    }

    return collision_detected;
}

// ============================================================================
// 检查所有碰撞（自碰撞 + 障碍物碰撞）
// ============================================================================
bool Collision_Joint_Checker::checkAllCollisions(const std::vector<double>& joint_angles)
{
    // 更新机器人姿态
    updateRobotPose(joint_angles);

    bool collision_detected = false;

    // 1. 检查自碰撞
    for (size_t i = 0; i < linkCollisionObjects.size() && !collision_detected; ++i) {
        for (size_t j = i + 2; j < linkCollisionObjects.size(); ++j) {
            // 跳过相邻连杆和某些特定组合
            if ((i == 0 && j == 2) || (i == 1 && j == 3)) continue;

            btCollisionObject* objA = linkCollisionObjects[i];
            btCollisionObject* objB = linkCollisionObjects[j];

            btCollisionObjectWrapper wrapA(nullptr, objA->getCollisionShape(), objA, objA->getWorldTransform(), -1, -1);
            btCollisionObjectWrapper wrapB(nullptr, objB->getCollisionShape(), objB, objB->getWorldTransform(), -1, -1);

            btCollisionAlgorithm* algorithm =
                dispatcher->findAlgorithm(&wrapA, &wrapB, nullptr, BT_CLOSEST_POINT_ALGORITHMS);

            btManifoldResult contactPointResult(&wrapA, &wrapB);
            algorithm->processCollision(&wrapA, &wrapB, collisionWorld->getDispatchInfo(), &contactPointResult);

            btPersistentManifold* manifold = contactPointResult.getPersistentManifold();
            if (manifold && manifold->getNumContacts() > 0) {
                collision_detected = true;
                std::cout << "   自碰撞: " << getJointName(i) << " 与 " << getJointName(j) << std::endl;
            }

            dispatcher->freeCollisionAlgorithm(algorithm);

            if (collision_detected) break;
        }
    }

    // 2. 检查与障碍物的碰撞
    if (!collision_detected) {
        for (size_t i = 0; i < linkCollisionObjects.size() && !collision_detected; ++i) {
            for (size_t j = 0; j < obstacleCollisionObjects.size(); ++j) {
                btCollisionObject* linkObj = linkCollisionObjects[i];
                btCollisionObject* obstacleObj = obstacleCollisionObjects[j];

                btCollisionObjectWrapper wrapA(nullptr, linkObj->getCollisionShape(),
                    linkObj, linkObj->getWorldTransform(), -1, -1);
                btCollisionObjectWrapper wrapB(nullptr, obstacleObj->getCollisionShape(),
                    obstacleObj, obstacleObj->getWorldTransform(), -1, -1);

                btCollisionAlgorithm* algorithm =
                    dispatcher->findAlgorithm(&wrapA, &wrapB, nullptr, BT_CLOSEST_POINT_ALGORITHMS);

                btManifoldResult contactPointResult(&wrapA, &wrapB);
                algorithm->processCollision(&wrapA, &wrapB, collisionWorld->getDispatchInfo(), &contactPointResult);

                btPersistentManifold* manifold = contactPointResult.getPersistentManifold();
                if (manifold && manifold->getNumContacts() > 0) {
                    collision_detected = true;
                    std::cout << "   障碍物碰撞: " << getJointName(i) << " 与 障碍物" << (j + 1) << std::endl;
                }

                dispatcher->freeCollisionAlgorithm(algorithm);

                if (collision_detected) break;
            }
        }
    }

    return collision_detected;
}

// ============================================================================
// 辅助函数：从RPY创建四元数
// ============================================================================
btQuaternion Collision_Joint_Checker::quatFromRPY_URDF(double roll, double pitch, double yaw)
{
    btQuaternion q;
    q.setEulerZYX(yaw, pitch, roll);
    return q;
}

// ============================================================================
// 辅助函数：创建复合形状
// ============================================================================
btCompoundShape* Collision_Joint_Checker::makeCompound(btCollisionShape* child,
    const btVector3& local_xyz,
    const btQuaternion& local_q)
{
    btCompoundShape* comp = new btCompoundShape();
    btTransform localT;
    localT.setIdentity();
    localT.setOrigin(local_xyz);
    localT.setRotation(local_q);
    comp->addChildShape(localT, child);
    return comp;
}