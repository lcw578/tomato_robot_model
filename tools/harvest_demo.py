"""采摘闭环演示 + 可达性抽检 (tomato_picker.xml, 含温室环境).

状态机 (对应论文 Figure 18):
  home -> 升降到作业高度 -> 接近点 C (果柄外 8cm) -> 采摘点 B (cut_point)
       -> 闭爪 -> [weld 切换: 断开植株 hold, 接通夹爪 grip] -> 回撤
       -> 筐上方 -> 松爪 -> 果串落筐 -> 进筐判定

用法:
  python3 tools/harvest_demo.py                 # 摘最近的一串成熟果
  python3 tools/harvest_demo.py --target p7_t1  # 指定果串
  python3 tools/harvest_demo.py --reach         # 全部果串可达性抽检 (IK)
"""
import argparse, json, sys
import numpy as np
import mujoco

XML = '/data/robot_assembly/model/tomato_picker.xml'
ARM = ['shoulder_joint', 'upperArm_joint', 'foreArm_joint',
       'wrist1_joint', 'wrist2_joint', 'wrist3_joint']
GEAR_HOME, GEAR_CLOSED = -0.474, 0.463
LIFT_KP_STEP = 400          # 升降到位所需步数
LAMBDA = 0.08               # DLS 阻尼
TOL = 0.006                 # IK 成功阈值 (m)
AXIS_W = 1.0                # 工具轴约束权重
COND_SOFT, COND_HARD = 50, 200   # 条件数缩放阈值 (lara_style_tracker)
MU_REST = 0.02              # q_rest Tikhonov 偏置 (破肘部翻转歧义)
ROT_ERR_HOLD = 2.2          # 转向误差 park 门限 (rad, >126°)
POS_ERR_MAX = 0.45          # 超工作空间截断 (防伸直锁死)
VMAX = np.array([2.618]*3 + [3.142]*3)   # 厂商关节速度限 (rad/s, i5 手册)
JOINT_MARGIN = 0.1          # 关节边界裕度


class Picker:
    def __init__(self):
        self.m = mujoco.MjModel.from_xml_path(XML)
        self.d = mujoco.MjData(self.m)
        self.tick = None          # 可视化钩子: 每个步进循环里周期调用 (harvest_view 用)
        self.stage = '初始化'
        self.cmd = None                            # 积分型关节位置命令 (LARA 语义)
        self._fk = mujoco.MjData(self.m)           # 命令轨迹 FK 专用 scratch
        m = self.m
        self.qadr = [m.jnt_qposadr[m.joint(n).id] for n in ARM]
        self.aadr = [m.actuator(a).id for a in ['j1', 'j2', 'j3', 'j4', 'j5', 'j6']]
        self.gear_a = m.actuator('gear_servo').id
        self.lift_a = m.actuator('lift_servo').id
        self.lift_j = m.joint('lift_joint').id
        self.machine_a = m.actuator('machine_servo').id
        self.machine_q = m.jnt_qposadr[m.joint('machine_slide').id]
        self.wheel_l = m.actuator('act_left_servo').id
        self.wheel_r = m.actuator('act_right_servo').id
        self.tcp = m.body('gear_link').id
        self.tool_frame = m.body('base_link').id  # 固定工具框架，不随夹爪电机开闭转动
        self.arm_dofs = [m.jnt_dofadr[m.joint(n).id] for n in ARM]
        self._last_path = None
        # TCP = 指尖中点: 沿工具轴 (gear 系 +z, 即腕部相机光轴方向) 偏 0.10m
        self.tcp_off = np.array([0.0, 0.0, 0.10])
        self.home_ctrl = np.array([-0.474, 0, 0, -1.59, -0.0611, 1.5, -1.59, -1.65, 3.05, 0.0, 0.0])
        self.trusses = self._enumerate_trusses()
        # 碰撞规划: 臂几何集 (含夹爪/相机) 与禁碰集 (植株主茎)
        import re as _re
        self.arm_geoms, self.forbid_geoms = set(), set()
        for g in range(m.ngeom):
            nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ''
            if _re.match(r'a_|base_link_mesh|gear_link_mesh|(left|right)_finger_link_mesh'
                         r'|cam_part|ad_part', nm):
                self.arm_geoms.add(g)
            elif _re.match(r'g_v\d+_stem', nm):
                self.forbid_geoms.add(g)
            else:
                bn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or ''
                if bn in ('chassis', 'lift'):        # 车体自身也参与规划避碰
                    self.forbid_geoms.add(g)

    def _enumerate_trusses(self):
        m = self.m
        out = []
        for s in range(m.nsite):
            nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SITE, s) or ''
            if not nm.startswith('cut_'):
                continue
            _, p, k = nm.split('_')          # cut_p{p}_t{k}
            rip = 'unknown'
            for g in range(m.ngeom):
                gn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ''
                if gn == f'g_tr{int(p[1:]) * 4 + int(k[1:])}_mesh':
                    mat = m.geom_matid[g]
                    mn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MATERIAL, mat) or ''
                    rip = mn.replace('mat_', '')
                    break
            out.append(dict(name=f'{p[1:]}_{k[1:]}', site=nm, ripe=rip in ('red', 'orange')))
        return out

    def reset(self, lift=0.0):
        m, d = self.m, self.d
        x_keep = float(d.qpos[self.machine_q])     # 保留底盘当前位置 (关键帧复位会拉回 x=0)
        mujoco.mj_resetDataKeyframe(m, d, 0)
        d.qpos[self.machine_q] = x_keep
        d.ctrl[:] = self.home_ctrl
        d.ctrl[self.lift_a] = lift
        d.ctrl[self.machine_a] = x_keep
        self.cmd = None                            # 复位后命令重锚到实测 (断流重锚语义)
        self._last_path = None
        mujoco.mj_forward(m, d)

    def _steps(self, n, every=2):
        """步进 n 步, 周期触发可视化钩子"""
        for i in range(n):
            mujoco.mj_step(self.m, self.d)
            if self.tick and i % every == 0:
                self.tick()

    # ---------- 基础动作 ----------
    def set_lift(self, lift, settle=True):
        self.d.ctrl[self.lift_a] = lift
        if settle:
            self._steps(LIFT_KP_STEP)

    def solve_ik(self, target, iters=300, axis_target=None):
        """离线数值 IK: 在 scratch 上迭代 (不步进物理), 多种子重启.
        axis_target 给定 -> 两阶段 (先纯旋转对齐工具轴, 再位置+姿态联合) ——
        沿果柄的抓取位姿使主茎落在指尖之外 (物理正确, 论文式 ±30° 朝向的精确化)。
        返回关节目标; 无解返回 None."""
        m = self.m
        dofs = [m.jnt_dofadr[m.joint(n).id] for n in ARM]
        qadr = self.qadr
        q_rest = np.array([-1.59, -0.0611, 1.5, -1.59, -1.65, 3.05])
        tgt_ax = np.array(axis_target, dtype=float) if axis_target is not None else None
        base_q = self.d.qpos[qadr].copy()
        best_q, best_e = None, float('inf')
        best_col_q = None                          # 备选: 收敛但带碰的解

        def fk(q):
            self._fk.qpos[:] = self.d.qpos
            self._fk.qpos[qadr] = q
            mujoco.mj_forward(m, self._fk)
            R = self._fk.xmat[self.tcp].reshape(3, 3)
            return R, self._fk.xpos[self.tcp] + R @ self.tcp_off

        def jac():
            jp = np.zeros((3, m.nv))
            jr = np.zeros((3, m.nv))
            mujoco.mj_jacBody(m, self._fk, jp, jr, self.tcp)
            off = self._fk.xmat[self.tcp].reshape(3, 3) @ self.tcp_off
            Jv = jp[:, dofs] + np.cross(jr[:, dofs].T, off).T
            return Jv, jr[:, dofs]

        def step(q, J, err, lam, clip):
            dq = J.T @ np.linalg.solve(J @ J.T + lam * np.eye(J.shape[0]), err)
            return np.clip(q + np.clip(dq, -clip, clip),
                           -(3.0543 - JOINT_MARGIN), 3.0543 - JOINT_MARGIN)

        col_tries = 0
        for seed in range(10):
            q = base_q.copy() if seed == 0 else \
                q_rest + np.random.default_rng(seed * 7 + 1).uniform(-0.6, 0.6, 6)
            col_flag = False
            if tgt_ax is not None:
                # Phase A: 纯旋转对齐 (150 迭代, 大步)
                for _ in range(150):
                    R, _p = fk(q)
                    ax = R[:, 2]
                    e_ang = float(np.arccos(np.clip(tgt_ax @ ax, -1, 1)))
                    if e_ang < 0.04:
                        break
                    axis = np.cross(ax, tgt_ax)
                    na = np.linalg.norm(axis)
                    if na < 1e-9:
                        axis = np.cross(ax, [1.0, 0, 0])
                        axis /= max(np.linalg.norm(axis), 1e-9)
                    else:
                        axis /= na
                    _Jv, Jr = jac()
                    q = step(q, Jr, AXIS_W * axis * min(e_ang, 1.0), 5e-3, 0.05)
            # Phase B: 位置 + 姿态 (或纯位置)
            for _ in range(iters * 2 if tgt_ax is not None else iters):
                R, p = fk(q)
                err = [target - p]
                rot = False
                if tgt_ax is not None:
                    ax = R[:, 2]
                    e_ang = float(np.arccos(np.clip(tgt_ax @ ax, -1, 1)))
                    if e_ang > 1e-6:
                        axis = np.cross(ax, tgt_ax)
                        na = np.linalg.norm(axis)
                        if na > 1e-9:
                            err.append(AXIS_W * (axis / na) * min(e_ang, 1.0))
                            rot = True
                else:
                    e_ang = 0.0
                err = np.concatenate(err)
                e_pos = float(np.linalg.norm(err[:3]))
                if e_pos < best_e:
                    best_e, best_q = e_pos, q.copy()
                if e_pos < TOL and e_ang < 0.06:
                    if self.collision_free(q):     # 解算须避开植株/车体: 优先无碰解
                        return q.copy()
                    if best_col_q is None:
                        best_col_q = q.copy()      # 记录带碰解, 换下一颗种子
                    col_flag = True
                    break
                Jv, Jr = jac()
                J = np.vstack([Jv, Jr]) if rot else Jv
                q = step(q, J, err, 5e-3, 0.05)
            # 修复 (2026-10-17): 原实现在种子 0 收敛带碰后立即 return best_col_q,
            # 其余种子从不尝试 ("多种子换无碰解"形同虚设)。现改为: 带碰则换下一颗
            # 种子 (有界 4 颗, 控制解算耗时), 全部试完才回退带碰解。
            if col_flag:
                col_tries += 1
                if col_tries >= 4:
                    break
        if best_col_q is not None:
            return best_col_q
        if best_e > 0.05:
            return None
        return best_q

    # ---------- 碰撞检测与规划 (关节空间 RRT, 直线优先) ----------
    def penetration(self, q):
        """scratch FK 下臂-环境及模型启用的臂自碰撞最大穿透 (m)。"""
        fd = self._fk
        fd.qpos[:] = self.d.qpos
        fd.qpos[self.qadr] = q
        mujoco.mj_forward(self.m, fd)
        worst = 0.0
        for c in range(fd.ncon):
            if fd.contact[c].dist >= 0:
                continue
            g1, g2 = fd.contact[c].geom1, fd.contact[c].geom2
            if self._forbidden_pair(g1, g2):
                worst = max(worst, -fd.contact[c].dist)
        return worst

    def shear_hit_at(self, q, cut, att):
        """在给定关节位姿下复算剪切判据 (果柄线段 attach->cut ∩ 剪切区盒)。"""
        fd = self._fk
        fd.qpos[:] = self.d.qpos
        fd.qpos[self.qadr] = q
        mujoco.mj_forward(self.m, fd)
        R = fd.xmat[self.tcp].reshape(3, 3)
        p = fd.xpos[self.tcp]
        seg = (R.T @ (np.array([att, cut]) - p).T).T
        for tt in np.linspace(0, 1, 25):
            qq = seg[0] * (1 - tt) + seg[1] * tt
            if 0.01 <= qq[2] <= 0.115 and np.hypot(qq[0], qq[1]) <= 0.026:
                return True
        return False

    def _forbidden_pair(self, g1, g2):
        return ((g1 in self.arm_geoms and g2 in self.forbid_geoms) or
                (g2 in self.arm_geoms and g1 in self.forbid_geoms) or
                (g1 in self.arm_geoms and g2 in self.arm_geoms))

    def collision_free(self, q, pen=0.003):
        """臂-环境及模型启用的臂自碰撞；允许的穿透上限为 pen。"""
        fd = self._fk
        fd.qpos[:] = self.d.qpos
        fd.qpos[self.qadr] = q
        mujoco.mj_forward(self.m, fd)
        for c in range(fd.ncon):
            if fd.contact[c].dist > -pen:
                continue
            g1, g2 = fd.contact[c].geom1, fd.contact[c].geom2
            if self._forbidden_pair(g1, g2):
                return False
        return True

    def _seg_free(self, qa, qb, n=8, pen=0.003):
        # 按关节变化加密；插值与执行端使用同一条关节线段。
        n = max(n, int(np.ceil(np.max(np.abs(qb - qa)) / 0.02)))
        for t in np.linspace(0, 1, n + 1)[1:]:
            if not self.collision_free(qa * (1 - t) + qb * t, pen=pen):
                return False
        return True

    def plan_path(self, q_goal, max_iter=15000, step=0.15, time_budget=20.0):
        """RRT-Connect；返回包含起点、终点的路径，所有边使用同一碰撞容差。"""
        q0 = self.cmd.copy()
        self._last_path = None
        if not self.collision_free(q_goal):
            return None
        if self._seg_free(q0, q_goal):
            self._last_path = [q0, q_goal.copy()]
            return self._last_path
        import time as _time
        _t0 = _time.time()
        rng = np.random.default_rng(0)
        bounds = 2.9

        def nearest(arr, n, q):
            return int(np.argmin(np.linalg.norm(arr[:n] - q, axis=1)))

        def step_from(qa, qb, s_):
            v = qb - qa
            L = np.linalg.norm(v)
            if L < 1e-9:
                return qa.copy()
            return np.clip(qa + v / L * min(s_, L),
                           -(3.0543 - JOINT_MARGIN), 3.0543 - JOINT_MARGIN)

        cap = max_iter + 8
        nodesA = np.zeros((cap, 6)); nodesA[0] = q0; nA = 1; parA = [-1]
        nodesB = np.zeros((cap, 6)); nodesB[0] = q_goal; nB = 1; parB = [-1]

        def path_from(nodes, par, k, reverse=False):
            out = []
            while k != -1:
                out.append(nodes[k].copy()); k = par[k]
            if reverse:
                out = out[::-1]
            return out

        for i in range(max_iter):
            if (i & 63) == 0 and _time.time() - _t0 > time_budget:
                break
            # A 树扩展 (目标偏置 50%)
            if i % 2 == 1:
                q_rand = q_goal + rng.normal(0, 0.25, 6)
            else:
                q_rand = rng.uniform(-bounds, bounds, 6)
            ia = nearest(nodesA, nA, q_rand)
            q_a = step_from(nodesA[ia], q_rand, step)
            if not self._seg_free(nodesA[ia], q_a, n=8):
                continue
            nodesA[nA] = q_a; parA.append(ia); nA += 1
            # B 树朝 A 新点贪心连接
            ib = nearest(nodesB, nB, q_a)
            while nB < cap:
                q_b = step_from(nodesB[ib], q_a, step)
                if not self._seg_free(nodesB[ib], q_b, n=8):
                    break
                nodesB[nB] = q_b; parB.append(ib); nB += 1
                if np.linalg.norm(q_b - q_a) < 1e-6 or \
                        self._seg_free(q_b, q_a, n=8):
                    # A: 根->叶，B: 叶->根。保留两树连接边，不能跳过 q_b。
                    path = path_from(nodesA, parA, nA - 1, reverse=True) + \
                           path_from(nodesB, parB, nB - 1)
                    self._last_path = path
                    return path
                ib = nB - 1
        print('  [plan] RRT-Connect %d 迭代未连通 (A=%d B=%d 节点)' % (max_iter, nA, nB), flush=True)
        return None          # 规划失败 (被植株完全挡住)

    def tcp_pos(self):
        R = self.d.xmat[self.tcp].reshape(3, 3)
        return self.d.xpos[self.tcp] + R @ self.tcp_off

    def _pose_at(self, q):
        fd = self._fk
        fd.qpos[:] = self.d.qpos
        fd.qpos[self.qadr] = q
        mujoco.mj_forward(self.m, fd)
        R = fd.xmat[self.tcp].reshape(3, 3)
        return (fd.xpos[self.tcp] + R @ self.tcp_off,
                fd.xmat[self.tool_frame].reshape(3, 3).copy())

    @staticmethod
    def _rotation_error(target, actual):
        quat, rotvec = np.zeros(4), np.zeros(3)
        mujoco.mju_mat2Quat(quat, (target @ actual.T).reshape(9))
        if quat[0] < 0:
            quat *= -1
        mujoco.mju_quat2Vel(rotvec, quat, 1.0)
        return rotvec

    def solve_pose(self, target, rotation, seed):
        """连续局部 IK：约束 TCP 位置与固定工具框架完整姿态，不切换随机分支。"""
        q = seed.copy()
        jp, jr = np.zeros((3, self.m.nv)), np.zeros((3, self.m.nv))
        for _ in range(100):
            p, R = self._pose_at(q)
            ep, er = target - p, self._rotation_error(rotation, R)
            if np.linalg.norm(ep) < 0.0001 and np.linalg.norm(er) < 0.001:
                return q
            mujoco.mj_jacBody(self.m, self._fk, jp, jr, self.tcp)
            off = self._fk.xmat[self.tcp].reshape(3, 3) @ self.tcp_off
            Jv = jp[:, self.arm_dofs] + np.cross(jr[:, self.arm_dofs].T, off).T
            J = np.vstack([Jv, 0.15 * jr[:, self.arm_dofs]])
            err = np.r_[ep, 0.15 * er]
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-5 * np.eye(6), err)
            q = np.clip(q + np.clip(dq, -0.08, 0.08),
                        -(3.0543 - JOINT_MARGIN), 3.0543 - JOINT_MARGIN)
        return None

    def plan_cut(self, cut, approach, axis, attach):
        """先选剪切完整姿态，再反向求同姿态直线；最后检查从当前状态到外侧的路径。"""
        q_b0 = self.solve_ik(cut, iters=800, axis_target=axis)
        if q_b0 is None:
            return None
        candidates = []
        bound = 3.0543 - JOINT_MARGIN
        for deg in (0, -15, 15, -30, 30, -45, 45, -60, 60, -75, 75,
                    -90, 90, -105, 105, -120, 120, -135, 135, -150, 150, -165, 165, -180):
            q = q_b0.copy()
            q[5] += np.radians(deg)
            if abs(q[5]) > bound:
                continue  # 限位不是角度取模，不能把不可行 roll 截断成另一候选
            _, rotation = self._pose_at(q)
            q = self.solve_pose(cut, rotation, q)
            if q is None or not self.shear_hit_at(q, cut, attach):
                continue
            self._pose_at(q)
            tool_axis = self._fk.xmat[self.tcp].reshape(3, 3)[:, 2]
            if np.arccos(np.clip(tool_axis @ axis, -1, 1)) > 0.06:
                continue
            path, worst = [q.copy()], self.penetration(q)
            if worst > 0.003:
                continue
            for s in np.linspace(0, 1, 21)[1:]:
                qn = self.solve_pose(cut + s * (approach - cut), rotation, path[-1])
                if qn is None or np.max(np.abs(qn - path[-1])) > 0.10 or \
                        not self._seg_free(path[-1], qn):
                    break
                worst = max(worst, self.penetration(qn))
                path.append(qn)
            if len(path) != 21:
                continue
            insertion = path[::-1]
            # 1mm 以内穿透近似等价；再偏好较小的总关节变化与末轴变化。
            score = (1000 * max(0.0, worst - 0.001) +
                     np.linalg.norm(insertion[0] - self.cmd) +
                     0.25 * abs(insertion[0][5] - self.cmd[5]))
            candidates.append((score, deg, worst, insertion, rotation))
        for _, deg, worst, insertion, rotation in sorted(candidates, key=lambda c: c[0]):
            approach_path = self.plan_path(insertion[0], time_budget=2.0, max_iter=3000)
            if approach_path is not None:
                return dict(approach=approach_path, insertion=insertion,
                            rotation=rotation, roll_deg=deg, penetration=worst)
        return None

    def point_tool_down(self, iters=80):
        """转动 j5/j6 使工具轴 (gear +z) 竖直向下, 其余关节不动 (果串随之垂挂)."""
        m, d = self.m, self.d
        qadr = [m.jnt_qposadr[m.joint(n).id] for n in ('wrist2_joint', 'wrist3_joint')]
        dofr = [m.jnt_dofadr[m.joint(n).id] for n in ('wrist2_joint', 'wrist3_joint')]
        tgt = np.array([0.0, 0.0, -1.0])

        def axis():
            mujoco.mj_forward(m, d)
            return d.xmat[self.tcp].reshape(3, 3)[:, 2].copy()

        for it in range(iters):
            ax0 = axis()
            e = tgt - ax0
            if np.linalg.norm(e) < 0.01:
                break
            if self.tick and it % 2 == 0:
                self.tick()
            J = np.zeros((3, 2))
            for jj, (qa, df) in enumerate(zip(qadr, dofr)):
                d.qpos[qa] += 0.02
                mujoco.mj_forward(m, d)
                J[:, jj] = (d.xmat[self.tcp].reshape(3, 3)[:, 2] - ax0) / 0.02
                d.qpos[qa] -= 0.02
            dq = np.linalg.solve(J.T @ J + 0.02 * np.eye(2), J.T @ e)
            for jj, (qa, aa) in enumerate(zip(qadr, [self.aadr[4], self.aadr[5]])):
                d.qpos[qa] = np.clip(d.qpos[qa] + np.clip(dq[jj], -0.06, 0.06),
                                     -3.0543, 3.0543)
                d.ctrl[aa] = d.qpos[qa]
            self._steps(1)
        self._steps(200)
        return float(np.linalg.norm(tgt - axis()))

    def _walk_to(self, q_target, speed=0.8, observe=None):
        """共同进度的五次插值；两次物理步/命令，保持旧执行器的有效速度上限。"""
        q_target = np.asarray(q_target, dtype=float)
        if (not np.isfinite(speed) or speed <= 0 or q_target.shape != (6,) or
                not np.all(np.isfinite(q_target)) or np.any(np.abs(q_target) > 3.0543)):
            raise ValueError('非法关节目标或速度')
        if self.cmd is None:
            self.cmd = self.d.qpos[self.qadr].copy()
        q0 = self.cmd.copy()
        delta = q_target - q0
        if np.max(np.abs(delta)) < 1e-9:
            return True
        dt = 2 * self.m.opt.timestep
        # 五次进度曲线最大导数 1.875；关节保持相同比例，不逐关节截断。
        duration = max(0.06, 1.875 * np.max(np.abs(delta) / (VMAX * speed * 0.5)))
        n = int(np.ceil(duration / dt))
        for i in range(n):
            u = (i + 1) / n
            s = 10*u**3 - 15*u**4 + 6*u**5
            self.cmd = q0 + s * delta
            self.d.ctrl[self.aadr] = self.cmd
            mujoco.mj_step(self.m, self.d)
            mujoco.mj_step(self.m, self.d)
            if observe is not None and not observe():
                # 异常停止：保持当前测量位置，避免继续追赶后续命令。
                self.cmd = self.d.qpos[self.qadr].copy()
                self.d.ctrl[self.aadr] = self.cmd
                return False
            if self.tick:
                self.tick()
        return True

    def execute_insertion(self, plan, approach, cut, log):
        """执行同姿态进刀，逐命令检查真实轨迹；失败不进入闭爪状态。"""
        self.stage = 'STRAIGHT INSERT'
        self._last_path = [q.copy() for q in plan['insertion']]
        axis = (cut - approach) / np.linalg.norm(cut - approach)
        samples, deviations, rotations, penetrations = [], [], [], []

        def observe():
            p = self.tcp_pos().copy()
            v = p - approach
            off = np.linalg.norm(v - (v @ axis) * axis)
            R = self.d.xmat[self.tool_frame].reshape(3, 3)
            angle = np.linalg.norm(self._rotation_error(plan['rotation'], R))
            pen = max((max(0.0, -float(c.dist)) for c in self.d.contact[:self.d.ncon]
                       if self._forbidden_pair(c.geom1, c.geom2)), default=0.0)
            samples.append(p); deviations.append(off); rotations.append(angle); penetrations.append(pen)
            return (off <= 0.01 and angle <= 0.05 and pen <= 0.003 and
                    np.max(np.abs(self.d.qpos[self.qadr] - self.cmd)) <= 0.08)

        ok = observe()
        if ok:
            for q in plan['insertion'][1:]:
                if not self._walk_to(q, speed=0.5, observe=observe):
                    ok = False
                    break
        if ok:
            for i in range(100):
                mujoco.mj_step(self.m, self.d)
                if i % 2 == 0:
                    if not observe():
                        ok = False
                        break
                    if self.tick:
                        self.tick()
        e = float(np.linalg.norm(self.tcp_pos() - cut))
        log.update(reach_err=round(e, 4),
                   insertion_path_mm=round(float(np.linalg.norm(np.diff(samples, axis=0), axis=1).sum()) * 1000, 2),
                   insertion_deviation_mm=round(max(deviations) * 1000, 2),
                   insertion_rotation_deg=round(float(np.degrees(max(rotations))), 2),
                   insertion_penetration_mm=round(max(penetrations) * 1000, 2))
        ok = ok and e <= 0.005 and rotations[-1] <= 0.03
        if not ok:
            self._last_path = None  # 未走完的路径不能冒充可原路回撤的历史。
            self.cmd = self.d.qpos[self.qadr].copy()
            self.d.ctrl[self.aadr] = self.cmd
        return ok

    def goto_reverse(self, speed=0.8):
        """沿上一条规划路径反向退回 (抓取位姿常在自己植株茎秆的包围袋里, 原路是唯一通道)。"""
        path = getattr(self, '_last_path', None)
        if not path:
            return False
        # 当前负载可能改变接触，退出前重新核对臂自碰撞/环境碰撞。
        previous = self.cmd
        for wp in path[-2::-1]:
            if not self._seg_free(previous, wp):
                self._last_path = None
                return False
            previous = wp
        for wp in path[-2::-1]:
            self._walk_to(wp, speed=speed)
        self._last_path = None
        return True

    def goto(self, target, iters=300, axis_target=None, speed=0.8):
        """离线解算 -> 碰撞规划 -> 同步关节插值；直线进刀由 plan_cut 单独处理。"""
        target = np.asarray(target, dtype=float)
        if self.cmd is None:
            self.cmd = self.d.qpos[self.qadr].copy()   # 命令重锚到实测
        # 轴对齐模式: 目标补偿迭代 —— 按测量残差修正解算目标 (消系统性局部偏差)
        if axis_target is not None:
            tgt = target.copy()
            q_best, e_best = None, 1e9
            for _round in range(6):
                q = self.solve_ik(tgt, iters=iters, axis_target=axis_target)
                if q is None:
                    break
                self._fk.qpos[:] = self.d.qpos
                self._fk.qpos[self.qadr] = q
                mujoco.mj_forward(self.m, self._fk)
                R = self._fk.xmat[self.tcp].reshape(3, 3)
                p = self._fk.xpos[self.tcp] + R @ self.tcp_off
                e_vec = target - p
                if np.linalg.norm(e_vec) < e_best:
                    e_best, q_best = float(np.linalg.norm(e_vec)), q.copy()
                if e_best < 0.008:
                    break
                tgt = tgt + e_vec              # 补偿: 下轮目标加上残差
            q_t = q_best
        else:
            q_t = self.solve_ik(target, iters=iters)
        if q_t is None:
            print('  [goto] 求解失败 target=%s axis=%s' % (np.round(target, 3), axis_target is not None))
            return False, float('inf')
        path = self.plan_path(q_t)
        if path is None:
            print('  [goto] 规划失败 (被植株挡住)')
            return False, float('inf')                 # 被植株挡住, 诚实失败
        print('  [goto] wp=%d' % len(path), flush=True)
        for wp in path:
            self._walk_to(wp, speed=speed)
        mujoco.mj_forward(self.m, self.d)
        e = float(np.linalg.norm(self.tcp_pos() - target))
        return e < 0.05, e

    def gear(self, pos, steps=250):
        self.d.ctrl[self.gear_a] = pos
        self._steps(steps)

    # ---------- 果串工具 ----------
    def cut_site_pos(self, name):
        return self.d.site_xpos[self.m.site(name).id].copy()

    def eq_ids(self, truss):
        hold = self.m.equality(f'hold_{truss}').id
        grip = self.m.equality(f'grip_{truss}').id
        return hold, grip

    def harvest(self, truss, snap=None, cut_override=None):
        """对一串果执行完整采集. truss 形如 '7_1'.
        cut_override: 感知给出的世界系剪切点 (混合方案); None 时用真值 site."""
        m, d = self.m, self.d
        site = f'cut_p{truss.split("_")[0]}_t{truss.split("_")[1]}'
        sid = m.site(site).id
        # 目标点先在 home/lift=0 下读一次世界坐标, 升降只影响臂不影响目标
        mujoco.mj_forward(m, d)
        if cut_override is not None:
            cut = np.asarray(cut_override, dtype=float).copy()
        else:
            cut = d.site_xpos[sid].copy()
        # 接近点: 从果串质心指向 cut 的水平方向外推 8cm (背离主茎朝走道)
        tr_body = m.body(f'truss_{truss.split("_")[0]}_{truss.split("_")[1]}').id
        mujoco.mj_forward(m, d)
        lateral = cut - d.xipos[tr_body]
        lateral[2] = 0
        nrm = lateral / max(np.linalg.norm(lateral), 1e-6)   # 果簇->藤蔓 方向 (判据/接近用)
        base = d.xpos[m.body('arm_base_mount').id]
        to_robot = cut - base
        to_robot[2] = 0
        to_robot /= max(np.linalg.norm(to_robot), 1e-6)      # 基座 -> 目标 (从走道伸向果串)
        dvec = cut - d.xipos[tr_body]
        dvec /= max(np.linalg.norm(dvec), 1e-9)              # 沿果柄 (指向主茎; 仅记录)
        approach = cut - to_robot * 0.10

        if cut[2] < 0.9:     # 低于论文作业带 (1.1-1.5m), 且低于台面板投影 -> 几何不可达
            log = dict(truss=truss, cut=np.round(cut, 3).tolist(), result='below_envelope')
            self.stage = 'BELOW ENVELOPE'
            return log
        # 抬升预检 (能低不升): 0 -> 0.25 -> 0.5, 解算全通过且臂未接近全伸
        # (基座->目标距离 <= 0.90m) 才接受; 无一通过则用最大抬升兜底
        self.stage = 'LIFT PRECHECK'
        base0 = d.xpos[m.body('arm_base_mount').id].copy()
        lift, best = 0.5, None
        for L in (0.0, 0.25, 0.5):
            self.reset(lift=L)
            self.set_lift(L)
            self.cmd = None
            q_c = self.solve_ik(approach, iters=600)
            # B = 横剪抓取: 工具轴水平指向走道侧 (to_robot), TCP 停在剪切点 ——
            # 腕部留在开放空间, 刀口从侧面横切竖直果柄 (送刀口深处=TCP 距齿轮 0.10)。
            # 旧式轴对齐沿 dvec: 切点上移到簇顶后 dvec 变竖直朝上, 整只手会被
            # 兜进果堆 (穿模 5.1cm) 且迫使臂绕下方入位, 见 model_notes 2026-10-12
            q_b = self.solve_ik(cut, iters=800, axis_target=to_robot)
            if q_b is None:
                q_b = self.solve_ik(cut, iters=800)
            e_b = float('inf')
            if q_b is not None:
                self._fk.qpos[:] = d.qpos
                self._fk.qpos[self.qadr] = q_b
                mujoco.mj_forward(m, self._fk)
                R = self._fk.xmat[self.tcp].reshape(3, 3)
                pp = self._fk.xpos[self.tcp] + R @ self.tcp_off
                e_b = float(np.linalg.norm(pp - cut))
            ok = (q_c is not None) and (q_b is not None) and (e_b <= 0.005)
            print('  [lift] 尝试 %.2f: 解算=%s B残差=%.4f -> %s' % (
                L, 'OK' if (q_c is not None and q_b is not None) else '失败', e_b,
                '采纳' if ok else '不满足'), flush=True)
            if best is None and q_c is not None and q_b is not None:
                best = L
            if ok:
                lift = L
                break
        else:
            lift = best if best is not None else 0.5
        log = dict(truss=truss, lift=round(lift, 3), cut=np.round(cut, 3).tolist())
        self.reset(lift=lift)
        self.set_lift(lift)

        self.stage = 'PLAN CUT'
        self.cmd = d.qpos[self.qadr].copy()
        _pp, _kk = truss.split('_')
        att_w = d.site_xpos[m.site(f'attach_p{_pp}_t{_kk}').id].copy()
        plan = self.plan_cut(cut, approach, to_robot, att_w)
        if plan is None:
            log['result'] = 'cut_path_unavailable'
            self.stage = 'CUT PATH UNAVAILABLE'
            return log
        log.update(axis_grasp=True, b_roll_deg=float(plan['roll_deg']),
                   b_stem_pen_mm=round(plan['penetration'] * 1000, 2))
        print('  [cut plan] roll=%d°, 直线进刀=%d点, 最大静态穿透=%.2fmm' % (
            plan['roll_deg'], len(plan['insertion']), plan['penetration'] * 1000), flush=True)
        self.stage = 'PREPARE CUT POSE'
        for q in plan['approach'][1:]:
            self._walk_to(q, speed=0.6)
        self._steps(100)
        mujoco.mj_forward(m, d)
        eC = float(np.linalg.norm(self.tcp_pos() - approach))
        log['approach_err'] = round(eC, 4)
        R = d.xmat[self.tool_frame].reshape(3, 3)
        if eC > 0.005 or np.linalg.norm(self._rotation_error(plan['rotation'], R)) > 0.03:
            log['result'] = 'approach_tracking_failed'
            self.stage = 'APPROACH TRACKING FAILED'
            return log
        if not self.execute_insertion(plan, approach, cut, log):
            log['result'] = 'insertion_tracking_failed'
            self.stage = 'INSERTION TRACKING FAILED'
            return log
        self.stage = 'CLOSE & CUT'
        # 剪切判据 (确定性几何): 果柄线段 (attach->cut, 即要剪的那段) 与夹爪剪切区盒相交
        # 盒 (gear 系): 轴向 [0.01, 0.115] (齿轮回转面到指端), 径向距工具轴 <= 0.026
        # (= 张开口 52mm 的一半 —— 果柄能落入张开的夹爪内部, 闭合即剪到)
        R_g = d.xmat[self.tcp].reshape(3, 3)
        p_g = d.xpos[self.tcp]                            # gear 原点 (世界)
        att = d.site_xpos[m.site(f'attach_p{truss.split("_")[0]}_t{truss.split("_")[1]}').id]
        seg_w = np.array([att, cut])                     # 果柄线段 = attach -> 剪切点 (世界)
        seg_g = (R_g.T @ (seg_w - p_g).T).T              # 变换到 gear 系
        hit = False
        for tt in np.linspace(0, 1, 25):
            q = seg_g[0] * (1 - tt) + seg_g[1] * tt
            if 0.01 <= q[2] <= 0.115 and np.hypot(q[0], q[1]) <= 0.026:
                hit = True
                break
        log['shear_hit'] = bool(hit)
        if not hit:
            # 空剪: 果柄未落入剪切区, 果串留在植株上
            self.stage = 'SHEAR MISS'
            self.gear(GEAR_CLOSED, steps=200)
            self.gear(GEAR_HOME, steps=200)
            log['result'] = 'shear_miss'
            return log
        # 闭爪 + weld 切换: 先把"当前相对位姿"写进 grip weld 再激活 (默认 relpose
        # 是编译零位下的, 直接激活会把果串猛拽走); torquescale=0 = 点抓
        # (只约束锚点平移, 果串在重力下绕剪切点自由垂挂摆动 —— 与真机一致)
        hold, grip = self.eq_ids(truss)
        base_id = m.body('base_link').id
        R1 = d.xmat[base_id].reshape(3, 3)
        R2 = d.xmat[tr_body].reshape(3, 3)
        p_cur = R1.T @ (d.xpos[tr_body] - d.xpos[base_id])
        q_cur = np.zeros(4)
        mujoco.mju_mat2Quat(q_cur, (R1.T @ R2).reshape(9))
        # 目标位姿: 果串铅垂垂挂于 TCP 正下方 (果柄竖直向下, 果簇在底)
        L_out = float(np.linalg.norm(d.xipos[tr_body] - d.xpos[tr_body]))
        v_w = (d.xipos[tr_body] - d.xpos[tr_body]) / max(L_out, 1e-9)   # 果串当前世界方向
        ax_w = np.cross(v_w, np.array([0.0, 0.0, -1.0]))
        sn = np.linalg.norm(ax_w)
        ang = float(np.arccos(np.clip(v_w @ np.array([0.0, 0.0, -1.0]), -1, 1)))
        q_d = np.zeros(4)
        if sn > 1e-9:
            mujoco.mju_axisAngle2Quat(q_d, ax_w / sn, ang)
        else:
            q_d[:] = [1, 0, 0, 0]
        q_new = np.zeros(4)
        mujoco.mju_mulQuat(q_new, q_d, q_cur)          # R2_new = R_delta @ R2
        R2_new = np.zeros(9)
        mujoco.mju_quat2Mat(R2_new, q_new)
        # 果串原点从 TCP 沿世界竖直方向再下移 HANG_DROP —— 让整串(含顶部果实)
        # 落到指板之下, 消除搬运期"果球穿指板"的残余视觉穿插 (2026-10-12)
        HANG_DROP = 0.05
        p_new_w = self.tcp_pos() + np.array([0.0, 0.0, -HANG_DROP])   # 垂挂于爪下
        p_new = R1.T @ (p_new_w - d.xpos[base_id])
        m.eq_data[grip][3:6] = p_cur                    # 从当前位姿起步
        m.eq_data[grip][6:10] = q_cur
        m.eq_data[grip][10] = 1.0
        d.eq_active[grip] = 1
        d.eq_active[hold] = 0
        # 闭爪 500 步内渐进插值 relpose -> 铅垂垂挂 (D2a: 视觉连续、确定性跟爪)
        p_tcp0 = self.tcp_pos().copy()
        for i in range(500):
            t = (i + 1) / 500
            m.eq_data[grip][3:6] = (1 - t) * p_cur + t * p_new
            qi = q_cur * (1 - t) + q_new * t            # nlerp (同半球)
            qi /= max(np.linalg.norm(qi), 1e-9)
            m.eq_data[grip][6:10] = qi
            # 闭爪命令斜坡 (2026-10-17): gear_servo 曾在 kv=0 下被阶跃命令激发
            # 142 rad/s / +519mrad 过冲振荡 = 夹爪抖动; 300 步线性拉满, 配合 kv=0.05
            ramp = min(1.0, (i + 1) / 300.0)
            d.ctrl[self.gear_a] = GEAR_HOME + (GEAR_CLOSED - GEAR_HOME) * ramp
            mujoco.mj_step(m, d)
            if self.tick and i % 2 == 0:
                self.tick()
        # 闭合时刻接近度 (2026-10-17): 果柄线段 (attach->cut) 的隐形查询胶囊
        # (不随垂挂下放移动) 与夹爪碰撞几何的最小表面距离 —— 为"闭合包络判据"
        # 提供物理量纲交叉验证。纯几何查询, 不参与成败判定; <=3mm 记 jaw_bite。
        mujoco.mj_forward(m, d)
        stub_id = m.geom(f'g_truss_{truss}_stemq').id
        jaw_d = []
        for gn in ['gear_link_mesh_1'] + [f'{s}_finger_link_mesh_{i}'
                                          for s in ('left', 'right') for i in range(1, 6)]:
            try:
                gid = m.geom(gn).id
            except KeyError:
                continue
            if m.geom_contype[gid] == 0:
                continue
            ft = np.zeros((6, 1))
            jaw_d.append(float(mujoco.mj_geomDistance(m, d, stub_id, gid, 0.05, ft)))
        if jaw_d:
            dmin = min(jaw_d)
            log['jaw_clearance_mm'] = round(dmin * 1000, 2)
            log['jaw_bite'] = bool(dmin <= 0.003)
            print('  [shear] jaw_clearance=%.1f mm%s' % (
                dmin * 1000, ' (bite 咬合)' if dmin <= 0.003 else ''), flush=True)
        log['result'] = 'cut'
        if snap:
            self._snap('_cut')
        # 完整进刀路径倒序退出冠层，再进入筐上方的自由空间运动。
        self.stage = 'RETRACT FROM PLANT'
        retract_start = self.tcp_pos().copy()
        if not self.goto_reverse(speed=0.5):
            log['result'] = 'retract_blocked'
            self.stage = 'RETRACT BLOCKED'
            return log
        self._steps(100)
        log['retract_distance_mm'] = round(float(np.linalg.norm(self.tcp_pos() - retract_start)) * 1000, 2)
        if np.linalg.norm(self.tcp_pos() - approach) > 0.005:
            log['result'] = 'retract_tracking_failed'
            self.stage = 'RETRACT TRACKING FAILED'
            return log
        self.stage = 'TRANSPORT TO BASKET'
        bw = self.basket_above()
        # 高空绕行路点: 先到筐上方 30cm (从上方越过筐口), 再降落到筐口上方 —— 把
        # 一次性大跨度的规划拆成两个小问题 (车体真实碰撞后直连规划易失败)
        bw_hi = bw.copy()
        bw_hi[2] += 0.30
        self.goto(bw_hi, iters=800)
        okT, eT = self.goto(bw, iters=800)
        if not okT:      # 30cm 直降规划失败 -> 5cm 小步下降 (每段关节直线几乎必通)
            import numpy as _np
            print('  [transport] 直降失败, 改小步下降', flush=True)
            for zz in _np.linspace(bw_hi[2], bw[2], 7)[1:]:
                self.goto(_np.array([bw[0], bw[1], zz]), iters=600)
            okT, eT = self.goto(bw, iters=600)
        bid = m.body(f'truss_{truss.split("_")[0]}_{truss.split("_")[1]}').id
        mouth_z = bw[2] - 0.22
        bwlo = bw.copy()
        bwlo[2] = mouth_z + 0.11 + HANG_DROP   # 释放深度: 指尖离筐底留 ~2cm (实测 1mm 浅擦 -> 抬升 2cm); +悬挂下移量补偿
        self.goto(bwlo, iters=400)
        self.stage = 'RELEASE'
        # 原地慢开爪; 在最终释放位姿上迭代对准 (此后到释放前无任何运动)
        self.gear(GEAR_HOME, steps=400)
        for _ in range(8):
            self._steps(400)
            mujoco.mj_forward(m, d)
            off = d.xipos[bid][:2] - bw[:2]
            log['align_err'] = round(float(np.linalg.norm(off)), 4)
            if np.linalg.norm(off) < 0.03:
                break
            corr = off.copy()
            if np.linalg.norm(corr) > 0.10:   # 限幅: 防止单次大修正引发摆动正反馈
                corr *= 0.10 / np.linalg.norm(corr)
            bw2 = bwlo.copy()
            bw2[:2] -= corr
            self.goto(bw2)
        log['transport_err'] = round(eT, 4)
        # 释放: 原地解除约束, 果串竖直落筐底; 之后手臂保持完全静止
        d.eq_active[grip] = 0
        self._steps(1000, every=4)
        if snap:
            self._snap('_above_basket')
        # 落筐判定: 筐内腔矩形 0.33x0.44 (半宽 0.167/0.222), 留 2cm 余量
        mujoco.mj_forward(m, d)
        bw0 = self.basket_above(0.0)
        off = d.xipos[bid][:2] - bw0[:2]
        inside = bool(d.xipos[bid][2] < bw0[2] + 0.02
                      and abs(off[0]) < 0.25 and abs(off[1]) < 0.28)
        log['settle_z'] = round(float(d.xipos[bid][2]), 3)
        log['settle_off'] = np.round(off, 3).tolist()
        log['result'] = 'in_basket' if inside else 'dropped'
        return log

    # ---------- 底盘移动 (轨道平移, 运动学语义) ----------
    def drive_to(self, x, speed=0.12):
        """沿轨道平移到站点 x (smoothstep 速度剖面, 驱动轮同步旋转做视觉)。"""
        x0 = float(self.d.qpos[self.machine_q])
        dist = x - x0
        if abs(dist) < 1e-3:
            return 0.0
        dt = self.m.opt.timestep * 2
        n = int(abs(dist) / (speed * dt)) + 1
        r = 0.1                                  # 驱动轮半径 (视觉滚转)
        sgn = 1.0 if dist > 0 else -1.0
        for i in range(n):
            t = (i + 1) / n
            u = t * t * (3 - 2 * t)              # smoothstep
            self.d.ctrl[self.machine_a] = x0 + dist * u
            self.d.ctrl[self.wheel_l] = sgn * speed / r
            self.d.ctrl[self.wheel_r] = sgn * speed / r
            mujoco.mj_step(self.m, self.d)
            mujoco.mj_step(self.m, self.d)
            if self.tick:
                self.tick()
        self.d.ctrl[self.wheel_l] = 0.0
        self.d.ctrl[self.wheel_r] = 0.0
        mujoco.mj_forward(self.m, self.d)
        return float(self.d.qpos[self.machine_q])

    def pick_best_truss(self, z_lo=1.0, z_hi=1.55):
        """真值选择: 成熟 + 高度带内 + 离所有主茎最远 (最好摘的一串)。"""
        mujoco.mj_forward(self.m, self.d)
        best = None
        for t in self.trusses:
            if not t['ripe']:
                continue
            cut = self.d.site_xpos[self.m.site(t['site']).id].copy()
            if not (z_lo <= cut[2] <= z_hi):
                continue
            clear = 1e9
            for gid in self.forbid_geoms:        # 主茎圆柱集
                gp = self.d.geom_xpos[gid]
                gm = self.d.geom_xmat[gid].reshape(3, 3)
                axis = gm[:, 2]
                half = self.m.geom_size[gid][1]
                v = cut - gp
                tt = float(np.clip(v @ axis, -half, half))
                clear = min(clear, float(np.linalg.norm(v - tt * axis)))
            score = (clear, -abs(cut[2] - 1.28))  # 离茎秆越远越好, 越靠近带中心越好
            if best is None or (tuple(score) > tuple(best[0])):
                best = (score, t['name'], cut, clear)
        return best

    def basket_above(self, h=0.22):
        s = self.d.site_xpos[self.m.site('basket_site').id]
        return np.array([s[0], s[1], s[2] + h])

    def _snap(self, tag):
        try:
            from PIL import Image
            r = mujoco.Renderer(self.m, 540, 960)
            r.update_scene(self.d, camera='cam_scene_yp_l515')
            Image.fromarray(r.render()).save(f'/tmp/pipe_check/harvest{tag}.png')
            r.close()
        except Exception:
            pass


def run_chain(pk, z_lo=1.0, z_hi=1.55, target=None, percept=False, weights=None):
    """链路验证: 真值挑最好摘的一串 -> 驶到最佳站点 -> 有碰撞环境下完成采摘。
    target 指定时跳过选串 (用于演示特定目标, 仍会驶到其站点)。
    percept=True: 混合方案 (stem=分割真值, tomato/trunk=真机 YOLO), 用感知剪切点替代真值。"""
    if target:
        pk.reset()
        cut = pk.cut_site_pos(f'cut_p{target.split("_")[0]}_t{target.split("_")[1]}')
        name = target
        c0 = 1e9                              # 指定目标的净空 (日志用)
        for gid in pk.forbid_geoms:
            gp = pk.d.geom_xpos[gid]
            gm = pk.d.geom_xmat[gid].reshape(3, 3)
            axis = gm[:, 2]
            half = pk.m.geom_size[gid][1]
            v = cut - gp
            tt = float(np.clip(v @ axis, -half, half))
            c0 = min(c0, float(np.linalg.norm(v - tt * axis)))
        print('指定果串: %s  cut=%s  离主茎=%.3fm' % (
            name, np.round(cut, 3).tolist(), c0))
    else:
        best = pk.pick_best_truss(z_lo=z_lo, z_hi=z_hi)
        if best is None:
            print('无符合条件的成熟串')
            return None
        (clear, _), name, cut, c0 = best
        print('选定果串: %s  cut=%s  离主茎=%.3fm' % (
            name, np.round(cut, 3).tolist(), clear))
    station = float(np.clip(cut[0], -2.0, 2.0))
    print('  站点 x=%.2f' % station)
    pk.stage = 'DRIVE TO STATION'
    x = pk.drive_to(station)
    print('已到站 x=%.3f' % x)
    cut_use, pr = None, None
    if percept:
        import sim_perception
        pr = sim_perception.perceive(name, station, weights, gt_cut=cut)
        print('  [percept] ' + json.dumps(pr, ensure_ascii=False))
        if not pr.get('ok'):
            print('  [percept] 感知失败, 链路中止 (不降级到真值)')
            return pr
        cut_use = pr['world_xyz']
    log = pk.harvest(name, cut_override=cut_use)
    if pr is not None:
        log['percept'] = pr
    log['station'] = round(station, 3)
    log['stem_clearance'] = round(c0, 3)
    print(json.dumps(log, ensure_ascii=False, indent=1))
    return log


def reachability():
    pk = Picker()
    print('果串可达性抽检 (按目标高度自适应升降):')
    print('  注: 车停在 x=0, 行向 ±1.8m 的果串大多在本工位外 (论文式"发现即停车"逐站采收)')
    ok = tot = ok_near = tot_near = 0
    for t in pk.trusses:
        pk.reset(lift=0)
        cut = pk.cut_site_pos(t['site'])
        if cut[2] < 0.9:     # 低于论文作业带 (1.1-1.5m), 且低于台面板投影 -> 几何不可达
            log = dict(truss=truss, cut=np.round(cut, 3).tolist(), result='below_envelope')
            self.stage = 'BELOW ENVELOPE'
            return log
        lift = float(np.clip(cut[2] - 0.97, 0.0, 0.5))
        pk.set_lift(lift)
        good, e = pk.goto(cut, iters=300)
        tot += 1
        ok += good
        if abs(cut[0]) <= 0.6:
            tot_near += 1
            ok_near += good
        else:
            t['r'] = round(e, 3)
    print(f'  全部果串: {ok}/{tot}')
    print(f'  本工位 (|x|<=0.6): {ok_near}/{tot_near}')
    return ok, tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target', default=None, help='果串名, 如 7_1; 默认挑最近成熟串')
    ap.add_argument('--reach', action='store_true')
    ap.add_argument('--chain', action='store_true', help='链路验证: 驶到最佳站点摘一串')
    ap.add_argument('--percept', action='store_true',
                    help='混合方案: stem=渲染分割真值, tomato/trunk=真机 YOLO')
    ap.add_argument('--weights', default='/home/lcw/robot_ws_dev/tomato-seg2025063018-yolov8-xucan.pt')
    args = ap.parse_args()
    if args.reach:
        reachability()
        return
    pk = Picker()
    mujoco.mj_resetDataKeyframe(pk.m, pk.d, 0)
    if args.chain:
        run_chain(pk, target=args.target, percept=args.percept, weights=args.weights)
        return
    mujoco.mj_forward(pk.m, pk.d)
    base = pk.d.xpos[pk.m.body('arm_base_mount').id]
    if args.target:
        truss = args.target
    else:
        ripe = [t for t in pk.trusses if t['ripe']]
        ripe.sort(key=lambda t: np.linalg.norm(pk.cut_site_pos(t['site']) - base))
        truss = ripe[0]['name']
        print(f'候选成熟串 {len(ripe)} 个, 选最近: {truss}')
    log = pk.harvest(truss, snap='/tmp/pipe_check/x')
    print(json.dumps(log, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
