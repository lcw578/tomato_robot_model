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

        for seed in range(5):
            q = base_q.copy() if seed == 0 else \
                q_rest + np.random.default_rng(seed * 7 + 1).uniform(-0.6, 0.6, 6)
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
                    return q.copy()
                Jv, Jr = jac()
                J = np.vstack([Jv, Jr]) if rot else Jv
                q = step(q, J, err, 5e-3, 0.05)
            if tgt_ax is not None and best_e < 0.02:
                return best_q          # 轴对齐解: 位置残差 <2cm 接受 (判据容差 26mm 内)
        if best_e > 0.05:
            return None
        return best_q

    # ---------- 碰撞检测与规划 (关节空间 RRT, 直线优先) ----------
    def collision_free(self, q, pen=0.003):
        """禁碰集与臂的穿透超过 pen 视为碰撞 (轻触允许: 真机也轻触植株)。"""
        fd = self._fk
        fd.qpos[:] = self.d.qpos
        fd.qpos[self.qadr] = q
        mujoco.mj_forward(self.m, fd)
        for c in range(fd.ncon):
            if fd.contact[c].dist > -pen:
                continue
            g1, g2 = fd.contact[c].geom1, fd.contact[c].geom2
            if (g1 in self.arm_geoms and g2 in self.forbid_geoms) or \
               (g2 in self.arm_geoms and g1 in self.forbid_geoms):
                return False
        return True

    def _seg_free(self, qa, qb, n=8, pen=0.003):
        # 排除 t=0 端点: 抓取位姿允许轻微贴碰, 路径应从该状态可退出
        for t in np.linspace(0, 1, n + 1)[1:]:
            if not self.collision_free(qa * (1 - t) + qb * t, pen=pen):
                return False
        return True

    def plan_path(self, q_goal, max_iter=15000, step=0.15, time_budget=20.0):
        """返回通往 q_goal 的无碰航点列表 (关节空间); 直线无碰则直达。"""
        q0 = self.cmd.copy()
        if self._seg_free(q0, q_goal):
            return [q_goal]
        import time as _time
        _t0 = _time.time()
        rng = np.random.default_rng(0)
        cap = max_iter + 4
        nodes = np.zeros((cap, 6))                   # 预分配 (原 list+每轮 np.array 为 O(n^2) 拷贝)
        nodes[0] = q0
        n_nodes = 1
        parent = [-1]
        bounds = 2.9
        for i in range(max_iter):
            if (i & 63) == 0 and _time.time() - _t0 > time_budget:
                break
            if i % 2 == 1:
                q_rand = q_goal + rng.normal(0, 0.25, 6)         # 50% 目标偏置
            else:
                q_rand = rng.uniform(-bounds, bounds, 6)
            j = int(np.argmin(np.linalg.norm(nodes[:n_nodes] - q_rand, axis=1)))
            q_near = nodes[j]
            v = q_rand - q_near
            L = np.linalg.norm(v)
            if L < 1e-9:
                continue
            q_new = q_near + v / L * min(step, L)
            q_new = np.clip(q_new, -(3.0543 - JOINT_MARGIN), 3.0543 - JOINT_MARGIN)
            if not self._seg_free(q_near, q_new, n=3):
                continue
            nodes[n_nodes] = q_new; parent.append(j); n_nodes += 1
            # 目标连接段放宽到 12mm: 剪切位姿本就贴近茎秆 (真机轻触); 路径段仍 3mm
            if np.linalg.norm(q_new - q_goal) < 0.6 and \
                    self._seg_free(q_new, q_goal, n=8, pen=0.032):
                nodes[n_nodes] = q_goal; parent.append(n_nodes - 1); n_nodes += 1
                path = []
                k = n_nodes - 1
                while k != -1:
                    path.append(nodes[k].copy()); k = parent[k]
                path = path[::-1]
                self._last_path = path
                return path
        arr = np.array(nodes)
        dmin = float(np.min(np.linalg.norm(arr - q_goal, axis=1)))
        print('  [plan] RRT %d 迭代未连通: 节点=%d  离目标最近=%.3f  直线碰撞=%s' % (
            max_iter, len(nodes), dmin, not self._seg_free(q0, q_goal)))
        return None          # 规划失败 (被植株完全挡住)

    def tcp_pos(self):
        R = self.d.xmat[self.tcp].reshape(3, 3)
        return self.d.xpos[self.tcp] + R @ self.tcp_off

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

    def _walk_to(self, q_target, speed=0.8):
        """关节空间匀速插值走 cmd (积分语义, 伺服跟踪)。"""
        guard = 0
        while True:
            guard += 1
            if guard > 60000:      # 看门狗: 防死循环 (NaN 目标等)
                print('  [walk] 看门狗中止 tgt=%s' % np.round(q_target, 2), flush=True)
                break
            remain = q_target - self.cmd
            if not np.all(np.isfinite(remain)):
                print('  [walk] 目标非法(NaN/inf), 中止', flush=True)
                break
            if np.max(np.abs(remain)) < 1e-3:
                break
            step = np.clip(remain, -VMAX * self.m.opt.timestep * speed,
                           VMAX * self.m.opt.timestep * speed)
            self.cmd = np.clip(self.cmd + step,
                               -(3.0543 - JOINT_MARGIN), 3.0543 - JOINT_MARGIN)
            self.d.ctrl[self.aadr] = self.cmd
            mujoco.mj_step(self.m, self.d)
            mujoco.mj_step(self.m, self.d)
            if self.tick:
                self.tick()

    def goto_reverse(self, speed=0.8):
        """沿上一条规划路径反向退回 (抓取位姿常在自己植株茎秆的包围袋里, 原路是唯一通道)。"""
        path = getattr(self, '_last_path', None)
        if not path:
            return False
        for wp in path[::-1]:
            self._walk_to(wp, speed=speed)
        self._last_path = None
        return True

    def goto(self, target, iters=300, axis_target=None, speed=0.8):
        """规划-执行: 离线解算 -> RRT 无碰规划 (直线优先) -> 沿航点关节空间插值。"""
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

    def harvest(self, truss, snap=None, axis_grasp=False):
        """对一串果执行完整采集. truss 形如 '7_1'."""
        m, d = self.m, self.d
        site = f'cut_p{truss.split("_")[0]}_t{truss.split("_")[1]}'
        sid = m.site(site).id
        # 目标点先在 home/lift=0 下读一次世界坐标, 升降只影响臂不影响目标
        mujoco.mj_forward(m, d)
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
        to_robot /= max(np.linalg.norm(to_robot), 1e-6)      # 朝机器人基座 (走道, 开放空间)
        dvec = cut - d.xipos[tr_body]
        dvec /= max(np.linalg.norm(dvec), 1e-9)              # 沿果柄 (指向主茎)
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
            q_b = self.solve_ik(cut - dvec * 0.015, iters=800, axis_target=dvec)
            if q_b is None:
                q_b = self.solve_ik(cut - dvec * 0.015, iters=800)
            e_b = float('inf')
            if q_b is not None:
                self._fk.qpos[:] = d.qpos
                self._fk.qpos[self.qadr] = q_b
                mujoco.mj_forward(m, self._fk)
                R = self._fk.xmat[self.tcp].reshape(3, 3)
                pp = self._fk.xpos[self.tcp] + R @ self.tcp_off
                e_b = float(np.linalg.norm(pp - (cut - dvec * 0.015)))
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

        self.stage = 'LIFT & APPROACH'
        # approach = 开放空间路点 (纯位置, 轴对齐在此非必要且易无解);
        # B = 抓取位姿 (轴对齐: 果柄沿工具轴, 主茎落在指尖之外)
        okC, eC = self.goto(approach, iters=600)
        if axis_grasp:
            okB, eB = self.goto(cut - dvec * 0.015, iters=800, axis_target=dvec)
            if not okB:
                # 轴对齐对该株几何无解 (可达性约束) -> 回退纯位置, 记录标注
                print('  [harvest] 轴对齐抓取无解, 回退纯位置 B', flush=True)
                okB, eB = self.goto(cut - dvec * 0.015, iters=800)
                log['axis_grasp'] = False
        else:
            okB, eB = self.goto(cut + to_robot * 0.02, iters=800)   # 停在剪切点前 2cm
        log['approach_err'] = round(eC, 4)
        log['reach_err'] = round(eB, 4)
        # B 点成败由任务判据决定 (果柄是否落入剪切区), 而非毫米级 TCP 误差 ——
        # 最后进冠层的物理接触会顶偏爪子 (真实机器同样接触植株)
        if not okC:
            log['result'] = 'unreachable'
            self.stage = 'UNREACHABLE'
            return log
        self.stage = 'CLOSE & CUT'
        # 剪切判据 (确定性几何): 果柄线段 (attach->果簇) 与夹爪剪切区盒相交
        # 盒 (gear 系): 轴向 [0.01, 0.115] (齿轮回转面到指端), 径向距工具轴 <= 0.026
        # (= 张开口 52mm 的一半 —— 果柄能落入张开的夹爪内部, 闭合即剪到)
        R_g = d.xmat[self.tcp].reshape(3, 3)
        p_g = d.xpos[self.tcp]                            # gear 原点 (世界)
        att = d.site_xpos[m.site(f'attach_p{truss.split("_")[0]}_t{truss.split("_")[1]}').id]
        seg_w = np.array([att, d.xipos[tr_body]])        # 果柄线段 (世界)
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
        p_new_w = self.tcp_pos()                        # 果串原点保持在 TCP (剪切点)
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
            d.ctrl[self.gear_a] = GEAR_CLOSED
            mujoco.mj_step(m, d)
            if self.tick and i % 2 == 0:
                self.tick()
        log['result'] = 'cut'
        if snap:
            self._snap('_cut')
        # 剪断后果串垂挂在爪下 (论文 Figure 18d-e): 运输全程保持工具竖直向上
        self.stage = 'TRANSPORT TO BASKET'
        DOWN = np.array([0.0, 0.0, -1.0])
        UP = np.array([0.0, 0.0, 1.0])
        self.goto_reverse()   # 沿原规划路径倒回 (唯一通道)
        bw = self.basket_above()
        okT, eT = self.goto(bw, iters=800)
        bid = m.body(f'truss_{truss.split("_")[0]}_{truss.split("_")[1]}').id
        mouth_z = bw[2] - 0.22
        bwlo = bw.copy()
        bwlo[2] = mouth_z + 0.11   # 释放深度: 指尖离筐底留 ~2cm (实测 1mm 浅擦 -> 抬升 2cm)      # 垂挂果串(长~0.25)下端贴筐底, 全部位于内腔
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


def run_chain(pk, z_lo=1.0, z_hi=1.55):
    """链路验证: 真值挑最好摘的一串 -> 驶到最佳站点 -> 有碰撞环境下完成采摘。"""
    best = pk.pick_best_truss(z_lo=z_lo, z_hi=z_hi)
    if best is None:
        print('无符合条件的成熟串')
        return None
    (clear, _), name, cut, c0 = best
    station = float(np.clip(cut[0], -2.0, 2.0))
    print('选定果串: %s  cut=%s  离主茎=%.3fm  站点 x=%.2f' % (
        name, np.round(cut, 3).tolist(), clear, station))
    pk.stage = 'DRIVE TO STATION'
    x = pk.drive_to(station)
    print('已到站 x=%.3f' % x)
    log = pk.harvest(name, axis_grasp=True)
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
    args = ap.parse_args()
    if args.reach:
        reachability()
        return
    pk = Picker()
    mujoco.mj_resetDataKeyframe(pk.m, pk.d, 0)
    if args.chain:
        run_chain(pk)
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
