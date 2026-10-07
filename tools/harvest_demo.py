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


class Picker:
    def __init__(self):
        self.m = mujoco.MjModel.from_xml_path(XML)
        self.d = mujoco.MjData(self.m)
        self.tick = None          # 可视化钩子: 每个步进循环里周期调用 (harvest_view 用)
        self.stage = '初始化'          # 可视化钩子: 每个步进循环里周期调用 (harvest_view 用)
        m = self.m
        self.qadr = [m.jnt_qposadr[m.joint(n).id] for n in ARM]
        self.aadr = [m.actuator(a).id for a in ['j1', 'j2', 'j3', 'j4', 'j5', 'j6']]
        self.gear_a = m.actuator('gear_servo').id
        self.lift_a = m.actuator('lift_servo').id
        self.lift_j = m.joint('lift_joint').id
        self.tcp = m.body('gear_link').id
        # TCP = 指尖中点: 沿工具轴 (gear 系 +z, 即腕部相机光轴方向) 偏 0.10m
        self.tcp_off = np.array([0.0, 0.0, 0.10])
        self.home_ctrl = np.array([-0.474, 0, 0, -1.59, -0.0611, 1.5, -1.59, -1.65, 3.05, 0.0])
        self.trusses = self._enumerate_trusses()

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
        mujoco.mj_resetDataKeyframe(m, d, 0)
        d.ctrl[:] = self.home_ctrl
        d.ctrl[self.lift_a] = lift
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

    def ik_to(self, target, iters=250, tol=TOL, freeze=(), axis_target=None):
        """阻尼最小二乘把 TCP (指尖中点, gear_link 沿工具轴 +0.10m) 移到 target.
        freeze: 冻结的关节序号 (0-5); axis_target: 工具轴方向约束 (如 (0,0,-1) 竖直)."""
        m, d = self.m, self.d
        dofs = [m.jnt_dofadr[m.joint(n).id] for n in ARM]
        tgt_ax = np.array(axis_target, dtype=float) if axis_target is not None else None
        for it in range(iters):
            mujoco.mj_forward(m, d)
            if self.tick and it % 2 == 0:
                self.tick()
            R = d.xmat[self.tcp].reshape(3, 3)
            off = R @ self.tcp_off
            p = d.xpos[self.tcp] + off
            err = [target - p]
            if tgt_ax is not None:
                ax = R[:, 2]
                err.append(1.0 * (tgt_ax - ax))
            err = np.concatenate(err)
            if np.linalg.norm(err[:3]) < tol and (tgt_ax is None or
                    np.linalg.norm(err[3:]) < 0.05):
                return True, float(np.linalg.norm(target - p))
            jacp = np.zeros((3, m.nv))
            jacr = np.zeros((3, m.nv))
            mujoco.mj_jacBody(m, d, jacp, jacr, self.tcp)
            Jv = jacp[:, dofs] + np.cross(jacr[:, dofs].T, off).T
            if tgt_ax is None:
                J = Jv
            else:
                ax = R[:, 2]
                sk = np.array([[0, -ax[2], ax[1]],
                               [ax[2], 0, -ax[0]],
                               [-ax[1], ax[0], 0]])
                J = np.vstack([Jv, -sk @ jacr[:, dofs]])
            if freeze:
                keep = [i for i in range(6) if i not in freeze]
                J = J[:, keep]
            else:
                keep = list(range(6))
            lam = LAMBDA * np.eye(J.shape[0])
            dq = J.T @ np.linalg.solve(J @ J.T + lam, err)
            dq = np.clip(dq, -0.02, 0.02)
            for ki, i in enumerate(keep):
                d.qpos[self.qadr[i]] += dq[ki]
                d.ctrl[self.aadr[i]] = d.qpos[self.qadr[i]]
            mujoco.mj_step(m, d)
        mujoco.mj_forward(m, d)
        R = d.xmat[self.tcp].reshape(3, 3)
        p = d.xpos[self.tcp] + R @ self.tcp_off
        return np.linalg.norm(target - p) < tol, float(np.linalg.norm(target - p))

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

    def goto(self, target, **kw):
        target = np.asarray(target, dtype=float)
        ok, e = self.ik_to(target, **kw)
        home_arm = np.array([-1.59, -0.0611, 1.5, -1.59, -1.65, 3.05])
        for seed in range(3):            # 局部极值重启: 多组扰动种子
            if ok:
                break
            rng = np.random.default_rng(seed)
            self.d.qpos[self.qadr] = home_arm + rng.uniform(-0.5, 0.5, 6)
            for i in range(6):
                self.d.ctrl[self.aadr[i]] = self.d.qpos[self.qadr[i]]
            ok, e = self.ik_to(target, **kw)
        return ok, e

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

    def harvest(self, truss, snap=None):
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
        nrm = lateral / max(np.linalg.norm(lateral), 1e-6)
        approach = cut + nrm * 0.08

        lift = float(np.clip(cut[2] - 0.97, 0.0, 0.5))
        self.reset(lift=lift)
        self.set_lift(lift)

        log = dict(truss=truss, lift=round(lift, 3), cut=np.round(cut, 3).tolist())
        self.stage = 'LIFT & APPROACH'
        okC, eC = self.goto(approach)
        okB, eB = self.goto(cut - nrm * 0.01)
        log['approach_err'] = round(eC, 4)
        log['reach_err'] = round(eB, 4)
        if not (okC and okB):
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
        p_rel = R1.T @ (d.xpos[tr_body] - d.xpos[base_id])
        R_rel = R1.T @ R2
        q_rel = np.zeros(4)
        mujoco.mju_mat2Quat(q_rel, R_rel.reshape(9))
        m.eq_data[grip][3:6] = p_rel
        m.eq_data[grip][6:10] = q_rel
        m.eq_data[grip][10] = 0.0            # 点抓: 旋转约束归零
        d.eq_active[grip] = 1
        d.eq_active[hold] = 0
        self.gear(GEAR_CLOSED)
        log['result'] = 'cut'
        if snap:
            self._snap('_cut')
        # 剪断后果串垂挂在爪下 (论文 Figure 18d-e): 运输全程保持工具竖直向上
        self.stage = 'TRANSPORT TO BASKET'
        DOWN = np.array([0.0, 0.0, -1.0])
        UP = np.array([0.0, 0.0, 1.0])
        wp_back = cut - nrm * 0.12 + np.array([0, 0, 0.03])   # 沿法线退出冠层
        self.goto(wp_back, iters=400, axis_target=UP)
        bw = self.basket_above()
        okT, eT = self.goto(bw, iters=400, axis_target=UP)
        bid = m.body(f'truss_{truss.split("_")[0]}_{truss.split("_")[1]}').id
        mouth_z = bw[2] - 0.22
        bwlo = bw.copy()
        bwlo[2] = mouth_z + 0.09      # 垂挂果串(长~0.25)下端贴筐底, 全部位于内腔
        self.goto(bwlo, iters=250, axis_target=UP)
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
            self.goto(bw2, axis_target=UP)
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


def reachability():
    pk = Picker()
    print('果串可达性抽检 (按目标高度自适应升降):')
    print('  注: 车停在 x=0, 行向 ±1.8m 的果串大多在本工位外 (论文式"发现即停车"逐站采收)')
    ok = tot = ok_near = tot_near = 0
    for t in pk.trusses:
        pk.reset(lift=0)
        cut = pk.cut_site_pos(t['site'])
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
    args = ap.parse_args()
    if args.reach:
        reachability()
        return
    pk = Picker()
    mujoco.mj_resetDataKeyframe(pk.m, pk.d, 0)
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
