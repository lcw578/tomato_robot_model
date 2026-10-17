"""P0-2 试点感知模块 (2026-10-17): 9_1 混合闭环。

输入: MuJoCo 场景相机 RGB + 米制深度 + 分割标签图
  - stem   : 分割标签图里真实果柄枝 stalk geom 的像素 (真值几何, 非模型)
  - tomato : 真机 YOLO (tomato-seg2025063018-yolov8-xucan.pt) 的 tomato 掩码
  - trunk  : 同上 (本试点不用)
几何: 复用真机老路径 stemGrowthAndPoints 的采摘点算法 (骨架/生长方向/中线交点)
输出: 世界系剪切点 + 像素/深度/误差信息

必须用 conda 环境运行 (mujoco + ultralytics):
  /home/lcw/anaconda3/envs/tomato_sim/bin/python
"""
import os
import sys
import types

import cv2
import numpy as np


# ---------------------------------------------------------------- cv2 补丁
def _ensure_thinning():
    """老路径用 cv2.ximgproc.thinning; opencv-headless 无 contrib —— 用形态学骨架替代。"""
    if hasattr(cv2, 'ximgproc') and hasattr(cv2.ximgproc, 'thinning'):
        return
    def thinning(binary, thinningType=None):
        work = (binary > 0).astype(np.uint8) * 255
        skel = np.zeros_like(work)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        while True:
            opened = cv2.morphologyEx(work, cv2.MORPH_OPEN, element)
            temp = cv2.subtract(work, opened)
            eroded = cv2.erode(work, element)
            skel = cv2.bitwise_or(skel, temp)
            work = eroded.copy()
            if cv2.countNonZero(work) == 0:
                break
        return skel
    cv2.ximgproc = types.SimpleNamespace(thinning=thinning, THINNING_GUOHALL=0)


def _load_old_geometry():
    """导入真机老路径 stemGrowthAndPoints (纯 cv2/numpy, 无 ROS 依赖)."""
    _ensure_thinning()
    path = '/data/robot_assembly/robot_ws_dev/src/xc_robot_main_control/xc_robot_main_control'
    if path not in sys.path:
        sys.path.insert(0, path)
    from stemGrowthAndPoints import StemGrowthAndPoints
    obj = StemGrowthAndPoints()
    obj.save_img = False          # 不写调试图
    return obj



def _project_mesh_mask(m, d, cid, gid, img_w, img_h, fovy_rad):
    """把 mesh geom 的三角面投影到相机图像, 返回 uint8 掩码 (0/255).
    世界顶点 = geom_xpos + geom_xmat @ mesh_vert (MuJoCo 网格重定心规则)."""
    import mujoco
    mid = int(m.geom_dataid[gid])
    vadr, vnum = int(m.mesh_vertadr[mid]), int(m.mesh_vertnum[mid])
    verts = np.array(m.mesh_vert[vadr:vadr + vnum], dtype=np.float64)
    R = d.geom_xmat[gid].reshape(3, 3)
    world = verts @ R.T + d.geom_xpos[gid]
    Rc = d.cam_xmat[cid].reshape(3, 3)
    rel = (world - d.cam_xpos[cid]) @ Rc
    f = (img_h / 2) / np.tan(fovy_rad / 2)
    cx, cy = img_w / 2, img_h / 2
    z = -rel[:, 2]
    ok = z > 1e-3
    u = np.where(ok, cx + f * rel[:, 0] / np.where(ok, z, 1), -1e4)
    v = np.where(ok, cy - f * rel[:, 1] / np.where(ok, z, 1), -1e4)
    fadr, fnum = int(m.mesh_faceadr[mid]), int(m.mesh_facenum[mid])
    faces = np.array(m.mesh_face[fadr:fadr + fnum], dtype=np.int64)
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    for tri in faces:
        if not (ok[tri[0]] and ok[tri[1]] and ok[tri[2]]):
            continue
        pts = np.stack([u[tri], v[tri]], axis=1).astype(np.int32)
        cv2.fillConvexPoly(mask, pts, 255)
    return mask


# ---------------------------------------------------------------- 匹配 (老路径 combine_boxes)
def _iou_stem(box1, box2):
    x_left, y_top = max(box1[0], box2[0]), max(box1[1], box2[1])
    x_right, y_bottom = min(box1[2], box2[2]), min(box1[3], box2[3])
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    inter = (x_right - x_left) * (y_bottom - y_top)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    return inter / area1 if area1 > 0 else 0.0


def _gap(boxA, boxB):
    if boxA[2] <= boxB[0]:
        x_gap = boxB[0] - boxA[2]
    elif boxB[2] <= boxA[0]:
        x_gap = boxA[0] - boxB[2]
    else:
        x_gap = -1
    if boxA[3] <= boxB[1]:
        y_gap = boxB[1] - boxA[3]
    elif boxB[3] <= boxA[1]:
        y_gap = boxA[1] - boxB[3]
    else:
        y_gap = -1
    return x_gap, y_gap


def _can_shift(box1, box2, img_w, img_h, x_ratio=0.15, y_ratio=0.15, iou_th=0.0):
    if box1[3] > box2[3]:
        return False, 0.0
    iou = _iou_stem(box1, box2)
    if iou > iou_th:
        return True, iou
    x_gap, y_gap = _gap(box1, box2)
    th = min(img_w * x_ratio, img_h * y_ratio)
    best = 0.0
    if 0 <= x_gap <= th:
        a = [box1[0] + x_gap, box1[1], box1[2] + x_gap, box1[3]]
        b = [box2[0] - x_gap, box2[1], box2[2] - x_gap, box2[3]]
        best = max(best, _iou_stem(a, b))
    if 0 <= y_gap <= th:
        a = [box1[0], box1[1] + y_gap, box1[2], box1[3] + y_gap]
        b = [box2[0], box2[1] - y_gap, box2[2], box2[3] - y_gap]
        best = max(best, _iou_stem(a, b))
    return best > iou_th, best


def _match_stem_to_tomato(stem_box, tomato_boxes, img_w, img_h):
    """老路径语义: stem 底部必须在番茄中心上方; 先 IoU 再平移匹配; 兜底最近中心。"""
    best = None
    for i, t in enumerate(tomato_boxes):
        if stem_box[3] > (t[1] + t[3]) / 2:
            continue
        iou = _iou_stem(stem_box, t)
        if best is None or iou > best[1]:
            best = (i, iou)
    if best is None or best[1] <= 0:
        for i, t in enumerate(tomato_boxes):
            if stem_box[3] > (t[1] + t[3]) / 2:
                continue
            ok, iou = _can_shift(stem_box, t, img_w, img_h)
            if ok and (best is None or iou > best[1]):
                best = (i, iou)
    if best is None:
        # 兜底: 最近中心 (不强制上下关系)
        cs = np.array([(stem_box[0] + stem_box[2]) / 2, (stem_box[1] + stem_box[3]) / 2])
        d = [np.linalg.norm(cs - np.array([(t[0] + t[2]) / 2, (t[1] + t[3]) / 2]))
             for t in tomato_boxes]
        best = (int(np.argmin(d)), 0.0)
    return best


# ---------------------------------------------------------------- 感知主流程
def perceive(truss, station_x, weights, camera=None, gt_cut=None,
             img_w=1280, img_h=720):
    import mujoco
    from ultralytics import YOLO

    m = mujoco.MjModel.from_xml_path('/data/robot_assembly/model/tomato_picker.xml')
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    jid = m.joint('machine_slide').id
    d.qpos[m.jnt_qposadr[jid]] = station_x
    mujoco.mj_forward(m, d)

    p_, k_ = truss.split('_')
    if camera is None:
        camera = 'cam_scene_yp_l515' if int(p_) < 5 else 'cam_scene_yn_l515'
    cid = m.camera(camera).id

    r_rgb = mujoco.Renderer(m, img_h, img_w)
    r_rgb.update_scene(d, camera=camera)
    rgb = r_rgb.render().copy()
    r_dep = mujoco.Renderer(m, img_h, img_w)
    r_dep.update_scene(d, camera=camera)
    r_dep.enable_depth_rendering()
    depth = r_dep.render().astype(np.float32)
    r_dep.disable_depth_rendering()

    # stem 掩码 = 真实果柄枝 mesh geom 的几何投影
    stem_name = f'g_v{p_}_stalk_{k_}'
    try:
        stem_gid = m.geom(stem_name).id
    except KeyError:
        return dict(truss=truss, ok=False, reason=f'missing {stem_name}')
    fovy_rad = np.radians(float(m.cam_fovy[cid]))
    stem_mask = _project_mesh_mask(m, d, cid, stem_gid, img_w, img_h, fovy_rad)
    if stem_mask.sum() == 0:
        return dict(truss=truss, ok=False, reason='stalk not visible from camera')
    ys, xs = np.where(stem_mask > 0)
    stem_box = [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]

    # YOLO: tomato / trunk
    yolo = YOLO(weights)
    # ultralytics 的 numpy 输入按 BGR 解释 (与 cv2.imread 一致)
    res = yolo.predict(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), imgsz=640, conf=0.35,
                       iou=0.5, device='cuda', verbose=False)[0]
    tomato_masks, tomato_boxes, tomato_conf = [], [], []
    if res.masks is not None and res.boxes is not None:
        masks = res.masks.data.detach().cpu().numpy()
        boxes = res.boxes.xyxy.detach().cpu().numpy()
        classes = res.boxes.cls.detach().cpu().numpy().astype(int)
        confs = res.boxes.conf.detach().cpu().numpy()
        for mask, box, cls, conf in zip(masks, boxes, classes, confs):
            if cls != 1:          # 只用 tomato
                continue
            mk = cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST) > 0.5
            tomato_masks.append(mk.astype(np.uint8) * 255)
            tomato_boxes.append([float(v) for v in box])
            tomato_conf.append(float(conf))
    if not tomato_boxes:
        return dict(truss=truss, ok=False, reason='no tomato detected')

    ti, iou = _match_stem_to_tomato(stem_box, tomato_boxes, img_w, img_h)
    tomato_box = tomato_boxes[ti]
    tomato_mask = tomato_masks[ti]

    # 老路径采摘点
    geom = _load_old_geometry()
    h, w = rgb.shape[:2]
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    stem_cx = (stem_box[0] + stem_box[2]) / 2
    tom_cx = (tomato_box[0] + tomato_box[2]) / 2
    aspect = (stem_box[2] - stem_box[0]) / max(stem_box[3] - stem_box[1], 1e-6)
    out = geom.determine_growth_direction_and_picking_point(
        0, stem_box, tomato_box, stem_mask, tomato_mask, bgr,
        stem_cx, tom_cx, aspect, w, h)
    ok = bool(out[0])
    pick = out[3] if ok else None
    growth = out[1] if ok else 'unknown'
    if pick is None or (pick[0] <= 0 and pick[1] <= 0):
        # 兜底: 果柄掩码质心
        pick = (float(xs.mean()), float(ys.mean()))
        growth = 'centroid_fallback'
    u, v = float(pick[0]), float(pick[1])

    # 深度 -> 相机系 -> 世界系
    # 老路径的采摘点来自膨胀后骨架, 可能落在细枝条的像素缝里 (mask=0, 深度=哨兵);
    # 先吸附到掩码内最近的有效深度像素, 再反投影。
    valid = (stem_mask > 0) & np.isfinite(depth) & (depth > 0.05) & (depth < 10.0)
    uu, vv = int(round(u)), int(round(v))
    if 0 <= vv < depth.shape[0] and 0 <= uu < depth.shape[1] and valid[vv, uu]:
        u_d, v_d = u, v
    else:
        ys_v, xs_v = np.where(valid)
        if len(xs_v) == 0:
            return dict(truss=truss, ok=False, reason='no valid depth on stalk mask')
        k = int(np.argmin((xs_v - u) ** 2 + (ys_v - v) ** 2))
        u_d, v_d = float(xs_v[k]), float(ys_v[k])
    z = float(depth[int(round(v_d)), int(round(u_d))])
    fovy = np.radians(m.cam_fovy[cid])
    f = (img_h / 2) / np.tan(fovy / 2)
    cx, cy = img_w / 2, img_h / 2
    x_c = (u_d - cx) * z / f
    y_c = -(v_d - cy) * z / f
    z_c = -z
    R = d.cam_xmat[cid].reshape(3, 3)
    world = d.cam_xpos[cid] + R @ np.array([x_c, y_c, z_c])

    err = None
    if gt_cut is not None:
        err = float(np.linalg.norm(world - np.asarray(gt_cut)))

    # 感知调试图: stem 绿 / tomato 黄 / 框 / 采摘点
    dbg = bgr.copy()
    dbg[stem_mask > 0] = (0, 255, 0)
    dbg[tomato_mask > 0] = (0.5 * dbg[tomato_mask > 0] + 0.5 * np.array([0, 255, 255])).astype(np.uint8)
    cv2.rectangle(dbg, (int(stem_box[0]), int(stem_box[1])),
                  (int(stem_box[2]), int(stem_box[3])), (0, 255, 0), 2)
    cv2.rectangle(dbg, (int(tomato_box[0]), int(tomato_box[1])),
                  (int(tomato_box[2]), int(tomato_box[3])), (0, 255, 255), 2)
    cv2.circle(dbg, (int(round(u)), int(round(v))), 5, (0, 0, 255), -1)
    cv2.circle(dbg, (int(round(u_d)), int(round(v_d))), 3, (255, 0, 0), -1)
    cv2.putText(dbg, f'{truss} tomato={tomato_conf[ti]:.2f} pick=({u:.0f},{v:.0f}) '
                     f'depth={z:.3f}m err={None if err is None else round(err*1000,1)}mm',
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.imwrite(f'/tmp/percept_{truss}.png', dbg)

    return dict(truss=truss, ok=True, camera=camera,
                stem_box=[round(v2, 1) for v2 in stem_box],
                tomato_box=[round(v2, 1) for v2 in tomato_box],
                tomato_conf=round(tomato_conf[ti], 3), match_iou=round(iou, 3),
                pick_pixel=[round(u, 1), round(v, 1)],
                depth_pixel=[round(u_d, 1), round(v_d, 1)], depth=round(z, 4),
                world_xyz=[round(float(v2), 4) for v2 in world],
                gt_xyz=None if gt_cut is None else [round(float(v2), 4) for v2 in gt_cut],
                err_mm=None if err is None else round(err * 1000, 1),
                growth_direction=growth,
                stem_px=int((stem_mask > 0).sum()),
                tomato_px=int((tomato_mask > 0).sum()))
