"""Phase C: assemble the full MuJoCo model tomato_picker.xml (units: meters, frame = V/1000).

Tree:
  world
  ├── chassis (static: 309 kept CAD parts + trimmed lift column)
  │    └── 10 wheel bodies (hinge; 2 actuated with velocity servos)
  ├── shoulder_Link ... wrist3_Link  (official AUBO i5 chain, mounted at the
  │    CAD base disc position, z-down; root at world so arm-vehicle collisions work)
  │    └── gripper base_link (baseline MJCF subtree, design-intent mount)
  │         └── gear/right/left fingers + equality + excludes + gear servo
  │         + camera-module geoms (CAD poses relative to gripper)
"""
import sys, re, struct
import xml.etree.ElementTree as ET
sys.path.insert(0, '/data/robot_assembly')
from geomlib import *
import numpy as np, json, os

OUT = '/data/robot_assembly'
ASM = f'{OUT}/model'
os.makedirs(f'{ASM}/meshes', exist_ok=True)

tg = json.load(open(f'{TOMATO_DIR}/parts.json'))['parts']
um = json.load(open(f'{TOMATO_DIR}/user_model.json'))
gp = json.load(open(f'{GRIPPER_DIR}/parts.json'))['parts']
cls = json.load(open(f'{OUT}/part_classes_final.json'))
T_VG = np.load(f'{OUT}/T_V_from_G.npy')

byid = {p['part_id']: p for p in tg}
umpart = {}
for l in um['links']:
    for pid in l['part_ids']:
        umpart[pid] = l

# per-part mass: volume x plastic density, calibrated to the tool's robot_arm total
ra_link = [l for l in um['links'] if l['name'] == 'robot_arm'][0]
Vtot = sum(byid[pid]['volume_estimate'] for pid in ra_link['part_ids'])
RHO = ra_link['mass'] / Vtot          # kg/mm^3
print(f"plastic density: {RHO*1e6:.4f} g/cm3 (calibrated on robot_arm total {ra_link['mass']:.2f} kg)")
def part_mass(pid):
    return RHO * byid[pid]['volume_estimate']

# ---- final class overrides ----
adapter = {'part_362', 'part_371'}   # printed wrist-camera adapter (362 = the L-bracket with the flange disc, 371 = the spacer ring on the gripper face)
# part_358, part_360 & part_361 = stylized replicas of the arm's wrist3 -> removed (the real wrist3 already exists on the arm).
# 360 is a small boss hanging off the replica (0.0 mm to 361, >70 mm to everything else), so it goes with it.
remove = set(cls['remove_arm']) | {'part_332', 'part_337'} | set(json.load(open(f'{OUT}/stray_parts.json')))\
         - {'part_365', 'part_369'} - adapter   # 365/369 rejoin camera; adapter rejoins the gripper assembly
# part_331 (CAD replica of the i5 base) removed: official base mesh used instead
static = [p for p in cls['static'] if p not in ('part_332', 'part_337', 'part_333', 'part_339', 'part_346')]  # 339/346 = planting-rack pipes (environment)
static = [p for p in static if p not in remove]   # enforce: anything in remove never enters the chassis
grip_solid = set(cls['gripper_solids'])    # removed (duplicated by baseline gripper meshes)
camera = set(cls['camera_module']) | {'part_365', 'part_369'}   # camera + its stalk pieces
print(f"static={len(static)} remove={len(remove)} grip={len(grip_solid)} cam={len(camera)}")

# ---- trim lift column part_333 (drop shoulder-ball branch below z=-757mm) ----
TRIM = {}  # part_333 (stylized shoulder ball + stray vertex) removed entirely
os.makedirs(f'{ASM}/meshes', exist_ok=True)
for pid, zcut in TRIM.items():
    v, f = load_stl(f"{TOMATO_DIR}/{byid[pid]['mesh_file']}")
    keep = (v[f[:, 0], 2] > zcut) & (v[f[:, 1], 2] > zcut) & (v[f[:, 2], 2] > zcut)
    tris = v[f[keep]]
    with open(f'{ASM}/meshes/{pid}_trimmed.stl', 'wb') as fh:
        fh.write(b'\0' * 80)
        fh.write(struct.pack('<I', len(tris)))
        for t in tris:
            n = np.cross(t[1] - t[0], t[2] - t[0])
            nn = n / np.linalg.norm(n) if np.linalg.norm(n) > 0 else np.zeros(3)
            fh.write(struct.pack('<3f', *nn.astype(np.float32)))
            for pnt in t:
                fh.write(struct.pack('<3f', *pnt.astype(np.float32)))
            fh.write(struct.pack('<H', 0))
    print(f"trimmed {pid}: kept {keep.sum()}/{len(f)} tris")

# ---- composite inertia helper (CAD mm, kg, kg*m^2) ----
def composite(parts):
    M = 0.0; c = np.zeros(3); I = np.zeros((3, 3))
    m_list = []
    for pid in parts:
        p = byid[pid]
        m = part_mass(pid)
        bb = p['bbox']
        ci = (np.array(bb['min']) + np.array(bb['max'])) / 2000.0   # m (bbox center)
        m_list.append((m, ci))
        M += m; c += m * ci
    c /= M
    for m, ci in m_list:
        d = ci - c
        I += m * (np.dot(d, d) * np.eye(3) - np.outer(d, d))       # point masses
    return M, c, I

veh_M, veh_c, veh_I = composite(static)
print(f"chassis: M={veh_M:.2f}kg com={np.round(veh_c,3)}")

# ---------------- XML assembly ----------------
X = []
X.append('<mujoco model="tomato_picker">')
X.append('  <compiler angle="radian" eulerseq="xyz" autolimits="true"/>')
X.append('  <option timestep="0.002" integrator="implicitfast"/>')
X.append('  <visual><global offwidth="1920" offheight="1080"/></visual>')
X.append('  <asset>')

def mesh_line(name, file, scale="0.001 0.001 0.001"):
    return f'    <mesh name="{name}" file="{file}" scale="{scale}" inertia="shell"/>'

# vehicle meshes
for pid in sorted(static):
    mf = f'{ASM}/meshes/{pid}_trimmed.stl' if pid in TRIM else f'{TOMATO_DIR}/{byid[pid]["mesh_file"]}'
    X.append(mesh_line('v_' + pid, mf))
# wheel meshes
WHEELS = {'small_right_front': 'part_011', 'small_left_front': 'part_015',
          'small_right_behind': 'part_004', 'small_left_behind': 'part_008',
          'big_left_front': 'part_062', 'big_right_front': 'part_056',
          'big_right_behind': 'part_044', 'big_left_behind': 'part_050',
          'act_right': 'part_260', 'act_left': 'part_039'}
for w, pid in WHEELS.items():
    X.append(mesh_line('w_' + w, f'{TOMATO_DIR}/{byid[pid]["mesh_file"]}'))
# aubo meshes (meters): STL = collision, DAE material groups = visual
import json as _json
DAE_GROUPS = _json.load(open(f'{OUT}/dae_split/groups.json'))
for n in ['base_link', 'shoulder_Link', 'upperArm_Link', 'foreArm_Link', 'wrist1_Link', 'wrist2_Link', 'wrist3_Link']:
    X.append(mesh_line('a_' + n, f'{AUBO_DIR}/models/{n}.STL', scale="1 1 1"))
    for grp in DAE_GROUPS.get(n, {}):
        if grp == 'none':
            continue   # internal parts (incl. far-flung junk geometry) not used
        X.append(mesh_line(f'd_{n}_{grp}', f'{OUT}/dae_split/{n}_{grp}.stl', scale="1 1 1"))
# gripper meshes (from baseline robot.xml: same files, absolute paths)
gsrc = open(f'{GRIPPER_DIR}/gripper_scene.xml').read()
for m in re.finditer(r'<mesh name="(part_\d+_solid_\d+)" file="([^"]+)"', gsrc):
    X.append(mesh_line('g_' + m.group(1), f'{GRIPPER_DIR}/{m.group(2)}'))
# camera module meshes (V-frame mm)
for pid in sorted(camera | adapter):
    X.append(mesh_line('c_' + pid, f'{OUT}/model/meshes_baked/{pid}_baked.stl', scale="1 1 1"))
X.append('  </asset>')

# ground-stance flip: the CAD assembly is modeled upside-down (wheels up, arm down).
# Wrap everything in a 180deg-about-x root body so wheels are at the bottom, arm on top.
_act_top = max(byid[pid]['bbox']['max'][2] for pid in WHEELS.values()
               if pid in ('part_260', 'part_039'))
FLIP_T = _act_top / 1000.0     # m; act wheel tops land on z=0
print(f"ground-stance flip: t={FLIP_T:.4f} m (act wheel top V z={_act_top:.1f} mm)")
X.append('  <worldbody>')
X.append('  <light name="top" directional="true" pos="0 0 3" dir="0 0 -1" diffuse="0.9 0.9 0.9" specular="0.3 0.3 0.3"/>')
X.append('  <light name="side" directional="true" pos="2 -2 2" dir="-1 1 -0.8" diffuse="0.5 0.5 0.55" specular="0.1 0.1 0.1"/>')
X.append('  <light name="back" directional="true" pos="-2 2 1.5" dir="1 -1 -0.5" diffuse="0.35 0.35 0.4"/>')
X.append('  <geom name="floor" type="plane" size="6 6 0.05" contype="0" conaffinity="0" rgba="0.22 0.24 0.27 1"/>')
X.append(f'  <body name="machine" pos="0 0 {FLIP_T:.6f}" quat="0 1 0 0">')
# ---------------- chassis (static) ----------------
M, c, I = veh_M, veh_c, veh_I
X.append(f'  <body name="chassis" pos="0 0 0">')
X.append(f'    <inertial pos="{c[0]:.6f} {c[1]:.6f} {c[2]:.6f}" mass="{M:.6f}" '
         f'fullinertia="{I[0,0]:.8f} {I[1,1]:.8f} {I[2,2]:.8f} {I[0,1]:.10f} {I[0,2]:.10f} {I[1,2]:.10f}"/>')
for pid in sorted(static):
    X.append(f'    <geom name="v_{pid}" type="mesh" mesh="v_{pid}" group="1"/>')
# wheels
ACT = {'act_left', 'act_right'}
for w, pid in WHEELS.items():
    l = umpart[pid]
    mc = np.array(l['mass_properties']['com_xyz_cad']) / 1000.0
    Ii = l['inertia']
    X.append(f'    <body name="{w}" pos="0 0 0">')
    X.append(f'      <inertial pos="{mc[0]:.6f} {mc[1]:.6f} {mc[2]:.6f}" mass="{l["mass"]:.6f}" '
             f'fullinertia="{Ii["ixx"]:.10f} {Ii["iyy"]:.10f} {Ii["izz"]:.10f} '
             f'{Ii["ixy"]:.12f} {Ii["ixz"]:.12f} {Ii["iyz"]:.12f}"/>')
    if w in ACT:
        X.append(f'      <joint name="{w}_joint" type="hinge" pos="{mc[0]:.6f} {mc[1]:.6f} {mc[2]:.6f}" '
                 f'axis="0 1 0" damping="0.002" armature="0.004"/>')
    else:
        X.append(f'      <joint name="{w}_joint" type="hinge" pos="{mc[0]:.6f} {mc[1]:.6f} {mc[2]:.6f}" '
                 f'axis="0 1 0" damping="0.001" frictionloss="0.02"/>')
    X.append(f'      <geom name="w_{w}" type="mesh" mesh="w_{w}" group="1"/>')
    X.append('    </body>')
X.append('  </body>')   # chassis

# ---------------- AUBO chain (rooted at world) ----------------
# mount: base flange face at z=-0.704, z-down; shoulder joint 122mm below flange
CHAIN = [
    ('shoulder_Link', (0, 0, 0.122),   (-3.67321e-06, 0, 0, 1), 80.0),
    ('upperArm_Link', (0, 0.1215, 0),  (0.499998, -0.5, -0.5, -0.500002), 80.0),
    ('foreArm_Link',  (0.408, 0, 0),   (-3.67321e-06, -1, 0, 0), 80.0),
    ('wrist1_Link',   (0.376, 0, 0),   (-2.59734e-06, 0.707105, 0.707108, -2.59735e-06), 40.0),
    ('wrist2_Link',   (0, 0.1025, 0),  (0.707105, -0.707108, 0, 0), 40.0),
    ('wrist3_Link',   (0, -0.094, 0),  (0.707105, 0.707108, 0, 0), 40.0),
]
# inertials from aubo_i5.xml (parse)
aubo = open(f'{AUBO_DIR}/mjcf/aubo_i5.xml').read()
inert = {}
for b in re.finditer(r'<body name="(\w+)".*?(<inertial[^/]*/>)', aubo, re.S):
    inert[b.group(1)] = b.group(2)

X.append('  <body name="arm_base_mount" pos="0.002 0.004 -0.704" quat="0 1 0 0">')  # 180deg about x
X.append('    <inertial pos="0 0 0.025" mass="0.9" diaginertia="0.002 0.002 0.002"/>')  # i5 base approx
X.append('    <geom name="a_base_link" type="mesh" mesh="a_base_link" group="1"/>')
for i, (name, pos, quat, force) in enumerate(CHAIN):
    X.append(f'    <body name="{name}" pos="{pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}" '
             f'quat="{quat[0]:.6f} {quat[1]:.6f} {quat[2]:.6f} {quat[3]:.6f}">')
    X.append('      ' + inert[name])
    X.append(f'      <joint name="{name.split("_")[0]}_joint" pos="0 0 0" axis="0 0 1" '
             f'range="-3.0543 3.0543" damping="0.2" armature="0.01"/>')
    X.append(f'      <geom name="a_{name}_col" type="mesh" mesh="a_{name}" group="3" rgba="0 0 0 0"/>')
    GRP_COL = {'m1': '0.99 0.37 0.06 1', 'm2': '0.10 0.10 0.11 1', 'none': '0.18 0.18 0.20 1'}
    for grp in DAE_GROUPS.get(name, {}):
        if grp == 'none':
            continue
        X.append(f'      <geom name="a_{name}_v{grp}" type="mesh" mesh="d_{name}_{grp}" group="1" '
                 f'contype="0" conaffinity="0" rgba="{GRP_COL[grp]}"/>')
# ----- gripper mount on the REAL tool flange (+z face of wrist3, coaxial with J6) -----
# The tool flange is the +z end of wrist3: a planar annulus at z_w3 = 0 (verified:
# 167 face verts all at exactly z=0.00), coaxial with the J6 axis (z_w3), radii
# 15.7..31.5 mm (Ø63 OD, Ø31.4 cable bore) with 4×M6 bolt holes at r≈25 mm.
# The +y end is NOT a flange - just a 40° arc patch of the rounded bulge; the -z
# end is the wrist2 seat (isolated Ø75 rim). Tool axis = +z_w3.
v3, _ = load_stl(f'{AUBO_DIR}/models/wrist3_Link.STL')
face = v3[v3[:, 2] > v3[:, 2].max() - 0.0003]
face_c_w3 = np.array([0.0, 0.0, float(face[:, 2].mean())])   # ON the J6 axis, plane z=0
# gripper body orientation in w3 (from the CAD's stylized wrist copy 361 <-> real
# wrist3 correspondence: 361's z-span/ bulges match w3 with axes permuted):
#   file +x -> +y_w3, file +y -> +x_w3, file +z -> -z_w3   (det=+1)
# => gripper tool (-z_G) along +z_w3 (away from wrist2, which sits at z_w3=-0.094)
R_gw = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
q_gw = quat_from_R(R_gw)
# adapter/camera parts live around the gripper shell (CAD poses via T_VG^-1),
# NOT between flange and gripper: the gripper mount face mates the flange directly.
Rt_mm = T_VG[:3, :3]; tv_mm = T_VG[:3, 3]
face_rel = np.array([-0.0074, 0.0, 0.0435])   # gripper mount-face center in body frame (m)
# printed adapter: its flange-side (top) face must mate the flange.
# find that face's center in body coords, then align it with the flange center.
S = -1e9; top_pts = []; all_pts = []
for pid in sorted(adapter):
    vv, _ = load_stl(f'{OUT}/model/meshes_baked/{pid}_baked.stl')
    all_pts.append(vv)
    S = max(S, float(vv[:, 2].max()))
for pid in sorted(adapter):
    vv, _ = load_stl(f'{OUT}/model/meshes_baked/{pid}_baked.stl')
    m = vv[:, 2] > S - 0.0015
    if m.any():
        top_pts.append(vv[m])
top = np.vstack(top_pts)
# mating center: part_362's top face = the print's flange interface.  Like the
# real annulus it tessellates as arc patches -> its center is ON the file tool
# axis (0,0), not at the vertex centroid.
v362, _ = load_stl(f'{OUT}/model/meshes_baked/part_362_baked.stl')
top362 = v362[v362[:, 2] > v362[:, 2].max() - 0.0003]
c_disc = np.array([0.0, 0.0, float(top362[:, 2].mean())])
print(f"  print top face: z={c_disc[2]*1000:.2f}mm (n={len(top362)}) -> mates flange plane y={face_c_w3[1]*1000:.2f}mm, axis<->axis")
standoff = S - face_rel[2]
print(f"printed adapter: face z={S*1000:.1f}mm -> gripper standoff={standoff*1000:.1f}mm")
# mate the disc face onto the flange face (concentric + flush)
gpos = face_c_w3 - R_gw @ c_disc
print(f"flange center={np.round(face_c_w3, 4)}  gripper body pos={np.round(gpos, 4)}")
X.append(f'      <body name="base_link" pos="{gpos[0]:.6f} {gpos[1]:.6f} {gpos[2]:.6f}" '
         f'quat="{q_gw[0]:.6f} {q_gw[1]:.6f} {q_gw[2]:.6f} {q_gw[3]:.6f}">')
# gripper base inertial: baseline + camera module (expressed in gripper base frame)
base_inert = re.search(r'<body name="base_link"[^>]*>\s*(<inertial[^/]*/>)', gsrc).group(1)
m0 = float(re.search(r'mass="([\d.e+-]+)"', base_inert).group(1))
p0 = np.array([float(x) for x in re.search(r'pos="([^"]+)"', base_inert).group(1).split()])
Rt = T_VG[:3, :3].T; tv = T_VG[:3, 3]
qi = re.search(r'quat="([^"]+)"', base_inert)
di = [float(x) for x in re.search(r'diaginertia="([^"]+)"', base_inert).group(1).split()]
R0 = R_from_quat(np.array(qi.group(1).split(), float)) if qi else np.eye(3)
I0 = R0 @ np.diag(di) @ R0.T

def cam_inG(pid):
    mi = part_mass(pid)
    cV = np.array(byid[pid]['bbox']['min']) + (np.array(byid[pid]['bbox']['max']) - np.array(byid[pid]['bbox']['min'])) / 2
    cV = cV / 1000.0
    cG = Rt @ (cV - tv / 1000.0)
    IG = np.zeros((3, 3))   # point-mass approximation for camera bits
    return mi, cG, IG

bodies = [(m0, p0, I0)] + [cam_inG(pid) for pid in sorted(camera)]
mtot = sum(b[0] for b in bodies)
mc = sum(b[0] * b[1] for b in bodies) / mtot
Icom = np.zeros((3, 3))
for m, ci, Ii in bodies:
    d = ci - mc
    Icom += Ii + m * (np.dot(d, d) * np.eye(3) - np.outer(d, d))
X.append(f'        <inertial pos="{mc[0]:.6f} {mc[1]:.6f} {mc[2]:.6f}" mass="{mtot:.6f}" '
         f'fullinertia="{Icom[0,0]:.9f} {Icom[1,1]:.9f} {Icom[2,2]:.9f} {Icom[0,1]:.11f} {Icom[0,2]:.11f} {Icom[1,2]:.11f}"/>')
# baseline base_link geoms + child bodies (gear/fingers), verbatim structure
groot = ET.fromstring(gsrc)
gbase = [b for b in groot.iter('body') if b.get('name') == 'base_link'][0]
for g in gbase.findall('geom'):
    a = dict(g.attrib)
    a['mesh'] = 'g_' + a['mesh']
    a['rgba'] = '0.58 0.59 0.62 1'
    X.append('        ' + ET.tostring(ET.Element('geom', a), encoding='unicode'))
# camera geoms: transform inv(T_VG) (mm) applied to V-frame meshes
# camera + adapter geoms: meshes baked in gripper-body frame (pos=0)
for pid in sorted(camera):
    X.append(f'        <geom name="cam_{pid}" type="mesh" mesh="c_{pid}" group="1" rgba="0.15 0.15 0.16 1"/>')
for pid in sorted(adapter):
    X.append(f'        <geom name="ad_{pid}" type="mesh" mesh="c_{pid}" group="1" rgba="0.62 0.63 0.66 1"/>')

# gear + finger bodies verbatim (from baseline), meshes renamed
for bname in ['gear_link', 'right_finger_link', 'left_finger_link']:
    b = [x for x in groot.iter('body') if x.get('name') == bname][0]
    attrs = {k: v for k, v in b.attrib.items()}
    a_str = ' '.join(f'{k}="{v}"' for k, v in attrs.items())
    X.append(f'      <body {a_str}>')
    j = b.find('joint'); X.append('        ' + ET.tostring(ET.Element('joint', dict(j.attrib)), encoding='unicode'))
    bi = b.find('inertial'); X.append('        ' + ET.tostring(ET.Element('inertial', dict(bi.attrib)), encoding='unicode'))
    for g in b.findall('geom'):
        a = dict(g.attrib); a['mesh'] = 'g_' + a['mesh']
        a['rgba'] = '0.50 0.51 0.54 1'
        X.append('        ' + ET.tostring(ET.Element('geom', a), encoding='unicode'))
    X.append('      </body>')
X.append('      </body>')   # base_link
X.append('    </body>')     # wrist3
for _ in range(6):
    X.append('  </body>')   # wrist2..shoulder(5) + arm_base_mount
X.append('  </body>')       # machine
X.append('  </worldbody>')

# ---------------- equality / contact / actuators ----------------
X.append('  <equality>')
for e in groot.iter('joint'):
    if e.tag.endswith('joint') and e.get('joint1'):
        X.append('    ' + ET.tostring(ET.Element('joint', dict(e.attrib)), encoding='unicode'))
X.append('  </equality>')
X.append('  <contact>')
for e in groot.iter('exclude'):
    X.append('    ' + ET.tostring(ET.Element('exclude', dict(e.attrib)), encoding='unicode'))
for w in WHEELS:   # wheels must spin free inside their mounts
    X.append(f'    <exclude body1="chassis" body2="{w}"/>')
X.append('    <exclude body1="arm_base_mount" body2="shoulder_Link"/>')  # official meshes overlap at the joint by design
for _wb in ['wrist1_Link', 'wrist2_Link', 'wrist3_Link']:   # end-effector bolts flush onto the flange
    X.append(f'    <exclude body1="base_link" body2="{_wb}"/>')
X.append('  </contact>')
X.append('  <actuator>')
for a in groot.iter('position'):
    X.append('    ' + ET.tostring(ET.Element('position', dict(a.attrib)), encoding='unicode'))
for w in ['act_left', 'act_right']:
    X.append(f'    <velocity name="{w}_servo" joint="{w}_joint" kv="8" ctrlrange="-40 40"/>')
X.append('    <position name="j1" joint="shoulder_joint" kp="1500" kv="90" ctrlrange="-3.0543 3.0543" forcerange="-80 80"/>')
X.append('    <position name="j2" joint="upperArm_joint" kp="1500" kv="90" ctrlrange="-3.0543 3.0543" forcerange="-80 80"/>')
X.append('    <position name="j3" joint="foreArm_joint" kp="900" kv="60" ctrlrange="-3.0543 3.0543" forcerange="-80 80"/>')
X.append('    <position name="j4" joint="wrist1_joint" kp="300" kv="22" ctrlrange="-3.0543 3.0543" forcerange="-40 40"/>')
X.append('    <position name="j5" joint="wrist2_joint" kp="300" kv="22" ctrlrange="-3.0543 3.0543" forcerange="-40 40"/>')
X.append('    <position name="j6" joint="wrist3_joint" kp="200" kv="15" ctrlrange="-3.0543 3.0543" forcerange="-40 40"/>')
X.append('  </actuator>')
X.append('  <keyframe>')
_q1,_q2,_q3,_q4,_q5,_q6 = 1.570796, -0.802851, 0.802851, 0.034907, 1.570796, 0.000000   # J4/J5 re-solved for the NEW tool axis (+z_w3) pointing straight down; in-range margins 173/85 deg, residual 1.3e-5
_gear = 0.0
_kf = ['0.000000']*19
_kf[10:16] = [f'{_q1:.6f}', f'{_q2:.6f}', f'{_q3:.6f}', f'{_q4:.6f}', f'{_q5:.6f}', f'{_q6:.6f}']
_kf[16] = f'{_gear:.6f}'; _kf[17] = '0.000000'; _kf[18] = '0.000000'
X.append(f'    <key name="home" qpos="{" ".join(_kf)}"/>')
X.append('  </keyframe>')
X.append('</mujoco>')

import xml.etree.ElementTree as ET2  # ensure ET imported
open(f'{ASM}/tomato_picker.xml', 'w').write('\n'.join(X))
print("written", f'{ASM}/tomato_picker.xml')
