"""
record_demo.py — 录制 Box2D-Lite Demo 视频
═══════════════════════════════════════════════════════════════

场景: 斜坡 + 多个方块掉落碰撞 + 摆锤

运行:
    cd Py_box2d
    python record_demo.py

输出:
    demo.mp4
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # 离屏渲染，不弹窗
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyBboxPatch, Circle
from Body import Body
from World import World
from Joint import Joint

# ── 录制参数 ─────────────────────────────────────────────────
TOTAL_FRAMES = 600
FPS = 60
DT = 1.0 / FPS
RES_W, RES_H = 10, 8  # 图像尺寸 (英寸)
DPI = 120
OUTPUT_FILE = "demo.mp4"
FRAMES_DIR = "_frames"


def make_box(name, pos, width, height, mass=1.0, friction=0.3, angle=0.0):
    body = Body()
    body.name = name
    body.mass = mass
    body.position = np.array(pos, dtype=float)
    body.angle = angle
    body.velocity = np.array([0.0, 0.0])
    body.angular_velocity = 0.0
    body.force = np.array([0.0, 0.0])
    body.torque = 0.0
    body.width = width
    body.height = height
    body.friction = friction
    body.restitution = 0.2
    if mass == float("inf"):
        body.invMass = 0.0
        body.I = float("inf")
        body.invI = 0.0
    else:
        body.invMass = 1.0 / mass
        body.I = mass * (width**2 + height**2) / 12.0
        body.invI = 1.0 / body.I
    return body


# ── 颜色方案 ─────────────────────────────────────────────────
COLORS = {
    "Ground":  {"face": "#4a4a4a", "edge": "#2a2a2a"},
    "Ramp":    {"face": "#8B4513", "edge": "#5C2D0E"},
    "Pivot":   {"face": "#888888", "edge": "#555555"},
    "default": [
        {"face": "#4682B4", "edge": "#2C5F8A"},
        {"face": "#CD5C5C", "edge": "#8B3A3A"},
        {"face": "#3CB371", "edge": "#2E8B57"},
        {"face": "#DAA520", "edge": "#B8860B"},
        {"face": "#9370DB", "edge": "#6A4EB0"},
        {"face": "#FF7F50", "edge": "#E06030"},
    ],
}
_color_idx = 0


def get_body_color(name):
    global _color_idx
    if name in COLORS:
        return COLORS[name]
    c = COLORS["default"][_color_idx % len(COLORS["default"])]
    _color_idx += 1
    return c


def draw_body(ax, body, color):
    half_w = body.width / 2.0
    half_h = body.height / 2.0
    vertices = np.array([
        [-half_w, -half_h],
        [ half_w, -half_h],
        [ half_w,  half_h],
        [-half_w,  half_h],
    ])
    cos_t = np.cos(body.angle)
    sin_t = np.sin(body.angle)
    rot = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
    world_v = vertices @ rot.T + body.position

    poly = Polygon(
        world_v, closed=True,
        facecolor=color["face"], edgecolor=color["edge"],
        linewidth=1.5, alpha=0.9, zorder=2,
    )
    ax.add_patch(poly)

    # 旋转指示线
    if body.mass != float("inf"):
        center = body.position
        front = center + np.array([cos_t, sin_t]) * min(half_w, half_h) * 0.8
        ax.plot([center[0], front[0]], [center[1], front[1]],
                color="white", linewidth=1.5, zorder=3)


def draw_joint(ax, joint):
    r1 = joint.body1.position + joint.r1
    r2 = joint.body2.position + joint.r2
    ax.plot([r1[0], r2[0]], [r1[1], r2[1]], "k-", linewidth=2, zorder=1)
    ax.scatter([r1[0], r2[0]], [r1[1], r2[1]], c="red", s=30, zorder=5)


def build_scene():
    """构建演示场景"""
    world = World(gravity=np.array([0.0, -9.81]), iterations=20)
    body_colors = {}

    # 地面
    ground = make_box("Ground", pos=(0, -0.5), width=30, height=1, mass=float("inf"), friction=0.5)
    world.add_body(ground)
    body_colors[id(ground)] = get_body_color("Ground")

    # 左侧斜坡
    ramp = make_box("Ramp", pos=(-2.0, 3.0), width=6, height=0.3, mass=float("inf"), friction=0.4, angle=-0.3)
    world.add_body(ramp)
    body_colors[id(ramp)] = get_body_color("Ramp")

    # 右侧平台
    platform = make_box("Ground", pos=(4.0, 1.8), width=3, height=0.3, mass=float("inf"), friction=0.4)
    world.add_body(platform)
    body_colors[id(platform)] = get_body_color("Ground")

    # 摆锤: 固定铰点 + 悬挂方块
    pivot = make_box("Pivot", pos=(6.0, 7.0), width=0.3, height=0.3, mass=float("inf"))
    world.add_body(pivot)
    body_colors[id(pivot)] = get_body_color("Pivot")

    bob = make_box("Bob", pos=(8.0, 5.5), width=0.8, height=0.8, mass=2.0, friction=0.3)
    world.add_body(bob)
    body_colors[id(bob)] = get_body_color(bob.name)

    joint = Joint()
    joint.Set(pivot, bob, np.array([6.0, 7.0], dtype=float))
    world.add_joint(joint)

    return world, body_colors


def spawn_box(world, body_colors, name, pos, width, height, mass=1.0, angle=0.0):
    """动态添加方块"""
    box = make_box(name, pos=pos, width=width, height=height, mass=mass, angle=angle, friction=0.4)
    world.add_body(box)
    body_colors[id(box)] = get_body_color(box.name)
    return box


# ── 动态投放时间表 ───────────────────────────────────────────
# (帧号, 名称, 位置, 宽, 高, 质量, 角度)
SPAWN_SCHEDULE = [
    (1,   "A", (-3.0, 6.0), 0.7, 0.7, 1.5, 0.1),
    (30,  "B", (-1.5, 7.0), 0.5, 0.9, 1.0, -0.2),
    (60,  "C", (-4.0, 7.5), 0.6, 0.6, 1.2, 0.3),
    (120, "D", (0.0, 8.0),  0.8, 0.5, 2.0, 0.0),
    (180, "E", (4.0, 6.0),  0.5, 0.5, 0.8, 0.5),
    (240, "F", (-2.0, 9.0), 0.6, 0.8, 1.5, -0.1),
    (300, "G", (3.5, 8.0),  0.7, 0.7, 1.0, 0.4),
    (360, "H", (-3.5, 8.5), 0.9, 0.4, 1.8, 0.2),
    (420, "I", (1.0, 9.0),  0.5, 0.5, 1.0, -0.3),
    (480, "J", (5.0, 7.0),  0.6, 0.6, 1.2, 0.0),
]


def main():
    os.makedirs(FRAMES_DIR, exist_ok=True)

    world, body_colors = build_scene()

    fig, ax = plt.subplots(figsize=(RES_W, RES_H), dpi=DPI)

    print(f"\n开始录制 {TOTAL_FRAMES} 帧 → {OUTPUT_FILE}")
    print(f"  分辨率: {RES_W * DPI}x{RES_H * DPI}  |  帧率: {FPS} fps")
    print(f"  预计时长: {TOTAL_FRAMES / FPS:.1f} 秒\n")

    for frame in range(TOTAL_FRAMES):
        # 动态投放
        for t, name, pos, w, h, m, ang in SPAWN_SCHEDULE:
            if frame == t:
                spawn_box(world, body_colors, name, pos, w, h, m, ang)
                print(f"  帧 {frame:>4d}: 投放方块 {name} @ ({pos[0]:.1f}, {pos[1]:.1f})")

        world.Step(DT)

        # 绘制
        ax.clear()
        ax.set_xlim(-8, 10)
        ax.set_ylim(-1.5, 10)
        ax.set_aspect("equal")
        ax.set_facecolor("#1a1a2e")
        fig.patch.set_facecolor("#1a1a2e")
        ax.grid(True, alpha=0.15, color="white")
        ax.tick_params(colors="#666666", labelsize=7)
        for spine in ax.spines.values():
            spine.set_color("#333333")

        ax.set_title(
            f"Box2D-Lite  |  帧 {frame + 1}/{TOTAL_FRAMES}  |  "
            f"物体 {len(world.bodies)}  接触 {len(world.arbiters)}",
            color="white", fontsize=12, pad=10,
        )

        for body in world.bodies:
            color = body_colors.get(id(body), COLORS["default"][0])
            draw_body(ax, body, color)
        for j in world.joints:
            draw_joint(ax, j)

        # 保存帧
        fig.savefig(
            os.path.join(FRAMES_DIR, f"{frame:06d}.png"),
            dpi=DPI, facecolor=fig.get_facecolor(),
        )

        if (frame + 1) % 100 == 0:
            print(f"  帧 {frame + 1:>4d}/{TOTAL_FRAMES}  |  物体 {len(world.bodies)}")

    plt.close(fig)

    # ffmpeg 合成
    print("\n正在合成视频...")
    cmd = (
        f'ffmpeg -y -framerate {FPS} -i {FRAMES_DIR}/%06d.png '
        f'-c:v h264_mf -pix_fmt nv12 -b:v 5M "{OUTPUT_FILE}"'
    )
    ret = os.system(cmd)

    if ret == 0 and os.path.exists(OUTPUT_FILE):
        size_mb = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)
        print(f"\n录制完成: {OUTPUT_FILE} ({size_mb:.1f} MB)")
    else:
        print("\n视频合成失败，帧图片保留在", FRAMES_DIR)
        return

    # 清理帧文件
    import shutil
    shutil.rmtree(FRAMES_DIR, ignore_errors=True)
    print("临时帧文件已清理")


if __name__ == "__main__":
    main()
