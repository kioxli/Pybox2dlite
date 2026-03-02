import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from Body import Body
from World import World
from Joint import Joint

def make_box(name, pos, width, height, mass=1.0, friction=0.3, angle=0.0, restitution=0.2):
    """创建矩形刚体：修正了变量名，确保与 Body 类属性一致"""
    body = Body()
    body.name = name
    body.mass = mass
    body.position = np.array(pos, dtype=float)
    body.angle = angle  # ⚠️ 统一使用 angle
    
    body.velocity = np.array([0.0, 0.0])
    body.angular_velocity = 0.0
    body.force = np.array([0.0, 0.0])
    body.torque = 0.0

    body.width = width
    body.height = height
    body.friction = friction
    body.restitution = restitution
    if mass == float('inf'):
        body.invMass = 0.0
        body.I = float('inf')
        body.invI = 0.0
    else:
        body.invMass = 1.0 / mass
        # 矩形转动惯量公式: I = m * (w^2 + h^2) / 12
        body.I = mass * (width**2 + height**2) / 12.0
        body.invI = 1.0 / body.I
    return body

def draw_body(ax, body):
    """绘制矩形刚体：修正为读取 body.angle"""
    half_w = body.width / 2.0
    half_h = body.height / 2.0
    
    # 局部坐标系下的四个顶点
    vertices = np.array([
        [-half_w, -half_h],
        [ half_w, -half_h],
        [ half_w,  half_h],
        [-half_w,  half_h]
    ])
    
    # 使用正确的 angle 计算旋转矩阵
    cos_t = np.cos(body.angle)
    sin_t = np.sin(body.angle)
    rot = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
    
    # 转换到世界坐标系
    world_vertices = vertices @ rot.T + body.position
    
    facecolor = 'saddlebrown' if body.name == "Ramp" else 'steelblue'
    edgecolor = 'maroon' if body.name == "Ramp" else 'darkblue'
    alpha = 0.4 if body.name == "Ramp" else 0.95
    
    poly = Polygon(world_vertices, closed=True, 
                   facecolor=facecolor, edgecolor=edgecolor, 
                   linewidth=2.0, alpha=alpha, zorder=2)
    ax.add_patch(poly)
    
    # 绘制指向前进方向的中心线（辅助观察旋转）
    center = body.position
    front = center + np.array([cos_t, sin_t]) * 0.5
    ax.plot([center[0], front[0]], [center[1], front[1]], color='white', linewidth=1, zorder=3)


def draw_joint(ax, joint):
    """绘制关节连线（铰点与两体锚点）"""
    r1 = joint.body1.position + joint.r1
    r2 = joint.body2.position + joint.r2
    ax.plot([r1[0], r2[0]], [r1[1], r2[1]], 'k-', linewidth=2, zorder=1)
    ax.scatter([r1[0], r2[0]], [r1[1], r2[1]], c='red', s=40, zorder=5)


def run_pendulum():
    """摆锤：固定铰点 + 悬挂刚体，用 Joint 约束共点"""
    world = World(gravity=np.array([0.0, -9.81]), iterations=20)

    # 铰点（静止小方块）
    pivot = make_box("Pivot", pos=(0.0, 5.0), width=0.3, height=0.3, mass=float('inf'))
    world.add_body(pivot)

    # 摆锤（悬挂的矩形，中心在铰点下方一段距离）
    rod_length = 2.0
    bob = make_box(
        "Bob", 
        pos=(1.5, 3.5), # 偏向右侧一点
        width=0.6, 
        height=0.4, 
        mass=1.0,
        angle=0.5       # 给一个初始偏角
    )    
    world.add_body(bob)

    # 关节：世界坐标锚点 = 铰点位置，使 pivot 与 bob 在该点共点（相当于无质量杆）
    anchor = np.array([0.0, 5.0], dtype=float)
    joint = Joint()
    joint.Set(pivot, bob, anchor)
    world.add_joint(joint)

    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 8))
    dt = 1.0 / 60.0

    while plt.fignum_exists(fig.number):
        world.Step(dt)
        ax.clear()
        ax.set_xlim(-5, 5)
        ax.set_ylim(-2, 8)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title("Pendulum (Joint)")

        for body in world.bodies:
            draw_body(ax, body)
        for j in world.joints:
            draw_joint(ax, j)

        plt.pause(0.01)

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    run_pendulum()
