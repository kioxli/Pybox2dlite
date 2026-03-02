import numpy as np
from Body import Body
from World import World
from Arbiter import Arbiter   
from Collide import Contact
# -------------------------
# 1. World 开关
# -------------------------
World.accumulateImpulses = True
World.positionCorrection = True
World.warmStarting = True

dt = 1.0 / 60.0
inv_dt = 60.0

# -------------------------
# 2. 创建刚体
# -------------------------
bodyA = Body(
    mass=1.0,
    width=2.0,
    height=2.0,
    pos=[0.0, 0.0],
    angle=0.0,
    velocity=[0.0, 0.0],
    angular_velocity=0.0,
    force=[0.0, 0.0],
    torque=0.0,
    friction=0.3,
    name="StaticBox"
)

bodyB = Body(
    mass=1.0,
    width=1.0,
    height=1.0,
    pos=[0.8, 0.9],
    angle=np.pi / 6,   # 30°
    velocity=[0.0, 0.0],
    angular_velocity=0.0,
    force=[0.0, 0.0],
    torque=0.0,
    friction=0.3,
    name="RotatedBox"
)
# print('物体A',bodyA.name, '质量：',bodyA.mass, '宽度：',bodyA.width, '高度：',bodyA.height, '位置：',bodyA.position, '角度：',bodyA.angle, '线速度：',bodyA.velocity, '角速度：',bodyA.angular_velocity, '力：',bodyA.force, '扭矩：',bodyA.torque, '摩擦力：',bodyA.friction,'invMass:',bodyA.invMass, 'invI:',bodyA.invI)
# print('物体B',bodyB.name, '质量：',bodyB.mass, '宽度：',bodyB.width, '高度：',bodyB.height, '位置：',bodyB.position, '角度：',bodyB.angle, '线速度：',bodyB.velocity, '角速度：',bodyB.angular_velocity, '力：',bodyB.force, '扭矩：',bodyB.torque, '摩擦力：',bodyB.friction,'invMass:',bodyB.invMass, 'invI:',bodyB.invI)

def print_contacts(arbiter):
    print("\n=== Contacts before solving ===")
    print(f"Number of contacts: {arbiter.numContacts}")

    for i, c in enumerate(arbiter.contacts):
        print(f"\nContact {i}:")
        print(f"  position     = {c.position}")
        print(f"  normal       = {c.normal}")
        print(f"  separation   = {c.separation}")
        print(f"  feature id   = {c.feature}")
        print(f"  Pn / Pt      = ({c.Pn}, {c.Pt})")

# -------------------------
# 4. 创建 Arbiter
# -------------------------
arbiter = Arbiter(bodyA, bodyB)
# ✅ 关键：先打印接触
# print_contacts(arbiter)

# -------------------------
# 5. 求解
# -------------------------
print("\n=== PreStep ===")
arbiter.PreStep(inv_dt)
print('After PreStep:')
for i in range(arbiter.numContacts):
    c = arbiter.contacts[i]
    print(f"接触点 [{i}]:")
    print(f"  - 位置: ({c.position[0]:.4f}, {c.position[1]:.4f})")
    
    print(f"  - [辅助参数]")
    print(f"    法向有效质量 (massNormal):  {c.massNormal:.4f}")
    print(f"    切向有效质量 (massTangent): {c.massTangent:.4f}")
    print(f"    位置修正偏置 (bias):        {c.bias:.4f}")

    print(f"  - [累计冲量 (Pn/Pt)]")
    print(f"    Pn: {c.Pn:.4f} | Pt: {c.Pt:.4f}")


print("\n=== Iterative Solver ===")
for it in range(10):
    arbiter.ApplyImpulse()

    c = arbiter.contacts[0]
    print(f"\nIteration {it+1}")
    print(f"  Pn (normal impulse): {c.Pn:.6f}")
    print(f"  Pt (tangent impulse): {c.Pt:.6f}")
    print(f"  Bias: {c.bias:.6f}")

    print(f"  BodyA vel: {bodyA.velocity}, ω: {bodyA.angular_velocity:.6f}")
    print(f"  BodyB vel: {bodyB.velocity}, ω: {bodyB.angular_velocity:.6f}")
