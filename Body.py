import numpy as np

class Body:
    
    def __init__(self, mass=1.0, width=1.0, height=1.0, pos=[0.0, 0.0], angle=0.0, velocity=[0.0, 0.0], angular_velocity=0.0, force=[0.0, 0.0], torque=0.0, friction=0.3, restitution=0.3, name=None):

        self.name = name
    
        # --- 质量 ---
        self.mass = mass
        self.invMass = 0.0 if mass==float('inf') else 1.0 / mass

        # --- 尺寸 ---
        self.width = width # 全尺寸的宽度
        self.height = height # 全尺寸的高度

        # --- 状态 ---
        self.position = np.array(pos, dtype=np.float64)
        self.angle = angle

        self.velocity = np.zeros(2, dtype=np.float64)   # 线速度
        self.angular_velocity = angular_velocity                          # 角速度 ω

        self.force = np.zeros(2, dtype=np.float64)
        self.torque = 0.0
        self.friction = friction

        self.I = self.mass * (self.width * self.width + self.height * self.height) / 12.0 if mass<=float('inf') else float('inf')
        self.invI = 0.0 if self.I ==float('inf') else 1.0 / self.I

        self.restitution = restitution
    def add_force(self, force):
        self.force += force

if __name__ == "__main__":
    body = Body(1.0, 1.0, 1.0, [0.0, 0.0], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "A")
    print('物体：', '名字：',body.name, '质量：',body.m, '宽度：',body.w, '高度：',body.h, '位置：',body.pos, '角度：',body.angle, '线速度：',body.v, '角速度：',body.angular_velocity, '力：',body.f, '扭矩：',body.torque, '摩擦力：',body.friction)
    body.add_force([1.0, 1.0])
    print('物体：', '名字：',body.name, '质量：',body.m, '宽度：',body.w, '高度：',body.h, '位置：',body.pos, '角度：',body.angle, '线速度：',body.v, '角速度：',body.angular_velocity, '力：',body.f, '扭矩：',body.torque, '摩擦力：',body.friction)
