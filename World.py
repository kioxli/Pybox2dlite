import numpy as np
from Arbiter import Arbiter


class World:

    accumulateImpulses=True # 是否积累冲量
    warmStarting=True
    positionCorrection=True

    def __init__(self, gravity, iterations):
        self.gravity=gravity
        self.iterations=iterations
        self.bodies=[]
        self.joints=[]
        self.arbiters={}


    def add_body(self, body):
        self.bodies.append(body)

    def add_joint(self, joint):
        self.joints.append(joint)

    def clear(self):
        self.bodies=[]
        self.joints=[]
        self.arbiters=[]

    def BoardPhase(self):
        for i in range(len(self.bodies)):
            for j in range(i+1,len(self.bodies)):

                if (self.bodies[i].invMass==0.0 and self.bodies[j].invMass==0.0):
                    continue

                newArb=Arbiter(self.bodies[i],self.bodies[j])
                key=(min(id(self.bodies[i]),id(self.bodies[j])),max(id(self.bodies[i]),id(self.bodies[j])))
                # print("key:",key)
                if newArb.numContacts > 0:
                    if key in self.arbiters:
                        # 找到旧的，更新它
                        self.arbiters[key].Update(newArb.contacts, newArb.numContacts)
                    else:
                        # 存入新的
                        self.arbiters[key] = newArb
                else:
                    # 分开了，删除记录
                    # print("删除Arbiter:", key)
                    self.arbiters.pop(key, None)

    def Step(self,dt):
        inv_dt=1.0/dt

        # BroadPhase
        self.BoardPhase()

        # Integrate Forces
        for i in range(len(self.bodies)):

            b=self.bodies[i]    
            if b.invMass==0.0:
                continue

            b.velocity+=dt*(self.gravity+b.invMass*b.force)
            b.angular_velocity+=dt*b.invI*b.torque

        # Perform pre-steps
        for arb in self.arbiters.values():
            arb.PreStep(inv_dt)

        for j in range(len(self.joints)):
            self.joints[j].PreStep(inv_dt)
            
        # Perform iterations
        for _ in range(self.iterations):
            for arb in self.arbiters.values():
                arb.ApplyImpulse()
            
            for j in range(len(self.joints)):
                self.joints[j].ApplyImpulse()

        #Integrate Velocities
        for i in range(len(self.bodies)):
            b=self.bodies[i]
            if b.invMass==0.0:
                continue

            b.position+=dt*b.velocity
            b.angle+=dt*b.angular_velocity

            b.force=np.array([0.0,0.0])
            b.torque=0.0

            