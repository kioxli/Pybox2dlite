from Body import Body
import numpy as np

from Math import FromAngleToMatrix

class Joint:
    M = np.zeros((2,2))
    localAnchor1=np.zeros((2))
    localAnchor2=np.zeros((2))
    r1=np.zeros((2))
    r2=np.zeros((2))
    bias=np.zeros((2))
    P=np.zeros((2))  # 累计冲量
    body1=Body()    # 物体1
    body2=Body()    # 物体2
    biasFactor=0.2
    softness=0.2
    anchor=np.zeros((2))

    def Set(self, body1:Body, body2:Body, anchor:np.array):
        self.body1=body1
        self.body2=body2

        Rot1=FromAngleToMatrix(body1.angle)
        Rot2=FromAngleToMatrix(body2.angle)
        
        Rot1T=Rot1.T
        Rot2T=Rot2.T

        self.localAnchor1=Rot1T @ (anchor-body1.position)
        self.localAnchor2=Rot2T @ (anchor-body2.position)

        self.P=np.zeros((2))
        self.biasFactor=0.2
        self.softness=0.0

    def PreStep(self, inv_dt:float):

        Rot1=FromAngleToMatrix(self.body1.angle)
        Rot2=FromAngleToMatrix(self.body2.angle)

        self.r1=Rot1 @ self.localAnchor1
        self.r2=Rot2 @ self.localAnchor2

        K1=np.array([[self.body1.invMass+self.body2.invMass, 0.0], [0.0, self.body1.invMass+self.body2.invMass]])

        K2=np.array([[self.body1.invI*self.r1[1]*self.r1[1], -self.body1.invI*self.r1[0]*self.r1[1]], [-self.body1.invI*self.r1[0]*self.r1[1], self.body1.invI*self.r1[0]*self.r1[0]]])

        K3=np.array([[self.body2.invI*self.r2[1]*self.r2[1], -self.body2.invI*self.r2[0]*self.r2[1]], [-self.body2.invI*self.r2[0]*self.r2[1], self.body2.invI*self.r2[0]*self.r2[0]]])

        K=K1+K2+K3
        K[0,0] += self.softness
        K[1,1] += self.softness

        self.M=np.linalg.inv(K)

        p1=self.body1.position+self.r1
        p2=self.body2.position+self.r2
        dp=p2-p1

        from World import World
        if World.positionCorrection:

            self.bias=-self.biasFactor*inv_dt*dp
        else:
            self.bias=np.zeros((2))

        if World.warmStarting:
            self.body1.velocity-=self.body1.invMass*self.P
            self.body1.angular_velocity-=self.body1.invI*np.cross(self.r1,self.P)
            self.body2.velocity+=self.body2.invMass*self.P
            self.body2.angular_velocity+=self.body2.invI*np.cross(self.r2,self.P)
        else:
            self.P=np.zeros((2))
    
    def ApplyImpulse(self):

        dv=self.body2.velocity+self.body2.angular_velocity*(self.r2[[1,0]]*[-1,1])-self.body1.velocity-self.body1.angular_velocity*(self.r1[[1,0]]*[-1,1])
        impulse=self.M @ (self.bias-dv-self.softness*self.P)

        self.body1.velocity-=self.body1.invMass*impulse
        self.body1.angular_velocity-=self.body1.invI*np.cross(self.r1,impulse)
        self.body2.velocity+=self.body2.invMass*impulse
        self.body2.angular_velocity+=self.body2.invI*np.cross(self.r2,impulse)
        
        self.P+=impulse