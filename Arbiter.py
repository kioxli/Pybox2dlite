from Body import Body
import numpy as np
from Collide import Collide, Contact
import copy



class Arbiter:  
    
    MAX_POINTS=2

    def __init__(self, bodyA, bodyB):
        self.body1 = bodyA  
        self.body2 = bodyB

        # 1. 预分配两个空的 Contact 对象
        # 这样 self.contacts[0] 和 self.contacts[1] 永远存在
        self.contacts = [Contact() for _ in range(self.MAX_POINTS)]
        self.numContacts = 0

        self.friction = np.sqrt(self.body1.friction * self.body2.friction)

        # 2. 调用 Collide 获取本次碰撞的实际数据
        num, new_contacts = Collide(self.body1, self.body2)
        self.restitution = max(self.body1.restitution , self.body2.restitution)
        # 3. 将新数据“填充”进预分配好的列表中，而不是直接覆盖 self.contacts
        self.numContacts = num
        for i in range(num):
            self.contacts[i] = new_contacts[i]
        # print("=================================")
        # print('From Arbiter.py')
        # print('物体A',self.body1.name, '质量：',self.body1.mass, '宽度：',self.body1.width, '高度：',self.body1.height, '位置：',self.body1.position, '角度：',self.body1.angle)
        # print('物体B',self.body2.name, '质量：',self.body2.mass, '宽度：',self.body2.width, '高度：',self.body2.height, '位置：',self.body2.position, '角度：',self.body2.angle)
    
        # print("========== 接触检测结果 ==========")
        # print(f"接触点数量 numContacts = {self.numContacts}")

        # for i, c in enumerate(self.contacts):
        #     print(f"\n--- Contact {i} ---")
        #     print(f"位置 position   = {c.position}")
        #     print(f"法线 normal     = {c.normal}")
        #     print(f"分离 separation = {c.separation}")
        #     print(f"Pn (normal impulse) = {c.Pn}")
        #     print(f"Pt (tangent impulse)= {c.Pt}")
        #     print(f"feature.value  = {c.feature}")

        # print("=================================")
    def Update(self, newContacts, numNewContacts):

        mergedContacts=[None]*self.MAX_POINTS

        for i in range(numNewContacts):
            cNew=newContacts[i]
            k=-1
            for j in range(self.numContacts):
                cOld=self.contacts[j]
                if cNew.feature==cOld.feature:
                    k=j
                    break
            
            if k > -1:
                # 找到匹配点：
                # A. 必须拷贝新点的几何数据（位置、法线、深度）
                c_merged = copy.copy(cNew) 
                cOld = self.contacts[k]
                from World import World
                # B. 继承旧点的冲量（热启动）
                if World.warmStarting:
                    c_merged.Pn = cOld.Pn
                    c_merged.Pt = cOld.Pt
                    c_merged.Pnb = cOld.Pnb
                else:
                    c_merged.Pn = 0.0
                    c_merged.Pt = 0.0
                    c_merged.Pnb = 0.0
                
                mergedContacts[i] = c_merged
            else:
                # 没找到匹配点：这是一个全新的碰撞点
                mergedContacts[i] = copy.copy(cNew)
        print("numNewContacts:",numNewContacts)
        print("numContacts:",self.numContacts)
        for i in range(numNewContacts):
            
            self.contacts[i]=mergedContacts[i]
        
        self.numContacts=numNewContacts

    def PreStep(self, inv_dt):
        # 计算contact中的massnormal和masstangent
        # 如果warm starting，修改body的velocity和angular velocity。
        from World import World
        print("World.positionCorrection:",World.positionCorrection)
        k_allowedPenetration=0.01
        k_biasFactor=0.2 if World.positionCorrection else 0.0

        # 弹性速度阈值：如果撞击速度低于这个值，就不弹了（防止静止时在地上抖动）
        restitution_threshold=1.0

        for i in range(self.numContacts):
            c=self.contacts[i] # 对于第i个接触

            r1=c.position-self.body1.position
            r2=c.position-self.body2.position

            rn1=np.dot(r1,c.normal)
            rn2=np.dot(r2,c.normal)

            kNormal=self.body1.invMass+self.body2.invMass
            kNormal+=self.body1.invI*(np.dot(r1,r1)-rn1*rn1)+self.body2.invI*(np.dot(r2,r2)-rn2*rn2)
            c.massNormal=1.0/kNormal
            tangent=c.normal[[1, 0]] * [1, -1] # 顺时针旋转90°，(x,y)->(-y,x)
            rt1=np.dot(r1,tangent)
            rt2=np.dot(r2,tangent)

            kTangent=self.body1.invMass+self.body2.invMass
            kTangent+=self.body1.invI*(np.dot(r1,r1)-rt1*rt1)+self.body2.invI*(np.dot(r2,r2)-rt2*rt2)
            c.massTangent=1.0/kTangent

            # 
            position_bias=-k_biasFactor*inv_dt*min(0.0,c.separation+k_allowedPenetration)

            # --- 1. 计算当前的相对速度 vn (这部分是新加的) ---
            # v_rel = (v2 + w2 x r2) - (v1 + w1 x r1)
            v1 = self.body1.velocity + self.body1.angular_velocity * (r1[[1,0]] * [-1,1])
            v2 = self.body2.velocity + self.body2.angular_velocity * (r2[[1,0]] * [-1,1])
            dv = v2 - v1
            vn = np.dot(dv, c.normal)
            # restitution bias
            resitituion_bias=0.0
            if vn<-restitution_threshold:
                resitituion_bias=-self.restitution*vn
            c.bias=position_bias+resitituion_bias

            if World.accumulateImpulses:
                P=c.Pn*c.normal+c.Pt*tangent # 上一帧法向和切向的冲量
                self.body1.velocity-=self.body1.invMass*P # 更新速度
                self.body1.angular_velocity-=self.body1.invI*np.cross(r1,P) # 更新角速度
                self.body2.velocity+=self.body2.invMass*P # 更新速度
                self.body2.angular_velocity+=self.body2.invI*np.cross(r2,P) # 更新角速度


                
    def ApplyImpulse(self):
        b1=self.body1
        b2=self.body2

        for i in range(self.numContacts):
            # print('Begin Apply Impluse for contact:',i)
            c=self.contacts[i]

            c.r1=c.position-b1.position
            c.r2=c.position-b2.position

            # 施加法向冲量
            # 在接触点的相对速度
            dv=b2.velocity+b2.angular_velocity*(c.r2[[1,0]]*[-1,1])-b1.velocity-b1.angular_velocity*(c.r1[[1,0]]*[-1,1])
            vn=np.dot(dv,c.normal) # 法向速度
            dPn=c.massNormal*(-vn+c.bias) # 法向冲量

            from World import World
            if World.accumulateImpulses:
                Pn0=c.Pn
                c.Pn=max(Pn0+dPn,0.0)
                dPn=c.Pn-Pn0
            else:
                dPn=max(dPn,0.0)
            # print('dPn after clamping:',dPn)
            Pn=dPn*c.normal
            # print('Pn:',Pn)
            b1.velocity-=b1.invMass*Pn
            b1.angular_velocity-=b1.invI*np.cross(c.r1,Pn)

            b2.velocity+=b2.invMass*Pn
            b2.angular_velocity+=b2.invI*np.cross(c.r2,Pn)

            # 施加切向冲量
            dv=b2.velocity+b2.angular_velocity*(c.r2[[1,0]]*[-1,1])-b1.velocity-b1.angular_velocity*(c.r1[[1,0]]*[-1,1])

            tangent=c.normal[[1, 0]] * [1, -1] # 顺时针旋转90°，(x,y)->(-y,x)
            # print('tangent',tangent)
            vt=np.dot(dv,tangent)
            dPt=c.massTangent*(-vt)
            if World.accumulateImpulses:
                maxPt=self.friction*c.Pn
                oldTangentImpulse=c.Pt
                c.Pt=np.clip(oldTangentImpulse+dPt,-maxPt,maxPt)
                dPt=c.Pt-oldTangentImpulse
            else:
                maxPt=self.friction*dPn
                dPt=np.clip(dPt,-maxPt,maxPt)
            
            Pt=dPt*tangent
            b1.velocity-=b1.invMass*Pt
            b1.angular_velocity-=b1.invI*np.cross(c.r1,Pt)
            b2.velocity+=b2.invMass*Pt
            b2.angular_velocity+=b2.invI*np.cross(c.r2,Pt)
    