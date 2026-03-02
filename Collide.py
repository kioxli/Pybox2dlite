from ast import main
from typing import Any
from numpy import absolute, char, dtype, floating
import numpy as np
from Math import Vec2
from Math import FromAngleToMatrix
from Body import Body
import math

class FeaturePair:
    def __init__(self, inE1=0, outE1=0, inE2=0, outE2=0):
        self.inEdge1 = inE1
        self.outEdge1 = outE1
        self.inEdge2 = inE2
        self.outEdge2 = outE2

    def __repr__(self):
        return f"FeaturePair(inEdge1={self.inEdge1}, outEdge1={self.outEdge1}, inEdge2={self.inEdge2}, outEdge2={self.outEdge2})"

class ClipVertex:
    def __init__(self, v: np.ndarray, fp: FeaturePair):
        self.v = v # 点坐标
        self.fp = fp # 特征对

class EdgeNumbers:
    NO_EDGE = 0
    EDGE1 = 1
    EDGE2 = 2
    EDGE3 = 3
    EDGE4 = 4

class Axis:
    FACE_A_X = 0
    FACE_A_Y = 1
    FACE_B_X = 2
    FACE_B_Y = 3

class Contact:
    Pn=0.0 # 累计法向冲量
    Pt=0.0 # 累计切向冲量
    Pnb=0.0 # 累计法向冲量用于位置偏差
    massNormal=0.0 # 法向质量
    massTangent=0.0 # 切向质量
    bias=0.0 # 偏差
    feature=FeaturePair(0, 0, 0, 0) # 特征对
    position=np.array([0.0, 0.0]) # 位置
    normal=np.array([0.0, 0.0]) # 法线
    r1=np.array([0.0, 0.0]) # 相对位置1
    r2=np.array([0.0, 0.0]) # 相对位置2
    separation=0.0 # 分离距离

def Flip(fp: FeaturePair):
    fp.inEdge1, fp.inEdge2 = fp.inEdge2, fp.inEdge1
    fp.outEdge1, fp.outEdge2 = fp.outEdge2, fp.outEdge1
    return fp

def ClipSegmentToLine(
    vIn: tuple[ClipVertex, ClipVertex],
    normal: np.ndarray,
    offset: float,
    clipEdge: char
):
    
    vOutList = []

    # 计算线段两个端点到直线的举例
    distance0 = np.dot(normal, vIn[0].v) - offset
    distance1 = np.dot(normal, vIn[1].v) - offset

    # 如果距离小于等于0，则将点加入vOutList
    if distance0 <= 0.0:
        vOutList.append(vIn[0])
    if distance1 <= 0.0:
        vOutList.append(vIn[1])

    # 如果距离乘积小于0
    if distance0 * distance1 < 0.0:
        # 计算交点
        interp = distance0 / (distance0 - distance1)
        intersect_v = vIn[0].v + (vIn[1].v - vIn[0].v) * interp
        # 根据线段方向，生成交点的 FeaturePair   
        if distance0 > 0.0:
            # 场景：从外进到内 (distance0 是正数，表示在外面)
            # 继承起点 vIn[0] 的特征，并标记入边为当前 clipEdge
            new_fp = FeaturePair(
                clipEdge, 
                vIn[0].fp.outEdge1,
                EdgeNumbers.NO_EDGE, 
                vIn[0].fp.outEdge2
            )
        else:
            # 场景：从内出到外 (distance1 是正数，表示在外面)
            # 继承终点 vIn[1] 的特征，并标记出边为当前 clipEdge
            new_fp = FeaturePair(
                vIn[1].fp.inEdge1, 
                clipEdge,
                vIn[1].fp.inEdge2, 
                EdgeNumbers.NO_EDGE
            )

        vOutList.append(ClipVertex(v=intersect_v, fp=new_fp))

    return vOutList, len(vOutList)

'''
计算入射边
参数：
    half_height: 半高度
    half_width: 半宽度
    pos: 位置
    Rot: 旋转矩阵
    normal: 法线
返回：
    c: 入射边
    len(c): 入射边长度
'''

def ComputeIncidentEdge(
    half_height: float, half_width: float,
    pos: np.ndarray,
    Rot: np.ndarray,
    normal: np.ndarray
) -> tuple[ClipVertex, ClipVertex]:

    c = [ClipVertex(v=np.array([0, 0]), fp=FeaturePair(0, 0, 0, 0)), ClipVertex(v=np.array([0, 0]), fp=FeaturePair(0, 0, 0, 0))]

    RotT = Rot.T # 旋转矩阵的转置即为其逆矩阵
    n = np.array(-(RotT @ normal)) # 将法线转入局部坐标系并反向
    nAbs = np.abs(n) # 获取绝对值，用于快速判断法线主要落在哪个轴上

    if nAbs[0] > nAbs[1]: # 如果x轴的绝对值大于y轴的绝对值，则法线主要落在x轴上
        if n[0] > 0.0: # 如果x轴的值大于0，则法线主要落在x轴的正方向上
            c[0].v = np.array([half_width, -half_height]) # 位于局部坐标系的右下角
            c[0].fp.inEdge2 = EdgeNumbers.EDGE3
            c[0].fp.outEdge2 = EdgeNumbers.EDGE4

            c[1].v = np.array([half_width, half_height])
            c[1].fp.inEdge2 = EdgeNumbers.EDGE4
            c[1].fp.outEdge2 = EdgeNumbers.EDGE1

        else: # 如果x轴的值小于0，则法线主要落在x轴的负方向上
            c[0].v = np.array([-half_width, half_height]) # 位于局部坐标系的左下角
            c[0].fp.inEdge2 = EdgeNumbers.EDGE1
            c[0].fp.outEdge2 = EdgeNumbers.EDGE2

            c[1].v = np.array([-half_width, -half_height]) # 位于局部坐标系的左上角
            c[1].fp.inEdge2 = EdgeNumbers.EDGE2
            c[1].fp.outEdge2 = EdgeNumbers.EDGE3
    else:
        if n[1] > 0.0: # 如果y轴的值大于0，则法线主要落在y轴的正方向上
            c[0].v = np.array([half_width, half_height]) # 位于局部坐标系的右下角
            c[0].fp.inEdge2 = EdgeNumbers.EDGE4
            c[0].fp.outEdge2 = EdgeNumbers.EDGE1

            c[1].v = np.array([-half_width, half_height])
            c[1].fp.inEdge2 = EdgeNumbers.EDGE1
            c[1].fp.outEdge2 = EdgeNumbers.EDGE2
        else: # 如果y轴的值小于0，则法线主要落在y轴的负方向上
            c[0].v = np.array([-half_width, -half_height]) # 位于局部坐标系的左上角
            c[0].fp.inEdge2 = EdgeNumbers.EDGE2
            c[0].fp.outEdge2 = EdgeNumbers.EDGE3

            c[1].v = np.array([half_width, -half_height])
            c[1].fp.inEdge2 = EdgeNumbers.EDGE3
            c[1].fp.outEdge2 = EdgeNumbers.EDGE4

    c[0].v = pos + Rot @ c[0].v
    c[1].v = pos + Rot @ c[1].v

    return c

def Collide(bodyA:Body, bodyB:Body):

    # 在A的面上，检测是否有碰撞
    # 计算公式：AB的中心距离减去A的半长减去B在A轴向上的投影
    dP = bodyB.position-bodyA.position
    
    RotA=FromAngleToMatrix(bodyA.angle)
    RotB=FromAngleToMatrix(bodyB.angle)
    RotAT=RotA.T
    RotBT=RotB.T
    # dA: 将位移向量转到 A 的局部坐标系中 (A 看来 B 在哪)
    dA=RotAT @ dP
    dB=RotBT @ dP
    C=RotAT @ RotB
    absC=np.abs(C)
    absCT=absC.T

    # 修正后（正确）：
    hA = np.array([bodyA.width, bodyA.height]) * 0.5  # 半宽高
    hB = np.array([bodyB.width, bodyB.height]) * 0.5 
    faceA=np.abs(dA)-hA-absC @ hB
    faceB=np.abs(dB)-absCT @ hA-hB


    if (faceA[0] > 0.0 or faceA[1] > 0.0):
        return 0, []

    if (faceB[0] > 0.0 or faceB[1] > 0.0):
        return 0, []
    
    # ----------------------------------------
    
    axis=Axis.FACE_A_X # 最佳分离轴
    separation=0.0 # 分离距离
    normal=Vec2(0.0, 0.0) # 法向向量
    separation=faceA[0]
    normal=RotA[:,0] if dA[0]>0.0 else -RotA[:,0] # Vec2

    relativeTol=0.95
    absoluteTol=0.01

    threshold = relativeTol * separation + absoluteTol * hA[1]

    if faceA[1] > threshold:
        separation = faceA[1]  
        axis = Axis.FACE_A_Y
        normal = RotA[:, 1] if dA[1] > 0.0 else -RotA[:, 1]

    threshold = relativeTol * separation + absoluteTol * hB[0]

    if faceB[0] > threshold:
        separation = faceB[0]  
        axis=Axis.FACE_B_X
        normal = RotB[:, 0] if dB[0] > 0.0 else -RotB[:, 0]

    threshold = relativeTol * separation + absoluteTol * hB[1]

    if faceB[1] > threshold:
        separation = faceB[1]  
        axis=Axis.FACE_B_Y
        normal = RotB[:, 1] if dB[1] > 0.0 else -RotB[:, 1]

    # ----------------------------------------

    frontNormal=np.array([0,0]) # 参考面主法线
    sideNormal=np.array([0,0]) # 参考面切线
    incidentEdge=[] # 被裁剪的边

    front, negSide, posSide=0.0,0.0,0.0
    neg_edge, pos_edge=EdgeNumbers.NO_EDGE, EdgeNumbers.NO_EDGE

    if axis == Axis.FACE_A_X:
        frontNormal=normal
        front=np.dot(bodyA.position, frontNormal)+hA[0]
        sideNormal=RotA[:,1]
        side=np.dot(bodyA.position, sideNormal)
        negSide=-side+hA[1]
        posSide=side+hA[1]
        neg_edge=EdgeNumbers.EDGE3
        pos_edge=EdgeNumbers.EDGE1
        incidentEdge=ComputeIncidentEdge(hB[1], hB[0], bodyB.position, RotB, frontNormal)
    elif axis==Axis.FACE_A_Y:
        frontNormal=normal
        front=np.dot(bodyA.position, frontNormal)+hA[1]
        sideNormal=RotA[:,0]
        side=np.dot(bodyA.position, sideNormal)
        negSide=-side+hA[0]
        posSide=side+hA[0]
        neg_edge=EdgeNumbers.EDGE2
        pos_edge=EdgeNumbers.EDGE4
        incidentEdge=ComputeIncidentEdge(hB[1], hB[0], bodyB.position, RotB, frontNormal)
    elif axis==Axis.FACE_B_X:
        frontNormal=-normal
        front=np.dot(bodyB.position, frontNormal)+hB[0]
        sideNormal=RotB[:,1]
        side=np.dot(bodyB.position, sideNormal)
        negSide=-side+hB[1]
        posSide=side+hB[1]
        neg_edge=EdgeNumbers.EDGE3
        pos_edge=EdgeNumbers.EDGE1
        incidentEdge=ComputeIncidentEdge(hA[1], hA[0], bodyA.position, RotA, frontNormal)
    elif axis==Axis.FACE_B_Y:
        frontNormal=-normal
        front=np.dot(bodyB.position, frontNormal)+hB[1]
        sideNormal=RotB[:,0]
        side=np.dot(bodyB.position, sideNormal)
        negSide=-side+hB[0]
        posSide=side+hB[0]
        neg_edge=EdgeNumbers.EDGE2
        pos_edge=EdgeNumbers.EDGE4
        incidentEdge=ComputeIncidentEdge(hA[1], hA[0], bodyA.position, RotA, frontNormal)

    # ----------------------------------------

    points_num=0
    clipPoints1,points_num=ClipSegmentToLine(incidentEdge, -sideNormal, negSide,neg_edge)

    if points_num<2: return 0, []

    clipPoints2,points_num=ClipSegmentToLine(clipPoints1, sideNormal, posSide,pos_edge)

    if points_num<2: return 0, []

    numContacts=0
    contacts=[]
    for i in range(points_num):
        separation=np.dot(frontNormal, clipPoints2[i].v)-front
        if separation<=0:
            current_contact=Contact()
            current_contact.separation=separation
            current_contact.normal=normal
            current_contact.position=clipPoints2[i].v-separation*frontNormal

            current_contact.feature = FeaturePair(
                clipPoints2[i].fp.inEdge1,
                clipPoints2[i].fp.outEdge1,
                clipPoints2[i].fp.inEdge2,
                clipPoints2[i].fp.outEdge2
            )
            if axis == Axis.FACE_B_X or axis == Axis.FACE_B_Y:
                Flip(current_contact.feature)

            contacts.append(current_contact)
            numContacts+=1

    return len(contacts), contacts

