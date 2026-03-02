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
    position=Vec2(0.0, 0.0) # 位置
    normal=Vec2(0.0, 0.0) # 法线
    r1=Vec2(0.0, 0.0) # 相对位置1
    r2=Vec2(0.0, 0.0) # 相对位置2
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

    print("\n" + "="*70)
    print(f"COLLISION DEBUG: {bodyA.name} vs {bodyB.name}")
    print("="*70)
    
    # 位移向量（带形状和dtype）
    print(f"\n[1] 位移向量 dP (世界坐标系, 从A→B):")
    print(f"    shape={dP.shape}, dtype={dP.dtype}")
    print(f"    dP = bodyB.pos - bodyA.pos = [{dP[0]: .4f}, {dP[1]: .4f}]")
    print(f"    物理意义: B 相对于 A 的位置偏移")
    
    # 旋转矩阵（带形状）
    print(f"\n[2] 旋转矩阵 (A angle={bodyA.angle:.2f} rad, B angle={bodyB.angle:.2f} rad):")
    print(f"    RotA (shape={RotA.shape}):\n{RotA}")
    print(f"    RotAT (shape={RotAT.shape}, A的逆):\n{RotAT}")
    print(f"    RotB (shape={RotB.shape}):\n{RotB}")
    print(f"    RotBT (shape={RotBT.shape}, B的逆):\n{RotBT}")
    
    # 局部坐标系位移（关键：检查是否为1D）
    print(f"\n[3] 局部坐标系位移 (⚠️ 期望 shape=(2,) ):")
    print(f"    dA (shape={dA.shape}): RotAT @ dP = [{dA[0]: .4f}, {dA[1]: .4f}]")
    print(f"        → A视角看B的位置")
    print(f"    dB (shape={dB.shape}): RotBT @ dP = [{dB[0]: .4f}, {dB[1]: .4f}]")
    print(f"        → B视角看A的位置")
    
    # 相对旋转（检查2x2矩阵）
    print(f"\n[4] 相对旋转矩阵 C = RotAT @ RotB (期望 shape=(2, 2)):")
    print(f"    C (shape={C.shape}):\n{C}")
    print(f"    |C| (shape={absC.shape}, 绝对值):\n{absC}")
    print(f"    |C|ᵀ (shape={absCT.shape}):\n{absCT}")
    
    # 半尺寸（检查1D向量）
    print(f"\n[5] 半长宽向量 (⚠️ 期望 shape=(2,) ):")
    print(f"    hA (shape={hA.shape}): [{hA[0]: .4f}, {hA[1]: .4f}]  (A: {bodyA.width}×{bodyA.height})")
    print(f"    hB (shape={hB.shape}): [{hB[0]: .4f}, {hB[1]: .4f}]  (B: {bodyB.width}×{bodyB.height})")
    
    # 分离距离（核心诊断点）
    print(f"\n[6] ⚠️  分离距离 (SAT 核心结果, 期望 shape=(2,) ):")
    print(f"    faceA (shape={faceA.shape}): |dA| - hA - |C|@hB = [{faceA[0]: .4f}, {faceA[1]: .4f}]")
    print(f"        → x轴: {'穿透' if faceA[0] < 0 else '分离'} ({faceA[0]: .4f})")
    print(f"        → y轴: {'穿透' if faceA[1] < 0 else '分离'} ({faceA[1]: .4f})")
    print(f"    faceB (shape={faceB.shape}): |dB| - |C|ᵀ@hA - hB = [{faceB[0]: .4f}, {faceB[1]: .4f}]")
    print(f"        → x轴: {'穿透' if faceB[0] < 0 else '分离'} ({faceB[0]: .4f})")
    print(f"        → y轴: {'穿透' if faceB[1] < 0 else '分离'} ({faceB[1]: .4f})")
    
    if (faceA[0] > 0.0 or faceA[1] > 0.0):
        return 0, []

    if (faceB[0] > 0.0 or faceB[1] > 0.0):
        return 0, []

    axis=Axis.FACE_A_X # 最佳分离轴
    separation=0.0 # 分离距离
    normal=Vec2(0.0, 0.0) # 法向向量
    # 1. FaceA的x轴作为分离轴
    separation=faceA[0]
    normal=RotA[:,0] if dA[0]>0.0 else -RotA[:,0] # Vec2

    # 2. 容差设置：
    # relativeTol (相对容差): 0.95 表示新轴必须比旧轴“好” 5% 以上才会切换
    # absoluteTol (绝对容差): 微小的偏移量，防止在极小穿透时发生轴抖动
    # 这种“偏见”逻辑是为了增加物理模拟的稳定性，防止法线在两个接近的平面间来回跳变（抖动）。
    relativeTol=0.95
    absoluteTol=0.01

    print("\n" + "="*70)
    print("BEST AXIS SELECTION (寻找最佳分离轴)")
    print("="*70)
    print(f"\n[初始] 假设 FACE_A_X 为最佳轴:")
    print(f"    separation = {separation:.6f}")
    print(f"    normal = [{normal[0]:.6f}, {normal[1]:.6f}]")
    
    # 3. 检测A的y轴
    threshold = relativeTol * separation + absoluteTol * hA[1]
    print(f"\n[检查] FACE_A_Y:")
    print(f"    faceA[1] = {faceA[1]:.6f}")
    print(f"    阈值 = {threshold:.6f} (relativeTol={relativeTol} * {separation:.6f} + absoluteTol={absoluteTol} * {hA[1]:.6f})")
    if faceA[1] > threshold:
        separation = faceA[1]  
        axis = Axis.FACE_A_Y
        normal = RotA[:, 1] if dA[1] > 0.0 else -RotA[:, 1]
        print(f"    ✅ 切换！新的 separation = {separation:.6f}, normal = [{normal[0]:.6f}, {normal[1]:.6f}]")
    else:
        print(f"    ❌ 不切换 (faceA[1] <= 阈值)")

    # 4. 检测B的x轴
    threshold = relativeTol * separation + absoluteTol * hB[0]
    print(f"\n[检查] FACE_B_X:")
    print(f"    faceB[0] = {faceB[0]:.6f}")
    print(f"    阈值 = {threshold:.6f}")
    if faceB[0] > threshold:
        separation = faceB[0]  
        axis=Axis.FACE_B_X
        normal = RotB[:, 0] if dB[0] > 0.0 else -RotB[:, 0]
        print(f"    ✅ 切换！新的 separation = {separation:.6f}, normal = [{normal[0]:.6f}, {normal[1]:.6f}]")
    else:
        print(f"    ❌ 不切换 (faceB[0] <= 阈值)")

    # 5. 检测B的y轴
    threshold = relativeTol * separation + absoluteTol * hB[1]
    print(f"\n[检查] FACE_B_Y:")
    print(f"    faceB[1] = {faceB[1]:.6f}")
    print(f"    阈值 = {threshold:.6f}")
    if faceB[1] > threshold:
        separation = faceB[1]  
        axis=Axis.FACE_B_Y
        normal = RotB[:, 1] if dB[1] > 0.0 else -RotB[:, 1]
        print(f"    ✅ 切换！新的 separation = {separation:.6f}, normal = [{normal[0]:.6f}, {normal[1]:.6f}]")
    else:
        print(f"    ❌ 不切换 (faceB[1] <= 阈值)")
    
    axis_names = {Axis.FACE_A_X: "FACE_A_X", Axis.FACE_A_Y: "FACE_A_Y", 
                  Axis.FACE_B_X: "FACE_B_X", Axis.FACE_B_Y: "FACE_B_Y"}
    print(f"\n[最终结果]")
    print(f"    最佳轴: {axis_names[axis]} (axis={axis})")
    print(f"    最终分离距离: {separation:.6f} (负数表示穿透深度)")
    print(f"    最终法向量: [{normal[0]:.6f}, {normal[1]:.6f}] (从A指向B)")
    print(f"    法向量长度: {np.linalg.norm(normal):.6f} {'✅' if np.isclose(np.linalg.norm(normal), 1.0, atol=1e-6) else '⚠️'}")

    frontNormal=np.array([0,0]) # 参考面主法线
    sideNormal=np.array([0,0]) # 参考面切线
    incidentEdge=[] # 被裁剪的边

    # front   ：参考面所在平面到原点的投影距离（Dot(n, x) = front）
    # negSide ：参考面负侧边平面的位置
    # posSide ：参考面正侧边平面的位置
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

    print("\n" + "="*70)
    print("REFERENCE FACE DEBUG INFO")
    print("="*70)
    
    # 1. 参考面法线与切线
    print("\n[1] 参考面方向向量:")
    print(f"    frontNormal (主法线): [{frontNormal[0]: .6f}, {frontNormal[1]: .6f}]")
    print(f"        → 长度: {np.linalg.norm(frontNormal):.6f} {'✅ (单位向量)' if np.isclose(np.linalg.norm(frontNormal), 1.0, atol=1e-6) else '⚠️ (非单位向量!)'}")
    print(f"        → 方向角: {np.degrees(np.arctan2(frontNormal[1], frontNormal[0])):.2f}°")
    
    print(f"    sideNormal (切线):    [{sideNormal[0]: .6f}, {sideNormal[1]: .6f}]")
    print(f"        → 长度: {np.linalg.norm(sideNormal):.6f} {'✅ (单位向量)' if np.isclose(np.linalg.norm(sideNormal), 1.0, atol=1e-6) else '⚠️ (非单位向量!)'}")
    print(f"        → 与法线垂直性: {np.dot(frontNormal, sideNormal):.6e} {'✅ (正交)' if np.isclose(np.dot(frontNormal, sideNormal), 0.0, atol=1e-6) else '❌ (不正交!)'}")
    
    # 2. 参考面平面方程
    print("\n[2] 参考面平面方程: Dot(frontNormal, x) = front")
    print(f"    front = {front:.6f}")
    print(f"    物理意义: 参考面到世界原点的有符号距离")
    
    # 3. 侧边平面位置
    print("\n[3] 侧边平面位置 (定义参考面有效范围):")
    print(f"    negSide = {negSide:.6f}  → 负侧边界 (特征边: {neg_edge})")
    print(f"    posSide = {posSide:.6f}  → 正侧边界 (特征边: {pos_edge})")
    print(f"    有效宽度 = {posSide + negSide:.6f} (应 ≈ 物体尺寸)")

    # 4. 入射边信息
    print("\n[4] 入射边 (Incident Edge) 顶点:")
    v0, v1 = incidentEdge[0], incidentEdge[1]
    print(f"    顶点0: 位置=[{v0.v[0]}, {v0.v[1]}], 特征ID={v0.fp}")
    print(f"    顶点1: 位置=[{v1.v[0]}, {v1.v[1]}], 特征ID={v1.fp}")
    
    # 计算顶点在侧边法线上的投影（验证是否在有效范围内）
    s0 = float(np.dot(sideNormal, v0.v))  # 强制转换为 Python float
    s1 = float(np.dot(sideNormal, v1.v))

    print(f"\n    顶点0侧向投影: {s0:.6f} → {'✅ 有效' if (negSide <= s0 <= posSide) else '❌ 越界'} (范围 [{negSide:.4f}, {posSide:.4f}])")
    print(f"    顶点1侧向投影: {s1:.6f} → {'✅ 有效' if (negSide <= s1 <= posSide) else '❌ 越界'} (范围 [{negSide:.4f}, {posSide:.4f}])")
    
    # 计算顶点到参考面的距离
    d0 = np.dot(frontNormal, v0.v) - front
    d1 = np.dot(frontNormal, v1.v) - front
    print(f"    顶点0到参考面距离: {d0} → {'✅ 在负侧' if d0 <= 1e-6 else '⚠️ 在正侧'}")
    print(f"    顶点1到参考面距离: {d1} → {'✅ 在负侧' if d1 <= 1e-6 else '⚠️ 在正侧'}")

    points_num=0
    clipPoints1,points_num=ClipSegmentToLine(incidentEdge, -sideNormal, negSide,neg_edge)

    if points_num<2: return 0, []

    clipPoints2,points_num=ClipSegmentToLine(clipPoints1, sideNormal, posSide,pos_edge)

    if points_num<2: return 0, []

    print("\n" + "="*70)
    print("CLIPPING PROCESS DEBUG (Sutherland-Hodgman Algorithm)")
    print("="*70)
    print("第一次裁剪")
    print("\n"+"初始顶点为：", incidentEdge[0].v,"和", incidentEdge[1].v,"裁剪法向量为：",-sideNormal,"裁剪距离为：",negSide)
    print("\n"+"裁剪后的顶点为：", clipPoints1[0].v,"和", clipPoints1[1].v)
    print("\n"+"第二次裁剪")
    print("\n"+"初始顶点为：", clipPoints1[0].v,"和", clipPoints1[1].v,"裁剪法向量为：",sideNormal,"裁剪距离为：",posSide)
    print("\n"+"裁剪后的顶点为：", clipPoints2[0].v,"和", clipPoints2[1].v)

    numContacts=0
    contacts=[]
    for i in range(points_num):
        separation=np.dot(frontNormal, clipPoints2[i].v)-front
        if separation<=0:
            current_contact=Contact()
            current_contact.separation=separation
            current_contact.normal=normal
            current_contact.position=clipPoints2[i].v-separation*frontNormal

            # 修复：先复制 feature，再 Flip
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

# if __name__ == "__main__":

#     # 创建测试物体：A静止，B以45°旋转嵌入A的右上角
#     bodyA = Body(
#         mass=1.0,
#         width=2.0,   # 较大盒子
#         height=2.0,
#         pos=[0.0, 0.0],
#         angle=0.0,   # 无旋转
#         velocity=[0.0, 0.0],
#         angular_velocity=0.0,
#         force=[0.0, 0.0],
#         torque=0.0,
#         friction=0.3,
#         name="StaticBox"
#     )

#     bodyB = Body(
#         mass=1.0,
#         width=1.0,   # 小盒子
#         height=1.0,
#         pos=[1.8, 1.9],  # 紧贴A的右上角内部
#         angle=math.pi / 6,   # 45°旋转 → 形成菱形
#         velocity=[0.0, 0.0],
#         angular_velocity=0.0,
#         force=[0.0, 0.0],
#         torque=0.0,
#         friction=0.3,
#         name="RotatedDiamond"
#     )
#     print('物体A',bodyA.name, '质量：',bodyA.mass, '宽度：',bodyA.width, '高度：',bodyA.height, '位置：',bodyA.pos, '角度：',bodyA.angle, '线速度：',bodyA.velocity, '角速度：',bodyA.angular_velocity, '力：',bodyA.force, '扭矩：',bodyA.torque, '摩擦力：',bodyA.friction)
#     print('物体B',bodyB.name, '质量：',bodyB.mass, '宽度：',bodyB.width, '高度：',bodyB.height, '位置：',bodyB.pos, '角度：',bodyB.angle, '线速度：',bodyB.velocity, '角速度：',bodyB.angular_velocity, '力：',bodyB.force, '扭矩：',bodyB.torque, '摩擦力：',bodyB.friction)
#     contact=Contact()

#     Collide(bodyA,bodyB)