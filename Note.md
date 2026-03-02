Arbiter.py

Arbiter:
一个类，用于判断两个物体之间的碰撞
属性包括：2个物体(Body)，1个contact(长度为2，因为是2d矩形之间的碰撞)，numContact，friction，restitution

PreStep:
在正式迭代之前进行处理，目的：计算接触Contact的MassNormal，TangentNormal，bias。
如果开启热启动：WarmStarting，将上一帧的冲量直接赋予这一帧的物体，加快收敛速度。

MassNormal:单位法向冲量产生的法向相对速度变化量，要产生单位速度的变化，需要多少冲量。

对于物体A，质量m1，施加-Pn/m1*n的冲量。
对于物体B，质量m2，施加+Pn/m2*n的冲量。

TangentNormal：单位切向冲量产生的切向相对速度变化量。与法向单位质量相似，注意由于库仑定律，施加的切向冲量$Abs(Pt)<= \mu Pn$

bias：由于数值积分的误差，物体会互相“穿透”（Sinking）。为了把它们推开，引入一个修正速度。
$$b = \frac{\beta}{\Delta t} \cdot \max(0, \text{penetration} - \text{slop})$$
slop为允许的最大穿透深度，penetration为当前两个物体的穿透深度，beta为修复速度，当beta为0时，完全不修复，当beta为1时，在1帧内修复完毕，即在1/beta帧数内完全修复当前的穿透。

AppluImpulse:
通过施加冲量Pn，将两个接触物体的相对速度修改为0，bias作用为：由于穿透，额外施加一个小的分离速度。

Update:

根据新的Collide返回的Contact，和上一帧的接触进行对比，当feature相同时，认为是同一对接触，将上一帧的对应接触的冲量继承过来，将该帧的位置，法线，穿透深度更新，如果找不到对应的接触，生成新的接触。


Joint.py

Joint，一个类，用于生成一个关节，确保两个物体之间的某种几何关系保持，例如钟摆。

属性包括：

2个物体，

r1，r2：两个锚点距离各自物体中心的矢量，

localAnchor1，2：两个锚点在各自物体的局部坐标系下的位置，

P：累计冲量，用于WarmStarting，

M：质量矩阵，类似Arbiter中的MassNormal和TangentNormal，用于计算冲量和速度的关系，

bias，biasFactor和Arbiter中的元素意义一致，

softness：关节柔软程度，为0时关节为绝对刚体，大于0时表现为类似弹簧，可增加数值稳定性。

PreStep：

准备阶段，计算M，bias，如果WarmStarting，将上一帧的冲量直接应用到这一帧

ApplyImpulse:

施加冲量，-dv项可以消除相对速度，bias为防止穿透，-softness*P模仿弹簧阻力，当softness为0时，完全刚体，

Body.py

定义Body类型，为矩形，属性：
mass：质量，invMass：质量倒数（当质量为无限大时为0），width，height，全尺寸的宽高，均为float，
position：质心位置，长度为2的矢量，angle：旋转角度，float

velocity：平动速度，长度为2的矢量，angular_velocity：角速度，float

force：受力，长度为2的矢量，torque：力矩，float

friction：摩擦系数，restitution：回弹系数，均为float

I：转动惯量，invI：转动惯量倒数，float

Collide.py

处理两个物体之间的碰撞

FeaturePair类：
四条边用于判断接触是否相同。

ClipVertex类：
1个点的坐标，以及这个点对应的FeaturePair

Contact类：
核心变量，属性包括：
Pn，Pt，Pnb：累计法向冲量，切向冲量，用于修复位置偏差的法向冲量，float
massNormal,massTangent：法向质量，切向质量，float
bias：用于修复位置偏差
feature：FeaturePair类，接触点的几何特征
position：接触点的位置
normal：接触外法线，规定始终由BodyA指向BodyB。
r1：接触点到物体a的矢量
r2：接触点到物体b的矢量
separation：接触点的分离距离：
• >0：分离（无接触）
• =0：恰好接触
• <0：穿透深度（绝对值）

Collide函数：
检测碰撞，通过SAT定理，然后计算出碰撞的外法线，位置，分离距离等

Step1：
通过SAT检测是否有碰撞，从两个物体的四个面的外法线方向投影，检查是否有重叠。

公式：faceA = (AB的中心距离在轴上的投影)-(A的半长)-(B在A轴向上的投影)，结果为2维向量，分别代表在横轴和纵轴上的分离距离。

term1: dp=bodyB.position-bodyA.position, dA=RotAT @ dp (RotAT为A的旋转角度矩阵的转置，Position.local = RT @ Position.world, Position.world = R @ Position.local，注意RT=R-1即逆矩阵)，这里dA的意义是“在 A 看来，B 的质心在哪里”。
term2: hA=[bodyA.width, bodyA.height]*0.5
term3: RotAT @ RotB @ hB, RotB将向量从B的局部空间转为世界空间，RotAT从世界空间转向A的局部空间
faceB的结果类似。

如果在四个轴向方向该结果均大于0说明没有接触，直接跳出。

Step2：
确定最佳分离轴，以及对应的分离距离和法线方向。

首先将A的横轴作为最佳分离轴备选，此时的分离长度为在这个方向上的SAT检测结果(faceA[0]，)，如果此时B在A的右侧，法线为RotA的第一列，即物体A的局部x轴在世界空间中的向量，反之取负，这是为了保证力使得AB分开。

之后对剩下的三个轴方向逐个筛选，取分离长度为负且最接近0的轴(面)为最佳分离轴(面)，法线和分离距离同样如上。

Step3：
为最后裁剪做准备，确定参考面(Reference Face)和受击面(Incident Face)，以及裁剪边界。

通过上一步确定的最佳分离轴确认这些数据，例如当最佳分离轴为A的x轴时，此时的参考面为A的x面，切线为A的y面法线，裁剪的负边界为A的下面，正边界为A的上面(正负是根据切线的方向确定)，Incident Face为B的4条边中最靠近参考面，且法线方向和参考面法线最接近的面。

Step4：
裁剪

1.将Incident Face根据negside(负边界)进行裁剪。
2.根据posside(正边界)进行裁剪。
3.重新对每个接触点计算separation(由于SAT计算的结果为确定重叠了多少，当非平行接触时，这两个点的穿透深度会不一样，所以要重新计算)，即接触点到参考面的有符号距离，符号根据法线决定。
4.将每个接触点存入Contact类中，包括separation，normal，position(接触点的位置，注意不是Incident Face上的点，是根据法线将该点投影到Reference Face上的对应点，)，feature(为FeaturePair类，确定接触由哪几条边组成)

接触点位置投影的原因：
“虽然在这一帧，物体 B 因为速度太快“钻”进了 A 内部，但我们认为碰撞本质上是发生在 A 的表面上的。我们把点投影回表面，就是为了模拟出：“如果在撞击表面的那一瞬间我们就处理了碰撞，应该是什么样子的。”

World.py

World类，
包括gravity：重力大小，float
iteration：单个时间步内迭代求解次数，int
bodies：多个Body
joints：多个joint
arbiters：多个arbiter

BoardPhase：
时间复杂度为O(n2)，

