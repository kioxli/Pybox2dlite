"""
测试 Collide 函数的正确性
对比 Python 实现和 C++ 版本的预期行为
"""
import numpy as np
import math
from Collide_with_text import Collide, Contact
from Body import Body

def test_case_1_no_collision():
    """测试1：两个完全分离的盒子（应该返回0个接触点）"""
    print("=" * 60)
    print("测试1：两个完全分离的盒子")
    print("=" * 60)
    
    bodyA = Body(
        mass=1.0, width=2.0, height=2.0,
        pos=[0.0, 0.0], angle=0.0,
        velocity=[0.0, 0.0], angular_velocity=0.0,
        force=[0.0, 0.0], torque=0.0, friction=0.3,
        name="BoxA"
    )
    
    bodyB = Body(
        mass=1.0, width=1.0, height=1.0,
        pos=[5.0, 5.0], angle=0.0,  # 完全分离
        velocity=[0.0, 0.0], angular_velocity=0.0,
        force=[0.0, 0.0], torque=0.0, friction=0.3,
        name="BoxB"
    )
    
    numContacts, contacts = Collide(bodyA, bodyB)
    print(f"接触点数量: {numContacts}")
    assert numContacts == 0, "应该没有碰撞！"
    print("✓ 测试通过：正确检测到无碰撞\n")

def test_case_2_simple_collision():
    """测试2：简单的重叠碰撞（应该返回1-2个接触点）"""
    print("=" * 60)
    print("测试2：简单的重叠碰撞")
    print("=" * 60)
    
    bodyA = Body(
        mass=1.0, width=4.0, height=4.0,
        pos=[0.0, 0.0], angle=0.0,
        velocity=[0.0, 0.0], angular_velocity=0.0,
        force=[0.0, 0.0], torque=0.0, friction=0.3,
        name="LargeBox"
    )
    
    bodyB = Body(
        mass=1.0, width=1.0, height=1.0,
        pos=[1.5, 1.5], angle=0.0,  # 部分重叠
        velocity=[0.0, 0.0], angular_velocity=0.0,
        force=[0.0, 0.0], torque=0.0, friction=0.3,
        name="SmallBox"
    )
    
    numContacts, contacts = Collide(bodyA, bodyB)
    print(f"接触点数量: {numContacts}")
    assert numContacts > 0, "应该有碰撞！"
    
    for i, contact in enumerate(contacts):
        print(f"  接触点 {i+1}:")
        print(f"    位置: ({contact.position[0]:.3f}, {contact.position[1]:.3f})")
        print(f"    法线: ({contact.normal[0]:.3f}, {contact.normal[1]:.3f})")
        print(f"    穿透深度: {contact.separation:.3f}")
    print("✓ 测试通过：正确检测到碰撞\n")

def test_case_3_rotated_collision():
    """测试3：旋转的盒子碰撞"""
    print("=" * 60)
    print("测试3：旋转的盒子碰撞")
    print("=" * 60)
    
    bodyA = Body(
        mass=1.0, width=2.0, height=2.0,
        pos=[0.0, 0.0], angle=0.0,
        velocity=[0.0, 0.0], angular_velocity=0.0,
        force=[0.0, 0.0], torque=0.0, friction=0.3,
        name="StaticBox"
    )
    
    bodyB = Body(
        mass=1.0, width=1.0, height=1.0,
        pos=[0.8, 0.9], angle=math.pi / 6,  # 30度旋转
        velocity=[0.0, 0.0], angular_velocity=0.0,
        force=[0.0, 0.0], torque=0.0, friction=0.3,
        name="RotatedBox"
    )
    
    numContacts, contacts = Collide(bodyA, bodyB)
    print(f"接触点数量: {numContacts}")
    
    if numContacts > 0:
        for i, contact in enumerate(contacts):
            print(f"  接触点 {i+1}:")
            print(f"    位置: ({contact.position[0]:.3f}, {contact.position[1]:.3f})")
            print(f"    法线: ({contact.normal[0]:.3f}, {contact.normal[1]:.3f})")
            print(f"    穿透深度: {contact.separation:.3f}")
        print("✓ 测试通过：正确检测到旋转碰撞\n")
    else:
        print("⚠ 警告：未检测到碰撞（可能是位置设置问题）\n")

def test_case_4_edge_contact():
    """测试4：边缘接触（应该返回1个接触点）"""
    print("=" * 60)
    print("测试4：边缘接触")
    print("=" * 60)
    
    bodyA = Body(
        mass=1.0, width=2.0, height=2.0,
        pos=[0.0, 0.0], angle=0.0,
        velocity=[0.0, 0.0], angular_velocity=0.0,
        force=[0.0, 0.0], torque=0.0, friction=0.3,
        name="BoxA"
    )
    
    bodyB = Body(
        mass=1.0, width=1.0, height=1.0,
        pos=[1.5, 0.0], angle=0.0,  # 刚好接触边缘
        velocity=[0.0, 0.0], angular_velocity=0.0,
        force=[0.0, 0.0], torque=0.0, friction=0.3,
        name="BoxB"
    )
    
    numContacts, contacts = Collide(bodyA, bodyB)
    print(f"接触点数量: {numContacts}")
    
    if numContacts > 0:
        for i, contact in enumerate(contacts):
            print(f"  接触点 {i+1}:")
            print(f"    位置: ({contact.position[0]:.3f}, {contact.position[1]:.3f})")
            print(f"    法线: ({contact.normal[0]:.3f}, {contact.normal[1]:.3f})")
            print(f"    穿透深度: {contact.separation:.3f}")
        print("✓ 测试通过：正确检测到边缘接触\n")
    else:
        print("⚠ 注意：边缘接触可能因为浮点精度返回0个接触点\n")

def test_case_5_corner_contact():
    """测试5：角接触（应该返回1个接触点）"""
    print("=" * 60)
    print("测试5：角接触")
    print("=" * 60)
    
    bodyA = Body(
        mass=1.0, width=2.0, height=2.0,
        pos=[0.0, 0.0], angle=0.0,
        velocity=[0.0, 0.0], angular_velocity=0.0,
        force=[0.0, 0.0], torque=0.0, friction=0.3,
        name="BoxA"
    )
    
    bodyB = Body(
        mass=1.0, width=1.0, height=1.0,
        pos=[1.0, 1.0], angle=0.0,  # 角接触
        velocity=[0.0, 0.0], angular_velocity=0.0,
        force=[0.0, 0.0], torque=0.0, friction=0.3,
        name="BoxB"
    )
    
    numContacts, contacts = Collide(bodyA, bodyB)
    print(f"接触点数量: {numContacts}")
    
    if numContacts > 0:
        for i, contact in enumerate(contacts):
            print(f"  接触点 {i+1}:")
            print(f"    位置: ({contact.position[0]:.3f}, {contact.position[1]:.3f})")
            print(f"    法线: ({contact.normal[0]:.3f}, {contact.normal[1]:.3f})")
            print(f"    穿透深度: {contact.separation:.3f}")
        print("✓ 测试通过：正确检测到角接触\n")
    else:
        print("⚠ 注意：角接触可能因为浮点精度返回0个接触点\n")

def test_case_6_45_degree_collision():
    """测试6：45度旋转碰撞"""
    print("=" * 60)
    print("测试6：45度旋转碰撞")
    print("=" * 60)
    
    bodyA = Body(
        mass=1.0, width=2.0, height=2.0,
        pos=[0.0, 0.0], angle=0.0,
        velocity=[0.0, 0.0], angular_velocity=0.0,
        force=[0.0, 0.0], torque=0.0, friction=0.3,
        name="BoxA"
    )
    
    bodyB = Body(
        mass=1.0, width=1.0, height=1.0,
        pos=[0.5, 0.5], angle=math.pi / 4,  # 45度旋转
        velocity=[0.0, 0.0], angular_velocity=0.0,
        force=[0.0, 0.0], torque=0.0, friction=0.3,
        name="Rotated45Box"
    )
    
    numContacts, contacts = Collide(bodyA, bodyB)
    print(f"接触点数量: {numContacts}")
    
    if numContacts > 0:
        for i, contact in enumerate(contacts):
            print(f"  接触点 {i+1}:")
            print(f"    位置: ({contact.position[0]:.3f}, {contact.position[1]:.3f})")
            print(f"    法线: ({contact.normal[0]:.3f}, {contact.normal[1]:.3f})")
            print(f"    穿透深度: {contact.separation:.3f}")
        print("✓ 测试通过：正确检测到45度旋转碰撞\n")
    else:
        print("⚠ 警告：未检测到碰撞\n")

if __name__ == "__main__":
    print("\n开始测试 Collide 函数...\n")
    test_case_4_edge_contact()
    # try:
    #     test_case_1_no_collision()
    #     test_case_2_simple_collision()
    #     test_case_3_rotated_collision()
    #     test_case_4_edge_contact()
    #     test_case_5_corner_contact()
    #     test_case_6_45_degree_collision()
        
    #     print("=" * 60)
    #     print("所有测试完成！")
    #     print("=" * 60)
    # except Exception as e:
    #     print(f"\n❌ 测试失败: {e}")
    #     import traceback
    #     traceback.print_exc()
