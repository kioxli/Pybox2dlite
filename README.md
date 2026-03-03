# Py-Box2D-Lite

[Box2D-Lite](https://github.com/erincatto/box2d-lite) 的 Python 复现，用于学习 2D 刚体物理引擎的核心算法。

## Demo

<video src="https://github.com/user-attachments/assets/b3b90b25-7c70-4e23-b06c-e71c6f952af9" controls width="700"></video>

## 特点

- 纯 Python + NumPy 实现，代码可读性高
- SAT（分离轴定理）碰撞检测
- Sequential Impulse 约束求解（支持摩擦、恢复系数）
- 关节约束（Joint）— 铰链 / 摆锤
- Warm Starting + 位置修正

## 文件说明

| 文件 | 职责 |
|------|------|
| `Main.py` | 入口：场景搭建、matplotlib 渲染循环 |
| `Body.py` | 刚体定义（质量、尺寸、状态） |
| `Collide.py` | SAT 碰撞检测 + 接触点裁剪 |
| `Arbiter.py` | 碰撞仲裁：冲量求解、Warm Starting |
| `Joint.py` | 关节约束求解 |
| `World.py` | 物理世界：Broad Phase、时间步进 |
| `Math.py` | 旋转矩阵等数学工具 |

## 运行

```bash
pip install numpy matplotlib
python Main.py
```

## 算法概要

| 模块 | 方法 |
|------|------|
| 碰撞检测 | SAT（分离轴定理）— 4 轴投影 + 最佳分离面选择 |
| 接触裁剪 | Sutherland-Hodgman 裁剪 → 参考面 / 受击面 |
| 约束求解 | Sequential Impulse — 法向 + 切向冲量迭代 |
| 位置修正 | Baumgarte Stabilization（bias 项防穿透） |
| 加速收敛 | Warm Starting（继承上一帧冲量） |

## 参考

- Catto E. *"Iterative Dynamics with Temporal Coherence."* GDC, 2005.
- [box2d-lite 原版 C++ 代码](https://github.com/erincatto/box2d-lite)
